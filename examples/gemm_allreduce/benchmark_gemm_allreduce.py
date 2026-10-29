"""TP=2 latency comparison: TileLang/SHMEM versus serial torch.mm + HCCL.

Run with torchrun on two local NPUs. --impl unfused needs neither TileLang nor SHMEM.
Reported times are synchronized end-to-end host latency, not device kernel duration.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import numpy as np

from gemm_allreduce_config import validate_shape, workspace_shapes
from test_gemm_allreduce import compare, make_case


def summarize(rank_samples: list[list[float]]) -> dict:
    """Use the slower rank for every iteration before computing statistics."""
    samples = np.asarray(rank_samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] != 2 or samples.shape[1] == 0:
        raise ValueError("Require two equal nonempty rank sample lists")
    if not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("Latencies must be positive and finite")
    critical = samples.max(axis=0)
    return {
        "mean_ms": float(critical.mean()),
        "p50_ms": float(np.median(critical)),
        "p95_ms": float(np.percentile(critical, 95)),
        "min_ms": float(critical.min()),
        "max_ms": float(critical.max()),
        "rank_samples_ms": samples.tolist(),
    }


def unfused_reference(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Model the local FP16 GEMM output rounding before the FP32 SUM."""
    return sum(
        (a[r].astype(np.float32) @ w[r].astype(np.float32).T)
        .astype(np.float16)
        .astype(np.float32)
        for r in range(2)
    )


def serial_step(a, weight_t, local_half, output, torch, dist, group) -> None:
    """Full GEMM -> FP32 cast/copy -> SUM, with no compute/communication overlap.

    HCCL waits for the producer on the current stream; async_op=False makes the
    consumer stream wait for the collective before returning to subsequent work.
    No separate compute stream, async collective, or fused matmul API is used.
    """
    torch.mm(a, weight_t, out=local_half)
    output.copy_(local_half)
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group, async_op=False)


def check_precision(actual, expected, dist) -> tuple[float, float]:
    """Fail both ranks together before either enters the next timed collective."""
    error = None
    result = None
    try:
        result = compare(actual, expected)
    except AssertionError as exc:
        error = str(exc)
    errors = [None, None]
    dist.all_gather_object(errors, error)
    if any(item is not None for item in errors):
        raise AssertionError(f"Precision check failed across TP ranks: {errors}")
    return result


def measure(step, warmup: int, iterations: int, torch, dist) -> dict:
    for _ in range(warmup):
        step()
    torch.npu.synchronize()
    dist.barrier()
    samples = []
    for _ in range(iterations):
        # Alignment is deliberately outside the measured region. The previous
        # call has completed on both ranks; this also limits SHMEM epoch skew.
        dist.barrier()
        start = time.perf_counter_ns()
        step()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    rank_samples = [None, None]
    dist.all_gather_object(rank_samples, samples)
    return summarize(rank_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("both", "fused", "unfused"), default="both")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k-local", type=int, default=64, help="Global K = 2 * K_local")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--heap-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, help="Rank 0 JSON result path")
    args = parser.parse_args()
    validate_shape(args.m, args.n, args.k_local, rank=0)
    if args.warmup < 1 or args.iterations < 1 or args.heap_mib < 16:
        parser.error("Require warmup >= 1, iterations >= 1, heap-mib >= 16")
    if args.warmup + args.iterations + 4 >= 2**31:
        parser.error("Too many calls for the signed SHMEM epoch")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (
        int(os.environ.get("WORLD_SIZE", "0")) != 2
        or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 2
        or rank not in (0, 1)
        or local_rank != rank
        or int(os.environ.get("GROUP_RANK", "0")) != 0
    ):
        parser.error("Require torchrun --standalone --nnodes=1 --nproc-per-node=2 (TP=2)")

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    torch.npu.set_device(local_rank)
    if torch.npu.device_count() < 2:
        raise RuntimeError("Two visible Ascend devices are required")
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    device = torch.device(f"npu:{local_rank}")
    shmem = None
    initialized = False
    successful = False
    allocations = []
    hccl = None
    try:
        a_cpu, w_cpu, expected = make_case(
            args.m, args.n, args.k_local, "random", seed=args.seed
        )
        a = torch.from_numpy(a_cpu[rank]).to(device)
        w = torch.from_numpy(w_cpu[rank]).to(device)
        results = {
            "tp": 2,
            "shape": {
                "m": args.m, "n": args.n, "k_local": args.k_local,
                "k_global": 2 * args.k_local,
            },
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "metric": "synchronized_host_e2e_ms_per_iteration_max_of_two_ranks",
            "input_dtype": "float16",
            "communication_output_dtype": "float32",
            "torch_version": str(torch.__version__),
            "torch_npu_version": str(torch_npu.__version__),
            "device": torch.npu.get_device_name(local_rank),
            "execution_order": [
                name for name in ("unfused", "fused") if args.impl in ("both", name)
            ],
            "unfused_local_output_dtype": "float16_then_cast_to_float32",
            "fused_local_output_dtype": "float32",
            "precision_report_rank": 0,
            "precision": {},
        }

        if args.impl in ("both", "unfused"):
            hccl = dist.new_group(ranks=[0, 1], backend="hccl")
            local_half = torch.empty((args.m, args.n), dtype=torch.float16, device=device)
            baseline = torch.empty((args.m, args.n), dtype=torch.float32, device=device)
            weight_t = w.T

            def unfused_step():
                serial_step(a, weight_t, local_half, baseline, torch, dist, hccl)

            unfused_step()
            torch.npu.synchronize()
            baseline_cpu = baseline.cpu().numpy()
            rounded_reference = unfused_reference(a_cpu, w_cpu)
            ratio, maximum = check_precision(baseline_cpu, rounded_reference, dist)
            results["precision"]["unfused"] = {
                "matched_ratio": ratio, "max_abs_vs_rounded_reference": maximum,
                "max_abs_vs_fp32_golden": float(np.max(np.abs(baseline_cpu - expected))),
            }

        if args.impl in ("both", "fused"):
            import shmem
            from gemm_allreduce import build_gemm_allreduce

            partial_shape, flag_shape = workspace_shapes(args.m, args.n, args.k_local)
            required = (int(np.prod(partial_shape)) + int(np.prod(flag_shape))) * 4
            if required + 8 * 1024**2 > args.heap_mib * 1024**2:
                raise ValueError("SHMEM workspace exceeds heap budget with 8 MiB reserve")
            if shmem.set_conf_store_tls(False, "") != 0:
                raise RuntimeError("SHMEM TLS setup failed")
            objects = [shmem.aclshmem_get_unique_id() if rank == 0 else None]
            dist.broadcast_object_list(objects, src=0)
            if not isinstance(objects[0], bytes) or not objects[0]:
                raise RuntimeError("Invalid SHMEM unique ID")
            status = shmem.aclshmem_init_using_unique_id(
                rank, 2, args.heap_mib * 1024**2, objects[0]
            )
            if status != 0:
                raise RuntimeError(f"SHMEM initialization failed: {status}")
            initialized = True
            partial = shmem.aclshmem_create_tensor(
                list(partial_shape), dtype=torch.float32, device_id=local_rank
            )
            allocations.append(partial)
            flags = shmem.aclshmem_create_tensor(
                list(flag_shape), dtype=torch.int32, device_id=local_rank
            )
            allocations.append(flags)
            if flags.data_ptr() % 64:
                raise RuntimeError("SHMEM flags must be 64-byte aligned")
            flags.copy_(torch.zeros(flag_shape, dtype=torch.int32, device="cpu"))
            fused = torch.empty((args.m, args.n), dtype=torch.float32, device=device)
            kernel = build_gemm_allreduce(args.m, args.n, args.k_local, rank)
            epoch = 0

            def fused_step():
                nonlocal epoch
                epoch += 1
                kernel(a, w, partial, flags, fused, epoch)

            torch.npu.synchronize()
            dist.barrier()
            fused_step()
            torch.npu.synchronize()
            ratio, maximum = check_precision(fused.cpu().numpy(), expected, dist)
            results["precision"]["fused"] = {
                "matched_ratio": ratio, "max_abs_vs_fp32_golden": maximum,
            }

        # All selected implementations must pass precision before any timed runs.
        dist.barrier()
        if args.impl in ("both", "unfused"):
            results["unfused"] = measure(
                unfused_step, args.warmup, args.iterations, torch, dist
            )
            check_precision(baseline.cpu().numpy(), rounded_reference, dist)
        if args.impl in ("both", "fused"):
            results["fused"] = measure(fused_step, args.warmup, args.iterations, torch, dist)
            check_precision(fused.cpu().numpy(), expected, dist)

        if args.impl == "both":
            results["speedup_mean"] = (
                results["unfused"]["mean_ms"] / results["fused"]["mean_ms"]
            )
        torch.npu.synchronize()
        dist.barrier()
        if rank == 0:
            print("Metric: synchronized host E2E, per-iteration max across TP ranks.",
                  flush=True)
            print("Local GEMM output: unfused FP16 then FP32 cast; fused FP32. "
                  "These paths have different local rounding.", flush=True)
            print("TP=2 | implementation | mean ms | p50 ms | p95 ms", flush=True)
            for name in ("unfused", "fused"):
                if name in results:
                    row = results[name]
                    print(f"{name:>24} | {row['mean_ms']:.6f} | "
                          f"{row['p50_ms']:.6f} | {row['p95_ms']:.6f}", flush=True)
            if "speedup_mean" in results:
                print(f"unfused / fused = {results['speedup_mean']:.4f}x", flush=True)
            print("Precision: " + json.dumps(results["precision"]), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        dist.barrier()
        successful = True
    finally:
        # Collective cleanup is only safe when both ranks completed successfully.
        # On device/peer failure rely on torchrun and the external watchdog.
        if successful:
            for tensor in reversed(allocations):
                shmem.aclshmem_free_tensor(tensor)
            if initialized:
                shmem.aclshmem_finialize()
            if hccl is not None:
                dist.destroy_process_group(hccl)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
