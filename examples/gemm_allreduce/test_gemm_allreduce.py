"""Two local 910C NPUs: deterministic data, CPU golden, repeated SHMEM epochs.

Run via run.sh. Imports of NPU dependencies are deferred so --help works on CPU.
"""

import argparse
from datetime import timedelta
import os
from pathlib import Path
import time

import numpy as np

from gemm_allreduce_config import validate_shape, workspace_shapes


def make_case(m: int, n: int, k: int, case: str, seed: int = 2026) -> tuple:
    """Construct all inputs on CPU; rank values differ to detect a missing SUM."""
    validate_shape(m, n, k, rank=0)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((2, m, k)) * 0.125).astype(np.float16)
    w = (rng.standard_normal((2, n, k)) * 0.125).astype(np.float16)
    if case == "zeros":
        a.fill(0)
    elif case == "ones":
        a.fill(1)
        w.fill(1)
    elif case == "rank_distinct":
        a[0].fill(1)
        a[1].fill(2)
        w[0].fill(0.25)
        w[1].fill(-0.5)
    elif case != "random":
        raise ValueError(f"Unknown case: {case}")
    expected = sum(a[r].astype(np.float32) @ w[r].astype(np.float32).T for r in range(2))
    return a, w, expected


def compare(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """FP32 mixed tolerance; nonfinite values fail the finite input contract."""
    if actual.shape != expected.shape or actual.dtype != np.float32:
        raise AssertionError("Output must have the reference shape and FP32 dtype")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Nonfinite output for finite test inputs")
    error = np.abs(actual - expected)
    ratio = float(np.mean(error <= 2**-16 + 2**-10 * np.abs(expected)))
    maximum = float(np.max(error))
    if ratio < 0.99 or maximum > 1e-2:
        raise AssertionError(f"matched_ratio={ratio:.6f}, max_abs_error={maximum:.6g}")
    return ratio, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("l0", "l1", "all"), default="all")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k-local", type=int, default=64)
    parser.add_argument("--heap-mib", type=int, default=128)
    parser.add_argument("--dump-source", type=Path)
    parser.add_argument("--skew-ms", type=float, default=10.0)
    args = parser.parse_args()
    validate_shape(args.m, args.n, args.k_local, rank=0)
    if args.heap_mib < 16 or not np.isfinite(args.skew_ms) or args.skew_ms < 0:
        parser.error("Require heap-mib >= 16 and finite skew-ms >= 0")
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if (
        int(os.environ.get("WORLD_SIZE", "0")) != 2
        or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 2
        or local_rank != rank
        or rank not in (0, 1)
        or int(os.environ.get("GROUP_RANK", "0")) != 0
    ):
        parser.error("Use torchrun --standalone --nnodes=1 --nproc-per-node=2")

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    import shmem
    import tilelang

    from gemm_allreduce import build_gemm_allreduce

    tilelang.disable_cache()
    torch.npu.set_device(local_rank)
    if torch.npu.device_count() < 2:
        raise RuntimeError("Two visible Ascend devices are required")
    device = torch.device(f"npu:{local_rank}")
    # Gloo only bootstraps SHMEM and controls tests. No tensor collective uses it.
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    initialized = False
    allocations = []
    successful = False
    try:
        if shmem.set_conf_store_tls(False, "") != 0:
            raise RuntimeError("SHMEM TLS setup failed")
        objects = [shmem.aclshmem_get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0)
        uid = objects[0]
        if not isinstance(uid, bytes) or not uid:
            raise RuntimeError("Invalid SHMEM unique ID")
        status = shmem.aclshmem_init_using_unique_id(rank, 2, args.heap_mib * 1024**2, uid)
        if status != 0:
            raise RuntimeError(f"SHMEM initialization failed: {status}")
        initialized = True

        shapes = [(args.m, args.n, args.k_local)]
        if args.level in ("l1", "all"):
            shapes = list(dict.fromkeys(shapes + [(32, 64, 64), (96, 192, 128)]))
        epoch = 0
        for m, n, k in shapes:
            partial_shape, flag_shape = workspace_shapes(m, n, k)
            required = int(np.prod(partial_shape)) * 4 + int(np.prod(flag_shape)) * 4
            if required + 8 * 1024**2 > args.heap_mib * 1024**2:
                raise ValueError("Symmetric workspace exceeds heap budget with 8 MiB reserve")
            # Same order and size on both ranks; never use ordinary torch.empty here.
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
            # Explicit CPU -> device initialization, outside the operator itself.
            flags.copy_(torch.zeros(flag_shape, dtype=torch.int32, device="cpu"))
            output = torch.empty((m, n), dtype=torch.float32, device=device)
            kernel = build_gemm_allreduce(m, n, k, rank)
            if args.dump_source:
                args.dump_source.mkdir(parents=True, exist_ok=True)
                destination = args.dump_source / f"rank{rank}_{m}_{n}_{k}.cpp"
                destination.write_text(kernel.get_kernel_source(), encoding="utf-8")
            torch.npu.synchronize()
            dist.barrier()

            for case in ("zeros", "ones", "rank_distinct", "random"):
                a_cpu, w_cpu, expected = make_case(m, n, k, case)
                w = torch.from_numpy(w_cpu[rank]).to(device)
                for repeat, sign in enumerate((1, -1, 1)):
                    a = torch.from_numpy((a_cpu[rank] * sign).astype(np.float16)).to(device)
                    # No per-epoch barrier: READY must handle either rank arriving late.
                    if rank == repeat % 2:
                        time.sleep(args.skew_ms / 1000)
                    epoch += 1
                    if epoch >= 2**31:
                        raise RuntimeError("Restart before epoch overflow")
                    kernel(a, w, partial, flags, output, epoch)
                    torch.npu.synchronize()
                    try:
                        ratio, maximum = compare(output.cpu().numpy(), expected * sign)
                    except AssertionError:
                        print(
                            f"[PRECISION_FAIL] rank={rank} shape={m,n,k} "
                            f"case={case} epoch={epoch}",
                            flush=True,
                        )
                        raise
                    print(
                        f"[PRECISION_PASS] rank={rank} shape={m,n,k} case={case} "
                        f"epoch={epoch} matched={ratio:.6f} max_abs={maximum:.6g}",
                        flush=True,
                    )
                dist.barrier()
            # Both ranks finish all GETs before collective symmetric deallocation.
            torch.npu.synchronize()
            dist.barrier()
            for tensor in reversed(allocations):
                shmem.aclshmem_free_tensor(tensor)
            allocations.clear()

        dist.barrier()
        successful = True
        if rank == 0:
            print("Test Passed! (two-rank NPU precision)", flush=True)
    finally:
        # On failure, let torchrun/timeout terminate peers instead of entering
        # collective cleanup while a peer may be stuck in a device wait.
        if successful and initialized:
            shmem.aclshmem_finialize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
