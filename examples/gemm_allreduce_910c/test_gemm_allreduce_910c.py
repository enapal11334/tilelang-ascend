"""Run with torchrun on 2/4/8 local NPUs; --level selects the validation stage."""

import argparse
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
import os
import re
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

import tilelang
from comm import GemmAllReduce910CContext, finalize_shmem
from gemm_allreduce_910c import BK, BM, BN, gemm_allreduce_910c

L0 = ((16, 256, 128), (32, 512, 256))
L1 = ((16, 128, 128), (1, 67, 65), (3, 67, 65), (17, 129, 17), (33, 257, 129))
SENTINELS = ((1, 8192, 4096), (8, 4096, 1024))
DTYPES = (torch.float16, torch.bfloat16)


@contextmanager
def watchdog(seconds: int):
    rank = dist.get_rank()

    def abort():
        print(f"[TIMEOUT] rank={rank} after {seconds}s", flush=True)
        os._exit(124)

    timer = threading.Timer(seconds, abort)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def make_inputs(shape, dtype, rank, case):
    m, n, k = shape
    generator = torch.Generator(device="cpu").manual_seed(9100 + rank)
    a = torch.randn((m, k), generator=generator, device="cpu") * 0.125
    b = torch.randn((k, n), generator=generator, device="cpu") * 0.125
    if case == "zeros":
        a.zero_()
    elif case == "ones":
        a.fill_(1)
        b.fill_(1)
    elif case == "rank":
        a.fill_((rank + 1) / 16)
        b.fill_(1 / 8)
    elif case == "special":
        a[0, 0] = float("nan")
        a[1, 0] = float("inf")
        a[2, 0] = -float("inf")
        b.abs_().add_(1 / 16)
    elif case == "overflow":
        a.zero_()
        b.zero_()
        a[0, 0] = 65504
        b[0, 0] = 2
    return a.to(dtype), b.to(dtype)


def golden(shape, dtype, world_size, case):
    result = None
    for rank in range(world_size):
        a, b = make_inputs(shape, dtype, rank, case)
        local = a.float() @ b.float()
        result = local if result is None else result + local
    return result.to(dtype)


def check_source(kernel, dtype):
    source = kernel.get_kernel_source()
    assert source.count("aclshmem_quiet();") == 3
    for name in (
        "shmem_ub_get_nbi<float>",
        "shmem_ub_put_nbi<int>",
        "copy_l1_to_l0a",
        "copy_l1_to_l0b",
    ):
        assert name in source, f"Missing generated {name}"
    assert ("bfloat16_t" if dtype == torch.bfloat16 else "half") in source
    copies = re.findall(r"copy_gm_to_l1<[^;]+;", source)
    # A single K panel needs no prefetch. Check both panel shapes regardless of how
    # many copy sites survive lowering.
    for operand, rows, cols in (("A", BM, BK), ("B", BK, BN)):
        panel = rf"copy_gm_to_l1<[^,]+,\s*{rows}\s*,\s*{cols}\s*>"
        assert any(re.match(panel, call) for call in copies), f"Missing L1 copy for {operand}"
    assert all(re.search(r",\s*true\s*\);$", call) for call in copies), "L1 tail clear missing"


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Output shape/dtype differs from golden")
    a, g = actual.float(), expected.float()
    for mask in (torch.isnan, torch.isposinf, torch.isneginf):
        if not torch.equal(mask(a), mask(g)):
            raise AssertionError("NaN/+Inf/-Inf positions differ")
    finite = torch.isfinite(g)
    if not finite.any():
        return 1.0, 0.0
    atol, rtol, cap = (
        (2**-14, 2**-9, 0.1) if expected.dtype == torch.float16 else (2**-10, 2**-6, 1.0)
    )
    error = (a[finite] - g[finite]).abs()
    ratio = float((error <= atol + rtol * g[finite].abs()).float().mean())
    maximum = float(error.max())
    if ratio < 0.99 or maximum > cap:
        raise AssertionError(f"matched={ratio:.6f}, max_abs={maximum:.6g}")
    return ratio, maximum


def check(output, expected, label, boundary=False):
    error = None
    try:
        ratio, maximum = compare(output.cpu(), expected)
    except AssertionError as exc:
        error = str(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    tag = "BOUNDARY" if boundary else "PRECISION"
    if any(errors):
        print(f"[{tag}_{'WARN' if boundary else 'FAIL'}] {label}: {errors}", flush=True)
        if not boundary:
            raise AssertionError(str(errors))
    elif dist.get_rank() == 0:
        print(f"[{tag}_PASS] {label} matched={ratio:.6f} max_abs={maximum:.6g}", flush=True)


def precision_cases(shapes, boundary=False):
    device = torch.device(f"npu:{torch.npu.current_device()}")
    for dtype in DTYPES:
        for shape in shapes:
            with watchdog(600):
                a_cpu, b_cpu = make_inputs(shape, dtype, dist.get_rank(), "random")
                a, b = a_cpu.to(device), b_cpu.to(device)
                comm = GemmAllReduce910CContext(a, b)
                check_source(comm.kernel, dtype)
            with watchdog(120), comm:
                cases = ("special",) if boundary else ("zeros", "ones", "rank", "random")
                if not boundary and shape[0] >= 4:
                    cases += ("special",)
                if not boundary and dtype == torch.float16:
                    cases += ("overflow",)
                for case in cases:
                    a_cpu, b_cpu = make_inputs(shape, dtype, comm.rank, case)
                    expected = golden(shape, dtype, comm.world_size, case)
                    b = b_cpu.to(device)
                    for repeat, sign in enumerate((1, -1, 1)):
                        a = (a_cpu * sign).to(device)
                        # No per-call host barrier: a different rank arrives late each time.
                        if comm.rank == repeat % comm.world_size:
                            time.sleep(0.01)
                        output = comm(a, b)
                        comm.stream.synchronize()
                        if (
                            comm.world_size == 4
                            and shape == (16, 256, 128)
                            and dtype == torch.float16
                            and case == "ones"
                            and repeat == 2
                        ):
                            output = output.cpu()
                            p = comm.partial.cpu()[0]
                            print(
                                f"[TILE_DEBUG] rank={comm.rank} "
                                f"Y0={output[:, :BN].unique().tolist()} "
                                f"Y1={output[:, BN:].unique().tolist()} "
                                f"P0={p[0].unique().tolist()} P1={p[1].unique().tolist()}",
                                flush=True,
                            )
                        check(output, expected * sign, f"{shape} {dtype} {case} {repeat}", boundary)


def protocol_inputs(rank, generation):
    """Bit payload: generation/rank weight/tile id survive both output dtypes exactly."""
    a = np.zeros((1, BK), dtype=np.float32)
    a[0, 0] = 2**rank
    b = np.zeros((BK, 5 * BN), dtype=np.float32)
    columns = np.arange(5 * BN)
    lanes = columns % 16
    bits = (generation >> np.minimum(lanes, 11)) & 1
    bits[lanes == 12] = 1  # Sum of distinct rank weights must be 2**P-1.
    tail = lanes >= 13
    bits[tail] = ((columns[tail] // BN) >> (lanes[tail] - 13)) & 1
    b[0] = bits
    return a, b


def protocol():
    device = torch.device(f"npu:{torch.npu.current_device()}")
    for dtype in DTYPES:
        a_cpu, b_cpu = protocol_inputs(dist.get_rank(), 0)
        a = torch.from_numpy(a_cpu).to(dtype).to(device)
        b = torch.from_numpy(b_cpu).to(dtype).to(device)
        with watchdog(600):
            comm = GemmAllReduce910CContext(a, b)
            check_source(comm.kernel, dtype)
        with watchdog(120), comm:
            for generation in range(4096):
                _, b_cpu = protocol_inputs(comm.rank, generation)
                b = torch.from_numpy(b_cpu).to(dtype).to(device)
                if generation % 128 == 0 and comm.rank == (generation // 128) % comm.world_size:
                    time.sleep(0.01)
                output = comm(a, b)
                comm.stream.synchronize()
                expected = torch.from_numpy(b_cpu[0:1] * (2**comm.world_size - 1)).to(dtype)
                if not torch.equal(output.cpu(), expected):
                    raise AssertionError(
                        f"Stale/missing payload: rank={comm.rank} gen={generation}"
                    )
            if comm.rank == 0:
                print(f"[PRECISION_PASS] protocol {dtype}: 4096 generations", flush=True)


def invalid_inputs():
    device = torch.device(f"npu:{torch.npu.current_device()}")
    a = torch.empty((16, 128), dtype=torch.float16, device=device)
    for b in (
        torch.empty((127, 128), dtype=torch.float16, device=device),
        torch.empty((128, 128), dtype=torch.bfloat16, device=device),
    ):
        try:
            comm = GemmAllReduce910CContext(a, b)
        except ValueError:
            if dist.get_rank() == 0:
                print("[BOUNDARY_PASS] incompatible input rejected", flush=True)
        else:
            comm.close()
            print("[BOUNDARY_WARN] incompatible input accepted", flush=True)


def measure(step, stream):
    for _ in range(30):
        output = step()
        stream.synchronize()
        del output
    samples, enqueue = [], []
    for _ in range(10):
        events, outputs = [], []
        for _ in range(100):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            before = time.perf_counter_ns()
            outputs.append(step())
            enqueue.append((time.perf_counter_ns() - before) / 1e6)
            end.record()
            events.append((start, end))
        stream.synchronize()
        samples.append([start.elapsed_time(end) for start, end in events])
        del outputs
    rank_samples = [None] * dist.get_world_size()
    dist.all_gather_object(rank_samples, samples)
    return np.max(np.asarray(rank_samples), axis=0), float(np.median(enqueue))


def speedup_interval(serial, fused):
    rng = np.random.default_rng(910)
    ratios = []
    for _ in range(10000):
        rounds = rng.integers(0, len(serial), size=len(serial))
        indices = rng.integers(0, serial.shape[1], size=serial.shape)
        serial_sample = serial[rounds[:, None], indices]
        indices = rng.integers(0, fused.shape[1], size=fused.shape)
        fused_sample = fused[rounds[:, None], indices]
        ratios.append(np.median(serial_sample) / np.median(fused_sample))
    return np.percentile(ratios, (2.5, 97.5))


def serial_call(comm, kernel, a, b):
    m, n, _ = comm.shape
    output = torch.empty((m, n), dtype=comm.dtype, device=comm.device)
    seq = comm.seq_base
    comm.seq_base += comm.rounds
    kernel(a, b, comm.partial, comm.flags, output, seq, stream=comm.stream.npu_stream)
    return output


def hccl_call(comm, local, cast, a, b, local_fp32, group):
    m, n, _ = comm.shape
    output = torch.empty((m, n), dtype=comm.dtype, device=comm.device)
    local(a, b, local_fp32, stream=comm.stream.npu_stream)
    dist.all_reduce(local_fp32, op=dist.ReduceOp.SUM, group=group, async_op=False)
    cast(local_fp32, output, stream=comm.stream.npu_stream)
    return output


def performance():
    from baseline import cast_output, local_gemm

    device = torch.device(f"npu:{torch.npu.current_device()}")
    with watchdog(600):
        hccl = dist.new_group(ranks=list(range(dist.get_world_size())), backend="hccl")
    for dtype in DTYPES:
        for shape in SENTINELS:
            with watchdog(600):
                a_cpu, b_cpu = make_inputs(shape, dtype, dist.get_rank(), "random")
                a, b = a_cpu.to(device), b_cpu.to(device)
                comm = GemmAllReduce910CContext(a, b)
                serial = gemm_allreduce_910c(
                    *shape,
                    str(dtype).removeprefix("torch."),
                    comm.rank,
                    comm.world_size,
                    comm.groups,
                    slots=1,
                )
                check_source(comm.kernel, dtype)
                check_source(serial, dtype)

                # Same compute/communication, one in-flight tile; no alternate math path.
                serial_step = partial(serial_call, comm, serial, a, b)
                local = local_gemm(*shape, str(dtype).removeprefix("torch."), comm.groups)
                cast = cast_output(shape[0], shape[1], str(dtype).removeprefix("torch."))
                local_fp32 = torch.empty(shape[:2], dtype=torch.float32, device=device)
                hccl_step = partial(hccl_call, comm, local, cast, a, b, local_fp32, hccl)

                expected = golden(shape, dtype, comm.world_size, "random")
                output = comm(a, b)
                comm.stream.synchronize()
                check(output, expected, f"perf prerequisite {shape} {dtype}")
                output = serial_step()
                comm.stream.synchronize()
                check(output, expected, f"serial prerequisite {shape} {dtype}")
                output = hccl_step()
                comm.stream.synchronize()
                check(output, expected, f"HCCL prerequisite {shape} {dtype}")
            with watchdog(120), comm:
                serial_samples, serial_enqueue = measure(serial_step, comm.stream)
                fused_samples, fused_enqueue = measure(partial(comm, a, b), comm.stream)
                hccl_samples, hccl_enqueue = measure(hccl_step, comm.stream)
                interval = speedup_interval(serial_samples, fused_samples)
                if comm.rank == 0:
                    for name, values, enqueue in (
                        ("serial", serial_samples, serial_enqueue),
                        ("fused", fused_samples, fused_enqueue),
                        ("separate_hccl", hccl_samples, hccl_enqueue),
                    ):
                        print(
                            f"{shape} {dtype} {name}: p50={np.median(values):.6f}ms "
                            f"p95={np.percentile(values, 95):.6f}ms "
                            f"enqueue={enqueue:.6f}ms",
                            flush=True,
                        )
                    tag = "PASS" if interval[0] > 1 else "FAIL"
                    print(f"[PERF_{tag}] speedup 95% CI={interval.tolist()}", flush=True)
                    print("Timeline overlap still requires device profiling.", flush=True)
                if interval[0] <= 1:
                    raise AssertionError("Measured overlap benefit did not exceed noise")
    dist.destroy_process_group(hccl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level", default="l0", choices=("l0", "l1", "l2", "boundary", "protocol", "perf", "all")
    )
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) not in (2, 4, 8):
        parser.error("Use single-node torchrun --nproc-per-node=2/4/8")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(1)
    tilelang.disable_cache()
    dist.init_process_group("gloo", timeout=timedelta(seconds=600))
    successful = False
    try:
        if args.level in ("protocol", "perf", "all"):
            protocol()
        if args.level in ("l0", "all"):
            precision_cases(L0)
        if args.level in ("l1", "all"):
            precision_cases(L1)
        if args.level in ("l2", "all"):
            invalid_inputs()
        if args.level in ("boundary", "all"):
            precision_cases(((4, 67, 65),), boundary=True)
        if args.level in ("perf", "all"):
            performance()
        successful = True
        if dist.get_rank() == 0:
            print("Test Passed!", flush=True)
    finally:
        if successful:
            with watchdog(120):
                finalize_shmem()
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
