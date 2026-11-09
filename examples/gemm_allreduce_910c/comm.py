"""Collective setup and fixed-shape lifetime for the 910C fused kernel."""

from math import prod
from socket import gethostname

import shmem
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from gemm_allreduce_910c import BK, BM, BN, clear_flags, gemm_allreduce_910c

_runtime = None
_active = False
_failed = False
_finalized = False


def _init_shmem(rank: int, world_size: int, device: int) -> None:
    global _runtime, _failed
    if _runtime is None:
        _failed = True
        if shmem.set_conf_store_tls(False, "") != 0:
            raise RuntimeError("SHMEM TLS configuration failed")
        uid = [shmem.aclshmem_get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(uid, src=0)
        if not isinstance(uid[0], bytes) or not uid[0]:
            raise RuntimeError("Invalid SHMEM unique ID")
        status = shmem.aclshmem_init_using_unique_id(rank, world_size, 128 * 1024**2, uid[0])
        if status != 0:
            raise RuntimeError(f"SHMEM initialization failed: {status}")
        _runtime, _failed = (rank, world_size, device), False


def finalize_shmem() -> None:
    """Collectively finish the job after all contexts close; no reinitialization."""
    global _failed, _finalized
    if _failed or _active:
        raise RuntimeError("Cannot finalize SHMEM with a failed or active context")
    if not _finalized:
        _failed = True
        if _runtime is not None:
            torch.npu.synchronize()
            dist.barrier()
            status = shmem.aclshmem_finialize()
            if status not in (None, 0):
                raise RuntimeError(f"SHMEM finalization failed: {status}")
        _finalized, _failed = True, False


class GemmAllReduce910CContext:
    """Own fixed-shape workspace on a process-wide SHMEM runtime.

    Create and close collectively on an initialized, single-node Gloo group.
    Contexts are sequential; call finalize_shmem() once when the job is done.
    Calls use the construction stream; retain A, B and Y until it completes.
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor) -> None:
        global _active, _failed
        if not dist.is_initialized() or dist.get_backend() != "gloo":
            raise RuntimeError("Initialize a single-node Gloo process group first")
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.closed = True
        self.failed = False
        error, metadata = None, None
        try:
            if self.world_size not in (2, 4, 8):
                raise ValueError("Require 2, 4 or 8 ranks")
            if _active or _failed or _finalized:
                raise RuntimeError("Close the active context; finalized/failed jobs cannot restart")
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                raise ValueError("Require A[M,K_local] and B[K_local,N]")
            self.shape = (int(a.shape[0]), int(b.shape[1]), int(a.shape[1]))
            if min(self.shape) <= 0:
                raise ValueError("M, N and K_local must be positive")
            m, n, k = self.shape
            if max(m * k, k * n, m * n) > 2**31 - 1:
                raise ValueError("Kernel element offsets must fit signed int32")
            if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
                raise ValueError("A and B must share FP16 or BF16 dtype")
            if not a.is_contiguous() or not b.is_contiguous():
                raise ValueError("A and B must be contiguous")
            if a.device.type != "npu" or b.device != a.device:
                raise ValueError("A and B must be on the same NPU")
            if a.device.index != torch.npu.current_device():
                raise ValueError("Select the input NPU before constructing the context")
            self.dtype, self.device = a.dtype, a.device
            self.stream = torch.npu.current_stream()
            if _runtime not in (None, (self.rank, self.world_size, self.device.index)):
                raise ValueError("SHMEM rank group and NPU must remain fixed for this job")
            cores = int(torch.npu.get_device_properties(self.device.index).cube_core_num)
            if cores <= 0:
                raise RuntimeError("Could not query the current NPU's Cube core count")
            metadata = (self.shape, str(self.dtype), BM, BN, BK, 2, gethostname(), cores)
        except Exception as exc:
            error = f"rank {self.rank}: {exc}"
        peers = [None] * self.world_size
        dist.all_gather_object(peers, (error, metadata))
        errors = [item[0] for item in peers if item[0] is not None]
        if errors:
            raise ValueError("Invalid collective inputs: " + "; ".join(errors))
        configs = [item[1] for item in peers]
        if any(config[:-1] != configs[0][:-1] for config in configs):
            raise ValueError(
                "All ranks must share one host, shape, dtype and protocol configuration"
            )
        m, n, k = self.shape
        tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)
        self.groups = min(min(config[-1] for config in configs), max(1, tiles // 2))
        self.rounds = (tiles + self.groups - 1) // self.groups
        self.seq_base = 0
        inbox_shape = [self.groups, 2, self.world_size, BM, BN]
        flags_shape = [self.groups, 2, 2, self.world_size, 16]
        workspace_bytes = 4 * (prod(inbox_shape) + prod(flags_shape))
        if workspace_bytes > 120 * 1024**2:
            raise ValueError("Symmetric workspace exceeds the fixed 128 MiB heap budget")

        _init_shmem(self.rank, self.world_size, self.device.index)
        _failed = True
        self.inbox = shmem.aclshmem_create_tensor(
            inbox_shape, dtype=torch.float32, device_id=self.device.index
        )
        self.flags = shmem.aclshmem_create_tensor(
            flags_shape,
            dtype=torch.int32,
            device_id=self.device.index,
        )
        if self.inbox.data_ptr() % 64 or self.flags.data_ptr() % 64:
            raise RuntimeError("Symmetric workspace must be 64-byte aligned")
        self.kernel = gemm_allreduce_910c(
            m,
            n,
            k,
            str(self.dtype).removeprefix("torch."),
            self.rank,
            self.world_size,
            self.groups,
        )
        clear_flags(self.groups, self.world_size)(self.flags, stream=self.stream.npu_stream)
        self.stream.synchronize()
        dist.barrier()
        _active, _failed = True, False
        self.closed = False

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        global _failed
        if self.closed or self.failed:
            raise RuntimeError("Context is closed or failed")
        self.failed = _failed = True
        m, n, k = self.shape
        if (
            tuple(a.shape) != (m, k)
            or tuple(b.shape) != (k, n)
            or a.dtype != self.dtype
            or b.dtype != self.dtype
            or a.device != self.device
            or b.device != self.device
            or not a.is_contiguous()
            or not b.is_contiguous()
        ):
            raise ValueError("Inputs must match this context's fixed metadata")
        if (
            torch.npu.current_device() != self.device.index
            or torch.npu.current_stream().npu_stream != self.stream.npu_stream
        ):
            raise RuntimeError("Use the NPU and stream bound to this context")
        if self.seq_base + self.rounds >= 2**31:
            raise OverflowError("Close and collectively rebuild before sequence overflow")
        output = torch.empty((m, n), dtype=self.dtype, device=self.device)
        seq_base = self.seq_base
        self.seq_base += self.rounds
        self.kernel(a, b, self.inbox, self.flags, output, seq_base, stream=self.stream.npu_stream)
        self.failed = _failed = False
        return output

    def close(self) -> None:
        """Collective normal completion; never call after a peer/device failure."""
        global _active, _failed
        if not self.closed:
            if self.failed or _failed:
                raise RuntimeError("A failed context requires terminating all workers")
            _failed = True
            self.stream.synchronize()
            dist.barrier()
            shmem.aclshmem_free_tensor(self.flags)
            shmem.aclshmem_free_tensor(self.inbox)
            self.closed = True
            _active, _failed = False, False

    def __enter__(self) -> "GemmAllReduce910CContext":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        global _failed
        if exc_type is None:
            self.close()
        else:
            # A peer may be blocked in a device wait; the launcher must stop all ranks.
            self.failed = _failed = True
