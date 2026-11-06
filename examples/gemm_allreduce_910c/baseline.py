"""FP32 local GEMM and final cast for the separate HCCL comparison.

The local GEMM uses the fused kernel's tiles, group mapping and explicit
double-buffered Cube pipeline. It writes FP32 directly, without rounding a
local result to FP16/BF16 before the collective. Both callables have explicit
output tensors: ``local(A, B, local_fp32)`` and ``cast(local_fp32, output)``.
"""

import tilelang
import tilelang.language as T

from gemm_allreduce_910c import BK, BM, BN, _EXPERT, _compute_panel, _load_panel


@tilelang.jit(out_idx=[], pass_configs=_EXPERT)
def local_gemm(M: int, N: int, K: int, dtype: str, groups: int):
    """Build ``(A, B, local_fp32)`` using the same Cube schedule as fusion."""
    if min(M, N, K, groups) < 1 or dtype not in ("float16", "bfloat16"):
        raise ValueError("Require positive dimensions/groups and FP16/BF16 inputs")
    n_tiles = (N + BN - 1) // BN
    tiles = ((M + BM - 1) // BM) * n_tiles
    rounds = (tiles + groups - 1) // groups
    k_tiles = (K + BK - 1) // BK

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        output: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(groups, is_npu=True) as (gid, _):
            a_l1_0 = T.alloc_L1((BM, BK), dtype)
            a_l1_1 = T.alloc_L1((BM, BK), dtype)
            b_l1_0 = T.alloc_L1((BK, BN), dtype)
            b_l1_1 = T.alloc_L1((BK, BN), dtype)
            a_l0_0 = T.alloc_L0A((BM, BK), dtype)
            a_l0_1 = T.alloc_L0A((BM, BK), dtype)
            b_l0_0 = T.alloc_L0B((BK, BN), dtype)
            b_l0_1 = T.alloc_L0B((BK, BN), dtype)
            acc = T.alloc_L0C((BM, BN), "float32")
            T.annotate_address(
                {
                    a_l1_0: 0,
                    a_l1_1: BM * BK * 2,
                    b_l1_0: 2 * BM * BK * 2,
                    b_l1_1: 2 * BM * BK * 2 + BK * BN * 2,
                    a_l0_0: 0,
                    a_l0_1: BM * BK * 2,
                    b_l0_0: 0,
                    b_l0_1: BK * BN * 2,
                    acc: 0,
                }
            )
            with T.Scope("C"):
                for version in T.unroll(2):
                    T.set_flag("MTE1", "MTE2", version)
                    T.set_flag("M", "MTE1", version)
                T.set_flag("FIX", "M", 0)
                for j in T.serial(rounds):
                    tile = gid + j * groups
                    if tile < tiles:
                        m = tile // n_tiles * BM
                        n = tile % n_tiles * BN
                        valid_m = T.min(BM, M - m)
                        valid_n = T.min(BN, N - n)
                        _load_panel(A, B, a_l1_0, b_l1_0, m, n, 0, valid_m, valid_n, K, 0)
                        T.wait_flag("FIX", "M", 0)
                        for k in T.serial(k_tiles):
                            if k + 1 < k_tiles:
                                if (k + 1) % 2 == 0:
                                    _load_panel(
                                        A, B, a_l1_0, b_l1_0, m, n, k + 1, valid_m, valid_n, K, 0
                                    )
                                else:
                                    _load_panel(
                                        A, B, a_l1_1, b_l1_1, m, n, k + 1, valid_m, valid_n, K, 1
                                    )
                            if k % 2 == 0:
                                _compute_panel(a_l1_0, b_l1_0, a_l0_0, b_l0_0, acc, k, 0)
                            else:
                                _compute_panel(a_l1_1, b_l1_1, a_l0_1, b_l0_1, acc, k, 1)
                        T.set_flag("M", "FIX", 0)
                        T.wait_flag("M", "FIX", 0)
                        T.copy(acc, output[m : m + valid_m, n : n + valid_n])
                        T.set_flag("FIX", "M", 0)
                for version in T.unroll(2):
                    T.wait_flag("MTE1", "MTE2", version)
                    T.wait_flag("M", "MTE1", version)
                T.wait_flag("FIX", "M", 0)

    return main


@tilelang.jit(out_idx=[], pass_configs=_EXPERT)
def cast_output(M: int, N: int, dtype: str):
    """Build ``(local_fp32, output)`` for one final FP16/BF16 conversion."""
    if min(M, N) < 1 or dtype not in ("float16", "bfloat16"):
        raise ValueError("Require positive dimensions and FP16/BF16 output")
    n_tiles = (N + BN - 1) // BN
    tiles = ((M + BM - 1) // BM) * n_tiles

    @T.prim_func
    def main(
        local_fp32: T.Tensor((M, N), "float32"),
        output: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(tiles, is_npu=True) as (tile, vid):  # noqa: SIM117
            with T.Scope("V"):
                acc_ub = T.alloc_ub((BM, BN), "float32")
                out_ub = T.alloc_ub((BM, BN), dtype)
                T.annotate_address({acc_ub: 0, out_ub: BM * BN * 4})
                if vid == 0:
                    m = tile // n_tiles * BM
                    n = tile % n_tiles * BN
                    valid_m = T.min(BM, M - m)
                    valid_n = T.min(BN, N - n)
                    T.copy(local_fp32[m : m + valid_m, n : n + valid_n], acc_ub)
                    T.set_flag("MTE2", "V", 0)
                    T.wait_flag("MTE2", "V", 0)
                    T.tile.cast(out_ub, acc_ub, "CAST_RINT", count=valid_m * BN)
                    T.set_flag("V", "MTE3", 0)
                    T.wait_flag("V", "MTE3", 0)
                    T.copy(out_ub, output[m : m + valid_m, n : n + valid_n])
                    T.barrier_all()

    return main
