"""One-launch Expert GEMM + two-rank SHMEM SUM, targeting Ascend 910C.

Adapted from D:/mKernel/examples/ascend_gemm_allreduce. This is a single-core
protocol prototype, not a tuned collective. See README.md for the supported domain.
"""

import re
from typing import Any

import tilelang
import tilelang.language as T

from gemm_allreduce_config import BK, BM, BN, FLAG_LINE_WORDS, FLAG_WORDS, validate_shape

_EXPERT_CONFIG = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@tilelang.jit(out_idx=[], pass_configs=_EXPERT_CONFIG)
def _gemm_allreduce(M: int, N: int, K: int, rank: int):
    tiles = (M // BM) * (N // BN)
    n_tiles = N // BN
    peer = 1 - rank

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        W: T.Tensor((N, K), "float16"),
        partial: T.Tensor((tiles, BM, BN), "float32"),
        flags: T.Tensor((tiles, FLAG_WORDS), "int32"),
        output: T.Tensor((M, N), "float32"),
        epoch: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            a_l1 = T.alloc_L1((BM, BK), "float16")
            w_l1 = T.alloc_L1((BN, BK), "float16")
            acc_l0c = T.alloc_L0C((BM, BN), "float32")
            local_ub = T.alloc_ub((BM, BN), "float32")
            peer_ub = T.alloc_ub((BM, BN), "float32")
            epoch_ub = T.alloc_ub((FLAG_LINE_WORDS,), "int32")
            poll_ub = T.alloc_ub((FLAG_LINE_WORDS,), "int32")
            T.annotate_address(
                {
                    a_l1: 0,
                    w_l1: BM * BK * 2,
                    acc_l0c: 0,
                    local_ub: 0,
                    peer_ub: BM * BN * 4,
                    epoch_ub: 2 * BM * BN * 4,
                    poll_ub: 2 * BM * BN * 4 + FLAG_LINE_WORDS * 4,
                }
            )

            with T.Scope("C"):
                for tile in T.serial(tiles):
                    # Two tokens allow tile t+1 GEMM to overlap tile t communication.
                    T.wait_cross_flag(2 + tile % 2)
                    for kk in T.serial(K // BK):
                        T.copy(A[(tile // n_tiles) * BM, kk * BK], a_l1)
                        T.copy(W[(tile % n_tiles) * BN, kk * BK], w_l1)
                        T.barrier_all()
                        T.gemm_v0(a_l1, w_l1, acc_l0c, transpose_B=True, init=(kk == 0))
                        T.barrier_all()
                    T.copy(acc_l0c, partial[tile, 0, 0])
                    T.barrier_all()
                    T.set_cross_flag("FIX", tile % 2)
                # Drain all outstanding consumers before the next launch can write.
                T.wait_cross_flag(2)
                T.wait_cross_flag(3)

            with T.Scope("V"):
                seen = T.alloc_var("int32", init=0)
                if vid == 0:
                    T.tile.fill(epoch_ub, epoch)
                    T.barrier_all()
                # Both AIVs participate in every local C/V handshake.
                T.set_cross_flag("MTE2", 2)
                T.set_cross_flag("MTE2", 3)
                for tile in T.serial(tiles):
                    T.wait_cross_flag(tile % 2)
                    if vid == 0:
                        # PUT a full control line to the peer; keep it immutable.
                        T.shmem_ub_put_nbi(
                            epoch_ub, flags, FLAG_LINE_WORDS, peer, tile * FLAG_WORDS
                        )
                        T.barrier_all()
                        seen = 0
                        while seen < epoch:
                            # Finish the previous scalar UB read before reusing it.
                            T.barrier_all()
                            T.copy(flags[tile, 0], poll_ub)
                            T.set_flag("MTE2", "S", 0)
                            T.wait_flag("MTE2", "S", 0)
                            seen = poll_ub[0]
                        T.copy(partial[tile, 0, 0], local_ub)
                        # A metadata-only GM view supplies the existing GET's
                        # source offset; access_ptr includes elem_offset.
                        peer_tile = T.decl_buffer(
                            (BM * BN,),
                            "float32",
                            data=partial.data,
                            elem_offset=tile * BM * BN,
                        )
                        T.shmem_ub_get_nbi(peer_ub, peer_tile, BM * BN, peer)
                        T.set_flag("MTE2", "V", 0)
                        T.wait_flag("MTE2", "V", 0)
                        T.tile.add(local_ub, local_ub, peer_ub)
                        T.set_flag("V", "MTE3", 0)
                        T.wait_flag("V", "MTE3", 0)
                        T.copy(local_ub, output[(tile // n_tiles) * BM, (tile % n_tiles) * BN])
                        T.barrier_all()
                        # ACK only after GET completion; peer must not overwrite early.
                        T.shmem_ub_put_nbi(
                            epoch_ub,
                            flags,
                            FLAG_LINE_WORDS,
                            peer,
                            tile * FLAG_WORDS + FLAG_LINE_WORDS,
                        )
                        T.barrier_all()
                        seen = 0
                        while seen < epoch:
                            T.barrier_all()
                            T.copy(flags[tile, FLAG_LINE_WORDS], poll_ub)
                            T.set_flag("MTE2", "S", 0)
                            T.wait_flag("MTE2", "S", 0)
                            seen = poll_ub[0]
                    T.set_cross_flag("MTE3", 2 + tile % 2)

    return main


def build_gemm_allreduce(m: int, n: int, k_local: int, rank: int) -> Any:
    """Build callable (A, W, partial, flags, output, epoch), all outputs explicit.

    partial/flags must be identically allocated SHMEM tensors on both ranks.
    Zero flags once before the first launch. Epochs are equal on both ranks,
    positive, strictly increasing, and below 2**31. No concurrent workspace reuse.
    Caller retains every tensor until completion. Device waits have no timeout.
    """
    validate_shape(m, n, k_local, rank)
    kernel = _gemm_allreduce(m, n, k_local, rank)
    source = kernel.get_kernel_source()
    for name in ("shmem_ub_get_nbi", "shmem_ub_put_nbi"):
        if not re.search(r"tl::ascend::" + name + r"\s*<", source):
            raise RuntimeError(f"Missing generated {name} call")
    return kernel
