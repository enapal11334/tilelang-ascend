"""Expert GEMM + FP32 SUM AllReduce on one 910C HCCS node."""

import tilelang
import tilelang.language as T

BM, BN, BK = 16, 128, 128

_EXPERT = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


@T.macro
def _load_panel(A, B, a_l1, b_l1, m, n, k, valid_m, valid_n, K, version):
    valid_k = T.min(BK, K - k * BK)
    T.wait_flag("MTE1", "MTE2", version)
    # Each logical L1 buffer starts at offset zero, including the K tail.
    T.copy(A[m : m + valid_m, k * BK : k * BK + valid_k], a_l1)
    T.copy(B[k * BK : k * BK + valid_k, n : n + valid_n], b_l1)
    T.set_flag("MTE2", "MTE1", version)


@T.macro
def _compute_panel(a_l1, b_l1, a_l0, b_l0, acc, k, version):
    T.wait_flag("MTE2", "MTE1", version)
    T.wait_flag("M", "MTE1", version)
    T.copy(a_l1, a_l0)
    T.copy(b_l1, b_l0)
    T.set_flag("MTE1", "MTE2", version)
    T.set_flag("MTE1", "M", version)
    T.wait_flag("MTE1", "M", version)
    T.mma(a_l0, b_l0, acc, init=(k == 0))
    T.set_flag("M", "MTE1", version)


@tilelang.jit(out_idx=[], pass_configs=_EXPERT)
def clear_flags(groups: int, world_size: int):
    """Initialize a newly allocated symmetric flags tensor, before any peer PUT."""

    @T.prim_func
    def main(flags: T.Tensor((groups, 2, 2, world_size, 16), "int32")):
        with T.Kernel(groups, is_npu=True) as (gid, vid):  # noqa: SIM117
            with T.Scope("V"):
                zeros = T.alloc_ub((16,), "int32")
                T.annotate_address({zeros: 0})
                if vid == 0:
                    T.tile.fill(zeros, 0)
                    T.set_flag("V", "MTE3", 0)
                    T.wait_flag("V", "MTE3", 0)
                    for slot in T.unroll(2):
                        for kind in T.unroll(2):
                            for peer in T.unroll(world_size):
                                T.copy(zeros, flags[gid, slot, kind, peer, :])
                    T.barrier_all()

    return main


@tilelang.jit(out_idx=[], pass_configs=_EXPERT)
def gemm_allreduce_910c(
    M: int,
    N: int,
    K: int,
    dtype: str,
    rank: int,
    world_size: int,
    groups: int,
    slots: int = 2,
):
    """Build (A, B, partial, flags, Y, seq_base); slots=1 is the serial baseline.

    The caller owns symmetric partial/flags and advances seq_base by ceildiv(Q,G).
    All ranks use identical shapes, groups, slot count and sequence intervals.
    """
    n_tiles = (N + BN - 1) // BN
    tiles = ((M + BM - 1) // BM) * n_tiles
    rounds = (tiles + groups - 1) // groups
    k_tiles = (K + BK - 1) // BK

    @T.macro
    def wait_state(flags, poll, gid, slot, kind, peer, seq):
        seen = T.alloc_var("int32", init=0)
        while seen != seq:
            # Finish the preceding scalar UB read before another DMA overwrites it.
            T.barrier_all()
            T.copy(flags[gid, slot, kind, peer, :], poll)
            T.set_flag("MTE2", "S", 0)
            T.wait_flag("MTE2", "S", 0)
            seen = poll[0]

    @T.macro
    def read_rank(partial, flags, poll, dst, peer_tile, gid, slot, peer, seq, valid_m):
        if peer == rank:
            T.copy(partial[gid, slot, 0:valid_m, :], dst)
        else:
            wait_state(flags, poll, gid, slot, 0, peer, seq)
            T.shmem_ub_get_nbi(dst, peer_tile, valid_m * BN, peer)
        T.set_flag("MTE2", "V", 0)
        T.wait_flag("MTE2", "V", 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        partial: T.Tensor((groups, 2, BM, BN), "float32"),
        flags: T.Tensor((groups, 2, 2, world_size, 16), "int32"),
        Y: T.Tensor((M, N), dtype),
        seq_base: T.int32,
    ):
        with T.Kernel(groups, is_npu=True) as (gid, vid):
            a_l1_0 = T.alloc_L1((BM, BK), dtype)
            a_l1_1 = T.alloc_L1((BM, BK), dtype)
            b_l1_0 = T.alloc_L1((BK, BN), dtype)
            b_l1_1 = T.alloc_L1((BK, BN), dtype)
            a_l0_0 = T.alloc_L0A((BM, BK), dtype)
            a_l0_1 = T.alloc_L0A((BM, BK), dtype)
            b_l0_0 = T.alloc_L0B((BK, BN), dtype)
            b_l0_1 = T.alloc_L0B((BK, BN), dtype)
            acc_l0c = T.alloc_L0C((BM, BN), "float32")
            acc_ub = T.alloc_ub((BM, BN), "float32")
            incoming_ub = T.alloc_ub((BM, BN), "float32")
            out_ub = T.alloc_ub((BM, BN), dtype)
            seq_ub = T.alloc_ub((16,), "int32")
            poll_ub = T.alloc_ub((16,), "int32")
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
                    acc_l0c: 0,
                    acc_ub: 0,
                    incoming_ub: BM * BN * 4,
                    out_ub: 2 * BM * BN * 4,
                    seq_ub: 2 * BM * BN * 4 + BM * BN * 2,
                    poll_ub: 2 * BM * BN * 4 + BM * BN * 2 + 64,
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
                        slot = j % slots
                        m = tile // n_tiles * BM
                        n = tile % n_tiles * BN
                        valid_m = T.min(BM, M - m)
                        valid_n = T.min(BN, N - n)
                        T.wait_cross_flag(2 + slot)
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
                                _compute_panel(a_l1_0, b_l1_0, a_l0_0, b_l0_0, acc_l0c, k, 0)
                            else:
                                _compute_panel(a_l1_1, b_l1_1, a_l0_1, b_l0_1, acc_l0c, k, 1)
                        T.set_flag("M", "FIX", 0)
                        T.wait_flag("M", "FIX", 0)
                        T.copy(acc_l0c, partial[gid, slot, :, :])
                        T.set_flag("FIX", "M", 0)
                        T.set_cross_flag("FIX", slot)

                T.wait_cross_flag(2)
                T.wait_cross_flag(3)
                for version in T.unroll(2):
                    T.wait_flag("MTE1", "MTE2", version)
                    T.wait_flag("M", "MTE1", version)
                T.wait_flag("FIX", "M", 0)

            with T.Scope("V"):
                # Both AIVs participate even though only AIV0 handles communication.
                T.set_cross_flag("MTE2", 2)
                T.set_cross_flag("MTE2", 3)
                for j in T.serial(rounds):
                    tile = gid + j * groups
                    if tile < tiles:
                        slot = j % slots
                        T.wait_cross_flag(slot)
                        if vid == 0:
                            m = tile // n_tiles * BM
                            n = tile % n_tiles * BN
                            valid_m = T.min(BM, M - m)
                            valid_n = T.min(BN, N - n)
                            seq = seq_base + j + 1
                            peer_tile = T.decl_buffer(
                                (BM * BN,),
                                "float32",
                                data=partial.data,
                                elem_offset=(gid * 2 + slot) * BM * BN,
                            )

                            # Release payload before a peer can observe READY.
                            T.shmem_quiet()
                            T.tile.fill(seq_ub, seq)
                            T.set_flag("V", "MTE3", 0)
                            T.wait_flag("V", "MTE3", 0)
                            for peer in T.unroll(world_size):
                                if peer != rank:
                                    T.shmem_ub_put_nbi(
                                        seq_ub,
                                        flags,
                                        16,
                                        peer,
                                        ((gid * 2 + slot) * 2 * world_size + rank) * 16,
                                    )
                            T.shmem_quiet()

                            # Identical FP32 rank order on every destination rank.
                            read_rank(
                                partial,
                                flags,
                                poll_ub,
                                acc_ub,
                                peer_tile,
                                gid,
                                slot,
                                0,
                                seq,
                                valid_m,
                            )
                            for peer in T.unroll(1, world_size):
                                T.set_flag("V", "MTE2", 0)
                                T.wait_flag("V", "MTE2", 0)
                                read_rank(
                                    partial,
                                    flags,
                                    poll_ub,
                                    incoming_ub,
                                    peer_tile,
                                    gid,
                                    slot,
                                    peer,
                                    seq,
                                    valid_m,
                                )
                                T.tile.add(
                                    acc_ub[0:valid_m, :],
                                    acc_ub[0:valid_m, :],
                                    incoming_ub[0:valid_m, :],
                                )
                            T.tile.cast(out_ub, acc_ub, "CAST_RINT", count=valid_m * BN)
                            T.set_flag("V", "MTE3", 0)
                            T.wait_flag("V", "MTE3", 0)
                            T.copy(out_ub, Y[m : m + valid_m, n : n + valid_n])
                            T.set_flag("MTE3", "S", 0)
                            T.wait_flag("MTE3", "S", 0)

                            # ACK follows all GETs and output completion, before any ACK wait.
                            for peer in T.unroll(world_size):
                                if peer != rank:
                                    T.shmem_ub_put_nbi(
                                        seq_ub,
                                        flags,
                                        16,
                                        peer,
                                        (((gid * 2 + slot) * 2 + 1) * world_size + rank) * 16,
                                    )
                            T.shmem_quiet()
                            for peer in T.unroll(world_size):
                                if peer != rank:
                                    wait_state(flags, poll_ub, gid, slot, 1, peer, seq)
                        T.set_cross_flag("MTE3", 2 + slot)

    return main
