"""Metadata contract for the two-rank Expert example; has no NPU dependencies."""

BM, BN, BK = 32, 64, 64
WORLD_SIZE = 2
# SHMEM scalar cache lines are 64 bytes. READY and ACK each own a full line.
FLAG_LINE_WORDS = 16
FLAG_WORDS = 2 * FLAG_LINE_WORDS


def validate_shape(m: int, n: int, k_local: int, rank: int) -> None:
    """Reject dimensions outside the intentionally small first-version domain."""
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("Require exactly ranks 0 and 1")
    for name, value, block in (("M", m, BM), ("N", n, BN), ("K_local", k_local, BK)):
        if type(value) is not int or value <= 0 or value % block:
            raise ValueError(f"{name} must be a positive integer divisible by {block}")


def workspace_shapes(m: int, n: int, k_local: int) -> tuple[tuple[int, ...], ...]:
    """Symmetric allocations must use these shapes in this order on both ranks."""
    validate_shape(m, n, k_local, rank=0)
    tiles = (m // BM) * (n // BN)
    return (tiles, BM, BN), (tiles, FLAG_WORDS)
