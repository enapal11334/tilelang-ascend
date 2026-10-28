"""CPU contract/golden checks only. These do not execute the NPU kernel."""

import unittest

import numpy as np

from gemm_allreduce_config import validate_shape, workspace_shapes
from test_gemm_allreduce import compare, make_case


class TestContract(unittest.TestCase):
    def test_supported_shapes_and_flags(self) -> None:
        for m, n, k in ((32, 64, 64), (64, 128, 64), (96, 192, 128)):
            with self.subTest(shape=(m, n, k)):
                partial, flags = workspace_shapes(m, n, k)
                self.assertEqual(np.prod(partial), m * n)
                self.assertEqual(flags, ((m // 32) * (n // 64), 32))
                # Two independent 64B control lines per output tile.
                self.assertEqual(flags[1] * 4, 128)

    def test_unsupported_domain(self) -> None:
        for values in (
            (0, 64, 64, 0),
            (33, 64, 64, 0),
            (32, 65, 64, 0),
            (32, 64, 65, 0),
            (32, 64, 64, 2),
            (32, 64, 64, True),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_shape(*values)

    def test_analytic_golden(self) -> None:
        for name, value in (("zeros", 0), ("ones", 128), ("rank_distinct", -48)):
            _, _, expected = make_case(32, 64, 64, name)
            np.testing.assert_array_equal(expected, np.full((32, 64), value, dtype=np.float32))

    def test_split_k_semantics(self) -> None:
        a, w, expected = make_case(32, 64, 64, "random")
        # Independently form the global K matrix multiplication in FP64 on CPU.
        full_a = np.concatenate((a[0], a[1]), axis=1).astype(np.float64)
        full_w = np.concatenate((w[0], w[1]), axis=1).astype(np.float64)
        compare((full_a @ full_w.T).astype(np.float32), expected)
        _, _, repeated = make_case(32, 64, 64, "random")
        np.testing.assert_array_equal(expected, repeated)

    def test_comparison_rejects_local_only_stale_and_nonfinite(self) -> None:
        a, w, expected = make_case(32, 64, 64, "rank_distinct")
        local_only = a[0].astype(np.float32) @ w[0].astype(np.float32).T
        for actual in (local_only, -expected, np.full_like(expected, np.nan)):
            with self.assertRaises(AssertionError):
                compare(actual, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
