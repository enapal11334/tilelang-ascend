"""CPU contract/golden checks only. These do not execute the NPU kernel."""

import unittest
from types import SimpleNamespace

import numpy as np

from gemm_allreduce_config import validate_shape, workspace_shapes
from test_gemm_allreduce import compare, make_case
from benchmark_gemm_allreduce import check_precision, serial_step, summarize, unfused_reference


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


class TestBenchmark(unittest.TestCase):
    def test_peer_precision_failure_prevents_timing(self) -> None:
        def gather(objects, local):
            self.assertIsNone(local)
            objects[:] = [None, "peer output is invalid"]

        dist = SimpleNamespace(all_gather_object=gather)
        valid = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(AssertionError, "peer output is invalid"):
            check_precision(valid, valid, dist)

    def test_latency_uses_per_iteration_slowest_rank(self) -> None:
        result = summarize([[1, 9], [8, 2]])
        self.assertEqual(result["mean_ms"], 8.5)
        self.assertEqual(result["p50_ms"], 8.5)
        self.assertEqual(result["max_ms"], 9)
        for samples in ([], [[], []], [[1], [2], [3]], [[0], [1]], [[np.nan], [1]]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                summarize(samples)

    def test_baseline_reference_models_half_rounding(self) -> None:
        a, w, expected = make_case(32, 64, 64, "random")
        partials = [a[r].astype(np.float64) @ w[r].astype(np.float64).T for r in range(2)]
        rounded = sum(p.astype(np.float16).astype(np.float32) for p in partials)
        compare(unfused_reference(a, w), rounded)
        self.assertTrue(np.any(rounded != expected))

    def test_serial_baseline_order_and_output_overwrite(self) -> None:
        calls = []

        class Output:
            value = 99

            def copy_(self, source):
                calls.append("copy")
                self.value = source[0]

        output = Output()
        local = [0]

        def mm(a, w, *, out):
            calls.append("mm")
            out[0] = a * w

        def all_reduce(tensor, *, op, group, async_op):
            calls.append("all_reduce")
            self.assertEqual((op, group, async_op), ("SUM", "HCCL", False))
            tensor.value += 7

        torch = SimpleNamespace(mm=mm)
        dist = SimpleNamespace(all_reduce=all_reduce, ReduceOp=SimpleNamespace(SUM="SUM"))
        for _ in range(2):
            serial_step(2, 3, local, output, torch, dist, "HCCL")
            self.assertEqual(output.value, 13)
        self.assertEqual(calls, ["mm", "copy", "all_reduce"] * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
