"""Binary tile operations compare dynamic region sizes by value, not identity."""

import pytest

import tilelang.language as T
from tvm import arith, tir


@pytest.mark.parametrize("scalar", [False, True])
def test_binary_op_accepts_equal_dynamic_row_prefixes(scalar):
    rows = tir.Min(tir.Var("rows", "int32"), 16)
    dst = tir.decl_buffer((16, 128), "float32", scope="shared.ub")
    src0 = tir.decl_buffer((16, 128), "float32", scope="shared.ub")
    src1 = tir.decl_buffer((16, 128), "float32", scope="shared.ub")
    rhs = 1.0 if scalar else src1[:rows, :]
    call = T.tile.add(dst[:rows, :], src0[:rows, :], rhs)
    assert arith.Analyzer().can_prove_equal(call.args[-1], rows * 128)


@pytest.mark.parametrize("operand", ["src0", "src1"])
def test_binary_op_rejects_different_dynamic_row_prefixes(operand):
    rows = tir.Var("rows", "int32")
    dst = tir.decl_buffer((16, 128), "float32", scope="shared.ub")
    src0 = tir.decl_buffer((16, 128), "float32", scope="shared.ub")
    src1 = tir.decl_buffer((16, 128), "float32", scope="shared.ub")
    rows0 = rows + 1 if operand == "src0" else rows
    rows1 = rows + 1 if operand == "src1" else rows
    with pytest.raises(AssertionError, match="size must be same"):
        T.tile.add(dst[:rows, :], src0[:rows0, :], src1[:rows1, :])
