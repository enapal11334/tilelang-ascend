"""SHMEM quiet must survive lowering on Vector and preserve PUT/GET order."""

import pytest

import tilelang
import tilelang.language as T
from tvm import IRModule, arith, tir


def _module(body):
    block = tir.Block([], [], [], "tilelang_root", body)
    return IRModule({"main": tir.PrimFunc([], tir.BlockRealize([], True, block))})


def _quiet_calls(body):
    calls = []

    def visit(node):
        if isinstance(node, tir.Call) and getattr(node.op, "name", None) == "tl.ascend_shmem_quiet":
            calls.append(node)

    tir.stmt_functor.post_order_visit(body, visit)
    return calls


def test_quiet_is_opaque_and_survives_noop_removal():
    call = T.shmem_quiet()
    assert len(call.args) == 0
    assert int(call.op.get_attr("TCallEffectKind")) == int(tir.CallEffectKind.Opaque)
    module = tir.transform.RemoveNoOp()(_module(tir.Evaluate(call)))
    assert len(_quiet_calls(module["main"].body)) == 1


def test_shmem_get_preserves_dynamic_symmetric_view_offset():
    group = tir.Var("group", "int32")
    slot = tir.Var("slot", "int32")
    rows = tir.Var("rows", "int32")
    partial = tir.decl_buffer((8, 2, 16, 128), "float32")
    offset = (group * 2 + slot) * 16 * 128
    view = tir.decl_buffer((2048,), "float32", data=partial.data, elem_offset=offset)
    local = tir.decl_buffer((16, 128), "float32", scope="shared.ub")
    call = T.shmem_ub_get_nbi(local, view, rows * 128, 1)
    analyzer = arith.Analyzer()
    assert analyzer.can_prove_equal(call.args[2].args[2], offset)
    assert analyzer.can_prove_equal(call.args[3], rows * 128)


def test_quiet_is_vector_owned_and_rejects_cube_scope():
    body = tir.Evaluate(T.shmem_quiet())
    combined = tilelang.transform.CombineCV()(_module(body))
    tilelang.transform.AscendResourceScopeVerify()(combined)
    branches = combined["main"].body.block.body.seq
    assert not _quiet_calls(branches[0])
    assert len(_quiet_calls(branches[1])) == 1

    cube = tir.AttrStmt(tir.IntImm("int32", 0), "resource_scope", 0, body)
    with pytest.raises(Exception, match="Vector operation must be inside"):
        tilelang.transform.CombineCV()(_module(cube))


@pytest.mark.parametrize("combine", [False, True])
def test_quiet_codegen_preserves_shmem_order(combine):
    @T.prim_func
    def main(Data: T.Tensor((16,), "float32"), Flags: T.Tensor((16,), "int32")):
        with T.Kernel(1, is_npu=True):
            data_ub = T.alloc_ub((16,), "float32")
            flag_ub = T.alloc_ub((16,), "int32")
            T.annotate_address({data_ub: 0, flag_ub: 64})
            with T.Scope("V"):
                T.shmem_ub_put_nbi(flag_ub, Flags, 16, 1)
                T.shmem_quiet()
                T.shmem_ub_get_nbi(data_ub, Data, 16, 1)
                T.shmem_quiet()

    with tilelang.transform.PassContext(config={"tl.ascend_auto_cv_combine": combine}):
        source = tilelang.lower(main, target="ascendc", platform="A3").kernel_source
    quiet = "aclshmem_quiet();"
    assert source.count(quiet) == 2
    assert source.index("shmem_ub_put_nbi<int>") < source.index(quiet)
    assert source.index(quiet) < source.index("shmem_ub_get_nbi<float>")
    assert source.index("shmem_ub_get_nbi<float>") < source.rindex(quiet)
