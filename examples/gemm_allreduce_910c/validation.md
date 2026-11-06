# 验证记录

记录日期：2026-11-05。当前机器为Windows，无torch/torch_npu、TVM运行库、CANN或NPU。
以下严格区分本地检查与设备验收；不把静态检查当作DSL lowering或NPU通过。

## 已执行

- Python `compile`：新算子文件、两个前端回归及修改的Python前端语法通过。
- Ruff静态检查：算子与修改的Python前端/回归通过；算子和新回归的100列格式检查通过，
  已有前端文件按仓库默认格式检查通过。
- NumPy执行测试脚本中的协议payload生成函数：4096个世代互异；验证2/4/8 rank的
  数学SUM、漏rank、单rank旧世代及错误tile的检测能力，FP16最终转换不掩盖这些突变。
- NumPy执行分层bootstrap函数：对确定性的2倍合成时间给出 `[2,2]` 区间。
  这是统计辅助函数检查，不是NPU性能数据。
- 临时CPU mock检查实际context的进程生命周期：跨context只初始化一次，终结后不能重建，
  多机/元素偏移溢出被拒绝，捕获launch异常后不进入集体free/finalize。不是SHMEM设备测试。
- 源码审计发现并修复：多维flags初始化只搬最后两轴；动态tile.add长度按对象身份比较；
  每shape重复SHMEM init/finalize可能复用已关闭的bootstrap plugin handle。

## 实际启动尝试

使用本机可用的Python执行：

```text
python examples/gemm_allreduce_910c/test_gemm_allreduce_910c.py --level l0
python examples/gemm_allreduce_910c/test_gemm_allreduce_910c.py --level perf
```

两次均退出1，在 `import torch` 处报 `ModuleNotFoundError: No module named 'torch'`。
未进入bootstrap、JIT或任何设备操作，因此不是L0/性能通过，也不是算子kernel错误的证据。
前端pytest/lowering尚未执行；当前环境也缺pytest和编译后的TileLang/TVM库。

## 必要实现审计

```text
[DISPATCH-COVERAGE-AUDIT]
supported_domain: matching contiguous FP16/BF16 A[M,K_local],B[K_local,N];
                  positive dimensions, element offsets<=2^31-1; single-node P=2/4/8
generic_fallback: one ceildiv/valid-extent Expert kernel for both dtypes
uncovered_input: none found in static domain audit; device behavior unverified
specialization_cases: both dtypes; L0 aligned, required L1 tails, both decode sentinels
result: static coverage checked; not a device correctness pass

[HOST-METADATA-AUDIT]
operator: shape/stride/dtype/device/stream validation, integer scheduling,
          empty buffer allocation, symmetric ownership, seq reservation, launch
input_stride -> output_stride: A=(K_local,1),B=(N,1),Y=(N,1)
shares_storage / same_data_ptr: original A/B passed unchanged; distinct owned Y/workspace
aclnn_or_physical_copy: no operator-side tensor math/reorder/cast; flags via TileLang
test: CPU generation/cast/golden, explicit H2D/D2H; HCCL only in separate baseline
result: source audit passed; device execution unverified

[REORDER-COST-AUDIT]
path: both input dtypes, BM/BN/BK=16/128/128; no full-input reorder pass
gm_passes: tiled input reuse; one partial write, one local read, P-1 remote reads,
           one final output write; local GM->UB helper may also initialize UB padding
dma_transactions: 2QL input + 2QL L1/L0 + (P+2)Q partial/read/output,
                  2(P-1)Q control PUTs plus measured polling iterations
average_dma_bytes: M1 case input=16512 B, GET=512 B; M8 input=17408 B, GET=4096 B;
                   control/poll=64 B (P=8), per design section 5.3
gm_scalar_accesses: 0 for payload and mutable flags; scalar comparisons read UB
address_div_mod_per_element: 0; tile decode per task
active_cores: G Cube + G AIV0, AIV1 participates only in local handshake
serial_tasks_per_core: floor/ceil(Q/G)
maximum_case_timeout: both dtype x P=8 x [(1,8192,4096),(8,4096,1024)], warm 120 s
result: costs statically accounted; sentinels NOT RUN on device
```

## 尚未验证

- 新C++ intrinsic能否在目标CANN/SHMEM版本编译，quiet scope及调用顺序回归。
- symbolic region修复的真实TVM前端回归；主kernel生成代码中的双L1清零、布局及pipe归属。
- 910C资源容量、两AIV cross-flag计数、FIX→READY→GET可见性和GET→ACK→覆写顺序。
- 真实4096代协议、L0、尾块、特殊值，以及2→4→8 rank正确性。
- 两个decode哨兵的实际延迟、相对串行/HCCL结果和设备时间线。

因此，当前状态为“实现与测试代码已写入、静态检查通过，设备验收未完成”。
