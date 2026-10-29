# 单机两卡 GEMM + AllReduce（Expert，SHMEM）

目标硬件 Ascend 910C。**当前为初版实现，未在 NPU 上编译或运行验证。**
CPU 检查不代表 910C 双卡正确性或通算融合性能通过。

每卡持有 K 维分片 `A[M,K_local]`、`W[N,K_local]`，两卡都输出完整结果：

```text
Y = A_rank0 @ W_rank0.T + A_rank1 @ W_rank1.T
```

FP16 输入，FP32 累加、跨卡通信和输出；SUM 不除以 2。
要求连续输入、正整数且 M/N/K_local 分别被 32/64/64 整除。
初版不支持尾块、其他 dtype、多机、多于两卡、autograd 或并发复用工作区。

## 通算融合方式

同一个 TileLang 混合 C/V kernel：Cube 逐 tile 计算 GEMM 并写 SHMEM 对称 GM；
Vector 收到 FIX 完成信号后用 `T.shmem_ub_put_nbi` 发布 ready，等待 peer 的对应 tile，
通过 `T.shmem_ub_get_nbi` 读入 UB，
在 Vector 上求和并写结果。Cube 可以计算下一 tile，因此不是完整 GEMM 后再启动通信。
两组本地 token 限制生产者领先量；GET 完成后的 ACK 和递增 epoch 保护重复调用。
ready/ACK 分别占独立 64B 缓存行。
等待状态时使用 `T.copy` 将 GM 控制行搬至 UB，完成 MTE2→S 同步后再读 UB 中的 epoch；
不直接轮询 GM scalar load。GET 源 tile 用共享 `partial.data`、带 `elem_offset` 的
`T.decl_buffer` 视图表示，不复制数据或新增对称分配。

每卡仅用一个逻辑 Cube group，两 AIV 参与本地同步，AIV0 做通信/求和。
这是最小协议原型；没有实现 mKernel 的自适应 SM 分工、多机分层归约，尚无加速比结论。

## Linux 双卡环境准备（待执行）

需要匹配的 CANN、PyTorch/torch_npu、本仓库及 SHMEM 子模块。环境须有 Gloo backend，
两张 910C 必须能够经 SHMEM 互访。按实际位置加载 CANN 环境，再在仓库根目录执行：

```bash
bash install_ascend.sh --enable-shmem
source 3rdparty/shmem/install/set_env.sh
```

本例仅复用仓库现有 SHMEM 接口，不依赖 `T.call_extern`、自定义 helper 或 codegen 修改。
已有 SHMEM 代码生成路径会处理 Buffer 的 `elem_offset`；构建函数检查 GET/PUT 调用存在。
SHMEM 使用本仓库子模块版本，不能用旧文档中的三参数 `aclshmem_init` 接口替代。

## 检查与运行

CPU 只需 Python + NumPy，不需要 torch/TileLang/NPU：

```bash
python -m unittest discover -s examples/gemm_allreduce -p test_cpu.py -v
python examples/gemm_allreduce/test_gemm_allreduce.py --help
```

安装可运行的 TileLang 后，先检查原生 SHMEM 的 Buffer 视图偏移（此处尚未执行）：

```bash
python -m pytest testing/python/language/test_tilelang_ascend_language_shmem_view.py -v
```

910C 两卡上运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 bash examples/gemm_allreduce/run.sh --level l0 --dump-source generated
# 单 tile、默认 case、9 tiles + 两次 K 循环：
ASCEND_RT_VISIBLE_DEVICES=0,1 bash examples/gemm_allreduce/run.sh --level all --dump-source generated
# 指定其他对齐尺寸；外层 watchdog 默认 600 秒，包含首次编译：
TEST_TIMEOUT=900 bash examples/gemm_allreduce/run.sh --m 96 --n 192 --k-local 128 --level l0
```

测试构造零、一、rank 不同常数和固定 seed 随机数。所有输入和 golden 均在 CPU 生成；
同一工作区按正/负/正多次调用，并轮流延迟一个 rank 的 launch。默认误差门槛：
atol=2^-16、rtol=2^-10、matched_ratio≥0.99 且 max_abs_error≤1e-2；非有限输出失败。
成功标记只有真机测试全部完成后才打印。

torchrun/Gloo 用于 bootstrap、CPU unique-id 广播及测试 barrier，算子不使用 HCCL 或
PyTorch matmul。SHMEM 默认在该本机测试中关闭 TLS；所用端口由 unique-id bootstrap 管理。
`ACLSHMEM_MASTER_ADDR/PORT` 可按 bundled SHMEM 的环境约定设置。
设备侧 wait 没有超时，保留 run.sh 的外部 watchdog；失败由 torchrun 终止 peer。

## TP=2 串行基线与融合时延对比

新增 `benchmark_gemm_allreduce.py` 和带 watchdog 的 `benchmark.sh`。
基线完全不使用 TileLang：每卡显式调用
`torch.ops.aten.mm.dtype_out(A_rank, W_rank.T, torch.float32, out=output_fp32)`，
将整个本地 GEMM 的结果直接写入 FP32，再以 HCCL
`all_reduce(SUM, async_op=False)` 得到完整输出。
计算和通信通过当前流依赖串行执行，没有计算/通信重叠，也没有 FP16 局部输出或转换。
两卡都保存完整结果，TP=2，`K_global = 2 * K_local`。

需要支持 `aten.mm.dtype_out` 的匹配 PyTorch/torch_npu/CANN 环境。
[Ascend 官方配置](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/config/op_plugin_functions.yaml)
从 v2.9 开始注册该 overload；[后端实现](https://raw.githubusercontent.com/Ascend/op-plugin/master/op_plugin/ops/opapi/MmKernelNpuOpApi.cpp)
将指定的 FP32 输出直接传给 `aclnnMm`。旧版或目标设备不支持时直接失败，不回退。
脚本在创建 HCCL group 前执行设备探针，检查 `1 + 2^-12` 和 `1 + 2^-12 - 1`
的结果，以拒绝 FP16 累加舍入或 FP16 输出再转 FP32 的常见路径。
探针不计时；其通过也不能证明所有输入下的内部累加行为。

在上述 Linux 双卡环境准备完成后运行（额外需要 HCCL backend）：

```bash
# 同输入、同 shape 比较两条路径，输出 mean/p50/p95 和 mean 加速比。
ASCEND_RT_VISIBLE_DEVICES=0,1 bash examples/gemm_allreduce/benchmark.sh \
    --m 96 --n 192 --k-local 128 --warmup 20 --iterations 100 \
    --output benchmark_results/tp2.json

# 单独运行非 TileLang 基线；仅需 PyTorch、torch_npu、NumPy、Gloo 和 HCCL。
ASCEND_RT_VISIBLE_DEVICES=0,1 bash examples/gemm_allreduce/benchmark.sh --impl unfused
# 单独测融合路径（不创建 HCCL group）。
ASCEND_RT_VISIBLE_DEVICES=0,1 bash examples/gemm_allreduce/benchmark.sh --impl fused
```

计时范围为**同步后的完整调用 host E2E 时延**：从 host 发起调用到设备全部完成，
包括 Python dispatch 和末尾设备同步。每轮 Gloo barrier 在计时外；初始化、编译、
分配、H2D/D2H、预热和精度检查均不计入。两条路径先通过精度检查才开始测量，
正式测量后再次校验。每轮取两 rank 时延的最大值，再计算 mean/p50/p95；
`speedup_mean = unfused_mean_ms / fused_mean_ms`，大于 1 表示本次融合路径更快。
JSON 保留每卡原始样本、shape、版本、设备、执行顺序及误差，只有 rank 0 写文件。
两 rank 均执行精度校验；JSON 的误差数值来自 rank 0。
默认先测 unfused 后测 fused；可单独运行两种模式并调换顺序，检查热状态影响。
需要纯设备 kernel 时延或重叠分析时，应另外使用 msprof 检查设备时间线；
此处 E2E 数据不能代替设备 profiler 数据。

**统一精度约定**：两者使用同一份 FP16 输入，要求 FP32 累加、局部输出、通信和最终输出。
已删除局部 FP16 舍入 golden；两条路径均调用 `make_case` 返回的同一个 FP32 CPU golden，
并通过同一个 `compare` 检查：atol=2^-16、rtol=2^-10、matched_ratio≥0.99，
且 max_abs_error≤1e-2，非有限输出失败。测量前后均检查，任一卡失败则不输出加速比。
JSON 为两条路径记录相同的误差字段和统一容差。
GEMM 实现及通信库（HCCL/SHMEM）仍不同，加速比是完整实现的比较，不能单独归因于融合。
当前环境没有双卡 NPU，尚无真实时延数据；910C 上该 CANN 路径的支持、内部累加精度和
融合 kernel 正确性仍需目标版本源码及真机核验，不能据 CPU 检查判定通过。

## 接口与资源所有权

`build_gemm_allreduce(M,N,K_local,rank)` 返回低层 callable：

```python
kernel(A, W, symmetric_partial, symmetric_flags, output, epoch)
```

两 rank 必须同序、同大小分配 SHMEM partial/flags，shape 由 `workspace_shapes` 给出。
flags 首次使用前归零且 64B 对齐。epoch 两卡一致、严格递增、0<epoch<2^31。
所有输入、输出、工作区必须存活至设备完成；同一 workspace 不能并发使用。
对称分配和释放须两卡协调，完整生命周期示例见 `test_gemm_allreduce.py`。

## 来源

- `D:/mKernel/src/gemm_ar.cu`（本地 HEAD `ef6225fdde493f4119f519c99bfa4ad9234e323b`）：
  compute 在输出 chunk 就绪后通知通信侧，通信侧消费并归约。
- `D:/mKernel/examples/ascend_gemm_allreduce`：未验证的 Ascend 原型；本次迁入后调整
  64B flag 隔离、unique-id bootstrap、原生 SHMEM GET/PUT 及测试。
- 本库 `examples/deepseek_v4/int8_gemm.py`、`examples/dispatch_combine/dispatch_combine_shmem.py`。
- SHMEM 子模块 `157eecf7d2102c7282269db55e54362e90dbac7b` 的 device signal/wait/MTE GET、
  `examples/python_extension/test/unique_id_test.py` 的初始化方式。

实现和同步设计详见 `design.md`。静态检查记录见 `validation.md`。
