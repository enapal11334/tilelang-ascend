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
