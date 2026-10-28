# GEMM + AllReduce 初版设计

## 1. 语义与范围

目标：Ascend 910C，单机两卡，用户明确选择 Expert。每卡持有 K 维分片：
`A_r[M,K_local]`、`W_r[N,K_local]`，两卡输出均为
`Y = A_0 @ W_0.T + A_1 @ W_1.T`。输入 FP16；本地累加、跨卡传输、求和和输出均为 FP32。
不做平均。输入在 CPU 构造，不需要模型文件。

初版支持正整数且 M/N/K_local 分别被 32/64/64 整除；仅有限值、静态 shape、连续输入。
不支持尾块、转置属性、多机、两卡以外、autograd、并发复用 workspace。
这是单逻辑 Cube group 的通算重叠原型，尚无性能或 NPU 正确性结论。

## 2. 模式与参考

Expert：显式 L1/L0C/UB、C/V scope、地址规划、pipeline flag 和 cross flag；关闭四项自动
同步/划分/内存规划。以模式指南为准，不套用 Developer 的 CV 自动同步配置。

参考 mKernel `src/gemm_ar.cu` 的 compute producer / communication consumer：输出分块可用后
立即通知通信侧，而不是等完整 GEMM。CUDA 的 SM 角色在这里映射为 Cube / Vector 的固定分工。
初版不移植动态 SM 配比、NVLink multicast、跨机 reduction tree 或 GPU controller。
同时参考其 `examples/ascend_gemm_allreduce` 未经 NPU 验证的原型，以及本库
`examples/deepseek_v4/int8_gemm.py` 的 C/V workspace 交接和
`examples/dispatch_combine/dispatch_combine_shmem.py` 的 SHMEM 接入。

## 3. API 与算法

| 步骤 | TileLang / SHMEM |
|---|---|
| GM→L1，K 分片 GEMM，L0C→对称 GM | `T.copy`, `T.gemm_v0(init=kk==0)` |
| 本地生产/消费 | `T.set_cross_flag`, `T.wait_cross_flag` |
| 远端就绪与 ACK | `T.shmem_ub_put_nbi` 发布 epoch；`T.copy` + MTE2→S 轮询 |
| peer GM→UB | `T.decl_buffer(elem_offset=...)` + `T.shmem_ub_get_nbi` |
| 等待读取完成、UB 求和、输出 | `T.set_flag/wait_flag`, `T.tile.add`, `T.copy` |

每卡一个 launch。Cube 顺序生产 tile，Vector 等本地 FIX 完成后向 peer 发布 ready(epoch)，
等 peer ready，读取双方 FP32 tile、相加、写出。GET 完成后发送 ACK 并等待 peer ACK。
两卡都执行 pull+sum，因此两卡均获得完整结果，无额外广播 kernel。

## 4. 数据与资源

固定 BM/BN/BK=32/64/64；tile_count=(M/32)*(N/64)。

| 缓冲区 | shape / 大小 | 位置 |
|---|---|---|
| A/W tile | 32×64 / 64×64 FP16，共 12 KiB | L1 |
| accumulator | 32×64 FP32，8 KiB | L0C |
| local/peer tile | 各 8 KiB，共 16 KiB | 每 AIV 的 UB |
| epoch/poll control | 各 64B，共 128B | 每 AIV 的 UB |
| partial | [tile_count,32,64] FP32 | SHMEM 对称 GM |
| flags | [tile_count,32] int32，128B/tile | SHMEM 对称 GM |
| output | [M,N] FP32 | 普通 GM |

每 tile 的 ready 在 flags[tile,0]，ACK 在 flags[tile,16]，各占独立 64B scalar cacheline。
两卡必须同序、同大小分配 partial 和 flags；flags 首次运行前清零。保留 tensor 生命周期至完成。

## 5. Tiling 与约束检测

只使用一维 `T.Kernel(1,is_npu=True)`，不传非法 threads；tile/K 的 for 循环上界由 JIT
常量给出，状态轮询 while 根据运行时 epoch 退出，无设备侧超时。
L0C 8 KiB 小于既有 A2/A3 128 KiB 上限。GM↔L1/UB、L0C→GM 均遵循已有路径。
非整除输入在 host metadata 校验时拒绝，不用 host padding/crop；未声明尾块支持。

```text
[DISPATCH-COVERAGE]
supported_domain: aligned contiguous finite FP16 A/W, ranks 0/1, one node
generic_fallback: single Expert path covers the entire declared domain
specializations: none
unsupported_inputs: dimensions outside 32/64/64 alignment; other dtype/topology
result: pass (metadata/domain analysis only)
```

无物理布局重排。每 tile 每 K chunk 两次二维输入搬运，完整输出写一次 partial，local/peer
各读一次，写一次 output。无 numel 级 GM 标量数据循环；控制状态每轮搬运 64B 至 UB。
单 tile 控制/数据调度规模 O(1)，active Cube group=1；每核串行 tile_count 个任务。
关键哨兵 96×192×128：9 tiles、18 K chunks、36 次输入 tile copy、每卡传输 72 KiB peer 数据。

## 6. 流水线

两组 free token（cross flag 2/3）限制 Cube 最多领先两个 tile；ready token 用 0/1。
每个 tile 使用独立 partial 槽，消费者不阻塞下一 tile 的 GEMM。
只用 AIV0 通信和归约，两个 AIV 都参与所有本地 cross flag。固定两组 token，结束全部 drain，
包括 tile_count=1 时未用的初始 token，避免跨 launch 残留。

## 7. 同步与复用

`GM→L1 → barrier → GEMM → barrier → L0C→GM → FIX cross flag`。
Vector 远端 GET 后必须完成 MTE2→V，再求和；V→MTE3 完成后写 output。
ACK 必须在 GET 完成之后；接到 peer ACK 才返还 free token。Cube 退出前 drain 两组 token，
后续 launch 才能覆写 partial。epoch 两卡一致、正数、严格递增且 <2**31；无并发 workspace 复用。
不在部分核心中调用 SHMEM 集体 barrier。

## 8. 编译与 bootstrap

只复用仓库原生 SHMEM GET/PUT，不使用 call_extern、自定义 helper 或 codegen 修改。
GET 接收 Buffer，因此在循环内用 `T.decl_buffer(data=partial.data, elem_offset=tile*BM*BN)`
创建源视图；已有 SHMEM lowering 会通过 access_ptr 保留 elem_offset，不额外分配或拷贝。
ready/ACK 通过 64B epoch UB 的 SHMEM PUT 发布，完成 MTE3 后才进入等待或返还 token。
等待通过 GM→UB DMA 和 MTE2→S 同步读状态，避免 GM scalar DCache 轮询；复用 polling UB
前用 barrier 保证先前 scalar 读取完成。采用本库 dispatch_combine 的 DMA 状态轮询模式。
源码检查 GET/PUT 调用存在，并提供动态视图偏移 lowering 回归；都不替代 CANN 编译和运行。

host 设置 local device，Gloo 广播 SHMEM unique ID，调用 `aclshmem_init_using_unique_id`。
不依赖 InitAttr 的未暴露 comm_args。Gloo 只负责 bootstrap/测试同步，无输出数据 collective。
kernel 不使用 PyTorch/HCCL 做数学计算或 AllReduce。

## 9. 验证

CPU golden：`sum(A[r].float32 @ W[r].float32.T for r in [0,1])`。
L0：默认 64×128×64，零、一、rank 不同常数、固定 seed 随机数。
L1/哨兵：单 tile 32×64×64，奇数 tile 且两次 K 循环 96×192×128。
每个输入按正/负/正连续调用并复用 workspace/flags；轮流延迟一个 rank 的 launch，检查旧 epoch。
外层 timeout=600s（包含首次编译），单 shape 建议独立运行同样 timeout。

| 输入 dtype / 输出 dtype | atol | rtol | max abs | matched ratio |
|---|---|---|---|---|
| float16 / float32 | 2^-16 | 2^-10 | 1e-2 | ≥0.99 |

只声明有限值测试域；输出非有限值直接失败。CPU 单测验证解析式 golden、split-K 语义和比较器
能拒绝 local-only/stale 输出；language 回归测试检查原生 SHMEM GET 视图及 PUT 状态偏移。

## 10. 状态与风险

当前用户仅要求实现和静态检查；本机为 Windows、无 NPU/CANN，禁止将 CPU 结果称为双卡通过。
待真机验证：Bisheng 编译、910C 双卡精度、C/V 与跨卡可见性、重入 epoch、故意启动偏斜。
设备 wait 无超时；必须使用外部 watchdog。失败时由 torchrun 终止 peer，不进入集体资源释放。
单核心和单 AIV 通信限制吞吐，本轮不提供性能收益结论。

## 11. 交付与 host 审计

kernel、metadata config、CPU/NPU 测试、run.sh、proto.yaml、README.md，以及 SHMEM 视图
lowering 回归。算子 host 仅做整数/shape 检查与编译；测试输入/随机数/golden 在 CPU
生成，明确 H2D 初始化和 D2H 验证，无 NPU torch matmul、重排或后处理。

```text
[HOST-METADATA-AUDIT]
operator_host: shape/rank validation, source read, JIT launch; no tensor writes
test_preparation: CPU data generation and explicit H2D (including initial flags)
test_verification: D2H then CPU comparison
core_math_or_tensor_layout_on_host: none
result: source audit; device behavior pending hardware validation
```
