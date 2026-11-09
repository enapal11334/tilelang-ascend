# 910C 小 M GEMM + AllReduce 融合算子设计

## 1. 概述

### 1.1 目标与交付状态

算子名：`gemm_allreduce_910c`。目标平台为 Ascend 910C，单机 HCCS scale-up，
通信规模为 2、4、8 个 rank，每个进程绑定一个可见逻辑 NPU。物理卡与逻辑设备的对应关系、
互访能力和可用资源由部署环境核验，不假定“一张物理卡等于一个 rank”。

初始设计交付为本设计及 `proto.yaml`；后续代码开发已补入纯 JIT kernel、context、测试和
必要的前端扩展，文件见 §11。资源和性能数字仍为设计推导，不是设备通过记录。
本地只做了静态及 CPU 辅助检查，尚未执行 CANN 编译、910C 正确性或性能验证。

### 1.2 数学语义

每个 rank 持有沿全局 K 维切分的输入：

\[
A_r\in D^{M\times K_{local}},\quad B_r\in D^{K_{local}\times N},\quad
D\in\{\mathrm{float16},\mathrm{bfloat16}\}.
\]

先计算局部 FP32 GEMM，再执行 FP32 SUM，最终只转换一次：

\[
X_r=\operatorname{GEMM}_{FP32}(A_r,B_r),\qquad
Y_r=\operatorname{cast}_{D}\left(X_0+X_1+\cdots+X_{P-1}\right).
\]

所有 rank 获得完整的 `[M,N]`，不平均、不转置输出。归约按照全局 rank 从 0 到 P−1
顺序累加，不能各卡“先加自身，再加 peers”。不要求与任意 HCCL 归约树逐 bit 相同。
本地 GEMM 的 FP32 累加顺序受 Cube 实现影响，按 §9.3 容差验证。

### 1.3 支持域与边界

- 各 rank 的 M、N、K_local、输入 dtype 一致；M/N/K_local 为正整数，允许非整除。
- A、B 均为连续二维张量、dtype 相同；B 物理布局为 `[K_local,N]`。
- shape 在 context 创建时固定，按 shape/dtype JIT 特化；不引入 tensor 值驱动的动态 shape。
- 当前实现要求 `max(M*K_local,K_local*N,M*N)<=2^31-1`，避免 int32 元素偏移溢出。
- 限于目标设备可分配内存、当前后端地址与搬运参数可表达范围；索引乘积先使用宽整数检查，
  超范围明确报错，不做有符号截断或仅因 tile 非整除而拒绝。
- 首版不提供转置、bias、alpha/beta、低精度通信、autograd、多机或并发共享 workspace。
- FP16/BF16 的最终转换可产生合法 Inf；特殊值按 §9.3 位置契约比较。

### 1.4 性能目标与算法选择

优先小 M 的 decode 延迟。目标是完整输出 tile 就绪后立即通信，与后续 tile GEMM 重叠，
而不是对每个 K chunk 单独 AllReduce，也不是等待整张 GEMM 完成后才开始通信。

首版选择 **all-peer pull**：每卡读取其余卡的局部结果并本地求和，省去 owner 归约后
再次传播结果的协议阶段。已接受其在 P=2/4/8 时，相对按相同 padded payload 计算的
带宽最优 ReduceScatter+AllGather 分别有 1/2/4 倍全机数据量。首版不声称 8 rank
具有带宽扩展优势；owner/ring 算法属于后续独立优化。

验收同时要求正确性、时间线上的真实重叠以及关键小 M case 的实测端到端收益。
tile 太少或 GEMM 太短时，通信可能只被部分隐藏；Q=1 不存在后续 tile 可重叠。

### 1.5 数据流

```text
rank r: A_r/B_r → L1 → L0A/L0B → FP32 L0C → 对称 GM partial[g,slot]
                                                │ FIX 完成、本地 READY
                                                ▼
        peers 的对称 partial ── SHMEM GET ──→ FP32 UB 顺序求和
                                                │ 最终一次 cast
                                                ▼
                                          普通 GM Y_r[M,N]

Cube:   [GEMM tile j] [GEMM tile j+1] [等待/继续下一 tile]
Vector:               [通信/求和 j] [通信/求和 j+1]
```

host 不进行输入物理重排、padding 或输出裁剪；运行热路径只检查 metadata、保留序号和 launch。

## 2. 编程模式选型

用户已明确选择 **Expert**。需要显式控制 Cube producer、Vector consumer、双槽 ownership、
远端 READY/ACK 与完成顺序，自动同步不能替代这些协议。

| 维度 | 选择 |
|---|---|
| 内存 | 显式 `T.alloc_L1/alloc_L0A/alloc_L0B/alloc_L0C/alloc_ub` |
| 计算 | `T.copy`、`T.mma`、`T.tile.add`、`T.tile.cast` |
| 执行侧 | 手写 `T.Scope("C")` / `T.Scope("V")`，保留 `vid` |
| 同步 | 手动 pipe event、cross flag、SHMEM 完成与 READY/ACK |
| 并行 | 一维 `T.Kernel(G,is_npu=True) as (cid,vid)`，不指定大 threads |

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}
```

全部硬件操作必须位于正确的 C/V scope。该配置是本设计明确选择的全手动方案，
不是所有含 GEMM 或手写 scope 的算子的通用默认值。

## 3. API 映射设计

### 3.1 现有 API 与来源

下表表示当前源码中可查，不表示本设计已完成 DSL/CANN 编译或运行。

| 步骤 | 接口与使用约束 | 仓库依据 |
|---|---|---|
| 分配与地址规划 | `T.alloc_L1/L0A/L0B/L0C/ub`、`T.annotate_address` | `examples/gemm/example_gemm_intrinsic.py` |
| 有效输入与输出搬运 | `T.copy(src_region,dst)`，显式 valid extent | `src/op/ascend.cc`、`src/tl_templates/ascend/common.h` |
| Cube 计算 | `T.mma(a_l0,b_l0,acc_l0c,init=(k==0))` | `examples/gemm/example_gemm_intrinsic.py:94` |
| 核内依赖 | `T.set_flag(src_pipe,dst_pipe,event)` / `T.wait_flag(...)` | 同上及 `tilelang/language/ascend.py` |
| 本地 C/V 交接 | `T.set_cross_flag(pipe,id)` / `T.wait_cross_flag(id)` | `examples/gemm_allreduce/gemm_allreduce.py` |
| 远端连续读 | `T.shmem_ub_get_nbi(dst_ub,src_gm,nelems,pe)` | `examples/shmem/shmem_mm_test_ub_get_nbi.py` |
| READY/ACK 发布 | `T.shmem_ub_put_nbi(src_ub,dst_gm,nelems,pe,offset)` | `examples/shmem/shmem_mm_test_ub_put_nbi.py` |
| 带偏移对称视图 | `T.decl_buffer(...,data=partial.data,elem_offset=...)` | 现有 GEMM+AllReduce 原型；须补生成代码回归 |
| FP32 求和 | `T.tile.add(acc_ub[0:valid_m,:],acc_ub[0:valid_m,:],incoming_ub[0:valid_m,:])` | `tilelang/language/ascend_tile.py` 的连续 BufferRegion 支持 |
| 最终转换 | `T.tile.cast(out_ub,acc_ub,"CAST_RINT",count=valid_m*BN)` | `examples/deepseek_v4/int8_gemm.py` |

SHMEM 的 `nelems` 和 `offset` 按接口要求使用元素数，不能混用字节数。
当前 UB GET/PUT 的底层为 `aclshmem_ptr` 远端地址解析及 `DataCopyPad`，属于 MTE 直访远端 GM，
不将其描述成已经实现的 RDMA/SDMA 通信路径。

### 3.2 实现前置扩展：`T.shmem_quiet()`

**已新增源码，需重新编译安装本仓库。** 无参数的 V 侧同步接口，codegen 直接调用已有
`aclshmem_quiet()`。注册为 opaque 副作用操作，并在 C/V 分离中归属 Vector，
codegen 拒绝 Cube scope；补入调用保留/顺序/归属回归，实际回归尚未运行。
不使用未声明的 `call_extern` 绕过正式接口。

依据：`3rdparty/shmem/include/device/gm2gm/shmem_device_mo.h:24-32` 声明纯 HCCS scale-up
上的完成与全局可见性契约；`src/device/gm2gm/shmem_device_mo.hpp:17-26` 实现为
`PipeBarrier<PIPE_ALL>()` 加 `dcci_entire_cache()`。
`T.barrier_all()` 只有当前执行核的 pipe barrier，不等价于 quiet 或跨 rank barrier。

每 tile 固定三次 quiet，位置见 §7。V quiet 不替代 Cube FIX 完成；
跨 C/V 发布链仍必须先通过 910C 可见性门禁，不能只凭函数名宣布正确。

### 3.3 调度伪代码

以下为算法级伪代码，`produce/read/publish` 等名称不是新增 DSL API。

```text
context: BM=16, BN=128, BK=128, n_tiles=ceildiv(N,BN)
Q=ceildiv(M,BM)*n_tiles; J=ceildiv(Q,G)
for each resident group g in parallel:
    Cube:
        initialize K-pipeline free events
        for j in [0,J):
            tile=g+j*G
            if tile<Q:
                slot=j%2; wait local FREE[slot]
                m=(tile//n_tiles)*BM; n=(tile%n_tiles)*BN
                valid_m=min(BM,M-m); valid_n=min(BN,N-n)
                for k in [0,ceildiv(K_local,BK)):
                    valid_k=min(BK,K_local-k*BK)
                    load valid A/B into zero-padded L1 version k%2
                    transfer L1→L0A/B; mma(init=(k==0))
                wait M→FIX; copy full physical L0C tile to partial[g,slot]
                publish local READY[slot] only after FIX completion
        drain both FREE tokens and K-pipeline events
    Vector (both AIVs):
        publish initial FREE[0], FREE[1]
        for each valid tile in the same order:
            wait local READY[slot]
            AIV0: perform §7 cross-rank protocol with seq=seq_base+j+1
            both AIVs: return local FREE[slot]
```

### 3.4 约束检测与同类实现差异

| 检测 | 设计处理 |
|---|---|
| 三维 Kernel | 仅一维 G；tile 在组内解码，不使用三维 launch |
| threads | 不传 GPU 风格线程数，沿用显式 `(cid,vid)` Expert 结构 |
| tensor 值动态循环边界 | Q/J/K 循环为 JIT 常量；设备 while 只用于已有示例支持的状态轮询 |
| GPU API | 不使用 CUDA/NVSHMEM 原语；T.mma 与 SHMEM 取自本库 |
| 非整除 | M/N/K 全部 ceildiv，输入和输出显式 valid 区域 |
| L0C | 16×128×4=8192 B，仍需部署资源核验 |
| 隐式 host 拷贝 | 禁止 host pad/crop/contiguous/转换与 tensor 计算 |

主要参考：`examples/gemm_allreduce/` 的双卡协议、`examples/gemm/example_gemm_intrinsic.py`
的显式 Cube 流水、`examples/shmem/` 的四个原生搬运例子。
现有 GEMM+AllReduce 原型只支持双卡 FP16、`W[N,K]` 转置 B、FP32 输出、整除形状和单 group；
本设计不直接复用其窄域声明，也不覆盖或改写该目录。它引用的
`testing/python/language/test_tilelang_ascend_language_shmem_view.py` 当前不存在，
不能当作已有通过回归。

## 4. 数据规格与内存规划

### 4.1 输入与输出

| 参数 | shape | dtype | 布局与语义 |
|---|---|---|---|
| A | `[M,K_local]` | float16 / bfloat16 | 连续 row-major，各 rank 持有 K 分片 |
| B | `[K_local,N]` | 与 A 相同 | 连续 row-major，不做 host 转置 |
| Y | `[M,N]` | 与 A 相同 | 普通 GM，独立输出，每 rank 完整复制 |

支持 dtype 全集以 §9.3 为准。FP32 只用于内部累加、workspace 和通信，不是额外公开输入 dtype。

### 4.2 对称与普通 GM

| buffer | shape / dtype | 大小 | ownership |
|---|---|---|---|
| partial | `[G,2,16,128]` FP32 | 16384G B | 本 rank Cube 写；本地及所有远端读者消费 |
| flags | `[G,2,2,P,16]` int32 | 256GP B | kind=READY/ACK；每源 rank 独占 64B 行 |
| Y | `[M,N]` FP16/BF16 | 2MN B | 每个有效输出 tile 仅本 rank 对应 group 写 |

partial 与 flags 在各 rank 上按相同顺序、大小、对齐分配。每行控制数据 64B 对齐，
不得让不同发送者共享或覆盖同一控制行；自身 rank 对应行保留但不轮询。
flags 仅在新 context 中用 TileLang 初始化 kernel 清零，完成后才能开始首次调用；
不在热路径用 `torch.zeros/fill_` 重置状态，不在复用时回零序号。

### 4.3 每组片上预算

FP16 与 BF16 均为 2 B，物理形状固定：

| 存储区 / buffer | 形状与版本数 | 字节 |
|---|---|---:|
| L1 A | 两个独立 `[16,128]` buffer | 8192 |
| L1 B | 两个独立 `[128,128]` buffer | 65536 |
| **L1 总计** | | **73728（72 KiB）** |
| L0A | 两个 `[16,128]` buffer | 8192 |
| L0B | 两个 `[128,128]` buffer | 65536 |
| L0C | 一个 `[16,128]` FP32 | 8192 |
| 每 AIV UB acc / incoming | 两个 `[16,128]` FP32 | 16384 |
| 每 AIV UB out | 一个 `[16,128]` FP16/BF16 | 4096 |
| 每 AIV UB seq / poll | 两个 `[16]` int32 | 128 |
| **每 AIV UB 总计** | | **20608** |

L0A 与 L0B 是不同存储区，不能合并预算后误判；目标 CANN platform 配置需核实
L1/L0/UB 容量与编译器实际保留空间。使用 `T.annotate_address` 在各区域内部做不重叠布局，
两个版本各有独立逻辑基址。未核实容量或生成代码不满足预算则禁止开始性能测试。

### 4.4 完整搬运路径

```text
GM A/B → L1 A/B → L0A/L0B → T.mma → L0C → FIX → 对称 GM partial
本地/远端 partial → MTE2 → acc/incoming UB → FP32 add → cast → out UB
out UB → MTE3 → 普通 GM Y
seq UB → SHMEM PUT/MTE3 → peer flags；本地 flags → MTE2 → poll UB → scalar 比较
```

没有 GM→L0 越级搬运，也不声称 Cube→Vector 硬件片上直连。

### 4.5 公开入口与 context

为减少冗余接口，落地入口由拟定的三参数函数简化为固定配置 context 的直接调用：

```python
with GemmAllReduce910CContext(A, B) as comm:
    Y = comm(A, B)
    comm.stream.synchronize()  # A/B/Y 保持存活至该 stream 完成。
# 同一作业所有 context 关闭后，全 rank 共同调用一次。
finalize_shmem()
```

`GemmAllReduce910CContext` 位于 `comm.py`，从 A/B 推导固定的 M/N/K_local/dtype/P/G、
逻辑设备和 stream，持有对称 workspace、JIT callable 和 invocation counter。
原始 callable 显式接收 `(A,B,partial,flags,Y,seq_base)`，`out_idx=[]`，不设 `workspace_idx`：
对称 workspace 由 context 提供，不能让普通 JIT workspace 分配器代为分配普通 GM。

setup 流程：调用方设置本地设备并初始化 Gloo → 查询各 rank 当前设备的 Cube 数、核对
全部 metadata/hostname、取共同 H → 首个 context 广播 unique ID 并初始化进程级 SHMEM
→ 同序对称分配 → 初始化 flags → 预编译并确认全部 rank ready。
参考现原型的 unique-ID bootstrap，不照搬 shmem 例子中的固定 IP、硬编码 8 rank。

metadata 核对包含 shape、dtype、world_size、tile、G、协议版本、workspace 字节数与初始序号。
创建或切换配置时才做 host collective。热路径各 rank 按相同调用顺序本地保留新序号区间，
不做 Gloo 往返；不一致调用属于 collective 使用错误，由外部 watchdog 终止。

context 不允许跨 stream 并行复用；允许绑定 stream 上顺序排队。
所有 tensor/context 保留至异步调用完成。正常 context 销毁先等待本地 stream 和全 rank
完成再释放 workspace；进程级 SHMEM 仅在全部 context 关闭后由 `finalize_shmem()` 终结一次，
终结后不再创建 context。底层 default bootstrap 的 finalize 关闭但不清空 plugin handle，
因此不采用“每个 shape 都 init/finalize”的重入假设。
失败或超时走 worker 终止流程，不尝试可能阻塞的集体释放。

### 4.6 Host metadata 审计

```text
[HOST-METADATA-AUDIT]
operation: shape/stride/dtype/device checks, integer scheduling, seq reservation
input_stride -> output_stride: A=(K_local,1), B=(N,1); input storage untouched
shares_storage / same_data_ptr: true for inputs; no input replacement/view needed
aclnn_or_physical_copy: false
result: allow (design audit)

operation: allocate scratch/output; initialize flags via TileLang kernel
input_stride -> output_stride: not a tensor transform; separate owned allocations
shares_storage / same_data_ptr: input unchanged; buffers intentionally distinct
aclnn_or_physical_copy: no aclnn computation; initialization is explicit kernel work
result: allow

operation: test preparation/verification and framework comparison baseline
input_stride -> output_stride: CPU data/golden, explicit H2D/D2H outside operator
shares_storage / same_data_ptr: not an operator preprocessing path
aclnn_or_physical_copy: H2D/D2H explicit; framework baseline isolated from operator
result: allow as harness only; no NPU torch math in the operator path
```

### 4.7 影响计算路径的属性（proto 派生源）

这些属性描述 builder/context，不是公开入口增加同名 kwargs。dtype 从 §9.3 派生。
M/N/K_local 从输入 shape 推导；seq_base 属于调用状态，不是可任意覆盖的用户 attr。

| name | type | default | required | 约束与来源 |
|---|---|---|---|---|
| world_size | Int | null | true | §1，2/4/8 |
| rank | Int | null | true | §1，0<=rank<world_size |
| cube_groups | Int | null | true | §5，必须等于共同 metadata 计算的 G |
| block_m | Int | 16 | false | §5，首版固定值 |
| block_n | Int | 128 | false | §5，首版固定值 |
| block_k | Int | 128 | false | §5，首版固定值 |
| pipeline_slots | Int | 2 | false | §6，正式路径固定2；1仅用于串行诊断基线 |
| algorithm | Enum | all_peer_pull | false | §1，首版唯一通信算法 |
| reduction | Enum | sum | false | §1，固定 SUM |
| protocol_version | Int | 1 | false | §7，所有 rank 一致 |

## 5. Tiling 策略

### 5.1 任务映射与支持域覆盖

```text
BM=16; BN=128; BK=128
MT=ceildiv(M,BM); NT=ceildiv(N,BN); Q=MT*NT
H=min(current_device_cube_core_num[r] for all ranks)
G=min(H,max(1,Q//2)); J=ceildiv(Q,G)
tile=g+j*G; m=(tile//NT)*BM; n=(tile%NT)*BN
```

资源查询失败即报错，不回退硬编码 20 核。G 不超过共同可用 group 数，并要求本算子执行时
这些 group 可驻留，避免等待还未调度的 producer。Q≥2 时每个 group 至少有两个有效 tile；
Q=1 时 G=1，仅融合不承诺重叠。奇数 Q 的最后一轮以 `tile<Q` 保护，C/V 分支完全相同。

```text
[DISPATCH-COVERAGE]
supported_domain: contiguous matching FP16/BF16 A[M,K_local], B[K_local,N];
                  positive shapes within device/backend limits; P in {2,4,8}
generic_fallback: single Expert path using ceildiv, valid extents and zero-padded L1
specializations: none; two dtypes share scheduling and FP32 wire format
unsupported_inputs: noncontiguous/mismatched dtype or shape, other P/topology,
                    nonpositive or unrepresentable dimensions, concurrent context reuse
result: pass (design coverage; hardware correctness unverified)
```

### 5.2 尾块与物理布局

每次搬入使用 `A[m:m+valid_m,k0:k0+valid_k]`、
`B[k0:k0+valid_k,n:n+valid_n]` 的显式 BufferRegion。L1 保持物理 BM×BK、BK×BN，
无效位置归零；每个 tile 的首个 `T.mma` 使用 init=True，后续 K chunk 累加。

`common.h:56-105` 的 GM→L1 helper 支持真实 pitch、valid 行列和 NZ fractal 清零；
`pad_value=0` 不能泛指此 L1 路径，它在另一条 GM→UB 路径使用。
两个 L1 版本必须是独立逻辑 buffer，并从各自逻辑 offset=0 完整写入，避免非零子切片
被 `need_clear=false` 判定后遗漏尾部清零。生成代码须包含 A、B 两类 GM→L1 搬运，
并对每条实际生成的调用检查 `need_clear=true`、地址不重叠和完整 fractal 容量。
单个 K tile 的预取分支可被消除。若编译器合并或改写 allocation，
必须重新证明这些条件，不能仅凭 Python 中声明了四个变量视为保证。

L0C 可完整写入私有物理 partial tile；SHMEM 与本地读只消费前 `valid_m*BN` 个 FP32 元素。
N 尾保留每行 BN=128 的物理 pitch，未使用列不写入最终输出。
`T.tile.add` 无 count 参数，三个操作数必须显式使用 `[0:valid_m,:]` 整行前缀，
保持连续并避免读取未搬入的 UB 行；不能只在文字上宣称整 buffer 调用自动忽略无效行，
也不能改成多行窄列 `[:,0:valid_n]` 的非连续 region。
落地时发现 `binary_op` 对等价 symbolic count 使用 Python 对象身份比较，已将两处尺寸
断言改为 `arith.Analyzer().can_prove_equal`，并增加等价/不等价动态前缀回归；尚未执行真实前端测试。
cast 处理有效行前缀；`out_ub` 仍为 `[16,128]`，通过
`T.copy(out_ub,Y[m:m+valid_m,n:n+valid_n])` 写有效区域，不把 UB 重声明为窄行。

当前后端 UB→GM 的 GM pitch 为字节粒度，UB pitch 为 32B 单位；固定半精度 UB 行宽
256B 满足源侧约束。GM→L1 的 ND2NZ 使用原始元素 stride，N/K 为奇数不做 stride 截断。
极大 row stride 进入 CATLASS 逐行 ND2NZ 路径时须单独验收，不能因源码存在就声称已通过。

### 5.3 数据搬运与通信成本

无全张量转置或物理重排；以下统计是逻辑 DMA 发出次数，不把一次二维 DMA 等同于
单个总线 transaction，实际 ND2NZ 拆分、cache 命中和链路 transaction 由 profiler 核实。

记 `L=ceildiv(K_local,128)`、`S=M*N_pad`、`N_pad=128*NT`：

| 每 rank 项目 | 次数 / 字节 |
|---|---|
| GM→L1 输入 | 2QL 次二维 copy；有效字节 `2K_local*(NT*M+MT*N)` |
| L1→L0 | 2QL 次；固定物理 tile，K 尾补零 |
| L0C→partial | Q 次，8192Q B（完整物理 tile） |
| 本地 partial→UB | Q 次，4S B |
| peer partial GET | (P−1)Q 次，4S(P−1) B |
| 最终 UB→Y | Q 次，2MN B |
| READY/ACK PUT | 2(P−1)Q 次，共 128(P−1)Q B |
| 状态轮询 | 每轮 64B DMA；次数依 rank skew，须测量，不能假定为1 |
| quiet | 3Q 次，仅 active AIV0 |

数据路径无逐元素 GM scalar load/store；UB scalar 只读控制字。
每 tile 一次 div/mod 解码 m/n，地址计算不按 numel 执行。
G 个 Cube 与 G 个 AIV0 active，各组串行 tile 数为 floor/ceil(Q/G)。

下表使用 P=8；MiB=2^20 B。与 dtype 无关的通信均为 FP32，两种输入 dtype 都需测。

| (M,N,K_local) | Q / L | 输入 copy 数 / 平均有效字节 | 输入总量 | peer GET 数 / 平均字节 | peer 数据总量 | 控制 PUT 总量 |
|---|---|---|---|---|---|---|
| (1,8192,4096) | 64 / 32 | 4096 / 16512 B | 64 MiB + 512 KiB | 448 / 512 B | 224 KiB | 56 KiB |
| (8,4096,1024) | 32 / 8 | 512 / 17408 B | 8 MiB + 512 KiB | 224 / 4096 B | 896 KiB | 28 KiB |

例如共同 H=24 时，上述两例 G 分别为24和16，每组处理2–3或2个 tile；H=24仅为计算示例，
不是对910C核数的断言。M=1 仍有大量512B GET和64B控制消息，属于明确的延迟风险。

全机远端数据为 `4S*P*(P−1)`，带宽最优 RS+AG 按同一 padded payload 为 `8S*(P−1)`，
两者比值 P/2。N 很小时仍传物理128列，这部分浪费不靠 host padding 隐藏。

```text
[REORDER-COST]
path: shared FP16/BF16 Expert tile path; no standalone reorder kernel
gm_passes: tiled GEMM input rereads as above; one partial store/local read,
           P-1 peer reads and one output store; no full-input staging pass
dma_transactions: logical 2QL input copies + 2QL L1/L0 copies + (P+2)Q
                  partial/read/output copies + 2(P-1)Q control PUTs + polls
average_dma_bytes: per-class values/formulas above; polls are 64 B
gm_scalar_accesses: 0 for tensor data and mutable flags
address_div_mod_per_element: 0; tile decode once per task
active_cores / serial_tasks_per_core: G Cube + G AIV0 / floor-or-ceil(Q/G)
largest_case_timeout_gate: both key cases, both dtypes, P=8, warm 120 s/case
result: analytical cost recorded; device feasibility/performance gate NOT RUN
```

## 6. 循环与调度结构

### 6.1 Cube K 流水

使用双 L1 和双 L0A/B、单 L0C，固定 BK=128，每个 K chunk 对应一次 `T.mma`。
参考 intrinsic 示例的事件结构，去掉 `core_num=20`、`kk==3` 和整除索引：

1. 初始化两个 L1 free 事件 MTE1→MTE2、两个 L0 free 事件 M→MTE1，以及 L0C free FIX→M。
2. MTE2 获取版本 free 后搬入 A/B，置 MTE2→MTE1 ready；预取下一 K chunk 时遵守同一规则。
3. MTE1 等待 L1 ready 与 L0 free，完成两次 L1→L0 copy，归还该 L1 free，置 MTE1→M ready。
4. M 等待 L0 ready；首个 chunk 还需等待上个 tile 的 FIX→M，执行 init=True 的 mma。
   之后每个 chunk init=False；M 完成后归还 L0 free。
5. 所有 K 完成后 M→FIX，FIX 写 partial，产生 FIX→M 与本地 cross READY。

事件版本使用 K chunk 的 `k%2`，不能将每个 tile 重置的内层 `kk%2` 错当全程流水版本。
相同 pipe-pair 的事件按版本区分，MTE1 在读取 A/B 两者完成之后才返还 L1 free。
独立命名的 L1/L0 版本通过 DSL `if k%2==0` 分支或可证明的静态展开选择；
禁止用运行时 PrimExpr 索引 Python buffer 列表，也不回退成非零 offset 的合并 L1 子切片。
先完成所有在途操作再清理初始 free 事件，包括只用一个 K chunk 时未使用的版本。
K 流水事件跨 tile 保留、在 kernel 退出时统一 drain；不得在仍有 FIX 时重新初始化。
不使用 `T.Pipelined`、自动跨核流水或持久化动态任务队列。

### 6.2 双槽与本地 C/V token

READY 使用 cross flag 0/1，FREE 使用2/3；属于独立于核内 pipe event 的编号域。
两个 AIV 启动时各发布两个 FREE；每个有效 tile 均执行 READY wait 与对应 FREE return。
AIV1 可以提前到达 FREE，但 Cube 只有收到该 cross flag 所要求的两个 AIV 参与后才能复用；
生成代码及真机门禁必须确认计数语义，不得删除 AIV1 的参与。

Cube 退出时等待两个 FREE，将未使用槽位的初始 token 也消费干净；Vector 不等待不存在的
末尾 tile。C/V 都用相同的 `tile<Q` 判断，防止奇数 Q 或末轮空任务造成握手不配平。

### 6.3 序号与重入

`J=ceildiv(Q,G)`，初次 `seq_base=0`，第 i 次调用 `seq_base=i*J`；当前 tile 使用
`seq=seq_base+j+1`，所有 rank 相同。context 在 enqueue 时保留区间，即使异步也不重复使用。
每次调用要求 `seq_base+J<2^31`，接近上限则所有 rank 完成后销毁并重建 context，不能原地回零。

轮询必须 `seen==seq`，不能将旧 epoch 或未来世代视为成功。调用顺序错误、世代损坏或
任一 peer 失败可能使设备 wait 不退出，统一由外部 watchdog 终止全部 workers。
普通调用不清零 flags，旧值因不同 seq 自然失效。

## 7. 同步策略

### 7.1 READY/GET/ACK 协议

每个有效 tile 由 AIV0 执行以下固定顺序；slot=j%2，seq 为 §6.3 的值：

1. 等待 Cube 的本地 READY。该 READY 由 FIX 完成触发，不能提前到仅 enqueue partial 写回。
2. **quiet #1：发布前 release。** Cube FIX 完成及 C/V 交接是前提，V quiet 不代替它们。
3. 在 seq UB 填入当前 seq，完成 V→MTE3；向所有 peers 的
   `flags[g,slot,READY,self_rank,:]` PUT 同一个64B控制行。
4. **quiet #2：完成全部 READY PUT。** 发布前 quiet 不能省略成只有发布后 quiet，
   因为 peer 可能已经观察到 READY；控制 UB 在此完成之前保持不可变。
5. 对 p=0…P−1：p=self 时从本地 partial DMA；否则轮询本地
   `flags[g,slot,READY,p,0]==seq` 后 GET p 的对应 partial。
   p=0 的数据直接初始化 acc UB，其后逐项 FP32 add，禁止改变全局加法顺序。
6. 每次 MTE2→V 完成后才读取/求和；复用 incoming UB 前确认前一向量读取完成（V→MTE2）。
   add 的三个操作数显式使用 `[0:valid_m,:]`，cast 指定 `count=valid_m*BN`，
   经 V→MTE3 写有效 Y 区域并确认输出完成。
7. 全部 GET 完成后，向所有源 ranks 的 `flags[g,slot,ACK,self_rank,:]` PUT seq。
   ACK 前明确完成 GET→归约→V→MTE3→输出完成的链；不能把 GET 函数返回或仅 MTE2→V wait
   当成 ACK 所在 MTE3 已被正确排序。若重排为提前 ACK，必须另证 MTE2→MTE3 依赖，本版不做。
8. **quiet #3：完成全部 ACK PUT。** 再轮询本地全部 reader 的 `ACK==seq`。
9. 全部读者 ACK 收齐后，AIV0 返还本地 FREE；AIV1 完成相同握手，Cube 才可覆写该槽。

仅一个 writer 写入每条 READY/ACK 行；不是对同一字做多卡原子累加。
所有 rank 先发布全部 READY，再读取并发布全部 ACK，最后才等 ACK；
不在逐 tile 流水或部分核心分支中调用全 rank collective barrier。

### 7.2 轮询与缓存

状态读采用本地 GM flags→poll UB 的64B DMA，MTE2→S Set/Wait 后读取 UB 第0元素。
poll UB 再次被 DMA 覆写前，先完成前一次 scalar 读取；可使用当前执行侧 pipe barrier。
禁止直接 `while flags[...]` 做可缓存的 GM scalar 轮询。

DMA 避免 mutable GM scalar DCache 轮询问题，但 SHMEM 的 L2 参数并不等于禁用所有 cache。
当前 GET/PUT 为异步发出；quiet 与 pipe event 的具体完成性，以及 FIX→READY→GET 的
跨核可见性需由 §9.2 的 payload/epoch 压力门禁验证。失败时先诊断接口或顺序，
不得通过加 host barrier、每 tile 全卡 barrier 或忽略旧数据来宣布通过。

## 8. CV 融合设计

这是 Cube GEMM、Vector 通信/归约/转换的 CV 融合，采用显式对称 GM workspace 中转。
一个 rank 的一次计算调用只 launch 一个融合 kernel；context 初始化与性能诊断独立进行。

| 执行侧 | 输入 ownership | 工作 | 释放条件 |
|---|---|---|---|
| Cube | 收到 FREE 的 partial 槽 | 完整 K 累加，FIX 写回 | FIX 完成后交给 V/peers |
| AIV0 | READY 指定世代的各 rank partial | 发布、GET、FP32 SUM、cast、写 Y、ACK | 全部 readers ACK 后返还 FREE |
| AIV1 | 本地 cross flag 参与者 | 初始化、等待、返还 token | 每次 valid tile 配平 |

partial 生命周期为 FREE→Cube 写→FIX 完成→远端可读→全部 reader 读完→FREE。
ACK 只说明读者已不再依赖源槽，不依赖跨 stream host 同步；输出写完是本版 ACK 的保守前提。
两个槽使 Cube 最多领先两个在途 tile；单 L0C 的复用另由 FIX→M 事件保护，不能用 FREE 代替。

诊断串行基线固定为一个在途 tile：相同 G、tile、math、三次 quiet 和 GET/ACK 协议，
每次固定使用 slot=0，不采用 `j%2`；Cube 每轮只获取一次 FREE0，再生产当前 tile，
两个 AIV 只发/等对应 READY0/FREE0。第二槽仅初始化 FREE1 并在退出时 drain。
正式入口使用双槽，单槽只作基准配置，不能当作第二个未说明的生产调度路径。

## 9. 验证方案

### 9.1 CPU Golden

```python
def golden_gemm_allreduce_910c(a_by_rank, b_by_rank):
    # 输入已在 CPU 按目标 FP16/BF16 量化；调用顺序即全局 rank 顺序。
    import torch

    dtype = a_by_rank[0].dtype
    result = None
    for a, b in zip(a_by_rank, b_by_rank, strict=True):
        local = a.to(torch.float32) @ b.to(torch.float32)
        result = local if result is None else result + local
    return result.to(dtype)
```

CPU 输入实际值先量化再计算；不把任意高精度未量化输入当作本算子的输入。
测试必须使用 rank 不同的数据，包含能区分 FP32 通信与提前半精度舍入的确定性样例。
Golden 与随机数、特殊值构造在 CPU 完成，operator 运行段只做 H2D→TileLang→D2H。

### 9.2 L0 计划与协议前置门禁

以下 L0 用例为规则、block 整除形状；两种 dtype 与2/4/8 rank均覆盖：

| 用例 | (M,N,K_local) | 核查目标 |
|---|---|---|
| l0_basic | (16,256,128) | 两个 tile、初始双槽、一次 K chunk |
| l0_multigroup | (32,512,256) | 多 group、多 K chunk 与不同 rank 数据 |

输入组合包括零、一、不同 rank 常数、固定 seed 随机数；检查所有 rank 的完整输出，
不能只检 rank0 或全部 rank 相同输入。完整 L1/L2/Boundary 套件由后续
`tilelang-op-test-design` 根据实际实现生成；本设计只规定必须覆盖的场景和少量门禁锚点。

**先于性能测试执行的发布/复用门禁：** P=2/4/8，每组固定两槽，至少4096代，
每代 payload 可区分 generation/tile/rank（整数保持在 FP32 精确表示范围内）。
轮流延迟 producer、consumer 和一个 rank 的 launch，验证看见新 READY 后必能读取对应新 payload，
且 GET 完成及全部 ACK 之前生产者绝不覆写。实现采用真实 GEMM 的4096次调用：
`A[0,0]=2^rank`，B 的有效行编码12个 generation bit、rank marker和3个 tile-id bit，
SUM 的整数值不超过255，在FP16/BF16最终输出中保持精确；因此不向生产kernel添加诊断模式。
测试覆盖端到端槽位复用及rank启动偏斜，独立的核内producer/consumer延迟仍需设备级检查。
若失败，禁止进入性能验收。

同时要求覆盖单 tile、奇数任务、两个槽多次复用、跨调用旧 seq、全部 token drain，
并用 `(1,67,65)`、`(3,67,65)`、`(17,129,17)` 锚定 M/N/K 尾块与窄输出 pitch。
读取生成代码确认两版 L1 均清零；后续测试覆盖跨 ND2NZ stride 上限路径及非法 metadata。
这些锚点不是已运行的测试，也不代替后续完整分层套件。

### 9.3 精度标准（支持 dtype 全集）

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio |
|---|---|---|---|---|
| float16 | 2^-14 | 2^-9 | 0.1 | 0.99 |
| bfloat16 | 2^-10 | 2^-6 | 1.0 | 0.99 |

有限值逐元素采用 `abs(actual-golden)<=atol+rtol*abs(golden)`，同时要求匹配率和最大误差通过。
先在完整输出上分别严格比较 NaN、正 Inf、负 Inf mask，再只对双方有限的位置计算误差。
不能复用旧原型“任何非有限输出都失败”的比较器。

特殊值输入采用有限值与稀疏 NaN/±Inf 混合，确保存在特殊位置和有限输出，
另覆盖最终 FP16 cast 溢出；全 NaN/全 Inf 只能作为补充，不能掩盖索引或数据重排错误。
还需让比较器拒绝 local-only、漏 rank、错误 rank、旧 seq 和提前半精度舍入的错误结果。

### 9.4 性能可行性哨兵与收益门槛

| 用例 | (M,N,K_local) | dtype / P | 覆盖与超时 |
|---|---|---|---|
| decode_m1 | (1,8192,4096) | FP16/BF16 × 2/4/8 | 最短有效行、较多tile/控制消息；warm 120 s/case |
| decode_m8 | (8,4096,1024) | FP16/BF16 × 2/4/8 | 更多通信字节、短K；warm 120 s/case |

两例及最高 P、两种 dtype 都是必跑哨兵，不能归入默认跳过的 large/L1。
未声明有限最大 shape，部署方的最大输入超出这两例时应增加对应资源/timeout 哨兵，
不得把上述关键 case 说成所有正整数 shape 的性能上界。

测量三组实现：

1. 正式双槽流水；
2. §8 的同 G、同计算与通信协议、单在途 tile 串行版本，用来隔离重叠收益；
3. 分离 GEMM→框架/HCCL FP32 AllReduce→最终 cast，同输入、同 SUM、同输出 dtype。
   为保持舍入契约，复用相同 TileLang local GEMM 输出 FP32，不能用默认半精度输出的
   框架 matmul，也不能把输入改成 FP32 来冒充同计算路径。该基线独立于 operator 路径，
   计入全部独立 launch 和最终 cast；不把低精度通信当作等价基线。

预编译及首次设备初始化不计入 warm 延迟。预热30次，再做10轮、每轮100次独立调用，
每次使用新序号；各 rank 同时参与，按最慢 rank 的设备完成延迟评价，不能取 rank 平均值。
host barrier、CPU Golden 和 profiler 采样不进入被计时 kernel；另报告 host enqueue 成本。
同一 workspace 的调用严格顺序执行，不以多流并发隐藏未完成的工作。

报告 p50、p95、波动、相对两种 baseline 的 speedup。采用固定 seed 的10000次分层 bootstrap
（先按测量轮重采样，再按轮内样本重采样）给出相对串行版本的 p50 speedup 95%区间；
每个关键 case 的下界须大于1才记为“实测收益通过”。否则记录真实结果，继续调优，
不得只挑有利 rank、dtype 或单次最快结果宣布目标完成。框架 baseline 如实对比，不隐去退化。

单独 profile 核查 tile t 的跨卡 MTE/归约与同组后续 Cube GEMM 有正时长交集，
并报告各阶段和未隐藏通信尾部。无归属证据的总时间下降不能单独证明跨卡搬运发生重叠。
Q=1 单独报告融合延迟，不纳入“后续 tile 通算重叠”通过数量。

### 9.5 执行顺序与超时

静态/生成代码检查 → CANN 编译 → 可见性门禁 → 2 rank 正确性 → 4 rank → 8 rank
→ 协议复用/尾块门禁 → 性能哨兵与时间线。所有步骤均为后续执行计划。

冷编译每 case 外部 watchdog 600秒，预编译后每 case 120秒；设备轮询本身无超时。
一个 worker 超时或失败时终止全部 workers，保存阶段、rank、group、seq 与日志；
不进入可能继续等待失败 peer 的 collective cleanup。

## 10. 风险点与注意事项

| 风险 | 处理与停止条件 |
|---|---|
| quiet 需重新编译安装 | 源码已接入，但未在本机编译或运行；旧安装包不包含此intrinsic |
| FIX/CV/HCCS 可见性 | 先运行代际 payload 门禁；未通过不调优，不靠 host barrier 掩盖 |
| 小 M 短消息和三次 quiet 开销 | 测量512B GET、64B控制与quiet成本；不能承诺所有shape全隐藏 |
| 8 rank 冗余数据 | 明确4倍带宽代价；后续RS+AG需要独立协议设计，不悄然切算法 |
| G 减少并行计算核数 | 同 G 的串行版本隔离重叠收益，框架基线反映整体代价 |
| 双缓冲尾块残留 | 每版独立逻辑基址、生成代码清零检查、奇数K测试 |
| UB 窄行或输出越界 | 物理pitch固定，GM valid区域显式，检查相邻tile/guard区域 |
| AIV1 token 或退出 drain 缺失 | 两个AIV完整参与；测试单tile、奇数tile、跨调用复用 |
| 某 rank 未参与/失败 | 标准collective同序调用契约，外部watchdog终止全体 |
| 资源与极大stride | 当前device/platform查询与预编译验收；超出可表达范围明确拒绝 |

当前只能确认仓库接口和算法推导。910C 的具体 SKU、CANN/SHMEM 版本、核数和互联拓扑，
以及多 group 并发 SHMEM/quiet 的实际行为，需要部署时核验；不据此虚构验证结论。

## 11. 交付清单

### 11.1 文件

```text
examples/gemm_allreduce_910c/
├── design.md                     # 设计与落地差异
├── proto.yaml                    # dtype/attrs规格
├── gemm_allreduce_910c.py         # 纯JIT融合kernel、flags初始化、共享Cube宏
├── comm.py                       # 固定shape context及进程级SHMEM生命周期
├── baseline.py                   # 同Cube计算的FP32 local GEMM与末尾cast
├── test_gemm_allreduce_910c.py    # 精度、协议和性能入口
├── README.md                     # 运行方式
└── validation.md                 # 实际检查与未验证项
```

保留 `examples/gemm_allreduce/` 原型。
proto 的 `inputs[].dtype` 与 §9.3 一致；`attrs[].name` 覆盖 §4.7，固定配置仍说明约束。

### 11.2 验证顺序

1. 在910C环境重新编译安装含quiet与动态region修复的仓库，执行前端/codegen回归。
2. 执行协议门禁和L0，确认双槽与FP32语义。
3. 现有L1/Boundary为设计锚点；L0收敛后再扩展完整分层套件。
4. 按 §9 完成2→4→8 rank、尾块与复用，再测收益和实际时间线。

### 11.3 设计自检状态

| 检查项 | 当前状态 |
|---|---|
| 模式、接口、内存路径、双槽ownership、序号与退出 | 本文已规定，需实现验证 |
| 支持域/尾块/host约束与成本审计 | 已给出设计记录；不是设备通过证据 |
| Golden、dtype精度表、L0及关键哨兵 | 已规定；没有执行 |
| proto一致性与本地文档检查 | 已检查11章、Python片段语法、空白/换行、2种dtype及10个attrs/default/type的一致性 |
| 资源与关键case成本 | 已用整数公式复算片上预算、对称workspace与两组输入/通信统计 |
| 910C编译、正确性、时间线与端到端收益 | 未执行，禁止标记通过 |

proto维度核对实际调用了仓库 `coverage_check.py` 的 `load_proto_dtypes_attrs`；
本地未装PyYAML，使用其文本回退路径，并额外核对attrs/default/type与输出dtype。
这不等于完整YAML parser验证，也不等于尚无kernel/test可供执行的测试覆盖门禁通过。

### 11.4 关键证据索引

- `examples/gemm_allreduce/gemm_allreduce.py`：完整K后发布、双token、GET/ACK与退出原型。
- `examples/gemm/example_gemm_intrinsic.py`：T.mma、显式双缓冲和pipe event。
- `examples/shmem/`：四个原生GM/UB搬运例子，不是现成的AllReduce实现。
- `src/tl_templates/ascend/common.h`：L1 NZ清零、字节粒度GM pitch、SHMEM helper。
- `src/op/ascend.cc`、`src/target/codegen_ascend.cc`：BufferRegion、偏移与need_clear生成。
- `3rdparty/catlass/include/catlass/gemm/tile/copy_gm_to_l1.hpp`：ND2NZ stride及逐行路径。
- `3rdparty/shmem/include/device/gm2gm/shmem_device_mo.h` 及对应 `.hpp`：quiet契约和实现。
- `testing/python/language/test_tilelang_ascend_language_copy_ub_to_gm_alignment.py`、
  `test_tilelang_ascend_language_gemm_v0_k_tail.py`：已有尾块/stride回归来源，非本次通过记录。
