# 910C GEMM + AllReduce

单机2/4/8个逻辑NPU，K分片输入 `A[M,K_local]`、`B[K_local,N]`。
FP16/BF16输入，GEMM与SUM均为FP32，末尾一次转换为输入dtype，每卡返回完整 `[M,N]`。
固定 Expert 模式、tile=16×128×128，使用双槽 all-peer push，支持 M/N/K 尾块。
生产端 PUT FP32 tile 到各接收端的 inbox，等待 `quiet()` 完成后发布 READY。
接收端等到同代 READY 后读取本卡 inbox，读完并写出结果后回 ACK。

## 使用

先在910C上配置CANN、torch_npu和SHMEM，并重新编译安装本仓库。
安装版本需包含 `T.shmem_quiet()` 的 C++ 注册和 codegen。已有该接口的环境可直接更新示例。

同一进程内先设置NPU并初始化单机Gloo进程组；在该示例目录中：

```python
from comm import GemmAllReduce910CContext, finalize_shmem

with GemmAllReduce910CContext(A, B) as comm:
    Y = comm(A, B)
    comm.stream.synchronize()

# 可以在此之前顺序创建不同shape/dtype的context。
finalize_shmem()  # 全rank调用一次，然后才销毁Gloo进程组。
```

context从输入和当前设备推导所有配置，不暴露tile/算法/归约参数。
A/B必须连续、同dtype、位于当前NPU；调用使用构造时的stream。
异步调用期间保留A/B/Y；不跨stream并发复用context。
每个 context 的对称 inbox 和 flags 合计占用 `16640×G×P` 字节，
其中 `G` 为 Cube group 数，`P` 为 rank 数。SHMEM 在进程内仅初始化和终结一次。

## 测试

从仓库根目录执行；将进程数依次改为2、4、8：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 examples/gemm_allreduce_910c/test_gemm_allreduce_910c.py --level l0
torchrun --standalone --nnodes=1 --nproc-per-node=2 examples/gemm_allreduce_910c/test_gemm_allreduce_910c.py --level protocol
torchrun --standalone --nnodes=1 --nproc-per-node=2 examples/gemm_allreduce_910c/test_gemm_allreduce_910c.py --level all
```

唯一CLI参数为 `--level`：`l0`、`l1`、`l2`、`boundary`、`protocol`、`perf`、`all`。
两种dtype总是一起覆盖；`perf`先跑4096代协议门禁，再测两个固定decode case。
冷编译case watchdog为600秒，warm验证为120秒；由torchrun在worker退出后终止其他进程。

性能比较双槽、同协议单在途tile，以及相同Cube计算→独立HCCL FP32 SUM→末尾cast。
输出最慢rank设备时间的p50/p95、host enqueue及相对串行版本的95%区间。
`PERF_PASS`只表示相对串行测量的收益，仍须使用目标CANN版本的设备profiling工具核实
Cube与跨卡MTE的真实重叠；Event计时不能替代时间线。

8 rank 下 all-peer push 的远端数据量约为带宽最优 RS+AG 的 4 倍，小 M 下可能只部分隐藏通信。
PUT 版本的 CANN 编译、910C 精度和性能待验证。实际检查记录见 [validation.md](validation.md)。
