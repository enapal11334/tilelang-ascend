# 检查记录（2026-10-28）

范围：GEMM+AllReduce 原生 SHMEM GET/PUT 版本；已移除本轮添加的 extern 生成修复和 helper。
环境：Windows；按用户要求仅实现和静态检查，没有 910C 双卡运行环境。

| 检查 | 结果 | 范围与限制 |
|---|---|---|
| `python -m unittest discover -s examples/gemm_allreduce -p test_cpu.py -v` | 5/5 通过 | CPU contract/golden/比较器，未执行 kernel |
| Python AST 解析 | 5 文件通过 | 4 个示例 Python 文件 + 1 个 language 回归文件 |
| Python 行宽 ≤100 | 通过 | 同上 5 文件 |
| `test_gemm_allreduce.py --help` | 通过 | CPU 可加载 CLI，无提前导入 NPU 依赖 |
| `bash -n examples/gemm_allreduce/run.sh` | 本轮未执行 | 上一轮通过且脚本未改；本轮 Git Bash 启动失败，退出码 -1073741502 |
| `git diff --ignore-submodules=all --check` | 通过 | 已跟踪文件的差异空白检查 |
| 新增文件的结尾换行、尾随空格、tab 检查 | 通过 | 本目录交付文本及新增 language 回归 |
| Ruff / clang-format | 未执行 | 工具未安装，下载未获授权 |
| SHMEM GET 视图/PUT 状态偏移 lowering 两个回归用例 | 未执行 | 缺少可运行 TileLang/TVM 环境 |
| C++/Bisheng 编译 | 未执行 | 缺少 Linux CANN 工具链 |
| 910C 两卡精度、同步和启动偏斜 | 未执行 | 无硬件；不能据 CPU 检查判定通过 |
| 性能/加速比 | 未执行 | 单核心原型，不提供性能结论 |

`git diff --ignore-submodules=all --exit-code -- src/target/codegen_ascend.cc` 返回 0：
该文件已恢复至原状态。新 kernel 不含 `T.call_extern` / `T.import_source`，不再依赖 helper。

已做源码审计：Buffer.access_ptr 计入 elem_offset、现有 SHMEM codegen 保留该偏移；
PUT 发布状态、DMA→UB 轮询、C/V token 数量、MTE2 完成后 ACK、64B control line 隔离、
unique-id bootstrap、对称内存成功释放顺序。codegen_ascend.cc 无本轮改动。
这些是代码层面的检查，不能证明设备内存可见性和运行时进度。

普通 git diff/status 会被 catlass 子模块 Git 目录权限阻断，因此检查主仓变更时使用
`--ignore-submodules=all`；未修改子模块内容。

后续真机入口与构建步骤见 README.md。回归测试包含 GET 的动态 elem_offset 和 PUT 的状态
目标偏移；双卡测试覆盖单 tile、默认 4 tiles、9 tiles 和两次 K 循环，
每种数据按正/负/正 epoch 复用 workspace，并轮流延迟其中一个 rank。

## 2026-10-29：TP=2 串行基线与 E2E 对比入口

新增 `benchmark_gemm_allreduce.py`、`benchmark.sh`，补充 README 和 CPU 检查。
本轮不修改融合 kernel。基线为 torch.mm（FP16 局部输出）→ FP32 转换 → HCCL SUM；
对比入口记录同步 host E2E 时延，不能当作设备 profiler 或纯融合收益数据。

- CPU unittest：9/9 通过，涵盖原 5 项以及逐轮两卡最大时延统计、局部 FP16 舍入、
  串行执行及重复覆盖输出、任一 rank 精度失败时两 rank 一致拒绝后续测量。
- `benchmark_gemm_allreduce.py --help`：通过，未导入 torch_npu/TileLang/SHMEM。
- 本轮两个 Python 文件 AST 解析及行宽 ≤100 检查：通过。
- `git diff --check --ignore-submodules=all`：通过（已跟踪改动）。
- Linux launcher、NPU GEMM/HCCL、融合 SHMEM、多流依赖与真实双卡时延：未执行。

原 2026-10-28 记录仅保留为历史记录；其中引用的
`testing/python/language/test_tilelang_ascend_language_shmem_view.py` 当前仓库不存在，
不属于本次新增内容或本次验证结果。

### 同日更新：统一精度约定（替代上节 FP16 局部输出基线）

- 基线改用 `aten.mm.dtype_out(..., torch.float32, out=...)`，直接产生 FP32 局部输出，
  再串行执行 FP32 HCCL SUM。移除 half 中间缓冲、cast 和单独的舍入 golden。
- 两条路径测量前后使用同一个 `make_case` FP32 golden 和同一个 `compare` 容差。
- 增加 NPU FP32 结果探针；不支持显式 output-dtype overload 或探针失败时，两 rank
  一致停止，不回退到低精度基线。探针不能证明任意输入下所有内部累加均为 FP32。
- CPU unittest：10/10 通过，新增覆盖显式 FP32 overload、拒绝 half 输出再 cast 的
  探针（使用 NumPy mock，不代表设备通过）、缺少接口时禁止回退。
- CLI help、两 Python 文件 AST/行宽及主仓 diff 空白检查：通过。
- 真机探针、目标 CANN 内部 FP32 累加路径、两卡精度和时延：仍未验证，无性能结论。
