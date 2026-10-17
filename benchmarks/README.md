# HunyuanVideo-1.5 可迁移 baseline

本目录随框架仓库一起使用，不依赖 `exp/`、兄弟仓库或本机绝对路径。默认复现官方 720p I2V SSTA、121 帧、50 步、seed 123、CFG distilled、BF16、SR/rewrite/cache/compile/FP8 关闭。输入使用固定版本的 LongCat 咖啡馆示例。模型加载一次，完整预热后计时，MP4 保存包含在 E2E 中。

## 环境和模型

```bash
# 在 clone 的 HunyuanVideo-1.5 根目录执行；需要 uv、CUDA toolkit 和 C++ 编译器。
bash benchmarks/setup.sh
# 仅检查已有环境，不重装：
bash benchmarks/setup.sh --check
.venv/bin/python benchmarks/fetch_sample.py
```

`requirements.lock` 固定原 baseline 的 Python 依赖（Python 3.11、torch 2.6.0/CUDA 12.4、diffusers 0.35.0、transformers 4.57.1）。FlashAttention 2.7.4.post1 单独安装，flex-block-attn 固定源码 commit 并编译。可通过 `UV`、`BENCH_VENV`、`CUDA_HOME`、`MAX_JOBS`、`FLASH_ATTN_PACKAGE` 指定工具或安装源；默认环境 `.venv`，不硬编码 CUDA 安装位置或 GPU 名称。未随脚本自动下载模型；按根目录 [checkpoints-download.md](../checkpoints-download.md) 准备官方完整组件。

**硬件支持由所选模型及官方内核决定。** 当前官方 SSTA/flex-block-attn 仅支持其声明的 Hopper 架构；脚本可迁移不代表稀疏模型支持所有 GPU。不会静默切换 dense attention。在其他受官方支持的设备上，可复制 `baseline.json`，显式选择可用的 dense 模型（`sparse_attn=false`），用 `--config` 运行，环境安装时加 `--without-ssta`；不同模型结果不与 SSTA baseline 混算。

模型目录结构包含 `transformer/720p_i2v_distilled_sparse/`、`vae/`、`scheduler/`、`text_encoder/`、`vision_encoder/`。权重可以在任意目录，通过 `--model-path` 传入。SigLIP 等组件的下载权限按官方要求准备。默认输入下载使用固定 Git commit 并校验 SHA-256；也可自备 `--image-path` 和 `--prompt-file` / `--prompt`。

## 运行与后台监控

```bash
# 将 /your/checkpoints 替换为权重路径；其余路径由脚本位置推导。
.venv/bin/python benchmarks/run.py --model-path /your/checkpoints \
  --gpus 8 4 2 1 --dry-run

# 默认关闭 offload，仅 dit 模式：每卡数完整预热 1 次、正式 1 次。
.venv/bin/python benchmarks/run.py --model-path /your/checkpoints \
  --gpus 8 4 2 1 --background

# 只测 E2E：不会创建 CUDA timing events，也不会注册 DiT hooks。
.venv/bin/python benchmarks/run.py --model-path /your/checkpoints \
  --gpus 1 --offload off --modes e2e --background

# run.py 输出 SUITE 路径。监控和简报只依赖 Python 标准库。
python3 benchmarks/monitor.py outputs/benchmarks/SUITE --watch
python3 benchmarks/report.py outputs/benchmarks/SUITE
```

`--output` 可以指定不存在的结果目录；已存在时拒绝覆盖。默认 `outputs/benchmarks/UTC时间/`。参数路径相对调用者当前目录解释，从任意 cwd 调用脚本均可。通过 `CUDA_VISIBLE_DEVICES` 选择设备，保留已有设备映射，每组使用可见设备的前 N 张；GPU 数须整除 16 heads。`--warmup` / `--repeats` 控制每种模式次数。每次实验保存配置与脚本快照，重复执行会新建 suite。

后台启动会输出 PID、launcher 日志、监控命令；退出当前终端后继续运行。进度脚本显示每组已完成 warmup/正式次数、当前模式与日志中的最新步数。进度来自计时区外的 JSON 和原生 tqdm，监控进程不插入推理路径。每完成一个进程组以及矩阵结束时自动生成 `report.md`、`summary.json`。运行中也可手动重建报告；未完成样本不会被当作完整矩阵。

后续实验默认使用 **`--modes dit --warmup 1 --repeats 1 --offload off`**：同时记录 E2E 和 DiT，每个卡数共生成两次视频，group offload 也关闭。需要复查波动或插桩开销时，可显式指定 `--repeats 3 --modes dit e2e`。当前 H20 的历史 1/2/4/8 卡实验均已验证可容纳完整模型；其他设备显存不足时可显式使用 `--offload on`，或 `--offload on off` 比较两种策略。

`baseline.json` 只保留生成所需的 11 个参数；关闭的 cache、FP8 等功能使用官方默认初始化，不再暴露其子参数。offload 由命令行控制。默认检查输出张量形状和有限值；加 `--validate-video` 可额外解码 MP4 检查帧数/尺寸/FPS，并计算文件 hash，全部位于计时区外。报告记录该开关。

每组 worker 从该 suite 的 `scripts/` 快照执行；启动器自动传入框架根目录，运行过程中修改 benchmark 源码不会改变后续组使用的实现。

## 计时位置与定义

阅读 [worker.py](worker.py) 中 `E2E START` / `E2E END` 和 [timing.py](timing.py) 中 `DiT START` / `DiT END`：

- `response_e2e_s`：rank 0 在 `pipe(**requests)` 前开始，官方 `save_video` 返回后结束。pipeline 的原生 `.cpu()` 产生 CPU 视频；不在 pipeline 与保存之间增加同步或检查。
- `distributed_e2e_s`：另记录保存后 barrier 返回时间，跨 rank 取最大值；每个 rank 的原始值保存在 `records[].per_rank[].distributed_e2e_s`。
- `dit_forward_sum_max_rank_s`：完整 `pipe.transformer` forward 前后记录 CUDA events，每 rank 先累计再取最大值。`dit_forward_step_max_sum_s` 则逐步跨 rank 取最大再求和；原始 `[rank, step]` 毫秒值保留。排除 scheduler 和去噪循环外的整模块搬运；若启用 group offload，forward 内的搬运/等待属于该指标。
- events 在所有请求前创建、record 初始化并同步；请求内 hooks 仅 record 和计数，无逐步同步。事件查询、跨 rank 聚合、输出检查、日志和 JSON 都在 E2E 结束后。原始 DiT 逐步值仅保存于 `records[].per_rank[].dit_step_ms`，不再复制整个 rank×step 矩阵。
- 显式同时运行 `dit` 和 `e2e` 时，两者复用同一 pipeline、各自预热，正式按 AB/BA 顺序逐轮反转。进入 e2e 请求前移除 DiT hooks；单独运行 `--modes e2e` 时连 events 都不创建。相同 seed/输入，无额外输出缓存。

原始 baseline 计时 schema 2 有 pipeline 后同步和区间内输出检查；本实现 schema 3 去掉它们，仅保留 E2E 和 DiT 重点指标。历史对照是复测差异，不能直接解释为算法加速；同次交替对照用于估计 DiT hooks/events 对 E2E 的影响。

## 报告

`report.md` 包含推理参数、模型/attention 配置、输入校验、源码版本、环境与硬件、group offload 实际取值、输出形状、E2E/DiT 均值 ± 样本标准差、逐次值、多卡加速比和两种计时的配对差异。加速比以相同 offload、相同模式的完整单卡结果为基准；e2e 模式没有 DiT 数据。

仅一次正式测量时报告标为“单次”，标准差不估计（JSON 中为 null）；仍可计算多卡加速比。两种模式的对照只在显式运行两种模式时提供。

`model_manifest.json` 保存模型配置内容/SHA-256 和权重文件大小（不声称权重内容已逐字节校验）。每组 `effective_config.json` 保存运行环境和实际策略；`metrics.json` 保留 warmup、正式请求和每 rank 原始值。失败/OOM 独立记录，不产生伪性能数据；标准差/均值 > 2% 或持续变快时提示复查。

历史对比默认关闭。运行 `run.py` 时显式添加 `--baseline benchmarks/reference/h20_20261013.json` 才会复制参考并生成对比；也可在实验完成后按需生成，无需重新推理：

```bash
python3 benchmarks/report.py SUITE --baseline benchmarks/reference/h20_20261013.json
```

不带 `--baseline` 重生成报告会去掉历史章节及 JSON 历史字段，即使 suite 中保存过历史参考。
历史参考仅本地保存、不随脚本提交；使用 `--baseline` 时需提供本地参考 JSON。输入/模型参数/硬件/核心依赖/推理源码不匹配时不计算历史百分比；时钟、驱动及系统负载变化仍可能影响复测。

CPU 检查：`.venv/bin/python -m unittest discover -s benchmarks -p 'test_*.py' -v`。


## 运行状态与失败提示

运行器自动更新实验目录的 `status.json`（原子替换文件），监控命令会直接显示错误摘要和日志位置。

- `status`：`waiting` / `running` / `complete` / `complete_with_failures` / `interrupted_or_failed`。
- `has_failures`：是否已有失败；失败后即使继续其他组也保留为 true。
- `failure_count`：已知失败组数；运行器自身错误由 `launcher_error` 单独记录。
- `last_error`：最近错误的 `summary`、`phase`、`directory`（组错误）、`log_file` 与 `error_files`。
- `completed`：每组结果、退出码和失败详情；`current` 为当前组，`updated_at_utc` 为状态更新时间。

`log_file` / `error_files` 相对于实验目录。后台启动日志位于实验目录旁的 `<实验名>.launcher.log`；
前台运行器自身异常也会写状态，完整堆栈显示在终端。各 rank 原始错误保存在组目录 `error_rankN.json`。
状态文件用于查看，不是控制入口：手动修改不会触发重试、跳过或恢复，并可能被下一次更新覆盖。
启动前参数校验失败时尚未创建实验目录；运行器被强制杀死也无法自行更新文件，监控会提示进程已退出。
已经启动的旧运行器保持原快照；新版监控可从其旧状态和 rank 错误文件展示失败信息，不改写其状态文件。


## GPU 空闲等待与统一启动行为

启动器现在与另外两个框架使用相同的 GPU 等待、锁和状态文件协议：

- `CUDA_VISIBLE_DEVICES` 指定 nvidia-smi 索引或唯一 GPU UUID 前缀；每组取前 N 张，worker 使用完整 UUID。
- 目标 GPU 有计算进程、已用显存超过 256 MiB 或利用率非零时等待；每 30 秒检查一次，连续两次空闲才开始。
- 同机同用户的三个框架共用 `/tmp/video-benchmark-gpus-UID/` 下的 GPU 文件锁，持锁至该组结束。
  不遵守锁的外部程序仍可使用 GPU；推理期间不轮询 GPU。此实现面向整卡，不支持 MIG 分区选择。
- `--wait-for /another/suite/status.json` 等待前置实验结束；失败/进程消失则报错。等待原因保存在 `status.json/waiting_reason`。
- 普通 worker 错误停止矩阵；加载/预热阶段明确的 OOM 记录后继续下一组，正式测量阶段 OOM 按失败停止。
  不会自动重试、降配或改变 offload。三个框架的监控均显示已知失败，后续成功不会隐藏失败提示。
- Hunyuan 设备/SSTA 检查由取得 GPU 后的原生 worker 加载执行，运行器等待时不创建 CUDA context。

默认仍为 `dit`、一次 warmup、一次正式测试，关闭 offload。报告提供 warmup 与正式耗时对照；历史比较按需开启。

多次正式测量 CV > 2% 或至少三次连续变快且首尾下降 > 2% 时，报告提示复查；不自动增加请求次数。双模式对照保留均值和每轮配对差异。
