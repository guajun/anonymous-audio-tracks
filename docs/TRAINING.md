# 最小训练、恢复与保留集评估（TRAINING，issue #8）

## 目标训练与现有基线

项目目标以 [身份活动向量与局部拼接规范](IDENTITY_ACTIVITY_DESIGN.md)（#48）为准。
本文其余部分是 **main 当前冻结 AuT / Query / E/P 概率训练及历史实测**；
组级 BCE 分配、正负原型对比与概率 F1 尚未迁移为连续 shape 主监督。

目标主链：每窗非零 V → 基于 E 的可微局部关联 → 同权重搬运 A → 整段一次 PIT
→ 连续包络 shape。不得逐帧 PIT 抹掉换轨。同形状可有 E 接近与歧义，分岔后让 shape
反传错误；cosine 是打分、soft/hard 是匹配权重的使用，TimeCycle 是独立辅助开关，
cycle=0 时主关联仍执行。hard argmax 的选择本身不能提供 shape→E 梯度；逐行 softmax
可能多对一，容量约束须明确。具体 loss、权重、软匹配与 gate 梯度仍需实验。

连续参考来自实际效果后进入混音的分轨线性 RMS；可比较 L1、校准 Huber、面积 IoU、
Huber+IoU、多尺度回归及后续时间运输/有界偏移。L1/Huber 和面积 IoU 仍同 t 对齐，
IoU 不自动容忍时间平移；静音/未用输出惩罚单列。先分轨监督，再独立比较混音训练，
并区分“分轨预训练后混音继续训练”和“全程不用分轨标签”。

课程从简单单音色不重叠到双音色交错/局部重复、overlap、较真实 DawDreamer 音乐。
训练段长度不能替代单次 E 输入的真实共同事件证据。每次研究应报告原时间轴预测、raw/搬运 A、
漏/多报、来源数、PIT 并列、空匹配/熵、shape→E 梯度、时间边界与成本。
工程通过不等于研究成功；本次不运行训练，也不把下文历史数值当新设计验收。

本文件说明 `src/aat/training/`、`scripts/train.py`、`configs/train/` 与 `tests/training/` 的实现、
配置、checkpoint/resume 语义、评估协议与已实测证据。范围是**冻结 AuT 特征 + K 查询 E/P 头**的
最小可复现训练；不做全量微调、昂贵超参扫描、声源重建或灯光映射。

两个模式从头到尾分开标识，结果不得混用：

| 模式 | 编码器 | 产物标识 | 能证明什么 |
|---|---|---|---|
| `fake` | `aat.encoders.fake.FakeAutEncoder`（NumPy，无权重、无 torch 模型） | `result_kind = fake-encoder-smoke (engineering pipeline only, not a real model result)` | 数据→batch→头→loss→checkpoint→resume→评估的工程链路 |
| `aut` | 冻结 Qwen3-Omni AuT（issue #5，独立加载、只读权重） | `result_kind = aut-frozen-training (real frozen AuT features)` | 真实特征上的短训练与保留集结果 |

`run.json`、`summary.json`、`checkpoint.pt`、逐 step 日志和评估报告都带模式与 `result_kind`，
查看器/评审不得把 fake 输出当作真实模型结果。

## 1. 固定的特征与 batch 策略

两条路径共享同一套固定策略（`aat.training.encoding`，配置里校验，不能静默更换）：

- **统一独立固定 2 s 窗口**：训练与推理都通过 `FakeAutEncoder.extract_windows` /
  `AutEncoder.extract_windows` 对每个中心独立编码；不使用“整段编码再切片”，也不允许训练/推理
  采用不同窗口。`encoder.window_seconds` 必须为 2.0，且必须与数据集索引的
  `plan.window_seconds` 一致，否则拒绝启动。
- **block-diagonal SDPA**：真实路径固定 `attention = "block_diagonal"`（issue #5 审计的掩码注入，
  `sdpa` 后端）；`unmasked_global` 被配置校验拒绝。
- **fp32**：`dtype = "float32"`，issue #5 实测可把批量与单条的差异压到 2.6e-4 量级；bf16 被拒绝。
- **固定分块与尾块**：每个 song block 内按 `encoder.batch_windows`（默认 8）顺序分块编码，
  最后一块可以更小；block 组成只由该 step 的采样 seed 决定，resume 后完全一致。分块策略、
  尾块行为、`extraction = "independent_windows"` 都写入 checkpoint 的 encoder provenance。
- **head 输入一致**：帧绝对时间 `frame_times = 实际窗口起点 + 编码器帧中心`，窗口中心用
  `center_times - W/2` 的同一 half-up 取整换算；`frame_valid` 为 False 的补零帧不参与注意力，
  `center_valid=False` 的行不产生有效槽位，padding 槽位不进入 loss。
- **推理边界与模式**：`HeadInference.predict_windows` 只接受该采样率下精确的 2 s 样本数，
  校验 `sample_rate` 为正整数、start/center 在同一采样网格取整约定内一致；`valid_samples`
  含补零的行（或调用方显式给出的 `center_valid=False`）在头部被强制为全零无效槽位，
  边界补零窗不会伪装成受监督预测。预测期间模型强制 `eval()`（dropout 关闭、不消耗 RNG），
  并在异常/正常返回后恢复原模式；`predictor_for_centers` 支持 `audio_duration_seconds` 以在
  callback 路径获得相同的边界掩码。

## 2. 数据：真实 DawDreamer 渲染与分组切分

### 2.1 数据计划

`configs/train/dataset_local.toml` 定义一个 8 首曲目的本地语料计划（渲染配置的
`preset_ref`/`sample_ref` 作为家族资产 ID）：

- `train-01..04`：共享 `aat/train/*` 音色/素材家族（4 首通过共享 voice 连成一个连通分量，
  因此索引把它们整体保留在同一集合）；
- `val-01/02`：`aat/val/*` 家族；
- `test-01/02`：`aat/test/*` 家族。

所有声音都由 DawDreamer 内置 sampler/效果链用显式 seed 本地合成，**没有下载任何素材**。
本机没有安装 Surge XT，因此本语料只覆盖自生成 sampler 音色参数范围，不宣称 Surge 或真实乐器
音色多样性。每首 2–3 个独立来源，8 s + 1.5 s 尾音，44.1 kHz。

### 2.2 生成命令与校验

```sh
uv sync --locked --extra render
uv run --no-sync python scripts/train.py prepare-data \
    --plan configs/train/dataset_local.toml \
    --out runs/train-data --index-out runs/train-index/index.json
```

`prepare-data` 逐曲调用 issue #3 的 `render_sample` 渲染、issue #4 的标签器写入
`activity.json/.npz`，最后用 issue #6 的 `build_dataset_index` 做格式/摘要/容量/泄漏校验并落索引。
退出码 `4` 表示检测到跨集合资产泄漏（本实现不会用“比例凑数”拆开连通分量）。

本机实测（Windows，CPU，2026-09-29）：

- 8 首渲染成功，`stem_sum max_abs_error_lsb = 1.0`（容差 3.0）；`test-02` 尾音 margin 0.05 s，
  渲染器自身容差通过；
- 每首 `valid_count/center_count = 376/476`（2 s 窗）；
- 索引 `split_counts = {train: 4, val: 2, test: 2}`，`split_ratios_actual` 与请求一致；
- `leak_free = true`，`cross_split_assets` 三个命名空间全为 0，`empty_splits = []`；
- 数据集内容摘要（checkpoint 里也记录）：
  `c073f9cd8a6e44832c5943dd90d960b26e5b9a3e5279d012d0aed6c39b391563`。

音频、索引与运行产物只留在被忽略的 `runs/`；索引记录 portable 相对路径，可整根移动后校验。

### 2.3 CI 用的合成语料

`aat.training.dataset.make_smoke_dataset` 用 NumPy 直接写真实协议文档与 PCM（不经过
DawDreamer），供 CPU smoke 与测试使用：4 首、三个不共享资产的家族，索引同样做泄漏校验。
它明确标记 `synthetic`，不能替代 `prepare-data` 的真实渲染证据。

## 3. 训练入口与配置

```sh
uv sync --locked --extra ml

# CPU fake 冒烟（自包含：生成合成语料 -> 训练 -> 评估）
uv run --no-sync python scripts/train.py smoke --out runs/train/smoke-0001 --steps 40

# 真实冻结 AuT（权重目录运行时传入，配置里是占位路径）
uv run --no-sync python scripts/train.py train \
    --config configs/train/aut_short.toml \
    --model-dir /path/to/encoder-checkpoint \
    --device cuda:0

# 断点续训 / 延长步数（配置身份必须一致）
uv run --no-sync python scripts/train.py train \
    --config configs/train/fake_local.toml --resume runs/train/fake-local --steps 400

# 用 checkpoint 对保留集评估（不训练）
uv run --no-sync python scripts/train.py eval \
    --config configs/train/aut_short.toml \
    --checkpoint runs/train/aut-short/checkpoint.pt --split val --max-songs 1
```

`configs/train/`：

| 配置 | 用途 |
|---|---|
| `fake_smoke.toml` | 合成语料 + fake 编码器，CPU CI/冒烟 |
| `fake_local.toml` | 本地真实 DawDreamer 语料 + fake 编码器（工程链路 + 过拟合观察） |
| `aut_short.toml` | 真实冻结 AuT，100 step 单次 GPU 预算，val/test 各 1 首 |

配置为 TOML，未知键/越界值/未知模式直接报错（带键路径）。所有影响训练状态的字段一起参与
`config_identity`；输出目录、run 名字、eval 协议、checkpoint 间隔与 step 预算不参与身份比较，
但 `--steps` 只能向前延长（checkpoint 已到的 step 不能被缩短）。

### 3.1 采样与随机性

一个 step = 每首歌一个 group（同曲多个非相邻中心，`min_center_gap` 网格步距，GT 来源列固定），
由 issue #6 的 `sample_batch` 采样，绝不混两首歌。每步的 batch seed 用显式序列

```text
step_seed(base_seed, step) = SeedSequence([base_seed, step, 0xA17]).generate_state(1)[0]
```

因此同一 `(base_seed, step)` 的 batch 与历史无关：中断后继续与不中断训练在同一 step 得到完全
相同的 batch。checkpoint 记录 `sampler.scheme = "seed-sequence-v1"`、base seed 和已完成 step。

### 3.2 Checkpoint 内容与恢复校验

每 `checkpoint.interval_steps`（默认 20）和训练结束时原子写入 `checkpoint.pt`（临时文件 +
`os.replace`）。payload 包含：

- `step`、`sampler`（显式 seed 序列）、head `model_config` 与 `model_state`、AdamW
  `optimizer_config`/`optimizer_state`；
- Python/NumPy/torch/CUDA RNG 状态；
- Python/NumPy/torch/CUDA/平台版本；
- 完整 `config` 快照、`config_sha256` 与 `config_identity`；
- Git SHA + dirty 标记（可用时）、`uv.lock` 的 SHA-256；
- 数据集身份：索引文件 SHA-256、数据根、`plan`、整库与逐 split 内容摘要（覆盖每条
  `sample_sha256` 与全部文件摘要）；
- encoder identity 与完整 provenance（模式、model id/revision/revision_source、layer、dtype、
  attention、extraction、分块策略、fake seed 等）；
- 最后一个 step 的 loss 分项与累计有效监督计数；
- `resources`：设备、整体 elapsed、每进程 torch CUDA allocator 的 peak allocated/reserved
  （在 `Trainer` 构造、编码器加载前 reset；CPU 明确写 `null` 并给 `unavailable_reason`，
  不伪造 0）。

恢复时在**训练任何一步之前**逐项核对：config identity（含 head/loss/optim 超参、seed、数据
采样参数）、数据集 digest 与当前 split digest、encoder identity、model shape。任何不一致抛
`ResumeMismatchError` 并列出差异；数据文件被改写但未重建索引时，启动时的全量重校验（
`DatasetIndex.load(verify_files=True)`）先失败。不会静默替换数据、shape 或超参。

### 3.3 逐 step 机器可读记录

`train_log.jsonl` 每行一个紧凑 JSON，包含：`step`/`step_seed`/`sample_ids`/`windows`/
`source_counts`、`batch_sha256`、`loss`（total/activity/empty_slots/positive/negative）、
`loss_weights`、`stats`（matched/skipped/**ambiguous**/**truncated**/identity_masked/
supervision_masked 与四类有效 term 数）、`effective_supervision`、`grad_norm`、`lr`、
`elapsed_seconds`。`summary.json` 汇总累计计数与最终状态。

**零有效监督不算成功**：若整个 run 的 `activity_terms == 0`（例如所有中心窗口都越界），
`summary.status = "no_effective_supervision"`，CLI 退出码 1，但诊断与 checkpoint 仍保留。

**中断日志对账**：恢复时不把上一次未 checkpoint 的日志尾巴直接拼接。`_reconcile_train_log`
只保留与 committed checkpoint step 严格对齐的 `0..step` 前缀（任何不一致的 step 序列直接抛
`CheckpointError`）；checkpoint 之后的完整行、以及写入中断产生的半行 JSON，统一保存为
`train_log.discarded-<timestamp>.jsonl` 证据后，把 canonical 日志原子重写为前缀。恢复后的
step seed 序列不变，因此被丢弃的行会被同 seed 重新计算，canonical 日志与 checkpoint/totals
始终唯一且对齐。

### 3.4 覆盖安全

输出目录已有已知产物时，不带 `--overwrite` 直接拒绝；带 `--overwrite` 时把已知文件重命名为
`*.bak-<UTC 时间戳>`（只处理白名单文件名，不删除任何内容），再开新 run。软链接/目录外内容不
会被触碰。

## 4. 评估协议（固定，不用 test 调参）

`eval` 用 issue #9 的整曲关联与评估：对每首保留曲目在 `activity.valid` 的中心上预测，经
`associate_sequence`（一对一、余弦门控、跨窗身份记忆）得到全曲轨迹，再以
`evaluate_trajectory` 做**整曲全局**的 source→track 映射，绝不逐帧重新最优匹配来掩盖 ID 切换。
阈值固定 0.5（`eval.threshold`，同时用于关联与评估），时间容差 1e-6 s。

报告 `eval_<split>.json` 含：协议快照（阈值、关联参数、全局映射说明、fake/real 标识）、
`micro`（TP/FP/FN、P/R/F1、ID switch、歧义归属帧、来源数误差、边界误差）、逐来源行、
有效量（valid centers / active labels / songs）、未定义原因（例如无正预测时 precision 为
`null` 而不是 0），以及三个常量基线（同一协议）：

- `all_inactive`：无候选，`precision = null`、`recall = 0`、`F1 = 0`；
- `all_active`：所有槽位全时刻激活、常量身份，recall = 1、precision 反映过预测；
- `no_identity`：使用模型 P，但所有 E 换成同一个常量单位向量，检验身份信息是否起作用。

在评估报告与轨迹 provenance 中，基线标为 `data_kind = "mock"`、模型输出标为 `"model"`。

边界语义：评估只在 `activity.valid` 的中心预测，且 `HeadInference` 会把越界/补零中心输出为
全零槽位（`center_valid=False`、`slot_valid` 全 False、`E=P=0`），与训练时无效中心一致；
`build_prediction_data` 路径传入 `audio_duration_seconds` 时获得相同掩码。`extract_windows_at_times`
仍拒绝对音频范围外的中心（维持既有 windowing 策略）。

## 5. 实测证据

### 5.1 本地测试

```sh
uv sync --locked --extra ml --extra render
uv run --no-sync pytest -q -p no:cacheprovider              # 686 passed（ml + render）
uv run --no-sync pytest -q -p no:cacheprovider tests/training  # 44 passed
# 无 torch 的 base job / 无 torch 的 render job（同 CI）：
UV_PROJECT_ENVIRONMENT=runs/venv-base uv sync --locked
UV_PROJECT_ENVIRONMENT=runs/venv-base uv run --no-sync pytest -q -m "not integration and not ml"  # 574 passed, 12 skipped
UV_PROJECT_ENVIRONMENT=runs/venv-render uv sync --locked --extra render
UV_PROJECT_ENVIRONMENT=runs/venv-render uv run --no-sync pytest -q -m "not ml"  # 579 passed, 10 skipped
```

`tests/training/` 覆盖：配置解析/未知键/pinned 模式拒绝、合成语料无泄漏与家族不交叉、摘要随
内容变化、padded 来源列与补零帧不进入 loss、越界中心零有效监督、fake smoke 的 finite loss 与
机器可读日志、**中断恢复与不中断训练 bitwise 一致**（模型/优化器/batch/loss/torch RNG，含
\"checkpoint 之间真实中断 + 部分 JSON 行\"的日志对账）、配置与数据变化拒绝恢复、覆盖归档、
checkpoint→推理接口、边界掩码与 callback 一致性、窗口长度/时间戳校验、dropout 下推理模式与
RNG 不消耗、评估协议与基线、CLI 端到端。CPU 测试与本地运行设置 `OMP_NUM_THREADS=4` 以避免
小矩阵线程开销。

### 5.2 fake 编码器（工程链路，非模型结论）

在真实 DawDreamer 语料上运行 `configs/train/fake_local.toml`：200 step 与 600 step 两次
（`runs/train/fake-local*`，忽略目录）。逐 step loss 分项正常下降/波动，checkpoint/resume/评估
链路完整；但 600 step 后模型在评估中心上的最大 P 仍约 0.3，低于固定阈值 0.5，轨迹数为 0，
train/val/test F1 均为 0。**fake 链路证明工程可运行，不证明学到来源身份**。

独立证据：对固定 batch（单曲、固定 4 中心，seed 12345）单独过拟合时，activity loss 从
0.6523 降到 ~0.17–0.25，活跃窗口的 P 达到 ~0.9999（`runs/` 内实验记录）。说明头与 loss 有
拟合能力；随机采样 fake 特征在这点预算下没有收敛到阈值以上，与本语料/特征强度有关，不作为
真实模型能力判断。

诚实记录：调参期间观察过训练集 loss 与 **val** 单曲的 P 分布（用于判断是否收敛），因此 val
不是完全未触碰的保留集；**test 没有参与任何配置选择**，但同样只有 1 首、只用于最终报告。

### 5.3 真实冻结 AuT 短训练（已完成，两次同配置 100 step）

两个 100 step 运行使用同一 `aut_short.toml`、seed 与数据，未做任何超参选择：

- **首次运行**（§5.3.1）：代码 `dc0c8a70a468533854d5a0d0882964048a163a39`，原始产物保留在
  远端 `runs/train/aut-short` 与本地回拷，未被覆写；
- **资源测量验证运行**（§5.3.2）：review 修复后的代码 `3649aba`，在新克隆/新输出目录重跑，
  逐 step loss 与保留集指标与首次运行完全一致；额外记录设备、整体耗时与 CUDA 峰值。

#### 5.3.1 首次真实短训练（dc0c8a7）

在授权 GPU 服务器（A6000，共享环境只读复用 issue #5 的模型与 venv）执行：

```sh
PYTHONPATH=src OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    timeout 600 /path/to/venv/bin/python scripts/train.py train \
    --config configs/train/aut_short.toml \
    --model-dir /path/to/encoder-checkpoint --device cuda:0
```

预算与结果：

- 配置 `steps = 100`、`groups_per_step = 2`、`centers_per_item = 4`、`batch_windows = 8`、
  fp32、独立 2 s 窗；评估 val/test 各 1 首；`timeout 600` 未触发；
- `status = completed`，100/100 step 有活动监督；逐 step 计算时间合计 **87.1 s**
  （首步 6.5 s 含预热，其后约 0.85–0.93 s/step）；
- 训练 loss（activity / empty / positive / negative）：step 1 `0.7026 / 0.7345 / 0.0004 /
  0.6148` → step 50 `0.0678 / 0.0414 / 0.0 / 0.0` → step 100 `0.1030 / 0.3190 / 0.0050 /
  0.3297`；身份负例项在后期才出现；
- 有效计数：`activity_terms = 2400`、`empty_terms = 4000`、`positive_terms = 149`、
  `negative_terms = 572`、`ambiguous_groups = 126/200`（并列最优匹配时身份监督被 mask）、
  `truncated = 0`、`supervision_masked = 0`；
- encoder provenance：`Qwen/Qwen3-Omni-30B-A3B-Instruct` revision
  `26291f793822fb6be9555850f06dfe95f2d7e695`（explicit）、fp32、`block_diagonal`/`sdpa`、
  layer `final`、`independent_windows`；数据集 digest 与 §2.2 一致、`leak_free = true`；
- checkpoint 还记录 Git SHA、`uv.lock` SHA-256
  `5bdbf6873db97152dcb7b801c323d03dce6b26484eaeee83277da7b4be8fdb55` 与全部 RNG 状态。

保留集结果（阈值 0.5，整曲全局映射，各 1 首、每首 376 个有效中心）：

| split | active labels | TP | FP | FN | precision | recall | F1 | ID switch | 来源数绝对误差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 364 | 329 | 856 | 35 | 0.278 | 0.904 | 0.425 | 0 | 5 |
| test | 236 | 218 | 297 | 18 | 0.423 | 0.924 | 0.581 | 3 | 2 |

同一协议的基线（F1）：

| split | all_inactive | all_active | no_identity |
|---|---:|---:|---:|
| val | 0.0 | 0.216 | 0.425 |
| test | 0.0 | 0.145 | 0.578 |

#### 5.3.2 资源测量验证运行（3649aba）

同一命令、同一 seed/配置/数据，在新输出目录重跑（前两次尝试因 torch 2.14 在 CUDA 懒初始化下
拒绝显式 device index 而立即失败、未消耗 GPU 计算；已改为 `torch.cuda.init()` + 选中设备后
reset，由此运行验证）：

- `status = completed`，100/100 step，`timeout 600` 未触发；逐 step 计算时间合计 **92.4 s**
  （首次 87.1 s，属共享机器波动）；
- 逐 step loss（step 1/50/100）与累计有效计数、val/test 指标、三个基线均与 §5.3.1 完全一致；
- `resources`：`device = cuda:0`、`device_index = 0`、整体 `elapsed_seconds = 189.73`
  （含编码器加载与评估）、`peak_allocated_bytes = 2,951,400,960`（2.95 GB）、
  `peak_reserved_bytes = 3,323,985,920`（3.32 GB）；scope 为**本进程** torch allocator
  统计，覆盖构造（编码器加载前）到 run 结束，其他进程不可见；reset 只发生在构造时一次。
- CPU 路径明确记 `peak_allocated_bytes = null`、`peak_reserved_bytes = null` 与
  `unavailable_reason`，不伪造 0。

**诚实结论**：100 step 的真实短训练在保留集上明显优于全不活跃/全活跃基线（召回高、过预测导致
precision 低）；但 `no_identity`（丢弃全部身份向量）与模型结果几乎相同（val 甚至完全一致，
test 只低约 0.003），在这个极小预算与数据规模下，**没有测到身份向量带来的收益**。ID switch
在 val 为 0、test 为 3，来源数误差 5/2。这不是质量结论，也不是可放大的泛化声明；val 在开发中
被观察过、每 split 仅 1 首、训练集只有 4 首自合成曲目。后续需要更多数据与更长训练才能讨论
身份学习曲线。

### 5.4 结果文件位置

所有音频、checkpoint、日志与评估 JSON 只写在忽略目录（本地 `runs/train*`，远端授权工作根下
的 `issue-8-training/`）。公共仓库只包含代码、配置与本文件的脱敏汇总，不含绝对路径、权重、
checkpoint、音频或凭据。

## 6. 限制与未验证项

- 真实运行只做了 100 step / 100 次 optimizer step，单卡单次，未做超参扫描；这不构成对 AuT
  特征质量或 E/P 协议上限的结论。
- 数据是 4+2+2 首自合成短曲，音色只覆盖内置 sampler 参数；未验证 Surge、真实乐器、强效果与
  强重叠；没有做学习曲线。
- 评估每 split 只有 1 首，指标方差未知；val 在开发期被查看过。没有把 test 用于调参，但也不
  宣称 test 是统计学意义上的独立泛化集。
- 固定 batch 过拟合证据只说明头/loss 的拟合能力，不代表随机采样训练会收敛。
- `ambiguous_groups` 采用 issue #7 的既有语义（并列最优时身份项 mask、活动项对称平均），本
  issue 未修改 loss 或匹配策略；126/200 的训练组出现并列，是当前数据/特征的可辨识度证据。
- fake 与 real 的数值差异（批量 vs 单条）按 issue #5 实测记录在 checkpoint provenance 中；本
  issue 固定 fp32 和分块策略以减少差异，但没有做逐位等价声明。
- 推理接口 `aat.training.inference.HeadInference` 只负责 E/P 预测与协议对象；整歌轨迹关联仍
  由 issue #9 的 tracker 消费，本 issue 不重新实现关联。
- 未实现特征缓存：每 step 重新编码，避免 stale 特征；如未来加入缓存，key 必须覆盖音频摘要、
  窗口/时间/mask、模型 revision/layer/attention/dtype/预处理。
- 推理模式修复（`eval()` + dropout 关闭）不会改变已记录的 100 step 指标：该运行 `dropout = 0`，
  且 3649aba 验证运行逐 step loss 与保留集指标与之完全一致，故不以此归罪旧结果。
- callback 路径在未提供 `audio_duration_seconds` 时只能判定起点边界，终点边界无法知悉；
  需要精确边界掩码时使用 `predict_at_times` 或显式传 duration。
- 日志对账会丢弃并保留 checkpoint 之后未审计的 tail；若 canonical 日志与 checkpoint 前缀不
  兼容（缺失/乱序/非整数 step），直接抛 `CheckpointError`，不会拼接伪造历史。
- GPU 峰值是**本进程** torch allocator 统计，不含其他进程/系统显存；编码器加载前的瞬态峰值
  已在 reset 前覆盖（scope 写明从构造开始）。
