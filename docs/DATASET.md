# 数据集索引与多窗口采样（DATASET，issue #6）

## 目标数据与当前索引边界

[身份活动向量与局部拼接规范](IDENTITY_ACTIVITY_DESIGN.md)（#48）要求连续分轨 RMS、
真实静音、原时间轴和统一强度尺度。本文其余章节记录 **当前 v0.1.0 概率标签、索引与采样实现**；
`source_present` 是监督来源列掩码，不是推理中持久静音 E 或额外 Q 的要求。
旧 0/1 标签不能静默改作 RMS；迁移须显式版本化，并同步标签、索引摘要、读写校验与训练消费方。

目标课程从简单不重叠到交错/局部重复、overlap 与更真实音乐；采样须覆盖共同事件和形状分岔。
非相邻 same_source_pairs 不自动证明单窗有共同证据，也不要求互不相交上下文 E 天然相似。
长训练段含 A-B-A 不代表生成一次 E 的输入能看见两端；保持实际 W/H、有效音频和原时间追溯。

当前 build/sample_batch 拒绝 `S>K`（包括静音参考来源），不是未来显著性竞争策略。
目标 K 是局部输出容量，整曲来源可更多；较弱/溢出活动可不被输出，后续需部分参考匹配与
可表示集合评估，不能强迫 K 输出拟合所有来源。#46 本轮固定 K≥来源数，容量不能解释该轮失败。
本次不改索引或容量代码，先分轨监督再独立比较混音方案。

本文说明 `src/aat/data/`、`scripts/build_dataset_index.py` 与 `tests/data/` 如何把真实渲染
分轨和声学中心活动标签组织成可训练样本。范围只覆盖**扫描校验、分组切分、多中心窗口采样**；
不做训练、不做跨歌曲叠加增强，也不通过改写标签掩盖渲染问题。

上游接口完全复用已合并协议：

- 样本目录、`manifest.json`、`sources.json`、`controls.json`、`activity.json/.npz` 均按
  [SCHEMAS.md](SCHEMAS.md) v0.1.0 读取；未知版本直接拒绝。
- 标签由 [ACTIVITY_LABELS.md](ACTIVITY_LABELS.md)（`scripts/label_sample.py`）生成；
  本任务不重新定义阈值。
- 窗口和时间取整复用 `aat.windowing`，与标签器的 `valid` 语义一致。
- 真实样本由 [RENDERING.md](RENDERING.md)（`scripts/render_sample.py`）产出。

## 1. CLI 快速开始

```sh
# 渲染一个真实样本（需要 render extra）
uv sync --locked --extra render
uv run --no-sync python scripts/render_sample.py render \
    --config configs/render/smoke.toml --out runs/dataset/real-smoke-0001
uv run --no-sync python scripts/label_sample.py runs/dataset/real-smoke-0001

# 扫描一个数据根下的全部 labeled 样本，校验并切分
uv run --no-sync python scripts/build_dataset_index.py build \
    --data-root runs/dataset \
    --out runs/dataset-index/index.json \
    --seed 20260929 --ratios 0.8,0.1,0.1 --slots 8 --window-seconds 2.0

# 重新校验（重新计算摘要、重读文档、重查泄漏）
uv run --no-sync python scripts/build_dataset_index.py verify \
    --index runs/dataset-index/index.json

# 查看计划/实际比例、超大组、空集合；--samples 追加逐样本行
uv run --no-sync python scripts/build_dataset_index.py summary \
    --index runs/dataset-index/index.json --samples

# 采样一个同曲多中心 batch 并打印（不会写音频文件）
uv run --no-sync python scripts/build_dataset_index.py batch \
    --index runs/dataset-index/index.json --split train \
    --items 1 --centers 4 --min-gap 5 --seed 20260929
```

退出码：`0` 成功，`1` 校验发现问题，`2` 输入/数据不可构建。索引与 batch 输出都应放在被忽略的
目录（`runs/`）中；索引文件本身只记录数据根相对路径，不写本机绝对路径。

## 2. 样本发现与校验

`discover_samples` 在数据根下按**排序后的目录序**递归查找 `manifest.json`；数据根自身不允许
是样本目录（相对路径是可移植索引的前提）。每个样本必须满足：

| 校验 | 规则 |
|---|---|
| 阶段 | `manifest.stage == "labeled"`，且带成对的活动标签路径 |
| 摘要 | 重算 `manifest.content_sha256` 覆盖的**全部**文件（mix、stems、sources、controls、activity.json、activity.npz）；缺失或不符即失败 |
| 标签 | `activity.json/.npz` 通过 `ActivityData.load`；`arrays_path` 与 manifest 声明一致 |
| 来源列 | `activity.source_ids` 与 `sources.json` 的 `source_ids` **顺序和取值完全一致**；`manifest.stem_paths` 键集合一致 |
| 标签版本 | `label_params` 必须存在；`LabelConfig.from_label_params` 校验 `activity-label-v1` 版本与 SHA-256，未知版本拒绝 |
| 窗口 | 标签快照的 `center_window_seconds` 必须等于构建时的 `--window-seconds`，否则报错并给出应使用的参数 |
| 时间范围 | 所有 `activity.center_times` 落在 `[track_start_seconds, track_start_seconds + duration_seconds]` |
| 实际音频 | mix 的采样率等于 manifest；时长差不超过 `--duration-tolerance-seconds`（默认 0.01 s） |
| 容量 | `len(source_ids) <= K`（默认 `--slots 8`）；**静音来源也算数**，超限直接拒绝 |
| 身份 | 跨目录 `sample_id` 重复直接拒绝，并列出冲突路径 |

错误信息统一带样本位置，例如
`sample 'a/s1': content_sha256['mix.wav']: expected …, got …`。

`controls.json` 只做结构校验、来源引用校验和事件统计；**MIDI 永不参与标签**。

## 3. 索引文档（`dataset-index-v1`）

索引是纯 JSON，无时间戳，同 seed/同输入重建字节一致。

```json
{
  "index_version": "dataset-index-v1",
  "plan": {"seed": 20260929, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
           "slots": 8, "window_seconds": 2.0},
  "layout": {"data_root": "../dataset", "sample_path_base": "data_root"},
  "summary": {"sample_count": 1, "split_counts": {"train": 1, "val": 0, "test": 0},
              "split_ratios_requested": {...}, "split_ratios_actual": {...},
              "component_count": 1, "largest_component_size": 1,
              "empty_splits": ["val", "test"], "oversized_components": [...],
              "warnings": [...], "cross_split_assets": {"composition": 0, "preset": 0, "sample_origin": 0},
              "leak_free": true},
  "components": [{"component_id": "c0000", "split": "train", "size": 1,
                  "sample_ids": ["smoke-0001"], "assets": {...}, "oversized": true}],
  "samples": [ ... ]
}
```

`samples[]` 每条记录：

| 字段 | 单位/类型 | 说明 |
|---|---|---|
| `sample_id` | string | manifest 的稳定 ID |
| `path` | POSIX string | 相对数据根的样本目录 |
| `split` | `train`/`val`/`test` | 整体分组切分结果 |
| `sample_rate` | Hz | mix 采样率 |
| `duration_seconds` / `track_start_seconds` | s | 原曲时间轴信息 |
| `groups` | object | `composition` 标量；`preset`/`sample_origin` 排序去重列表，可为空 |
| `source_ids` | array[string] | `sources.json` 列顺序（与 activity 列一致） |
| `labels` | object | `metadata_path`/`arrays_path`/`center_count`/`valid_count`/`hop_seconds`/`center_window_seconds`/`label_version`/`label_config_sha256` |
| `controls` | object | `event_count`/`note_on_count`/`note_min`/`note_max` 与逐来源音高范围（仅证据） |
| `audio` | object | mix 路径、采样率、声道数、帧数、时长 |
| `content_sha256` | object | manifest 记录的逐文件摘要（相对样本目录） |
| `usable` | bool | 是否有来源且 `valid_count > 0`；即默认 `valid_only=True` 下可用。`valid_only=False` 仍可消费只有无效边界的样本（`center_count > 0`） |
| `warnings` | array[string] | 逐样本提示（无有效中心、全静音等） |
| `sample_sha256` | hex | 整条记录（除本字段）的 canonical JSON SHA-256 |

`sample_sha256` 在解析索引时立即重算校验，索引文件被篡改会直接报错。加载时若能从
`layout.data_root`（或显式 `--data-root`）解析数据根，`DatasetIndex.load` 默认调用
`verify_dataset`：重新扫描样本、重算全部摘要、对比记录与当前输入，并重新检查泄漏。发现
文件变化/输入过期/资产跨集合时会报出全部问题。

## 4. 分组与切分

三个分组键使用**独立命名空间**，空列表不制造共享资产：

```text
("composition", <标量>)          # 每个样本恰好一个
("preset", <列表元素>)            # 空列表 -> 无键
("sample_origin", <列表元素>)     # 同上
```

样本按共享任一命名空间资产做并查集合并，得到连通分量。A-B 共享 preset、B-C 共享
sample_origin 时，A/B/C 属于同一分量（传递关联），整体进入同一个集合，因此
`train/val/test` 之间不存在共享资产的交叉；索引同时写入
`cross_split_assets`（必须全为 0）与 `leak_free: true` 作为证据，加载/校验时会重新检查。

切分规则：

- 分量不可拆分；优先保证“无泄漏”，其次才接近请求比例。
- 分量按 `(-size, 最小 sample_id)` 排序，逐个分配给当前 deficit 最大的集合；deficit 相同时用
  显式 `seed` 的确定性随机数打破平局。固定 seed + 固定输入 → 字节级相同索引。
- 请求比例会归一化；非法输入（负数、非有限、全 0、数量不对、未知键、非法 seed）直接报错。
- 空数据根是合法输入：生成 0 样本索引，三个集合都空并给出 warning。
- 超大分量（`size > 最大集合目标`）整体保留，`summary.oversized_components` 与 `warnings`
  给出分量 ID、大小和资产；`empty_splits` 与 `split_ratios_actual` 报告真实比例。
  **不会为了凑比例拆开连通组，也不会把空集合粉饰成“已验证的泛化集”**。
- 记录 `split_ratios_requested` 与 `split_ratios_actual` 两个字段，便于 review 时核对偏差。

## 5. 多中心采样接口（`sample_batch`）

一次调用返回 `DatasetBatch`：`blocks[]` 每个元素对应**一首歌**，绝不把两首歌的音频混在一起。
同一 block 内来源列顺序固定为 `sources.json` 顺序，跨多个中心时刻不重排来源。

```python
from aat.data import DatasetIndex, sample_batch

index = DatasetIndex.load("runs/dataset-index/index.json")   # 默认重新校验输入
batch = sample_batch(
    index,
    "runs/dataset",
    seed=20260929,            # 显式采样 seed
    split="train",
    items=1,                  # 选几首歌；None = 该集合全部
    centers_per_item=4,       # 每首歌几个中心时刻
    min_center_gap=5,         # 中心网格最小间隔（单位：hop 步）
    valid_only=True,          # 只用 activity.valid 为 True 的中心（默认）
    activity_threshold=0.5,   # same_source_pairs 的活动阈值（作用于声学标签）
    verify_digests=True,      # 默认核验实际消费的 mix/activity 摘要（可关，见 §5.6）
)
block = batch.blocks[0]
```

### 5.1 数组 shape 与单位

| 字段 | shape | dtype | 单位/语义 |
|---|---|---|---|
| `source_ids` | `(S,)` | string | 真实来源 ID，固定列顺序（来自 `sources.json`） |
| `slot_ids` | `(K,)` | string \| None | 槽位映射；前 `S` 个是来源，其余 `None` 为 padding 空槽 |
| `center_times` | `(N,)` | float64 | **原曲绝对秒**，保留追溯信息 |
| `center_indices` | `(N,)` | int64 | 在 `activity.npz` 中心网格中的行号 |
| `center_valid` | `(N,)` | bool | 来自 `activity.valid`：该中心是否可作监督 |
| `activity` | `(N, K)` | float32 | 中心活动标签 `[0,1]`；padding 槽恒 0 |
| `source_present` | `(N, K)` | bool | 真实来源列为 True（**即使静音**），padding 槽为 False |
| `audio` | `(N, W)` | float32 | mix 中心窗；多声道按算术平均下混为单声道；边界补零 |
| `audio_valid` | `(N, W)` | bool | True 表示该样本是真实音频，False 是补零 |
| `same_source_pairs` | `(P, 3)` | int | `(slot, 行a, 行b)`：同一来源在两个中心都活动且网格间隔 ≥ `min_center_gap` |
| `source_note_ranges` | `(S, 2)` | int \| None | 来自 `controls.json` 的音高范围，仅证据、不参与标签 |

`track_start_seconds`、`sample_rate`、`window_seconds`、`window_samples`、`hop_seconds`、
`activity_sha256`、`mix_sha256`、`slots`、`path`、`split` 也记录在 block 上，便于复盘与重放。

### 5.2 三个掩码的区别

| 掩码 | 含义 | 用途 |
|---|---|---|
| `source_present` | 该槽位是否是这首歌的真实来源（`sources.json` 声明） | 区分“静音来源”（True 且 activity=0）与“padding 空槽”（False 且 activity=0） |
| `center_valid` | 标签器的模型中心窗是否完整落在渲染音频内（`activity.valid`） | **监督门**：False 的行不得用于训练损失 |
| `audio_valid` | 该窗口样本是否来自真实音频（边界补零为 False） | 音频侧补零遮罩；与 `center_valid` 不是同一个概念 |

默认 `valid_only=True` 时只会抽到 `center_valid=True` 的行，且当
`window_seconds == label_params.config.center_window_seconds` 时 `audio_valid` 全为 True。
`valid_only=False` 可显式包含首尾无效行用于检查：此时 `center_valid=False`，
`audio_valid` 可能出现补零；两者都必须由消费方屏蔽，不能把无效首尾当有效监督。
当标签窗口大于样本时长（例如 0.5 s 样本 + 2.0 s 窗口）时 `activity.valid` 可能全为
False：默认 `valid_only=True` 会按 §5.5 拒绝/跳过，而 `valid_only=False` 仍可采样（26
个中心里的任意非空中心），返回的 `center_valid` 全 False、`audio_valid` 标记补零、
`same_source_pairs` 为空，消费方必须全部屏蔽。
`window_seconds` 覆盖成与标签窗口不同时，两个掩码的差别会更明显（有测试覆盖）。

### 5.3 非相邻窗口与不同音高

- `min_center_gap` 是最小网格步距：`0`/`1` 允许相邻窗，`2` 起禁止紧邻窗口。选择算法
  （`_select_centers`）先用反向贪心判断可行性，再在可行区间内用 `seed` 随机取点，
  因此同 seed 可复现、不同 seed 不同，且**不会退化成只取相邻窗口**。
- `same_source_pairs` 给出同一来源在多个非相邻中心的活动配对，可直接用于同源正样本；
  测试验证了间隔、活动阈值与来源正确性。
- 不同音高由渲染控制（`patterns` 的 `note`）产生，索引与 block 记录
  `controls`/`source_note_ranges` 作为证据；**标签仍然只来自分轨声学 activity**。
  测试覆盖“改 controls 音高 → 标签数组不变”。
- 不允许用偶然相同的 `source_id` 字符串当作跨曲身份：来源列只在同一样本内有意义，
  block 也只包含单一样本的数据。

### 5.4 采样率、时间取整与非零起点

`audio` 由 `aat.windowing.extract_windows_at_times` 截取：半开窗、**half-up** 采样换算、
首尾补零并由 `audio_valid` 标记；`track_start_seconds != 0` 时 `origin_seconds` 传入原曲
绝对起点，`center_times` 始终是绝对时间。16 kHz 与 44.1 kHz 都有测试；非零
`track_start_seconds`（含 44.1 kHz）也有对照测试。

### 5.5 容量与异常处理

- `S > K`（含静音来源）在 `build` 与 `sample_batch` 都会**明确拒绝**，不静默截断来源；
  `slots` 覆盖更小的 K 同样拒绝。
- `valid_only=True`（默认）下 `valid_count == 0` 的样本不可用：默认 `on_unusable="error"`
  报错；`on_unusable="skip"` 时跳过并在 `DatasetBatch.skipped` 记录
  `(sample_id, reason)`，`requested_items` 与 `returned_items` 都会体现差额。
- `valid_only=False` 时不因 `valid_count == 0` 拒绝：只要来源非空且
  `center_count > 0` 就会尝试采样（仍受 `centers_per_item`/`min_center_gap` 可行性约束）；
  真正的零中心（`center_count == 0`）与无来源在任何模式下都不可用。
- 可用中心少于 `centers_per_item`、`min_center_gap` 不可行时也走 `on_unusable` 策略。
- 空集合采样返回空 batch；显式 `items > 可用样本数` 报错而不是重复采样。
- 索引记录 `usable` 与逐样本 `warnings`，`verify_dataset` 会重查这些条件。

### 5.6 输入摘要核验与 opt-out

`DatasetIndex.load` 的校验只代表**加载时刻**的数据；采样时可能已换 root、或在 build 之后
重写了文件。因此 `sample_batch` 默认 `verify_digests=True`：每个 block 在解析前重新计算
**实际消费的三个文件**（mix、activity metadata、activity arrays）的 SHA-256，并与索引记录
逐项比对；不一致时报错并给出 `sample_id`、文件角色/相对路径、索引记录值、实际值以及重建
索引的命令提示。它只哈希这三个文件，不会全库重扫；解析后还会把实际 frames/采样率/来源列/
标签计数与索引记录再比对一次（同 shape 重写无法蒙混）。

受控热路径可显式传 `sample_batch(..., verify_digests=False)` 或 CLI
`batch --no-verify-digests`：这表示**明确接受未经验证的输入**，block 的 `mix_sha256`/
`activity_sha256` 仍写索引记录值而不是实际消费字节的摘要，且不会在采样时回写索引；调用方
应先自行运行 `verify_dataset`。摘要校验发生在解析之前，同一文件的并发写入竞态不在范围内。

## 6. 真实证据（2026-09-29，本机 Windows，CPU）

以下为完整真实链路，使用已合并的 DawDreamer 渲染器和 `label_sample.py`：

```sh
uv sync --locked --extra render
uv run --no-sync python scripts/render_sample.py render \
    --config configs/render/smoke.toml --out runs/dataset/real-smoke-0001
uv run --no-sync python scripts/label_sample.py runs/dataset/real-smoke-0001
uv run --no-sync python scripts/build_dataset_index.py build \
    --data-root runs/dataset --out runs/dataset-index/index.json --seed 20260929
uv run --no-sync python scripts/build_dataset_index.py verify --index runs/dataset-index/index.json
uv run --no-sync python scripts/build_dataset_index.py batch \
    --index runs/dataset-index/index.json --split train --items 1 --centers 4 --min-gap 5
```

实测结果：

- 渲染：4 来源 20 s 音乐 + 3 s 尾音（23.0 s），分轨求和 `max_abs_error_lsb = 1.0`
  （容差 3.5），尾音 margin 2.5 s。
- 标签：4 个分轨，1151 个中心（1051 valid），active fraction 0.325/0.174/0.143/0.412。
- 索引：1 个样本、1 个连通分量、`leak_free = true`、`cross_split_assets` 全 0；
  请求 0.8/0.1/0.1 时 `split_counts = {train: 1, val: 0, test: 0}`，
  `empty_splits = ["val","test"]`，warning 明确说明单个连通组无法填满三集合。
- 校验：`verify: ok: 1 sample(s), 9 digest-checked file(s)`。
- batch（`min-gap 5`）：44100 Hz、`window_samples = 88200`、4 个中心
  `[5.44, 11.24, 16.94, 20.94]`，`slot_ids = [s01,s02,s03,s04,null,null,null,null]`，
  padding 槽 activity 全 0 且 `source_present=False`，`audio_valid` 全 True；
  另一次 `min-gap 8` 采样得到 6 个中心（最小间隔 11），`same_source_pairs = ((3, 0, 2),)`，
  `source_note_ranges = ((36,36),(36,36),(43,48),(60,67))`（不同音高来自渲染 controls）。

CI 不跑 20 s 真实渲染：`tests/data/` 以临时合成 fixture 为主（真实 WAV + 真实协议文档 +
共享标签器），`tests/data/test_integration_render_chain.py` 使用 `smoke_ci.toml` 的 2 s 真实
DawDreamer 渲染并在缺少 render extra 时自动 skip。以上单个真实样本只证明链路可用，
**不用于宣称切分比例或模型质量**。

## 7. 限制与未验证项

- 真实证据只有单个 2–4 来源自合成样本；未验证多曲比例是否合理、未做学习曲线或模型指标。
- 分组只基于 manifest 声明的 `composition`/`preset`/`sample_origin`。若素材复用未被如实写入
  manifest，索引无法发现；这属于上游渲染配置的责任。
- 切分是确定性的贪心 deficit 分配，不是比例误差最优解；超大组/单连通组会明显偏离请求比例，
  由 `empty_splits`、`split_ratios_actual`、`oversized_components` 与 warnings 如实报告。
- `sample_batch` 每次调用会重新读取 mix 与 activity；默认还会重算这三个文件的 SHA-256
  （每个 block 3 次哈希，大 mix 的 I/O 成本可见，可用 `verify_digests=False` 显式关闭），
  没有特征缓存、DataLoader worker 或 `torch` 依赖，面向 M1 规模。后续训练 issue 需要缓存/
  多进程时另行设计。
- 多声道下混采用**算术平均**；标签能量使用等功率平均（见 ACTIVITY_LABELS.md），两者用途不同，
  未验证强空间化/相位相反的素材。
- 采样选择是“可行区间内带 seed 的随机”，不是均匀分布，也没有按活动比例分层。
- 只验证了 16 kHz 与 44.1 kHz；其它采样率会走同一套 `aat.windowing` 换算但未逐一测试。
- `verify_dataset` 的摘要重算需要读取全部音频/标签文件；大库上属于显式 I/O 成本，可在热路径用
  `verify_files=False` 跳过文件重算（索引记录本身的 `sample_sha256` 始终校验）。
- 不提交任何音频、索引或 batch 转储；`runs/` 已被忽略。

## 8. 测试

```sh
# 基础环境：dataset 单测 + CLI + fixture 链路（无需 DawDreamer）
uv sync --locked
uv run --no-sync pytest -q -p no:cacheprovider tests/data

# 完整环境：包含真实渲染 → 标签 → 索引 → batch 集成链路
uv sync --locked --extra render
uv run --no-sync pytest -q -p no:cacheprovider
```

`tests/data/` 覆盖：固定 seed 重建同一索引；三种分组键在 train/val/test 无交叉；
A-B/B-C 传递关联防泄漏；超大组/空集合真实比例；重复 `sample_id`、坏摘要、未知版本、
来源列不一致、时间越界、音频时长不符、容量超限；静音来源 vs padding 空槽；
`center_valid`/`audio_valid` 区别；非相邻窗口与同源配对；不同音高证据不改标签；
44.1 kHz、16 kHz 与非零 `track_start_seconds`；非法 seed/ratio/slots/参数；空数据与空集合。

近一步的过期输入回归：同 shape 重写 mix 后默认 `sample_batch` 报
`mix 'mix.wav' sha256 mismatch`（`verify_digests=False` 仍可显式消费，但 block 保留索引
记录摘要、不静默重写）；同 shape 改活动值/标签 metadata 同样被拦；索引与数据根被复制到
另一个 root 后同名同 shape 的改写也会被检出；边界-only（`valid_count=0`、`center_count>0`）
样本默认拒绝、`valid_only=False` 可采样并给出全 False/补零掩码，零中心样本两种模式都
明确不可用。
