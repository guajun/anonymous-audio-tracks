# 数据协议 v0.1.0（SCHEMAS）

本文件固化 issue #2 的共享 JSON/NPZ 接口。后续渲染（#3）、活动标签（#4）、特征（#5）、输出头/损失（#7）、轨迹（#9）和试听页（#10）都必须使用这里的字段、单位、形状和空值语义，不得各自发明并行协议；确需变更时先向主会话申报并同步提升版本号。

Python 侧实现与校验位于 `src/aat/contracts/` 和 `src/aat/windowing.py`；测试位于 `tests/contracts/`，全部使用合成小数组，不加载真实音频或模型。

## 0. 通用约定

### 0.1 版本与文档类型

每个 JSON 文档必带两个头部字段：

| 字段 | 类型 | 必填 | 取值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | 当前 `"0.1.0"` | 协议版本；读取方遇到未知版本必须拒绝，不得猜测 |
| `kind` | string | 是 | `sample_manifest` / `sources` / `controls` / `activity` / `feature` / `prediction` / `trajectory` | 文档类型，与协议一一对应 |

NPZ 文件自身不带版本号；版本、单位、形状和列顺序由同目录 JSON sidecar 记录。sidecar 是权威元数据。

### 0.2 时间轴：保留原曲起点

- 所有时间字段都是**原曲时间轴上的绝对秒**（float64，单位 s）；`t = 0` 是原曲开头。
- 渲染切片若从原曲 `T0` 秒开始，则 `manifest.track_start_seconds = T0`，且 `activity/feature/prediction/trajectory` 的时间仍是绝对时间，**不得重置为切片内相对时间**。
- `aat.windowing.extract_windows_at_times(..., origin_seconds=T0)` 用 `origin_seconds` 表示 `audio[0]` 的原曲时刻；首尾补零只发生在原曲/切片边界。
- 控制事件、活动中心时间、特征帧时间和轨迹中心时间之间可直接按秒对齐，不需要再叠加偏移。

### 0.3 采样率、秒与样本索引

| 概念 | 约定 |
|---|---|
| 采样率 | `sample_rate` 正整数，单位 Hz；16 kHz 与 44.1 kHz 都只是一次换算输入 |
| 秒 → 样本 | `seconds_to_samples(seconds, rate)`，四舍五入（half up）：`floor(seconds * rate + 0.5)` |
| 样本 → 秒 | `samples_to_seconds(samples, rate)`，精确除法 |
| 窗口样本数 | `window_sample_count(window_seconds, rate)`；小于 1 个样本报错 |
| 全局窗口/步长 | 原型 `window_seconds = 2.0`、`hop_seconds = 0.02`（来自 `configs/experiment.toml`，待实验确认） |

例：0.02 s 在 16 kHz 是 320 样本，在 44.1 kHz 是 882 样本；2 s 窗口分别是 32000 与 88200 样本。

### 0.4 空输入、无活动与空曲线表示

| 场景 | 规范表示 |
|---|---|
| 空时间轴（无窗口/无帧） | 对应数组首维长度 0，例如 `center_times.shape == (0,)`、`activity.shape == (0, S)`；不得用 NaN 占位 |
| 无来源 | `sources.json` 中 `sources: []`；活动数组第二维长度 0，如 `(T, 0)` |
| 某来源无活动 | 该列/该行概率全部为 0.0，数组保留；不得删除行或写 NaN |
| 窗口中心处零长度音轨 | `center_times(duration=0, ...)` 返回空数组，不生成“全补零中心” |
| 预测无候选 | 该窗口 `slot_valid` 全 False、`embeddings` 全零、`activity` 全零；形状仍为 `(N, K)` / `(N, K, D)` |
| 轨迹空曲线 | 直接从 `tracks` 中省略；不允许长度为 0 的 track |
| 静音但保留身份 | track 保留完整时间序列，`activity` 可为全 0（身份记忆），见 §7 |

### 0.5 槽位（slot）与身份（track_id）

- **原始来源身份（source_id）**：生成期控制句柄，定义在 `sources.json`，只在本样本监督内有效；同一 `source_id` 在不同歌曲间没有继承关系。
- **槽位（slot）**：一次前向输出的局部容量编号，`0 <= slot < K`；K 默认 8、可配置。槽位只表示容量，**不等于身份**，相邻窗口的同一槽位可能换成另一个来源。
- **稳定轨迹身份（track_id）**：由跨窗口一对一关联产生的身份，跨窗口、短暂静音和重现保持稳定，整首歌累计轨迹数不必等于 K。`trajectory.json` 以 `track_id` 为主键；原始槽位来源只以可选的 `slot_indices` 记录。

### 0.6 强制校验（违者拒绝）

- 任何数组/数值出现 NaN 或 Inf；
- 概率不在 `[0, 1]`；
- 需要严格递增的时间数组不是严格递增（控制事件只要求非递减，见 §3）；
- `source_ids` 或 `track_id` 重复；
- 同一文档内数组长度/形状不一致；
- JSON 版本未知或 `kind` 不匹配；
- 有效候选 embedding 不是单位向量（见 §6）；
- 路径字段是绝对路径、含 `..` 或使用反斜杠。

### 0.7 参考实现入口

| 用途 | 入口 |
|---|---|
| 版本、K/D 默认、单位向量容差 | `aat.contracts.version` |
| JSON 文档（manifest/sources/controls/trajectory） | `aat.contracts.documents` |
| NPZ 文档（activity/feature/prediction） | `aat.contracts.arrays` |
| 统一导出 | `aat.contracts` |
| 窗口与时间换算 | `aat.windowing` |
| 异常 | `ContractError`（数据违约）、`SchemaVersionError`（版本不符）、`WindowError` / `WindowRangeError`（窗口参数或越界） |

## 1. `manifest.json`（`kind = sample_manifest`）

描述一个渲染样本的随机种子、采样率、时长、原曲偏移、划分分组、产物路径和内容摘要。

| 字段 | 类型 | 必填 | 单位/取值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | `"0.1.0"` | 版本 |
| `kind` | string | 是 | `"sample_manifest"` | 类型 |
| `sample_id` | string | 是 | 非空 | 样本稳定 ID |
| `seed` | int | 是 | `>= 0` | 渲染随机种子，保证同 seed 可追溯 |
| `sample_rate` | int | 是 | Hz，`>= 1` | 样本音频采样率 |
| `duration_seconds` | float | 是 | s，`> 0` | 样本音频有效时长（不含尾部补零） |
| `track_start_seconds` | float | 是 | s，`>= 0` | `mix.wav` 的 0 时刻在原曲时间轴上的位置；全曲样本为 0 |
| `groups` | object | 是 | 见下 | 划分分组键 |
| `groups.composition` | string | 是 | 非空 | 曲目/作品分组 |
| `groups.preset` | string | 是 | 非空 | 音色 preset 分组 |
| `groups.sample_origin` | string | 是 | 非空 | 素材来源分组 |
| `mix_path` | string | 是 | 相对 POSIX 路径 | 混音文件，如 `mix.wav` |
| `stem_paths` | object | 是 | `source_id -> 相对路径` | 至少一项；键必须与 `sources.json` 的 `source_id` 一致 |
| `sources_path` | string | 是 | 相对 POSIX 路径 | 通常 `sources.json` |
| `controls_path` | string | 是 | 相对 POSIX 路径 | 通常 `controls.json` |
| `activity_metadata_path` | string | 是 | 相对 POSIX 路径 | 通常 `activity.json` |
| `activity_arrays_path` | string | 是 | 相对 POSIX 路径 | 通常 `activity.npz` |
| `versions` | object | 是 | string -> string | 必须含 `renderer`；记录渲染器/插件/管线版本 |
| `content_sha256` | object | 是 | 路径 -> 64 位小写 hex | 必须覆盖 `mix_path`、全部 stems、`sources_path`、`controls_path`、`activity_metadata_path`、`activity_arrays_path` |
| `render_latency_seconds` | float | 否 | s，`>= 0` | 插件/渲染延迟记录 |
| `tail_seconds` | float | 否 | s，`>= 0` | 尾音保留时长 |
| `notes` | string | 否 | 任意 | 自由备注 |

无活动/空表示：manifest 本身描述非空样本，因此 `duration_seconds > 0` 且至少一个 stem；不描述“无来源样本”。

示例：

```json
{
  "schema_version": "0.1.0",
  "kind": "sample_manifest",
  "sample_id": "synth-0001",
  "seed": 20260929,
  "sample_rate": 16000,
  "duration_seconds": 4.0,
  "track_start_seconds": 0.0,
  "groups": {"composition": "comp-01", "preset": "preset-a", "sample_origin": "surge-factory"},
  "mix_path": "mix.wav",
  "stem_paths": {"s01": "stems/s01.wav", "s02": "stems/s02.wav"},
  "sources_path": "sources.json",
  "controls_path": "controls.json",
  "activity_metadata_path": "activity.json",
  "activity_arrays_path": "activity.npz",
  "versions": {"renderer": "dawdreamer-0.x"},
  "content_sha256": {"mix.wav": "0000000000000000000000000000000000000000000000000000000000000000"}
}
```

（示例只示意结构；实际 `content_sha256` 必须覆盖全部必需路径。）

## 2. `sources.json`（`kind = sources`）

定义生成期原始来源身份及其顺序。数组列顺序只由本文件决定。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema_version` | string | 是 | 版本 |
| `kind` | string | 是 | `"sources"` |
| `sample_id` | string | 否 | 关联样本 ID |
| `sources` | array | 是 | 来源对象数组，可为空 `[]`（此时无来源） |
| `sources[].source_id` | string | 是 | 非空且全局唯一 |
| `sources[].index` | int | 是 | `>= 0`，必须等于数组位置（0-based） |
| `sources[].renderer` | string | 否 | 渲染器引用，如 `surge_xt` |
| `sources[].preset_ref` | string | 否 | 音色/预设引用，相对路径或标识 |
| `sources[].sample_ref` | string | 否 | 采样素材引用 |
| `sources[].seed` | int | 否 | `>= 0`，该来源的随机种子 |
| `sources[].notes` | string | 否 | 备注 |

协议不加入乐器/track 语义字段（如 instrument、BPM）；需要时由上层配置单独保存。

## 3. `controls.json`（`kind = controls`）

生成期控制事件（MIDI、参数自动化、素材触发），与声学活动标签分离。

| 字段 | 类型 | 必填 | 单位/取值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | `"0.1.0"` | 版本 |
| `kind` | string | 是 | `"controls"` | 类型 |
| `sample_id` | string | 否 | 非空 | 关联样本 ID |
| `events` | array | 是 | 可为空 `[]` | 事件按时间非递减排列 |
| `events[].time_seconds` | float | 是 | s，`>= 0`，原曲绝对时间 | 允许同一时刻多个事件（同时起音） |
| `events[].source_id` | string | 是 | 必须存在于 `sources.json` | 目标来源 |
| `events[].event_type` | string | 是 | 非空，如 `note_on` / `note_off` / `param` / `trigger` | 事件类型（开放集合） |
| `events[].data` | object | 否 | 默认 `{}` | 事件负载，如 `{"note": 60, "velocity": 100}` |

无控制事件：`events: []`。note-on 不等于可听起音，note-off 不等于尾音结束；可听活动以 §4 的活动标签为准。`aat.contracts.documents.Controls.validate_source_references(source_ids)` 用于跨文件检查未知来源引用。

## 4. `activity.json` + `activity.npz`（`kind = activity`）

逐来源的中心时刻活动标签。`P` 类语义：某来源在**窗口中心时刻**是否可听活动，不代表“整个窗口曾出现”。

### 4.1 `activity.json` 元数据

| 字段 | 类型 | 必填 | 单位/取值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | `"0.1.0"` | 版本 |
| `kind` | string | 是 | `"activity"` | 类型 |
| `sample_id` | string | 否 | 非空 | 关联样本 ID |
| `sample_rate` | int | 是 | Hz，`>= 1` | 标签与音频的采样率 |
| `source_ids` | array[string] | 是 | 唯一、非空字符串 | 列顺序；必须与 `sources.json` 顺序一致，长度 = S |
| `hop_seconds` | float | 否 | s，`> 0` | 中心时间标称步长 |
| `label_params` | object | 否 | 任意 | 标签阈值/时间窗/滞回等参数快照（#4 校准后写入） |
| `arrays_path` | string | 是 | 相对路径 | 通常 `"activity.npz"` |
| `arrays` | object | 是 | 见 §4.2 | 每个数组的 dtype/shape/unit 声明；读取方校验 |

### 4.2 `activity.npz` 数组

| 键 | dtype | 形状 | 单位/值域 | 说明 |
|---|---|---|---|---|
| `center_times` | float64 | `[T]` | s，原曲绝对时间，`>= 0` 且严格递增 | `t=0` 为原曲起点 |
| `activity` | float32 | `[T, S]` | 概率 `[0, 1]` | `activity[t, s]` 为 `source_ids[s]` 在 `center_times[t]` 的活动概率 |
| `valid` | bool | `[T]` | bool | `False` 表示该中心窗口需要补零、应由消费方屏蔽 |

空/无活动：`T = 0` 表示无中心时间；`S = 0` 表示无来源；某来源无活动即该列为全 0；不得用 NaN。数组键必须恰好是上表三项。

## 5. `feature.json` + `feature.npz`（`kind = feature`）

一维/二维音频特征（如冻结 AuT 或 Mel/CNN 局部特征）。特征**不带来源语义**，只是输出头的输入。

### 5.1 `feature.json` 元数据

| 字段 | 类型 | 必填 | 单位/取值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | `"0.1.0"` | 版本 |
| `kind` | string | 是 | `"feature"` | 类型 |
| `sample_id` | string | 否 | 非空 | 关联样本 ID |
| `feature_name` | string | 是 | 非空 | 提取器/层配置标识，如 `"aut.layer-12"`；随代码/配置变更 |
| `feature_dim` | int | 是 | `>= 1` | 特征维度 D，必须等于数组第二维 |
| `sample_rate` | int | 是 | Hz，`>= 1` | 源音频采样率 |
| `frame_origin_seconds` | float | 是 | s，`>= 0` | 第 0 帧的原曲绝对时刻（通常 0.0 或首帧时间） |
| `hop_seconds` | float | 是 | s，`> 0` | 帧网格标称步长；`frame_times` 才是权威时间 |
| `backend` | string | 否 | 非空 | 提取器标识（如 `aut`、`mel`） |
| `preprocessing` | object | 否 | 任意 | 归一化/重采样参数快照 |
| `arrays_path` | string | 是 | 相对路径 | 通常 `"feature.npz"` |
| `arrays` | object | 是 | 见 §5.2 | 数组声明 |

### 5.2 `feature.npz` 数组

| 键 | dtype | 形状 | 单位/值域 | 说明 |
|---|---|---|---|---|
| `frame_times` | float64 | `[T]` | s，原曲绝对时间，`>= 0` 且严格递增 | 权威帧时间，可与标称 hop 网格存在实现级偏差 |
| `features` | float32 | `[T, D]` | 实数，禁止 NaN/Inf | `features[t, d]` 为第 t 帧第 d 维 |
| `valid` | bool | `[T]` | bool | `False` 表示该帧由补零音频计算，输出头应忽略或降权 |

空表示：`T = 0`，`features` 形状为 `(0, D)`；`D` 仍由 `feature_dim` 声明。帧数、维度、掩码长度任一不一致即拒绝。

## 6. `prediction.json` + `prediction.npz`（`kind = prediction`）

输出头的核心协议：`E[N, K, 128]` 与 `P[N, K]`。N 为窗口数，K 为槽位容量（默认 8，可配置），128 为默认 embedding 维度（可配置，由 `embedding_dim` 记录）。

### 6.1 `prediction.json` 元数据

| 字段 | 类型 | 必填 | 单位/取值 | 说明 |
|---|---|---|---|---|
| `schema_version` | string | 是 | `"0.1.0"` | 版本 |
| `kind` | string | 是 | `"prediction"` | 类型 |
| `sample_id` | string | 否 | 非空 | 关联样本 ID |
| `slots` | int | 是 | `>= 1`，默认配置 8 | 槽位容量 K，必须等于 `E` 第二维 |
| `embedding_dim` | int | 是 | `>= 1`，默认配置 128 | 身份向量维度 D，必须等于 `E` 第三维 |
| `hop_seconds` | float | 否 | s，`> 0` | 窗口步长 |
| `provenance` | object | 否 | 见 §7.2 | 推理运行来源 |
| `arrays_path` | string | 是 | 相对路径 | 通常 `"prediction.npz"` |
| `arrays` | object | 是 | 见 §6.2 | 数组声明 |

### 6.2 `prediction.npz` 数组

| 键 | dtype | 形状 | 单位/值域 | 说明 |
|---|---|---|---|---|
| `center_times` | float64 | `[N]` | s，原曲绝对时间，`>= 0` 且严格递增 | `center_times[n]` 是窗口 n 的中心时刻；`P[n]` 属于该时刻 |
| `embeddings` | float32 | `[N, K, D]` | 有效候选为 L2 单位向量（容差 `1e-3`） | `E[n, k]`；无效槽位必须全零 |
| `activity` | float32 | `[N, K]` | 概率 `[0, 1]` | `P[n, k]` 为中心活动概率；无效槽位必须为 0 |
| `slot_valid` | bool | `[N, K]` | bool | `True` 表示该槽位是可用候选；仍可能 `P = 0`（静音但保留身份） |
| `center_valid` | bool | `[N]` | bool | `True` 表示窗口完全位于音频内部；`False` 表示首尾补零窗口，消费方可降权 |

槽位语义：

- 有效候选（`slot_valid = True`）：`E` 必须有限且 L2 范数为 1（容差 `1e-3`）；`P` 可为 0——这是短暂静音的“身份记忆”状态，低活动概率的向量不更新轨迹原型。
- 无效槽位（`slot_valid = False`）：`E` 全零、`P = 0`，不携带身份；关联时不得匹配。
- 槽位编号不是身份；跨窗口关联输出稳定 `track_id`（§7）。
- 空表示：`N = 0` 时形状为 `(0, K, D)`、`(0, K)`、`(0,)`、`(0, K)`；不得用 NaN。

数值/形状违规（NaN、概率越界、非单位有效向量、长度不一致等）直接报 `ContractError`。

## 7. `trajectory.json`（`kind = trajectory`）

整歌轨迹关联输出：稳定身份、中心时刻、活动概率和运行来源。

### 7.1 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema_version` | string | 是 | 版本 |
| `kind` | string | 是 | `"trajectory"` |
| `sample_id` | string | 否 | 关联样本/曲目 ID |
| `slots` | int | 否 | `>= 1`；给出时用于校验 `slot_indices` 范围 |
| `provenance` | object | 是 | 运行来源，见 §7.2 |
| `params` | object | 否 | 关联参数快照，如 `{"activity_threshold": 0.5, "match_threshold": 0.7}` |
| `tracks` | array | 是 | 轨迹数组，可为空 `[]` |

### 7.2 `provenance` 运行来源

| 字段 | 类型 | 必填 | 单位/取值 | 说明 |
|---|---|---|---|---|
| `run_id` | string | 是 | 非空 | 运行唯一标识 |
| `model_id` | string | 否 | 非空 | 模型/检查点标识 |
| `git_commit` | string | 否 | 7–40 位小写 hex | 代码版本 |
| `config_hash` | string | 否 | 非空 | 推理配置摘要 |
| `created_at_utc` | string | 否 | ISO-8601 且带 UTC 时区（如 `...Z`） | 创建时间 |
| `notes` | string | 否 | 任意 | 备注 |

### 7.3 `tracks[]` 轨迹对象

| 字段 | 类型 | 必填 | 单位/取值 | 说明 |
|---|---|---|---|---|
| `track_id` | string | 是 | 非空、文件内唯一 | 稳定身份，不是槽位编号 |
| `center_times` | array[float] | 是 | s，原曲绝对时间，`>= 0` 且严格递增 | 至少 1 个点；长度为 0 必须省略整个 track |
| `activity` | array[float] | 是 | `[0, 1]`，长度与 `center_times` 相同 | 中心活动概率；全 0 表示身份在静音中保持 |
| `confidence` | array[float] | 否 | `[0, 1]`，长度相同 | 关联置信度 |
| `slot_indices` | array[int] | 否 | `0 <= x < slots`，长度相同 | 原始槽位来源，便于追溯；不参与身份判定 |
| `notes` | string | 否 | 任意 | 备注 |

空曲线规范：没有轨迹时 `tracks: []`；已经存在的轨迹不能写成空数组，必须省略；静音重现保留同一 `track_id` 并在曲线中记 0，而不是新建身份。

## 8. 目录布局与 NPZ 键汇总

一个样本目录（二进制音频被仓库忽略，不提交）：

```text
sample/
  mix.wav
  stems/<source_id>.wav
  manifest.json
  sources.json
  controls.json
  activity.json
  activity.npz        # center_times, activity, valid
  feature.json
  feature.npz         # frame_times, features, valid
  prediction.json
  prediction.npz      # center_times, embeddings, activity, slot_valid, center_valid
  trajectory.json
```

NPZ 约定：由 `numpy.savez` 写出、`numpy.load(..., allow_pickle=False)` 读取；键必须恰好等于上表，禁止 pickle 对象。这样后续纯浏览器读取只需解析 zip/.npy。

## 9. 窗口与时间工具（`aat.windowing`）

| 函数 | 语义 |
|---|---|
| `seconds_to_samples(seconds, rate)` | 秒到样本，四舍五入 half-up |
| `samples_to_seconds(samples, rate)` | 样本到秒 |
| `window_sample_count(window_seconds, rate)` | 窗口样本数，`< 1` 报错 |
| `uniform_times(origin, hop, count)` | 均匀时间网格，返回原曲绝对秒 |
| `center_times(duration, hop, origin_seconds=0)` | 覆盖 `[origin, origin + duration]` 的中心时间网格；起点精确为 `origin`；非整步长时最后一个中心不超过末尾；`duration == 0` 返回空 |
| `centered_window_bounds(center, width)` | 半开窗口边界 `[c - W//2, c - W//2 + W)`；奇数长度多出的样本在右侧 |
| `extract_centered_windows(audio, center_samples, window_samples)` | 逐中心截取，越界补零；返回 `(windows[N, W], valid[N, W])` |
| `extract_windows_at_times(audio, center_seconds, rate, window_seconds, origin_seconds=0)` | 绝对时间版本；`valid` 为 True 处是真实样本 |

窗口规则：中心必须有限、非负、严格递增且位于音频时间范围内；首尾明确补零并返回有效掩码；消费方可用 `mask.all()` 判定“完整窗口”。越界抛 `WindowRangeError`，参数非法抛 `WindowError`，不静默截断。

## 10. 版本策略

- 任何字段语义、单位、形状或空值语义的变化都必须提升 `schema_version`，并在本文件同步更新。
- 允许新增**可选**字段且保持旧字段语义不变；为此读取方容忍未知额外字段，但本文档列出的必填字段缺失即拒绝。
- 历史文档无迁移脚本；发现版本不符时拒绝读取，交由相应 issue 显式处理。
