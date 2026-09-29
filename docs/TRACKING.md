# 轨迹关联与身份评估（issue #9）

本文件说明 `src/aat/tracking/`、`src/aat/evaluation/` 与 `scripts/track.py` 的关联算法、配置、协议映射和评估定义。协议仍为 v0.1.0；本模块只读写 `docs/SCHEMAS.md` 已冻结的 `prediction` / `trajectory` / `activity` 文档，不新增字段、不修改依赖。

所有时间都是原曲绝对秒：`trajectory.audio.track_start_seconds` 记录被分析音频 `audio[0]` 的原曲位置，中心时间和边界误差都不换算成切片相对时间。

## 1. 数据流

```text
prediction.json + prediction.npz          activity.json + activity.npz（参考）
        │  PredictionSequence                      │
        ▼                                          │
associate_sequence(config)                         │
        │                                          │
        ▼                                          ▼
trajectory.json  ────────────────►  evaluate_trajectory(config) ──► 指标对象
```

滑窗模型入口（fake callback 或未来训练好的输出头）：

```text
audio ──extract_windows_at_times──► windows[N, W]
      ──predictor(windows)──► E[N, K, 128], P[N, K][, slot_valid]
      ──predict_windows 归一化/规范化──► PredictionData（协议校验）──► 轨迹
```

模块入口：

| 用途 | 入口 |
|---|---|
| 关联配置（全部阈值） | `aat.tracking.TrackingConfig` |
| 预测序列视图与校验 | `aat.tracking.PredictionSequence` |
| 余弦与一对一分配 | `aat.tracking.cosine_similarity` / `maximum_assignment` |
| 整曲关联 | `aat.tracking.associate_sequence` / `track_prediction` |
| 滑窗回调与全链路 | `aat.tracking.predict_windows` / `build_prediction_data` / `track_audio` |
| 整曲评估 | `aat.evaluation.evaluate_trajectory` |
| CLI（prediction → trajectory） | `scripts/track.py` |

## 2. 关联算法

### 2.1 每窗处理顺序

1. **过期**：`t - last_match_time > retention_seconds` 的轨迹终止，之后不再补点。
2. **窗口有效性**：`center_valid[n] = False`（首尾补零窗口）不匹配、不出生、不补记忆点。
3. **门控**：候选对（轨迹原型, 有效槽位向量）余弦 `< match_threshold` 的先禁用（“阈值外边先禁用”）。有效槽位由 `slot_valid[n, k]` 决定；`P` 不参与门控，`P = 0` 的有效候选仍是身份记忆。
4. **一一分配**：在可行对集合上先最大化匹配数，再最大化总余弦（“最大可行匹配数再最高余弦”）。K 小默认 8，使用位掩码 DP 精确求解；当任一维度较小而另一维超过阈值时自动转置后精确求解；只有两个方向都超过安全阈值才退化为“按相似度贪心”，仍然一一对应且经过门控，但不保证最大基数（`maximum_assignment` 文档已注明）。
5. **匹配结果**：匹配轨迹记录该中心时刻的 `P`、`confidence = clip(cosine, 0, 1)` 和原始 `slot`；未匹配的存活轨迹补 `activity = 0`、`slot_indices = null` 的记忆点，不伪造槽位。
6. **原型更新**：只有 `P >= activity_threshold` 的匹配候选更新原型：`proto ← normalize(alpha·proto + (1-alpha)·E)`；`prototype_alpha = 0` 表示直接替换。低 P 候选可以匹配、记活动、刷新存活时间，但**绝不动原型**。
7. **出生**：没有匹配上的候选，若 `P >= birth_threshold`（默认等于 `activity_threshold`）则新建 `trk-XXXX`，以该候选向量为初始原型。低 P 噪声不会产生新身份。

### 2.2 静音、重现与终止

- 轨迹在 `retention_seconds` 内没有门控匹配即终止；内存中每隔一个有效窗口补全零活动点。
- 来源在保留时间内重现并再次通过余弦门控时接回**同一 `track_id`**；超出保留时间则产生新的 `track_id`（整歌累计 ID 数可以大于 K）。
- 槽位编号只是容量编号：同一轨迹可在不同窗口对应不同槽位，匹配只看 embedding，绝不按槽位下标继承身份。

### 2.3 配置

| 参数 | 默认 | 含义 |
|---|---|---|
| `activity_threshold` | 0.5 | `P` 达到该值才算“活动”，才更新原型/参与出生 |
| `match_threshold` | 0.7 | 余弦门控；低于它的候选对先禁用 |
| `retention_seconds` | 1.0 | 静音身份保留时长；超时终止，超时后重现算新身份 |
| `prototype_alpha` | 0.9 | 原型 EMA 的旧值权重，`[0, 1)` |
| `birth_threshold` | 同 `activity_threshold` | 另可单独配置的出生阈值 |
| `max_exact_slots` | 16 | 位掩码精确匹配的槽位数上限，超过则贪心退化 |

`trajectory.params` 会记录上述参数的 JSON 快照和 `"matching": "gated_max_cardinality_then_cosine"`，便于回溯运行配置。

## 3. 滑窗回调入口

回调签名：`predictor(windows: np.ndarray[N, W]) -> WindowPrediction | (E, P) | (E, P, slot_valid)`。

- `E` 形状必须为 `[N, K, 128]`，`P` 为 `[N, K]`，`P ∈ [0, 1]`；维度必须等于请求的 `slots`，embedding 维度按协议固定 128。
- `predict_windows` 会把有效 embedding 重新归一化为单位 L2 向量，把无效槽位规范化为全零 `E` 且 `P = 0`，随后交给契约对象校验（`PredictionData` 构造即验证）。
- 未显式给 `slot_valid` 时，零范数 embedding 视为无效槽位；显式把零向量标为有效则报 `TrackingError`。
- `track_audio(...)` 完成音频 → 窗口 → 回调 → `PredictionData` → `Trajectory` 全链路，`track_start_seconds` 用于非零起点的切片，时间保持原曲绝对秒。

测试中使用的 fake callback 是确定性的能量/整型构造，不加载模型；真实模型前向验证仍依赖 #5/#7/#8。

## 4. 轨迹 JSON

输出由 `Trajectory` 契约对象序列化并通过协议校验：

- `audio` 记录被分析音频范围；所有轨迹点必须落在范围内（容差 `1e-6 s`）。
- `provenance` 原样传递调用方的 `RunProvenance`，`data_kind` 区分 `model` / `annotation` / `mock`。
- `tracks[].center_times` 严格递增；`activity` 长度一致；`confidence = cosine`；`slot_indices` 为原始槽位或 `null`（静音记忆）。
- 空曲线直接省略轨迹；`duration_seconds = 0` 时 `tracks` 必须为空。

CLI 示例（预测文档不记录音频范围，因此必须显式提供时长与起点）：

```sh
uv run --no-sync python scripts/track.py \
  --prediction outputs/run-01 --output outputs/run-01/trajectory.json \
  --duration-seconds 120.0 --run-id run-01 --data-kind mock
```

## 5. 评估定义（`aat.evaluation.evaluate_trajectory`）

输入是 `ActivityData`（参考）与 `Trajectory`（预测）。配置：`activity_threshold`（默认 0.5）、`time_tolerance_seconds`（默认 `1e-6`）。

### 5.1 时间对齐

预测点先对齐到**完整**的参考中心时间网格（含 `valid = False` 帧），再用 `valid` 掩码决定哪些帧参与评估：

- 落在显式无效帧上的预测点被忽略（既不是有效证据也不是误检）；
- 容差内不落在任何参考网格时间的预测点是真正的“仅预测帧”，活动时计入误检；
- 已对齐的二值化结果显式与 `valid` 掩码相与；没有预测点（未观测）的填充 0 不会被二值化，即使 `activity_threshold = 0` 也不会被当成真实活动；观测到的预测值才按 `P >= activity_threshold` 判定（阈值为 0 时观测到的 0 也算活动，这是显式阈值语义）；
- 参考帧没有对应预测点时按预测活动 0 处理；无效帧不参与活动、ID、边界任何统计，且会打断参考活动段，不被当作连续区段或声学边界。

容差对齐不插值概率。同一轨迹多点映射同一帧时取最大活动。

### 5.2 全曲一致映射与活动 P/R/F1

对二值化后的参考活动与预测活动构建来源 × 轨迹的重叠计数矩阵；在“重叠 > 0”的可行对上做一一分配，**最大化全曲总重叠（TP）而不是匹配对数**，允许来源/轨迹未匹配。在固定的 GT/预测活动总量下，最大化总 TP 等价于最大化 micro-F1，因此不会为了多配一对而牺牲主重叠。之后**整曲固定这套映射**计算：

- 每对映射：`TP = 参考活动 ∧ 预测活动`，`FP = ¬参考 ∧ 预测`，`FN = 参考 ∧ ¬预测`；
- 未映射来源：其活动帧全部记 FN；
- 未映射轨迹与仅预测活动帧：记 FP；
- 报告微平均与逐来源的 precision / recall / F1。

在线关联仍保持“最大配对数优先”的原目标，不受此处影响。参考标签**不逐帧重排**：交换身份的输出会掉 F1，不会因为逐帧重映射而伪装成满分。

### 5.3 ID switch

对每个参考来源，沿时间逐帧检查其有效活动帧：

- 仅当该帧**恰好一个有效活动 GT 来源且恰好一条活动预测轨迹**时才建立可辨认 owner；owner 与上一个可辨认 owner 不同时计一次 ID switch；
- 有预测但多个 GT 来源同时活动、或多条预测轨迹活动，或两者兼有 → 无法仅凭活动真值归属该轨迹，该帧对每个受影响来源计入 `ambiguous_owner_frames`，不任意编造身份，并保留上一个可辨认 owner；
- 没有任何预测轨迹活动 → 缺测，跳过但不重置记忆。

记忆跨越缺测、静音记忆、GT 重叠歧义帧保留：来源消失后重现或歧义结束后若换了轨迹，会在下一次可辨认帧计为 switch。两首来源重叠严重的音乐里大量帧会被记为歧义（该诊断的覆盖限制），ID switch 仍是独立诊断，不替代活动 F1；连续活动期间的交换由逐帧 owner 变化与 F1 共同体现。

### 5.4 来源数误差

`reference_source_count` 为有任意有效活动帧的参考来源数，`predicted_track_count` 为有任意有效活动帧（含仅预测帧）的轨迹数；落在无效帧上的活动不计。报告有符号 `source_count_error` 与 `source_count_abs_error`。全静音或空输入时为 0。

### 5.5 边界误差

边界统计覆盖**全部参考来源**（含完全未映射的来源，其所有活动段计入 `boundary_segments_missed`）：

- 参考活动段按有效帧切分，无效帧会打断连续段；
- 映射轨迹的预测活动段忽略落在无效帧上的点；
- 每个参考段与映射轨迹中重叠最大的预测段配对；没有重叠段的参考段计入 `boundary_segments_missed`；单点段用容差判定“命中”；
- 与无效帧相邻的段边缘是未知位置、不是声学边界，不计入 onset/offset 误差；只有真实的起止边缘才累计 `|预测边界 − 参考边界|`。

只有比较过至少一段时才报告 MAE，否则为 `None`。

### 5.6 空值与边界语义

- `F1 = 2TP / (2TP + FP + FN)`，只要该分母非零就定义：有 GT 但完全没预测时 F1 = 0，只有误检没有 GT 时 F1 = 0；只有 `TP = FP = FN = 0` 才为 `None`；
- `activity_threshold` 可为 0；未观测的填充 0 不会因阈值为 0 变成预测，只有实际观测的预测点参与阈值判定；
- precision 和 recall 各自按自己的分母独立判定是否 `None`（无预测时 precision 为 `None`、无 GT 正例时 recall 为 `None`）；
- 没有真实边界可比时对应 MAE 为 `None`；
- 空输入、全静音、单点段都不伪造精度；
- `to_dict()` 输出 JSON 安全（无 NaN/Inf），未定义项为 `null`。

## 6. 证据与未验证项

已验证（`tests/tracking/`，可控 embedding / 合成音频，不依赖模型）：

- 槽位随机置换、相似 embedding、同时活动、低 P 不拖动原型；
- 静音后保留期内接回原轨、超时终止后重现产生新 ID、整歌累计 ID 多于 K；
- 空输入、全静音、K 可变、补零窗口跳过；
- 故意交换身份的序列被 ID switch 与 F1 同时检测；连续活动内换轨、跨缺测/歧义记忆、GT 来源重叠不虚报 owner、歧义后真实换轨计一次、全未映射来源的边界漏检均有回归；
- `activity_threshold = 0` 的显式阈值语义：全无效帧、部分有效、无预测点填充、观测到的 0 活动四种情形；
- 全曲评估映射以总重叠为目标的反例（100 帧主重叠 vs 2 条小重叠）与来源/轨迹顺序置换不变性；
- 无效参考帧掩码：落在无效帧的预测不误检、真正仅预测帧仍计 FP、无效间隙不桥接边界段；
- `F1` 在“有 GT 无预测”“仅误检”时为 0、只有全零才 `None`；
- trajectory 协议校验与 `save`/`load` 往返；
- fake callback 的音频 → 窗口 → E/P → 轨迹全链路与绝对时间保持；
- 位掩码精确分配与暴力枚举一致（随机小矩阵回归，含 `total_score` 目标）。

未验证 / 风险：

- 没有真实训练模型：全部“模型前向”证据来自 fake callback；真实 AuT 输出头（#5/#7/#8）接入后需要重新校准 `match_threshold`、`activity_threshold`、`retention_seconds` 与 `prototype_alpha`；
- ID switch 只在“唯一有效活动 GT 来源且唯一活动预测轨迹”的可辨认帧建立 owner；GT 来源重叠或预测轨迹重叠的帧按受影响来源计为 `ambiguous_owner_frames` 而不猜 owner（重叠音乐覆盖有限，需要固定真实音乐抽查 #11 复核）。
- 合成音频只验证管线正确性，不代表分离或跟踪质量；
- `slots > max_exact_slots` 且转置后也无法精确求解时退化为贪心一一分配，可能损失基数；默认 K=8 走精确 DP，来源数少而轨迹多时通过转置仍可精确求解。
