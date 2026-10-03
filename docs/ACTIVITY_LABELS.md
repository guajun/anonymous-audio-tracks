# 活动标签（ACTIVITY_LABELS，issue #4）

## 连续强度目标与旧标签

目标见 [身份活动向量与局部拼接规范](IDENTITY_ACTIVITY_DESIGN.md)（#48）：
使用实际效果后、进入混音的分轨连续线性 RMS，而非存在概率或 ADSR 控制值。
应保留真实静音、原时间轴与同一强度尺度，记录能量窗、声道聚合、采样率、增益、效果与延迟；
不得逐窗峰值归一化或制造持续底音。共享非线性母带处理时，需说明分轨参考生成边界。

本文其余章节记录 **main 当前标签器与 activity-label-v1 / schema v0.1.0**：
RMS 经 dBFS 阈值、滞回及 release_hold 后产出 0/1 probability 代理；这些值不是连续 RMS。
`release_hold_seconds` 是旧标签后处理，不是身份活动的 E 下边沿，也不是 ADSR R。
E 下边沿是进入零的表征事件，严格零不携带身份；弱活动/容量挤出/漏识别的无输出也不是真实零。

未来连续标签需显式版本/单位迁移，不能仅改同名 activity 数组的解释；与
[SCHEMAS](SCHEMAS.md)、[DATASET](DATASET.md)、[TRAINING](TRAINING.md) 同步。
连续 RMS 标签的导出/训练消费与新协议尚未实现；现有 RMS 能量计算仍用于旧阈值标签。
本次不修改阈值或重写历史标签。

本文说明 `src/aat/labels/` 与 `scripts/label_sample.py` 如何从真实分轨音频生成
协议 v0.1.0 的**中心时刻活动标签**（`activity.json` + `activity.npz`）和可读摘要
（`activity_summary.json`）。范围只覆盖活动标签；身份跟踪、重复起音检测、对未知歌曲
自动编造真值都不在范围内。

## 1. 两种窗口必须分开

| 窗口 | 配置字段 | 默认 | 用途 |
|---|---|---|---|
| 模型中心窗 | `center_window_seconds` | 2.0 s | **只**用于计算协议 `valid`：中心窗是否完整落在渲染音频内 |
| 局部能量窗 | `energy_window_seconds` | 0.05 s | **只**用于计算声学 RMS 包络与阈值 |

`valid[t]` 的语义完全遵循 `docs/SCHEMAS.md` §4.2：`center_times[t] ± W_c/2` 完整落在
`[track_start_seconds, track_start_seconds + duration_seconds]` 内时为 `True`，否则消费方
应屏蔽该行。局部能量窗再短也不会改变 `valid`；`valid` 为 `False` 的中心仍会计算声学
标签，但它们不是模型窗的有效监督。

所有时间都是**原曲绝对秒**：`center_times = track_start_seconds + i * hop_seconds`，
不会重设为切片相对时间。`track_start_seconds` 偏移只平移时间轴，不改变标签序列。

## 2. 包络与阈值

1. **包络**：对每个采样点计算局部矩形窗内的均方值（多声道取等功率平均），再转成
   dBFS。窗口边界只用实际存在的样本求平均；数字静音向下钳到 `-240 dBFS`，保证 JSON
   摘要里不出现 `-inf`。
2. **每分轨阈值**（都在配置中显式化）：

   ```text
   peak              = max(envelope_db)
   noise_floor       = percentile(envelope_db, noise_floor_percentile)
   noise_floor_gate  = noise_floor + noise_floor_margin_db
   relative_gate     = peak - peak_relative_threshold_db
   on_threshold      = max(absolute_threshold_dbfs, min(noise_floor_gate, relative_gate))
   off_threshold     = on_threshold - hysteresis_db
   ```

   - 绝对阈值保证很安静的素材不会被强行标成活动；
   - 噪声底门限在底噪远离峰值时抬高阈值；
   - `min(noise_floor_gate, relative_gate)` 把自适应门限封顶在
     `peak - peak_relative_threshold_db`，防止“持续信号把自适应底噪抬到自己身上”
     导致全静音；代价是当噪声与峰值相差小于 `peak_relative_threshold_db` 时，
     阈值不保证高于底噪，背景噪声可能被标成活动（`low_snr` 只是提示）。
   - 三者都是配置项，随 `label_params` 快照保存并可用 SHA-256 校验；默认值是
     M1 分析起点，尚未在真实分轨上校准。
3. **滞回与尾音**：状态机在电平达到 `on_threshold` 时打开，只有低于 `off_threshold`
   才关闭；关闭后再保留 `release_hold_seconds` 的尾音。因此 note-off 之后仍在衰减的
   尾音可以继续保持活动，而 note-on（`controls.json` 中的事件）从不参与状态机，
   不会被强制当作听觉 onset。尾音窗远小于模型窗，中心标签不会变成整个 2 秒窗的 OR。
4. **概率**：输出为 `0.0/1.0` 的二值声学代理。它是可复现的阈值判断，不声称等于主观
   可听性，也不做“新起音”检测。

## 3. 控制事件与音频分离

`controls.json` 只作为参考数据读取：校验来源引用、统计事件数量写入摘要，并在摘要里
标注 `control_events_reference.used_for_labels = false`。标签完全由分轨音频推导；
修改 note-on/note-off 不会改变任何标签（有测试覆盖）。

## 4. 配置、版本与可重建性

- 配置类：`aat.labels.LabelConfig`（冻结 dataclass），完整字段写入
  `activity.json` 的 `label_params.config`，配 `label_params.version =
  "activity-label-v1"` 和 `label_params.sha256 = config_sha256()`（按字段排序的紧凑
  JSON 的 SHA-256）。
- 重建与校验：

  ```python
  from aat.labels import LabelConfig
  config = LabelConfig.from_label_params(activity_metadata["label_params"])
  assert config.config_sha256() == activity_metadata["label_params"]["sha256"]
  ```

  版本不匹配或 SHA 不匹配会抛 `LabelError`，不会静默套用旧语义。

## 5. CLI 用法

```sh
uv run --no-sync python scripts/label_sample.py <sample_dir> [--config label_config.json|.toml]
```

步骤与校验：

1. 加载 `manifest.json`、`sources.json`、`controls.json`，校验 `sources` 引用和
   `manifest.stem_paths` 与 `sources.source_ids` 完全一致（列顺序按 `sources.json`）。
2. **在写入任何文件之前**先解析并检查所有输出路径：分别解析
   `activity.json`、`activity.npz`、summary 与 manifest 的最终路径，拒绝绝对路径、
   盘符路径和 `..` 穿越；任何输出与其他输出或输入（manifest、sources、controls、
   `mix.wav`、各 stem）碰撞时立即失败，因此自定义 summary 文件名不能覆盖受保护文件。
3. 逐一核对 `manifest.content_sha256` 中的每个路径摘要；缺失或摘要不符立即失败。
4. 读取每个 stem 的 PCM WAV：采样率必须等于 `manifest.sample_rate`，时长与
   `manifest.duration_seconds` 的差不得超过 `--duration-tolerance-seconds`（默认
   0.01 s；必须是有限且非负的值，`NaN`/`Inf` 会被拒绝）。支持 PCM 8/16/24/32-bit、
   IEEE float 32/64-bit 与 `WAVE_FORMAT_EXTENSIBLE`；压缩格式直接拒绝。
5. 生成 `activity.json` + `activity.npz`（通过 `ActivityData.save`，写后重新
   `ActivityData.load` 校验），并写 `activity_summary.json`。
6. 默认把 manifest 推进到 `stage = labeled`，写入两个标签路径和两条 SHA-256；
   其余渲染摘要保持不变。`--no-manifest-update` 可只写标签文件。

`--config` 接受 JSON 或 TOML：扁平字段直接覆盖 `LabelConfig`；TOML 中若存在
`[task]` 表，则 `task.window_seconds → center_window_seconds`、
`task.hop_seconds → hop_seconds`，因此可直接读取 `configs/experiment.toml` 的窗口
起点。

## 6. 摘要字段（`activity_summary.json`）

顶层包含 `summary_version`、生成器、样本/采样率/时长/原曲偏移、hop、两种窗口、
完整配置与 `config_sha256`、`center_count`、`valid_count`、`source_order`、
`control_events_reference` 和每个来源的 `sources[]`：

| 字段 | 含义 |
|---|---|
| `active_fraction` / `valid_fraction` | 活动/有效中心占比（全部中心行为分母） |
| `active_seconds` | 样本级活动总时长 |
| `segments` | 活动连续段数量 |
| `first_active_seconds` / `last_active_seconds` | 首/末活动绝对时间，可为 `null` |
| `peak_dbfs` / `noise_floor_dbfs` / `snr_db` | 包络统计与信噪比 |
| `on_threshold_dbfs` / `off_threshold_dbfs` | 实际生效阈值 |
| `silent` | 整段无活动 |
| `low_snr` | 有活动且 `snr_db < low_snr_threshold_db` |
| `ambiguous_fraction` / `ambiguous` | 中心电平落在 on 阈值 ± `ambiguity_margin_db` 的比例，及其是否超过 `ambiguity_fraction_threshold` |

这些标记只是可读提示，不改变 `activity.npz` 的协议内容。

## 7. 测试与未验证项

- 测试全部使用程序合成的小波形（`tests/labels/signals.py`），覆盖脉冲、慢 attack、
  延音、静音、重复音、尾音、滞回、平移协变、原曲偏移、16k/22.05k/44.1k 采样率、
  WAV 解码、协议校验、manifest 摘要与 CLI 端到端。没有真实 DawDreamer/Surge 分轨
  参与校准，默认阈值只是 M1 起点。
- 立体声按等功率平均合并；未验证强空间化或相位相反的素材。
- 不做重采样：stem 采样率必须与 manifest 一致。公共 API 要求 declared
  `duration_seconds` 与解码帧数之差不超过容差（默认 0.01 s），网格仍由 declared
  duration 生成以对齐预测窗；`valid` 依据实际样本边界与
  `centered_window_bounds` 语义计算，越界/补零中心为 `False`，不会把最后一个样本
  重复当作越界中心的能量。
- `valid` 依据实际解码样本边界和 `aat.windowing` 的 half-up 采样换算与
  `centered_window_bounds` 语义计算（有与
  `extract_windows_at_times(...).mask.all(axis=1)` 的对照测试），包括奇数窗宽、
  非整数样本窗、44.1 kHz 和非零原曲起点。
- 阈值是声学代理，不是主观可听性真值；未知歌曲不自动生成“真值”。
