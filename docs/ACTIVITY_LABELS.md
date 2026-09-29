# 分轨中心活动标签（#4）

本文件说明 `src/aat/labels/`、`scripts/label_sample.py` 与 `tests/labels/` 实现的行为：如何从渲染分轨生成与滑窗中心对应的活动标签、所有可复现参数、静音/低信噪比/歧义标注，以及当前未验证的边界。

标签器只做一件事：把每个真实分轨的短时声学能量映射到协议中心时刻并给出 `[0, 1]` 活动概率。**不做**身份跟踪、重复起音检测，也不根据主观假设给未知歌曲自动造真值。

## 1. 数据流

```text
manifest.json ──> sample_rate / duration / track_start_seconds
sources.json  ──> source_ids 列顺序
stems/<id>.wav ─┬─> 短时 RMS 包络（frame_seconds/frame_hop_seconds）
                │      └─> dBFS、噪声底、平滑
                ├─> 在 center_times 插值
                ├─> 绝对/相对阈值 + 滞回 + release 尾音
                └─> 概率（soft / binary）
                       └─> activity.json + activity.npz（协议校验）
                       └─> activity_summary.json（可读摘要）
                       └─> manifest.json stage=labeled（默认更新）
```

MIDI/控制事件（`controls.json`）**不是**输入。note-on 不必等于可听起音，note-off 不必等于尾音结束；活动标签完全来自渲染音频。

## 2. 输入

| 输入 | 要求 |
|---|---|
| `manifest.json` | `sample_id`、`sample_rate`、`duration_seconds`、`track_start_seconds`、`stem_paths`；`rendered` 或已 `labeled` 阶段均可 |
| `sources.json` | 决定 `source_ids` 列顺序；必须与 `manifest.stem_paths` 键顺序一致 |
| 分轨 WAV | 未压缩 PCM 8/16/24/32-bit 或 IEEE float 32/64-bit（含 `WAVE_FORMAT_EXTENSIBLE`）；压缩格式显式拒绝 |
| 采样率 | 必须等于 `manifest.sample_rate`；标签器不做重采样 |
| 时间轴 | `audio[0]` 位于 `manifest.track_start_seconds`（原曲绝对秒）；`center_times` 也是绝对秒，不重置为片段相对时间 |

多声道分轨按“逐声道帧功率求平均”处理，反相信道不会互相抵消。帧窗越界（片段首尾）与音频缺失处按零填充；`valid` 掩码描述的是**模型窗口**（默认 2 s）是否完整落在渲染音频内，而不是能量窗。

## 3. 能量包络与中心采样

每个分轨先在短帧网格上计算 RMS 能量：

- 帧中心网格：`origin + i * frame_hop_seconds`，`origin = track_start_seconds`；
- 帧窗：`frame_seconds`，以帧中心为中心，越界补零；
- 功率：逐声道 `mean(x²)` 后对各声道取平均；
- dB：`10 * log10(power / full_scale²)`，下限 `floor_db`。

随后把 dB 包络线性插值到每个 `center_times[t]`，并对中心网格运行阈值/滞回状态机。能量窗默认 20 ms、步长 10 ms，远小于 2 s 模型窗，因此**中心标签不会退化成“整个窗口的 OR”**：例如 1.0 s 处的 50 ms 脉冲不会让 1.5 s（其 2 s 窗仍覆盖该脉冲）的中心变为活动。

若某个中心没有任何分析帧覆盖（自定义中心网格且帧步长不对齐时可能发生），该中心按数值下限处理为不活动，而不是沿用最近一帧的数值。

## 4. 阈值、滞回与尾音

1. 噪声底：默认取帧 dB 的 10 百分位（只用完整落在音频内的帧）；也可用 `noise_floor_db` 显式覆盖。
2. 激活阈值：`on_db = max(abs_on_db, noise_floor_db + rel_on_db)`。
3. 关闭阈值：`off_db = on_db - hysteresis_db`。
4. 滞回状态机（在中心网格上）：
   - `db >= on_db` 立即激活；
   - 激活状态下 `db <= off_db` 开始 release 计时；持续 `release_seconds` 后才关闭；
   - 回弹到 `off_db` 以上会取消 release 计时，无需再次达到 `on_db`。
5. 平滑：可选的对称移动平均（`smoothing_seconds`，默认关闭），不产生时间偏移。
6. 概率：
   - `soft`：`p = state * clip((db - off_db) / (on_db - off_db), 0, 1)`，阈值带内为中间值；
   - `binary`：`p = state`，即 0/1 的滞回决策（适合直接当监督标签）。

`release_seconds` 让低于关闭阈值的短暂凹陷/尾音保持活动：note-off 后只要声学能量仍高于 `off_db`，标签继续活动并按能量衰减；跨过 `off_db` 后的短暂释放由 `release_seconds` 桥接。所有这些参数都来自配置并随输出保存，不存在写死的阈值。

## 5. 配置

`LabelConfig`（`src/aat/labels/config.py`）是唯一参数来源；`configs/experiment.toml` 的 `window_seconds`/`hop_seconds` 与默认值一致，但标签器不读取该文件。

| 字段 | 默认 | 含义 |
|---|---|---|
| `labeler_version` | `activity-labeler-0.1.0` | 算法/默认参数集版本 |
| `hop_seconds` | 0.02 | 未提供中心时刻时的默认步长 |
| `model_window_seconds` | 2.0 | 计算 `valid` 掩码的模型窗宽 |
| `frame_seconds` | 0.02 | 短时能量窗宽 |
| `frame_hop_seconds` | 0.01 | 能量包络步长（必须 ≤ `frame_seconds`） |
| `full_scale` | 1.0 | dBFS 参考满量程 |
| `floor_db` | -120.0 | dB 数值下限 |
| `noise_floor_percentile` | 10.0 | 噪声底百分位 |
| `noise_floor_db` | `null` | 显式覆盖噪声底 |
| `abs_on_db` | -45.0 | 绝对激活阈值（dBFS） |
| `rel_on_db` | 6.0 | 高于噪声底的激活余量（dB） |
| `hysteresis_db` | 3.0 | 滞回带宽；0 退化为硬阈值 |
| `release_seconds` | 0.05 | 低于关闭阈值后的尾音保持时长 |
| `smoothing_seconds` | 0.0 | dB 包络平滑窗，0 关闭 |
| `probability_mode` | `soft` | `soft` 或 `binary` |
| `min_snr_db` | 10.0 | 低于该峰-噪比的非静音分轨标记为低信噪比/歧义 |
| `summary_active_probability` | 0.5 | 摘要中判定“活动中心”的概率门限 |

CLI 的 `--config` 接受**部分** JSON 覆盖（未知字段报错），覆盖结果会合并进最终配置：

```json
{"probability_mode": "binary", "abs_on_db": -50.0, "release_seconds": 0.1}
```

## 6. 输出

### 6.1 `activity.json` + `activity.npz`

由 `aat.contracts.ActivityData` 写出并校验（协议 0.1.0）：

- `center_times` float64 严格递增，`origin = original_track_start`；
- `activity` float32，形状 `[T, S]`，列顺序 = `sources.json`；
- `valid` bool，`True` 表示 `model_window_seconds` 完整落在渲染音频内；
- `label_params` 写入：`labeler_version`、`config_sha256`、完整 `config`、`origin_seconds`、`duration_seconds`、`n_centers`；
- 静音分轨保留整列 0.0，不写 NaN、不删行。

### 6.2 `activity_summary.json`（标签器本地可读摘要，非共享协议）

| 字段 | 含义 |
|---|---|
| `kind` = `activity_summary` | 文档类型 |
| `labeler_version` / `config_sha256` / `config` | 可复现信息（与 `label_params` 一致） |
| `origin_seconds` / `duration_seconds` / `hop_seconds` / `model_window_seconds` | 时间轴与窗参数 |
| `n_centers` / `n_valid_centers` | 中心数与有效中心数 |
| `sources[]` | 逐来源统计：`noise_floor_db`、`peak_db`、`on_db`、`off_db`、`active_fraction`、`active_centers`、`n_segments`、`max_activity`、`mean_activity`、`mean_db`、`ambiguous_fraction`、`silent`、`low_snr` |
| `warnings[]` | 静音、低信噪比/歧义等文字说明 |

统计默认只覆盖 `valid=True` 的中心；当没有任何有效中心（时长小于模型窗）时回退到全部中心，并在 `warnings` 中说明。

`silent` = 有效中心里没有任何概率 ≥ `summary_active_probability`。`low_snr` = 非静音且 `peak_db - noise_floor_db < min_snr_db`（诊断，不影响标签数值）。`ambiguous_fraction` = dB 落在 `(off_db, on_db)` 滞回带内的中心比例。**这些标记都不等于主观可听性判断**：阈值只是一个可复现的声学代理。

### 6.3 `manifest.json`

默认把 manifest 重写为 `labeled` 阶段：

- 成对写入 `activity_metadata_path` / `activity_arrays_path`；
- 两者的 SHA-256 都写入 `content_sha256`（其余哈希原样保留）；
- `versions.labeler` 记录 `labeler_version`；
- `--no-manifest-update` 可跳过，此时 manifest 保持 `rendered`，但活动文件仍然写出。

## 7. 可复现

同一 `--config` 与同一音频必然产生相同的 `config_sha256` 与标签数组。重建步骤：

1. 读取 `activity.json` 的 `label_params.config`；
2. `LabelConfig.from_dict(config)`；
3. 校验 `config_sha256 == LabelConfig.from_dict(config).fingerprint()`；
4. 用同一音频和 `center_times` 重新运行 `aat.labels.label_stems`。

`labeler_version` 覆盖算法行为，`config_sha256` 覆盖参数，两者都在 `label_params` 与摘要中。

## 8. CLI

```sh
uv sync --locked
uv run python scripts/label_sample.py path/to/sample_dir
uv run python scripts/label_sample.py path/to/sample_dir --config overrides.json
uv run python scripts/label_sample.py path/to/sample_dir --no-manifest-update
```

退出码：`0` 成功；`2` 表示输入/配置错误（缺 manifest、采样率不符、未知配置字段等），错误信息写到 stderr。脚本只读该样本目录内的音频并把输出写回同一目录，不访问网络或其他路径。

## 9. 已验证行为（合成测试）

`tests/labels/` 全部使用程序合成的小波形，不读取任何真实音频：

- 脉冲/慢 attack/延音/静音/重复音的中心定位；
- 中心标签不等于 2 s 窗 OR（1.0 s 脉冲不影响 1.5 s 中心）；
- 音频平移后标签等量平移；`track_start_seconds` 偏移后绝对时间轴与标签不变；
- 16 k/22.05 k/44.1 k 采样率下标签无系统偏移（质心差 < 5 ms）；
- note-off 后尾音仍活动，`release_seconds` 可延长；note-on 不强制 onset；
- 滞回可桥接低于关闭阈值的短暂凹陷；
- 静音、低信噪比、阈值带歧义在摘要中显式标注；
- `ActivityData` 保存/加载校验、`valid` 边界语义、配置指纹可重建；
- WAV 读取（PCM 8/16/24/32、float32/64、extensible、反相立体声、拒绝压缩格式与非有限样本）；
- CLI 端到端：生成 `activity.*` 与摘要、把 manifest 更新为 `labeled` 并写入两个哈希；
  `controls.json` 中与音频不符的 note-on/note-off 不会改变标签。

## 10. 未验证与边界

- 尚未在真实 DawDreamer/Surge 分轨上校准阈值；默认值是 M1 分析起点，不是听觉真值。
- 不做重采样；分轨采样率必须与 manifest 一致。
- 不做音高/乐器识别、身份跟踪或重复起音检测；重复音只表示为多个离散活动段。
- `valid` 描述模型窗，不描述能量窗；模型窗小于能量窗或自定义中心网格时，边缘行为以本文档和测试为准。
- 立体声只在“各声道功率平均”意义下处理；未评估强空间化/相位复杂的分轨。
- 摘要统计默认只覆盖有效中心；极短片段可能没有有效中心。
