# 渲染（issue #3：DawDreamer 冒烟样例与 Surge 探针）

本文说明 `src/aat/render/`、`scripts/render_sample.py` 和 `configs/render/` 的用法与证据语义。范围是**生成一首音乐内部的 2–4 个可独立控制声音层，并导出混音与分轨**；不做大规模数据生成、训练、活动阈值定义，也不购买或下载任何商业插件。

## 1. 快速开始

```sh
uv sync --locked --extra render
uv run --no-sync python scripts/render_sample.py render \
    --config configs/render/smoke.toml \
    --out runs/render/smoke-0001
```

`smoke.toml` 生成 **20 秒**（+3 秒尾音）的四来源片段：kick、snare、带滤波的 plucked bass 和 pad。所有声音素材由 `src/aat/render/synth.py` 用显式 seed 在本地合成，不使用任何外部素材。同 seed 重复运行输出字节级一致（集成测试会校验 `mix.wav` 与全部分轨）。

输出目录（`runs/` 已被 `.gitignore` 忽略，音频不入库）：

```text
runs/render/smoke-0001/
  mix.wav               # 线性求和后的混音（PCM16）
  stems/<id>.wav        # 效果后分轨（PCM16），求和 ≈ mix
  dry/<id>.wav          # 效果前干声引用（PCM16）
  sources.json          # 协议 v0.1.0：生成期来源身份（source_id/renderer/seed/引用）
  controls.json         # 协议 v0.1.0：note_on/note_off 控制事件（原曲绝对秒）
  manifest.json         # 协议 v0.1.0：stage = "rendered"，含采样率/时长/尾音/摘要
  render_report.json    # 渲染器私有证据：延迟、起音、尾音、求和残差、削波统计、配置快照
  surge_probe.json      # 仅当显式传入 Surge 插件路径时生成
```

## 2. 配置参考（`configs/render/*.toml`）

配置由 Python 标准库 `tomllib` 解析，映射为冻结 dataclass，未知键/越界值/非有限值一律报错（不会静默改变渲染）。**提交的配置不包含任何本机路径。**

`[render]` 必填字段：

| 字段 | 说明 |
|---|---|
| `sample_id` / `composition` | 样本 ID 与作品 ID（`manifest.groups`） |
| `seed` | 显式渲染种子，写入 manifest 并派生各来源默认 sample seed |
| `sample_rate` / `block_size` | DawDreamer 引擎参数（8000–384000 Hz / 1–8192） |
| `bpm` | 引擎 BPM（当前模式全部用秒制，字段仅作为配置记录） |
| `duration_seconds` | 音乐时长（秒）；每个导出配置必须 > 0，冒烟样例为 15–30 秒 |
| `tail_seconds` | 额外渲染的尾音（秒）；必须覆盖最后一个 note-off + release |
| `track_start_seconds` | 片段在原曲时间轴上的起点（默认 0） |

`[surge]`（可选）：`probe_duration_seconds`、`note`、`velocity`。**不要提交 `plugin_path`**；插件位置只通过 `--surge-plugin-path` 或本地未跟踪的覆盖文件传入。

`[[sources]]`（2–8 个，冒烟样例 4 个）：

- `id`：不带乐器语义的来源 ID，也是分轨文件名与 `sources.json` 的键。
- `sample`：`type ∈ {kick, snare, hat, pluck, pad}`、可覆盖 `seed`、`params`（未知参数报错）。
- `gain`：来源级增益（0–4），在效果链末端施加，因此干声/分轨/混音的层次关系固定。
- `center_note`：sampler 的 pitch 基准（note 等于该值时按原速播放）。
- `amp`：sampler ADSR（`attack_ms`/`decay_ms`/`sustain`/`release_ms`）。note-off 后的 release 是尾音的一部分。
- `effects`：有序效果链，支持 `{ type = "filter", mode = "low"|"high", frequency_hz, q, gain }` 与 `{ type = "gain", gain }`；其他类型明确报错。
- `preset_ref` / `sample_ref`：写入 `sources.json` 的音色/素材引用（默认 `aat/...`，不是本机路径）。
- `pattern`：`step_seconds` + 可选 `loop_steps`（循环重复直到 `duration_seconds`）+ `notes`（`step`/`note`/`velocity`/`length_steps`）。显式 MIDI 全部记录进 `controls.json` 和报告中的配置快照。

## 3. 测量证据与容差

`render_report.json` 中的证据只引用实际导出的 PCM：

- **stem 求和容差**：`mix.wav` 与全部 `stems/*.wav` 读回为 int16 后逐样本求和。每个样本最多贡献半 LSB 的舍入误差，因此容差取
  `tolerance_lsb = 0.5 * (num_stems + 1) + 1`（`analysis.stem_sum_tolerance_lsb`），报告同时给出 `max_abs_error_lsb` 与对应的浮点值。超过容差会抛 `StemSumError` 而不是放行。
- **原曲绝对时间与渲染局部时间**：`controls.json`、`sources.json`、报告中的事件时刻始终是原曲绝对秒；DawDreamer 引擎时间轴从渲染缓冲区起点开始，因此 `graph.py` 在送入 MIDI 时减去 `track_start_seconds`，`pipeline.required_tail_seconds` 也以局部秒计算并与实际渲染长度比较。报告同时给出 `tail.required_seconds`（局部）与 `tail.required_absolute_seconds`（绝对）。只改变 `track_start_seconds` 元数据时，PCM 内容、延迟、尾音 margin 完全不变，控制事件与 onset 绝对时刻整体平移（集成测试覆盖 0 / 0.1 / 12 s）。
- **控制触发→声学起音延迟**：对每个来源，取该来源第一个 `note_on` 绝对时间与分轨首个超过阈值样本的时间差，报告 `onset_offset_seconds`。这是**声学证据**，包含音源 attack，不写入 `manifest.render_latency_seconds`。
- **插件/引擎延迟**：对每个来源，用单样本脉冲跑同一条效果+增益链，报告 `effect_latency_samples/seconds`。`manifest.render_latency_seconds` 只取这些独立测量值的最大值，并在 `render_latency` 证据块中说明scope与覆盖情况；如有来源无法测得，则省略该可选字段并给出 `omitted_reason`。内置 sampler/filter/gain 链为 0 samples。
- **尾音**：`tail.required_seconds` = 局部坐标系下最后一个 note 的 `start + duration + release`；必须小于渲染总长并保留 guard，且最后 100 ms 的 RMS 比例低于 `TAIL_DECAY_RATIO_LIMIT = 0.05`。否则报 `RenderValidationError` 并提示增大 `tail_seconds`，不会静默截断。
- **异常检测**：空音频、NaN/Inf、混音或分轨削波（`|x| >= 1.0`）在写出 JSON 前报错；报告固定包含 `clipped_samples`、`non_finite_samples`、`silent`、`peak_dbfs`、`rms_dbfs`。
- **配置可追溯**：报告含 `config_sha256`（规范化配置摘要）、`config_file_sha256`（配置文件摘要，仅 basename，不含本机路径）、`seed`、`git_commit`（可用时）与完整解析后配置。

## 4. Surge XT 探针

```sh
uv run --no-sync python scripts/render_sample.py probe-surge \
    --plugin-path "<path to Surge XT.vst3>" \
    --out runs/render/surge-probe
```

或在渲染时附带探针结果：

```sh
uv run --no-sync python scripts/render_sample.py render \
    --config configs/render/surge_probe.toml \
    --out runs/render/surge-probe \
    --surge-plugin-path "<path to Surge XT.vst3>"
```

探针通过 `DawDreamer.PluginProcessor` 载入插件、发送一个 MIDI note、渲染并检查输出。结果写入 `surge_probe.json`：

- `status = "passed"` 仅当输出非空、有限且峰值超过静音下限（约 -80 dBFS），同时记录插件通道数与 `latency_samples`。
- `status = "failed"` 以稳定 `reason` 分类：`not_configured`（未提供路径）、`path_not_found`、`plugin_load_failed`、`graph_rejected`、`midi_rejected`、`render_failed`、`empty_output`、`silent_output`、`non_finite_output`。命令行退出码为 4。
- **绝不把自生成 sampler 冒烟样例当作 Surge 验证通过**；没有插件时探针明确失败，Surge 保持"未验证"。

## 5. 限制与未验证项

- 本机（Windows）未安装 Surge XT，因此 Surge 集成探针**未在本机验证通过**；已验证的是缺插件/错路径时的明确失败路径，以及自生成 sampler 链路的真实 DawDreamer 渲染。
- `DawDreamer 0.9.0` 的 `ReverbProcessor` 与 `DelayProcessor` 在本机实测参数不生效或输出异常（reverb 静音/参数赋值后数值爆炸），因此效果链只启用可验证的 `filter`/`gain`。其他机器/版本上使用前需要重新做脉冲/尾音验证。
- 主链路假设是"每来源独立处理 + 线性求和"；一旦加入共享非线性母带处理，分轨求和一致性不再强制成立，需要新的容差定义并在报告中标记。
- 采样合成是确定性信号生成，不代表真实乐器音色多样性；音色/素材多样性扩大留给数据集 issue。

## 6. 测试命令

```sh
# 基础环境：协议 + 渲染单元测试，集成渲染自动跳过
uv sync --locked
uv run --no-sync pytest -q -p no:cacheprovider

# 完整渲染：真实 DawDreamer 集成（含 20 秒冒烟样例与可复现性）
uv sync --locked --extra render
uv run --no-sync pytest -q -p no:cacheprovider tests
```
