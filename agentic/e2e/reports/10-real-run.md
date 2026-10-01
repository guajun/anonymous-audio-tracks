# 10 — 真实端到端运行证据（REAL，issue #33）

> **真实性声明**：本文件记录的是一次**真实**运行（真实 Gemini API 调用 + 真实桥接音频字节 +
> 真实 SAM GPU 分离 + 真实 DSP 脚本）。其中没有任何预写输出、mock 分离或人工润色。
> 全文脱敏（`<WS>`=workspace、`<REPO>`=仓库 checkout、`<SAM_ROOT>`=SAM checkout、
> `<AUDIO>`=真实源音乐）；完整事件流/会话只在本地 `<WS>/local/e2e/<RUN_ID>/`（ignored）。

## 0. 一览

| 项 | 值 |
|---|---|
| run id | `e2e-real-20261001-02`（`outputs/e2e/LATEST.txt` 指向它） |
| 结果 | **pass**（runner exit 0；数据校验 15 项全绿 + 执行证据门 9 项全绿） |
| R1 修订 | 读取器/抽查统计 bug 已修并**离线重算**；旧 “isolation≈0.95–1” 数字作废，见 [`reports/20-spotcheck.md`](20-spotcheck.md) §0（原 result/events/stems 未改动） |
| 输入 | `e2e-clip-001.wav`（16.0 s / 44.1 kHz / 2ch，sha256 `42f968e2…c7f0`，clip 时轴 t=0=源 4.0 s） |
| 模型 | 全部 assistant 消息 `model=gemini-3.8-flash`（**未传 `--model`**，来自项目设置 + `pi --approve` project trust） |
| 音频通路 | `bridge`：`audio_attach {"path":"e2e-clip-001.wav"}` ×2（第 2 次用于抽查试听）；**非** Pi 原生（unsupported，#30） |
| 真实 SAM 分离 | **3/3 预算**：`drums` / `bass` / `synthesizer`（各先 `--dry-run` 再真跑；`cuda`+`bfloat16`） |
| Agent DSP | 自写 `outputs/e2e/<RUN_ID>/scripts/detect_events.py`（三频带能量通量），实际运行 |
| 结果 | 110 个 onset（全部 `source="dsp"` `method="energy-flux"`）；tempo 按 unknown 规则写 null |
| 用量 | 59 轮 / 58 工具调用；input 312,380、output 33,021、cacheRead 2,543,902、cacheWrite 0 = totalTokens 2,889,303（逐步累计可对账）；cost ≈ **0.549**（Pi 记账近似，非账单真值） |
| 耗时 | wall **354.9 s**（SAM 上游推理 21.6 / 22.2 / 22.2 s）；GPU RTX 3070 Laptop 8 GB |

## 1. 为什么这是“Agent 自主”，不是手工 pipeline（trace 证据）

`<WS>/local/e2e/<RUN_ID>/events.jsonl` 的工具调用链（58 次；bash 36 / read 16 / audio_attach 2 /
write 3 / edit 1），关键顺序（脱敏摘要）：

| # | 工具 | 调用（摘要） | 说明 |
|---|---|---|---|
| 1–5 | bash/read | `ls local/ audio/inputs/ .pi/skills/sam-audio/`、读 `local/config.json`、`audio/inputs/manifest.json`、`SKILL.md` | 自主勘查 workspace 与 skill |
| 6–10 | read/bash | 读 `agentic/schema/README.md`（两次）、`harness/dsp_onset.py`、跑 `validate.py --help` + 校验 schema 自带 fixture | 自主学习冻结协议与校验器 |
| 11–12 | bash | `audio_toolbox.py sam check-environment --sam-root <SAM_ROOT> --python <SAM_ROOT>/.venv/…`；`python -c "import torch…"` | 真实环境预检（CUDA 可用） |
| 13 | bash | 自算 clip sha256/时长（`hashlib`/`wave`） | result 的 `audio` 字段来源 |
| 14 | **audio_attach** | `{"path":"e2e-clip-001.wav"}` → `{attached, mime:"audio/wav", bytes:2822444, sha256:42f968e2…}` | **bridge 听音**（下一次请求注入 inlineData） |
| 15–20 | read | `spotcheck.py`、`validate_result.py` 等 helper | 自主决定如何满足校验/抽查要求 |
| 21–23 | bash/write | `mkdir -p outputs/e2e/<RUN_ID>/{stems,scripts}`；写 `notes.md` 假设 | 声音层假设：drums / bass / synthesizer（见 §3） |
| 24–29 | bash | **6 次 `audio_toolbox.py sam separate`** = 3 × (`--dry-run` + 真跑) | **真实 SAM GPU 分离**（预算内） |
| 30–33 | bash/write/edit | 自写并运行 `scripts/detect_events.py` → onset/tempo JSON；算 stem hash | **自主 DSP**（非预写脚本；参数见 §4） |
| 34–37 | bash/write | 跑 `validate.py`（退出 0）+ `validate_result.py --level real`（全绿）→ 写 `result.json` | 冻结校验一次通过 |
| 38–… | **audio_attach** + bash | 再次挂载 clip 做抽查试听；跑 overlay/统计；写 `notes.md` 抽查表 | **试听抽查**（见 reports/20） |

> harness **只**做：造 clip、发 prompt、限时启动 pi、收 trace、跑校验、写 manifest。harness 没有
> 写过 `result.json`/stem/onset 的任何内容（`run_e2e.py` 里没有生成逻辑，可审计）。

## 2. SAM 真实分离（GPU）

| stem | description | device/dtype | 上游推理 | 产物（`outputs/e2e/<RUN_ID>/stems/<stem>/`） | target sha256（前 16） |
|---|---|---|---|---|---|
| drums | `drums` | cuda / bfloat16 | 22.23 s | target.wav、residual.wav、request.json、report.json | `392e02bc722366c0…` |
| bass | `bass` | cuda / bfloat16 | 21.63 s | 同上 | `c64ebecba7489490…` |
| synthesizer | `synthesizer` | cuda / bfloat16 | 22.22 s | 同上 | `38bb97853310ec99…` |

采样率 48 000 Hz（SAM 输出）；分离是“目标/残差”两路**假设**，不是真值（串音见 reports/20）。

## 3. Agent 的声音层/乐器假设（听自 bridge 注入的音频）

| 层 | label | 假设依据（Agent 记录于 notes.md） | source/confidence |
|---|---|---|---|
| 1 | drums | 清晰 kick/snare/clap/hi-hat 律动，瞬态贯穿全段 | sam / 0.85 |
| 2 | bass | 连贯合成低音（synth bass / 808），随和弦与底鼓铺陈 | sam / 0.75 |
| 3 | synthesizer | 中高频电子琴/电钢与主旋律；7–9 s 有微弱人声点缀 | sam / 0.70 |

（乐器标签是假设；真实音乐无真值，不作准确率主张。）

## 4. Agent 的 DSP 参数与结果（`scripts/detect_events.py`，实际运行）

| 来源 | 频带 | frame/hop | threshold | min_gap | 事件数 |
|---|---|---|---|---|---|
| inst-drums | all | 1024/256 | 1.2 | 0.10 s | 36 |
| inst-bass | low | 2048/256 | 1.0 | 0.15 s | 33 |
| inst-synthesizer | mid | 1024/256 | 1.1 | 0.12 s | 41 |

- 事件：110 个，`id` 全文档唯一（`inst-<stem>-ev-XXX`），按 `(onset_seconds, id)` 升序，秒（clip 时轴），
  **每个事件显式 `source="dsp"` + `method="energy-flux"`**（没有继承乐器标签的 sam/llm 来源）。
- tempo：IOI 直方图主峰 123 ms≈121.95 BPM 但置信度 0.27、drums 轴 171 ms≈175.44 BPM 置信度 0.19
  → 依协议写 `{"bpm": null, "source": "unknown", "confidence": 0}`，**不猜、不量化**。
- `pitch`/`duration_seconds`：无充分证据，**未填**（协议“没证据不填”）。

## 5. 校验（两级，均通过）：**数据校验** + **执行证据门**

> 定位声明（R1）：文件/schema/一致性/隐私检查只是**数据校验**，不证明“真实执行”；
> 真实执行由**执行证据门**（观测到的模型/附件/成功分离/预算/stage）证明，两者都必须过。

**① 数据校验**：冻结协议校验（#32）`python <REPO>/agentic/schema/validate.py <WS>/outputs/e2e/<RUN_ID>/result.json`
→ `OK (engine=stdlib)`，退出码 0；+ `harness/validate_result.py --level real`（15/15 pass）：

| 检查 | 结果 | 证据 |
|---|---|---|
| schema | pass | 冻结 validator，0 问题 |
| audio_filename_safe | pass | `e2e-clip-001.wav` 安全相对路径 |
| clip_meta / clip_sha256 / clip_sample_rate / clip_duration_seconds | pass | result = clip 记录 = `42f968e2…`；44 100 Hz；16.0 s |
| audio_file | pass | **现算文件字节** sha256 一致（actual hash） |
| timeline | pass | 110 事件全在 `[0, 16.0]` 秒（clip 时轴一致） |
| event_provenance | pass | 110/110 显式 `source`+`method`；110 个 `source="dsp"` |
| provenance_pins | pass | 含 `gemini-3.8-flash`、`audio-toolbox.sam pin dfbc40a9541f (sam c603de8794cc)` |
| provenance_chain | pass | steps.source = {bridge, llm, sam, dsp} |
| no_native_audio_claim | pass | 音频输入标 bridge，未冒充 native |
| no_mock_or_native_claims | pass | 全文档（tempo/instruments/events/steps）无 mock/native 冒称 |
| privacy | pass | 无绝对路径/key 形态/长 base64 |
| stem_files | pass | 3 个 stem 存在且 sha256 一致（相对 run 目录解析） |

**② 执行证据门**（`run_e2e.py` execution gate `agentic-e2e-execution-gate/v2`，观测证据而非文档自述，9/9 pass）：

| 检查 | 结果 | 观测证据（结构化载荷/产物关联，不只信工具 isError） |
|---|---|---|
| observed_model | pass | 事件流中 assistant `model=['gemini-3.8-flash']` `provider=['google']`（精确匹配；缺 provider 即 fail） |
| no_terminal_provider_failure | pass | 无 provider error/aborted 终态 |
| attachment_observed | pass | `audio_attach` **结构化成功** 2/2（载荷 `attached=true`、`mime=audio/wav`、`bytes=2,822,444>0`、工具未报错） |
| sam_separation_observed | pass | 真·SAM 分离**成功 3** 次（wrapper 载荷 `schema=audio-toolbox.sam/v1`、`action=sam.separate`、`ok=true`、`exit_code=0`）；dry-run 3、失败 0、无结构化载荷 0 **分开计数** |
| sam_artifacts_correlated | pass | 3 个成功调用逐一关联到 `stems/<name>/report.json`（可解析、含 outputs）且 target/residual 非空（不是任意 report.json） |
| budget_compliance | pass | 真实分离**尝试** 3（含失败/unknown）≤ 预算 3 |
| required_stages_present_nonfailed | pass | model_run/result/validate/spotcheck **均存在**且 ok（空 stages 即 fail） |
| execution_evidence_present | pass | `events.jsonl` 存在，58 次工具调用 |
| original_runner_success | pass | 原 runner exit=0（`--verify` 不会把失败 run 洗成 pass） |

**result 摘要（纯 JSON，脱敏）**

```json
{
  "schema_version": "agentic-audio-tracks/v1",
  "audio": {"filename": "e2e-clip-001.wav", "sha256": "42f968e2…c7f0", "duration_seconds": 16.0, "sample_rate": 44100},
  "tempo": {"bpm": null, "source": "unknown", "confidence": 0},
  "instruments": [
    {"id": "inst-drums",        "label": "drums",        "source": "sam", "confidence": 0.85, "events": 36},
    {"id": "inst-bass",         "label": "bass",         "source": "sam", "confidence": 0.75, "events": 33},
    {"id": "inst-synthesizer",  "label": "synthesizer",  "source": "sam", "confidence": 0.70, "events": 41}
  ],
  "provenance": {"steps": ["bridge:audio_attach", "llm:gemini-3.8-flash listening", "sam:audio-toolbox.sam pin dfbc40a9541f (sam c603de8794cc)", "dsp:scripts/detect_events.py energy-flux"]},
  "limitations": ["真实音乐无精确真值…", "SAM 目标/残差存在串音…", "能量通量可能漏弱瞬态…", "IOI 置信度低→tempo unknown"]
}
```

（示例为**摘录**；完整文档在本地 `outputs/e2e/<RUN_ID>/result.json`，sha256 `d789cab…8311`。）

## 6. run manifest（本地，schema `agentic-e2e-run-manifest/v2`）

`<WS>/outputs/e2e/<RUN_ID>/run-manifest.json` 含：clip 源/clip hash 与 offset、模型/工具/bridge pin、
参数、GPU 与 run 前显存、wall 与 SAM 耗时、usage（含 cacheRead/cacheWrite 与对账说明）、
tool trace 摘要（真·分离 3 成功 / 0 失败 / 3 dry-run、2 次 audio_attach）、逐 stage 状态、
数据校验与执行门摘要、限制、脱敏复现命令。**可归属性（R1）**：`code` 块记录 git revision
（run 时未记录 → 保持 `unknown`，**不伪造**；verification 时 revision 分开标注）+ harness 各组件
sha256 + **Agent DSP 脚本 sha256 `f8b97e45…e298a`**（`scripts/detect_events.py`，14,582 bytes）。
**时间戳**：`created_at` 与 `timings` 保持原 run 值；离线复核追加 `verification_runs[]`
（`kind=offline-reverify`）而不改写历史。失败 stage 语义：本 run 全部 `ok`；若失败会写
`fail`/`blocked` 并保留事件流。

## 7. 产物位置（本地，均 ignored）

```text
<WS>/outputs/e2e/e2e-real-20261001-02/
  result.json            Agent 产出（协议文档）
  notes.md               Agent 过程/假设/参数/抽查表
  scripts/detect_events.py  Agent 自写 DSP（实际运行）
  stems/{drums,bass,synthesizer}/  target.wav + residual.wav + request.json + report.json
  validation.json        harness 校验报告（数据校验 15 项 + 执行门 9 项）
  spotcheck/             overlay.wav + spotcheck.json + overlay.json（harness，R1 修复后重算）
  overlay.wav, spotcheck-stats.json  Agent 自产的同名辅助产物（原样保留）
  stem-map.json          唯一 stem 文件名映射（R1：hash/相对路径/采样率；原件不动）
  stems-unique/          唯一文件名副本（stem-<instrument-id>-target.wav）
  run-manifest.json      run manifest（v2）
<WS>/local/e2e/e2e-real-20261001-02/
  prompt.txt events.jsonl stderr.txt trace-summary.json   本地证据（不入 git）
```

## 8. 一条命令复现（Junior）

```sh
python agentic/e2e/harness/run_e2e.py --workspace "<WS>" --run-id <RUN_ID> --clip-name e2e-clip-001.wav --timeout 2400
```

成功判据：退出码 0 + `e2e pass run_id=…` + `validation.json` 全 pass；失败时看打印的 stage
（`preflight.*` / `model_run` / `result` / `validate`）与 `local/e2e/<RUN_ID>/events.jsonl`。
