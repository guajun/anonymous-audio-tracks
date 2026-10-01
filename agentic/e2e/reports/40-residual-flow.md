# 40 — 残差逐层剥离对照 run（REAL，issue #42）

> **真实性声明**：本文件记录一次**真实**运行（真实 Gemini API 调用 + 真实桥接音频字节 + 真实
> SAM GPU 分离 + 真实 DSP 脚本）。没有预写输出、手工预固定分轨后 LLM 润色、mock 分离或人工
> 修改。全文脱敏（`<WS>`=workspace、`<REPO>`=仓库 checkout、`<SAM_ROOT>`=SAM checkout）；
> 完整事件流/会话只在本地 `<WS>/local/e2e/<RUN_ID>/`（ignored）。
> **角色**：实施/开发 worker = `openrouter/xiaomi/mimo-v2.6-pro`（本目录代码与文档）；处理音乐的
> Agent = `google/gemini-3.8-flash`（固定，经明确非 native 的 Pi `audio_attach` bridge）。

## 0. 一览

| 项 | 值 |
|---|---|
| 新 run | `e2e-real-20261001-03-rflow`（`--profile residual`，flow `residual-peeling-v1`） |
| 结果 | **pass**（一次真实 run，未复跑）：数据校验 15 项 + 执行证据门 12 项 + 链核对 16 项全绿 |
| 基线 run | `e2e-real-20261001-02`（**只读复用**；run 前后 26 个文件 hash 全等，result.json `d789cabc…8311` 未变） |
| 输入 | 同一个 16 s clip `e2e-clip-001.wav`（`42f968e2…c7f0`，44.1 kHz/2ch），**同一零点**（t=0=clip 开头） |
| 剥离链 | drums（复用基线）→ R1 → bass（真实分离）→ R2 → synthesizer（真实分离）→ R3 → 停（budget 2/2） |
| 预算 | 复用 drums（不计）+ **2/2** 真实 SAM（单次 `--timeout 900`，实际 59.0 s / 37.6 s，cuda/bfloat16，48 kHz）；Gemini 整体 timeout 2400 s；校验修复 0 次 |
| 听音实证 | 4 次成功 `audio_attach`：原 clip + R1/R2/R3 监听副本（PCM16，1,536,044 bytes 各；顺序 clip→R1→SAM→R2→SAM→R3） |
| 结果 | 164 个 onset（drums 36 复用 / bass 48 / synthesizer 80；全部 `source="dsp"`）；tempo=null/unknown |
| 用量/成本 | 79 轮 / 78 工具调用（bash 49 / read 21 / audio_attach 4 / write 3 / edit 1）；input 413,532 + output 49,899 + cacheRead 4,364,712 + cacheWrite 0 = totalTokens 4,828,143；cost ≈ **0.825**（Pi 记账近似，非账单真值） |
| 耗时/GPU | wall **632.9 s**；RTX 3070 Laptop 8 GB（启动前 3,281 MiB 空闲；仅本任务使用，未动他人进程） |
| 代码归属 | 启动时 git `c3be512`（**clean**）+ 10 个 harness 组件 hash 快照（`agentic-e2e-run-manifest/v3`） |

## 1. 链证据：每级 input = 前一级 **raw residual**（真实 request + trace）

| stage | 层 | input（kind/sha 前 16） | target（sha 前 16） | residual（= 下一级 input，sha 前 16） | 证据 |
|---|---|---|---|---|---|
| 0（复用） | drums | `mix / 42f968e24dee4300`（原 clip） | `392e02bc722366c0` | `ea03d1cc74c01843` | 基线 request/report + 逐字节副本 hash 核对 |
| 1 | bass | `residual / ea03d1cc74c01843` **== stage0 residual** | `cad70e55beee425a` | `f33042a0ebea8160` | `stems/bass/request.json` 的实际 audio hash + trace `--audio` |
| 2 | synthesizer | `residual / f33042a0ebea8160` **== stage1 residual** | `dd3ea0b53d077e43` | `4ed0a3330434abb9`（R3，stop.residual_left） | `stems/synthesizer/request.json` + trace `--audio` |

- trace 中真实 SAM 调用（脱敏）：`--audio "outputs/e2e/<RUN_ID>/stems/drums-reused/residual.wav"`
  → 产 bass；`--audio "outputs/e2e/<RUN_ID>/stems/bass/residual.wav"` → 产 synthesizer。
  **没有任何一次分离的输入是原混音**（旧行为是三次都输入原 clip）。
- 链核对（`stage-chain-verify.json`，`agentic-e2e-stage-chain-verify/v1`）16 项全绿：
  `chain_input_prev_residual` / `no_mix_swap` / `request_proves_input` / `trace_input_chain` /
  `residual_listen_observed` / `listen_before_next_separation` / `stop_policy` /
  `residual_left_is_final_residual` / `sam_budget` / `reused_baseline_integrity` /
  `baseline_same_clip` / `reused_events_carried_into_result` / `sidecar_privacy` 等。任何“各自从
  原混音独立分离”“PCM16 监听代理当 SAM 输入”“听音晚于分离/同轮并行假装听后选择”
  “残余引用不是最终级 residual”都会 FAIL（离线回归有专门反例）。
- 时间轴：每个 target/residual 按**自身采样率**（48 000 Hz）核等长 16.000 s = clip 时长；无人为
  trim/shift/时间量化/换采样率（监听副本 PCM16 转换仅幅度量化，帧数/采样率/声道不变）。
  **口径澄清**：算法内部的采样率转换（clip 44.1 kHz 进 SAM、输出 48 kHz）是既有行为、允许；
  禁止的是人为改时间轴或让监听副本与 raw residual 采样率/帧数不一致（幅度量化 ≠ 时间量化）。
- 监听副本（`audio/inputs/<RUN_ID>-listen/`，本 run 派生目录，raw/proxy hash + 转换参数 + 等长
  全记录）：R1 `93367a950451290b` / R2 `4b3cb9b6351a5f54` / R3 `b833e1363a5e627a`
  （均 PCM16、48 kHz、768 000 帧；raw = 各级 residual `ea03d1cc…`/`f33042a0…`/`4ed0a333…`）。
  SAM 输入始终是 raw residual（float32），不是代理。

## 2. drums 如何保留（用户局部定性反馈的处理）

- 复用基线 drums **target `392e02bc…5640` / residual `ea03d1cc…aa82`**（新 run 内为逐字节副本
  `stems/drums-reused/`，hash 与基线原件逐一相同；基线文件 run 前后 hash 全等，未被改动）。
- 基线 **36 个事件原样进入新 result**（events canonical sha `34c40d4b7329…`；校验项
  `reused_events_carried_into_result` pass；新 `inst-drums` 36 点与基线 36 点 ±50 ms 全重合
  ——复用是构造使然，不是重新检测）。
- 反馈口径如实记录：**局部定性**（“drums 是准的”），据此只复用、不重跑；**不**推导 drums target
  完全干净，**不**把 R1 当真值（sidecar `reused.user_feedback` 与 result `limitations` 均写明）。

## 3. 旧独立分离 vs 新残差剥离（**描述性对照，不是准确率**）

| 层（标签=假设） | 旧 run 事件（各自从原混音分离） | 新 run 事件（残差剥离） | ±50 ms 重合（新→旧） | 备注 |
|---|---|---|---|---|
| drums | 36 | 36（复用） | 36/36（100%） | 用户已局部认可；复用保持一致 |
| bass | 33 | 48 | 33/48（69%） | 旧 33 点全部被覆盖，新增 15 点：可能更完整，也可能是误检/归属变化 |
| synthesizer | 41 | 80 | 30/80（38%） | 新层 = R2 上的再分离；点位差异大（含 8–9 s 人声切片附近事件），**不确定**哪个更准 |

- 同一次抽查（抽 5 点，能量上升 3/5；独立检测器差异：疑似漏检 0 / 疑似 17——只是同族检测器
  在 0.1 s 容差下的差异**迹象**，不是误检率）；已作废的“隔离比”不使用，跨轴能量比只作代理。
- **DSP 参数混杂（必须显式声明）**：两次 run 的 onset 检测参数**并不相同**，所以事件数变化
  **不是**“只换分离方式”的受控消融（not a controlled separation-only ablation）：

| 层 | 旧 run（原混音独立分离）DSP | 新 run（残差剥离）DSP | 阈值 |
|---|---|---|---|
| drums | all-band，frame 1024，hop 256，min-gap 0.10 | **复用基线同脚本/同参数的 36 事件**（不重检） | 1.2 |
| bass | low-band，frame **2048**，hop 256，min-gap **0.15** | low-band，frame **1024**，hop 256，min-gap **0.12** | 1.0（相同） |
| synthesizer | mid-band，frame 1024，hop 256，min-gap 0.12 | mid-band，frame 1024，hop 256，min-gap 0.12（**相同**） | 1.1（相同） |

（两版脚本的置信度归一化也略有不同：旧 strength/4.0 下限 0.1，新 bass strength/5.0、synth
  strength/6.0、下限 0.2；不影响 onset 位置，只影响 confidence 展示。）
- 差异的可能来源（**不承诺方向**）：①分离对象不同（新 target 来自 R1/R2，不是原混音）；②层归属
  随剥离顺序改变（“synthesizer”在 R2 语义下包含更多中高频内容）；③SAM target/residual 本身是
  估计，**逐层误差会累积**；④**DSP 参数本身变了**（上表：bass frame/min-gap 不同，synth 相同）。
  **不能**据此宣称残差剥离更准；原 164 事件不为“好看”而修改。

## 4. 分栏：实测 / 复用 / 推测

| 内容 | 分栏 | 依据 |
|---|---|---|
| 2 次真实 GPU 分离、4 次 bridge 挂载、监听副本等长、result 过冻结 schema | **实测**（工具/文件证据） | trace 结构化载荷、`request.json`/`report.json`、hash、校验 15+12+13 项 |
| 每级 input == 前级 raw residual 的 hash 链 | **实测** | sidecar × request × trace 三重核对全绿 |
| drums target/residual/36 事件 | **复用**（基线 + 用户局部定性反馈） | 基线 hash / 事件 canonical hash 核对 |
| 层标签（drums/bass/synthesizer）、各层假设内容 | **推测**（LLM 听感 + SAM text prompt） | 不是真值；标签正确性未验证 |
| onset 164 点、tempo=null | **推测/估计**（DSP 能量通量；IOI 无法定拍） | 不承诺准确率；不量化、不反推 BPM |
| R3 内容（微弱人声 FX “go!”、相位伪影、背景噪声） | **推测**（听感描述） | 无真值判定；stop=budget 时残余未剥离 |

## 5. 成本 / 限制 / 未验项

- **成本**：API 记账 ≈ 0.825（含 cacheRead 4.36M 的计费口径差异；非账单真值）；GPU 2 次分离
  59.0 s + 37.6 s；wall 632.9 s；全程只用本任务预算（1 次真实 run，未复跑、无参数扫）。
- **限制**：residual/target 均为模型估计，**逐层误差会累积**，不承诺比旧独立分离更准；单 clip、
  单次对照、剥离顺序只有一种（drums→bass→synthesizer，由 Agent 听 R1/R2 自主选择）；stop 原因
  `budget`（2/2），**R3 残余未剥离**（含微弱人声切片与相位伪影，已写入 limitations）；监听副本
  为 PCM16 幅度量化（仅供听音；SAM 仍用 raw residual）；tempo 仍为 null/unknown。
- **未验项**（留给人类验收/后续）：人工听辨与网页对照（#28 人类验收 pending）；bass/synthesizer
  标签正确性；R3 残余内容真值；不同剥离顺序（如先合成器后贝斯）的差异；多次 run 稳定性；
  R1“干净度”（用户反馈只覆盖 drums 的局部感知）。

## 6. 网页交接（#34 / #41；同一 16 s 时轴）

| 资源 | 相对路径（`<WS>/`） | sha256（前 16） |
|---|---|---|
| 新 result | `outputs/e2e/e2e-real-20261001-03-rflow/result.json` | `49ecf9bd8c807006…` |
| 输入 clip（同一零点） | `audio/inputs/e2e-clip-001.wav` | `42f968e24dee4300…` |
| drums target（复用） | `outputs/e2e/e2e-real-20261001-03-rflow/stems/drums-reused/target.wav` | `392e02bc722366c0…` |
| bass target（新） | `outputs/e2e/e2e-real-20261001-03-rflow/stems/bass/target.wav` | `cad70e55beee425a…` |
| synthesizer target（新） | `outputs/e2e/e2e-real-20261001-03-rflow/stems/synthesizer/target.wav` | `dd3ea0b53d077e43…` |
| 最终残差 R3 | `outputs/e2e/e2e-real-20261001-03-rflow/stems/synthesizer/residual.wav` | `4ed0a3330434abb9…` |
| 剥离链 sidecar | `outputs/e2e/e2e-real-20261001-03-rflow/stage-chain.json`（+ `stage-chain-verify.json`） | `e0bc22179e232617…` |
| 唯一 stem 映射 | `outputs/e2e/e2e-real-20261001-03-rflow/stem-map.json`（3 项，ok） | 见文件 |

- 旧 run（`e2e-real-20261001-02`，result `d789cabc…8311`、110 事件）**保持原样**，供并排对照；
  `outputs/e2e/LATEST.txt` 按冻结策略指向最近一次 run（= 新 run）。
- 展示口径不变：tempo `null/unknown` 允许人工输入 BPM；不写“准确率/串音率”；层标签标为
  假设（`source="sam"`）；事件 `source="dsp"`；音频通路 `bridge`（native=unsupported）。
- 播放滑条（#41）范围 = 真实音频 0..16 s；两侧 result/音频时轴相同，可逐层并排试听
  （drums 复用件 vs 新 bass/synthesizer 与 R3 残余）。

## 7. 评审修订记录（PR #44 R1，守卫加固；无新实验/GPU）

针对主 Agent 评审意见的修复（**不改**原 110 + 新 164 events、音频、raw trace；离线回归 + 既有 run
复核，无新 Gemini/SAM 调用）：

| # | 缺陷 | 修复 | 回归 |
|---|---|---|---|
| 1 | 基线可与当前 clip 不同源（只时长相同也可通过，可能继承另一段音频的 drums） | `check_baseline_clip`：基线 `result.audio.sha256/duration_seconds` + 原始 `request.json` 实测输入 hash 必须与当前 clip 一致；写 prompt（=启动/复用/复制）之前拒绝（`E_BASELINE_CLIP`），链核对新增 `baseline_same_clip` | 同长不同源基线负例（写前拒 + 链拒）+ 同源正例 |
| 2 | 监听目录 `startswith(run_id)` 可跨 run（`run-1` 写入 `run-10-listen/`）；已存在文件会被覆盖 | 精确命名空间 `<run-id>-listen`（不用 startswith）；`raw==out` 拒；已存在不同内容拒（`E_PROXY_EXISTS`，不重写既有听音/证据），逐字节相同幂等放行 | 兄弟前缀碰撞、既有文件哨兵、幂等、raw==out 负例 |
| 3 | `stop.residual_left` 可省略/可任指；听音与分离顺序未验证（同轮并行可冒充“听后选择”） | `residual_left_is_final_residual`：必须就是最后一级 residual（root/rel/sha，拒原混音/更早 residual）；`listen_before_next_separation`：听音须在下一次成功分离**之前**且中间隔模型轮次 | late-listen、同轮并行、缺/错 residual_left 负例 |
| 4 | 报告未显式声明 DSP 参数混杂 | §3 新增新旧 DSP 参数表 + “非分离方式受控消融”声明 | 文档；164 events 未改 |

prompt 口径同步澄清：禁的是**人为** trim/shift/时间量化/监听副本换采样率；算法内部采样率转换
（44.1 kHz 进 SAM → 48 kHz 输出）是既有行为、允许；PCM16 幅度量化 ≠ 时间量化。

复核结果（离线）：既有 run `e2e-real-20261001-03-rflow` 重新核对 **15 数据 + 12 执行门 + 16 链**
全绿（含新加的顺序/最终残余/同源检查；真实事件顺序本就合规）；`agentic/` 回归 **397 passed**；
viewer Node **162 passed**（LF 检出）。基线与新 run 的 result/events/音频/raw trace 均未改动。
