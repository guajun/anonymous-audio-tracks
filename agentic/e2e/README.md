# agentic/e2e — 真实端到端：用户输入复音音乐 → Agent + SAM → onset JSON（issue #33/#42）

**一句话**：本目录是“扮演用户的 harness + 预算内真实运行 + 冻结校验 + 抽查 + run manifest”。
真实链路：**用户 harness → Pi（`google/gemini-3.8-flash`）经 `audio_attach` bridge 听到音频字节 →
Agent 自主读 `sam-audio` skill → 真实 SAM 分离（GPU）→ 自写 DSP onset/BPM → 产出
`outputs/e2e/<run-id>/result.json`（协议 `agentic-audio-tracks/v1`）→ 冻结校验 → 抽查 → run manifest**。

**两种 profile（issue #42 新增）**：

| profile | 工作流 | prompt | 链证据 |
|---|---|---|---|
| `baseline`（#33） | 对原混音各自独立分离（drums/bass/synthesizer 都输入原 clip） | `harness/prompt_template.md` | 无 sidecar（历史形态） |
| `residual`（#42） | **残差逐层剥离**：原混音 → 复用已认可 drums → 听 R1 → 自主选下一层 → SAM `--audio`=**R1 raw residual** → 听 R2 → 再剥或停 | `harness/prompt_template_residual.md` | `stage-chain.json`（`agentic-e2e-stage-chain/v1`）+ 链核对（每级 input hash == 前级 residual raw hash） |

- **不是**预写输出脚本：harness 不产生任何分离/onset/result 内容，只准备输入、发任务、收集证据。
  result.json 必须由 Agent 的真实工具调用产出（trace 里能看到 bash/audio_attach/write 的调用链）。
- **真实 / mock 严格分开**：真实运行证据在 [`reports/`](reports/)（脱敏）与本地 run 产物（不入 git）；
  自生成 fixture 只用于离线测试，一律 `kind="mock"`，报告与 manifest 中永不冒充真实结果。
- **不夸大**：真实音乐无精确真值；抽查（overlay/能量/交叉检测）是**迹象**，不是总体准确率；
  SAM 分离与 LLM 判断都只是假设。

---

## 0. 真实 / mock 对照

| 内容 | 类型 | 位置 |
|---|---|---|
| clip 裁剪/hash/登记、trace 提取、校验、manifest、抽查 | 离线（纯标准库） | `harness/` |
| 离线自动测试（127 项 e2e，0 网络 0 API 0 GPU） | mock / 自生成 fixture | `tests/` |
| 残差剥离链核对 / 监听副本（PCM16/copy） | 离线（纯标准库） | `harness/residual_chain.py` |
| 自生成复音 fixture + 真值 | **mock**（构造真值） | `fixtures/make_mock_fixture.py` |
| 真实 Gemini + bridge + SAM GPU 运行 | **真实**（会产生 API 费用与 GPU 负载） | `harness/run_e2e.py`，证据 `reports/` |
| onset overlay / 抽查统计 | 离线 DSP（对真实 run 产物） | `harness/spotcheck.py` |

## 1. 前置条件（Junior 从这里开始）

**在哪运行**：仓库 checkout 内任意目录（脚本按自身路径定位仓库）。依赖：

- Windows 10/11 + git-bash（已验证；POSIX 同理）；Python 3.11+（纯标准库，无第三方依赖）；
- 一个**已部署**的 Pi workspace（issue #31 模板：`python agentic/workspace-template/bootstrap.py …`），
  且 `doctor.py --deep` 无失败（含 SAM 权重 SHA 校验）；
- Pi 0.87.1 + Google provider 认证就绪（`pi auth check --provider google --json`）；
- 本地 SAM Audio checkout（`.venv` + `model-cache/` 权重，**不自动下载**）与 FFmpeg 4–8 full-shared；
- 一段你有权使用的真实复音音乐 WAV（本地文件）；GPU（本机 8 GB 级）。
- **上传边界（准确口径）**：源音乐/clip/stems **不入 git、不上传 GitHub**；但**裁剪后的 clip 会以
  请求内联字节（inlineData）上传到 Google/Gemini API 做分析**（本 issue 明确允许上传“本任务音频”
  供该 API 分析；不用 Gemini Files API，无服务端临时文件）；**SAM 分离全程本地离线**。

**不做**：不下载权重、不改全局认证/全局 Pi、不动 `schema/`（#32）、`workspace-template/`（#31）、
`viewer/`（#34）与根锁文件。

## 2. 三步复现（单条命令级别）

```sh
# ① 裁剪受控 clip（守桥接预算 4 MiB/8 MiB；记录源 hash/offset/时轴）
python agentic/e2e/harness/make_clip.py \
  --source "<你的音乐>.wav" --offset 4.0 --duration 16.0 \
  --workspace "<WS>" --name e2e-clip-001.wav

# ② 真实端到端运行（真实 Gemini API + GPU SAM；硬超时 2400s；先 --dry-run 看 argv）
python agentic/e2e/harness/run_e2e.py --workspace "<WS>" --run-id e2e-real-YYYYMMDD-HHMMSS \
  --clip-name e2e-clip-001.wav --timeout 2400

# ③ 复核（冻结 schema + e2e 一致性；抽查统计/overlay）
python agentic/e2e/harness/validate_result.py \
  --result "<WS>/outputs/e2e/<RUN-ID>/result.json" --level real \
  --clip "<WS>/local/e2e/clip-e2e-clip-001.wav.json" \
  --audio-root "<WS>/audio/inputs" --run-dir "<WS>/outputs/e2e/<RUN-ID>"
python agentic/e2e/harness/spotcheck.py stats --result "<WS>/outputs/e2e/<RUN-ID>/result.json" \
  --audio "<WS>/audio/inputs/e2e-clip-001.wav" --run-dir "<WS>/outputs/e2e/<RUN-ID>" --kind real
```

**成功判据**：

1. `run_e2e.py` 退出码 `0`，且打印 `e2e pass run_id=…`；
2. `outputs/e2e/<RUN-ID>/result.json` 存在且是 Agent 写的（`local/e2e/<RUN-ID>/events.jsonl` 的
   `write`/`bash` 调用链可证）；`validation.json` 全部 `pass`；
3. **执行证据门**（`validation.json.execution`，baseline 9 项 / 残差流 12 项）全 pass：精确观测
   `google`+`gemini-3.8-flash`、无终态 provider 失败、**结构化**成功 `audio_attach`（MIME=audio/*、
   bytes>0、工具未报错）、**结构化**成功真·SAM 分离（wrapper 载荷 `action=sam.separate`/`ok=true`/
   `exit_code=0`；dry-run/失败/无载荷分开计数）、成功调用与实际产物**逐一关联**（report 可解析+
   非空 outputs）、分离**尝试**（含失败）≤预算、必要 stage（含 spotcheck，残差流再加
   `residual_chain`）**存在且非失败**、事件流存在、原 runner 成功；残差流追加 3 项（**只加不减**）：
   sidecar 已记录、链核对全过、trace 观测到对残差监听副本的真实挂载（自主残差听音实证）；
4. `run-manifest.json` 含输入 hash、模型/工具 pin、代码归属（revision/组件 hash）、GPU/耗时/用量
   （含 cache 口径）、tool trace 摘要、失败 stage（如有）。

> **数据校验 ≠ 执行证明**：文件/schema/一致性检查只证明文档合规；“真实 pass”必须同时过执行证据门。
> `--verify` 是**离线复核**（不产生新 API/GPU），它**不会**把失败的原始 run 或缺失的执行证据洗成 pass；
> 复核会保留原 `created_at`/`timings` 并追加 `verification_runs[]`。

**失败语义（退出码，冻结）**：`0` 成功 / `1` 运行完成但验收未过（result 缺失或校验失败）/
`2` 用法 / `3` 命令失败或无法启动 / `4` 超时（杀进程，blocked）/ `5` provider error /
`6` 事件流空/乱码/不完整。**blocked 不写成 success**；失败时产物（事件流/stderr/manifest）全保留。

## 2b. 残差逐层剥离 run（issue #42，`--profile residual`）

```sh
# 真实对照 run：复用基线 drums（target/residual/36 事件）+ 最多 2 次额外 SAM；
# 每级 input hash 必须 == 前级 residual raw hash（sidecar + 真实 request/trace 三重核对）
python agentic/e2e/harness/run_e2e.py --workspace "<WS>" --run-id <新 RUN_ID> \
  --clip-name e2e-clip-001.wav --profile residual --baseline-run <基线 RUN_ID> \
  --sam-separations 2 --sam-timeout 900 --timeout 2400

# 离线复核（链 + 数据校验 + 执行门；不产生新 API/GPU）
python agentic/e2e/harness/residual_chain.py verify \
  --chain "<WS>/outputs/e2e/<RUN_ID>/stage-chain.json" --run-dir "<WS>/outputs/e2e/<RUN_ID>" \
  --clip-json "<WS>/local/e2e/clip-e2e-clip-001.wav.json" --audio-root "<WS>/audio/inputs" \
  --baseline-run-dir "<WS>/outputs/e2e/<基线 RUN_ID>" --ws "<WS>" \
  --events "<WS>/local/e2e/<RUN_ID>/events.jsonl" --budget 2 \
  --result "<WS>/outputs/e2e/<RUN_ID>/result.json"
```

**残差流硬性规则（prompt + 链核对双重强制）**：①优先复用基线 drums（逐字节副本 + 事件原样，
基线只读不改）；②每级 SAM `--audio` = 前一级 **raw residual**（PCM16 监听代理不得当输入）；
③“各自从原混音独立分离”链断裂，不能伪称 sequential；④每级先听 residual（监听副本放
`audio/inputs/<run-id>-listen/`，记录 raw/proxy hash + 转换参数 + 等长）再选下一层/停止；
⑤停止理由 ∈ `near-silence|artifacts-only|no-identifiable-layer|uncertain|budget`，残余写
limitations（residual 是模型估计、误差会累积、不承诺更准）。

## 3. 输出布局与 latest 策略（冻结）

```text
<WS>/audio/inputs/e2e-clip-001.wav        输入 clip（#31 音频 manifest 已登记 sha256）
<WS>/outputs/e2e/<RUN-ID>/result.json     Agent 产出的协议结果（每 run 独立目录，不覆盖旧 run）
<WS>/outputs/e2e/<RUN-ID>/notes.md        Agent 的假设/参数/失败与不确定性/抽查记录
<WS>/outputs/e2e/<RUN-ID>/scripts/        Agent 自写 DSP 脚本（实际运行过）
<WS>/outputs/e2e/<RUN-ID>/stems/<层名>/   SAM 产物（target.wav/residual.wav/request.json/report.json）
<WS>/outputs/e2e/<RUN-ID>/validation.json 校验报告（数据校验 + 执行证据门）
<WS>/outputs/e2e/<RUN-ID>/stem-map.json   唯一 stem 文件名映射（hash/相对路径/采样率）
<WS>/outputs/e2e/<RUN-ID>/stage-chain.json 残差剥离阶段链 sidecar（issue #42；Agent 写）
<WS>/outputs/e2e/<RUN-ID>/stage-chain-verify.json 链核对报告（harness 写）
<WS>/audio/inputs/<RUN-ID>-listen/       本 run 派生的监听副本（PCM16/copy；raw/proxy hash 可审计）
<WS>/outputs/e2e/<RUN-ID>/stems-unique/   唯一文件名副本（stem-<instrument-id>-<basename>）
<WS>/outputs/e2e/<RUN-ID>/spotcheck/      overlay.wav + spotcheck.json（harness 写）
<WS>/outputs/e2e/<RUN-ID>/run-manifest.json run manifest（harness 写）
<WS>/outputs/e2e/LATEST.txt               最近一次 run id（单行 UTF-8）——**latest 指针**
<WS>/local/e2e/<RUN-ID>/prompt.txt|events.jsonl|stderr.txt|trace-summary.json  本地证据（不入 git）
```

**路径基准（冻结）**：`audio.filename` 相对 `audio/inputs`（= bridge 根 `PI_AUDIO_BRIDGE_ROOT`）；
`instruments[].stem.filename` 相对 run 目录 `outputs/e2e/<RUN-ID>/`。两者都必须是 #32 E_PATH
意义上的安全相对路径，且解析后不越出各自根目录（harness 现算 hash 核对）。

## 4. 冻结接口（#34 / 下游用；变更需在 issue #33 评论提出）

| 项 | 值 |
|---|---|
| 结果协议 | `agentic-audio-tracks/v1`（#32 冻结；校验 `python agentic/schema/validate.py <file>`） |
| 结果路径 | `outputs/e2e/<run-id>/result.json`；latest = `outputs/e2e/LATEST.txt` |
| run manifest | schema `agentic-e2e-run-manifest/v3`（= v2 + `flow`/`stage_chain`（issue #42），v2 字段语义不变：输入 hash、模型/工具/代码归属 pin、参数、耗时/GPU、用量含 cache 口径、trace 摘要、stages、数据校验+执行门、限制、复现命令；`verification_runs[]` 追加式） |
| 残差剥离 stage sidecar | schema `agentic-e2e-stage-chain/v1`（`stage-chain.json`；示例 `fixtures/stage-chain.example.json`）：每级 layer/描述/sam(performed|reused)/input/target/residual/request 的相对路径+hash、listen（audio_attach + proxy raw/proxy hash/转换参数/等长）、reused（基线 run/result hash/基线原件 hash/事件 canonical hash/用户局部反馈）、stop(reason/evidence/residual_left)、limitations；**加法 sidecar，不扩展 #32 冻结 result 协议** |
| 链核对 | schema `agentic-e2e-stage-chain-verify/v1`（`residual_chain.py verify`）：sidecar 结构 × 真实文件 hash × `request.json` 实际 input × trace `--audio` 参数链；**每级 input hash == 前级 raw residual hash**，原混音/监听代理偷换 → FAIL；时间轴（每文件自身采样率，时长 == clip，禁 trim/shift/量化）；监听代理（run 派生目录/PCM16 或 copy/等长）；停止/预算规则；基线复用只读完整性（hash + 36 事件 canonical hash `sha256(json.dumps(events, sort_keys=True, separators=(",", ":"), ensure_ascii=False))`） |
| 执行证据门 | schema `agentic-e2e-execution-gate/v2`：精确观测 model/provider（缺 provider 即 fail）+ 无终态 provider 失败 + 结构化成功附件（MIME=audio/* 且 bytes>0）+ 结构化成功真·SAM 分离（wrapper `action=sam.separate`/`ok=true`/`exit_code=0`，shell 掩盖失败不算成功）+ 成功调用与产物逐一关联 + 尝试数（含失败/unknown）≤预算 + 必要 stage（含 spotcheck）存在且非失败 + 原 runner 成功；真实 pass 的必要条件 |
| 唯一 stem 映射 | schema `agentic-e2e-stem-map/v1`：`stem-map.json` + `stems-unique/stem-<instrument-id>-<basename>`（真实 hash/相对路径/采样率；只做加法，不动原 Agent 产物、不改 schema、时轴不变） |
| 校验层级 | `--level real`（真实来源链 **bridge AND llm AND sam AND dsp** 全在 + 精确 SHA/前缀 pin + bridge 通路）/ `--level fixture`（自生成 mock） |
| 事件来源标注 | DSP 检出的事件必须 `source="dsp"` + `method=<检测法>`；乐器 `source` = 假设来源（`sam`/`llm`） |
| 音频通路 | `bridge`（`audio_attach`）；**Pi 原生音频 = unsupported**（#30），结果不得冒充 `native` |
| 模型 | 音乐 Agent `google/gemini-3.8-flash`（#30 冻结，bridge 硬锁定）；实施 worker `openrouter/xiaomi/mimo-v2.6-pro`（本 issue 执行约定） |
| 工具 pin | sam-audio skill `dfbc40a9541f…`；SAM 上游 commit `c603de8794cc…`；bridge `audio-bridge.ts` sha256 `dac9abe5…` |
| 预算 | Pi run 硬超时默认 2400s；真实 SAM 分离 ≤3 次/clip、单次 `--timeout 900`；校验修复循环 ≤2；无参数扫描 |

## 5. 离线测试（0 API / 0 GPU）

```sh
uv run --no-project --python 3.12 --with pytest==8.4.2 python -m pytest agentic/e2e/tests -q
```

成功判据：`127 passed`。覆盖：WAV 读取按实际编码 tag（**float32 不再当 int32**；PCM16/24/32 +
float32 + EXTENSIBLE fixture；不受支持/损坏格式拒收）、spotcheck 逐 stem 采样率（48k stem 不被
44.1k mix 时钟索引）与 `--sample 1` 不除零、clip 裁剪元数据/实际 hash/幂等（含小数参数）与
`E_HASH_CONFLICT`/`E_BRIDGE_SIZE`/`E_CLIP_EMPTY`/非有限数/预算守卫、写前路径守卫（run id /
clip 名白名单 + realpath 包含）、新 run 拒绝复用旧证据目录、执行证据门（失败分离单独计数、
失败 runner 不得被 verify 洗白、时间戳保留）、隐私扫描（Windows/UNC/POSIX 绝对路径含 JSON
转义；相对引用/URL 不误伤）、严格 pin（精确 SHA/前缀）与 mock/native 全字段拒绝、
唯一 stem 映射（原件不动）；mock fixture 真值与确定性；DSP helper 方法冒烟（**不代表真实音乐
准确率**）；**mock-native-bridge 区别**；trace 提取/分离计数；prompt 实例化；
`run_e2e --dry-run`（零 API）；脱敏/隐私扫描；run manifest 与 `LATEST.txt` 策略；
spotcheck overlay/统计的结构与“非总体准确率”口径。**残差剥离（issue #42）**：prompt 强制逐层
residual/复用 drums/安全停止与预算、链规则（每级 input==前级 raw residual；错误原混音/监听
代理/trace 偷换输入拒绝，不能误报 sequential）、时间轴（每 stem 自身采样率、等长、禁 trim/
shift）、停止/预算规则（budget 停止必须真实用尽预算）、监听副本（raw/proxy hash、转换参数、
等长、run 派生目录、PCM16/copy 编码）、基线复用完整性与**基线不可变**、路径/隐私、残差流
gate 只加不减、`--profile residual --dry-run`、verify CLI 退出码。

## 6. 隐私与清理

- 公开材料只放**脱敏摘要**（`--redact`）与纯 JSON 摘要：不放音频、权重、API key、完整会话、
  个人绝对路径；`validate_result.py` 的 `privacy` 检查会拒掉 key 形态/长 base64/**绝对路径**
  （Windows/UNC/POSIX，含 JSON 转义形态）；相对引用与 URL 不误伤。
- **上传边界**：只有裁剪后的 clip 以内联字节上传到 Google/Gemini API 做分析（本 issue 允许）；
  源音乐不上传、stems 不上传、**不使用 Gemini Files API**；SAM 分离本地离线；GitHub 上无音频。
- 音频/stems/事件流/会话都在 workspace 的 ignored 目录（`audio/`、`outputs/`、`local/`、`sessions/`）。
- 清理单次 run：删 `<WS>/outputs/e2e/<RUN-ID>` 与 `<WS>/local/e2e/<RUN-ID>`；清理 clip：
  删 `audio/inputs/e2e-clip-*.wav` 并从 `audio/inputs/manifest.json` 移除对应条目。
- 取消/失败只清理本任务自己的进程（run_bounded 杀真实 Pi 进程；SAM 调用自带 `--timeout`），
  **不做全局 taskkill**。

## 7. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `E_CLIP: clip 描述不存在` | 没先跑 `make_clip.py` | 先执行 §2 ① |
| `E_HASH_CONFLICT`（make_clip） | 同名 clip 内容/来源不同 | 换 `--name`；不要覆盖已有 clip |
| `E_BRIDGE_SIZE` | clip 超 4 MiB 预算 | 缩短 `--duration` 或降采样/转 mp3 |
| `preflight.auth = fail` | Pi Google 认证未就绪 | 用 Pi 自身认证流程；不要写 key 到文件 |
| `preflight.clip: hash 漂移` | clip 文件被改动 | 重新 `make_clip.py` 或恢复文件 |
| 退出码 4（blocked） | Pi run 超硬超时 | 看 `local/e2e/<RUN-ID>/events.jsonl` 定位阶段；可减 clip/减分离次数后重跑（有限次数） |
| 退出码 5 / 6 | provider error / 事件流不完整 | 按失败留证并报告，不换模型、不伪造 |
| `result 缺失`（退出码 1） | Agent 未产出/写错路径 | 读 trace 的工具调用与 notes.md 定位失败阶段 |
| `event_provenance FAIL` | 事件缺 `source`/`method`，或 DSP 法标了非 dsp 来源 | 按校验 detail 修正（≤2 次修复循环），不许篡改时间掩盖 |
| `privacy FAIL` | 结果里有绝对路径/key 形态/长 base64 | 改成相对路径/占位符，删除字节块 |

## 8. 已知限制（诚实口径）

- 真实音乐**没有**精确真值：onset/乐器/tempo 都是估计；抽查（overlay 试听、能量上升比、
  stem 串音比、独立检测器差异计数）只是**迹象**，不构成总体准确率，报告按此口径记录漏检/误检/串音。
- SAM 输出只有“目标/残差”两路，多于两个声源时只能多次分离，串音不可避免；乐器标签是假设。
- bridge ≠ Pi 原生音频（#30）；音频一次性注入、不落会话；模型对编码细节的感知不保证。
- BPM 由 onset 间隔估计（倍频折算），低置信度时协议允许 `bpm: null`。
- **残差不是真值**：target/residual 都是模型估计，逐层剥离误差会**累积**，不承诺比原独立分离
  更准；用户对 drums 的认可只是**局部定性**反馈，不代表 drums target 完全干净或 residual 是真值；
  逐层剥离后其余声层来源/边界不确定，报告按“复用/实测/推测”分栏记录。
- usage/cost 是 Pi 记账近似值，非账单真值；GPU 耗时随负载波动。

## 9. 证据索引

- 真实运行证据（脱敏）：[`reports/10-real-run.md`](reports/10-real-run.md)
- 抽查记录（漏检/误检/串音）：[`reports/20-spotcheck.md`](reports/20-spotcheck.md)
- 环境与预算：[`reports/00-environment.md`](reports/00-environment.md)
- 限制 / blocked 记录：[`reports/30-limits.md`](reports/30-limits.md)
- 残差逐层剥离对照 run（issue #42）：[`reports/40-residual-flow.md`](reports/40-residual-flow.md)
- 冻结接口与 pin：[`manifest.json`](manifest.json)
- 上游：[`../schema/README.md`](../schema/README.md)（#32 协议）、
  [`../workspace-template/README.md`](../workspace-template/README.md)（#31 workspace）、
  [`../audio-probe/README.md`](../audio-probe/README.md)（#30 bridge 冻结接口）
