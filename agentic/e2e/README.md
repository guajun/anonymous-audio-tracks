# agentic/e2e — 真实端到端：用户输入复音音乐 → Agent + SAM → onset JSON（issue #33）

**一句话**：本目录是“扮演用户的 harness + 预算内真实运行 + 冻结校验 + 抽查 + run manifest”。
真实链路：**用户 harness → Pi（`google/gemini-3.8-flash`）经 `audio_attach` bridge 听到音频字节 →
Agent 自主读 `sam-audio` skill → 真实 SAM 分离（GPU）→ 自写 DSP onset/BPM → 产出
`outputs/e2e/<run-id>/result.json`（协议 `agentic-audio-tracks/v1`）→ 冻结校验 → 抽查 → run manifest**。

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
| 离线自动测试（40 项，0 网络 0 API 0 GPU） | mock / 自生成 fixture | `tests/` |
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
- 一段你有权使用的真实复音音乐 WAV（本地文件，**不上传、不入 git**）；GPU（本机 8 GB 级）。

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
3. trace 中可见 `audio_attach`（bridge 听音）与真实 `sam separate`（非 `--dry-run`）调用；
4. `run-manifest.json` 含输入 hash、模型/工具 pin、GPU/耗时/用量、tool trace 摘要、失败 stage（如有）。

**失败语义（退出码，冻结）**：`0` 成功 / `1` 运行完成但验收未过（result 缺失或校验失败）/
`2` 用法 / `3` 命令失败或无法启动 / `4` 超时（杀进程，blocked）/ `5` provider error /
`6` 事件流空/乱码/不完整。**blocked 不写成 success**；失败时产物（事件流/stderr/manifest）全保留。

## 3. 输出布局与 latest 策略（冻结）

```text
<WS>/audio/inputs/e2e-clip-001.wav        输入 clip（#31 音频 manifest 已登记 sha256）
<WS>/outputs/e2e/<RUN-ID>/result.json     Agent 产出的协议结果（每 run 独立目录，不覆盖旧 run）
<WS>/outputs/e2e/<RUN-ID>/notes.md        Agent 的假设/参数/失败与不确定性/抽查记录
<WS>/outputs/e2e/<RUN-ID>/scripts/        Agent 自写 DSP 脚本（实际运行过）
<WS>/outputs/e2e/<RUN-ID>/stems/<层名>/   SAM 产物（target.wav/residual.wav/request.json/report.json）
<WS>/outputs/e2e/<RUN-ID>/validation.json 校验报告（harness 写）
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
| run manifest | schema `agentic-e2e-run-manifest/v1`（输入 hash、模型/工具 pin、参数、耗时/GPU、用量、trace 摘要、stages、校验、限制、复现命令） |
| 校验层级 | `--level real`（真实来源链 + pin + bridge 通路）/ `--level fixture`（自生成 mock） |
| 事件来源标注 | DSP 检出的事件必须 `source="dsp"` + `method=<检测法>`；乐器 `source` = 假设来源（`sam`/`llm`） |
| 音频通路 | `bridge`（`audio_attach`）；**Pi 原生音频 = unsupported**（#30），结果不得冒充 `native` |
| 模型 | 音乐 Agent `google/gemini-3.8-flash`（#30 冻结，bridge 硬锁定）；实施 worker `openrouter/xiaomi/mimo-v2.6-pro`（本 issue 执行约定） |
| 工具 pin | sam-audio skill `dfbc40a9541f…`；SAM 上游 commit `c603de8794cc…`；bridge `audio-bridge.ts` sha256 `dac9abe5…` |
| 预算 | Pi run 硬超时默认 2400s；真实 SAM 分离 ≤3 次/clip、单次 `--timeout 900`；校验修复循环 ≤2；无参数扫描 |

## 5. 离线测试（0 API / 0 GPU）

```sh
uv run --no-project --python 3.12 --with pytest==8.4.2 python -m pytest agentic/e2e/tests -q
```

成功判据：`40 passed`。覆盖：clip 裁剪元数据/实际 hash/幂等与 `E_HASH_CONFLICT`/桥接预算
`E_BRIDGE_SIZE`/范围守卫；mock fixture 真值与确定性；DSP helper 在 mock fixture 上的检出与
BPM 估计（**方法冒烟，不代表真实音乐准确率**）；real 级校验全绿与全部负例（hash/时轴/
来源标注/隐私/路径/stem）；**mock-native-bridge 区别**（mock 结果在 real 级必须被拒、冒充
`native` 必须被拒）；trace 提取与真实分离计数（区分 `--dry-run`）；prompt 实例化；
`run_e2e --dry-run`（零 API）；脱敏/隐私扫描；run manifest 与 `LATEST.txt` 策略；
spotcheck overlay/统计的结构与“非总体准确率”口径。

## 6. 隐私与清理

- 公开材料只放**脱敏摘要**（`--redact`）与纯 JSON 摘要：不放音频、权重、API key、完整会话、
  个人绝对路径；`validate_result.py` 的 `privacy` 检查会拒掉 key 形态/长 base64/绝对路径。
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
- usage/cost 是 Pi 记账近似值，非账单真值；GPU 耗时随负载波动。

## 9. 证据索引

- 真实运行证据（脱敏）：[`reports/10-real-run.md`](reports/10-real-run.md)
- 抽查记录（漏检/误检/串音）：[`reports/20-spotcheck.md`](reports/20-spotcheck.md)
- 环境与预算：[`reports/00-environment.md`](reports/00-environment.md)
- 限制 / blocked 记录：[`reports/30-limits.md`](reports/30-limits.md)
- 冻结接口与 pin：[`manifest.json`](manifest.json)
- 上游：[`../schema/README.md`](../schema/README.md)（#32 协议）、
  [`../workspace-template/README.md`](../workspace-template/README.md)（#31 workspace）、
  [`../audio-probe/README.md`](../audio-probe/README.md)（#30 bridge 冻结接口）
