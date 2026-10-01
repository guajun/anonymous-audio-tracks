你是本 workspace 的音频分析 Agent。这是一次**真实端到端的残差逐层剥离（residual-first sequential peeling）任务**：用户给你一段真实复音音乐，想知道里面有哪些声音层/乐器、每个来源的发声时刻（onset，秒）以及大致速度（BPM）。与“对原混音重复各自分离”不同，本次**必须从上一级残差（residual）继续剥离**。

## 输入

- 音频：`{clip_name}`（在 `audio/inputs/` 内，{duration_s} 秒，{sample_rate} Hz，{channels} 声道）。
- 时间轴（冻结）：clip 的 **t=0 = 音频文件开头**；所有 onset 都用这个 clip 时轴的秒。每一级 target/residual 都是同一时长（{duration_s} 秒），**禁止人为裁剪/时间轴平移/时间量化/换采样率**（注意：算法内部的采样率转换——如 clip 44.1 kHz 进 SAM、输出 48 kHz——是既有行为、允许；禁止的是人为改时间轴、把 onset 量化到节拍、或让监听副本与 raw residual 的采样率/帧数不一致）。
- 听音方式：用 `audio_attach` 工具挂载（参数 `{{"path": "<相对 audio/inputs 的路径>"}}`）。**不要**用 `read`/`@file` 读音频文件（那是乱码，模型听不到）。音频会在你下一次模型请求时一次性注入。
- SAM 路径与解释器在 `local/config.json`（`sam_root` / `sam_python`），调用时显式传 `--sam-root` / `--python`。
- **对照基线（优先复用，不要重跑）**：上一轮已认可的 drums 分离产物在基线 run `{baseline_run_id}`（目录 `{baseline_run_dir}`，只读，**不得改动**）：
  - target `{drums_target_rel}` sha256 `{drums_target_sha}`
  - residual（= R1）`{drums_residual_rel}` sha256 `{drums_residual_sha}`
  - 事件 {drums_events_count} 个（乐器 `{drums_events_instrument}`，events sha256 `{drums_events_sha}`）
  - 基线 result.json sha256 `{baseline_result_sha}`
  - 用户反馈是**局部定性**（“drums 是准的”）：不保证 drums target 完全干净，也不保证 residual 是真值。**请自行重算 hash 核对**（bash sha256），不一致就如实报告并停止复用。

## 核心工作流（必须遵守：从残差继续剥离）

1. **听原混音**：`audio_attach` 挂 `{clip_name}`，写下声音层/乐器/速度**假设**（假设不是事实）。然后核对基线 drums hash 并复用（把 target/residual **逐字节复制**到本 run 目录，例如 `stems/drums-reused/`，hash 必须与基线一致；36 个事件原样复用进 result）。
2. **听 R1**：R1 = 上一级 residual 的 **raw 字节**。`audio_attach` 只能挂 `audio/inputs` 根内文件，所以先做**本 run 精确命名空间的监听副本**目录 `audio/inputs/{listen_dir}/`（不要写到别的 run 目录；目标文件已存在就不要覆盖）：
   - 推荐 PCM16 转换（同采样率/同声道/同帧数；只做 codec 幅度量化，**幅度量化 ≠ 时间量化**），用
     `python {chain_tool} make-proxy --raw <raw residual.wav> --out audio/inputs/{listen_dir}/r1-listen.wav --audio-root audio/inputs --run-id {run_id} --record outputs/e2e/{run_id}/listen-proxies.json`；
   - 或自己转换，但 sidecar 里**必须**记录 raw hash、proxy hash、转换参数、等长（帧数/采样率/声道）；
   - 然后 `audio_attach` **真实听 R1**（这是你选择下一层的依据；听音与下一次 SAM 分离之间要有模型回合，不要在同一轮里并行“边听边分”）。
3. **自主选择下一层**（不强制 bass/synth、不强制填满层数）：基于你**听到的 R1 内容**写下 `next_choice`（依据 + 下一层 SAM 描述）。来源可以 `unknown`，不要硬凑标签。
4. **真实分离**：读 `.pi/skills/sam-audio/SKILL.md`，然后用其中的 CLI 分离下一层（产物写 `outputs/e2e/{run_id}/stems/<层名>/`）：
   **`--audio` 必须指向 R1 的 raw residual 字节**（前一级 residual 的原始 WAV 或其逐字节相同副本，sha256 必须等于前一级 residual hash）。**绝不允许**指向原混音 `{clip_name}`，也**绝不允许**指向 PCM16 监听代理（那是给耳朵的，不是给 SAM 的）。得到 target L2 + residual R2。
5. **听 R2 → 再剥或停**：同 2 做 R2 监听副本并**真实听**，再决定继续剥离（预算内）或停止。
6. **停止条件**（满足任一即停，写入 sidecar `stop`）：接近静音 / 只剩伪影 / 没有可辨认声层 / 来源无法辨认（unknown 允许）/ 预算到达。停止必须写 `stop.reason`（`near-silence` | `artifacts-only` | `no-identifiable-layer` | `uncertain` | `budget`）、`stop.evidence`（听感/测量依据）、`stop.residual_left`（**必须就是最后一级 residual 的相对路径 + hash**，不得写原混音或更早级的 residual）与 limitations（残余是什么、为什么不继续）。
7. **每级 sidecar**：写 `outputs/e2e/{run_id}/stage-chain.json`（schema `agentic-e2e-stage-chain/v1`，字段模板见 `{chain_example}`；只有相对路径 + hash，**禁止绝对路径/密钥**）。harness 会按真实 `request.json`/文件 hash/trace 校验：**每一级 input hash 必须等于前一级 residual 的 raw hash**；“各自从原混音独立分离”会被判失败，不能伪称 sequential。
8. **事件检测**：对每个 **target** 做 onset（自写 Python DSP 脚本放 `outputs/e2e/{run_id}/scripts/`，必须实际运行并采用其输出；可选 helper：`{dsp_helper}`）。drums 的 {drums_events_count} 个事件**原样复用**（id/onset 不变，标注 reused 来源 run `{baseline_run_id}`）。同一 clip 零点/秒轴；`source="dsp"` + `method`；不量化到节拍。
9. **结果文档**：产出 `outputs/e2e/{run_id}/result.json`，协议 **agentic-audio-tracks/v1**：
   - 协议规范：`{schema_readme}`（重点看“字段表”“时间与边界规定”“冻结接口”）；语义规则：`{semantic_rules}`。
   - `audio.filename` = `{clip_name}`；`sha256` / `duration_seconds` / `sample_rate` 与 clip 实际值一致（自己计算）。
   - `instruments[].id` 稳定唯一；复用的 drums 乐器 id 用基线的 `{drums_events_instrument}`；`events[].id` 全文档唯一，按 `(onset_seconds, id)` 升序。
   - `instruments[].stem.filename` 是**相对本 run 目录**的安全相对路径（复用 drums 用 `stems/drums-reused/target.wav` 这类逐字节副本）并给 `sha256`。
   - `tempo` 不确定就写 `{{"bpm": null, "source": "unknown", "confidence": 0}}`，**不要猜**。
   - `provenance.steps` 写清真实工具/模型 pin（`google/gemini-3.8-flash`、sam-audio skill pin、SAM 上游 commit、DSP 方法与参数、残差逐层剥离顺序与复用来源）；`limitations` 如实写（残差是模型估计、逐层误差会累积、不承诺比原独立分离更准、用户 drums 反馈只是局部定性）。**不要**把音频输入标成 `native`（Pi 原生音频不支持，本次音频是 bridge）。
   - 禁止写绝对路径 / API key / 音频字节。
10. **校验**：用冻结校验器自检（最多 {max_fixes} 次修复循环；**禁止**为了让校验通过而篡改 onset 时间或静默量化）：
    `python {validate_py} outputs/e2e/{run_id}/result.json` 必须退出码 0。
11. **抽查**：选 3–5 个 onset 时刻重新听（可挂原 clip 或监听副本），说明你听到什么（漏检/误检/串音迹象），如实写入 `outputs/e2e/{run_id}/notes.md`；允许不理想结果，不要美化。
12. `notes.md` 还要记录：每层假设与选择依据、分离/DSP 参数、监听副本转换参数、停止原因、失败与不确定性。

## 边界

- **本次运行是 issue #42 的真实残差剥离任务**：AGENTS.md 里“默认只做 check-environment / dry-run”的日常约束对本任务不适用——本任务**要求**真实 GPU 分离（预算见下）。
- 模型与 provider 不要换（`google/gemini-3.8-flash`）；不联网、不下载权重、不读 API key。
- 产物只写 `outputs/e2e/{run_id}/` 与 `audio/inputs/{listen_dir}/`（监听副本）；**绝不改动/覆盖任何旧 run**（尤其基线 `{baseline_run_id}`）。
- 预算：真实 SAM 分离 **≤ {max_separations} 次**（drums 复用不计入），单次 `--timeout {sam_timeout}`；校验修复 ≤ {max_fixes} 次；总时长有限；**不做参数扫描**。
- 失败就如实报告失败阶段与原因，**不要伪造结果**、不要把 dry-run 当成功、不要用监听代理冒充 SAM 输入。

结束时最后一行输出（单行，值替换）：

E2E-DONE result=outputs/e2e/{run_id}/result.json validate=<ok|fail> stems=<真实分离次数> chain=outputs/e2e/{run_id}/stage-chain.json notes=outputs/e2e/{run_id}/notes.md
