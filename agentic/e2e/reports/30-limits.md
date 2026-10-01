# 30 — 限制、失败语义与未交付项（issue #33）

## 1. 结论一句话

真实端到端链路**已打通并实证**（用户 harness → Pi/gemini-3.8-flash → bridge 听音 →
Agent 自主 SAM GPU 分离 → 自写 DSP onset/BPM → 冻结协议 JSON → 校验全绿 → 抽查 → run manifest）。
**没有**外部阻塞（模型 / GPU / 权重 / ABI 均可用），因此本 issue 以 `Closes #33` 提 PR；
但结果质量有明确局限（见下），**不作准确率承诺**。

## 2. 结果质量限制（诚实口径）

- 真实音乐**无精确真值**：onset、乐器、tempo 全是估计/假设；抽查只是迹象（reports/20）。
- **分离串音明显**（isolation ratio ≈0.95–1.0）：SAM 目标/残差两路在这段复音音乐上互相渗漏；
  乐器标签（drums/bass/synthesizer）来自分离描述 + 听感，不保证正确。
- **漏检**：独立检测器比对出 6 处可能漏检（弱拍/ghost notes/低频连奏）；**误检**：2 个 bass 弱
  onset 在全频能量口径下不成立（连奏滑音、结尾截断），保留原样未删改。
- **tempo 不可靠**：IOI 直方图置信度 0.19–0.27，按协议 `bpm: null` + `source: "unknown"`。
- **pitch/duration 未填**：无充分证据（协议“没证据不填”）；本项不产出 MIDI、不量化到节拍。
- DSP 方法单一（能量通量族）：独立复核检测器与 Agent 检测器同族，共享偏差不可见。

## 3. 工程限制

- **bridge ≠ Pi 原生音频**：Pi 0.87.1 原生声音模态 unsupported（#30）；音频一次性注入、不落会话、
  不转码；模型对编码细节感知不保证。`audio_attach` 预算：单文件 ≤4 MiB、排队 ≤8 MiB。
- **SAM 只有 target/residual 两路**，多于两个声源时必须多次分离（本次 3 次预算用满）；
  需要 4+ 层时预算与串音都会恶化。
- **GPU 单项占用**：本次 run 前空闲 5347 MiB，SAM 单次推理约 22 s（cuda/bfloat16）；
  未杀任何其它进程、未下载任何权重。显存紧张机器可能需要 `--dtype float32`/CPU（慢）。
- **usage/cost 是 Pi 记账近似值**（0.549 USD 级），非账单真值；totalTokens 含缓存/音频等口径差异。
- **harness 不做自动修复**：校验失败只给结构化 detail，修复循环 ≤2 且禁止改时间/量化掩盖；
  本 run 一次通过（0 次修复）。

## 4. 失败语义与 blocked 口径（冻结）

| 退出码 | 含义 | 处置 |
|---|---|---|
| 0 | 成功（验收全过） | — |
| 1 | 运行完成但验收未过（result 缺失/校验失败） | 看 stage 与 events 定位；不写成 success |
| 2 | 用法错误 | 修参数 |
| 3 | 命令失败/无法启动 | 看 stderr |
| 4 | 硬超时（杀 Pi 进程） | **blocked**：保留事件流，报告 blocked，不无限重试 |
| 5 | provider error / aborted | **blocked**：留证，不换模型 |
| 6 | 事件流空/乱码/不完整 | **failed**：不计入成功 |

若外部阻塞（模型下线 / GPU 或权重不可用 / ABI 不兼容），交付口径是：**pipeline PR + 实证 +
issue 改 `Refs` 并保持 blocked**，绝不用 mock 运行关闭（本次未发生）。
取消/失败只清理本任务自己的进程（`run_bounded` 杀真实 Pi 进程；SAM 调用自带 `--timeout`），
不做全局 `taskkill`。

## 5. 未交付 / 明确不做

- **不做网页**（#34 并行负责）；本项只交付真实 result/audio 的本地加载说明与协议 fixture。
- **不交付** 音高/精确时值/MIDI/乐谱；不做总体准确率评估（无真值）。
- **不上传** 私有音频（clip/stems/源音乐）、权重、API key、完整会话/事件流、个人绝对路径。
- **不修改** #31/#32 冻结接口、根锁文件、全局 Pi/认证、SAM 迁移代码（PR #26 只读）。

## 6. 隐私自检

- 公开提交：代码、纯 JSON mock fixture/真值、脱敏报告（本文档即示例）。
- 自动把关：`validate_result.py` 的 `privacy` 检查（绝对路径/key 形态/长 base64）在真实 result 上 pass；
  `run_e2e.py --redact` 输出全部路径→占位符；报告人工复查无 `C:\`、`F:\`、用户名、key。
- clip sha256 保留（单向哈希，可复核一致性，不泄漏内容）。
