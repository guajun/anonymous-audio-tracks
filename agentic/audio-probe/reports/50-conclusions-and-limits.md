# 50 · 结论、限制与 blocked 记录（issue #30）

## 50.1 结论（对应 issue 验收项）

1. **原生结论：Pi 0.87.1 不支持声音模态输入（native = unsupported）**。
   - 精确模型与版本、调用命令、实际输送内容、模型答案、限制 → 见 00 / 30（REAL）与 20（OFFLINE/SOURCE）。
   - 失败是确定性的：消息类型无 AudioContent、模型目录 `input:["text","image"]`、
     所有文档化入口经 `processImage` 拒绝音频或降级为 UTF-8 乱码文本；真实调用中模型自述无法感知音频。
2. **Gemini API 原生支持音频**（`inlineData` = MIME + 字节），Pi 管道经 **bridge** 可以把字节送达：
   REAL 验证（40）中模型对两段构造音频的音高方向判断与构造真值一致（支持性证据，
   n=2 非统计证明，见 40.4）。
3. **明确区分**：Gemini API 原生 ✅ / Pi 原生 ❌ / bridge ✅（非 native，类型标注借用 `image` 块的 workaround）。
   本目录任何文档不把 bridge 称为 native，不把文件名/乱码文本当音频输入成功。
4. **不主张时间戳/事件计数级真值准确度**（事件计数实测与构造不一致，40.4）。

## 50.2 桥接（bridge）限制 — 下游 workspace 必读

- **一次性注入**：每个附件只注入紧随工具调用的下一次模型请求；同一 run 后续请求不重发。
- **模型锁定**：attach 与注入都只对 `google/gemini-3.8-flash` 生效（`PI_AUDIO_BRIDGE_MODEL`，
  默认即该模型）；其他 provider/model 以 `E_AUDIO_MODEL` 拒绝，模型切换前的残留音频会被丢弃而非注入。
- **大小/路径/队列受控**：单文件 ≤4 MiB（`PI_AUDIO_BRIDGE_MAX_BYTES`），**排队总量 ≤8 MiB raw**
  （`PI_AUDIO_BRIDGE_MAX_QUEUE_BYTES`；base64 体积 +33%，仍低于目录声明的 20 MiB inline 上限），
  读取后按 `raw.length` 复核；仅限受控根目录（`PI_AUDIO_BRIDGE_ROOT`），realpath 防符号链接逃逸。
- **生命周期清理**：排队音频在 `session_start` / `session_shutdown` / `agent_settled` 边界清空，
  不会跨会话或在 run 中止后残留；音频字节只驻内存，不落 session/转录（40 已 grep 复核）。
- **失败语义**：无效输入时 `execute()` **抛错**（Pi 标记为 failed tool result），错误消息带稳定的
  `E_AUDIO_*` 码（ARGS/PATH/NOT_FOUND/NOT_FILE/SIZE/QUEUE/MIME/READ/MODEL）且不含绝对路径。
- **类型标注 workaround**：音频承载在 `type:"image"` 块中（MIME 为 audio/*）；按块类型判型的消费者会误判。
- **无转码**：按 sniff 到的容器 MIME 原样上传（wav/mp3/ogg/flac/m4a 白名单）。
- **工具结果通道**（能力矩阵 #4）payload 级可行但未真实验证，下游用 user 注入通道。

## 50.3 费用 / 上传 / 清理 / 凭据

- 真实调用：**2 次 Pi run**（native 1 + bridge 1）；**每次 Pi run 含 2 个 assistant requests**，
  合计 4 个 requests。实测用量（事件流 `usage`，非估算）：native 66,154 totalTokens / Pi 记账
  cost 0.0518；bridge 3,237 totalTokens / cost 0.0053。费用为 Pi 从 provider usage 元数据记账的
  USD 近似值，非账单真值；不作“费用可忽略”式断言。离线测试/探针 0 次调用。
- 调用边界：脚本经 `probe/run_bounded.py` 硬超时（默认 300s），失败/超时保留产物、非零退出，
  并识别 Pi 退出 0 但事件流含 provider error/aborted 的失败；不重试循环。
- 上传边界：仅本任务自产 fixture（允许上传给 Google API 做分析）；**不上传**私有音频/模型/key/完整会话。
- 传输形态：请求内联 `inlineData`；**不用 Gemini Files API**，因此**无服务端临时文件、无需删除**；
  本地清理：`python probe/clean_fixtures.py --yes`（fixtures/、tmp/ 均 gitignored）。
- 凭据：扩展与探针不读取/不输出 key；未执行 `pi auth print`；未改全局认证/settings；
  报告经 `probe/extract_evidence.py` 脱敏（base64、sk-/AIza/Bearer token、HOME/REPO 路径）。

## 50.4 自动化测试（mock/offline，与真实证据分开）

- 命令：`uv run pytest agentic/audio-probe/tests -v`；结果：**22 passed**（2026-09-30）。
- 覆盖：fixture 真值（过零率估计音高方向）、bridge 校验逻辑（MIME/大小/队列/路径逃逸/错误码与路径泄漏）、
  **extension 级 mock 测试**（mock Pi registry/context：多附件、一次性注入、模型锁定、
  失败输入、生命周期清理、不落转录）、payload 形状（含 `inlineData.mimeType="audio/wav"` 且 base64 脱敏）、
  原生路径拒绝音频、**fake pi 回归**（run_real_*.sh 的成功/失败/挂起/provider-error 四态与退出码）、
  脱敏器（先脱敏后截断、JSON 转义路径、repo 根路径、越界路径）。
- 真实模型证据**不在**自动测试内（避免 CI 发真实调用）；两者在 README/报告中分栏标注。

## 50.5 blocked 记录

- **无 blocked**。`pi auth check --provider google` = ready；`google/gemini-3.8-flash` 存在；
  两次真实运行均正常返回（`stopReason: stop`）；无模型切换、无网络卡死、无重试循环。
- 非 blocked 的本地修正：`@file` 与消息需分开 argv（首次 native 运行的脚本参数错误，已修复并留证，30.3）。
- SAM GPU 不在本项运行（依 issue 约定）。

## 50.6 给下游的稳定最小接口（冻结于本 PR）

见 README §9：`pi --extension ./bridge/audio-bridge.ts --tools audio_attach` +
工具契约 `audio_attach(path) -> {attached, mime, bytes, sha256, queueDepth, note}` +
`PI_AUDIO_BRIDGE_ROOT` / `PI_AUDIO_BRIDGE_MAX_BYTES`。接口变更需在 reports/ 记录。
