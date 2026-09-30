# agentic/audio-probe — Pi 声音模态输入探针与最小桥接（issue #30）

**一句话结论（先行）**：Pi 0.87.1 **没有**原生音频输入（native = unsupported）；本目录提供
(a) 可复现的失败证据，(b) 一个**明确标注为 bridge** 的最小 project-local 桥接，让
`google/gemini-3.8-flash` 真正听到音频。**桥接 ≠ Pi 原生**，两者在报告中严格分开。

- 研究模型固定：`google/gemini-3.8-flash`（不换模型；失败即 blocked）。
- 实测证据见 [`reports/`](reports/)；能力矩阵见 [`reports/10-capability-matrix.md`](reports/10-capability-matrix.md)。

---

## 1. 目录与角色

| 路径 | 内容 | 真实/离线 |
|---|---|---|
| `bridge/audio-bridge.ts` | Pi extension（工具 `audio_attach` + `context` 注入） | 用于真实运行 |
| `bridge/audio_guard.mjs` | 纯校验逻辑（路径/MIME/大小），可离线单测 | 离线 |
| `probe/make_fixture.py` | 生成构造真值明确的 WAV fixture（stdlib） | 离线 |
| `probe/payload_shape.mjs` | 用 **Pi 自带** `convertMessages` 打印将发送的 Gemini 请求形状 | 离线 |
| `probe/native_paths_probe.mjs` | 用 **Pi 自带** `processImage`/mime 代码复现原生输入路径拒绝 | 离线 |
| `probe/run_real_native.sh` | **真实** Gemini 调用：原生 `@file` + `read` 失败证据 | 真实（1 次调用） |
| `probe/run_real_bridge.sh` | **真实** Gemini 调用：桥接听音验证 | 真实（1 次调用） |
| `probe/extract_evidence.py` | 证据脱敏（base64/token/绝对路径 → 标记） | 离线 |
| `probe/clean_fixtures.py` | 清理本地 fixtures/ 与 tmp/ | 离线 |
| `tests/` | 离线自动化测试（8 项） | 离线（mock） |
| `reports/` | 运行证据、能力矩阵、结论与限制 | — |

`fixtures/`、`tmp/` 均被 `.gitignore`：**音频二进制与完整会话不入库**。

## 2. 环境与依赖（Junior 复现从这里开始）

在**仓库内**、目录 `agentic/audio-probe/` 下运行所有命令（脚本会自行 `cd` 到该目录）。

- Windows / git-bash 已验证；需要：
  - `pi` **0.87.1**（`pi --version`），Google provider 认证已就绪
    （`pi auth check --provider google --json` 只输出 `{"status":"ready",...}`，**不打印 key**）；
  - Node ≥ 20（离线探针执行 Pi 自带 JS 模块）；
  - Python 3.12+（本仓库 `uv run pytest` 即可，未改根 `pyproject.toml`/`uv.lock`）。
- 不修改任何全局配置/全局扩展；桥接只经 `pi --extension ./bridge/audio-bridge.ts` 显式加载。

## 3. 离线测试（mock，不发任何网络请求、不花钱）

```bash
cd agentic/audio-probe
uv run pytest tests -v          # 或在仓库根: uv run pytest agentic/audio-probe/tests -v
```

成功判据：`8 passed`。测试内容：fixture 真值（过零率估计音高方向）、bridge 校验逻辑、
payload 形状（必须含 `inlineData.mimeType="audio/wav"` 且 base64 被脱敏）、
原生路径必须拒绝音频、证据脱敏。

## 4. 离线探针（复现“原生为什么不支持”“payload 长什么样”）

```bash
cd agentic/audio-probe
python probe/make_fixture.py --all            # 生成 fixtures/fixture-a.wav、fixture-b.wav
node probe/native_paths_probe.mjs --file fixtures/fixture-a.wav
node probe/payload_shape.mjs
```

成功判据：
- `processImage: ok=false message="[Image omitted: could not be converted to a supported inline image format.]"`
- `image sniff: fromBuffer=null` → `@file` 把 .wav 当 **UTF-8 文本**（乱码），绝非音频；
- payload 形状：`"inlineData": {"mimeType": "audio/wav", "data": "<base64 redacted: …>"}`（MIME + 字节）。

## 5. 真实 Gemini 运行（少量调用；每次一个脚本、一次模型调用）

```bash
cd agentic/audio-probe
bash probe/run_real_native.sh    # 真实调用 1：原生失败证据（@file + read）
bash probe/run_real_bridge.sh    # 真实调用 1：桥接听音验证（2 个 audio_attach）
python probe/extract_evidence.py tmp/bridge-events.jsonl   # 脱敏查看
```

- 模型固定 `google/gemini-3.8-flash`；认证失败/模型不存在会直接报错并留证，**不会**换模型。
- 成功判据（bridge）：模型对 `fixture-a.wav` 判 "pitch goes up"、对 `fixture-b.wav` 判 "pitch goes down"
  （与 `probe/make_fixture.py` 构造真值一致）。音高**方向**是可核对的定性判断；**不**主张
  时间戳/事件计数级别的真值准确度。
- 产物：`tmp/*.jsonl`（原始事件流，gitignored）→ `probe/extract_evidence.py` 脱敏后进 `reports/`。

## 6. 准确的 payload 与调用方式（README 核心）

**原生路径（全部失败，见 reports/20、30）**：

| 入口 | 实际输送内容 | 结果 |
|---|---|---|
| `pi @fixtures/fixture-a.wav "…"` | WAV 字节按 UTF-8 解码成**乱码文本**，包在 `<file name="…">` 里 | 模型看不到/听不到音频 |
| `read` 工具读 .wav | 同上，乱码文本作为 tool result | 同上 |
| RPC `prompt.images` / SDK `prompt({images})` / `sendUserMessage` | 只接受 `ImageContent`，经 `processImage()` 归一化 | 音频被拒：`[Image omitted: …]` |

**桥接路径（bridge，非 native）**：

1. Agent 调用工具 `audio_attach {"path": "fixture-a.wav"}`（受控根目录内、≤4 MiB、
   magic-byte 命中 wav/mp3/ogg/flac/m4a 白名单，否则拒绝）；
2. extension 在**下一次模型请求**注入一条 user 消息：文本标注 + 一个内容块
   `{type:"image", mimeType:"audio/wav", data:<base64 原始字节>}`
   （Pi 没有 audio 内容块，`type:"image"` 只是载体标注，**MIME 与字节是音频**）；
3. Pi 自带 Google provider 转换器把它变成 Gemini API **原生音频**请求形状：

```json
{ "role": "user", "parts": [
  { "text": "[audio-bridge] attached audio \"fixture-a.wav\" (audio/wav, 24044 bytes, …)" },
  { "inlineData": { "mimeType": "audio/wav", "data": "<base64 音频字节>" } } ] }
```

即：**Gemini API 原生支持音频（inlineData）**；Pi 层不支持，桥接把两者接起来。

## 7. 费用 / 上传 / 隐私边界

- 真实调用共 **2 次运行**（native 1 + bridge 1），输入为 ~24 KB WAV + 短提示，费用可忽略；
  离线测试与探针 **0 次**调用。
- 音频以请求内联字节（`inlineData`）上传给 Google API 做分析（本任务音频、允许上传）；
  **不使用 Gemini Files API**，因此**没有服务端临时文件、无需删除**。
- 凭据：extension 与探针均**不接触** API key；模型调用由 Pi 自身认证解析；
  证据输出经 `probe/extract_evidence.py` 脱敏（base64/token/绝对路径）。
- 本地清理：`python probe/clean_fixtures.py --yes`（删除 `fixtures/`、`tmp/`）。
- 公开仓库不含：音频二进制、完整会话、key、个人绝对路径。

## 8. bridge 限制（下游 workspace 必读）

- **一次性注入**：每个附件只注入紧随其后的一次模型请求；同一 run 后续请求不重发字节。
- **类型标注**：内容块 `type:"image"` 承载 audio MIME；按块类型（而非 MIME）判断的消费者会误判。
- **不做转码**：按 sniff 到的容器 MIME 原样上传；模型对编码细节的感知不保证。
- **能力边界**：bridge 只证明“Gemini 能听 + Pi 管道能把字节送进去”；**不**把 Pi 变成原生多模态，
  下游不应据此声称 Pi 原生支持音频。

## 9. 给下游 workspace 的稳定最小接口（issue #31/#32 用）

```bash
pi --extension ./bridge/audio-bridge.ts --tools audio_attach \
   --provider google --model gemini-3.8-flash ...
```

- 工具契约：`audio_attach(path: string)` → 返回 JSON 文本
  `{attached, mime, bytes, sha256, queueDepth, note}`（只含文件名，无绝对路径）。
- 环境变量：`PI_AUDIO_BRIDGE_ROOT`（受控根目录，默认 `agentic/audio-probe/fixtures`）、
  `PI_AUDIO_BRIDGE_MAX_BYTES`（默认 4 MiB）。
- 行为约定：工具只负责校验+排队；真正的音频注入发生在 `context` 事件，一次性、不落 session。
- 该接口在本 PR 内冻结；变更需在 reports/ 记录。

## 10. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `Error: File not found: …fixture-a.wav Use the read tool…` | `@file` 与消息写成了同一个 argv | `-- "@file.wav" "消息"` 分开两个参数 |
| `audio_attach failed: path escapes the controlled audio root` | 路径穿越/越界 | 用受控根内的相对路径 |
| `audio_attach failed: file is not a recognized audio container` | 非白名单格式/伪装文件 | 仅 wav/mp3/ogg/flac/m4a |
| `Pi installation not found`（node 探针） | 未找到 pi 安装 | 设 `PI_INSTALL_DIR` 指向 `@earendil-works/pi-coding-agent` |
| `pi exited non-zero; see tmp/*-stderr.txt` | 认证/网络/模型问题 | 看 stderr 留证并报 blocked，**不要**换模型 |

## 11. 证据索引

- 环境与命令：[`reports/00-environment.md`](reports/00-environment.md)
- 能力矩阵（Gemini API 原生 / Pi CLI / read / 工具结果 / RPC / SDK / bridge）：[`reports/10-capability-matrix.md`](reports/10-capability-matrix.md)
- 原生源码+离线证据：[`reports/20-native-source-audit.md`](reports/20-native-source-audit.md)
- 真实原生失败运行（REAL）：[`reports/30-real-native-run.md`](reports/30-real-native-run.md)
- 真实 bridge 验证运行（REAL）：[`reports/40-real-bridge-run.md`](reports/40-real-bridge-run.md)
- 结论、限制、blocked 记录：[`reports/50-conclusions-and-limits.md`](reports/50-conclusions-and-limits.md)
