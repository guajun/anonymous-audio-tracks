# 20 · 原生不支持：源码审计 + 离线证据（issue #30）

标签：**OFFLINE**（读已安装 Pi 0.87.1 文档/源码，并用 Pi 自带模块真实执行；无网络、无凭据）。
Pi 安装位置以下记 `<HOME>/.pi/agent/install/releases/0.87.1/node_modules/@earendil-works/`。

## 20.1 文档层（Pi docs，全量检索）

- `pi-coding-agent/docs/` 全目录 `grep -ri audio`：**0 命中**。文档从未声明音频输入。
- `docs/cli.md`：`@path` = “Include a **text file or image** in the first prompt”；
  `read` 工具 = “Read **text files and supported images**”。
- `docs/rpc-commands.md`：`prompt`/`steer`/`follow_up` 的 `images` 仅 `ImageContent`
  （`{"type":"image","data":<base64>,"mimeType":"image/png"}`）。
- `docs/message-types.md`：`UserMessage.content: string | (TextContent | ImageContent)[]`；
  `ToolResultMessage.content: (TextContent | ImageContent)[]`；`CustomMessage` 同——**无 AudioContent**。

## 20.2 源码层

| 位置 | 事实 |
|---|---|
| `pi-ai/dist/types.d.ts` | 内容块联合只有 `TextContent \| ImageContent`；`audio` 仅出现在**计价字段**（`cost.audio`） |
| `pi-ai/dist/providers/data/google.json` | `gemini-3.8-flash`: `"input": ["text","image"]`，`inputLimits.images.resize`——**模型目录未声明 audio** |
| `pi-coding-agent/dist/utils/image-process.js` | 图片归一化只认 png/jpeg/gif/webp，其余尝试转 PNG；转不了 → `ok:false` |
| `pi-coding-agent/dist/core/agent-session.js` `_normalizePromptImages` | RPC/SDK/`sendUserMessage` 图片入口统一走 `processImage`，失败仅留文字提示、图片丢弃 |
| `pi-coding-agent/dist/cli/file-processor.js` | `@file`：`detectSupportedImageMimeTypeFromFile` 为 null → **UTF-8 文本分支** |
| `pi-coding-agent/dist/core/tools/read.js` | 非图片文件：`buffer.toString("utf-8")` → 文本 |
| `pi-ai/dist/api/google-shared.js` `convertMessages` | 非 text 块 → `inlineData{mimeType, data}`（user parts 与 toolResult `functionResponse.parts` 均如此）——**桥接可用性的根因** |
| `pi-coding-agent/dist/utils/tool-result-images.js` | 工具结果图片块 `processImage` 失败时**原样保留**（工具结果桥接通道的根因） |

## 20.3 离线执行 Pi 真实代码（OFFLINE）

### A. 原生输入路径必然拒绝音频 — `node probe/native_paths_probe.mjs`

```
file              : fixtures/fixture-a.wav (24044 bytes)
processImage      : ok=false message="[Image omitted: could not be converted to a supported inline image format.]"
  -> audio rejected: every documented input path (@file, read, RPC/SDK prompt images) hits this
image sniff       : {"fromBuffer":null,"fromFile":null,"interpretation":"@file treats the .wav as a TEXT file (no image detected)"}
@file text branch : {"decodedAs":"utf-8","previewFirst120":"RIFF·]··WAVEfmt ········@····>······data·]········X(·.L.·)··A·x···F·m·A···<···c········&f-·.i*%!··i·X·······8······!·9\n·", ...}
```

→ `@file` 与 `read` 把 WAV **字节当 UTF-8 文本**送给模型（乱码）——这正是“把文件名/文本当音频”的陷阱，
判定为 **unsupported**，不是音频输入。

### B. 桥接 payload 形状（MIME + 字节确实进请求） — `node probe/payload_shape.mjs`

使用 Pi 自带 `pi-ai/dist/api/google-shared.js` 的 `convertMessages`（与真实请求同一转换器）：

```
model             : google/gemini-3.8-flash (api=google-generative-ai, input=["text","image"])

--- case: user ---（bridge 注入形状）
[ { "role": "user", "parts": [
      { "text": "What do you hear?" },
      { "inlineData": { "mimeType": "audio/wav",
                        "data": "<base64 redacted: 704 chars, sha256 2ece270cbdf7735e…>" } } ] } ]

--- case: tool ---（工具结果形状，Gemini>=3 嵌进 functionResponse.parts）
[ ... { "role": "user", "parts": [ { "functionResponse": { "name": "audio_attach",
      "response": { "output": "(see attached image)" },
      "parts": [ { "inlineData": { "mimeType": "audio/wav",
                                  "data": "<base64 redacted: 704 chars, sha256 2ece270cbdf7735e…>" } } ],
      "id": "t1" } } ] } ]
```

→ 请求 payload **包含 MIME + 音频字节**（`inlineData`，无 file URI、无 Files API 上传）。
证据输出默认脱敏 base64（`--show-data` 才显示原文，且永不进报告）。

## 20.4 小结

- Pi 消息/工具/CLI/RPC/SDK 全链路**不存在音频内容块**；模型目录不声明 audio 模态 → **native unsupported**。
- 但 Pi 的 Google 转换器会把任意非文本块原样变成 `inlineData` → 为**桥接**留了一条合规缝隙
  （用 `type:"image"` 载体 + audio MIME，明确标注 workaround）。
