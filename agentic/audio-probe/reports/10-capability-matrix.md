# 10 · 能力矩阵：音频作为“模型输入”（issue #30）

问题定义：**Pi 能否把声音作为模型输入**（不是“Gemini API 能不能听音频”）。
每行标注证据类型：SOURCE（已安装 Pi 源码/文档）、OFFLINE（离线执行 Pi 真实代码）、
REAL（真实 Gemini 调用）。**Pi 原生音频 = 不支持**；bridge = 可用但不是 native。

| # | 通道 | payload 是否含 MIME + 音频字节 / file URI | 判定 | 证据 |
|---|---|---|---|---|
| 1 | **Gemini API 原生**（generativelanguage `inlineData`） | ✅ `inlineData{mimeType:"audio/wav", data:<base64>}`（字节内联；未用 Files API/file URI） | **API 原生支持** | OFFLINE（`probe/payload_shape.mjs` 用 Pi 自带转换器产出该形状）+ REAL（bridge 运行中模型对音高方向的判断与构造真值一致，支持性证据，见 40） |
| 2 | **Pi CLI `@file`**（`pi @x.wav "…"`） | ❌ WAV 字节按 UTF-8 解成**乱码文本**包进 `<file name="…">` | **unsupported**（文本冒充音频） | SOURCE（`cli/file-processor.js`：非图片走文本分支）+ OFFLINE（`native_paths_probe.mjs`）+ REAL（30：模型自述无法感知音频） |
| 3 | **Pi `read` 工具** | ❌ 同上，乱码文本作为 tool result | **unsupported** | SOURCE（`core/tools/read.js` 只支持文本/图片）+ REAL（30） |
| 4 | **Pi 工具结果（tool result 图片块）** | ⚠️ 转换后 `functionResponse.parts:[{inlineData:{mimeType:"audio/wav",…}}]`（块 type 仍是 `image`） | **payload 级可行**（未真实调用验证 API 是否接受 functionResponse 内音频） | OFFLINE（`payload_shape.mjs` case `tool`；`utils/tool-result-images.js` 对 `processImage` 失败的块原样保留） |
| 5 | **Pi RPC**（`prompt`/`steer`/`follow_up` 的 `images`） | ❌ `images` 只接受 `ImageContent`，进 `_normalizePromptImages`→`processImage` 拒绝音频：`[Image omitted: could not be converted to a supported inline image format.]` | **unsupported** | SOURCE（`rpc-commands.md` + `core/agent-session.js`）+ OFFLINE（同一 `processImage` 实测） |
| 6 | **Pi SDK**（`session.prompt({images})` / `sendUserMessage`） | ❌ 同一 `processImage` 关卡；消息类型无 AudioContent | **unsupported** | SOURCE（`sdk.md`、`message-types.md`、`core/extensions/types.d.ts`）+ OFFLINE |
| 7 | **扩展桥接（本目录 bridge）** | ✅ Agent 触发 `audio_attach` → `context` 注入 `type:"image"` 块但 `mimeType=audio/wav`、`data`=原始字节 → Pi 转换器产出 inlineData | **bridge 可用**（明确非 native） | OFFLINE（guard/extension 单测 + payload 形状）+ REAL（40：真实 Gemini 桥接运行，支持性证据） |

## 结论行

- **Pi 原生（#2/#3/#5/#6）：不支持**——没有 AudioContent 类型、模型目录无 audio 模态、
  所有文档化输入入口把音频降级为文本或直接丢弃。
- **Gemini API 原生（#1）：支持**——请求形状就是 `inlineData`（MIME + 字节）。
- **bridge（#7）：可用**——把 #1 与 Pi 管道接起来；类型标注借用 `image` 块（诚实记录的 workaround）。
- **#4**：工具结果通道在 payload 层可行但未真实验证，下游优先用 #7 的 user 注入通道。

> 术语纪律：本仓库文档中 "native" 仅指 Pi 自身输入能力；bridge 一律标注 bridge/workaround，
> 不把 bridge 称为 native，也不把文件名/乱码文本当作音频输入成功。
