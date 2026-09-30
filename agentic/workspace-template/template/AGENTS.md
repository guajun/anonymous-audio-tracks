# Pi 音频分析 workspace（issue #31 模板部署）

这是一个**研究 workspace**（不是训练项目）。它是一个真正的 Pi 项目：项目设置、skill、桥接扩展、
音频输入与输出目录都在本目录内。上游问题：`guajun/anonymous-audio-tracks#28`（父）、
`#27`（研究日志）。

## 0. 角色与模型（不要改）

- 当前模型由 `.pi/settings.json` 固定为 **google/gemini-3.8-flash**（研究模型，issue #30 冻结）。
  **不要切换模型、不要换 provider**：音频桥接只对该模型生效，其他模型会得到 `E_AUDIO_MODEL`。
- 本 workspace 的 Agent 任务：分析本地音频（分离/听音/结构化结果），不做训练、不下载权重。
- GPU 真实分离是 issue #33 的事；本 workspace 默认只做 check-environment / dry-run 等无 GPU 步骤。

## 1. 目录布局

| 路径 | 用途 | 是否提交 |
|---|---|---|
| `audio/inputs/` | 输入音频 + `manifest.json`（sha256/时长/采样率） | 否（ignored） |
| `outputs/` | 结果 `outputs/result.json`、运行产物 `outputs/runs/<run-id>/` | 否（ignored） |
| `local/` | 本机配置 `local/config.json`、运行证据 | 否（ignored） |
| `sessions/` | Pi 会话（sessionDir 已指向这里） | 否（ignored） |
| `.pi/settings.json` | 项目设置（模型固定、扩展、sessionDir） | 模板 |
| `.pi/extensions/audio-bridge.ts` | 音频桥接扩展（issue #30 冻结接口的 pin 副本） | 模板 |
| `.pi/skills/sam-audio/` | `gh skill install --pin` 安装的 SAM skill/CLI | 否（ignored） |

## 2. 音频输入：桥接，不是 Pi 原生

**Pi 0.87.1 没有原生音频输入**（实测结论见 `agentic/audio-probe/README.md`）。音频只能通过
**明确标注的桥接扩展**进入模型：

1. 用 `audio_attach` 工具：`{"path": "<相对 audio/inputs 的文件名>"}`。
   受控根目录由启动脚本设置的 `PI_AUDIO_BRIDGE_ROOT`（= `audio/inputs`）决定。
2. 成功返回 JSON：`{attached, mime, bytes, sha256, queueDepth, queuedBytes, note}`（无音频字节）。
3. 音频在**下一次模型请求**一次性注入（inlineData，Gemini API 原生音频形状）；之后不重发。
4. 失败抛错，稳定错误码 `E_AUDIO_ARGS / E_AUDIO_PATH / E_AUDIO_NOT_FOUND / E_AUDIO_NOT_FILE /
   E_AUDIO_SIZE / E_AUDIO_QUEUE / E_AUDIO_MIME / E_AUDIO_READ / E_AUDIO_MODEL`。
5. **不要**用 `@file` 或 `read` 工具读 .wav：那会把字节当 UTF-8 文本（乱码），模型听不到。

限制：单文件 ≤4 MiB、排队 ≤8 MiB；不转码；按 MIME 原样上传；音频字节不落盘、不进会话转录。

## 3. SAM skill / CLI

- skill 名 `sam-audio`（`.pi/skills/sam-audio/`），CLI：`python .pi/skills/sam-audio/scripts/audio_toolbox.py`。
- SAM checkout 路径与解释器在 `local/config.json` 的 `sam_root` / `sam_python`；调用时传
  `--sam-root` / `--python`（或设置 `SAM_AUDIO_ROOT`）。**不要**自动下载权重/联网（wrapper 强制
  `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`）。
- 推荐顺序：`--help` → `sam check-environment` → `sam separate --dry-run`（读时长/校验 anchors/FFmpeg/
  模型路径，**不加载模型、不用 GPU**）→（issue #33 才做）真实分离。
- 真实分离产物写 `outputs/runs/<run-id>/`（`target.wav`、`residual.wav`、`request.json`、`report.json`）。

## 4. 结果输出（接口约定）

- 最终结构化结果约定写 **`outputs/result.json`**；其 schema 由 **issue #32** 定义并合并后对接。
  schema 尚不存在时：**不要自造 schema 并宣称兼容**，把过程性结果写 `outputs/notes.md` 与
  `outputs/runs/<run-id>/`，并在文本中注明“等待 #32 schema”。
- 任何输出不得包含 API key、音频字节、模型权重或个人绝对路径。

## 5. 凭据与隐私

- API key 由 Pi 自身认证解析；**不读取、不打印、不写入任何文件**。
- 本 workspace 在本地忽略目录；`local/`、`audio/`、`outputs/`、`sessions/` 都被 `.gitignore`。
- 本地 git 仓库只为 `gh skill` 的 project 安装与 ignore 防护存在；**不要 commit 敏感内容、不要 push**。

## 6. 失败语义（要求可读错误，不是堆栈）

遇到缺 key / 缺权重 / 缺 FFmpeg / 音频不可读时，给出**可读的错误 + 修复建议**（错误码/缺失项/下一步），
不要抛裸 traceback、不要静默降级、不要换模型、不要无限重试。自检命令：

```sh
python agentic/workspace-template/doctor.py --workspace <本目录>        # 离线诊断
python agentic/workspace-template/doctor.py --workspace <本目录> --deep # + 真实 check-environment/dry-run
```
