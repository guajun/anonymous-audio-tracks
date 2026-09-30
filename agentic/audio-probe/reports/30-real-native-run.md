# 30 · 真实运行：原生 `@file` + `read` 工具（REAL，失败证据）

标签：**REAL**（真实 Gemini API 调用；模型 `google/gemini-3.8-flash`，provider `google`，
取自事件流 `message_end.message.model/provider`，**未换模型**）。运行脚本：
`bash probe/run_real_native.sh`（显式限定 `--tools read`、禁用扩展/skills/prompt templates/themes/context files）。

原始事件流：`tmp/native-events.jsonl`（gitignored，本地保留）；下方摘录经
`probe/extract_evidence.py` 脱敏（`<REPO>`=仓库根、`·`=不可打印字节、无 base64/key）。

## 30.1 实际输送内容（不是音频！）

`pi -- "@fixtures/fixture-a.wav" "…"` 的 user 消息（`message_end.message.user`）：

```
<file name="<REPO>\audio-probe\fixtures\fixture-a.wav">
RIFF·]··WAVEfmt ········@····>······data·]········X(·.L.·)··A·x···F·m·A···<···c········&f-·.i*%!··i·X··
·····8······!·9 ···%·,·.%+)#·.·U·8·$··············$·$·8·U···.·)#·.·,·.··9 !·… [truncated]
```

`read` 工具结果（`tool_execution_end.result`）同样是 WAV 字节按 UTF-8 解码的乱码文本：

```
RIFF·]··WAVEfmt ········@····>······data·]········X(·.L.·)··A·x···F·m·A···<···c·· … [truncated]
```

→ 模型收到的是**二进制乱码文本**与文件名文本；**没有任何音频字节 / MIME / file URI** 进入模型输入。

## 30.2 模型答案（verbatim）

```
I cannot actually perceive audio from this file.

The `read` tool only returns the raw file contents decoded as text (showing the binary WAV header
and sample data such as `RIFF...WAVEfmt...`), without audio playback or audio perception capabilities.
Because I cannot listen to the audio stream, I cannot determine how many distinct sound events occur
or whether the pitch goes up, down, or stays level.
```

（`stopReason: stop`，`model: gemini-3.8-flash`，`provider: google`）

## 30.3 判定

- **Pi 原生音频输入 = 不支持（REAL 复证）**：真实调用中模型明确表示无法感知音频；
  输送内容（30.1）证明原生入口只产生文本/乱码。
- 该运行**不**用于任何音频理解准确度声明；它只是失败证据。
- 备注：`@file` 与消息必须是**两个 argv**（`-- "@x.wav" "消息"`）；首次运行因合成一个参数报
  `File not found: …fixture-a.wav Use the read tool…`，已修正脚本并保留该 stderr 于 tmp/（属本地脚本错误，非 blocked）。
