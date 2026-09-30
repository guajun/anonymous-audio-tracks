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

本 run 实测用量（事件流 `usage`，**一次 Pi run = 2 个 assistant requests**，非 1 次 API 调用）：

| request | stopReason | input | output | totalTokens | Pi 记账 cost |
|---|---|---|---|---|---|
| 1 | toolUse | 21,946 | 359 | 22,305 | 0.0178 |
| 2 | stop | 43,479 | 370 | 43,849 | 0.0340 |
| 合计 | — | 65,425 | 729 | 66,154 | 0.0518 |

（乱码文本路径 token 消耗高：第 2 个 request 的 43K input 主要是 WAV 字节被当文本编码所致。）

## 30.3 判定

- **Pi 原生音频输入 = 不支持（REAL 复证）**：真实调用中模型明确表示无法感知音频；
  输送内容（30.1）证明原生入口只产生文本/乱码。
- 该运行**不**用于任何音频理解准确度声明；它只是失败证据。
- 备注：`@file` 与消息必须是**两个 argv**（`-- "@x.wav" "消息"`）；首次运行因合成一个参数报
  `File not found: …fixture-a.wav Use the read tool…`，已修正脚本并保留该 stderr 于 tmp/（属本地脚本错误，非 blocked）。
  该修正之后，`probe/run_real_*.sh` 均经 `probe/run_bounded.py` 加硬超时（默认 300s）并以退出码
  区分：0 成功 / 3 命令失败或无法启动 / 4 超时 / 5 事件流含 provider error 或 aborted（即使 Pi 退出 0）/
  6 事件流空/乱码/不完整（无成功终态 assistant 消息不计成功）；`pi …` 经 `probe/pi_launcher.py`
  解析为原生 Node+CLI 入口进程（argv 保真、超时杀到真实 Pi）；失败/超时均保留事件流与 stderr 产物；
  失败语义已用 fake pi 离线回归（`tests/test_real_run_scripts.py`，无 API 调用），
  并有实际 `pi --version` 冒烟（零 API，`tests/test_pi_launcher.py`）。
