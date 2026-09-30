# 40 · 真实运行：audio bridge 听音验证（REAL）

标签：**REAL**（真实 Gemini API 调用；模型 `google/gemini-3.8-flash`，provider `google`，
取自事件流，**未换模型**）。运行脚本：`bash probe/run_real_bridge.sh`
（`pi --extension ./bridge/audio-bridge.ts --tools audio_attach`，显式项目内加载，不装全局）。

> 重申：这是 **bridge**，不是 Pi 原生音频支持。音频经扩展注入 Pi 的请求管道，
> 由 Pi 自身认证调用 Gemini；扩展不接触凭据。

## 40.1 fixture 构造真值（by construction，`probe/make_fixture.py`）

| 文件 | 内容 | 真值 |
|---|---|---|
| `fixture-a.wav` | 16-bit PCM mono 8 kHz，3 个正弦 tone（440/660/880 Hz），各 0.30s + 0.20s 静音 | 3 个事件，**音高上升** |
| `fixture-b.wav` | 同上但 880/660/440 Hz | 3 个事件，**音高下降** |

文件名不泄露方向；提示词不透露答案。sha256（工具返回值实录）：
`fixture-a.wav=22a5960d60f2afc4…`、`fixture-b.wav=bf9236198f3e4ff4…`。

## 40.2 实际调用与输送内容（脱敏）

Agent 行为（事件流实录）：

```
ASSISTANT TOOLCALL: audio_attach {"path": "fixture-a.wav"}
ASSISTANT TOOLCALL: audio_attach {"path": "fixture-b.wav"}
TOOL: audio_attach → {"attached":"fixture-a.wav","mime":"audio/wav","bytes":24044,
                      "sha256":"22a5960d60f2afc4…","queueDepth":1, …}
TOOL: audio_attach → {"attached":"fixture-b.wav","mime":"audio/wav","bytes":24044,
                      "sha256":"bf9236198f3e4ff4…","queueDepth":2, …}
```

下一次模型请求由 `context` 注入（形状由 OFFLINE `probe/payload_shape.mjs` 用 Pi 自带转换器复核）：

```
{ "role": "user", "parts": [
  { "text": "[audio-bridge] attached audio \"fixture-a.wav\" (audio/wav, 24044 bytes, sha256 22a5960d60f2afc4…). The audio data follows as inline content." },
  { "inlineData": { "mimeType": "audio/wav", "data": <base64 原始 WAV 字节，24044 B，未入报告> } } ] }
```

→ payload **含 MIME + 音频字节**（inlineData 内联；无 file URI、无 Files API 上传、无服务端临时文件）。

## 40.3 模型答案（verbatim）

```
### `fixture-a.wav`
1. **Distinct sound events:** 1 (a single continuous tone/sweep)
2. **Pitch direction:** Goes **up** (rising pitch from low to high)

---

### `fixture-b.wav`
1. **Distinct sound events:** 1 (a single continuous tone/sweep)
2. **Pitch direction:** Goes **down** (falling pitch from high to low)
```

（`stopReason: stop`，`model: gemini-3.8-flash`，`provider: google`）

## 40.4 对照与判定

| 项目 | 构造真值 | 模型回答 | 判定 |
|---|---|---|---|
| fixture-a 音高方向 | 上升 | up | ✅ 一致 |
| fixture-b 音高方向 | 下降 | down | ✅ 一致 |
| 事件计数 | 3 个 tone | 1（听成连续 sweep） | ❌ 不一致 |

- 两个文件方向**不同**且文件名/提示不泄露，若模型只看到文本或文件名不可能双双答对方向
  → **模型确实听到了音频字节**；结合 OFFLINE payload 证据，可判定 audio 经 Pi 管道送达 Gemini。
- **不主张**时间戳/事件计数级真值准确度（事件计数已实测不一致）；仅“音高方向”这种
  定性判断作为听音证据。音乐/复音场景的能力上限不在此 issue 范围。
- session 隐私核查（grep 实测）：bridge session JSONL 中**无** base64 音频（注入是请求级、
  Pi 恢复自身状态）；工具结果只含文件名/sha256 等元数据。
