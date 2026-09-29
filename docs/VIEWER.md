# 本地音频与中心活动检查页（issue #10）

`viewer/` 是一个**纯静态、本地运行**的检查页：用户选择一首音频和一份协议
`trajectory.json`（协议 0.1.0，见 [SCHEMAS.md](SCHEMAS.md) §7），页面把每条匿名
轨迹的**中心活动概率**与音频播放对齐显示，用于人工检查漏检、误检和换轨。

它显示的是窗口中心时刻的活动概率，**不是** note-on 起音检测、不是乐器识别、
不是谱面编辑器；`track_id` 是关联出的匿名身份，不映射任何乐器名称。

## 1. 目录与运行方式

```text
viewer/
  index.html          页面骨架（只引用本地 app.js / styles.css）
  styles.css
  app.js              浏览器入口：文件选择、状态、渲染循环
  js/
    protocol.js       trajectory.json 读取与校验（镜像 Python 契约子集）
    timeline.js       本地/原曲绝对时间换算、插值、阈值区间、循环、窗口
    snapshot.js       单帧只读快照：所有轨迹共用同一个时钟
    session.js        纯状态机：加载/切换/错误时的清理语义、循环草稿
    media.js          blob: URL 生命周期（切换/清空即 revoke）
    overview.js       长歌整曲包络降采样（可缓存的纯函数）
    draw.js           canvas 绘制（窗口 + 整曲概览）
  tools/
    make-fixture-audio.mjs   本地合成 WAV fixture（不入 Git）
    browser-check.mjs        无依赖的 Chrome/Edge CDP 真机检查
tests/viewer/
  *.test.js          Node 内建测试（node:test），无 npm 依赖
  fixtures/*.json    纯文本协议 fixture（提交）
  fixtures/generated/  生成的音频/报告/截图（其 .gitignore 忽略内容）
```

启动静态服务（Python 只用于本地静态服务，页面本身不依赖 Python）：

```sh
uv run python -m http.server 8123 --directory viewer
# 浏览器打开 http://127.0.0.1:8123/
```

`file://` 直接打开不行：ES module 会被浏览器 CORS 规则拦截，必须经本地 HTTP
服务。页面没有任何远程请求、CDN、字体或遥测；音频只挂在 `blob:` URL 上。

运行前端测试（Node ≥ 22，本仓库记录使用 Node v24）：

```sh
node --test "tests/viewer/**/*.test.js"
```

现有 Python 测试不受影响，仍为：

```sh
uv sync --locked
uv run pytest
```

## 2. 操作说明

1. **加载音频**：本地文件选择；浏览器异步读取 metadata 期间显示“等待 metadata”，
   此时不会误判为时长错误。加载失败（非音频、解码失败）会暂停播放、清除
   blob URL 并给出明确错误；已加载的轨迹保留以便排查。
2. **加载 trajectory JSON**：选择新文件的瞬间即进入 pending：暂停播放、清空旧曲线/旧循环，使旧轨迹不再充当当前同步数据；随后只有**最新一次**读取结果（成功或失败）可以提交文件名、provenance 或错误，迟到/过期的旧读取被丢弃（`trajectoryLoadGeneration` 代次令牌）。校验失败时面板列出每条字段级问题。成功时重置窗口与来源开关。
3. **播放/暂停/定位**：滑块、窗口画布点击、整曲概览点击都直接写
   `audio.currentTime`；时间读数分“本地”（播放器秒）与“原曲绝对”（本地 +
   `track_start_seconds`）。
4. **慢放**：0.25× / 0.5× / 0.75× / 1× / 1.5× / 2×。
5. **选段循环**：播放中点击“设 A 点/设 B 点”记录当前时间；端点必须是有限秒、
   在音频时长内、A < B 且间隔 ≥ 0.05s，否则不启用并显示原因。循环通过播放循环
   对 `audio.currentTime` 取模回绕，变速时同样生效。
6. **长歌窗口导航**：局部窗口可选 5/10/20/30/60/120s；上一窗/下一窗/开头/结尾
   按钮按窗口步进并关闭“跟随播放”，勾选“跟随播放”后窗口随播放头翻页。
   整曲概览始终显示全曲包络、当前窗口矩形与播放游标，可点击定位。
7. **来源开关**：每条轨迹可单独显示/隐藏；隐藏只影响绘制，不影响数据与时钟。

## 3. 时间轴与校验语义

- 协议时间一律是**原曲绝对秒**；`audio.currentTime` 是**相对音频起点**的秒。
  换算为 `local = absolute - track_start_seconds`，页面从不把协议时间直接当作
  播放器秒数。`trajectory.clip.json`（`track_start_seconds = 12.0`）在本地
  0.24s 处对应绝对 12.24s、p=0.7。
- 校验（`viewer/js/protocol.js` 镜像 `src/aat/contracts/documents.py`）：
  `schema_version` 必须为 `0.1.0` 且 `kind = trajectory`；`audio`、
  `provenance`、`tracks` 必填；`provenance.run_id` 非空、`data_kind ∈
  {model, annotation, mock}`；`track_id` 唯一；`center_times` 有限、非负、严格
  递增且落在音频范围内；`activity` 与 `center_times` 等长且在 `[0, 1]`；
  `confidence`/`slot_indices` 可选但形状与范围一致。未知额外字段按协议容忍。
- 时长兼容性：本页要求载入**同一分析片段**，因此比较已加载音频时长与轨迹
  声明 `audio.duration_seconds` 的**双向差值**，容差 ±0.01s
  （`COMPATIBILITY_TOLERANCE_SECONDS`，用于吸收短文件 codec padding 等差异）；
  `track_start_seconds` 只参与绝对时间换算，**不参与长度比较**。差值超出容差为
  `mismatch`：立即暂停并禁用播放/定位/循环（按钮禁用；`window.__aatViewer` 的
  `play`/`setTime`/`setLoop` 同样拒绝，rAF 内的守卫也会暂停任何绕过路径），
  不自动接受整歌或更长音频；metadata 未就绪返回 `pending`，不误报为时长错误。
- 读取代次：`beginTrajectoryFileLoad` 先清空旧轨迹并置 pending；只有与
  `trajectoryLoadGeneration` 相等的最新读取才能调用 `loadTrajectoryText` 或
  `failTrajectoryLoad`。旧读取无论成功或失败都不会覆盖新文件名、数据或错误。
- `data_kind` 在 provenance 面板显著区分，`mock` 另有“不能当作模型结果”的警告。
- 所有用户 JSON 文本只经 `textContent`/`fillText` 渲染；测试会静态扫描
  `innerHTML`、`insertAdjacentHTML`、`document.write` 等 HTML sink。

## 4. Fixture 与生成

提交的纯文本 fixture（`tests/viewer/fixtures/`）：

| 文件 | 内容 |
|---|---|
| `trajectory.known-times.json` | 4s，3 条轨迹；trk-a 已知插值点与阈值区间 `[0.3125, 1.25]`、`[2.875, 3.0833]`；trk-b 全 0 的静音保持身份；trk-c 仅 `[1, 2]s` |
| `trajectory.clip.json` | `track_start_seconds = 12.0`，验证非零原曲起点换算 |
| `trajectory.empty.json` | `duration = 0`、`tracks = []` 空边界 |
| `trajectory.hostile-strings.json` | `track_id`/`notes` 含 `<img onerror>`、`<script>` 等，验证按文本渲染 |

这四份 fixture 同时通过合并后的 Python 契约实现（一次性交叉核对）：

```sh
uv run python -c "
import json
from pathlib import Path
from aat.contracts.documents import Trajectory
for path in sorted(Path('tests/viewer/fixtures').glob('*.json')):
    document = Trajectory.from_json_dict(json.loads(path.read_text(encoding='utf-8')))
    print('ok', path.name, 'tracks=', len(document.tracks))
"
# ok trajectory.clip.json tracks= 1
# ok trajectory.empty.json tracks= 0
# ok trajectory.hostile-strings.json tracks= 1
# ok trajectory.known-times.json tracks= 3
```

本地音频 fixture 由脚本合成，输出到被忽略的 `tests/viewer/fixtures/generated/`：

```sh
node viewer/tools/make-fixture-audio.mjs                    # 4s known-times.wav（44.1kHz）
node viewer/tools/make-fixture-audio.mjs --pattern long     # 600s long-600s.wav（8kHz）
node viewer/tools/make-fixture-audio.mjs --pattern tone --seconds 10 --out /tmp/tone.wav
```

`known-times.wav` 的两段音与 `trajectory.known-times.json` 的阈值区间对齐
（0.3125–1.25s、2.875–3.0833s），便于人耳核对；音频文件本身不入 Git。
浏览器检查还会即时合成与 `trajectory.clip.json` 配对的 0.5s 音频，以及
0.48s / 0.52s 两种不匹配长度，用于验证同片段策略。

## 5. 实际浏览器检查记录

检查脚本无 npm 依赖：Node 内建 `fetch`/`WebSocket` 连接本机 headless
Chrome/Edge 的 DevTools Protocol，用 `DOM.setFileInputFiles` 走真实
`<input type=file>` 路径，读取页面里 UI 所用的同一份状态。

```sh
node viewer/tools/browser-check.mjs \
  --report tests/viewer/fixtures/generated/browser-check.json \
  --screenshot tests/viewer/fixtures/generated/viewer-check.png
# 也可用 --chrome <path> 或环境变量 CHROME_PATH 指定浏览器
```

本 session 记录（2026-09-29 复审修复后的重跑，Windows，Chrome 150.0.7871.101
headless，`--autoplay-policy=no-user-gesture-required --mute-audio`）：

```text
PASS  browser: headless Chrome/Edge launched — {"browser":"Chrome/150.0.7871.101","page":"/viewer/index.html"}
PASS  trajectory 0.1.0 fixture validates and lists 3 anonymous tracks — {"dataKind":"model","runId":"fixture-known-times-01"}
PASS  audio file loads via local blob: URL (no upload) — {"duration":4}
PASS  shared clock: at local 0.75s absolute time stays 0.75s (track_start=0) — {"localTime":0.75,"absoluteTime":0.75}
PASS  interpolated activity at 0.75s (0.5s:0.8 -> 1.0s:0.9) equals 0.85 — {"value":0.8500000000000001}
PASS  threshold 0.5 marks the point active
PASS  silent-identity track reports 0.0 and stays present — {"value":0}
PASS  third track starts only at 1.0s, so it has no data at 0.75s — {"value":null}
PASS  second track is active at 1.0s
PASS  threshold intervals: [0.3125, 1.25] and [2.875, 3.08333] — [[0.3125,1.25],[2.875,3.0833333333333335]]
PASS  adjustable threshold 0.95 flips 0.9 activity to inactive
PASS  seek slider drag updates the shared audio clock — {"currentTime":3.4}
PASS  play at 1x tracks real wall-clock time — {"advance":0.705223,"wallElapsed":0.706}
PASS  pause freezes the shared clock
PASS  slow motion 0.5x advances at half wall-clock rate — {"slowAdvance":0.35494000000000003,"slowWallElapsed":0.711,"playbackRate":0.5}
PASS  valid loop A=1.0 B=1.5 activates — {"start":1,"end":1.5}
PASS  playback wraps inside the loop and never runs past B — {"currentTime":1.034713}
PASS  loop endpoint validation rejects A >= B — "循环区间至少需要 0.05s（当前 A=2s，B=1s）。"
PASS  loop endpoint validation rejects intervals shorter than 0.05s — "循环区间至少需要 0.05s（当前 A=0.5s，B=0.52s）。"
PASS  loop endpoint validation rejects B beyond the audio duration — "循环端点 B=99s 超出音频时长 4s。"
PASS  loop clears cleanly
PASS  4s audio vs declared 0.5s clip is a mismatch (old code accepted it as "covering") — {"message":"轨迹声明被分析音频时长 0.5s，已加载音频 4.000s （长于 3.500s，超出 ±0.01s 容差）；本页要求载入同一分析片段，不自动接受整歌或更长的音频。"}
PASS  mismatch disables play/seek/loop controls — {"play":true,"seek":true,"loopA":true}
PASS  mismatch refuses playback through the test hook too — {"ok":false,"message":"同步播放被禁止：音频与轨迹时长不匹配或未就绪。"}
PASS  mismatch refuses seek through the test hook — {"beforeRefusedSeek":1.034713,"afterRefusedSeek":1.034713}
PASS  mismatch refuses loop creation — {"refused":"音频与轨迹时长不匹配或未就绪，循环被拒绝。"}
PASS  origin 12s / duration 0.5s clip matches a real 0.5s audio (ok) — {"message":"已加载音频 0.500s 与轨迹声明的被分析音频时长 0.5s 相符（容差 ±0.01s）；原曲绝对起点为 12s。"}
PASS  non-zero track_start_seconds: local 0.24s maps to absolute 12.24s — {"localTime":0.24,"absoluteTime":12.24}
PASS  absolute center time 12.24s yields p=0.7 (not the 0.24s player value) — {"value":0.7}
PASS  valid loop works with the matching 0.5s clip — {"start":0.1,"end":0.3}
PASS  matching clip is actually playable through the hook — {"currentTime":0.220215,"paused":false}
PASS  declared 0.5s vs 0.48s audio is rejected (too short) — {"message":"轨迹声明被分析音频时长 0.5s，已加载音频 0.480s （短于 0.020s，超出 ±0.01s 容差）；本页要求载入同一分析片段，不自动接受整歌或更长的音频。"}
PASS  mismatch transition pauses playback and drops the active loop — {"paused":true,"loop":null}
PASS  too-short mismatch keeps play/seek disabled — {"play":true,"seek":true}
PASS  declared 0.5s vs 0.52s audio is rejected (too long) — {"message":"轨迹声明被分析音频时长 0.5s，已加载音频 0.520s （长于 0.020s，超出 ±0.01s 容差）；本页要求载入同一分析片段，不自动接受整歌或更长的音频。"}
PASS  empty trajectory (duration 0, tracks []) renders without crashing — {"duration":0.52,"durationSource":"audio"}
PASS  invalid trajectory rejected with field-level issues — {"paths":"trajectory.tracks[\"trk-invalid\"].center_times,trajectory.tracks[\"trk-invalid\"].activity"}
PASS  load error clears old curves, old playback and shows the error panel — {"trajectoryLoaded":false,"trackListText":"当前 trajectory 没有轨迹（空数组）。","errorPanelVisible":true,"paused":true,"trackCount":0}
PASS  hostile JSON strings render as text, never as HTML — {"pwned":0,"imgs":0}
PASS  new selection enters pending immediately, pauses and drops old curves — {"pending":"race-old-slow-a.json","loaded":false,"paused":true}
PASS  a delayed read still commits when it remains the newest selection
PASS  slow old failure cannot overwrite the newer fast success — {"runId":"race-new-fast-a","fileName":"race-new-fast-a.json"}
PASS  slow old success cannot overwrite the newer fast success — {"runId":"race-new-fast-b"}
PASS  newest failure is not overwritten by an older slow success — {"errorFileName":"race-new-fail.json","loaded":false}
PASS  switching audio revokes the previous blob URL — {"revokeCount":1}
PASS  non-audio file produces a clear load error and keeps the trajectory — {"error":"浏览器不支持该音频格式或源（MEDIA_ERR_SRC_NOT_SUPPORTED）。","trackCount":3}
PASS  600s trajectory + audio loads (long-song browsing input) — {"duration":600}
PASS  window next moves the local window by one window length — 20
PASS  window next reaches 80s after four clicks — 80
PASS  window last clamps to 580s for a 600s song — 580
PASS  window first returns to 0s — 0
PASS  clicking the overview seeks the shared clock to the middle of a 600s song — {"currentTime":299.705304}

52/52 browser checks passed.
```

同一轮检查还生成了页面截图（`tests/viewer/fixtures/generated/viewer-check.png`，
未提交），确认布局、provenance（含 mock 警告）、传输区与整曲/局部画布正常绘制。

Node 单元/集成测试（74 项，含时间换算、插值、阈值区间、循环边界、窗口导航、
长歌包络、会话清理与 pending/失败状态、双向时长容差、blob URL 生命周期、
HTML sink/网络静态扫描）：

```sh
node --test "tests/viewer/**/*.test.js"
# ℹ tests 74
# ℹ pass 74
# ℹ fail 0
```

## 6. 未验证项与限制

- 只在 Chromium（Chrome 150 headless，1280×900）验证；Firefox/Safari、移动端
  布局未验证。
- headless 使用 `--mute-audio`，只验证时钟与状态，未验证实际听感。
- 时长策略固定为“载入同一分析片段 + ±0.01s 双向容差”；若以后需要「整歌音频 +
  片段轨迹」的另一个原点，必须显式增加选择项，本次不做，也不会静默接受。
- 刷新时 `beforeunload` 里的 blob URL 清理只通过源码/静态测试与代码路径确认，
  未能自动断言页面销毁后的 revoke 计数。
- 页面只消费 `trajectory.json`，不解析 `activity.npz` / `feature.npz` /
  `prediction.npz`；这些 NPZ 仍是协议的一部分，读取器留给后续任务。
- 长歌性能只在 Node 层用 180k 点合成轨迹验证包络降采样（<1s）；未用小时级真实
  推理输出做内存/解析基准。
- 本页不做 LED 映射、发音起音检测或谱面编辑；不修改既有 `rgb-score-player`。

## 7. 隐私与网络

- 音频文件只通过 `File` + `URL.createObjectURL` 在本机解码，不上传；切换文件、
  加载失败和页面卸载都会 revoke 旧 URL。
- 页面代码无 `fetch`/XHR/WebSocket/EventSource，无远程 URL；`tests/viewer/`
  的静态检查会阻止这些依赖回流。
- 提交内容不含音频、权重、私人路径或完整代理日志；生成物在忽略目录。
