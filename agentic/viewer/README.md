# agentic/viewer — 高性能本地网页查看器（issue #34）

读取 `agentic-audio-tracks/v1`（issue #32 冻结协议）分析结果 JSON 与本地音乐文件，
**纵轴逐行显示乐器条目（label + 稳定 id，同类不同 id 绝不合并），横轴按音频秒显示 onset 事件与波形**，
支持 BPM 网格调节、滚轮锚点缩放、平移、播放/暂停/seek/playhead。**纯本地：file input 加载，不上传任何服务、无 CDN。**

- 数据契约：`agentic/schema/`（issue #32 冻结）。本目录 `schema/` 是字节一致的构建副本（`schema/SOURCE.json` 记录来源 sha256，测试校对）。
- demo / stress 数据全部是**自生成 mock**（`provenance.source=mock`），**不是研究结果**，不代表任何模型能力。
- 旧 `viewer/`（根目录）未做任何改动；本目录自包含（页面 + 自己的 tests/tools/docs）。

## 目录结构

```
agentic/viewer/
  index.html            页面（CSP：default-src 'none'，只加载同源资产）
  styles.css
  app.js                UI 装配（textContent/Canvas 文本，无 HTML sink）
  js/
    strict-json.js       严格 JSON：重复键/NaN/1e999/深度 64/数值政策（对齐 loader.py）
    structure.js         结构校验（Draft 2020-12 便携子集，对齐 structure.py）
    semantic.js          跨字段语义（E_* 冻结错误码，对齐 semantic.py）
    protocol.js          校验流水线 parse→version→structure→semantic（对齐 pipeline.py）
    model.js             视图模型（typed arrays；BPM 状态/beat grid；onset 永不被改动）
    timeline.js          时间窗数学（锚点缩放/平移/fit/约束、二分查找可视切片）
    peaks.js             波形 min/max 金字塔 + LOD 选择
    render.js            Canvas 绘制（viewport culling + 事件密度 LOD）
    transport.js         本地音频（文件名匹配/SHA-256/解码/ObjectURL/AudioContext/Worker 清理）
    validate-worker.js   大 JSON 校验 Web Worker
    peaks-worker.js      波形金字塔 Web Worker
  schema/                冻结契约副本 + SOURCE.json（hash/版本测试）
  fixtures/demo/         已提交的小型 demo JSON（mock）
  fixtures/generated/    生成物（demo wav / 100k stress JSON+wav）— gitignore，不提交
  tools/                 serve.mjs / make-demo.mjs / make-stress.mjs / browser-check.mjs
  tests/                 Node 单测（102 项）
  screenshots/ reports/  浏览器截图与报告 — gitignore，作为证据返回主 Agent
```

## 新手复现（在哪个目录、装什么、跑什么）

**运行目录**：仓库 worktree 根目录（本文件所在仓库的根，下面命令均为根目录相对路径）。
**依赖**：Node.js ≥ 22（实测 Node 24.19；`tools/browser-check.mjs` 使用全局 WebSocket，Node 22+ 才提供；
Node 20 可跑单测但跑不了浏览器工具）；Chrome 或 Edge 本机已安装（不下载浏览器、无 npm 安装、无根 lock 改动）。
Python 3 仅用于可选的差分测试与上游 CLI 校验；没有 Python 时对应测试会显式 skip，其余照常通过。

### 1. 启动本地页面

```bash
node agentic/viewer/tools/serve.mjs --port 8123
```

浏览器打开 <http://127.0.0.1:8123/index.html>。服务只绑定 127.0.0.1 且只暴露 `agentic/viewer/`
（不 serve 仓库根、不 serve `.local/`）。

### 2. 生成并加载 demo（自生成 mock，非研究结果）

```bash
node agentic/viewer/tools/make-demo.mjs
```

- 生成 `agentic/viewer/fixtures/generated/demo-track.wav`（gitignore；确定性，SHA-256 与已提交 JSON 一致）
- 已提交的 JSON 在 `agentic/viewer/fixtures/demo/`：
  `demo-track.json`（BPM 120）、`demo-unknown-tempo.json`（bpm=null 未知）、
  `demo-empty-events.json`（空事件）、`demo-malicious-labels.json`（恶意 label 安全测试）

页面加载顺序：**先选 JSON 文件，再选音频文件**（可选：再选 stem 音频，按 JSON `stem.filename` 文件名匹配）。

**成功视觉判据**（对照 `screenshots/01-demo-loaded.png`）：

1. 状态区绿色：`音频已加载并校验（SHA-256 匹配 JSON）：12.000s @ …Hz。`
2. 时间轴视图：左侧 5 行 = `kick(id: drums-kick)`、`hi-hat(id: hats-1)`、`bass(id: bass-1)`、
   `piano(id: piano-a)`、`piano(id: piano-b)` —— 两个 piano **同 label 不同 id，分两行不合并**。
3. 顶部整体波形蓝色；行内彩色竖条 = onset（`hi-hat` 行只有细竖线：**无 duration 只画 onset，不造音符长度**）。
4. `piano-b` 行若加载了 stem 文件，行内会出现半透明波形（各 stem 波形）。
5. 纵向淡蓝细线 = beat grid（120 BPM 时每 0.5s 一条）；红色竖线 = playhead。

### 3. 交互步骤（BPM / 滚轮 / seek）

| 操作 | 步骤 | 预期 |
| --- | --- | --- |
| BPM 预填 | 看「BPM」输入框 | 显示 JSON 的 `tempo.bpm`（120）；来源标注“JSON tempo.bpm” |
| 未知 BPM | 加载 `demo-unknown-tempo.json` | 输入框为空 + 占位 `unknown` + 提示“BPM 未知（JSON tempo.bpm=null）”，无网格，可手动输入 |
| 手动 BPM | 输入 240 → 点「应用 BPM」或回车 | 网格变为每 0.25s 一条；**onset 位置/秒值完全不变**（界面提示“onset 秒数组未改变，已验证”） |
| 非法 BPM | 输入 0 / 负数 / abc / 1e9 | 红字错误（“必须为正数”“必须是有限数字”“<= 1000”），**网格保持不变** |
| 滚轮缩放 | 在时间轴上移动鼠标滚轮 | 以指针位置为锚点缩放（指针下的秒值不动），窗口约束在 [0, 12s] |
| 平移 | Shift+滚轮 或 按住拖拽 | 时间窗平移，两端不越界 |
| fit / reset | 点按钮 | fit=显示整曲；reset=回到整曲 + BPM 回到 JSON 值 |
| seek | 单击标尺/波形/轨道任意位置 | playhead 跳到该处音频秒；「跳转（秒）」输入框同步 |
| 播放 | 点「播放」或空格 | playhead 按真实音频秒推进；←/→ 快退/快进 1 秒 |

### 4. 大文件 / 格式错误时会发生什么（可读状态）

- **格式错误 JSON**：状态区红色 `JSON 未通过 agentic-audio-tracks/v1 校验…`，下方逐条列出
  `<JSON Pointer> [layer/code] 中文说明`（例如 `/audio/filename [semantic/E_PATH] 路径不安全…`），**不绘制错误数据**。
- **未知字段/版本不符**：`additionalProperties` / `E_VERSION` 明确报错（版本不符只报这一条）。
- **重复键、NaN/Infinity 字面量、1e999、深度 > 64、BOM**：解析层受控拒绝（`E_DUPLICATE_KEY` / `E_PARSE` / `E_NONFINITE`）。
- **恶意 label**（`<img onerror=…>`、`<script>`）：按纯文本渲染（textContent / Canvas fillText），不执行（见 `screenshots/04-…`）。
- **音频与 JSON 不一致**：文件名不匹配→黄色警告；SHA-256 不一致→红色错误；时长不一致→红色错误；
  明确写明“按 JSON 秒轴显示，不重定时/缩放”，**绝不默默 shift/retime**。
- **大 JSON**（>64MB 提示）：读取与校验在 Web Worker 异步进行，页面不冻结；波形峰值计算也在 Worker。
- **空事件**：行照常显示，标注“（无事件）”，状态区报告 0 事件。

## 测试

```bash
# 单元测试（131 项：结构/语义负例、原型键/Unicode 对齐、字节级入口、BPM 不改 onset、
# 锚点缩放、持续事件裁剪、bounded beat grid、server 逃逸、差分回归、卫生检查…）
node --test "agentic/viewer/tests/*.test.js"

# 真实浏览器检查（40 项 demo/stress/lifecycle）+ 100k stress 量化 + 截图
node agentic/viewer/tools/make-stress.mjs --events 100000     # 生成 gitignore 的 stress 夹具
node agentic/viewer/tools/browser-check.mjs                    # 需要本机 Chrome/Edge（Node ≥ 22）

# 真实输出集成回归（+6 项，输入路径由 CLI 指定，不硬写私有路径）
node agentic/viewer/tools/browser-check.mjs \
  --real-json  <result.json> \
  --real-audio <clip.wav> \
  --real-stem  <stems/a/target.wav> --real-stem <stems/b/target.wav> --real-stem <stems/c/target.wav>
```

- 浏览器检查输出：`agentic/viewer/reports/browser-check.json`（含环境/测量方法/逐项结果/截图时状态快照），
  截图 `agentic/viewer/screenshots/*.png`（01 demo、02 BPM 网格、03 锚点缩放、04 恶意 label、05 错误态、
  06/07 100k stress、08 持续事件裁剪、10 真实集成）。**这些均本地 gitignore，不入 GitHub。**
- 差分测试会调用 `python agentic/schema/validate.py --json`，把本目录 JS 校验器与 #32 冻结 Python 参考的
  `(layer, code, pointer)` 集合逐文件对比；无 Python 时显式 skip（不会假报成功）。
- 所有测试只用相对路径，换 checkout 可直接跑；不提交私有路径/key/音频/权重/完整会话。

## 性能：测量方法与实测（不空喊）

**方法**：Chrome DevTools Protocol 驱动真实 Chrome（headless），计时取页面内 `performance.now()`；
帧采样**每帧执行整场景重绘**（标尺+波形+全部行）后记录 rAF 间隔与重绘耗时——不测“空转调度”就声称含重绘的 fps；
每个数据集（JSON/音频）加载时指标**重置**，`firstRenderMs` 只属于当前数据集。可见/候选/绘制事件数由渲染层直接上报。
测量条件见 `reports/browser-check.json` 的 `environment`（实测：Chrome 150.0.7871.101 / Windows / 16 核 / deviceMemory 32 / DPR 1）。

**100k 事件 stress**（8 行 × 12500 事件 + 300s 音频，`fixtures/generated/stress-100000.json`）：

| 指标 | 实测（每帧含整场景重绘） |
| --- | --- |
| JSON 解析+结构+语义校验（Worker） | ~0.93–1.27 s（一次性） |
| 音频解码（300s WAV） | ~0.41–0.70 s |
| 当前数据集首帧渲染 / 单帧渲染 | ~5.4 ms / ~2.6–2.9 ms |
| 60 帧 rAF 间隔 avg / max | 13.3 ms / 15.4 ms（含每帧整场景重绘） |
| 60 帧重绘耗时 avg / max | 2.7 ms / 3.3 ms |
| 整曲视图候选/可见事件 | 100000 / 100000 → **按像素密度聚合**（aggregatedRows=8，不画 10 万图元） |
| 放大 2s 窗口 | 可见/绘制 725（二分查找 culling + 边缘裁剪） |
| DOM 节点总数 | **103**（每乐器行 1 个标签节点；**没有每事件/每采样 DOM**） |

## 安全 / 隐私

- 只用 `<input type=file>` 本地读取；无上传、无遥测、无第三方 URL（页面代码中 `http(s)://` 零命中，测试强制）。
- **严格字节入口**：不用 `File.text()`（它会吞掉 UTF-8 BOM、把非法 UTF-8 替换成 `\uFFFD`）；原始字节先查 BOM、
  再严格 UTF-8 解码（`E_PARSE` 受控拒绝），与冻结 Python loader 同判定（字节级+真实 File+差分回归）。
- label / 错误文本一律 `textContent` 或 Canvas `fillText`；禁用 `innerHTML` 等 sink（测试强制），CSP `default-src 'none'`。
- JSON 里的 `audio.filename` / `stem.filename` **只做本地文件名匹配**，绝不按其路径 fetch（`../`、盘符、UNC、URL、
  保留名等由语义层 `E_PATH` 拒绝）；**同名 stem（如多份 `target.wav`）按 SHA-256 身份消歧**，hash/时长不符明确报错
  且不渲染、不报成功，绝不自动对齐/重定时。
- 本地服务只暴露 `agentic/viewer/`，**realpath 容器检查**（junction/symlink 逃逸也被拒，回归测试含合成 junction）。
- 音频走 ObjectURL + AudioContext：换文件/`dispose()` 即 revoke/close/terminate（浏览器测试断言全部归零，含加载中 dispose）。
- 生成的 wav/stress JSON/截图/报告/真实产物全部 gitignore，不进 Git；公开只出脱敏摘要。

## 限制与 pending（不假称全验收）

1. **真实输出集成（#33）已可加载并有集成回归**（CLI 传路径：3 行 drums/bass/synthesizer、110 原始事件、
   tempo=null/unknown、hash/时长核对、3 份同名 stem 按 hash 映射到各行、seek/缩放）；但 **#33 后端 spotcheck
   统计存在 float32 解码/采样率窗口问题、正在离线纠正**——UI 只展示冻结 schema 的原始事件与**标注为
   “文档自述（非验证结论）”的 limitations/provenance**，不展示任何 isolation/leak 等质量指标，**不宣称研究准确率**。
2. **人类验收 = pending**。截图/浏览器检查是主 Agent 自动验证证据，**不能代替人类验收**。
3. `pitch` 仅在事件确有音高证据时显示参考信息，不承诺 MIDI 真值；demo 的 pitch/confidence 是随意演示值。
4. 极端情况（>200k 事件单行、BPM/时长极端到超出可计算范围）已被拒绝或给出可读 “网格不支持” 状态，未做更极端压测。
5. 浏览器兼容：按 Chrome/Edge 现代特性实现；其他浏览器未验证。headless 截图需强制合成提交（工具已处理），
   否则可能拍到上一帧。

## 人类验收指南（按 issue #34 验收项）

1. `node agentic/viewer/tools/serve.mjs --port 8123` → 打开 <http://127.0.0.1:8123/index.html>。
2. 加载 `agentic/viewer/fixtures/demo/demo-track.json` + `agentic/viewer/fixtures/generated/demo-track.wav`
   （先跑 `node agentic/viewer/tools/make-demo.mjs` 生成 wav）。
3. 看：5 行乐器（两个 `piano` 分行）、横轴 0–12s、onset 与波形对齐音频秒；`hi-hat` 行只有 onset 细线。
4. 试：BPM 120→240（只改网格）、输入 0（报错不变）、滚轮在 4s 处缩放（4s 不动）、拖拽平移、单击 3s seek、
   播放看 playhead 推进、fit/reset。
5. 加载 `demo-unknown-tempo.json` 验证“BPM 未知”提示与手动输入；加载 `demo-malicious-labels.json` 验证
   恶意 label 不执行；故意选错音频（如 3s wav）验证 hash/时长 mismatch 明确报错且不重定时。
6. 真实结果验收：加载 #33 真实 `result.json` + 16s clip（+ 3 份 stem `target.wav`，按 hash 自动消歧到对应行），
   确认 3 行（drums/bass/synthesizer）、110 事件、BPM 显示 unknown 可手输；**不做准确率评价**。
