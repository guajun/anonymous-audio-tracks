# agentic/schema：`agentic-audio-tracks/v1` 结果 JSON 协议与校验器（issue #32）

本目录定义**新的版本化协议** `agentic-audio-tracks/v1`：把一次音频分析的结果写成一个 JSON 文档
（piano-roll-like：**行 = 乐器/声音来源，列 = 秒时间**），并提供结构 schema、跨字段语义校验器、
纯 JSON fixture 与自动测试。

- **不修改旧协议**：根目录 `docs/SCHEMAS.md` 里的 anonymous E/P 轨迹 schema 完全不动，两套协议互不冲突。
- **时间真值是秒**：`onset_seconds` 等字段直接是音频时间轴上的秒；**BPM 只是辅助网格**，不量化事件、不反推 MIDI。
- **piano-roll-like ≠ MIDI ≠ 音高真值**：只表达“来源 + 何时活动”，`pitch` 仅有证据时才填，且只是估计。
- **不夸大**：SAM 分离、LLM/bridge 猜测都只是**假设**，不是真值；本目录不承诺任何检测/分离精度。

冻结接口见 [“冻结接口（#33/#34 用）”](#冻结接口3334-用)。机器可读语义规则：[`schema/semantic_rules.json`](schema/semantic_rules.json)。

## 目录结构

```
agentic/schema/
  schema/agentic-audio-tracks-v1.schema.json   结构 schema（Draft 2020-12 便携子集，可在浏览器使用）
  schema/semantic_rules.json                  跨字段语义规则表（机器可读）
  agentic_schema/                             校验器实现（纯标准库 + 可选 jsonschema 引擎）
  validate.py                                 冻结 CLI 入口
  fixtures/valid_*.json                       正例（纯 JSON、自生成、可当网页开发 fixture）
  fixtures/negative/*.json                    负例（每个都必须被拒绝）
  fixtures/negative/expected.json             负例预期错误（layer/code/pointer）
  tests/                                      自动测试（与 cwd 无关）
  requirements.txt                            可选依赖 pin（不动根 uv.lock）
  README.md                                   本文档
```

## 依赖与安装（局部 pin，不碰根锁）

校验器**核心零依赖**（纯标准库）：没有 `jsonschema` 也能跑结构+语义校验（引擎显示 `stdlib`）。
官方 `jsonschema` 引擎是**可选增强**（引擎显示 `jsonschema`），用于交叉验证与浏览器同源行为，pin 见
[`requirements.txt`](requirements.txt)（`jsonschema==4.26.0`）。**不要改根 `uv.lock`**，用独立环境：

```sh
# 方式 A：一次性独立环境（推荐，零痕迹，不写任何 lock）
uv run --no-project --python 3.12 --with pytest==8.4.2 --with jsonschema==4.26.0 \
    python -m pytest agentic/schema/tests -q

# 方式 B：局部 venv + requirements pin
uv venv agentic/schema/.venv
uv pip install --python agentic/schema/.venv -r agentic/schema/requirements.txt
uv pip install --python agentic/schema/.venv pytest==8.4.2
agentic/schema/.venv/Scripts/python -m pytest agentic/schema/tests -q   # Windows
agentic/schema/.venv/bin/python -m pytest agentic/schema/tests -q       # Linux/macOS
```

不安装 `jsonschema` 时，`tests/test_engines_agree.py` 会**显式 skip**（不是通过）；安装 pin 后应 0 skip。

## 如何校验（Junior 按此操作）

1. 准备结果文档 `result.json`（构造方法见下节）。
2. 在**任意目录**运行（路径相对脚本自身解析，不依赖当前目录）：

```sh
python agentic/schema/validate.py path/to/result.json
python agentic/schema/validate.py --json path/to/result.json        # 机器可读
python agentic/schema/validate.py --engine stdlib path/to/result.json  # 强制零依赖引擎
```

3. 看输出与退出码：

| 退出码 | 含义 |
|---|---|
| `0` | 全部文档合法（`OK ...`） |
| `1` | 至少一个文档不合法（`FAIL ...` + 逐条错误） |
| `2` | 用法/文件/依赖错误（例如文件不存在、`--engine jsonschema` 但没装） |

输入过深（>64 层或递归超限）按**不合法文档**处理：`FAIL` + `E_PARSE`，退出码 1，**不吐 traceback**。

错误输出每行一条，格式固定：

```
<文件>: <JSON Pointer> [<layer>/<code>] <中文说明>
```

示例：

```
FAIL fixtures/negative/neg_onset_after_end.json (engine=stdlib)
  /instruments/0/events/0/onset_seconds [semantic/E_TIME_RANGE] onset_seconds 必须在 [0, 12.0] 秒内（时间边界=音频末尾），得到 12.5
```

### 如何看 JSON Pointer（返回路径定位 error）

- ``（空）= 文档根；`/audio/filename` = 根对象 `audio` 里的 `filename`；
- 数组用下标：`/instruments/0/events/2/onset_seconds` = 第 1 个乐器的第 3 个事件的 onset；
- 字段名里的 `~` 写成 `~0`、`/` 写成 `~1`（RFC 6901）。
- layer 取值：`parse`（JSON 解析）、`version`（版本）、`structure`（结构）、`semantic`（跨字段语义）。
- code 取值：语义层是稳定 `E_*` 码；结构层是 schema 关键字名（`type`/`required`/`pattern`/…）。

## 如何构造一个结果文档

按顺序填（每个字段的含义见下面字段表）：

1. `schema_version` 固定写 `"agentic-audio-tracks/v1"`。
2. `audio`：相对文件名 + `sha256` + 时长（秒）+ 采样率。哈希可用
   `python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" 音频文件`。
3. `tempo`：不知道 BPM 就写 `{"bpm": null, "source": "unknown", "confidence": 0}`（**不要**猜）。
4. `instruments`：一个来源一个条目；同类乐器可以多条（`label` 相同、`id` 不同）；实在没识别出来就写
   `"label": "unknown"` 并在 `description` 说明依据。没有任何来源就给空数组 `[]`。
5. 每个来源的 `events`：先写 `id` 与 `onset_seconds`（秒）；**有证据**才加 `duration_seconds` / `pitch` /
   `confidence` / `source` / `method`。写完按 `(onset_seconds, id)` 升序排序。
6. `provenance.steps`：写清这份 JSON 是怎么来的（工具 + 来源类别 + 备注）；`limitations`：写清已知不可靠之处。

最小可用示例（= `fixtures/valid_minimal.json`）：

```json
{
  "schema_version": "agentic-audio-tracks/v1",
  "audio": {
    "filename": "fixtures-audio/mock-etude.wav",
    "sha256": "787f9d428c8ceb9e009604253177cedc164ab3542acc24729329262717714c72",
    "duration_seconds": 12.0,
    "sample_rate": 44100
  },
  "tempo": { "bpm": 120.0, "source": "mock", "confidence": 0.9 },
  "instruments": [
    {
      "id": "inst-1",
      "label": "piano",
      "description": "合成 fixture：单一 mock 乐器来源",
      "source": "mock",
      "confidence": 1.0,
      "events": [
        { "id": "inst-1-ev-1", "onset_seconds": 0.0 },
        { "id": "inst-1-ev-2", "onset_seconds": 2.5, "duration_seconds": 1.25 }
      ]
    }
  ],
  "provenance": {
    "steps": [{ "tool": "aat-schema-fixtures", "source": "mock", "note": "自生成纯 JSON fixture，无真实音频" }]
  },
  "limitations": ["纯合成示例，不代表任何模型精度"]
}
```

更多示例：

| 文件 | 演示的策略 |
|---|---|
| `fixtures/valid_minimal.json` | 最小合法文档 |
| `fixtures/valid_multi_instrument.json` | 多乐器、同类多来源、同时 onset、stem 引用、pitch 证据 |
| `fixtures/valid_unknown_tempo.json` | tempo 未知（`bpm: null`）策略 |
| `fixtures/valid_empty_instruments.json` | 空结果策略（`instruments: []`） |
| `fixtures/valid_full.json` | 全字段 + 时间边界（0 秒 onset、onset=音频末尾、onset+duration=音频末尾）+ 多来源 provenance |

## 字段表（逐字段）

约定：**必填** = schema `required`；**可空** = 允许 `null`；所有数字必须是**有限数且在 float64 可互操作范围内**
（`|x| <= 1.7976931348623157e308`；拒绝 NaN/Inf/`1e999`/`10**400`），
布尔值不是数字；所有字符串均为 UTF-8。未知字段一律**拒绝**（`additionalProperties: false`），
加字段=破坏性变更=新 `schema_version`。文档嵌套深度（根=1）不得超过 **64**（见“数值与深度政策”）。

### 顶层

| 字段 | 类型 | 必填 | 可空 | 范围/约束 | 含义 |
|---|---|---|---|---|---|
| `schema_version` | string | 是 | 否 | 精确等于 `agentic-audio-tracks/v1` | 协议版本 |
| `audio` | object | 是 | 否 | 见下 | 被分析音频元数据 |
| `tempo` | object | 是 | 否 | 见下 | 节拍估计（可完全未知） |
| `instruments` | array | 是 | 否 | 可为空数组 | 乐器/声音来源列表（piano-roll 的行） |
| `provenance` | object | 是 | 否 | 见下 | 文档级来源记录 |
| `limitations` | string[] | 是 | 否 | 每条 1..1024 字符，可为空数组 | 已知限制（透明说明） |

### `audio`

| 字段 | 类型 | 必填 | 可空 | 范围/约束 | 含义 / 取值来源 |
|---|---|---|---|---|---|
| `filename` | string | 是 | 否 | 1..512 字符，**安全相对路径**（规则见 E_PATH） | 音频文件名，用于和本地音频根目录**匹配**，不做任意 fetch |
| `sha256` | string | 是 | 否 | `^[0-9a-f]{64}(?![\s\S])`（绝对结尾，64 小写 hex） | 文件内容哈希（DSP/人工计算；mock 可为自生成哈希） |
| `duration_seconds` | number | 是 | 否 | `> 0`，有限 | 音频总时长（秒）；一切时间边界以此为准 |
| `sample_rate` | integer | 是 | 否 | `>= 1` | 采样率（Hz） |

### `tempo`

| 字段 | 类型 | 必填 | 可空 | 范围/约束 | 含义 / 取值来源 |
|---|---|---|---|---|---|
| `bpm` | number \| null | 是 | **是** | 数值时 `0 < bpm <= 1000`（典型 20..400） | 每分钟节拍数；**只是辅助网格**，不用于量化事件 |
| `source` | enum | 是 | 否 | `human/dsp/llm/sam/mock/native/bridge/unknown` | BPM 从哪来 |
| `confidence` | number | 是 | 否 | `[0,1]` | BPM 置信度 |
| `beat_origin_seconds` | number | 否 | 否 | `[0, audio.duration_seconds]` | 可选节拍网格原点（秒） |

**未知/已知互斥（E_TEMPO）**：`bpm: null` ⟺ `source: "unknown"` 且 `confidence: 0` 且**没有** `beat_origin_seconds`；
`bpm` 是数值时 `source` 不得是 `"unknown"`。UI 在 `bpm: null` 时应允许人类输入 BPM。

### `instruments[]`（一行 = 一个来源）

| 字段 | 类型 | 必填 | 可空 | 范围/约束 | 含义 / 取值来源 |
|---|---|---|---|---|---|
| `id` | string | 是 | 否 | `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(?![\s\S])`，全文档唯一 | 稳定 ID（跨运行可复现） |
| `label` | string | 是 | 否 | 1..128 字符 | 人类可读名称；识别不出写 `"unknown"`；**允许重复**（同类多来源） |
| `description` | string | 是 | 否 | 1..1024 字符 | 来源描述（谁产出的、依据什么） |
| `source` | enum | 是 | 否 | 同 `tempo.source` | 识别来源类别（LLM/SAM/DSP/human/mock/native/bridge/unknown） |
| `confidence` | number | 是 | 否 | `[0,1]` | 对该来源判断的置信度 |
| `stem` | object | 否 | 否 | `{filename, sha256}` 均必填 | 可选分离轨引用（只有路径+哈希，不内嵌音频） |
| `events` | array | 是 | 否 | 可为空数组；**必须按 (onset,id) 升序** | 该来源的事件列表 |

### `instruments[].events[]`（一个格子 = 一个事件）

| 字段 | 类型 | 必填 | 可空 | 范围/约束 | 含义 / 取值来源 |
|---|---|---|---|---|---|
| `id` | string | 是 | 否 | 同上 ID 规则（绝对结尾），**全文档唯一（跨乐器）** | 稳定事件 ID |
| `onset_seconds` | number | 是 | 否 | `[0, audio.duration_seconds]` | 事件开始（秒）。**不量化到节拍** |
| `duration_seconds` | number | 否 | 否 | `> 0` 且 `onset + duration <= audio.duration_seconds` | 事件时长（秒）；**没证据不填** |
| `pitch` | object | 否 | 否 | `{midi, confidence, source}` 三个字段必须一起出现；`midi` 为整数 `0..127` | 音高**估计**（不是 MIDI 真值）；**没证据不填** |
| `confidence` | number | 否 | 否 | `[0,1]` | 事件置信度 |
| `source` | enum | 否 | 否 | 同上 | 事件取值来源；缺省视为继承所属 instrument |
| `method` | string | 否 | 否 | `^[a-z0-9][a-z0-9._-]{0,63}(?![\s\S])`（绝对结尾） | 产生该事件的方法（如 `spectral-flux`、`llm-estimate`、`human-tap`） |

### `provenance` / `limitations`

| 字段 | 类型 | 必填 | 可空 | 范围/约束 | 含义 |
|---|---|---|---|---|---|
| `provenance.steps` | array | 是 | 否 | **至少 1 项** | 产出文档的步骤链（顺序即执行顺序） |
| `provenance.steps[].tool` | string | 是 | 否 | 1..128 字符 | 工具/组件名（如 `audio-toolbox.sam`） |
| `provenance.steps[].source` | enum | 是 | 否 | 同上 | 该步骤来源类别 |
| `provenance.steps[].note` | string | 否 | 否 | 1..1024 字符 | 该步骤说明 |
| `provenance.notes` | string | 否 | 否 | 0..4096 字符 | 补充说明（版本/参数/偏差） |
| `limitations` | string[] | 是 | 否 | 每条 1..1024 字符 | 已知限制（推荐至少一条：精度不承诺等） |

`source` 枚举语义：`human`=人工标注/输入，`dsp`=传统信号处理，`llm`=语言/多模态模型推测，
`sam`=SAM Audio 分离结果，`mock`=自生成示例，`native`=Pi 原生声音模态，`bridge`=非原生桥接调用，
`unknown`=未知来源。**LLM/SAM/bridge/native 都只给假设，不自动成为真值。**

## 数值与深度政策（冻结）

- **数值**：所有数字必须是有限数且在 IEEE-754 float64 可互操作范围内（`|x| <= 1.7976931348623157e308`）。
  `NaN`/`Infinity` 字面量、`1e999`（溢出浮点）、`10**400`（超出 float64 范围的超大整数）一律拒绝并给出精确
  Pointer（`E_NONFINITE`/`E_FINITE`），**绝不静默转 Infinity**；布尔值不是数字。三个入口（`validate_text` /
  `validate_file` / `validate_document`）与两个结构引擎（stdlib / jsonschema）行为一致。
- **深度**：文档嵌套深度（根容器=1）不得超过 **64**；超出即受控拒绝（`E_PARSE`，退出码 1，**无 traceback**）。
  解析期递归超限（如 `'['*1200+'0'+']'*1200`）同样按 `E_PARSE` 受控拒绝；不做深嵌套 JSON 支持。
- **pattern 绝对结尾**：`id`/`sha256`/`method` 用 `^...(?![\s\S])` 而**不是 `$`**——Python/JS 的 `$` 可能匹配在
  末尾换行之前（`"inst-1\n"` 会被放过）；`(?![\s\S])` 在 Python `re` 与浏览器 ECMA 正则里语义一致，
  尾随 `\n`/`\r\n` 一律拒绝。

## 时间与边界规定（冻结）

- 时间轴：`t = 0` 是音频文件开头；单位一律**秒**；`audio.duration_seconds` 是唯一时间上限。
- `onset_seconds`：`0 <= onset <= audio.duration_seconds`（**允许 0，也允许恰好落在文件末尾**）。
- `duration_seconds`：必须 `> 0`；`onset + duration <= audio.duration_seconds`，**允许相等**（事件恰好在文件末尾结束）。
- `beat_origin_seconds`：`0 <= 值 <= audio.duration_seconds`。
- 浮点比较容差：`1e-9` 秒（只用于边界比较，不改变写入的数值）。
- 事件排序：同一 `instrument.events` 内按 `(onset_seconds 升序, id 升序)`；**允许同时 onset**。
- `instruments` 数组顺序 = 展示顺序，协议不要求排序（但 `id` 必须唯一）。

## 空结果 / 未知 tempo / 同类多来源策略（冻结）

- **空结果**：`instruments: []` 合法，表示“没找到可信来源”，**不表示音频无声**；下游当“无事件”处理，不要当错误。建议在 `limitations` 说明。
- **未知 tempo**：按上文 E_TEMPO 互斥规则写；UI 应允许人类补 BPM。
- **同类多来源**：多个 `label` 相同、`id` 不同的条目合法（例如 SAM 分出两路钢琴）；事件 `id` 全文档唯一。

## 跨字段语义规则与错误码

机器可读表：[`schema/semantic_rules.json`](schema/semantic_rules.json)（含反例）。稳定错误码：

| code | 层 | 拒绝什么 |
|---|---|---|
| `E_VERSION` | version | `schema_version` 不是 v1（只报这一个错并停止） |
| `E_PARSE` | parse | 非法 JSON / BOM / 非 UTF-8 / `NaN`·`Infinity`·`-Infinity` 字面量 |
| `E_DUPLICATE_KEY` | parse | 对象重复键（拒绝 last-wins；只报键名——已知限制） |
| `E_NONFINITE` | parse | 数字溢出/非有限（如 `1e999` → inf），带精确 Pointer |
| `E_FINITE` | semantic | 文档内任何非有限数字（兜底） |
| `E_BOOL_NUMBER` | semantic | 布尔冒充数字（`true` 不是 `1`） |
| `E_TIME_RANGE` | semantic | `onset_seconds` / `beat_origin_seconds` 越界或为负 |
| `E_TIME_DURATION` | semantic | `duration_seconds <= 0` 或 `onset + duration` 超出音频末尾 |
| `E_TIME_ORDER` | semantic | 事件未按 `(onset, id)` 排序 |
| `E_TEMPO` | semantic | tempo 未知/已知互斥规则被破坏 |
| `E_ID_UNIQUE` | semantic | `instrument.id` 或 `event.id` 重复 |
| `E_PATH` | semantic | 路径不安全（见下） |
| `E_RANGE` | semantic | `confidence` 不在 `[0,1]`、`pitch.midi` 不是 0..127 整数 |

### 路径安全规则（E_PATH，冻结）

`audio.filename` 与 `instruments[].stem.filename` 必须是**安全相对路径**，仅用于和本地音频根目录匹配，
**不做任意 fetch**。拒绝：空路径、超长（>512）、控制字符、反斜杠 `\`、`/` 开头的绝对路径、盘符（`C:`）、
UNC（`//` 或 `\\` 开头）、URL（`scheme://`）、`.`/`..` 段、空段（`//` 或结尾 `/`）、段首尾空格或结尾点、
Windows 保留设备名（`CON`/`NUL`/`COM1`…）、字符 `< > : " | ? *`。不做百分号解码（`%2e%2e` 只是普通字符）。
**使用方仍必须自己做“解析后仍在音频根目录内”的包含检查。**

## 常见错误与排查

| 现象 | 原因 / 修复 |
|---|---|
| `[/] [parse/E_PARSE] JSON 语法错误` | 多/少逗号、尾逗号、单引号、注释；用严格 JSON |
| `[parse/E_PARSE] 拒绝非标准 JSON 常量 'NaN'` | Python `json` 默认允许 `NaN/Infinity`；本协议拒绝，改成 `null` 或真实数值 |
| `[parse/E_NONFINITE] /audio/duration_seconds` | `1e999` 溢出数字或 `10**400` 超大整数；检查单位/量纲（数值政策见上） |
| `[parse/E_PARSE] 文档嵌套过深` | 嵌套深度 > 64 或递归超限；展平结构（深度政策见上） |
| `[parse/E_DUPLICATE_KEY]` | 同一对象写了两个同名键；删掉一个（本协议不 last-wins） |
| `[version/E_VERSION] /schema_version` | 版本字符串写错（必须精确 `agentic-audio-tracks/v1`） |
| `[structure/type] /tempo/bpm` | 把 `true`/字符串当数字；`bpm` 只能是数字或 `null` |
| `[structure/additionalProperties]` | 加了协议外字段；删掉或提出新版本 |
| `[structure/required]` | 缺必填字段（如 `pitch` 三个证据字段要一起出现） |
| `[structure/pattern]` | `id`/`sha256`/`method` 尾随换行（`"inst-1\n"`）或格式不对；去掉空白、用绝对结尾格式 |
| `[semantic/E_TIME_ORDER]` | 事件没排序；按 `(onset_seconds, id)` 升序重排 |
| `[semantic/E_TIME_RANGE]` | onset 超过音频末尾或为负；核对 `audio.duration_seconds` |
| `[semantic/E_TIME_DURATION]` | 时长 ≤ 0，或 `onset+duration` 超时 |
| `[semantic/E_ID_UNIQUE]` | 事件/乐器 id 重复；id 全文档唯一（跨乐器也唯一） |
| `[semantic/E_PATH]` | 用了绝对路径/`..`/URL/盘符/UNC；改安全相对路径 |
| `[semantic/E_TEMPO]` | `bpm: null` 但写了来源/置信度/节拍原点（或反过来） |

## 浏览器里使用

- **结构校验**：直接用 [`schema/agentic-audio-tracks-v1.schema.json`](schema/agentic-audio-tracks-v1.schema.json)
  （Draft 2020-12，且只用便携子集关键字，无外部引用），任何浏览器端 Draft 2020-12 校验器（如 `jsonschema` JS、
  hyperjump 等）都能用，错误同样按实例路径定位。
- **语义校验**：按 [`schema/semantic_rules.json`](schema/semantic_rules.json) 在前端复刻（规则都是纯 JSON 可判定的
  跨字段比较）；Python 参考实现在 `agentic_schema/semantic.py`。注意浏览器 JSON 解析器接受 `1e999→Infinity`、
  拒绝 `NaN` 字面量的行为与 Python 不同——请以本仓库 CLI 的判定为准。

## 冻结接口（#33/#34 用）

以下内容已冻结；如需变更请在 issue #32 评论提出，不要各自扩展：

1. **版本字符串**：`schema_version == "agentic-audio-tracks/v1"`。
2. **结构 schema**：`agentic/schema/schema/agentic-audio-tracks-v1.schema.json`（字段/类型/必填/范围以上表为准）。
3. **语义规则**：`agentic/schema/schema/semantic_rules.json`（错误码上表为准）。
4. **校验命令**（任意 cwd；退出码 0/1/2 如上）：

   ```sh
   python agentic/schema/validate.py [--json] [--engine auto|jsonschema|stdlib] <file.json> [more.json ...]
   ```

5. **Python API**（`agentic/schema` 加入 `sys.path` 后）：

   ```python
   from agentic_schema import (
       SCHEMA_VERSION, SCHEMA_PATH, SEMANTIC_RULES_PATH, Issue, Report,
       validate_file, validate_text, validate_document,
   )
   report = validate_file("outputs/result.json")   # report.ok / report.issues / report.engine
   ```

6. **输出约定**：#31/#33 的分析结果写为 `outputs/result.json`（audio.filename 相对音频根目录）；
   #34 用 `fixtures/valid_*.json` 作网页开发 fixture，`bpm: null` 时允许人类输入 BPM。
7. **错误格式**：`<file>: <JSON Pointer> [<layer>/<code>] <message>`；`--json` 输出
   `{source, ok, engine, schema_version, issues:[{layer, code, pointer, message}]}`。
8. **数值/深度/pattern 政策**：数字有限且 `|x| <= 1.7976931348623157e308`（float64 互操作范围）；嵌套深度 ≤ 64；
   `id`/`sha256`/`method` 绝对结尾 `^...(?![\s\S])`（拒绝尾随换行）。详见上文“数值与深度政策”。

## 限制（不夸大）

- 本目录只定义**格式与校验**，不产生、不评估任何音频分析结果；不承诺 onset/分离/BPM 的准确率。
- `pitch` 只是带来源的**估计**，piano-roll-like 不是 MIDI、不是乐谱、不是音高真值；事件不量化到节拍。
- SAM 分离、LLM/bridge/native 猜测都只是**假设**，`provenance` 只保证“说清来源”，不保证来源正确。
- fixture 是**纯 JSON 自生成 mock**（哈希为 `sha256("agentic-audio-tracks/v1:fixture:<名字>")`，仅占位），
  不含任何私有音频/权重/API key/个人路径。
- 重复键错误只报键名、不报完整路径（解析器限制）；非有限数会带精确 Pointer。
- 结构校验的 stdlib 引擎是 jsonschema 的子集参考实现；两者一致性由 `tests/test_engines_agree.py` 交叉验证。

## 测试

```sh
# 任意目录可运行；测试不写死 cwd、不访问私有路径、无网络
uv run --no-project --python 3.12 --with pytest==8.4.2 --with jsonschema==4.26.0 \
    python -m pytest agentic/schema/tests -q
```

覆盖：严格解析（NaN/Inf/重复键/溢出/编码）、结构（类型/必填/未知字段/范围/模式/可空/整数语义）、
语义（时间边界含 0/末尾/相等、排序与同时 onset、ID 唯一、路径安全 20+ 反例、tempo 互斥、范围兜底）、
fixture 契约（正例全过、负例命中 expected、纯 JSON、无私有信息）、CLI（退出码/定位/--json/cwd 无关）、
双引擎一致性、README/规则/版本契约，以及鲁棒性回归（review 复现项）：超大整数/溢出数字在
validate_text/file/document 三入口与双引擎下都给稳定 Pointer 错误、结构失败后语义不崩、
id/sha256/method 尾随 `\n`/`\r\n` 双引擎拒绝、超深嵌套受控拒绝（CLI 退出码 1 无 traceback）。
成功判据：全部 pass；装了 pin 依赖时 **0 skip**。
