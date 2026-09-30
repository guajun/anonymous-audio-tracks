# agentic/workspace-template — 可复现的 Pi 音频分析 workspace（issue #31）

**一句话**：用本目录的 `bootstrap.py` 在**本地忽略目录**部署一个真正的 Pi 项目（项目设置固定
`google/gemini-3.8-flash`、`sam-audio` skill（pin 安装）、明确标注的音频桥接扩展、音频输入与
输出目录），用 `doctor.py` 自检、用 `smoke.py` 做真实模型冒烟。它不是训练项目。

- 真实/mock 分离：本 README 的命令全部可实跑；**离线自动化测试**（`tests/`，18 项，不联网、
  不调用模型）与**真实验证**（实机 Windows 启动 / Pi 项目 trust / Gemini skill 发现 / SAM
  help 与 dry-run）分开记录，真实证据见 [`reports/30-deploy-verification.md`](reports/30-deploy-verification.md)。
- 冻结接口见 [`manifest.json`](manifest.json)：toolbox pin、SAM commit、桥接文件 SHA-256、
  模型、输出路径约定（`outputs/result.json`，**schema 归 issue #32**，本项不发明 schema）。

---

## 0. 真实 / mock 对照

| 内容 | 类型 | 位置 |
|---|---|---|
| 部署/幂等/冲突/诊断逻辑测试 | 离线（mock/tmp 目录） | `tests/test_workspace_template.py`（18 项） |
| Windows 实机启动（PowerShell 5.1 / pwsh / git-bash） | 真实 | `reports/30-deploy-verification.md` §3 |
| project trust 加载项目设置/扩展 | 真实（零 API 探针 + 真实运行模型证据） | 报告 §4 |
| Gemini 3.8 Flash skill 发现 + SAM CLI `--help` + `audio_attach` 错误契约 | 真实（1 次 Pi run，≈3 个 assistant 请求） | 报告 §5 |
| SAM `check-environment` + `separate --dry-run` | 真实（不加载模型、不跑 GPU） | 报告 §6 |
| GPU 真实分离 | **不做**（issue #33） | — |

## 1. 前置条件（Junior 从这里开始）

在哪运行：仓库内任意位置（脚本用自身路径定位仓库）。需要：

- Windows 10/11 + git-bash（已在 Windows 验证；POSIX 路径见括号注释）；
- Python 3.11+（`python --version`）；
- Node ≥ 22.19（`node --version`；Pi 0.87.1 的最低要求）；
- **Pi 0.87.1**（`pi --version`）且 Google provider 认证就绪
  （`pi auth check --provider google --json` 只输出 `{"status":"ready",...}`，**不打印 key**）；
- GitHub CLI（`gh --version`；`gh skill` 为 preview 命令）；
- FFmpeg 4–8 **full-shared** 构建在 PATH（Windows static 构建缺共享 DLL 不够）；
- 一个本地 SAM Audio checkout（本机迁移目录，含 `.venv`、`model-cache/` 权重）。

**不做**：不下载权重、不改全局设置/认证、不改 `~/.pi/agent/trust.json`、不跑 GPU 推理。

## 2. 从空目录部署

```sh
# 任意目录执行（<WS> 为本地忽略目录；示例用仓库内 .local/agentic/workspace）
python agentic/workspace-template/bootstrap.py \
  --workspace "<WS>" \
  --fixture \
  --audio "<某本地音频文件>" \
  --sam-root "<SAM checkout 根目录>" \
  --sam-python "<SAM checkout 的 .venv 解释器>"
```

成功判据（退出码 0，逐步 `[ok]`）：

1. `<WS>` 内出现 `AGENTS.md`、`.pi/settings.json`、`.pi/extensions/audio-bridge.ts`、
   `.pi/extensions/audio_guard.mjs`、`run.ps1`、`run.sh`、`local/config.json`、
   `audio/inputs/manifest.json`；
2. `[ok] gh skill install sam-audio @dfbc40a9541f… -> .pi/skills/sam-audio`（pin
   `dfbc40a9541f686207b65b93b1332bb505654261`；`gh skill list` 出现 `sam-audio  pi  project`）；
3. `audio/inputs/manifest.json` 每条含 `sha256/bytes/duration_s/sample_rate/channels/probe/source`。

**再次运行的语义（验收点）**：

- 已存在的模板文件、用户改动的 `AGENTS.md`/`.pi/settings.json`/`local/config.json`、已导入音频
  一律 `[skip]`/`[kept]`，**不覆盖**；fixture 生成同样不覆盖；
- 导入音频与已有文件**同名不同 sha256** → 显式失败（退出码 3，`E_HASH_CONFLICT`），原文件不动；
- 目标目录非空且不是本工具管理的 workspace → 拒绝（退出码 3，`E_WS_UNMANAGED`）；
- 桥接文件与冻结 pin 不一致 → 失败（`E_BRIDGE_PIN`/`E_BRIDGE_CONFLICT`），需显式 `--force-bridge`。

退出码：`0` 成功 / `2` 用法（如音频不存在）/ `3` 守卫或冲突 / `4` 外部依赖失败（git/gh 缺失、
`gh skill install` 失败）。

## 3. 启动（Windows 实测）

```powershell
cd <WS>
.\run.ps1                              # PowerShell 5.1 / pwsh 均验证
.\run.ps1 "分析 audio/inputs/fixture-a.wav"
```
```sh
cd <WS>
./run.sh                               # git-bash 验证
```

脚本做的事：切到 workspace 根、设 `PI_AUDIO_BRIDGE_ROOT=<WS>/audio/inputs`、读
`local/config.json` 的 `SAM_AUDIO_ROOT`、以 **`pi --approve`** 启动。

**project trust（明确约定）**：`--approve` 是**进程级**信任，加载本项目的 `.pi/settings.json`、
`.pi/extensions/`、`.pi/skills/`；**不写** `~/.pi/agent/trust.json`、不改全局设置/认证。交互式
用户想持久化可用 Pi 的 `/trust`（自担；本模板不代做）。

**模型固定**：`.pi/settings.json` 把 `defaultProvider/defaultModel` 固定为
`google/gemini-3.8-flash`（issue #30 冻结研究模型）；**不要**用 `--model` 覆盖。桥接扩展只对该
模型生效，其他模型得到 `E_AUDIO_MODEL`。

## 4. 给 Agent 提问（示例）

```text
/audio_attach 由你自己调用。请听 audio/inputs/fixture-a.wav：
1) 用 audio_attach 工具挂载（path 是相对 audio/inputs 的文件名）；
2) 判断音高是上行还是下行；
3) 用 sam-audio skill 对它跑 sam separate --dry-run（--sam-root/--python 取自 local/config.json），
   把 plan JSON 要点写到 outputs/notes.md。
```

要求 Agent 遵守的规则都写在 `<WS>/AGENTS.md`（模型固定、桥接用法、SAM 调用顺序、输出位置、
隐私边界、失败语义）。

## 5. 音频输入：桥接（bridge），不是 Pi 原生

**Pi 0.87.1 没有原生音频输入**（实测结论见 [`../audio-probe/README.md`](../audio-probe/README.md)）。
音频只能走**明确标注的桥接扩展**（`audio-bridge.ts`，issue #30 冻结接口的 pin 副本）：

| 项 | 值 |
|---|---|
| 工具 | `audio_attach {"path": "<相对 PI_AUDIO_BRIDGE_ROOT 的文件名>"}` |
| 成功返回 | `{attached, mime, bytes, sha256, queueDepth, queuedBytes, note}`（无音频字节） |
| 失败 | 抛错（Pi 标记 failed tool result）：`E_AUDIO_ARGS / E_AUDIO_PATH / E_AUDIO_NOT_FOUND / E_AUDIO_NOT_FILE / E_AUDIO_SIZE / E_AUDIO_QUEUE / E_AUDIO_MIME / E_AUDIO_READ / E_AUDIO_MODEL` |
| 注入 | 下一次模型请求一次性注入（Gemini inlineData 原生音频形状）；不落会话/磁盘 |
| 限额 | 单文件 ≤4 MiB（`PI_AUDIO_BRIDGE_MAX_BYTES`）、排队 ≤8 MiB（`PI_AUDIO_BRIDGE_MAX_QUEUE_BYTES`） |
| 根目录 | `PI_AUDIO_BRIDGE_ROOT`（run 脚本设为 `<WS>/audio/inputs`） |

**不要**用 `@file` / `read` 读 `.wav`：Pi 会把字节当 UTF-8 文本（乱码），模型听不到。
输入导入用 bootstrap 的 `--audio`（复制到忽略目录并登记 sha256/时长/采样率）或 `--fixture`
（自有确定性 WAV，真值 440/660/880 Hz 上行与下行）。

## 6. SAM skill / CLI 与输出

- skill：`sam-audio`（`.pi/skills/sam-audio/`，`gh skill install --pin`，自包含 3 文件）。
- CLI：`python .pi/skills/sam-audio/scripts/audio_toolbox.py ...`（纯标准库）；SAM 路径/解释器
  取自 `local/config.json`（`sam_root`/`sam_python`），显式传 `--sam-root`/`--python`。
- 推荐顺序：`--help` → `sam check-environment` → `sam separate --dry-run`（读真实时长、校验
  anchors/FFmpeg/模型路径，**不加载模型、不用 GPU**）→（issue #33）真实分离。
- 输出存放：最终结构化结果约定 **`outputs/result.json`**（schema 由 **issue #32** 定义后对接；
  schema 未合并前不要自造格式并宣称兼容，过程结果写 `outputs/notes.md`）；SAM 运行产物写
  `outputs/runs/<run-id>/`（`target.wav`、`residual.wav`、`request.json`、`report.json`）。

## 7. 自检：doctor 与 smoke

```sh
python agentic/workspace-template/doctor.py --workspace "<WS>"            # 离线诊断
python agentic/workspace-template/doctor.py --workspace "<WS>" --deep    # + 真实 check-environment / dry-run
python agentic/workspace-template/doctor.py --workspace "<WS>" --json --redact   # 公开粘贴用（路径脱敏）
python agentic/workspace-template/smoke.py  --workspace "<WS>"           # 真实 Pi run（少量 API 费用）
python agentic/workspace-template/smoke.py  --workspace "<WS>" --print-argv      # 只看启动 argv（零 API）
```

- doctor 检查项：workspace 布局/设置模型固定/桥接 pin/skill 完整/音频 manifest hash/outputs 可写/
  node/pi 版本/Pi google 认证状态（只读状态）/gh skill list/SAM 路径与权重布局/FFmpeg（**文件级
  探测，≠ TorchCodec/GPU 推理**）/`gpu.inference`（标记 blocked，留给 #33）。退出码：`0` 无失败、
  `1` 有失败、`2` 用法。
- smoke（真实，约 3 个 assistant 请求）判据：① 事件流全部 assistant 消息 `model=gemini-3.8-flash`
  （**不传 `--model`**，证明 trust 加载了项目设置）；② system prompt `skills` 段含 `sam-audio`；
  ③ bash 真实执行 `audio_toolbox.py --help`；④ `audio_attach` 失败返回 `E_AUDIO_NOT_FOUND`
  （注册 + 错误契约）；⑤ 终态含 `SMOKE-DONE`。失败语义沿用 run_bounded：`2` 用法 / `3` 命令失败 /
  `4` 超时（杀进程）/ `5` provider 错误 / `6` 事件流空/乱码/不完整——**blocked 不写成 success**。

## 8. 清理与恢复

```sh
rm -rf <WS>                      # 整个 workspace 是本地忽略目录，直接删除即恢复干净状态
python .../bootstrap.py ...      # 重新部署（幂等）
```
- 只清运行产物：删 `<WS>/outputs/*`、`<WS>/sessions/*`、`<WS>/local/smoke/*`；音频/配置保留。
- 装坏 skill：删 `<WS>/.pi/skills/sam-audio` 后重跑 bootstrap（显式重装，不做静默修补）。
- 不需要清理任何“服务端文件”：音频是请求内联字节（inlineData），**不使用 Gemini Files API**。

## 9. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| doctor `pi.auth.google` = `not_ready` | Pi 的 Google 认证未就绪 | 用 Pi 自身认证流程配置；**不要**把 key 写进配置文件 |
| doctor `sam.weights` = fail | `model-cache/` 缺权重 | 按上游 `model-manifest.json` 手动补齐 + `scripts/verify_models.py` 校验；**不自动下载** |
| doctor `ffmpeg.detect` = fail | FFmpeg 不在 PATH / static 构建 | 装 FFmpeg 4–8 full-shared；注意它≠ GPU/TorchCodec 就绪 |
| `E_AUDIO_MODEL` | 当前模型不是 `google/gemini-3.8-flash` | 回到固定模型，**不要**绕过锁定 |
| `E_AUDIO_MIME` / `E_AUDIO_SIZE` / `E_AUDIO_QUEUE` | 非白名单容器 / 超 4 MiB / 排队超 8 MiB | 换音频或调整 `PI_AUDIO_BRIDGE_MAX_*` |
| `E_HASH_CONFLICT`（bootstrap，退出 3） | 导入音频与已有同名文件内容不同 | 换文件名或人工确认；绝不静默覆盖 |
| `E_WS_UNMANAGED`（bootstrap，退出 3） | 目标目录非空且非本工具部署 | 换空目录 |
| `Cannot find module './audio_guard.mjs'` | 桥接只拷了一半 | 用 bootstrap 部署（两个文件都 pin）；`doctor` 会指出 |
| PowerShell 启动无输出/命令没执行 | PS 5.1 把无 BOM 脚本当 ANSI，中文注释吞掉下一行 | `run.ps1` 已固定 ASCII + UTF-8 BOM；改脚本时保持该约定（测试有回归） |
| 路径含空格 | argv 被错误拼接 | 全部脚本用 argv 列表（无 shell 拼接）；测试覆盖“ws dir”路径 |

## 10. 已知限制（诚实口径）

- **Pi 原生音频 = unsupported**；桥接 ≠ 原生。桥接限制（一次性注入、模型锁定、不转码、
  `type:"image"` 承载音频 MIME）见 [`../audio-probe/README.md`](../audio-probe/README.md) §8。
- `PI_AUDIO_BRIDGE_ROOT` 只能由启动环境设置（settings 无 env 项）；直接裸跑 `pi` 不带 run 脚本
  时，`audio_attach` 会因根目录不对而报 `E_AUDIO_*`——按 §3 用 run 脚本启动。
- GPU/真实分离验证留给 **issue #33**；`outputs/result.json` 的 schema 留给 **issue #32**。
- smoke 的费用是 Pi 记账近似值，非账单真值；完整事件流只留本地 `local/smoke/`，公开材料只贴
  脱敏摘要（`--redact`）。
- 公开仓库不含：音频、权重、API key、个人绝对路径、完整会话。

## 11. 证据索引

- 逐条验收实证（真实运行）：[`reports/30-deploy-verification.md`](reports/30-deploy-verification.md)
- 冻结接口与 pin：[`manifest.json`](manifest.json)
- 上游接口：[`../toolbox/manifest.json`](../toolbox/manifest.json)（#29）、
  [`../audio-probe/README.md`](../audio-probe/README.md)（#30，§9 给下游的稳定接口）
- 离线测试：`tests/test_workspace_template.py`（18 项，0 网络 0 API）
