# issue #31 部署验证实证（真实运行；mock 测试另见 tests/）

日期：2026-09-30 ~ 10-01（UTC+8）。执行：sub-agentic-31（实现模型 openrouter/xiaomi/mimo-v2.6-pro，
未切换）。研究模型固定 `google/gemini-3.8-flash`（issue #30 冻结）。

**脱敏约定**：本文件不含音频/权重/key/完整会话/个人绝对路径。占位符：`<WORKSPACE>`（本地忽略
部署目录）、`<SAM_ROOT>`（本机迁移的 SAM checkout）、`<SAM_PYTHON>`、`<本地样本>`（真实迁移
音频，只在本地 `audio/inputs/manifest.json` 记录路径）。完整事件流仅本地 `<WORKSPACE>/local/smoke/`。

## 0. 真实 / mock 口径

| 部分 | 口径 |
|---|---|
| `tests/test_workspace_template.py`（18 项） | **离线 mock**：tmp 目录、`--skip-skill`、断 PATH；0 网络 0 API 0 GPU |
| 本文件 §2–§7 | **真实**：真机 Windows、真 `gh skill install`、真 Pi run（少量 API）、真 SAM dry-run |
| GPU 真实分离 | **blocked → issue #33**（未执行、未伪称成功） |

## 1. 环境（真实）

`pi 0.87.1`（native argv：node 24.19.0 + CLI entry）、`gh 2.95.0`（skill preview 可用）、
Python 3.14（脚本）/ SAM checkout `.venv`（上游环境）、FFmpeg full-shared 在 PATH、
`pi auth check --provider google --json` → `{"status":"ready",...}`（**只读状态，未读取/打印 key**）。

## 2. 部署实证（bootstrap，真实目录）

目标：`<WORKSPACE>`（本地忽略目录；`.local/` 已被仓库 .gitignore 覆盖，workspace 自身另有
`.gitignore` 兜底 audio/outputs/local/sessions）。命令（真实执行，路径已脱敏）：

```sh
python agentic/workspace-template/bootstrap.py --workspace "<WORKSPACE>" \
  --fixture --audio "<本地样本>.wav" --sam-root "<SAM_ROOT>" --sam-python "<SAM_PYTHON>"
```

- 首次：退出码 0；`gh skill install sam-audio @dfbc40a9541f… -> .pi/skills/sam-audio`（真网络安装，
  pin `dfbc40a9…`）；桥接 2 文件按 pin 落位；`audio/inputs/manifest.json` 3 条（fixture×2 + 真实样本×1，
  含 sha256/bytes/duration_s/sample_rate/channels/probe）。
- **二次运行（幂等）**：退出码 0；模板文件全部 `[skip]`（hash 一致）、用户改过的 `AGENTS.md`
  与 `local/config.json`（含哨兵 key）`[kept]` 未覆盖、已导入音频 `[skip]` 且 hash 一致；
  只重写本地 `local/bootstrap-report.json`（证据文件）。
- **hash 冲突**：同名不同内容导入 → 退出码 3 + `E_HASH_CONFLICT`，原文件 sha256 不变（实测比对）。
- **守卫**：非空非托管目录 → 退出码 3 + `E_WS_UNMANAGED`；音频不存在 → 退出码 2 + `E_AUDIO_NOT_FOUND`。
  全程无裸 traceback（离线测试断言 stderr 不含 `Traceback`）。

开发期发现并修复的真实缺陷（详见 §8）：子进程输出 GBK 解码崩溃、桥接依赖 `audio_guard.mjs`
漏部署、PowerShell 5.1 无 BOM 中文注释吞行——均有回归测试。

## 3. Windows 实机启动（真实，零 API）

| 命令 | 结果 |
|---|---|
| `powershell -NoProfile -ExecutionPolicy Bypass -File <WORKSPACE>\run.ps1 --version` | `0.87.1`，退出 0 |
| 同上 `--list-models gemini-3.8` | 列出 `google gemini-3.8-flash`（含 context/thinking/images 列） |
| `pwsh -NoProfile -File <WORKSPACE>\run.ps1 --version` | `0.87.1`，退出 0 |
| `bash <WORKSPACE>/run.sh --version`（git-bash） | `0.87.1`，退出 0 |

三个入口都以 `pi --approve` 启动（进程级 project trust）并设 `PI_AUDIO_BRIDGE_ROOT=<WORKSPACE>/audio/inputs`。

## 4. project trust 证据（真实）

- **机制探针（零 API）**：临时项目 `.pi/settings.json` 声明 `extensions: ["extensions/probe-ext.ts"]`
  （注册 CLI flag）。`pi --approve --help` → flag 出现；`pi --no-approve --help` → flag 消失。
  即：`.pi` 资源（settings/extensions/skills）由 project trust 门控（与官方 security.md 一致），
  `--approve` 是进程级、不写 `~/.pi/agent/trust.json`、不改全局设置/认证。
- **workspace 真实证据（§5 smoke）**：启动命令**不带** `--model`/`--provider`（本机全局默认是
  其他 provider/模型），事件流中全部 assistant 消息 `model = gemini-3.8-flash` → 模型来自
  trust 加载的项目 `.pi/settings.json`；同一事件流 system prompt `tools` 段含
  `audio_attach: Attach a local audio file for listening (project audio bridge)` → 桥接扩展
  也是经 trust 的 settings 加载。

## 5. 真实 Pi smoke（1 次 run；少量 API 费用）

`python agentic/workspace-template/smoke.py --workspace "<WORKSPACE>" --redact`

判据（全部 pass，逐条对应 issue 验收“Pi 使用指定 Gemini 模型发现 skill，SAM dry-run 能执行”的 Pi 侧）：

| 判据 | 真实证据（事件流抽取） |
|---|---|
| model | 全部 assistant 消息 `model=['gemini-3.8-flash']`（未传 `--model`） |
| skill | system prompt `skills` 段含 `sam-audio`；agent 回复报告该 skill |
| sam_cli | `bash` 工具真实执行 `python .pi/skills/sam-audio/scripts/audio_toolbox.py --help`，结果含 `usage: audio-toolbox [-h] [--version] GROUP ...` |
| bridge_tool | `audio_attach {"path": "smoke-missing.wav"}` → **failed tool result** `E_AUDIO_NOT_FOUND: audio file not found inside the controlled audio root` |
| final_line | 终态含 `SMOKE-DONE skills=sam-audio help=usage: audio-toolbox … err=E_AUDIO_NOT_FOUND` |

- 计费口径：一次 Pi run = **4 条带 usage 的 assistant 请求**（非 1 次 API 调用）。本 run：
  `totalTokens=14221`、`cost=0.012997`（Pi 记账近似值，**非账单真值**）。
- 失败语义：run_bounded 硬超时默认 300s（超时杀真实 Pi 进程、退出 4、保留产物）；命令失败退出 3；
  provider error/aborted 退出 5；事件流空/乱码/不完整退出 6——**不把 blocked/fail 记为 success**。
- 开发期另有 1 次真实 run 因 runner 接线缺陷（事件文件指错）被误判 fail，但 Pi 实际完成：
  `totalTokens=14612`、`cost≈0.0138`（保留诚实计数）。本 issue 真实 smoke 总支出 ≈ 0.027（Pi 记账）。
- 离线测试与 doctor **0 次**模型调用；音频**未**上传任何 API（smoke 不挂载音频，只走错误契约）。

## 6. SAM CLI 真实验证（check-environment + dry-run；不加载模型、不跑 GPU）

`python agentic/workspace-template/doctor.py --workspace "<WORKSPACE>" --deep --redact`

- `audio-toolbox sam check-environment`（真跑，wrapper 8 项）→ 通过（sam_root/sam_entry/model_dir/
  text_encoder_dir/model_manifest/python/ffmpeg_preflight/offline_policy）。
- `audio-toolbox sam separate --audio <…> --description "melodic sound" --dry-run`（真跑）→ 通过：
  `duration_s=30.0`（真实样本时长由上游读出）、plan 合法、`device=cuda` 为 plan 值（**未执行推理**）。
- doctor 全量结论：`0 项失败`，`gpu.inference` 显式 `blocked`（留给 #33）。
- 口径强调：`ffmpeg.detect` 是**文件/PATH 级探测**，≠ TorchCodec/GPU 推理就绪；两者在输出中分列。

## 7. 逐条 issue 验收对照

| 验收项 | 结论 | 证据 |
|---|---|---|
| 从空本地目录按说明部署成功；已部署再运行不会覆盖用户输入/配置 | **满足** | §2（首跑成功、二跑 `[skip]/[kept]`、冲突退出 3 不覆盖；README §2） |
| Pi 使用指定 Gemini 模型发现 skill，SAM dry-run 能执行 | **满足** | §4–§6（模型来自 trust 的 settings；skills 段含 sam-audio；dry-run 真跑通过） |
| 音频与 outputs/secret 全被 ignore，配置模板只使用 env 或本地配置 | **满足** | 模板 `.gitignore`（audio/outputs/local/sessions/.pi/skills/.pi/extensions/*.wav/…）；`local/config.json` 只存本机路径；模板无 key；tests 断言 |
| 提供 smoke tests/doctor 和逐条部署实证；为任务 5 返回稳定命令及 workspace 路径 | **满足** | `tests/`（18 项离线）+ `smoke.py` + `doctor.py` + 本文件；稳定命令见 README §2/§3/§7 |

## 8. 开发期真实缺陷（诚实记录，均已修复 + 回归测试）

1. Windows 子进程输出用系统 GBK 解码 → `UnicodeDecodeError` 崩 reader 线程；改为 `utf-8 + replace`。
2. 桥接扩展 `import ./audio_guard.mjs`，只部署 `audio-bridge.ts` 会 `Cannot find module` →
   冻结接口改为 **2 文件 pin**（manifest `bridge.files`），doctor 校验两个 hash。
3. Windows PowerShell 5.1 把**无 BOM** `.ps1` 当 ANSI 读：奇数长度的 UTF-8 中文注释行会吞掉
   **下一行**（`pi` 启动行被“注释掉”，无输出且退出 0）→ `run.ps1` 固定 ASCII + UTF-8 BOM，测试回归。
4. `pi …` 作为 `-File` 脚本最后一条语句时子进程 stdout 被吞 → `run.ps1` 末尾显式 `exit $LASTEXITCODE`。
5. smoke 首版把 `--events` 指向空文件（pi `--mode json` 的事件流即 stdout）→ run_bounded 误判
   provider failure；改为同一文件作 `--stdout/--events`，并把判据从“全文 grep”改为结构化抽取
   （assistant 消息 model 字段、system skills 段、tool_execution_end 结果）。

## 9. 限制与 blocked

- GPU/真实分离 → **blocked：issue #33**（未执行）；`outputs/result.json` schema → **issue #32**
  （本项只约定路径，未发明 schema）。
- 桥接限制（一次性注入、模型硬锁、不转码、`type:"image"` 承载音频、单文件 4 MiB/排队 8 MiB）
  沿用 issue #30 §8；`PI_AUDIO_BRIDGE_ROOT` 由启动脚本设置（settings 无 env 项），裸跑 `pi`
  会得到可读的 `E_AUDIO_*` 错误。
- 本机真实样本只在本地登记（sha256/时长/采样率），公开材料不含音频与绝对路径。
