# SAM Audio 工具箱安装与调用说明（issue #29）

固定版本见 [`manifest.json`](manifest.json)：toolbox 仓库 `guajun/agentic-audio-toolbox`
pin `f8eeac443aaf9d8aa0f7ded0d26783abcb517a48`；SAM 上游
`guajun/anonymous-audio-tracks@infrastructure/audio-analysis/sam-audio` commit
`c603de8794cc16880dc01be0f1e868f6c2845417`（Meta SAM License，权重不分发）。
本文件中的路径一律用占位符：`<SAM_ROOT>`（SAM checkout 根目录）、`<PYTHON>`
（该 checkout 的解释器）、`<AUDIO>`（输入音频）、`<RUN_DIR>`（输出目录）、
`<TMP>`（临时目录）。**不要把真实绝对路径、音频、权重或密钥写入 git。**

## 1. 新手复现：安装 skill

在哪运行：任意已 `git init` 的项目根目录（本文用 `<PROJECT>` 表示）。

```sh
cd <PROJECT>
gh skill install guajun/agentic-audio-toolbox sam-audio \
  --agent pi --scope project --pin f8eeac443aaf9d8aa0f7ded0d26783abcb517a48
```

成功判据：命令输出 `✓ Installed sam-audio (from guajun/agentic-audio-toolbox@608e59a5...) in .pi\skills`，
且 `gh skill list` 出现一行 `sam-audio  pi  project  guajun/agentic-audio-toolbox`。

安装结果（自包含，仅 3 个文件，无需源仓库其它文件）：

```text
.pi/skills/sam-audio/
├── SKILL.md                     # Agent 操作说明
├── references/cli-reference.md  # 稳定 JSON schema / 错误码 / 环境变量
└── scripts/audio_toolbox.py     # CLI 实现（纯标准库）
```

常见报错：

| 报错 | 原因 | 处理 |
|---|---|---|
| `unknown command "skill"` | `gh` 版本过旧（skill 为 preview） | 升级 GitHub CLI |
| `! Skills are not verified by GitHub ...` | 正常的安全提示 | 用 `gh skill preview ...` 人工审阅后再用 |
| 安装到非 git 目录失败/落到其它位置 | project scope 依赖项目根 | 先 `git init` 或进入已有仓库根目录 |

## 2. 调用 CLI

依赖：wrapper 仅需 Python 3.11+（标准库）。**真实分离**另需 SAM checkout 的锁定环境
（`uv sync --locked`，Python 3.12、torch/CUDA）、Windows FFmpeg 4–8 **full-shared**
构建在 PATH、以及本地权重（`<SAM_ROOT>/model-cache/...`，字节/SHA-256 见上游
`model-manifest.json`，用上游 `python scripts/verify_models.py` 校验）。

在哪运行：任意目录；CLI 用 `--sam-root`/`SAM_AUDIO_ROOT` 定位 SAM checkout，
用 `--python` 指定带 SAM 环境的解释器。

```sh
# 0) 帮助（不加载 GPU/torch）
python <PROJECT>/.pi/skills/sam-audio/scripts/audio_toolbox.py --help

# 1) 环境预检（不加载 torch、不用 GPU、不联网）
python <PROJECT>/.pi/skills/sam-audio/scripts/audio_toolbox.py sam check-environment \
  --sam-root "<SAM_ROOT>" --python "<PYTHON>"

# 2) dry-run（读取真实音频时长、校验 anchors/FFmpeg/模型路径，仍不加载模型）
python <PROJECT>/.pi/skills/sam-audio/scripts/audio_toolbox.py sam separate \
  --audio "<AUDIO>" --description "melodic sound" --anchor 11.18,11.50 \
  --dry-run --sam-root "<SAM_ROOT>" --python "<PYTHON>"

# 3) 真实分离（GPU，分钟级；本任务不执行，留给 #33）
python <PROJECT>/.pi/skills/sam-audio/scripts/audio_toolbox.py sam separate \
  --audio "<AUDIO>" --description "melodic sound" --anchor 11.18,11.50 \
  --output-dir "<RUN_DIR>" --sam-root "<SAM_ROOT>" --python "<PYTHON>" --timeout 3600
```

输入/输出：

- 输入：`--audio`（本地音频文件）、`--description`（小写名词/动词短语）、可重复
  `--anchor START,END`（秒，`0 <= START < END <= 时长`）。
- 输出：stdout 恰好一个 JSON（schema `audio-toolbox.sam/v1`，含 `ok/action/exit_code/
  command/parameters/upstream/run_dir/outputs/report|plan/error`），失败时 stderr 一行摘要；
  真实运行在 `<RUN_DIR>` 生成 `target.wav`、`residual.wav`、`request.json`、`report.json`。

成功判据：退出码 0 且 JSON `ok=true`。dry-run 成功**不代表**已分离；只有
`outputs` 四个产物齐全才算真实分离结果。

退出码（稳定）：`0` 成功 / `2` 用法错误 / `3` 输入错误 / `4` 环境错误 /
`5` 上游 SAM 入口失败 / `6` 超时（进程已终止）/ `7` 产物缺失。
错误码与字段见 skill 内 `references/cli-reference.md`。

常见报错：

| 错误码 | 常见原因 | 处理 |
|---|---|---|
| `E_AUDIO_NOT_FOUND` | 路径拼错或未加引号（路径含空格必须引号） | 修正 `--audio` |
| `E_ANCHOR_FORMAT` / `E_ANCHOR_INVALID` | `--anchor` 不是 `START,END`，或 `START >= END`/负数 | 按格式改写 |
| `E_ENVIRONMENT` | `--sam-root` 未设/入口缺失/解释器不对/权重不在 `model-cache/` | 设置 `SAM_AUDIO_ROOT`、`--python`，补权重 |
| `E_UPSTREAM_FAILED` | CUDA 不可用、FFmpeg 缺共享 DLL（Windows static 构建不够）、音频不可读 | 看 `upstream.stderr_tail` 与 `error.detail.upstream_exit_code` |
| `E_UPSTREAM_TIMEOUT` | 长音频/慢设备超过 `--timeout` | 调大 `--timeout` 或截短输入 |
| `E_OUTPUT_MISSING` / `E_PLAN_INVALID` / `E_REPORT_INVALID` | 上游声称成功但产物缺失、dry-run 无合法 plan JSON、或 `report.json` 损坏/非对象 | 查 `upstream.stdout_tail`，检查磁盘空间/上游输出；此类结果不得当作成功 |

安全约定：wrapper 只用 argv 列表 + `shell=False` 调用上游（无 shell 注入面），
强制 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`，不下载、不上传、不读取任何密钥。