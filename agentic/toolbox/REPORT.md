# SAM Audio 工具箱封装报告（issue #29）

所有路径以占位符记录：`<SAM_ROOT>`（SAM checkout）、`<PYTHON>`（该 checkout 的
解释器）、`<AUDIO>`（输入音频）、`<TMP>`（临时目录）、`<PROJECT>`（安装 skill 的项目）。
**报告不含真实绝对路径、音频、权重、API key 或完整会话日志。**

## 1. 交付物

| 位置 | 内容 |
|---|---|
| toolbox 仓库 `guajun/agentic-audio-toolbox`（branch `agentic/issue-29-sam-toolbox`） | `skills/sam-audio/`（SKILL.md + references/cli-reference.md + scripts/audio_toolbox.py）、`bin/audio-toolbox(.cmd)`、`tests/`、MIT `LICENSE`、README |
| 本仓库 `agentic/toolbox/` | `manifest.json`（pin/固定来源/许可/权重策略）、`INSTALL.md`（安装/调用/新手复现）、`REPORT.md`（本报告）、`tests/test_manifest.py` |

设计取舍：**轻量 subprocess wrapper**，不复制 SAM 源码、不引入重复的巨大 SAM 依赖
（wrapper 仅 Python 标准库）；只以 argv 列表 + `shell=False` 调用固定上游入口
`<sam-root>/scripts/run_inference.py`；稳定机器输出 schema `audio-toolbox.sam/v1` 与
稳定退出码（0/2/3/4/5/6/7）；skill 目录自包含（安装后不依赖源仓库其它文件）。

## 2. 固定来源与许可

- toolbox pin：`dfbc40a9541f686207b65b93b1332bb505654261`（toolbox PR #1 squash
  合并到 `main` 的 merge commit；安装即 pin 该 commit，无需正式 release）。
- SAM 上游：`guajun/anonymous-audio-tracks@infrastructure/audio-analysis/sam-audio`
  commit `c603de8794cc16880dc01be0f1e868f6c2845417`（branch `infra/audio-analysis-migration`，
  对应 PR #26，不修改）。
- 许可：wrapper 代码 MIT；SAM Materials（`sam_audio` 包、入口脚本、
  `facebook/sam-audio-*` 权重）保留上游 **Meta SAM License**（`SAM-LICENSE.txt`），
  不随 toolbox 分发。
- 权重策略：不提交、不分发、不自动下载；本地 `model-cache/` 布局与字节/SHA-256
  以上游 `model-manifest.json` 为准，用上游 `scripts/verify_models.py` 校验。

## 3. 测试记录（真实执行，路径脱敏）

### 3.1 自动测试（无 GPU / 无 torch / 无网络；fake SAM entry）

命令（toolbox worktree）：`python -m unittest discover -s tests -v`

结果：`Ran 44 tests in 2.264s — OK`。

覆盖：`--help`/usage 表面、结构化错误与退出码、anchor 校验（格式/数值/时长边界由
上游校验并透传）、含空格与 shell 元字符路径的 argv 精确透传（无注入，源码扫描断言
无 `shell=True`/`os.system`/`eval`/`exec`）、timeout 终止、upstream 失败透传、
dry-run plan 校验（超长 plan / 无 JSON / 字段不全 / 嵌套片段不误认）、run-dir 与
产物收集、损坏/非对象 `report.json` 拒绝、离线环境变量强制、UTF-8 解码（非法字节
不崩溃、非 ASCII 路径、子进程 stdout 恰好一个 JSON 文档）、环境预检、skill
frontmatter/自包含布局。

### 3.2 `gh skill` 校验

命令：`gh skill publish --dry-run`
结果：`Dry run complete. Use without --dry-run to publish.`（仅 tag-protection
提示 warning；frontmatter/命名/安装元数据校验通过）。

### 3.3 真实环境 dry-run / preflight（不加载模型、不使用 GPU）

使用现有离线 SAM checkout（`<SAM_ROOT>`，含锁定 venv `<PYTHON>`）与真实音频
`<AUDIO>`（30s）：

| 命令（脱敏） | 结果 |
|---|---|
| `python ... audio_toolbox.py sam check-environment --sam-root <SAM_ROOT> --python <PYTHON>` | `ok=true, exit_code=0`；8/8 检查通过（sam_root/sam_entry/model_dir/text_encoder_dir/model_manifest/python/ffmpeg_preflight/offline_policy），upstream preflight `exit 0` |
| `python ... audio_toolbox.py sam separate --audio <AUDIO> --description "melodic sound" --anchor 11.18,11.50 --dry-run --sam-root <SAM_ROOT> --python <PYTHON>`（继承 `HF_HUB_OFFLINE=0`/`TRANSFORMERS_OFFLINE=0`） | `ok=true, exit_code=0`；plan 含 10 个必填字段、`duration_s=30.0`、`network="disabled"`；证明离线标志即使被父环境置 0 也在子进程强制为 1 |
| 同上但 `--anchor 40,41`（超时长，失败输入实证） | `exit 5`、`error.code=E_UPSTREAM_FAILED`、`upstream.exit_code=2`，`stderr_tail` 含上游校验 `require finite 0 <= START < END <= audio duration (30s)` |

### 3.4 远程 skill 隔离安装（pin 真实推送 commit）与安装后可用

临时 git 仓库 `<TMP>`：

```sh
gh skill install guajun/agentic-audio-toolbox sam-audio \
  --agent pi --scope project --pin dfbc40a9541f686207b65b93b1332bb505654261
```

结果：
- `Using ref dfbc40a9541f686207b65b93b1332bb505654261 (dfbc40a9)` → `✓ Installed sam-audio (from guajun/agentic-audio-toolbox@dfbc40a9541f686207b65b93b1332bb505654261) in .pi\skills`
- `gh skill list` → `sam-audio  pi  project  guajun/agentic-audio-toolbox`
- 安装目录自包含（仅 `SKILL.md`、`references/cli-reference.md`、`scripts/audio_toolbox.py`），
  从安装目录直接可用：
  - `python .pi/skills/sam-audio/scripts/audio_toolbox.py --version` → `ok=true, exit_code=0`
  - `... sam check-environment --sam-root <SAM_ROOT> --python <PYTHON>` → `ok=true`，全部检查通过
  - `... sam separate --audio <AUDIO> --description "melodic sound" --anchor 11.18,11.50 --dry-run ...` → `ok=true`，plan 10 字段、`network=disabled`
- **Pi skill 发现**：隔离目录内 `pi -a --model openrouter/xiaomi/mimo-v2.6-pro --print "…"` →
  系统提示中广告的 skill 名为 `sam-audio`。

### 3.5 主仓库集成测试

命令（本 worktree）：`python -m unittest discover -s agentic/toolbox/tests -v`
结果：`Ran 7 tests in 0.671s — OK`。校验内容：manifest pin/固定来源/退出码映射/
文档引用；以及 **git 跟踪列表卫生**（`git ls-files`，仅判断受控跟踪文件：禁止
tracked 音频/权重/密钥/个人路径/超大文件）。按 review 指令，卫生检查不再扫描
磁盘目录内容：#30/#31 在 gitignored 目录（`.local/`、`outputs/` 等）生成的合法
fixture/音频/会话不会被误判为 committed；并新增 temp repo 回归：被忽略的本地
fixture 不误报、真正被跟踪的二进制被拒绝、被跟踪的密钥被拒绝。根锁文件未改动
（改动仅 `agentic/toolbox/`）。

### 3.6 合并后修订（主仓库 PR review 指令）

- toolbox 已按授权 squash 合并（PR #1，merge commit `dfbc40a9541f686207b65b93b1332bb505654261`；
  合并前校验 HEAD 未变、mergeable、`--match-head-commit`，未用 `--admin`、未删分支；
  PR body 跨库 `Closes #29` 已改为 `Refs #29`，#29 由主仓库 PR 关闭）。
- 最终 pin 更新为该 merge SHA（manifest/INSTALL/REPORT/PR body），并在临时 git 仓库
  重新做远程隔离安装验证（见 3.4，pin 即 `dfbc40a…`）。
- 修正 INSTALL.md 成功判据输出残留旧 pin 的问题（现为 `Using ref dfbc40a9…` /
  `@dfbc40a9...`）。
- `test_manifest.py` 卫生检查改为 `git ls-files` 受控跟踪列表（见 3.5）。

## 4. Review 修复记录（PR #1 changes requested）

针对主 Agent 实测发现的 4 项（评论：`guajun/agentic-audio-toolbox/pull/1#issuecomment-5917269794`）：

1. **离线策略未强制**：`setdefault` 会继承父环境 `HF_HUB_OFFLINE=0`/`TRANSFORMERS_OFFLINE=0`。
   修复：子进程环境无条件置 `1`（父环境不变），回归测试用 fake entry 读取子进程视角断言。
2. **截断 stdout 解析 plan**：>2000 字符的合法 plan 会得到 `plan:null` 却 `ok:true`。
   修复：用完整 stdout 解析 plan（诊断 tail 仍限 2000 字符、不进结果 JSON）；要求
   plan 含全部 10 个顶层字段，缺失/损坏/嵌套片段 → `E_PLAN_INVALID`（exit 7）。
3. **损坏 report 仍报成功**：修复为 `E_REPORT_INVALID`（exit 7），非对象 JSON 同样拒绝；
   fake 成功用例均写合法 JSON report。
4. **Windows 解码依赖 locale**：`text=True` 在 GBK 下对不可解码字节触发 reader 线程
   traceback 并丢输出。修复：捕获 bytes、UTF-8+replace 解码，子进程 `PYTHONIOENCODING=utf-8`；
   测试覆盖非法字节/非 ASCII 路径/单 JSON 文档承诺，无 traceback。

修复 commit：`f8eeac443aaf9d8aa0f7ded0d26783abcb517a48`（内容经 squash 并入
merge commit `dfbc40a9541f686207b65b93b1332bb505654261`）；测试数 34 → 44。

## 5. 未验项与限制（不写成完成）

- **真实 GPU 推理/真实分离产物未执行**（`target.wav`/`residual.wav` 的真实分离），
  按约定留给 #33 统一执行；本报告中 dry-run/preflight/失败输入均不冒充真实分离。
- `gh skill publish` 正式发布未执行（非必需；固定 SHA 安装已验证满足 `gh skill`
  支持）。pin 已更新为 toolbox merge commit `dfbc40a9541f686207b65b93b1332bb505654261`。
- 上游 `--check-environment` 不保证 TorchCodec ABI / CUDA 驱动兼容性（上游文档已声明），
  本 wrapper 不额外验证。
- 自动测试与真实模型证据分开记录（3.1 为 fake entry 自动测试；3.3/3.4 为真实入口但
  无模型加载）。

## 6. 链接

- Toolbox PR: https://github.com/guajun/agentic-audio-toolbox/pull/1（已 squash 合并，merge commit `dfbc40a9541f686207b65b93b1332bb505654261`）
- 主仓库 PR: https://github.com/guajun/anonymous-audio-tracks/pull/36（`agentic/issue-29` → `agentic/minimal-agent`）
- Issue: https://github.com/guajun/anonymous-audio-tracks/issues/29（Closes #29）