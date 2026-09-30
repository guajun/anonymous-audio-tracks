# 00 · 环境与命令记录（issue #30）

所有命令在 `agentic/audio-probe/`（repo worktree `F:/…/issue-30`，以下记 `<REPO>`）执行。
本文件只记录状态与命令；不含 key、不含个人绝对路径（绝对路径已脱敏为 `<REPO>`/`<HOME>`）。

## 环境（OFFLINE，无 API 调用）

```
$ pi --version
0.87.1

$ pi auth check --provider google --json
{"status":"ready","provider":"google","authType":"api_key"}     # 仅状态，不打印 key

$ pi --list-models google | (head -1; grep gemini-3.8-flash)
provider    model                                      context  max-out  thinking  images
google      gemini-3.8-flash                           1.0M     65.5K    yes       yes
```

要点：
- 精确模型 `google/gemini-3.8-flash` 存在于 **google** provider（研究模型固定值，未替换）。
- `--list-models` 只有 `images` 能力列，**没有 audio 列**（能力矩阵的 CLI 层证据之一）。
- 认证：`authType=api_key`，`status=ready`；全过程未执行 `pi auth print`，未输出任何凭据。

## 本目录执行过的命令汇总

| 命令 | 类型 | 结果 |
|---|---|---|
| `uv run pytest tests -v`（或仓库根 `uv run pytest agentic/audio-probe/tests -v`） | OFFLINE | 8 passed |
| `python probe/make_fixture.py --all` | OFFLINE | fixture-a.wav / fixture-b.wav 各 24044 B |
| `node probe/native_paths_probe.mjs --file fixtures/fixture-a.wav` | OFFLINE | 见 20 |
| `node probe/payload_shape.mjs` | OFFLINE | 见 20 |
| `bash probe/run_real_native.sh` | **REAL**（1 次模型调用） | 见 30 |
| `bash probe/run_real_bridge.sh` | **REAL**（1 次模型调用） | 见 40 |
| `python probe/extract_evidence.py tmp/*.jsonl` | OFFLINE | 脱敏证据 |

真实调用预算：**2 次模型调用**（每次一个脚本、一次运行；脚本失败即停并保留 stderr，不重试循环）。
SAM GPU 不在本项运行（依 issue 约定）。
