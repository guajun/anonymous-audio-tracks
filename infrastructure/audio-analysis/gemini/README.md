# Gemini 音频分析

这是独立的 Gemini 音频分析工具。它使用 Google 官方 `google-genai` SDK，将参考片段和完整片段上传给 `gemini-3.8-flash`，把原始回答和通过 `anonymous-audio-tracks/v1` 基本校验的 `tracks.json` 保存到一个新的运行目录。它不依赖 SAM Audio，也不会加载本地模型。

## 设置

在本目录创建未跟踪的 `config.toml`（可由 `config.example.toml` 复制），按需改音频、提示词和输出路径。API key 只通过环境变量或本目录未跟踪的 `.env` 提供：

```powershell
Copy-Item config.example.toml config.toml
Copy-Item .env.example .env
$env:GEMINI_API_KEY = "<your-key>"
```

`config.toml` 中的相对路径以本目录为基准；当前 Breeze 样本位于仓库的 `data/audio-analysis/breeze/`。样本音频在本地迁移时复制了旧目录内容，但由于音频和运行结果属于本地资料，默认不会提交 Git。

## 运行

先做不联网检查：

```powershell
uv run --project . python analyze.py --dry-run
```

真实请求：

```powershell
uv run --project . python analyze.py
```

也可以用 `--audio`、`--reference`、`--prompt`、`--model` 和 `--auxiliary` 覆盖配置。`--auxiliary` 只用于辅助辨认，时间事件仍以原始完整片段为准。失败的模型原文会保留在 `response.txt`，不会被脚本猜测修复。

审计一个已保存结果（只标记可能需要试听复核的重复模式）：

```powershell
uv run --project . python scripts/audit_tracks.py runs/<timestamp>/tracks.json
```

离线单元测试不创建网络客户端：

```powershell
uv run --project . python -m unittest discover -s tests -p "test_*.py"
```

## 迁移关系

代码和提示词来自旧目录 `F:\LED\gemini-audio`；历史运行记录复制到 `history/runs/`，原目录未修改。旧的 `.env`、虚拟环境和缓存没有复制。迁移只整理工具，不声称历史输出是听觉真值。
