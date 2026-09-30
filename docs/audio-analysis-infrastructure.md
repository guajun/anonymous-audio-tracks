# 音频分析基础设施迁移说明

## 目录职责

`infrastructure/audio-analysis/gemini/` 是网络 API 工具：它管理提示词、文件上传、临时 Google Files 清理和 `anonymous-audio-tracks/v1` JSON 校验。每次真实请求写到一个新的运行目录，原始 `response.txt` 永远保留。

`infrastructure/audio-analysis/sam-audio/` 是本地 GPU 工具：它包含 SAM Audio 的可复用 Python 包、离线加载选项、文字/时间锚点推理脚本和模型 SHA-256 校验。它不包含 embedding、轨道聚类或 RGB 谱面生成。

## 旧路径映射

| 旧位置 | 仓库内位置 | 处理方式 |
|---|---|---|
| `F:\LED\gemini-audio\analyze.py` 和辅助脚本 | `infrastructure/audio-analysis/gemini/` | 复制并改为仓库相对默认路径 |
| `F:\LED\gemini-audio\prompt*.txt` | `infrastructure/audio-analysis/gemini/prompts/` | 保留通用模板和 Breeze 当前提示 |
| `F:\LED\gemini-audio\runs\`、`target-breeze.json` | `infrastructure/audio-analysis/gemini/history/` | 作为历史资料复制，未宣称为真值 |
| `F:\LED\sam-audio\sam_audio\` | `infrastructure/audio-analysis/sam-audio/sam_audio/` | 保留核心包和原 Meta 许可证 |
| `F:\LED\sam-audio\local_model_options.py` | `infrastructure/audio-analysis/sam-audio/` | 改为 `model-cache/` 和离线路径 |
| `C:\Users\MSI-NB\Documents\Codex\sam-audio-models` | `infrastructure/audio-analysis/sam-audio/model-cache/` | 本机副本；权重不进入 Git |
| `F:\LED\sam-audio\models` 联接 | 仍保留 | 原联接和源目录未修改，便于旧脚本回溯 |
| `F:\LED\sam-audio\clips\` | `data/audio-analysis/breeze/` | 本机样本复制；音频默认不提交 |

## 安全边界

真实 Gemini key 只能来自环境变量或未跟踪 `.env`；配置模板不含 key、token 或个人路径。SAM 不需要 API key，默认设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。模型下载由使用者显式完成，仓库只保存来源、大小和 SHA-256 清单。

不要把 `runs/`、`model-cache/`、`.env`、音频或任何模型权重用 `git add -f` 强行加入提交。提交前可检查：

```powershell
git status --short --ignored infrastructure/audio-analysis data/audio-analysis
rg -n -i "AIza|api[_-]?key|bearer|token|secret|C:\\Users|F:\\LED" infrastructure/audio-analysis --glob '!history/**'
```
