# SAM Audio 本地推理

这是与 Gemini 完全独立的 SAM Audio 基础设施。这里保留 Meta 发布的 `sam_audio` 推理包、一个本地离线推理入口和模型完整性校验；不包含 embedding、轨道聚类或 RGB 谱面逻辑。

## 依赖和设备

建议使用 Python 3.12、CUDA GPU 和 `uv`。当前 Windows 锁定 PyTorch 2.10.0+cu130、torchaudio 2.10.0+cu130 和 torchvision 0.25.0+cu130；如果你的驱动不支持 CUDA 13，请按本机驱动选择兼容的 CUDA wheel，并同步调整 `pyproject.toml`/`uv.lock`。SAM Audio small 加上 T5-base 权重约 5.9 GB，实际推理还需要显存和临时空间。安装环境：

```powershell
Copy-Item config.example.toml config.toml
uv sync --locked
```

大型权重不受 Git 管理。默认目录是：

```text
infrastructure/audio-analysis/sam-audio/model-cache/
├── sam-audio-small/checkpoint.pt
└── t5-base/model.safetensors
```

本机迁移时，权重从旧的 `C:\Users\MSI-NB\Documents\Codex\sam-audio-models` 复制到这个目录；旧目录和 `F:\LED\sam-audio\models` 联接保持不变。新目录中的权重仍由 `.gitignore` 排除。没有权重时，可以按 SAM Audio / ModelScope 对应仓库下载 `facebook/sam-audio-small` 和 `AI-ModelScope/t5-base` 的文件，再运行：

```powershell
uv run --project . python scripts/verify_models.py
```

脚本逐文件检查字节数和 `model-manifest.json` 中的 SHA-256；它不会反序列化 checkpoint，也不会自动下载或上传任何内容。

## 单独运行推理

只做路径和权重检查，不加载 GPU、不联网：

```powershell
uv run --project . python scripts/run_inference.py `
  --audio ..\..\..\data\audio-analysis\breeze\Breeze-first30s.wav `
  --description "melodic sound" `
  --anchor 11.18,11.50 `
  --dry-run
```

真实运行（时间锚点可以省略，省略时是纯文字提示）：

```powershell
uv run --project . python scripts/run_inference.py `
  --audio ..\..\..\data\audio-analysis\breeze\Breeze-first30s.wav `
  --description "melodic sound" `
  --anchor 11.18,11.50
```

也可以从本目录执行 `./scripts/run.ps1 --audio ... --description ...`，它只是上述命令的 PowerShell 包装，不会改变模型或输出位置。

每次运行会在 `runs/<timestamp>/` 产生 `target.wav`、`residual.wav`、`request.json` 和 `report.json`。`--model-dir`、`--text-encoder-dir`、`--device` 和 `--dtype` 可覆盖默认配置；脚本设置 Hugging Face/Transformers offline 标志，缺少本地文件时直接失败。SAM Audio 运行不读取 Gemini 的 key，也不读取 Gemini 的历史结果。

## 迁移关系和许可证

`sam_audio/`、`model-manifest.json` 和 `SAM-LICENSE.txt` 从旧目录 `F:\LED\sam-audio` 复制而来；旧仓库未修改。没有复制旧的虚拟环境、日志、notebook、评估数据或结果输出。SAM Materials 继续受随附的 Meta SAM License 约束。
