# 音频分析基础设施

这里集中放置两套可以单独运行的本地工具：

| 工具 | 用途 | 入口 | 外部要求 |
|---|---|---|---|
| `gemini/` | 上传参考片段和完整片段，让 Gemini 输出匿名声源轨迹 JSON | `gemini/analyze.py` | Gemini API key 和网络 |
| `sam-audio/` | 用文字和可选时间锚点做本地声音分离 | `sam-audio/scripts/run_inference.py` | 本地权重、Python/CUDA 环境 |

两者只共享 `data/audio-analysis/` 中的音频样本约定，不共享运行时环境、key、模型或输出目录。Gemini 不会加载 SAM；SAM 默认离线，不会调用 Gemini。

## 迁移原则

本目录是从 `F:\LED\gemini-audio` 和 `F:\LED\sam-audio` 复制整理的副本。旧文件没有覆盖、移动或删除；旧路径和本次迁移的关系写在各子目录 README 及 [docs/audio-analysis-infrastructure.md](../../docs/audio-analysis-infrastructure.md) 中。历史 Gemini 运行记录只作为可追溯资料，不能当作准确率或听觉真值。

大型权重和本地音频都通过忽略规则排除在 Git 提交之外；仓库只提交下载清单、SHA-256 和可复现的脚本。
