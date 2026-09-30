# Agentic anonymous audio track

总研究日志：[issue #27](https://github.com/guajun/anonymous-audio-tracks/issues/27)。
本路线探索具有音频理解能力的 Agent 自主调用 SAM Audio 与 DSP 工具，产生按乐器来源组织的 onset JSON，供本地可视化审阅；不替代旧 E/P 训练路线，不预先承诺准确率。

## 分支 / 首轮实施

- 研究分支 `research/agentic-anonymous-audio-track`，从 `main@eb05eb93300960c7e879330920e53367233fe724` 创建。
- 最小 Agent 集成分支 `agentic/minimal-agent`，父实施 [#28](https://github.com/guajun/anonymous-audio-tracks/issues/28)。实施 PR 先合此分支，后通过集成 PR 合研究分支；不自动合 main。
- 独立工具箱 [guajun/agentic-audio-toolbox](https://github.com/guajun/agentic-audio-toolbox)。SAM 来源为现有 `infra/audio-analysis-migration`，固定版本，不改 PR #26。

| 阶段 | 子 issue | 交付 |
| --- | --- | --- |
| A（并行） | [#29](https://github.com/guajun/anonymous-audio-tracks/issues/29)、[#30](https://github.com/guajun/anonymous-audio-tracks/issues/30) | SAM skill/CLI、Pi + Gemini 3.8 Flash 音频能力探针 |
| B（A 合并后并行） | [#31](https://github.com/guajun/anonymous-audio-tracks/issues/31)、[#32](https://github.com/guajun/anonymous-audio-tracks/issues/32) | 可复现 Pi workspace、版本化 JSON schema |
| C（B 合并后并行） | [#33](https://github.com/guajun/anonymous-audio-tracks/issues/33)、[#34](https://github.com/guajun/anonymous-audio-tracks/issues/34) | 真实 Agent/SAM 端到端、onset/波形/BPM/缩放网页 |

GitHub 使用原生 sub-issues 与 blocked-by，不仅靠此表。具体目标、步骤、验收与最新状态以 issue 为准。

## 开发 / 证据纪律

每项独立 branch/worktree/PR；Pi 实现 worker 使用 `openrouter/xiaomi/mimo-v2.6-pro`，研究音乐 Agent 使用 `google/gemini-3.8-flash`。主 Agent 实际 review diff、测试与证据，评论记录当前 HEAD 后放行 merge。不要自行合并。

日志包含命令、模型/代码/工具 pin、输入 hash、真实与 mock 区分、native/bridge 区分、结果、失败与限制。Gemini 支持音频不等于 Pi 原生支持音频；桥接不能写成原生。onset 是可听起音，不自动表示 MIDI note、音高或 duration。

音频、模型、密钥、个人路径、完整会话及未脱敏日志只留本地忽略目录。SAM 默认离线；API 音频上传只用于获准的模型分析。重型 GPU 推理串行。无法真实运行时保留 blocked 状态，不以 mock 关闭。

## 人类验收（尚待执行）

最终提供本地网页启动命令、真实 JSON/音乐加载路径、需检查的乐器与时间点，以及 BPM 调节、滚轮缩放操作。工程自动检查与人类对显示效果/每个 issue 执行详情的验收分开记录。研究总 issue 长期保持开放。
