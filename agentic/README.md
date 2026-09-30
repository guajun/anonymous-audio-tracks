# 最小 Agent 实施区

父 issue [#28](https://github.com/guajun/anonymous-audio-tracks/issues/28)，总研究日志 [#27](https://github.com/guajun/anonymous-audio-tracks/issues/27)，路线见 [`docs/AGENTIC_RESEARCH.md`](../docs/AGENTIC_RESEARCH.md)。

实施顺序：`(#29, #30) → (#31, #32) → (#33, #34)`。已交付：[`toolbox/`](toolbox/)（#29 SAM 工具箱 pin/安装文档）、[`audio-probe/`](audio-probe/)（#30 原生/桥接验证与冻结接口）、[`workspace-template/`](workspace-template/)（#31 workspace 模板 + bootstrap/doctor/smoke + 文档）；其余目录随各 PR 更新。

目录归属：`toolbox/`（#29 工具箱 pin/集成报告）、`audio-probe/`（#30 原生/桥接验证）、`workspace-template/` 与 bootstrap（#31）、`schema/`（#32）、`e2e/`（#33）、`viewer/`（#34）。必要的测试和报告跟随各目录，避免并行重写根锁文件。

本地部署/音频/模型/完整日志置于被忽略的 `.local/` 或明确忽略的 outputs。API key 使用 Pi 现有认证，不提交凭据或改全局配置。研究 Agent 使用 Gemini 3.8 Flash；开发 workers 使用 MiMo-V2.6-Pro。任务 PR base 为 `agentic/minimal-agent`；未经主 Agent 当前 HEAD review 放行不得 merge。

最终网页与真实输出的验收说明由 #33/#34 提供，人类验收前不将显示效果标记已验收。
