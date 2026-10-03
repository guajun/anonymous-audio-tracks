# Anonymous Audio Tracks

从音乐中跟踪匿名声音来源，将每个来源的活动轨迹映射为独立 RGB 灯带。

项目于 2026-09-29 立项。当前状态：数据渲染/中心活动标签、独立冻结 AuT 特征、E/P 输出头与最小训练、
滑窗轨迹关联与本地试听页都已实现并合并（M1/M2/M4 的最小闭环），并有一次真实冻结 AuT 短训练与
保留合成集评估、以及一对固定真实音乐片段的客观抽查证据；**尚未宣称 M3 研究成功、M5 完成，也没有 LED 谱面接入**。
本文中的窗口、步长与模型选择仍以实测文档为准，代表性结论与失败例见
[docs/reports/issue-11-integration.md](docs/reports/issue-11-integration.md) 与 [docs/TRAINING.md](docs/TRAINING.md)。

Issue #46 的冻结 Demucs、无 Query 向量头与连续包络课程实验在独立分支推进：
[设计与损失组合](docs/ISSUE_46_DESIGN.md)、[运行入口](docs/ENVELOPE_EXPERIMENT.md)、
[C0 实测与失败分析](docs/reports/issue-46-c0.md)、
[C1-local 实际窗口共现与对照](docs/reports/issue-46-local-context.md)。已实现连续包络及可微软关联研究链，
当前先纠正旧 C1 的跨上下文任务错配；正确关联与重叠课程仍待验证。
新关联训练、双音色接线诊断与 TensorBoard 入口见 [ENVELOPE_ASSOCIATION.md](docs/ENVELOPE_ASSOCIATION.md)。

## 当前状态与快速开始

| 环节 | 现状 | 入口 / 证据 |
|---|---|---|
| 数据渲染与标签 | DawDreamer 合成语料 + 中心活动标签；Surge 仅探针未训练 | `scripts/render_sample.py`、`scripts/label_sample.py`、[docs/RENDERING.md](docs/RENDERING.md)、[docs/ACTIVITY_LABELS.md](docs/ACTIVITY_LABELS.md) |
| 数据集索引 | 分组切分、泄漏安全、多窗口采样 | `scripts/build_dataset_index.py`、[docs/DATASET.md](docs/DATASET.md) |
| 冻结 AuT | 独立 Qwen3-Omni AuT 编码器，固定 revision 与 block-diagonal 注意力 | `scripts/probe_aut.py`、[docs/AUT_PROBE.md](docs/AUT_PROBE.md) |
| E/P 头与训练 | K 查询输出头、排列无关损失、checkpoint/恢复、fake 与 real 两条链 | `scripts/train.py`、[docs/TRAINING.md](docs/TRAINING.md) |
| 轨迹与评估 | 一对一关联、静音记忆、整曲全局映射指标与基线 | [docs/TRACKING.md](docs/TRACKING.md) |
| 端到端与试听 | 参数化 demo 入口 + 本地静态 viewer | `scripts/demo_pipeline.py`、[docs/REPRODUCE.md](docs/REPRODUCE.md)、[docs/VIEWER.md](docs/VIEWER.md) |

```sh
uv sync --locked --extra ml            # 训练/推理（CPU torch；GPU 另装同版本 CUDA 轮子）
uv sync --locked --extra render        # DawDreamer 渲染

# 真实渲染语料（可写目录）
uv run --no-sync python scripts/train.py prepare-data \
    --plan configs/train/dataset_local.toml --out runs/train-data --index-out runs/train-index/index.json

# CPU fake 冒烟（工程链路，非模型结果）
uv run --no-sync python scripts/demo_pipeline.py smoke --out runs/demo/smoke --steps 3

# 真实冻结 AuT 短训练与保留集评估（权重/输出都在忽略目录）
uv run --no-sync python scripts/train.py train --config configs/train/aut_short.toml \
    --model-dir /path/to/encoder-checkpoint --device cuda:0
uv run --no-sync python scripts/demo_pipeline.py synthetic \
    --checkpoint runs/train/aut-short/checkpoint.pt --index runs/train-index/index.json \
    --data-root runs/train-data --split test --model-dir /path/to/encoder-checkpoint --out runs/demo/aut-test

# 本地试听页（不上传音频；只监听 127.0.0.1）
uv run python -m http.server 8123 --bind 127.0.0.1 --directory viewer
```

完整复现步骤、运行预算与隐私边界见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

## 总目标

输入一首混合音乐，识别其中可辨认的声音层，跨音高、响度、重叠和短暂休止保持来源身份，为每个来源输出连续活动轨迹。最终一条轨迹驱动一条灯带，动态随真实音乐变化，不受游戏难度或固定物件密度约束。

不要求输出乐器名称、音高、歌词或传统乐谱。身份主要在同一首歌内部定义，不要求建立跨歌曲的全局乐器分类。音源数量以 8 为经验起点，槽位数 K 可配置；不要求每个槽位都被使用。

应用输入是一首音乐内部的多轨混音，不将两首完整歌曲相叠作为应用场景或数据增强方式。

## 核心模型协议

对中心时刻 t，截取长度 W 的音频窗口：

```text
f(audio[t-W/2 : t+W/2]) -> E[K,128], P[K]
```

- E[i]：第 i 个候选来源的单位归一化身份向量。
- P[i]：该来源在窗口中心时刻 t 的活动概率，不表示“整个窗口内曾出现”。
- 默认 K=8；这是容量设计，不是所有歌曲的声源数量假设。
- 沿整首歌按步长 H 移动窗口，得到逐中心时刻的预测，再做跨窗口身份关联，形成轨迹。
- 槽位编号不等于轨迹 ID。相邻窗口通过一对一匹配关联；短暂静音保留身份记忆，低活动概率的向量不更新轨迹原型。
- 第一版离线运行，允许窗口使用未来上下文；不声称实时零延迟。
- 原型计划 W=2 秒、H=20 ms；比较不同上下文长度和步长后再确定。首尾明确补零并保留有效区域掩码。

高频滑窗只是密集取样，不保证有效时间分辨率。独立窗口编码与整段编码后切片也不天然等价，缓存优化须核对窗口注意力与边界行为。

### 来源与活动的定义

训练来源首先定义为一个可独立控制的声音层：同一合成器的和弦音归为同一来源；kick/snare/hat 可分别构成来源。合成器实例 ID 是生成监督，不能假设人耳或模型能区分音色和演奏完全相同的两个实例。不可辨认的重复层应合并或在评估中标记歧义。

中心活动标签从实际渲染分轨的局部能量/包络派生，明确阈值、时间窗和滞回；控制事件另存。MIDI note-on 不必等于可听起音，note-off 不必等于尾音结束。合成活动标签是可复现的声学代理，不等于完美的人类可听性真值。

## 实现方案

### 数据：DawDreamer 可控渲染

使用 Python + DawDreamer 调度合成器、采样器与效果器。在一首音乐内部生成协调的节奏与旋律，再渲染各来源及混音。Suno 不作为主训练数据来源，机器拆分的 stems/MIDI 不作为精确真值。

每个样本保存：

```text
sample/
  mix.wav
  stems/<source_id>.wav
  controls.json       # MIDI、素材触发、参数自动化
  sources.json        # 来源 ID、渲染器/音色状态引用
  activity.npz        # 中心时间轴与逐来源活动标签
  manifest.json       # 随机种子、采样率、时长、版本、配置与内容摘要
```

优先验证 Surge XT 插件与 DawDreamer 的本机兼容性。保留干声、效果后分轨和混音的时间对应；记录插件延迟、渲染起点及尾音。先使用可加和的处理链，再加入压缩/侧链等效果；共享非线性母带处理后的混音不强制满足逐分轨相加一致性。

覆盖：音高与力度变化、短音与延音、快速重复、同时起音、强鼓遮盖旋律、来源停顿重现、滤波/失真/调制、段落密度变化。由简单的 2–4 来源逐步扩展至 K，音色多样性与合理音乐结构并重。避免固定“第 1 轨永远是鼓”等捷径。

按曲目、音色 preset、素材来源分组划分训练/验证/测试；不得把同曲相邻窗口随机分散到不同集合。另设未见合成器/效果配置测试，固定少量真实音乐片段作迁移评估。

### 骨干：Qwen3-Omni AuT

首选实验候选是 Qwen3-Omni 的音频编码器，独立提取特征，不加载完整语言模型和语音生成部分。官方报告约 650M 参数、16 kHz 输入、输出约 12.5 Hz（80 ms/步）。代码有独立 AudioEncoder；权重筛选、预处理和独立运行仍需实际验证。

实验 A：冻结 AuT，训练多来源输出头，直接输出 E/P。

实验 B：若活动边界不佳，为同一个输出头加入高时间分辨率的局部 Mel/CNN 特征。保持 E/P 协议不变，与 A 使用相同数据划分比较。

新模型不自动保证细粒度音乐跟踪更好。若冻结特征不足，再比较中间层、局部解冻或 LoRA。Qwen3-ASR 仅作为备选；ASR 的词时间戳不是匿名声音活动标签。后续骨干必须有可用权重、特征接口和可验证时间映射，不能仅凭音频问答能力入选。

### 多来源输出头与训练

使用 K 个可学习查询从音频时间特征中提取候选来源，输出单位向量和 sigmoid 活动概率。具体头结构在最小实验中确定。

训练损失包含：

1. 来源与预测的一对一排列无关匹配，以及匹配后的中心活动损失。
2. 同源跨窗口/跨音高/不同伴奏背景的对比损失，异源作负样本。
3. 跨窗口一致性与空槽约束，避免身份切换、重复占槽和全零预测。

仅凭一个中心时刻的多个 0/1 标签无法唯一匹配来源。训练批次应包含同曲多个时间点，利用一段活动序列与来源监督联合匹配；必要时以干净分轨表示作训练辅助。不能只做相邻重叠窗口相似度，否则模型可利用共有波形而非身份。

初始以冻结骨干和小头控制显存；若显存不足，可预提取特征或使用获准的远端 GPU。实际峰值、批大小与吞吐在 M2 测量，不预先承诺全量微调能在本地完成。

### 轨迹与灯光

输出包含原始中心时间、稳定 track_id、活动概率、关联置信度及实验来源。推理关联采用阈值门控的一对一匹配与轨迹记忆，支持出生、静音、重现和终止；K 是局部容量，整首歌累计轨迹数量不必等于 K。

先显示概率曲线和阈值活动区间，再映射 LED。活动上升沿可触发 attack；音量持续存在时不保证能区分每次连打，因此重复起音检测作为后续可选模块，不混淆“存在性”和“新起音”。LED decay/release 等渲染策略在适配层实现，不作为音频来源身份标签。

## 路线与验收

| 里程碑 | 交付 | 完成标准 |
|---|---|---|
| M0 立项 | README、配置、uv 工程、GitHub 仓库、计算资源说明 | 本地与远端仓库可核对，无数据或凭据入库 |
| M1 数据闭环 | 单首 2–4 来源渲染器、分轨、中心活动标签、试听 | 已知触发与分轨时间对齐；延迟/尾音有记录；同 seed 可追溯 |
| M2 骨干探针 | 独立 AuT 提取、时间映射、显存/吞吐记录 | 无需载入完整 Omni；保存特征形状；确认候选层可训练输出头 |
| M3 最小训练 | E/P 输出、排列匹配、跨窗身份损失 | 小数据可过拟合；未见曲目/音色测试优于常量和无身份基线 |
| M4 整歌轨迹 | 滑窗推理、轨迹关联、活动可视化 | 报告漏检/误检、来源数误差、身份切换与静音重现错误 |
| M5 复杂化与迁移 | 调制/效果/强重叠数据，固定真实音乐评估 | 在保留测试上比较改进，不用目标歌曲反复调参后当独立验证 |
| M6 LED 接入 | 轨迹 JSON、独立灯带、可选起音模块 | 可试听/定位/导入现有播放器，保留原时间轴与历史实验 |

执行顺序先 M1 与 M2 小规模探针，再决定是否扩大数据。首轮 30–60 分钟合成数据用于检验标签与管线；之后按学习曲线扩容，不把合成小时数当质量指标。

活动评估使用逐来源 precision/recall/F1；身份使用全曲一致匹配后的错误与 ID switch；边界报告时间误差。不能逐帧重新最优匹配来掩盖身份切换。真实片段没有完整真值时明确记为抽查，不宣称总体准确率。

## 工程与资源管理

GitHub 仓库：[guajun/anonymous-audio-tracks](https://github.com/guajun/anonymous-audio-tracks)（公开）。首轮开发由 [父 issue #1](https://github.com/guajun/anonymous-audio-tracks/issues/1) 管理；[开发任务图](docs/DEVELOPMENT.md) 说明并行范围与阻塞关系，[新主会话提示词](docs/PI_ORCHESTRATION_PROMPT.md) 规定本地 pi 的派发、等待、review 与合并流程。

Python 环境由 uv 管理，按用途分成 `render`（DawDreamer）与 `ml`（CPU torch、transformers、safetensors）两个 extra，
`uv.lock` 锁定版本；base 安装不拉取模型栈。每次运行都应保存配置快照、Git commit、依赖锁摘要、随机种子与数据划分摘要
（`scripts/demo_pipeline.py` 与训练 checkpoint 已自动记录）；GPU 运行需要同版本 CUDA torch 时在独立环境安装，不改锁文件。

```sh
uv sync --locked
uv sync --locked --extra ml
uv sync --locked --extra render
```

`configs/experiment.toml` 保存初始实验定义；`docs/OPERATIONS.md` 说明本地与共享服务器的边界。服务器连接信息放在忽略的 `.local/` 中，不进入远程仓库。

数据、模型权重、音频、缓存、运行输出和密钥均不提交 Git。每次运行应保存配置快照、Git commit、依赖锁摘要、随机种子与数据划分摘要；只提交脱敏的代码、文档与结果摘要。

## 参考资料

- [Qwen3-Omni 报告](https://arxiv.org/abs/2509.17765)：AuT 编码器与多模态训练。
- [Qwen3-Omni 仓库](https://github.com/QwenLM/Qwen3-Omni)：公开模型与使用方式。
- [DawDreamer](https://github.com/DBraun/DawDreamer)：Python 音频图、MIDI、插件渲染。
- [Surge XT](https://github.com/surge-synthesizer/surge)：合成器候选。
- [Wavesplit](https://arxiv.org/abs/2002.08933)：来源表示与聚类。
- [EEND-EDA](https://arxiv.org/abs/2106.10654)：匿名来源活动与可变来源数，语音领域参考。
- [PIT](https://arxiv.org/abs/1607.00325)：排列无关监督。
- [Slakh](https://www.slakh.com/)：合成分轨及对齐 MIDI 的数据路线。

以上工作支持研究方向，不构成对复杂电子音乐泛化效果的保证。
