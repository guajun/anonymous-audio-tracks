# Anonymous Audio Tracks

从音乐中跟踪匿名声音来源，将每个来源的活动轨迹映射为独立 RGB 灯带。

项目于 2026-09-29 立项。当前状态：数据渲染/中心活动标签、独立冻结 AuT 特征、E/P 输出头与最小训练、
滑窗轨迹关联与本地试听页都已实现并合并（M1/M2/M4 的最小闭环），并有一次真实冻结 AuT 短训练与
保留合成集评估、以及一对固定真实音乐片段的客观抽查证据；**尚未宣称 M3 研究成功、M5 完成，也没有 LED 谱面接入**。
本文中的窗口、步长与模型选择仍以实测文档为准，代表性结论与失败例见
[docs/reports/issue-11-integration.md](docs/reports/issue-11-integration.md) 与 [docs/TRAINING.md](docs/TRAINING.md)。

**目标设计（2026-10-04）**：以 [身份活动向量与局部拼接规范](docs/IDENTITY_ACTIVITY_DESIGN.md)
为准（[issue #48](https://github.com/guajun/anonymous-audio-tracks/issues/48)）。非零 V 的模长是连续活动强度，
方向 E 在局部共同事件证据下可匹配；严格零没有身份。核心监督是关联后连续包络 shape，整段一次 PIT。
main 当前 AuT/Query/E/P probability、静音原型记忆及历史结果是既有实现，尚未迁移到这一目标；
#46 的 Demucs 研究也不是 main 功能。下方快速开始运行的是现有 E/P 管线。

## 当前状态与快速开始

| 环节 | 现状 | 入口 / 证据 |
|---|---|---|
| 数据渲染与标签 | DawDreamer 合成语料 + 中心活动标签；Surge 仅探针未训练 | `scripts/render_sample.py`、`scripts/label_sample.py`、[docs/RENDERING.md](docs/RENDERING.md)、[docs/ACTIVITY_LABELS.md](docs/ACTIVITY_LABELS.md) |
| 连续包络课程数据 | 单来源、双来源交错与局部事件共现；效果后分轨 RMS，独立于模型 | `scripts/prepare_curriculum.py`、[docs/SYNTHETIC_CURRICULUM.md](docs/SYNTHETIC_CURRICULUM.md) |
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

输入一首混合音乐，跟踪其中可辨认的声音层，为被选中输出的来源活动形成连续强度轨迹。
局部共同事件证据支持跨短静音拼接；超出证据范围先保留碎片，离线回环是后续计划。
最终一条轨迹驱动一条灯带，动态随真实音乐变化，不受游戏难度或固定物件密度约束。

不要求输出乐器名称、音高、歌词或传统乐谱，也不要求 E 全局音色不变或跨歌曲不变。
K 以 8 为经验起点且可配置，是局部输出容量，不是整曲声源总数上限；较弱活动或容量溢出可无非零输出。
现有数据/训练仍拒绝来源数 > K，显著性竞争和部分参考评估尚待实现。

应用输入是一首音乐内部的多轨混音，不将两首完整歌曲相叠作为应用场景或数据增强方式。

## 目标模型语义与当前协议

目标候选是身份活动向量 V：非零时 `A=||V||₂`、`E=V/A`，A 是连续声学强度而非存在概率；
严格零不定义 E，不需要持久静音 E 或 Q 标量。E 下边沿是身份活动进入零的表征事件，不是 ADSR R。
gate 到零是待验证工程候选，尚未选定门控或梯度方案。候选排名/槽位不是 track_id。

短静音 `g<W/2` 可给两端非零事件双向共同证据，但仍要求候选存在、足够事件内容及边缘/步长余量；
在线关联可跨中间零输出的有限局部范围，不能机械限定连续两个中心，也不依赖长期陈旧原型。
首窗有效候选直接自对应 `C₀=I`，零首窗不建立身份种子。长缺口后续拟通过真实事件裁剪、
置于同窗重推理、验证并聚合碎片，再映射回原轴；不假定互不相交的旧 E 天然相似。

以下是 **main 当前 v0.1.0 E/P 协议**，并非以上 V 目标的已实现接口：

对中心时刻 t，截取长度 W 的音频窗口：

```text
f(audio[t-W/2 : t+W/2]) -> E[K,128], P[K]
```

- E[i]：第 i 个候选来源的单位归一化身份向量。
- P[i]：该来源在窗口中心时刻 t 的活动概率，不表示“整个窗口内曾出现”。
- 默认 K=8；这是容量设计，不是所有歌曲的声源数量假设。
- 沿整首歌按步长 H 移动窗口，得到逐中心时刻的预测，再做跨窗口身份关联，形成轨迹。
- 槽位编号不等于轨迹 ID。当前候选与保留的轨迹原型一对一匹配；短暂静音保留身份记忆，低活动概率的向量不更新轨迹原型。
- 第一版离线运行，允许窗口使用未来上下文；不声称实时零延迟。
- 原型计划 W=2 秒、H=20 ms；比较不同上下文长度和步长后再确定。首尾明确补零并保留有效区域掩码。

高频滑窗只是密集取样，不保证有效时间分辨率。独立窗口编码与整段编码后切片也不天然等价，缓存优化须核对窗口注意力与边界行为。

### 来源与活动的定义

训练来源首先定义为一个可独立控制的声音层：同一合成器的和弦音归为同一来源；kick/snare/hat 可分别构成来源。合成器实例 ID 是生成监督，不能假设人耳或模型能区分音色和演奏完全相同的两个实例。不可辨认的重复层应合并或在评估中标记歧义。

当前中心活动标签从实际渲染分轨能量经阈值、时间窗和滞回生成 0/1 probability；
目标参考则使用实际效果后进入混音的连续线性 RMS，保留真实静音、原时间轴和统一强度尺度，
不逐窗峰值归一化。控制事件另存；ADSR 不是严格声学真值，MIDI note-on 不必等于可听起音，
note-off 不必等于尾音结束。现有二值声学代理不等于完美的人类可听性真值，单位迁移须显式版本化。

## 当前实现与后续研究

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

### 当前骨干：Qwen3-Omni AuT

main 已实现独立冻结 Qwen3-Omni 音频编码器，不加载完整语言模型和语音生成部分；权重筛选、
预处理、时间网格与实测见 [AUT_PROBE](docs/AUT_PROBE.md)。它是现有 E/P 基线，目标设计的骨干仍需实验比较。

既有实验 A：冻结 AuT，训练多来源输出头，直接输出 E/P，历史结果见 [TRAINING](docs/TRAINING.md)。

旧 E/P 基线的候选实验 B：加入高时间分辨率的局部 Mel/CNN 特征，与 A 使用相同划分比较；尚未实现。

新模型不自动保证细粒度音乐跟踪更好。若冻结特征不足，再比较中间层、局部解冻或 LoRA。Qwen3-ASR 仅作为备选；ASR 的词时间戳不是匿名声音活动标签。后续骨干必须有可用权重、特征接口和可验证时间映射，不能仅凭音频问答能力入选。

### 现有输出头与目标训练的差异

现有 `SourceQueryHead` 使用 K 个可学习查询，输出单位 E 和 sigmoid P；结构已实现，见
[MODEL_HEAD](docs/MODEL_HEAD.md)。无 Query 是研究候选，不会自动给通道跨窗身份。

现有 E/P 损失包含（保留用于复现，不作为新目标必要约束）：

1. 来源与预测的一对一排列无关匹配，以及匹配后的中心活动损失。
2. 同源跨窗口/跨音高/不同伴奏背景的对比损失，异源作负样本。
3. 跨窗口一致性与空槽约束，避免身份切换、重复占槽和全零预测。

目标训练使用非零 V → E 的可微局部关联 → 同权重搬运 A → 整段一次 PIT → 连续包络 shape。
同形状允许 E 接近，分岔后由 shape 反传关联错误；不以无条件音色排斥替代。
cosine 是打分，soft/hard 是匹配权重的使用，TimeCycle 是可独立开关的辅助 loss；
逐行匹配可能多对一，hard argmax 配对选择不能承担 shape→E 梯度。完整约束见规范 §6。
先分轨监督，再独立比较混音训练；课程从简单不重叠到交错重复、overlap 和更真实音乐。

初始以冻结骨干和小头控制显存；若显存不足，可预提取特征或使用获准的远端 GPU。实际峰值、批大小与吞吐在 M2 测量，不预先承诺全量微调能在本地完成。

### 轨迹与灯光

现有输出包含原始中心时间、track_id、活动概率、关联置信度及实验来源，使用余弦门控一对一匹配与
原型记忆；“出生/终止/静音记忆”是其工程状态。目标中空匹配仅表示无输出对应，不能断言真实静音、
死亡或 E 下边沿；进入输出集合不等于新来源出生。关系、缺测与连续强度需显式版本迁移。

当前 viewer 显示概率曲线和阈值区间；目标连续强度显示与 LED 接入仍待实现。
活动上升沿可触发 attack；音量持续存在时不保证能区分每次连打，因此重复起音检测作为后续可选模块，
不混淆“活动”和“新起音”。LED decay/release 等策略属于适配层，不作为 E 下边沿或音频来源身份标签。

## 历史里程碑与目标验收

下表保留立项的 E/P 里程碑与验收口径；新设计实现须按
[规范 §10](docs/IDENTITY_ACTIVITY_DESIGN.md#10-后续实现验收与本次交付边界)另行验收。
本次仅对齐文档，未新增模型训练、容量处理、回环或研究成功声明。

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

现有概率活动评估使用逐来源 precision/recall/F1、全曲一致匹配与 ID switch、边界误差。
目标需另报告连续 shape、原时间轴 raw/搬运 A、漏/多报、来源数、PIT 并列、空匹配/熵、
shape→E 梯度和成本。不能逐帧重新匹配掩盖换轨；真实片段无完整真值时只记抽查，不宣称总体准确率。

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
