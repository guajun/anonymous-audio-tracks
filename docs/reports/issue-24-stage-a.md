# issue #24 阶段 A：DETR 式音频来源集合预测的设计与诊断

本报告对应 [issue #24](https://github.com/guajun/anonymous-audio-tracks/issues/24) 的**阶段 A（设计与诊断）**。
它只做三件事：把现有代码与 DETR 逐模块对齐、把对象/监督/跟踪的语义写清楚、并用确定性最小反例测量当前
匹配歧义与监督覆盖。**阶段 B 的实验在本报告中只是待审阅提案**，没有启动任何训练、没有替换主模型、没有
实现拼接后处理，也不声称 DETR 或端点拼接有效。

- 基线 commit：`eb05eb93300960c7e879330920e53367233fe724`（分支 `codex/issue-24-stage-a`，PR base 为
  `codex/research-issue-24`）。
- 代码事实全部指向本仓库的文件/符号；文献事实全部给出一手链接，并在附录 A 标注核验日期。
- 诊断入口：`uv run --no-sync python scripts/diagnose_issue24.py report`；Phase B 清单校验：
  `uv run --no-sync python scripts/diagnose_issue24.py validate-plan`。

## 0. 证据分级与边界

本报告把每条结论标注为以下三类之一：

| 级别 | 含义 | 举例 |
|---|---|---|
| **文献** | 一手论文/官方实现的陈述，已核验 arXiv 页面或源码 | DETR 的匈牙利匹配代价与 `num_boxes` 归一化 |
| **代码** | 当前仓库具体文件/符号的实际行为，可复现 | `head_loss` 的 `composition_ids[g]` 实际是 `sample_id` |
| **假设/未验证** | 本项目尚未实验证实的推断或设计选择 | “query self-attention 会改善重复占槽” |

真实试听的边界：issue #24 正文记录的用户反馈（固定真实音乐听感以钢琴为主、第一秒有两下低沉鼓声、
预测存在性与听感无明显关联）是**负向人工抽查证据**。#11 文档中 `manual_inspection.md` 状态仍为
pending；本报告不把它当作逐帧/逐来源 GT，不用它计算准确率，只把它当作“需要复核中心窗口边界与
监督覆盖”的问题线索。

---

## 1. 原始 DETR → 当前 `SourceQueryHead` 的逐模块映射

### 1.1 DETR 的一手事实（论文 + 官方实现）

DETR（Carion et al., ECCV 2020, [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)）把检测视为
直接集合预测：固定数量的可学习 object queries、transformer encoder-decoder、集合级损失，通过二分图
匹配实现一对一分配，去掉 NMS 与 anchor。官方实现
[facebookresearch/detr](https://github.com/facebookresearch/detr)：

- `models/matcher.py::HungarianMatcher.forward`：分类代价用 `-softmax(pred_logits)[target]`，
  框代价用 L1 与 `-GIoU`，按图像用 `scipy.optimize.linear_sum_assignment` 精确求解；
- `models/detr.py::SetCriterion.loss_labels`：匹配后分类是带 no-object 类的交叉熵；
  `loss_boxes` 的 L1/GIoU 以 `num_boxes = sum(len(t["labels"]))`（batch 内目标总数，`clamp(min=1)`）归一化；
  `loss_cardinality` 只做日志诊断；每个 decoder 层有 auxiliary loss；
- decoder 每层 = query 间 self-attention + 对 encoder memory 的 cross-attention + FFN，逐层迭代更新 query；
  推理不做 NMS，由 no-object/分数阈值决定输出集合。

后续一手工作指出 DETR 匹配在早期训练不稳定、收敛慢：DAB-DETR 把 query 改写为动态 anchor box 并逐层
更新（[arXiv:2201.12329](https://arxiv.org/abs/2201.12329)）；DN-DETR 额外把带噪 GT 送入 decoder 做
去噪重建以降低二分匹配难度（[arXiv:2203.01305](https://arxiv.org/abs/2203.01305)）；Deformable DETR
用稀疏采样注意力改善收敛（[arXiv:2010.04159](https://arxiv.org/abs/2010.04159)）。这些是阶段 B decoder
消融的直接参照，不是阶段 A 已实现内容。

### 1.2 逐模块映射表

| 模块 | DETR（文献） | 当前实现（代码） | 状态 |
|---|---|---|---|
| 输入 | 图像 → CNN 特征图 + sine 位置编码 | 独立 2 s 窗口 → 冻结 AuT 帧特征 `[B,T,F]` + `frame_times`/`frame_valid`；训练配置固定 fp32 与 `block_diagonal`/`sdpa`（`aat.training.config`、`aat.training.encoding`） | 有，输入域不同 |
| 可学习 query | `num_queries=100` object queries | `SourceQueryHead.queries [K,d_model]`，K 默认 8 可配置（`aat.models.head`） | 有 |
| encoder-decoder | 全局 encoder + 多层 decoder（query self-attn + cross-attn + FFN，逐层迭代） | 只有两条**单层**分支 attention：`identity_attention`（键/值只用内容）与 `center_attention`（键含时间嵌入 + 每 query 时间偏置），各自 residual+FFN+LayerNorm | **缺失** query 间 self-attention、多层迭代、auxiliary per-layer loss |
| 位置信息 | 图像 token 位置编码 + query 动态 anchor | Fourier 时间编码 `τ(Δ_t)` 加在 center 键上；identity 分支刻意不含时间；`center_time_bias` 为每 query 时间偏置 | 有，偏置而非 anchor |
| 对象分类 | 类别 softmax（含 no-object） | 每槽二值活动 logit `z`，`P=sigmoid(z)`；没有类别词表 | 有，语义不同 |
| 位置回归 | 连续框 `[cx,cy,w,h]` | 无框；identity 是 128 维单位向量 `E`；活动是中心时刻概率 `P` | **缺失**几何位置，换成匿名身份 |
| 空对象 | 显式 no-object 类 + `empty_weight`；未匹配 slot 即背景 | `slot_valid` 掩码 + `empty_slots` BCE 把未占容量推向 `P=0`；被匹配的静音来源保留身份记忆，**不是** no-object | 语义不同，需明确区分 |
| 匹配 | 每张图用 Hungarian 求解 `[queries × targets]`，代价 = 类别 + L1 + GIoU | 每组（同曲多个中心窗口、来源列固定）用精确 bitmask DP 求解，代价只用活动序列 BCE；**所有**最优 assignment 枚举（上限 64），embedding 不参与代价 | 有，代价信号少一类 |
| matched loss | 分类 CE + L1 + GIoU，除以 `num_boxes`；aux losses | matched 活动 BCE + 空槽 BCE + 同源正对比 + 异源负 hinge；只统计有效证据项 | 有，归一化语义不同 |
| 推理后处理 | 分数阈值 + no-object，无 NMS | `slot_valid` + `P` 阈值；跨窗口身份由 `aat.tracking.associate_sequence` 维护；`slot != track_id` | 有，身份在 head 外 |
| 评估 | COCO AP | 整曲固定映射的逐来源 P/R/F1、ID switch、来源数误差、边界 MAE（`aat.evaluation.evaluate_trajectory`） | 有，不用 mAP |

### 1.3 音频任务不能照搬的部分（代码 + 假设）

1. **对象边界不是矩形**：DETR 的目标是空间框；音频来源在时间上任意重叠，且没有“频率框”。中心 `P` 是
   单点标签，不是“整个窗口曾活动”（`docs/MODEL_HEAD.md`、`docs/SCHEMAS.md`）。
2. **没有固定类别词表**：身份是曲内匿名来源，GT 由渲染 `source_id` 提供；模型不要求学习跨曲统一的
   乐器类别或 embedding 坐标（issue #24 定义决策；README“身份主要在同一首歌内部定义”）。
3. **来源数未知且可变**：DETR 的 `num_queries` 固定但 no-object 表达“无目标”；这里 K 是容量设计，空槽
   并不等于“静音来源”。静音来源必须保留身份记忆以便休止后接回。
4. **标签来源不同**：GT 来自实际渲染分轨的局部能量活动（`aat.labels.pipeline`），MIDI/控制事件另存；
   真实音乐没有 GT，听感不是精确标签。
5. **时间尺度不同**：DETR 框是连续坐标；这里 `P` 是 20 ms 网格上的中心概率，AuT 帧约 80 ms
   （`docs/AUT_PROBE.md`），存在网格—帧对齐问题。窗口首尾还有补零无效区。
6. **身份是工程关联问题**：整曲 track_id 由关联层维护，不在 decoder 内；ORB-SLAM 类比只表示“局部观测
   + 轨迹维护 + 重新关联”的分工，不意味着音频有几何验证，也不意味着拼接已实现。
7. **训练预算极小**：当前证据是 100 step 短运行（`docs/TRAINING.md` §5.3），而 DETR 需要长日程与辅助
   损失；任何“结构不行”的结论在阶段 A 都不成立。

### 1.4 三种候选表示的比较与阶段选择

| 维度 | 中心 E/P（现状） | 逐来源时间活动 mask | 事件区间（onset/offset） |
|---|---|---|---|
| 预测 shape | `E[N,K,128]`、`P[N,K]` | `M[N,K,T]`（或 `[K,T]`） | 变长集合 `(onset, offset, id)` |
| 监督信号 | 中心时刻活动 BCE | 帧级/段级 Dice/IoU + BCE | 集合匹配（区间 + 类别/身份） |
| 与公共协议 | 已冻结（`docs/SCHEMAS.md` v0.1.0） | 新输出，需协议讨论 | 新输出，需协议讨论 |
| 能否区分完全相同活动序列 | 不能（本报告 §3.4 实测） | 不能（mask 相同则匹配歧义依旧存在） | 不能（区间相同则歧义依旧） |
| 边界分辨率 | 受中心网格与窗口限制 | 受帧网格限制，可能更高 | 直接建模边界，依赖事件定义 |
| 标签成本 | 已有（中心活动 npz） | 需要帧级/段级导出与对齐 | 需要区间化规则（尾音、休止、颗粒度） |
| 现有代码支持 | 完整（#4/#7/#8） | 无 | 无 |
| 阶段 A 选择 | **保留为局部对象主接口** | 仅作为 Stage B 可选**辅助训练输出**（b6），且必须单独论证 | 不采用；作为长期研究候选 |

选择理由（证据限定）：阶段 A 测到的首要问题是**匹配歧义与身份监督覆盖**（§3.4、§4），而不是
“中心 P 表达力不足”；在表示未验证前引入 mask/区间会同时改变标签、匹配、公共接口三个变量，无法
归因。mask 只有在 b6 的受控辅助实验（以及窗口变化 b9 若获批）显示边界或监督增益、且单独协议论证
通过后才进入实现。

---

## 2. 对象语义、GT/预测形状与边界语义

### 2.1 三个层次及其职责边界（用户决策 + 代码）

| 层次 | 定义 | 谁产生 | 作用域 | 不承诺 |
|---|---|---|---|---|
| 局部声源对象 | 某曲内一个可独立控制的声音层在局部上下文中的存在与区分 | 渲染 GT `source_id` 提供监督；模型用 K 个槽位表示容量 | 单窗口/单训练组内 | 不要求槽位编号有跨窗意义 |
| 片段轨迹 | 一次分析范围内的稳定 `track_id` 及其 E/P 曲线 | `aat.tracking.associate_sequence` / `trajectory.json` | 一次滑窗推理（一首曲子/一段拼接音频） | 不代表跨长间隔必然同源 |
| 全曲 track_id | 整首歌内跨片段关联后的身份 | 工程关联层（当前 = 一次整曲滑窗关联；拼接复核是待实现候选） | 整曲 | 不要求模型 embedding 形成全局坐标 |

局部身份作用域：`(composition_id, source_id)` 的等价类只在**同一首歌内**有意义（`aat.losses.head_loss`
文档如此声明）；跨曲同名 `source_id` 是负例。README 的“不因长间隔断开就把同源前后片段当成强负例”
在现有代码中确实没有被违反——跨 step 的同源窗口既不进正例也不进负例（§4.1）。

### 2.2 GT 与预测 shape（代码）

训练 batch（`aat.training.batching.TrainingBatch`）：

| 名称 | shape | 语义 |
|---|---|---|
| `features` | `[G,N,T,F]` | 冻结骨干帧特征；F 由 encoder 决定（AuT=2048） |
| `frame_times` / `frame_valid` | `[G,N,T]` | 帧绝对秒；`False` 为补零帧，注意力屏蔽 |
| `center_times` / `center_valid` | `[G,N]` | 窗口中心；`center_valid` 来自 `activity.valid` |
| `activity` | `[G,N,S]` ∈ {0,1}（当前标签） | 中心时刻活动真值，来自实际渲染分轨能量标签 |
| `source_valid` | `[G,S]` | 真实来源列（静音来源也为 True），padding 列为 False |
| `composition_ids` / `source_ids` | `[G]` / `[G][S]` | 身份 key 的两部分；**实现里 composition 用 `sample_id`** |
| 预测 `E` / `logits` | `[G,N,K,128]` / `[G,N,K]` | 单位身份向量、活动 logit；`slot_valid` 行外全零 |

音级推理协议（`aat.contracts` v0.1.0）：`E[N,K,128]`、`P[N,K]`、`slot_valid[N,K]`、`center_valid[N]`；
整曲参考 `ActivityData.activity[C,S]`、`valid[C]`；轨迹 `Trajectory.tracks[].center_times/activity/
confidence/slot_indices`。`K` 可配置、embedding 维固定 128。

### 2.3 valid mask、空目标、静音与休止后重现

- **valid mask（代码）**：标签器的 `activity.valid` 定义为“模型中心窗口完整落在渲染音频内”
  （`aat.labels.pipeline`；与 `aat.windowing.extract_windows_at_times` 同一 half-up 取整语义）。
  它是可监督掩码，不是声学边界。训练还叠加 `frame_valid`（窗口内补零帧）与 `slot_valid`。
  推理时 `center_valid=False` 的窗口输出全零槽位，严禁把无效区当漏检或静音。
- **空目标（代码）**：整组全静音仍是有效监督样本；来源列照样参与匹配、`P` 被推向 0；`empty_slots`
  只压未占容量。**空槽 ≠ 静音来源**；静音来源的 E 仍合法（身份记忆），不更新原型。
- **静音与重现（代码实测，见 §7）**：`P=0` 但 E 仍可匹配的候选会刷新 `last_match_time`，因此稳定
  身份记忆会让同一 track 跨任意长静音保持（`retention_seconds` 不会因“没有活动”而计时）；
  只有“没有候选通过余弦门控”超过 `retention_seconds` 才终止并在重现时产生新 `track_id`。
  `docs/TRACKING.md` §2.1/§2.2 已描述这一行为，但配置表里“静音身份保留时长”的措辞易被读成
  “无活动即计时”；本报告以代码实测为准（见 §4.1 与 §7.3）。
- **训练目标内的重现**：同一训练组内的静音窗口不打断固定来源列；但跨 step 的同源窗口不构成正例，
  也不被强制拉近（issue 定义决策：长间隔跨窗泛化可测量但不是训练目标）。

### 2.4 真实试听反馈的证据地位

- 只作为**负向人工证据**：固定真实片段听感（钢琴为主、第一秒两下低沉鼓声、预测存在性与听感关联不明）
  提示需要复核“中心窗口有效区”和“短事件采样覆盖”；
- 不当作逐帧/逐来源 GT；不据此报总体准确率；不在训练中把听感写进标签；
- #11 文档的人工检查状态是 pending，本报告不改变该状态。

---

## 3. Matching cost 与 matched training loss

所有公式对应 `docs/MODEL_HEAD.md` 与 `src/aat/losses/{matching,head_loss}.py`。约定：组 `g`、中心窗口
`n ∈ N_g`、来源 `s ∈ S_g`、槽位 `k`；`z_{g,n,k}` 为活动 logit，`a_{g,n,s}` 为活动真值，
`E_{g,n,k}` 为单位 embedding。

### 3.1 Matching cost（先算，`detach`，不反传）

当前实现（代码）：

```text
C_g[s, k] = (1 / |N_g|) · Σ_{n ∈ N_g} BCE( z_{g,n,k} ; a_{g,n,s} )     # detach
```

- 只在 `center_valid ∧ slot_valid ∧ source_valid` 的项上求平均（`cost_num / cost_den`）；
- `Z_g = argmin_{单射 z: S_g → K_g} Σ_s C_g[s, z(s)]`，用精确 bitmask DP；所有最优 assignment 全枚举
  （`max_optimal_assignments=64`），`num_optimal` 饱和计数；截断时该组全部监督被跳过；
- **embedding 不参与匹配**：避免“用当前身份表示自证身份”。

与 DETR（文献）的差异：

```text
C_DETR = λ_cls · ( -softmax(pred_logits)[target_class] )
       + λ_L1  · || box_pred - box_gt ||_1
       + λ_giou· ( -GIoU(box_pred, box_gt) )
```

DETR 的匹配代价使用分类概率 + 框几何；本任务唯一的逐来源目标是活动序列，没有类别/框可用。因此
相同活动序列的来源在匹配层面不可区分（§3.4）。

### 3.2 Matched training loss（匹配后，反传）

```text
L_act   = (1/|T_act|)   Σ_{(g,n,s)} BCE( z_{g,n,z(s)} ; a_{g,n,s} )          # 含 a=0 的匹配静音来源
L_empty = (1/|T_empty|) Σ_{(g,n,k): k 空槽} BCE( z_{g,n,k} ; 0 )             # 未占容量
L_pos   = (1/|T_pos|)   Σ_j ( 1 - cos( E_j ; p_{e}^{(-j)} ) )               # 同 key 留一原型
L_neg   = (1/|T_neg|)   Σ_{(j,e')} max( 0, cos( E_j ; p_{e'} ) - m )       # m=0.25
L       = w_act·L_act + w_empty·L_empty + w_pos·L_pos + w_neg·L_neg
```

其中 `p` 都由 `stop_gradient` 的均值 + normalize 得到；key `e = (composition_id, source_id)`；
anchor 要求匹配窗口 `center_valid ∧ slot_valid ∧ a ≥ identity_activity_threshold(=0.5)`。

#### 3.2.1 normalization / reduction（代码）

| 项 | reduction | 分母 | 空集行为 |
|---|---|---|---|
| `L_act` | 所有有效 `(g,n,s)` 项的 mean | `activity_denominator`（有效中心 × 匹配来源计数） | 0，保留零梯度路径 |
| `L_empty` | 所有有效空槽项的 mean | `empty_denominator` | 0，保留零梯度路径 |
| `L_pos` | 所有有效 `(anchor, leave-one-out)` 对的 mean | `positive_denominator`；count<2 的 identity 不产生项 | 0，保留零梯度路径 |
| `L_neg` | 所有有效 `(anchor, other-identity prototype)` 对的 mean | `negative_denominator` | 0，保留零梯度路径 |

DETR 的 `num_boxes` 归一化与这里的分项 mean 语义不同：这里活动/空槽是“有效监督项平均”，正负例是
“有效对平均”；没有对每张图/每组做固定除数。这个选择让小组与长组贡献不同权重，阶段 B 若要比较需要
固定数据与 group 结构（已写入 `configs/research/issue24_phase_b_plan.toml`）。

### 3.3 活动 BCE / mask Dice/IoU 的适用前提

- **活动 BCE**：适用于当前单点 `(n, s)` 二值/软标签；匹配代价与 matched loss 都用
  `binary_cross_entropy_with_logits`（数值稳定、可求导）。
- **mask Dice/IoU**：**只有先定义逐来源时间 mask 或区间才有意义**。
  `Dice(X,Y) = 2|X∩Y| / (|X|+|Y|)`、`IoU = |X∩Y| / |X∪Y|`；需要帧级目标、与 `frame_valid` 对齐、
  处理空 mask（两者皆空时 Dice/IoU 无定义，不能悄悄记 1）。它**不能**区分完全相同的 mask；对当前
  单个中心 `P` 不能当框使用，也不能因为“Dice 是集合损失”就自动获得来源区分能力。
  Phase A 不引入 mask 损失；若 b6 获批，它只作为**训练辅助**，公共推理仍为 `E/P`。
- **空槽**：只惩罚未使用容量，避免模型用“全槽激活”骗 recall（#11 已观测到 FP 主导与预测来源数虚高）。
- **分轨辅助**：可以给每个来源提供训练时局部表征目标，但必须（a）只在本窗口/组内定义，
  不强迫全局 embedding 聚类；（b）不在推理时输入 stem；（c）不绕过匹配直接用 GT 配对制造监督。
- **局部身份约束**：只在同一训练组内跨越窗口，不跨 step 持久化；不得强制长间隔同源聚成同一类。

### 3.4 相同活动序列的歧义与不可辨认来源策略

设两来源 `s1,s2` 在所有 `n ∈ N_g` 上 `a_{·,s1} ≡ a_{·,s2}`，则 `C_g[s1, ·] ≡ C_g[s2, ·]`（代价行
完全相同），交换两者占用的槽位不改变总代价 → `num_optimal > 1`。这是**标签结构**导致的歧义，与
音色是否可辨认无关（本报告 §7.2 用可辨认但同起停的 fixture 实测：`same-on-off-two-timbres`
`num_optimal=2`）。策略（代码现状 + 阶段 A 结论）：

1. **不伪造身份真值**：并列最优时对活动/空槽做对称平均，对 identity 项整体 mask
   （`GroupMatching.identity_masked`）；
2. **截断更保守**：最优 assignment 数超过枚举上限时不使用任何未认证 assignment，整组包括活动项
   都跳过（`silent-duplicates-truncated` 实测 `activity_terms=0`）；
3. **不可辨认重复层**：应合并或在评估中标记歧义（README）；`evaluate_trajectory` 在“多 GT 来源同时
   活动”时计 `ambiguous_owner_frames` 而不猜 owner；
4. **不能用 E 偷偷配对**：把 embedding 放进 matching 会让当前表示自证，且与“E 不属于 GT”矛盾；
   若确实要利用可辨识音色，应作为显式实验（例如 stem 辅助或内容匹配代价），并在报告里标注风险；
5. **不能把 loss 权重设零当方案**：把 `w_pos`/`w_neg` 长期归零只掩盖歧义，不产生“来源区分”能力。
   阶段 B 的候选是逐来源支持的身份项（b2）、负例范围（b3a）、负例权重控制消融（b3b）、采样距离
   （b4）；其中 b3b 只是检验负例项是否起作用的一次性对照，不是“关掉 loss 当解决”的推荐方案。

---

## 4. 现有监督与采样审计（按文件/符号）

### 4.1 事实清单

| 编号 | 事实 | 位置 |
|---|---|---|
| F1 | 身份 key 实际为 `(sample_id, source_id)`；`composition_ids.append(block.sample_id)` | `src/aat/training/batching.py::training_batch_from_blocks` |
| F2 | 一个 step = `groups_per_step` 首歌，每步**不重复选曲**（`sample_batch` 无放回） | `src/aat/data/batch.py::sample_batch`；`src/aat/training/trainer.py` |
| F3 | 正例只在同一 step 的同一首歌组内跨窗留一；跨 step 同源窗口既不进正例也不进负例 | `src/aat/losses/head_loss.py` anchors/prototypes |
| F4 | 负例是同一 step 内“其他身份”的 stop-grad 原型；包含同曲其他来源与同批其他曲来源 | `head_loss.py` negative 分支 |
| F5 | anchor 需 `center_valid ∧ slot_valid ∧ a ≥ 0.5`；count<2 的 identity 无正例 | `head_loss.py` |
| F6 | 匹配每组一次、跨该组 N 个窗口固定 assignment；无逐帧重匹配 | `src/aat/losses/matching.py::match_sources` |
| F7 | 歧义组 mask identity、对称平均 activity/empty；截断组全部跳过 | `head_loss.py`；`docs/MODEL_HEAD.md` §2/§3 |
| F8 | 训练采样默认 `centers_per_item=4`, `min_center_gap=5`；`min_center_gap` 是最小间距而非固定间距（`_select_centers` 随机选择更远中心） | `configs/train/aut_short.toml`；`src/aat/data/batch.py::_select_centers` |
| F9 | 标签 hop=20 ms 时，`min_center_gap=5` 只限制最接近允许配对（0.1 s）；该配对的**最坏情况**是 2 s 窗口共享 1.9 s = **95%**。smoke 语料 10 step 实测相邻 gap 最小 0.1 s、中位 0.17–0.36 s（§7.2），实际通常更远 | 本报告 `context_overlap_report` / `sampling_coverage_report`（§7） |
| F10 | 解码器后处理按窗口一对一匹配 + 原型 EMA；低 P 匹配候选刷新存活时间但不更新原型 | `src/aat/tracking/tracker.py` |
| F11 | 评估用整曲固定映射，不逐帧重配；歧义帧不猜 owner | `src/aat/evaluation/metrics.py` |
| F12 | #8 真实 100 step：`activity_terms=2400`、`empty_terms=4000`、`positive_terms=149`、`negative_terms=572`、`ambiguous_groups=126/200`、`truncated=0` | `docs/TRAINING.md` §5.3.1 |

### 4.2 与“强上下文局部身份”目标的符合/超出判定

| 机制 | 目标 | 判定 | 依据 |
|---|---|---|---|
| 组内跨窗同源正例 | 局部上下文内区分来源 | **符合**，但强度可疑 | F9：最接近允许配对的最坏情况重叠上界 95%，实际抽样中位 gap 0.17–0.36 s；正例仍可能奖励共有上下文稳定性 |
| 批内其他身份负例 | 局部异源分离 | **部分超出**：同批其他曲的负样本引入瞬时跨曲分离压力（每一步 batch 随机，坐标不锚定全局） | F4；需 b3a/b3b 消融判断是否有害 |
| 各组独立固定 assignment | 防止逐帧重匹配掩盖换轨 | **符合** | F6；训练没有逐帧重配，评估也不重配 |
| 歧义组 mask identity | 不伪造身份标签 | **符合**保守原则 | F7；但牺牲大量 identity 监督（#8 63% 组被 mask） |
| 跨 step 不聚合身份 | 不强制长间隔同源聚类 | **符合** | F3；跨窗泛化可测量但未强制 |
| `sample_id` 充当 composition key | 同曲同源才正例 | **符合**；名称与实际分组字段不同，仅文档风险 | F1 |
| P=0 候选刷新 retention | 静音保留身份 | **符合代码文档**（`TRACKING.md` §2.1），但配置表措辞易误读为“无活动即倒计时” | F10；§7.3 实测 |

### 4.3 identity loss 实际有效梯度比例（现有证据 + 本阶段补测）

- **真实证据（#8，100 step / 200 组）**：`positive_terms/activity_terms = 149/2400 ≈ 6.2%`；
  `negative_terms/activity_terms = 572/2400 ≈ 23.8%`；`126/200 = 63%` 的组因并列最优被 mask 身份项。
  这不必然归因于 decoder 结构；至少一半是标签/匹配歧义与 anchor 覆盖问题。
- **合成补测（本报告 §7.2，只是结构计数）**：可辨认同起停组 `same-on-off-two-timbres` 的
  `positive_terms=0`；而活动序列不同的组（如 `piano-two-short-drums`）`positive_terms=7`、
  `negative_terms=7`。即：身份监督是否执行，取决于活动序列可区分性与 anchor 数量，而不是 decoder。
- **采样覆盖（本报告 §7.2）**：在合成 smoke 语料上跑 10 个模拟 step（每步 2 曲 × 4 窗、gap=5），
  train 两首歌的 valid 行覆盖率约 17–22%，每个来源都能获得若干“≥2 anchors”的组；但这只是合成语料，
  不能外推到真实渲染语料。要回答“真实运行的身份监督比例”，需要 Phase B 在真实索引上重复该诊断
  （已作为 b1/b2 的记录字段）。

### 4.4 结论与阶段 A 的“不做”

- 当前代码的匹配歧义不是 decoder 缺陷，也不是“打开/关闭某个权重”能解决的；把 `w_pos=0` 只会更糟
  （身份项彻底消失）。
- 阶段 A 不修改 `head_loss`、matching、tracker、协议；只把可测事实固定下来，供 Phase B 单变量消融。
- `docs/TRACKING.md` 的 retention 措辞建议后续单独修文档（本 PR 未改，避免扩大范围；报告 §2.3/§7.3
  已给出准确语义）。

---

## 5. 端点拼接复核流程（设计，未实现）

目标：为“长间隔候选关联”提供比直接比较原始 E 更强的**局部共同上下文证据**。一轮 = 沿拼接时间轴按采样
步长逐中心时刻滑窗推理 + tracker，不是单次前向，也不要求整段音频塞进一个窗口（issue #24 定义决策）。
以下全部用**原混音**，不假设推理时有 stem。

### 5.1 输入、输出与层次

- 输入：两条既有片段轨迹 `T_left`（结束端）与 `T_right`（起始端）及其原曲时间范围、对应的原始混音区间；
- 拼接：取 `T_left` 结束端音频 `A_end`（长度 `L_A ≥ W`）与 `T_right` 起始端音频 `B_start`（`L_B ≥ W`），
  按同一采样率、同一单声道/多声道约定首尾相接，得到拼接音频 `S = concat(A_end, B_start)`；
- 时间映射：`S` 的每个样本映射回 `(原曲, 原曲绝对秒)`；映射是分段线性的，接缝处有明确跳变
  `t_seam = len(A_end)/rate`；不能用拼接时间当作原曲时间，也不能把拼接轨迹回写进原曲轨迹而不做映射；
- 输出：候选对的复核结论（确认同源 / 拒绝 / 不确定）+ 证据；不修改 `T_left`/`T_right` 的原始轨迹，
  不改原 `slot`。

### 5.2 一轮的完整步骤

1. **候选检索（宽松，可与确认分离）**：对每个轨迹结束/起始端，用原始 E 端点相似度、时间间隔、
   活动置信度与容量约束产生候选列表（top-N，允许低阈值召回）；记录检索分数与排名。检索不依赖
   未经验证的拼接；目标是候选召回，不在这里做最终同源判断。
2. **拼接构造**：对每个候选对构造 `S`（长度、接缝类型、是否交叉淡化、增益是否对齐都写入 manifest）。
   不混两首完整歌曲；只做同曲内端点拼接。
3. **沿拼接时间轴滑窗**：按步长 `H` 从最左有效中心滑到最右（中心需让窗口完整落在 `S` 内；首尾
   `W/2` 仍无有效中心，与协议一致）。每个中心独立编码 2 s 窗口，得到 `E[K,128]/P[K]`；窗口跨越
   接缝时同时看到左右两端，这是复核证据的来源。
4. **tracker 关联一轮**：用 `associate_sequence` 得到拼接轴的 `trajectory_splice`。轨道是新 `trk-*`，
   **不得**沿用原 slot 编号或假设重跑前后 embedding 坐标不变。
5. **参考区保存（逐样本同输入锚点）**：在接缝两侧各保留一段远离接缝的参考中心（例如左端 clip 的
   最后 `R` 秒、右端 clip 的最前 `R` 秒，`R ≥ 1.5 s`，且窗口不跨缝）。这些中心在原推理与拼接重跑中
   读取**完全相同的原曲采样**（同一绝对时间、同一采样率、参考区不做增益/重采样变换），因此窗口输入
   逐样本相同。保存原推理的逐中心 E/P/track 证据（原曲绝对时间 + track_id + 原始 slot），作为重跑
   对齐的锚点。
6. **逐侧重跑轨迹对齐（一对一、可验证、可拒绝）**：在左右参考区分别把 `trajectory_splice` 的轨迹与
   原始 `T_left` / `T_right`（以及同侧其他原轨迹）做**轨迹级一对一**对齐，证据包括：
   - 时间共现：同一参考中心上原轨迹与重跑轨迹同时活动；
   - P 序列一致性：共现中心的 P 差值在容差内；
   - E 可复核性：在样本完全相同的参考中心上，原 E 与重跑 E 的余弦/距离必须落在 fp32 确定性复现
     容差内——这是**测量**同输入下的可复现性，不是假设 embedding 坐标跨运行恒定；
   只有“唯一最佳、显著优于第二候选、且可复核误差在容差内”的对齐才标记为**可靠**；出现并列或
   超差时该侧记为不确定，不允许用最近邻近似硬配。
7. **确认规则（严格，允许拒绝）**：只有同时满足以下条件才确认同源：
   - 左右参考区各自得到**可靠**的一对一对齐，并且都锚定到**同一条**拼接轨迹 `X`（同一条 splice track
     在两侧分别对应候选 `T_left` 与 `T_right`；不是两条不同重跑轨迹各自占一边）；
   - `X` 在接缝两侧邻域都有足够锚点（例如接缝前后各 ≥ `N_anchor` 个 `P` 高于活动阈值的中心，且
     连续匹配不中断）；
   - 接缝邻域没有“旧轨迹终止 + 新轨迹出生”的证据；
   - 决策阈值比 tracker 的单次门控更严格：确认分数由参考区对齐可靠性 + 接缝邻域证据统计共同决定，
     不是“任何一对相似度 ≥ 门限就算”。
   任一条件不满足，或任一侧对齐不可靠/歧义/不可复核，输出拒绝或不确定；允许拒绝是系统能力，不是失败。
8. **证据留存**：保存拼接 manifest（输入哈希、采样率、区间、接缝类型、增益处理）、原曲/拼接时间映射、
   左右参考区的原推理与重跑逐中心 E/P/track、逐侧一对一 alignment 结果与可复核误差、接缝邻域逐中心
   E/P 摘要、决策分数与原因码。原始音频与数组只留在忽略目录。

### 5.3 原始 E 对照与消融

- **直接对照**：同一候选列表分别跑“原始 E 端点匹配”与“拼接重跑 tracker 复核”，比较候选级
  确认/拒绝结果；直接 E 相似度不得作为模型必须通过的指标（issue 定义决策）。
- **顺序对照**：`A_end→B_start` 与 `B_start→A_end` 都应给出兼容结论；顺序敏感性写入证据。
- **接缝/音量/伴奏对照**：硬切 vs 短交叉淡化；接缝处人工增益差；两端伴奏/效果不同；间隔长度不同。
  这些是**防伪影对照**，不是训练增强；若决策随这些控制显著变化，说明系统依赖拼接伪影，应停止。
- **复杂度**：候选数 `n_c` 下成本 ≈ `O(Σ_c (L_A+L_B)/H)` 次独立窗口编码；应设每轨候选上限与全局
  预算，并记录实际耗时。参考量级：#11 本地 CPU 上 8 s 音频、H=0.1 s（81 窗）约 29 s（含加载），
  拼接复核的额外成本必须按此显式报告。
- **失败方式**：接缝爆音/相位不连续被当成事件；音量跳变改变 E；两端共享同一伴奏导致伪相似；
  窗口首尾无效区没有证据；参考区太短或全静音导致 P/E 证据不足；同一端多条轨迹同时活动使一对一
  对齐不唯一；E 在样本相同的参考中心上重跑不可复现（批次/数值差异）导致无法复核；tracker 门控过松
  把异源连接；顺序依赖；重跑轨迹与端点轨迹对应错误。以上任一情况都必须落到“拒绝/不确定”，而不是
  强行合并。
- **对齐指标（Phase B b8）**：参考区一对一 alignment 的 accuracy/uniqueness、参考区同输入 E 的
  重跑复现误差、不可靠/歧义计数、确认/拒绝混淆。对齐不可靠时不允许把候选计入误合并分母之外。

### 5.4 与“首秒两下鼓”的关系

#11 的 8 s 真实片段在 W=2 s 下**裸裁剪**时只有 61/81 个有效中心，第一秒与最后一秒无法承载中心
标签（本报告 §7 实测）。正确处理不是把评价区间缩到 31–37 s，而是**从原曲读取目标区间前后各 W/2 的
上下文**：目标区间仍是原来那 8 s，原曲绝对时间轴不变，首秒两下鼓仍映射到 target-relative 0.1/0.6 s，
其绝对中心（例如 31.1/31.6 s）此时拥有完整窗口，可以监督与评价。端点拼接同理：为各自端点保留远离
接缝、窗口输入与原推理完全相同的参考区（§5.2 步骤 5–6），而不是排除接缝附近的中心。

---

## 6. Phase B 提案（**待主会话审批**，未执行）

结构化清单在 `configs/research/issue24_phase_b_plan.toml`，由
`scripts/diagnose_issue24.py validate-plan` 校验（只校验结构，不运行实验）。本报告不授权执行；
Phase B 只有在主会话逐条审阅监督/接口/实验方案后才可开始。

### 6.1 数据与隔离

- 合成渲染语料：`train-01..04`（训练）/ `val-01/02`（development validation）/ `test-01/02`
  （历史已观察）；按 composition/preset/sample_origin 连通分量隔离
  （`configs/train/dataset_local.toml`，`leak_free=true`）。
- **候选、阈值、停止与组合决策只使用 train + development validation**。`test-01/02` 在 #8/#11 已被
  观察，只能作为历史对照；它不冒充未调参的 virgin holdout，也不进入任何选择决策。
- **新增 frozen holdout check（b10，仅方案）**：配置冻结后、在生成任何 holdout 音频之前先登记；
  holdout 由与 train/val 不共享 composition/preset/sample_origin 的曲目构成；只对事先写定的 baseline
  与最终 selected candidate 各评估一次；Phase A 不生成、不查看、不运行它。
- 真实用户音乐**不进入训练**、不用于准确率；固定片段只能用于工程层后处理诊断（无 GT、明确标注）。
- 拼接对只在同一首合成曲内构造（同源长间隔 + 异源难负例），绝不放两首完整歌曲；端点保留样本级
  同输入参考区（§5.2）。

### 6.2 B0 过拟合门槛（先于一切比较）

在固定两首歌 × 每首 4 个非相邻窗口上过拟合当前 head/loss（300 step 上限）。通过条件：该固定 batch
活动 F1 ≥ 0.99，且至少一个 identity key 产生正例项。若连这都不过，说明监督/数据选择有问题，禁止
进入 B1–B9 比较（b10 是冻结后的最终检查，更不参与此前的选择）。#7 的 fake-feature 过拟合与 #8 的
真实 100 step 只是历史证据，不能替代本门槛。

### 6.3 实验表（与 TOML 一一对应）

| id | 单变量 | 基线 | 预算（上限） | 主要指标 | 停止条件要点 |
|---|---|---|---|---|---|
| b0-overfit-gate | 无（门槛） | zero-logit BCE / #7 overfit | 300 step, 15 min | activity F1、identity anchor 覆盖 | 不过则先修监督/数据 |
| b1-baseline-current | 无（重跑基线） | #8 aut_short | 100 step, 10 min, 2 seeds | F1、ID switch、边界、歧义/identity 覆盖、采样 gap 分布 | 基线不可复现则先诊断 |
| b2-supervision-ambiguity | identity mask 粒度 | b1 | 100 step, 10 min, 2 seeds | identity 覆盖、F1、ID switch | 只涨覆盖不涨指标→记录无效果，不叠加 |
| b3a-supervision-negative-scope | 负例范围（batch-global→同曲） | b1 | 100 step, 10 min, 2 seeds | negative/positive terms、F1、ID switch | 无效果→保留最简项 |
| b3b-supervision-negative-off | 负例权重（1.0→0，控制消融） | b1 | 100 step, 10 min, 2 seeds | positive terms、F1、ID switch | 无效果→记录负例项不可测 |
| b4-sampling-context | 最小间距 5→25 格（0.1→0.5 s，最坏重叠 95%→75%） | b1 | 100 step, 10 min, 2 seeds | anchor 覆盖、实测 gap、F1、边界 | 无效果→记录正例“上下文稳定性”解释 |
| b5-decoder-self-attention | head 结构（query self-attn + 迭代） | b1 | 100 step, 15 min, 2 seeds | F1、来源数误差、重复占槽、边界 | 两个 seed 无提升→停止 decoder 方向 |
| b6-time-mask-aux | 辅助时间 mask 头（公共接口不变） | b1–b5 最佳 | 100 step, 15 min, 2 seeds | 边界 MAE、F1、重复占槽 | 仅边界问题成立时启动；无改善→不引入协议 |
| b7-combination | 最佳监督 + 最佳 decoder | 两个父实验 | 100 step, 15 min, 2 seeds | 组合指标 | 不优于最佳父→报告冲突并停止组合 |
| b8-endpoint-splice-postproc | 直接 E vs 参考区对齐后的拼接复核（同一候选表） | 直接 E 匹配 | 60 对, 30 min | 候选召回、确认精度、误合并/漏关联、参考区对齐可靠/复现、耗时 | 不优于直接 E 或对齐不可靠/对伪影敏感→停止拼接方向 |
| b9-window-context | W 2.0→4.0 s（需协议论证） | b1 @W=2.0 | 100 step, 15 min, 2 seeds | 边界 MAE、F1、来源数 | 未批准不启动；无改善→保持 W=2.0 |
| b10-frozen-holdout-check | 无（最终一次性检查） | 事先写定的 b1 与 selected candidate | 每个配置评估一次, 15 min | 活动 P/R/F1、ID switch、来源数、identity 覆盖 | 冻结后登记、只看一次、不回馈调参 |

决策只用 train + development validation（`val-01/02`）；历史 `test-01/02` 不参与选择，frozen holdout
只在配置冻结后登记并一次性评估 baseline 与 selected candidate（§6.1）。

### 6.4 预算、seed、记录字段

- seed 固定 `20260929`；正值结论需在 `20260930` 复现同一方向，否则记为 unstable。
- 每次运行记录：代码 SHA + dirty、配置 SHA-256、`uv.lock` SHA-256、数据集索引 SHA-256 与内容摘要、
  `leak_free`、encoder revision/provenance、seed、设备、耗时、进程内 CUDA peak（CPU 记 null+原因）、
  逐 step loss 分项与统计（含 ambiguity/truncated/identity_masked/term 计数）。
- 单变量约束：每个候选预先固定一个水平，不能出现“or … chosen before the run”；验证器拒绝未决
  单变量；b7 只在父实验报告后启动；b6/b9 需单独协议论证。
- 数据约束：选择/停止/组合只用 train + development validation；历史 test 仅作历史对照；frozen holdout
  在配置冻结后登记并只评估事先写定的 baseline 与 selected candidate 一次。

### 6.5 指标与歧义统计

- 模型层（已有）：活动 micro/macro P/R/F1、ID switch、歧义归属帧、来源数绝对误差、onset/offset MAE。
- 监督层（已有，head_loss 统计）：ambiguous group 比例、identity_masked 比例、四类有效 term 计数、
  identity anchor 覆盖，以及实际采样相邻中心 gap 的 min/median/max。
- 新增（计划，未实现）：`duplicate_overlap_frames`（未映射轨迹与某映射来源活动重叠的帧数）、
  `candidate_recall_at_n`、`splice_confirmed_precision`、`splice_false_merge`、`splice_missed_link`、
  `splice_inference_seconds`，以及参考区对齐指标 `reference_zone_alignment_accuracy`、
  `reference_zone_alignment_uniqueness`、`reference_zone_e_reproducibility_error`、
  `splice_uncertain_count`。拼接指标只在合成长间隔对上定义；真实片段无 GT，不作为准确率。
- 歧义统计：除 group 级统计外，报告“同活动序列来源对数量”和“评估中的 ambiguous_owner_frames”。

### 6.6 停止/继续条件

- B0 不过 → 停止，修监督/数据，不得进入比较；
- 单变量在两个 seed 无方向性变化 → 记录 null result，不叠加到组合；
- B5/B6 在相同预算与 seed 下不优于 B1 → 停止 decoder/mask 方向，不靠加长训练“救”结论；
- B8 不优于直接 E、参考区对齐不可靠/不可复现、或对接缝/顺序/音量控制敏感 → 停止拼接方向；
- frozen holdout 只在冻结后登记并一次性评估事先写定的 baseline/selected candidate，不回馈调参；
- 任何实验无提升也如实交付；mock 只用于工程冒烟，不冒充真实结果。

---

## 7. 可复现诊断（阶段 A 交付）

### 7.1 运行方式

```sh
uv sync --locked --extra ml
uv run --no-sync python scripts/diagnose_issue24.py report --out runs/issue24-diagnostics/report.json
uv run --no-sync python scripts/diagnose_issue24.py validate-plan
uv run --no-sync pytest -q tests/diagnostics            # 基础 + ml 诊断测试
```

- 诊断不读取私人/外部/真实音频、权重、checkpoint 或训练模型；sampling section 只在调用方临时目录中
  生成并读取合成 smoke WAV 语料，进程结束后自动清理；其余 fixture 都在内存构造；
- 输出 JSON 的每个 section 带 `evidence_kind`，并汇总 `limitations`；
- `report --skip-torch/--skip-matching/--skip-corpus` 可在基础环境运行（缺 torch 时对应 section 记
  `unavailable`/`skipped`，不失败）；
- 代码位置：`src/aat/diagnostics/issue24.py`、`scripts/diagnose_issue24.py`、`tests/diagnostics/`。

### 7.2 覆盖场景与测得事实（合成 fixture，非模型证据）

| 场景 | 构造 | 测得（`matching_ambiguity` / `identity_supervision`） |
|---|---|---|
| 单来源 | 1 来源、4 窗、3 窗活动 | 唯一匹配；positive=3、negative=0 |
| 钢琴 + 两次短鼓 | 钢琴 5/6 窗 + 鼓两次孤立短击 | 唯一匹配；positive=7、negative=7（identity 监督确实执行） |
| 同开同停的两个可辨认音色 | 两列活动完全相同 | **歧义**：`num_optimal=2`、identity mask；positive=0 |
| 交替活动 | 两来源不重叠 | 唯一匹配；positive=4、negative=4 |
| 休止后重现 | 单来源 1,1,0,0,1,1 | 唯一匹配；positive=4；静音中心仍被监督 P=0 |
| 无声 | 两个有效静音来源 | 歧义（`num_optimal=12`）；activity 仍被监督；identity mask |
| 不可辨认重复层 | 3 列相同活动 1,0,1,1 | 歧义（`num_optimal=6`）；identity mask |
| 静音重复 + 枚举截断 | 3 列全零、K=8 | **truncated**（P(8,3)=336>64）；整组监督为 0 |
| 窗口边界（裸裁剪） | 8 s、W=2 s、H=0.1 s | 81 个中心中只有 61 个有效；首秒两下鼓只作为 context 可见，0 个有效中心可标注 |
| 窗口边界（补上下文） | 目标 8 s 不变，从原曲前后各读 W/2=1 s | 101 个中心中 81 个有效；首秒两事件仍在 target-relative 0.1/0.6 s（目标起点 30 s 时为绝对 31.1/31.6 s），各对应 1 个有效中心 |
| 最接近允许配对的重叠上界 | `min_center_gap=5` 格 × 20 ms | 最小允许间距 0.1 s；最坏重叠 1.9 s = **95%**（只是上界，不是每组实际间距） |
| 实际采样 gap/重叠 | smoke 语料、10 step × 2 曲 × 4 窗 | 相邻中心 gap：min 0.1 s、median 0.17–0.36 s、max 1.24–2.6 s；组内最坏重叠：min 0.81、median 0.89–0.95、max 0.95 |
| 采样覆盖 | smoke 语料、10 step × 2 曲 × 4 窗 | train 两首歌 valid 行覆盖约 17–22%；每来源都有 ≥2 anchor 的组（合成语料限定） |

### 7.3 原始模型输出 vs tracker/postprocess（分开报告）

`tracker_postprocess_report` 用显式 E/P fixture 把后处理影响单独列出：

1. **slot 置换**：两身份在窗口中交换 slot（窗口 6 起 `slot0↔slot1`），拼接轨迹仍是 2 条稳定 track，
   `single_stable_track_ids=true` → `slot` 不是身份，track_id 不跟随 slot。
2. **静音与 retention**：P=0 但 E 可匹配的候选会刷新 `last_match_time`，长间隔（2 s）仍保持同一 track；
   只有“间隙内没有任何候选通过门控”且超 `retention_seconds` 才产生新 track（实测
   `long_gap_without_candidate.track_count=2`）。
3. **门限敏感性**：两段余弦 0.68 的 E：默认门限 0.7 → 2 条 track；门限 0.6 → 1 条 track。说明宽松
   门限会强行连源，拼接复核确认必须用严格证据而不是门限放行。

这些数字来自合成 E/P；它们证明的是**后处理行为**，不是模型质量。#11 的真实轨迹（clip-A 8 条、clip-B
3 条 track，61/81 有效中心）只作为工程侧观察，不在本报告作为效果结论。

### 7.4 既有问题/最小修复

- 本阶段没有发现需要修改生产代码的 bug；`tests/diagnostics` 全部通过。
- 发现一处**文档措辞风险**（非代码 bug）：`docs/TRACKING.md` 配置表把 `retention_seconds` 描述为
  “静音身份保留时长；超时终止”，而代码是“低 P 候选可匹配并刷新存活时间”（`tracker.py` §2.1 与
  `TRACKING.md` §2.1 已写对）。本 PR 不改该文档以避免扩大范围，报告 §2.3/§7.3 给出准确语义，建议
  后续单独修文档。

---

## 8. 风险与未验证项

- 所有诊断为标签/匹配/后处理结构证据；没有真实模型输出参与结论。
- 合成 smoke 语料是 NumPy PCM，不是 DawDreamer 渲染；采样覆盖数字不能外推。
- #8 的 100 step / 4+2+2 首歌、#11 的 2 首合成评估与 2 段真实抽查都不足以判定架构优劣。
- mask/区间表示、query self-attention、拼接复核均为未实现假设；本报告不声称有效。
- 真实音乐没有 GT；用户负面试听只用于定位问题，不用于准确率。
- 阶段 B 预算、seed 与停止条件必须在主会话批准后才能执行。

---

## 附录 A：一手参考（核验日期 2026-09-29，均为 arXiv abs 页面或官方仓库）

| 工作 | 链接 | 与本报告的关系 |
|---|---|---|
| DETR | https://arxiv.org/abs/2005.12872 | 集合预测、二分匹配、encoder-decoder |
| DETR 官方实现 | https://github.com/facebookresearch/detr | `HungarianMatcher`/`SetCriterion`/decoder 结构 |
| Deformable DETR | https://arxiv.org/abs/2010.04159 | 收敛/稀疏注意力（阶段 B 参照） |
| DAB-DETR | https://arxiv.org/abs/2201.12329 | query 作为动态 anchor、逐层更新 |
| DN-DETR | https://arxiv.org/abs/2203.01305 | 匹配不稳定与去噪训练 |
| SED Transformer（DCASE 2022） | https://arxiv.org/abs/2210.09529 | 音频事件检测的事件级集合预测；边界仍配合帧级模型 |
| 官方实现 | https://github.com/965694547/Hybrid-system-of-frame-wise-model-and-SEDT | 事件级 SED 的匹配/推理结构 |
| SoundDet | https://arxiv.org/abs/2106.06969 | 把时空声音事件当作完整“sound-object”，含轨迹检测 |
| TadTR | https://arxiv.org/abs/2106.10271 | 时序动作检测的事件区间集合预测 |
| 官方实现 | https://github.com/xlliu7/TadTR | 区间 query/匹配实现参照 |
| EEND（self-attention） | https://arxiv.org/abs/1909.06247 | 说话人 diarization：匿名身份、重叠活动 |
| EEND 置换无关目标 | https://arxiv.org/abs/1909.05952 | 局部身份/排列无关监督的一手来源 |
| EEND-EDA | https://arxiv.org/abs/2106.10654 | 未知说话人数、attractor 局部身份 |
| EEND 官方实现 | https://github.com/hitachi-speech/EEND | diarization 的 local identity 实践 |
| Powerset diarization | https://arxiv.org/abs/2310.13025 | 重叠来源的可辨认性/歧义表示对照 |

引用边界：以上文献支持“局部身份 + 集合预测 + 后处理关联”的研究方向，不构成本项目复杂电子音乐效果
的任何保证；文献中的数据集、类别体系与评估协议均与匿名音乐来源跟踪不同。
