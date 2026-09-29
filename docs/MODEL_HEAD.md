# 多来源 E/P 输出头与排列无关训练损失（issue #7）

本文件记录 `src/aat/models/` 与 `src/aat/losses/` 的最小实现公式、假设、备选方案和未验证项。
本轮只使用 fake features 验证数学与工程行为，不加载真实 AuT、不写完整训练脚本、不做乐器分类或灯光渲染。

## 0. 协议边界（不可变）

- 输出固定为 `E[N, K, 128]` 与 `P[N, K]`；`D = 128`，只有 `K` 可配置（默认 8）。
- `P[n, k]` 是窗口中心时刻 `center_times[n]` 的活动概率，不是“窗口内曾活动”。
- 有效候选（`slot_valid = True`）的 `E` 必须是有限单位向量（容差 `1e-3`），`P ∈ [0, 1]`；即使 `P = 0`（短暂静音的 identity memory）也必须保留单位身份向量。
- 无效槽位 `E = 0`、`P = 0`。槽位编号不是身份；本实现先训练纯候选头，再由下游轨迹关联产生稳定 `track_id`。
- 产物必须能通过 `aat.contracts.PredictionData` 校验；协议本身不被本 issue 修改。

## 1. 输入：fake features 与训练组

头部按窗口读取特征帧：

```text
features:     [B, T, F]   每窗口 T 帧、F 维特征
feature_valid:[B, T]      True 为真实音频帧；False 为补零帧，必须屏蔽
frame_times:  [B, T]      权威帧时间（原曲绝对秒，可选）
```

训练组是**同一曲目的多个中心时刻窗口**（不是随机拼不同曲目）：

```text
E:              [N, K, 128]   头部输出（每个窗口同一组 K 个查询）
P_logits / P:   [N, K]
activity:       [N, S]        生成监督：该窗口中心时刻各来源活动概率（0/1 或软标签）
window_valid:   [N]           False 表示该窗口标签无效（如补零窗口），不参与损失
source_descriptors: [S, D_id] 可选辅助身份描述（如干净分轨摘要），用于匹配与对比
```

`S ≤ K`。若 `S > K`，损失直接报错（容量不足属于配置错误，不静默截断）。

## 2. 输出头（`src/aat/models/head.py`）

只有一套可学习参数，窗口之间共享，保证同一来源跨窗表示可比：

1. 固定正弦时间编码：`time_embed(t)[2i] = sin(2π t / T_i)`，`[2i+1] = cos(...)`，
   `T_i = shortest_period * 2^i`（默认 `shortest_period = 0.02 s`，`i = 0..d_model/2-1`）。
   无参数、对任意绝对秒数值稳定；默认最短周期对应原型 hop。
2. 特征投影：`x[n, t] = W_f features[n, t] + time_embed(frame_times[n, t])`。
3. K 个可学习查询 `q_k`，对帧做点积注意力，补零帧在 softmax 前置为有限大负数
   `-1e9` 并在 softmax 后乘 mask 归零（避免全 mask 行的 NaN）：

```text
a[n, k, t] = softmax_t( q_k · K(x[n, t]) / sqrt(d) ) · 1[valid[n, t]]
h[n, k]    = Σ_t a[n, k, t] · V(x[n, t])
```

4. 前馈：`g = FFN(h)`（两层 Linear + GELU）。
5. 身份分支：`E[n, k] = normalize(W_E g[n, k])`；若某窗口没有任何有效帧，
   用 `W_E q_k` 的单位化结果作为回退，保证有效槽位永远是单位向量。
6. 活动分支：`logit[n, k] = W_P g[n, k]`，`P = sigmoid(logit)`。
7. 本实现的 `slot_valid` 对所有窗口、所有槽位恒为 `True`（K 是容量，空槽由损失压到
   `P = 0`）。occupancy 头是备选方案（§6.3），本轮不实现。

`head_output_to_prediction_arrays()` 把 `HeadOutput` 转成 `PredictionData` 可接受的
numpy 字典：`E` 转为 float32，无效槽位清零，`P` 不变，`center_times` 用 float64。

## 3. 组内排列匹配（`src/aat/losses/matching.py`）

匹配只决定监督对应关系，不引入额外可学习参数。对每个槽位 `k` 与来源 `s` 计算代价：

```text
C_act[k, s] = BalancedBCE( logits[:, k], activity[:, s]; window_valid )
C_id[k, s]  = 1 - cos( normalize(mean_valid E[:, k]), normalize(descriptor[s]) )   # 有描述时
C[k, s]     = C_act[k, s] + w_desc · C_id[k, s]
```

`BalancedBCE` 对正负标签分别取均值再平均，避免“全零预测”在静音占多数时被当成成功：

```text
pos = Σ w_pos · BCE(logit, target) / Σ w_pos,   w_pos = target · valid
neg = Σ w_neg · BCE(logit, target) / Σ w_neg,   w_neg = (1 - target) · valid
BalancedBCE = 0.5 (pos + neg)    （某侧为空时退化为另一侧；双侧为空时为 0）
```

`S ≤ K` 时在所有**单射** `π: source → slot` 上求

```text
π* = argmin_π Σ_s C[π(s), s]
```

K 小（默认 8，最大 `8! = 40320` 个单射）直接按字典序枚举全排列，取第一个严格最小者，
因此算法确定性、无随机 tie-break；超过 `max_match_permutations` 时报错而不是近似。

## 4. 不可辨匹配的显式处理

仅凭“活动序列”可能无法区分来源：若两个来源在所有有效窗口上的标签列相同（且没有能区分的
描述子），则交换二者的任何 assignment 都具有相同匹配代价。实现上：

1. `find_ambiguous_classes()` 用并查集把“有效窗口上 activity 列相同（`atol` 内）且
   descriptor 相同”的来源合并成等价类。
2. `iter_ambiguous_assignments()` 生成等价类内部的**全部槽位置换**（笛卡尔积），
   基准 assignment 排在首位；总数 = ∏ |class|!，超过 `max_ambiguous_assignments`
   （默认 64）时确定性截断并置 `ambiguity_truncated = True`。
3. 损失对所有枚举 assignment 的每一项分别取平均，再按权重求和；不随机硬配。

同时满足简单不变式：等价类内的来源标签在整个组上完全相同，且当前所有损失项对类内槽位
置换对称（活动项按 `(slot, target)` 多集配对、同源项只依赖槽位、异源项按无序槽位对求和），
因此截断只影响极端 `8!` 规模的病态输入，不改变普通输入（如两个相同 0/1 序列）的结果。
测试会显式验证：GT 列交换、等价类槽位交换均不改变总损失。

## 5. 损失分项（`src/aat/losses/head_loss.py`）

对某个匹配 assignment `π`，记匹配到来源 `s` 的槽位为 `k_s`，其跨窗原型

```text
z_s[n]  = E[n, k_s]
μ_s     = normalize( Σ_{n: valid} z_s[n] / Σ_{n: valid} 1 )
```

- **活动 BCE**：`L_act = mean_s C_act[k_s, s]`，即匹配后的中心活动损失
  （`C_act` 与匹配代价完全一致，避免“匹配目标”和“监督目标”不一致）。
- **同源跨窗对比**：`L_same = mean_{n: valid, s} (1 - cos(z_s[n], μ_s))`。
  身份交换（同一槽位在部分窗口换成别来源的向量）会提高该项；`N < 2` 时自然为 0。
- **异源负例**：对有效窗口内不同来源的有序对用 hinge 惩罚原型碰撞
  （`m = negative_margin`，默认 0.25）：

```text
L_diff = Σ_{n,s≠s'} relu( cos(z_s[n], μ_{s'}) - m ) / (N_valid · S · (S-1))
```

- **空槽处理**：未被匹配的槽位（容量剩余）必须压低活动，不含任何 NaN：

```text
L_empty = Σ_{n: valid, k ∈ unmatched} BCE(logit[n, k], 0) / (N_valid · |unmatched|)
```

- **总损失**：

```text
L = w_act L_act + w_same L_same + w_diff L_diff + w_empty L_empty
```

默认权重均为 1。全部分项在 `LossOutput` 中暴露（`total / activity / same_source /
diff_source / empty_slots` 与 `matching`、`ambiguity_truncated`），便于 ablation。

边界行为：`S = 0`、全静音、`N = 0`、`N = 1`、所有窗口无效、无有效帧时各项都必须有限，
且返回值为连接到计算图的零（而不是常量），保证反向传播可用。

### 不接受的行为

- 全零预测不能“成功”：正标签进入 `pos` 项，`P ≡ 0` 时 `L_act > 0` 且梯度非零。
- 不对每个窗口独立重新匹配来掩盖身份错误：匹配以整组序列为代价，只做一次。
- 不把空槽的全零当输入身份；空槽只贡献 `L_empty`，不参与同源/异源项。

## 6. 备选方案与取舍

1. **匈牙利算法**：O(K³) 且需要 scipy。K ≤ 8 时全排列枚举上界仅 40320，
   无额外依赖、可读性更好，因此选枚举；接口留有 `max_match_permutations` 便于替换。
2. **Sinkhorn / soft matching**：可微但会引入温度超参和熵正则，且“软匹配”与协议要求的
   一对一候选语义距离更远；本轮以硬匹配 + 对称平均作为参考实现。
3. **occupancy 头**：让模型自己预测 `slot_valid` 可表达“活动来源数 < K”，但需要额外的
   occupancy 监督与阈值校准，协议允许“有效候选 P = 0”，故本轮先恒为 True，空槽用
   `L_empty` 压低；后续如发现动态来源数收益明显，再以独立 issue 增加。
4. **匹配代价加入 embedding**：把预测 embedding 直接用于匹配会和身份对比项形成自举
   （模型可通过让 embedding 互相靠近来获得更容易的匹配）。本轮匹配只用活动序列和
   可选外部描述子；embedding 只进对比项。
5. **每窗口独立 PIT**：实现更简单，但无法表达跨窗身份一致性，明确不采用。

## 7. 未验证项

- 只使用 fake features（合成帧特征、合成活动序列）；未接真实 AuT 特征、未验证时间
  分辨率与真实乐器重叠场景。
- 未写完整训练脚本、数据加载、checkpoint、学习率/优化器 sweep 或显存测量；
  overfit 测试只证明“可优化 + 数学正确”，不构成真实数据性能声明。
- 描述子对比路径仅用合成向量测试；真实干净分轨辅助训练的收益未验证。
- 未验证 K > 8 的实时性（匹配枚举上界）；当前按“K 小”的设计前提实现。
- 轨迹关联、乐器分类、LED 渲染不在本 issue 范围。
