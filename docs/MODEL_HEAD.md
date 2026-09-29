# E/P 输出头与排列无关训练损失（issue #7）

本文件固化 `aat.models`、`aat.losses`、`tests/models/` 的结构、公式与训练取舍。协议仍为 `docs/SCHEMAS.md` v0.1.0：`E[N,K,128]`、`P[N,K]`，K 可配置、embedding 维度固定 128；`P[n,k]` 是窗口中心时刻的活动概率，不是“整个窗口曾活动”。本模块只用 fake features，不加载 AuT、不做完整训练脚本、不做乐器分类或灯光。

## 1. 输出头（`aat.models.SourceQueryHead`）

输入（B 个窗口，T 帧，F 维特征）：

| 名称 | 形状 | 语义 |
|---|---|---|
| `features` | `[B, T, F]` | 冻结骨干帧特征，帧内无来源语义 |
| `frame_times` | `[B, T]` | 帧的原曲绝对秒（或任意公共时间轴上的秒） |
| `frame_valid` | `[B, T]` bool | `False` 为补零/无效帧，注意力屏蔽且不参与任何输出 |
| `center_times` | `[B]` | 窗口中心时刻；与 `frame_times` 同轴 |
| `center_valid` | `[B]` bool | 可选；窗口中心是否完全落在音频内，缺省等于“存在有效帧” |

相对时间 `Δ_t = frame_times - center_times`。时间编码使用 `L` 个倍频程频率 `f_j = f_0 · 2^j`（`j = 0..L-1`，默认 `f_0 = 1 Hz`，`L = 6`）：

```text
τ(Δ) = [ sin(2π f_j Δ), cos(2π f_j Δ) ]_{j=0..L-1}       # R^{2L}
```

1. 内容投影与时间嵌入（padding 处清零）：

```text
x_t = W_c features_t + b_c                          # 内容，不含时间
u_t = MLP_τ(τ(Δ_t))                                 # 时间嵌入
s_t = x_t + u_t                                     # 中心分支的键
```

2. K 个可学习查询 `q_k ∈ R^d`（`k = 0..K-1`），K 默认 8、可配置。两个分支各自做屏蔽多注意力（softmax 在有效帧上归一化）：

```text
identity 分支（身份不应随窗口中心漂移，键/值只用内容）:
    α^id_{k,t} = softmax_t( q_k · x_t / sqrt(d) )
    h^id_k     = Σ_t α^id_{k,t} x_t
center 分支（活动必须落在中心，键含时间、并加每查询时间偏置）:
    s_{k,t}   = q_k · s_t / sqrt(d) + u_t · g_k / sqrt(d)
    α^ctr_{k,t} = softmax_t(s_{k,t})
    h^ctr_k    = Σ_t α^ctr_{k,t} x_t
```

`g_k` 是每查询的可学习时间偏置向量：查询可以学会把注意力集中到 `Δ ≈ 0`（当前中心帧）而不是整窗平均。`softmax` 只在 `frame_valid` 上归一化；全屏蔽时退化为对零向量的均匀注意力，输出有限，随后按下面的有效性规则清零。

3. 残差 + FFN + LayerNorm（两分支各一份）：

```text
e_k   = E_head( q_k + h^id_k  + FFN_id(...) )        # R^128
logit_k = P_head( [ h^ctr_k + q_k , q_k ] )          # 对中心活动打分
E_k   = e_k / max(||e_k||, eps)                       # 单位化；退化时回退到固定单位向量
P_k   = sigmoid(logit_k)
```

`P_head` 同时看到查询自身（“我在找谁”）与中心注意力池化结果（“中心帧里有什么”），因此可以在“前后文有该来源、中心帧静音”时输出低 `P`。身份分支不含时间、且共享同一 `q_k`，所以静音中心不改变身份表示；`E` 仍是单位向量（身份记忆），`P` 可以接近 0。

4. 槽位有效性与协议映射：

- `center_valid[b] == False` 或该窗口没有有效帧：`slot_valid[b, :] = False`，`E = 0`、`P = 0`（无效槽位全零，不携带身份）。
- 否则 `slot_valid[b, :] = True`：`E` 必须有限且 L2 单位（容差 1e-3），`P` 可为 0（静音身份记忆）。
- `HeadOutput.to_prediction_data(center_times)` 产出 `aat.contracts.PredictionData`，可直接 `save/load`；无效槽位在映射时再次强制为全零。

## 2. 组级匹配（`aat.losses.matching`）

一个训练组 = 同一首曲子的多个中心窗口，来源列与 `source_valid` 跨窗固定。记组 `g` 的有效来源集合 `S_g`、有效中心窗口集合 `N_g`。匹配代价只用组内活动序列（不逐帧重配；embedding 不参与匹配，避免用当前身份表示自证）：

```text
C_g[s, k] = (1 / |N_g|) · Σ_{n ∈ N_g} BCE( logit_{g,n,k} ; a_{g,n,s} )     # detach
Z_g       = argmin_{单射 z : S_g → K_g} Σ_{s ∈ S_g} C_g[s, z(s)]           # K_g 为可用槽位
```

- 精确求解：小 K 用位掩码 DP（`aat.losses.matching.enumerate_optimal_assignments`），同时枚举**所有**最优 assignment，并用饱和计数给出 `num_optimal`；K 超过 `max_exact_slots`（默认 16）直接报错，不退回贪心或在线 tracker 式的基数优先近似。
- 若 `|S_g| > K_g`，说明来源数超过槽位容量，拒绝并报错。
- 不可辨来源（例如活动序列完全相同）会产生并列最优：`Z_g` 有多个元素。对**完整枚举**的等价 assignment 在活动项与空槽项上做对称平均；身份项属于“无依据的身份约束”，对并列组整体 mask（不算作假监督）。“对称平均”与“mask 掉无依据身份约束”都是事前确认允许的等价处理，这里按后者处理身份、前者处理活动/空槽，绝不任意固定列制造虚假真值。`MatchingResult` 显式报告 `ambiguous`、`num_optimal`、`truncated`，且 `GroupMatching.identity_masked` 对并列与截断都为 True。
- 枚举达到 `max_optimal_assignments`（默认 64）仍未穷尽时 `truncated=True`：无法认证完整最优集合，故该组活动、空槽、身份项**全部 mask**（不采用任何未认证的 assignment；等值前向下的梯度也会因列选择而不同，不能用“值相等”当作对称性）；`truncated_groups` / `supervision_masked_groups` 计数，`activity_terms` 不计入该组。不贪心退化，也不假装监督了无法枚举的约束。
- 索引约定：`GroupMatching.optimal.assignments[j]` 是来源 `source_indices[j]` 的**原始槽位下标**；`match_sources` 已把 `enumerate_optimal_assignments` 的局部下标映射回 batch 原始下标，消费方不得再经 `slot_indices` 重映射。
- 并列判等用 `atol/rtol = 1e-9`；枚举按槽位下标升序，结果稳定。

## 3. 训练损失（`aat.losses.head_loss`）

输入：`E[G,N,K,128]`（单位向量）、`logits[G,N,K]`、活动真值 `a[G,N,S]∈[0,1]`、`source_valid[G,S]`、`center_valid[G,N]`、`slot_valid[G,N,K]`、每组的 `composition_ids` 与 `source_ids`。所有项只统计有效证据，padding 与无效中心不进入分子分母。

### 3.1 匹配后中心活动 BCE（`activity`）

对每个最优 assignment `z`（并列时权重 `1/|Z_g|`）：

```text
T_act = { (g, n, s) : n ∈ N_g, s ∈ S_g, slot_valid[g,n,z(s)] }
L_act = (1/|T_act|) Σ_{(g,n,s)} BCE( logit_{g,n,z(s)} ; a_{g,n,s} )
```

这是“匹配后”的 BCE，包含被判为静音（`a=0`）的来源；中心有效性、槽位有效性与来源 padding 全部 mask。

### 3.2 空槽 / 重复占槽（`empty_slots`）

未被 assignment 占用的可用槽位是空槽（不是静音来源；静音来源仍被匹配、只降 `P`）：

```text
T_empty = { (g, n, k) : n ∈ N_g, k 可用且 k ∉ z(S_g) }
L_empty = (1/|T_empty|) Σ BCE( logit_{g,n,k} ; 0 )
```

一一匹配本身禁止一个来源占多个槽位、或一个槽位承载两个来源；`empty_slots` 再压低剩余容量。权重默认 `0.5`。

### 3.3 同源跨窗正对比（`positive`）

身份 key 为 `(composition_id, source_id)`：**必须同曲同 source_id** 才算同源；不同曲即使字符串相同也是不同身份。锚点限定在中心有效且活动真值 `a ≥ identity_activity_threshold`（默认 0.5）的匹配窗口——静音没有身份可辨信息时不强制更新身份。

对同 identity `e` 的锚点集合 `A_e`，对每个锚点做留一原型（`stop_gradient`）：

```text
p_e^{(-j)} = normalize( mean_{i ∈ A_e, i ≠ j} E_i )
L_pos = (1/|T_pos|) Σ_j ( 1 - cos( E_j, p_e^{(-j)} ) )
```

留一避免“只含自己的原型与自己完全相似”的零梯度；同一组内跨窗、以及批内同曲同 source_id 的窗口都可成为正例；只有一个锚点的身份不产生正例。

### 3.4 异源负例（`negative`）

对每个锚点，取批内所有其他 identity `e'` 的原型 `p_{e'}`（同样 `stop_gradient`，且只用活动锚点），hinge：

```text
L_neg = (1/|T_neg|) Σ_{(j,e')} max( 0, cos( E_j, p_{e'} ) - margin )
```

`margin` 默认 0.25；批内只有一个 identity 时无负例（项为 0，但保持梯度通路）。

### 3.5 总损失与权重

```text
L = w_act·L_act + w_empty·L_empty + w_pos·L_pos + w_neg·L_neg
```

默认 `w_act = 1.0`、`w_empty = 0.5`、`w_pos = 1.0`、`w_neg = 1.0`。分项、有效项计数、匹配结果与歧义统计全部从 `HeadLoss` 暴露，便于诊断；`LT_act/empty/pos/neg` 为空时该项为 0，并保留回到输入的零梯度通路，因此空来源、全静音、全无效中心批也能安全反传（有限梯度，不把 0 当成功）。

## 4. 环境与 CI（CPU-only torch）

`ml` 可选组只包含 `torch>=2.5,<3`，并通过 `[[tool.uv.index]] name = "pytorch-cpu"` 与 `[tool.uv.sources] torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]` 解析 CPU wheel：

- `uv sync --locked`（无 extra）不安装 torch，基础契约/窗口/render 测试照常；
- `uv sync --locked --extra ml` 在 Linux/Windows 从 PyTorch CPU 索引安装 `x.y.z+cpu`，不引入 nvidia/cuda 依赖；
- macOS 回退默认 PyPI（macOS 的 PyPI torch 本身是 CPU 版）；CI 只覆盖 Linux CPU。

CI 在原 `unit` / `render` job 之外新增 `ml` job，执行 `pytest -m ml` 并断言 `torch.version.cuda is None`；`unit` / `render` 通过 `-m "not ... ml"` 与 `pytest.importorskip("torch")` 在无 torch 环境下 skip 而非报错。`aat.models` / `aat.losses` 只属于 ml extra，基础环境不导入它们。

## 5. 实际验证与过拟合结果

本机（Windows 11，Python 3.12.11，CPU-only `torch 2.14.0+cpu`，uv 0.12.19）实际运行：

| 命令 | 结果 |
|---|---|
| `uv lock && uv sync --locked --extra ml` | 解析并安装 CPU wheel `2.14.0+cpu`，`torch.version.cuda is None` |
| `uv run --no-sync pytest -q`（ml + render 组合） | 442 passed |
| `uv sync --locked && uv run --no-sync pytest -q -m "not ml"` | 395 passed, 6 skipped（无 torch 时 ml 模块 importorskip，不报错） |
| `uv run --no-sync pytest tests/models -q` | 43 passed |

过拟合实验（`tests/models/test_overfit.py`，合成 fake features：3 组 × 5 个非相邻窗口 × 3 来源，K=4，250 步 Adam，CPU）：

```text
loss 1.6938 -> 0.0219; activity 0.6881 -> 0.000094 (zero-logits / P=0.5 baseline 0.6931);
F1=1.000; all-inactive (P=0) F1=0.000, accuracy=0.422; P(active)=0.9999 P(inactive)=0.0001
boundary P=0.000111 (n=7)   # 前后文有来源、中心静音
isolated P=0.999930 (n=12)  # 中心有声、前后文静音
identity cos same=0.9998 different=0.1593 (pairs 56/126)
```

零 logits（P=0.5）基线 BCE 为 `ln 2 ≈ 0.6931`，明显高于训练后的 0.000094；全静音预测（P=0）在该数据上 F1=0、准确率=负例占比（有正例时不可能完美）。持续静音 batch 的反传与优化步保持 finite；活动、空槽、正例、负例四个分项及歧义/截断统计都有独立断言。

## 6. 与既有模块的边界

- 不修改 `docs/SCHEMAS.md`、`src/aat/contracts/`、`src/aat/windowing.py`、`src/aat/tracking/`、`src/aat/evaluation/`、`src/aat/labels/`、`src/aat/render/`；`PredictionData` 只作为输出校验/序列化入口。
- 滑窗推理入口仍是 `aat.tracking.track_audio(predictor=...)`；本头可作为 predictor 适配器接入，但真实 AuT 特征与训练计划属于 #5/#8。
- 只使用 fake features 的 CPU 微型实验，不加载 AuT、不写完整训练脚本、不做乐器分类或灯光映射。
