# Issue #46：局部非零端点、直接锚定与轨迹碎片

日期：2026-10-04。规范性讨论引用 [项目设计澄清 #48，第 1..6、8 节](https://github.com/guajun/anonymous-audio-tracks/issues/48)。
本文只描述 #46 研究分支的可执行参考实现，不改 main 的项目协议或未来回环/容量设计。
历史原型实现保留在 `association.py`，旧数据/缓存/运行不覆写；原先结果见
[旧关联报告](reports/issue-46-association.md)、[输入共现修正](reports/issue-46-local-context.md)。
本轮实测见 [端点实验报告](reports/issue-46-endpoint.md)。

## 输出语义和门控

头部仍无学习 Query，共享映射产生方向与活动强度。非零输出对应 `V=A*E`，
`norm(V)=A`；A 是线性 RMS 强度。严格零 A 的 E 为严格零/无效，不能读取静音身份。
没有增加持久 E/Q 头、下边沿事件头、音色排斥或伪持续底音。
进入零是身份活动表征的边沿，具体事件编码尚未选择，不把它解释为 ADSR release 或物理结束。

本轮头部增加 `magnitude_gate=0.002`：低于门限的模长 forward 精确为零。
训练使用 `a + stop_gradient(gate(a)-a)`，门关闭时 A 的代理梯度仍为 1；
这不是硬门真实导数。E 的有效掩码使用门后非零输出，关闭区间没有描述子匹配梯度。
推理硬门的关闭区间梯度为零，测试分别验证；记录门前/门后活动、全零帧与关闭区间实际主损失梯度。
这是一项实验附加变量，不能把新旧差异全部归因于关联器。

soft 搬运会产生微小串扰。v2 起对搬运模长也使用 0.002 的同类门，
低于门限的结果不成为新端点或身份活动，避免别的来源在休止时不断覆盖端点。
直接锚定帧不重复门控，严格保留头部 A。搬运门也可能抑制真实弱活动，须同时记录
`transport_pregate_sum_mean`、门移除量和最终活动；不把这个损失量藏在 null 指标中。
v1 未使用搬运门的六组实际诊断独立保留，commit `12cd8cd`；v2 commit `2babaab`。
两版主头门相同，关联端点更新/轨迹碎片与辅助 cycle 的路径会随搬运门改变。
正式 v3 commit `89aea3f` 进一步排除 cycle 中的零候选容量列及无有效端点的行，
分别按同一有效非零端点集合条件归一，避免把空容量当作身份状态。
v1/v2/v3 各六组均实际执行，历史目录与语义版本独立。

## 初始化与每一步关联

局部候选容量 K=8，大于本轮实际来源数 2。匿名 lane 是软计算容量，不能把 lane 索引作为永久 track_id。

1. 首窗有效候选与匿名轨迹直接自对应，`C0=I`、`tracked_A0=A0`。
   完全零首窗没有种子；第一个有效活动帧仍直接锚定，不通过 self-cosine/birth/置信度证明自对应。
2. 每个 lane 存储最近一次有效非零输出的实际候选 E、对应权重与原始中心时间。
   这是有限范围内的真实事件端点分布，既不是新模型输出，也不是归零向量里的 E。
   不生成 normalized mean prototype，不按熵/可信度混合更新旧原型。
3. 对当前非零候选，打分是端点分布对实际候选 cosine 的加权平均。
   当前零候选没有 E 分数，只保留匿名空容量列。无有效端点的 lane 没有身份分数，使用中性出生容量。
   `similarity_gate=0.5` 在此是有证据与空容量的相对 logit 偏置，不是旧原型硬匹配门限。
4. 对 K×K logits 做 12 次独立行/列 log-normalization，并令列精确归一。
   当前 A 用同一矩阵搬运；有限迭代的行容量仍近似，报告最大残差。
   门前搬运严格保留当前非零 A 总量；没有把真实当前 A 送入 dustbin 的通道。
   null 只报告旧 lane 分给当前零候选容量的量，不是被隐去的新活动，也不是来源死亡判定。
5. 搬运门后有效输出无条件成为最新端点分布，更新不依赖匹配熵。
   归零输出不会覆盖端点。仍在范围内的旧非零事件记录可用于短静音两端匹配。
   新 lane 的有效活动开始一个匿名碎片；候选对称时出生/对应仍可能混合，须如实评估。

最新证据更新可避免旧实现的首窗滞留，但不保证正确来源对应。
无匹配时也可能出现强制混合或重复活动；本轮不以容量丢失解释这些错误。

## 短静音证据与真实断轨

使用 cache 中真实原时间轴中心和实际 2 秒输入边界，不用长训练段或 7.8 秒执行补零。
两端中心距离必须严格小于 `W/2 - edge_guard - evidence_margin`，本轮为
`1.0 - 0.05 - 0.04 = 0.91 秒`；两端各自位于对方真实窗口的内侧余量内。
这是一项保守可用条件，不是正确匹配定理，也不是 W/2 的普适必要界限。
来源连续零中心数量×hop 和两端正值中心距离不同；审计/指标分别保存正值端点时间。

当一个来源归零而另一个仍发声，搬运门可阻止微小串扰形成其有效端点；
最近真实非零端点保留到共同证据边界内，以关联后续非零片段。
中间零输出的 `fragment_ids=-1`、E 无效，不构造某乐器持续的零身份。
保存有限的先前事件记录及对应关系，不能被解释成零向量保留身份或无界声学记忆。

超过证据范围，删除端点及该 lane 对旧碎片的活动连接。后来活动即使使用同一个 lane，
也获得新的 fragment_id 和独立曲线列，不复用旧 track_id。
如果此时其他来源仍发声，空容量可能形成新的错误出生；这是待评估的重复/混合错误，
不能借 lane 重用把旧碎片偷偷接回。导出有效帧 ID、直接锚定帧、到期帧及所有有限端点连接。
本轮没有离线检索、裁剪音频重推理或 SLAM 回环。

当前接口每窗仍只输出中心候选。它没有显式输出“上下文内两个音符相同来源”的事件关系。
拼接器明确比较两端非零中心候选，使用双方输入已有共同事件的训练条件；
这个接口在形式上可表达跨零关系，但是否学会从共同上下文提取所需方向，仍须实测，不能从输入共现直接推出成功。

## 完整轨迹监督和 TimeCycle

片段重建为 `[T,F]` 的独立碎片曲线，F 可大于 K；同时活动仍只有 K lane。
没有把 long-gap 之后重用的 lane 压回同一曲线。全片段一次 PIT 将完整参考来源注入碎片列，
不能逐帧或逐事件重配 loss，也不把碎片预先按参考答案聚合。
同来源断成两条碎片时，一条不能拟合完整参考，另一条还承担 unused-output 代价，真实断轨受到惩罚。

F>8 时，本轮 1/2-source 课程使用精确 rectangular PIT（枚举来源注入的成本最优组合），
不用指数级 `2**F` 动态规划。并列最优仍对称平均；过多并列报错。
更多来源的 wide-fragment PIT 未实现，不能宣称已支持 C4。
空碎片/未用曲线的代价分母固定为 `max(1,K-source_count)`，不随 F 增加稀释误报惩罚。
这是与历史固定 K 曲线相比的目标/重建变更，须在因果比较中标记。

TimeCycle 的正反矩阵在同一组有限端点/当前窗口 logits 上分别归一；
辅助往返只保留有效旧端点行与当前非零候选列，再分别条件归一，零容量占位不参与身份往返。
反向不是正向硬逆。掩码取两端参考活动、唯一整段 PIT 和无相同包络歧义的来源。
跨度可以跨中间零帧；没有把零帧 E 加入 cycle，也不使用陈旧 prototype 的往返。
cycle=0 仍执行主关联，并且单独测量 raw cycle；辅助系数保持 0 / 0.001。
shape 经门的代理 A 梯度及可微软矩阵回传 E，测试验证后段分岔能够到达早期 E。
非零梯度、低 cycle 或排列不变都不证明正确轨迹。

## 运行和历史复现

正式六组保留现有 `cache-c1-local-v1`，相同 seed46/head 初始化/抽样序列/100步预算：

```sh
for family in l1 huber_iou; do
  for mode in index soft0 softcycle; do
    association=soft; cycle=0
    if [ "$mode" = index ]; then association=index; fi
    if [ "$mode" = softcycle ]; then cycle=.001; fi
    python scripts/envelope_experiment.py train --cache runs/envelope/cache-c1-local-v1 \
      --out runs/envelope/c1-endpoint-v3-${family}-${mode}-100 --device cuda:0 \
      --steps 100 --seed 46 --lr .0003 --slots 8 --segment-centers 180 --families "$family" \
      --association "$association" --association-backend local --magnitude-gate .002 \
      --cycle-weight "$cycle" --empty-weight 1 --null-weight 1 \
      --tensorboard-root runs/envelope/events --eval-every 50
  done
done
python scripts/report_envelope_endpoint.py --root runs/envelope \
  --out docs/reports/evidence/issue-46-endpoint.json
```

证据采集脚本同时读取 v1/v2 历史数值；没有这些文件时可先分别 checkout
`12cd8cd` / `2babaab` 跑同组六组，使用独立 `c1-endpoint-v1-*` / `c1-endpoint-v2-*` 名称，再回正式版本。
历史长期原型对照使用 `--association-backend legacy --magnitude-gate 0`；
首轮旧 C1 另有 `--allow-partial-recurrence`，不能把不同采样的旧结果混作控制变量。
checkpoint/result 显式保存 head gate、关联版本、配置、cache 摘要和代码 commit。
