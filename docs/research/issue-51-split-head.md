# #51 响度与身份拆头：匹配结构对照

本轮只比较输出头耦合与拆分。基准为修正版 joint-teacher fit-v2 + ID .2；
旧纯包络教师、joint fit-v1/fit-v2 以及原 C0 失败证据全部保留。

共享结构完全相同：769→128 投影、4 个 Conv3/GELU、center98.25 加完整197 token
mean/std，固定 384 维读出。耦合头输出 V[8,128]，A=norm(V)、E=normalize(V)。
拆头保留原输出层作为 Z[8,128]，另加 Linear(384,8)，A=softplus(logits)，
E=normalize(Z)。693,000 参数对比689,920，额外3,080（0.4464%）。A 为非负标量；
Z 的模长不参与响度、教师幅度、预测可靠度或幅度重排。

联合教师收到显式独立 A 与原 Z/E。冻结可靠度、完整 F=L_amp+.2 L_ID、
block16 单遍坐标、全段幅度上界+.02、逐GT非零节点误差保护+.02保持原样。
教师与训练完整目标独立对照；单遍搜索不是全局最优。P/G 一对一。
ID 使用同源正对及双向异源 margin.2+.05 pull，offset1/2/5/15/30/45、radius45。
GT0 无身份；原独立 A 接受 GT0/余轨线性监督，无 .001 训练 gate。

推理仍为相邻 cosine、A/(A+.01r) 连续可靠度、Hungarian。
显式 A 硬排列，总 A 和逐帧活动槽数量守恒。cosine 用 E，不能用 Z 模长代替 A。
不改变阈值、关联策略、预测-only验证选择规则或身份维度。

## 公平初始化与明确限制

两臂均从历史 runs/window-depth/fit-v1/depth4/C1-raw/head.pth 完整复制共享层和
原128维输出层。拆头新响度层不使用未经解释的随机初始化：仅加载 C0/C1 train
的8片段缓存，冻结共享层及 Z，用旧 head 的 A 为目标。先对 inverse-softplus A 做
带1e-6 ridge 的训练特征最小二乘，再对缓存384维特征运行1000次 Adam lr.001 MSE
校准。不读取 val/test，不拟合 GT，不更新共享/身份参数。

耦合控制做同一训练特征上的1000次 readout 前向，已有精确零残差，保持参数。
这给出相同 pass 数，**不等于相同优化工作或墙钟成本**；两者成本分别记录。
scalar linear softplus 无法精确表示128维 affine vector 的 norm，因此起始 A 仍有残差。
不能把起点差异归因于拆头训练收益。原权重、两臂初始化 SHA、逐片段 A/数量差、
共享/Z一致性和初始 actual/raw/teacher 诊断单独保存。

## 固定执行和审计

完整 full197x768 float16 缓存由 train_window_depth.read 读取，不做6token裁剪。
四课程 C1/C2-low/C2-high/C3 各1000更新，seed4650..4653，50/50 replay，
AdamW lr.001 decay.0001 clip1，每50步 val，7200秒安全预算。线性 r=.2591032087802887。
GPU0 bf16 head，损失/关系float32，TF32 enabled，SSAST冻结。
预测-only评分为 MAE/r+.1(1-meanIoU)+.1unusedmeanA/r+.01countMAE，所有已见val
等样本平均，test及teacher不选checkpoint。pilot每臂20次train-only更新，另行计成本。

结果独立保存在 runs/split-head/fit-v1。每阶段 selected/last、完整已见val/test的
原 Z/A/actual/teacher NPZ、重载验证、SHA、冻结源快照、执行PID/status/log保留。
报告原轴 IoU/MAE、静音/余轨FP、面积/数量/复制、跨静音同预测行与严格GT源行端点、
cos/shared gradient、遗忘、初始化差、成本/停止原因及近似/标签限制。

40项小验证通过：独立A/Z尺度不变性、输出路径梯度独立但共享特征两项梯度仍存在、
GT0/近零、完整教师/torch损失一致、P/G一对一与A/数量守恒，及原36项回归。
固定小样本单seed不能支持统计显著性、未见音色/输入增益泛化或SSAST-vs-RMS收益。
不合并PR，不启动子代理，不改共享checkout及#46资产。
