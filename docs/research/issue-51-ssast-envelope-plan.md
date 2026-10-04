# #51 SSAST 连续匿名包络：最小验证草案

状态：草案建档后，用户于2026-10-04授权开工；
执行进展见 [P0报告](../reports/issue-51-p0.md)。下文保留原草案阶段与预算，
其中“本轮仅建档/不执行”指最初建档轮，不覆盖后续用户授权。日期：2026-10-04（UTC+8）。
研究 [issue #51](https://github.com/guajun/anonymous-audio-tracks/issues/51)，
分支 `codex/issue-51-ssast-envelope`，
[draft PR #52](https://github.com/guajun/anonymous-audio-tracks/pull/52)。
[本研究主会话](codex://threads/01a1057b-fc7c-7612-91ba-ec3b9af45e41)；
[发起会话](codex://threads/01a1052d-03e2-7a40-817b-81e1baab0756)。

## 1. 研究边界与具体问题

从合入数据管线 [PR #50](https://github.com/guajun/anonymous-audio-tracks/pull/50)
后的最新 origin/main `5e78f39c42bd549299a1baee4d1e65cd26a7f769` 开始。
规范以 [IDENTITY_ACTIVITY_DESIGN](../IDENTITY_ACTIVITY_DESIGN.md) 与
[issue #48](https://github.com/guajun/anonymous-audio-tracks/issues/48) 为准。
本研究不是早期存在概率、静音身份保留或全局音色排斥方案的延续。

问题按可否证顺序：
1. P0：官方预训练 SSAST 的未池化 token，能否在真实 2 秒上下文中提供可校验的时间映射、幅值响应及可接受成本？
2. P1：冻结这些特征后，小型共享头能否拟合 C0 实测线性 RMS，包括起止边沿、静音和未用候选？绝对 RMS 旁路贡献多大？
3. P2：在 C1-local 的同来源真实局部共同证据下，保留时间结构是否改善形状及正确端点对应？整段 shape 是否能通过搬运权重向 E 传有效梯度，而非复制混合曲线？

SSAST 是新假设，不能预先声称胜过 Demucs。自监督频谱预训练不保证匿名实例分解或精确响度。
本轮不下载/加载权重、不渲染数据、不运行 P0/P1/P2、不训练、不合并研究 PR。
**TimeCycleLoss 不实现、不导入、不接入本轮损失**；后续局部关联不能依赖 cycle 开关。
用户对草案反馈之后再确定执行范围。

## 2. 官方依据与 checkpoint 选择

已读 [SSAST 论文 v2](https://arxiv.org/abs/2110.09784v2)、
[官方 README](https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/README.md)。
代码核查固定在官方提交 `a1a3eecb94731e226308a6812f2fbf268d789caf`，不是依赖浮动 main。
下表的“公开权重”表示官方 README 确实列出链接；**二进制可达性、SHA256、state_dict 与加载尚未验证**。

| 角色 | 官方公开权重及来源 | 选择理由/限制 |
|---|---|---|
| 首选 | [SSAST-Base-Frame-400.pth](https://www.dropbox.com/s/nx6nl4d4bl71sm8/SSAST-Base-Frame-400.pth?dl=1) | 全 AudioSet + Librispeech，自监督 base（官方约89M），128×2 形状；优先验证连续时间输出 |
| 后续对照 | [SSAST-Base-Patch-400.pth](https://www.dropbox.com/s/ewrzpco95n9jdz6/SSAST-Base-Patch-400.pth?dl=1) | 同训练数据、base、16×16；频率局部结构不同，不能把分类结论直接外推包络 |
| 成本失败备选 | [SSAST-Tiny-Frame-400.pth](https://www.dropbox.com/s/rx7g60ruzawffzv/SSAST-Tiny-Frame-400.pth?dl=1) | 官方约6M；不同容量是单独实验，不能替换后仍称同骨干对照 |

两类都采用联合 masked spectrogram modeling：辨别正确被遮盖频谱块的 InfoNCE/NCE，
以及重建频谱块的 MSE，官方预训练组合为 discriminative + 10×generative。
这是预训练任务，**不是本研究要重新训练的目标**。frame 将全128 mel、2时间帧组成一个块；
patch 将16 mel、16时间帧组成块。官方
[frame recipe](https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/pretrain/run_mask_frame.sh)
与 [patch recipe](https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/pretrain/run_mask_patch.sh)
均用400个 mask、1024帧输入、无 token 重叠。frame 使用时间 mask，patch 使用簇状频谱 mask。
官方 audio 分类更推荐 patch、speech 更推荐 frame；本研究优先 frame 的依据是时间接口，
不是已证实的音乐包络优势。

## 3. 输入、token 网格与位置映射（代码推导，待 P0 实测）

核查源：
[主 dataloader](https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/dataloader.py)，
[ASTModel](https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/models/ast_models.py)，
[官方 SUPERB audio 包装器](https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/finetune/superb/upstream/ast/audio.py)，
[expert](https://github.com/YuanGongND/ssast/blob/a1a3eecb94731e226308a6812f2fbf268d789caf/src/finetune/superb/upstream/ast/expert.py)，
[torchaudio 0.10 fbank 默认值](https://docs.pytorch.org/audio/0.10.0/compliance.kaldi.html#fbank)。

- 计划显式重采样为16 kHz单声道，与官方 SUPERB 的 SAMPLE_RATE=16000 对齐。主 dataloader
  直接传入文件 sr，**并不自动重采样**；不能将任意44.1/48 kHz文件直接当作已符合 checkpoint 输入。
  声道策略固定并记录；RMS旁路从原始声道功率平均计算，避免反相信号抵消。
- Kaldi log-fbank，128 mel，hanning，25 ms窗、10 ms移位、htk_compat=True、
  use_energy=False、dither=0、snip_edges=True。显式记录默认 preemphasis=.97、
  remove_dc_offset=True、low_freq=20、高频至Nyquist、FFT向上取2幂、log/power=True。
  主 dataloader 还先减波形均值。将依赖版本及实际参数写入特征manifest。
- 固定全数据统计归一化：(fbank - (-4.2677393))/(2×4.5689974)；不逐窗峰值归一化，
  不做每段 CMVN。固定统计不等于完全抹掉幅值，但log、DC处理与encoder可能削弱幅值校准；
  故另保留原尺度混音 RMS 时间序列。不能用它在C0的成功证明分离。
- 主 dataloader 先把log-fbank补0/截断，再归一化；SUPERB先归一化再补0，padding值不同。
  P0首选不补频谱：完整2秒16 kHz波形在上述默认下推导出198帧，input_tdim=198。
  所有波形边界补零另记真实有效掩码；不为兼容checkpoint把2秒伪装成10.24秒真实证据。
  若改成200帧，须作为独立padding核查，不静默改协议。

设裁剪起点为 s，帧中心 c_i=s+0.0125+0.010i（秒），帧支持区间
[s+0.010i, s+0.010i+0.025]。此处中心为连续区间约定，离散样本中点差异须P0实测记录。
无padding Conv2d：T_out=floor((T_in-tshape)/tstride)+1，
F_out=floor((128-fshape)/fstride)+1。token列 j 的投影中心：
s+0.0125+0.010(j×tstride+(tshape-1)/2)。

| 配置（频率×时间） | T_in=198 的投影网格 | 时间步长与支持 |
|---|---|---|
| frame首选 shape128×2，stride128×1（官方下游配置） | F=1,T=197；base token维度由模型代码为768，仍核查权重 | 10 ms中心步长，首中心17.5 ms；投影支持35 ms |
| frame成本对照 stride128×2（预训练stride） | F=1,T=99 | 20 ms步长；同35 ms投影支持 |
| patch后续 shape16×16，stride10×10（官方下游配置） | F=12,T=19 | 100 ms步长，首中心87.5 ms；投影支持175 ms |

上述是**投影网格推导，不是实测有效分辨率**。Transformer self-attention 的每个输出可看整段2秒；
不能把35 ms投影支持当作E仅看35 ms。频率token展平次序是频率外层、时间内层，
去除 cls/dist 两个特殊token后重排为[B,F,T,D]，可频率聚合为[B,T,D]，保留时间轴。
取 transformer blocks 后 norm 的非特殊token，绝不以ft_avgtok/ft_cls最终分类embedding作为包络特征。
不用官方包装器的固定downsample_rate=160概括patch网格。

主ASTModel加载预训练形状和input dims，要求形状相同，允许改变stride；
短时间输入中心裁剪预训练位置嵌入，频率/更长输入另有双线性插值。
P0记录旧/新位置张量形状、裁剪起点、插值路径、所有missing/unexpected keys：
不以strict=False无报错代替正确加载。绝对音频时间由裁剪起点与投影公式决定，
不是由预训练位置嵌入索引决定。代码对更长输入/频率裁剪有固定尺寸或索引假设；
本轮2秒frame/patch不扩展到任意长输入，运行前核对实际路径和输出shape。

## 4. 数据、架构与关联建议

直接复用 main 的 `aat.data.curriculum.prepare_c0/prepare_c1/prepare_c1_local`、
`aat.labels.envelope.EnvelopeData/label_sample` 与
`aat.data.local_context.audit_local_context`、`scripts/prepare_curriculum.py`。
不复制、不修改这些模块来隐式适配SSAST。模型侧适配放独立研究目录。
标签是效果后、进入混音分轨的线性RMS，默认20 ms能量窗/20 ms网格。
在实际预测中心重新调用标签提取/审计，记录重采样和对齐误差；不得无说明将20 ms标签插值宣称10 ms真值。

起点使用课程默认4/2/2（C0 seed4600，C1-local seed4640），固定pad/pluck。
这是独立事件、ADSR、gain等的留出，不是未见音色泛化。先在train小集拟合，
val选停止/checkpoint，test封存至配置冻结。重跑 seeds46/47/48；训练seed与渲染seed分别记录。
较长休止的C1仅作可见证据负例/碎片化检查，不要求2秒上下文重连长间隙。

2秒输入 W；建议滑窗中心步长 H=20 ms，中心附近token形成局部输出；
P0确认选择的token/插值位置，评分按**实际输出中心**重审C1-local（默认1..11秒评分区间）。
若一窗多中心输出，审计必须使用该次裁剪真实左右边界，不能为每token假设另一个中心2秒窗。
训练完整样本按中心窗提特征再拼整段轨迹，特征可缓存；冻结骨干eval/no_grad，避免缓存训练期漂移。

query-free共享头：频率聚合后保留token时间序列，投影至hidden128；
两层小型时序Conv1d（kernel3起点，可后续与小attention比较），在中心处输出K×128的V，K=8。
共享头指同参数用于各窗，候选槽本身不等于跨窗身份。绝对RMS旁路作为标量序列输入头，
不直接替代每源预测。不预承诺双头或Query；可对V参数化/gate另作单变量研究。

非零 A=||V||、E=V/A；严格零E未定义且不匹配。训练先检查门前输出；
门控阈值建议评估1e-3 linear RMS并扫0.5e-3/2e-3，阈值不是存在概率。
硬gate造成关闭区间无梯度，训练门控/代理方式尚待反馈选择；不得默认硬门训练成功。
必须分开报告raw、门前/门后与阈值敏感性。

P2主链：最新有限局部共同事件E打分 → 可微局部匹配C → 同C搬运A →
整段一次PIT → shape。首有效候选直接以单位对应锚定，首窗全零不建身份；
不soft self-match，不用长期陈旧原型。跨多个零中心保存局部关系，零输出不提供E。
只连接真实共同证据内的两端，长间隙断为片段；本轮不实现回环。

建议采用带空匹配的容量约束soft assignment（例如有限迭代Sinkhorn）而不是逐行softmax复制。
起点temperature=.1、迭代20，归一化/空槽质量/退化条件须实现前明示并核验；
这只是候选算法，不把double-stochastic或低熵称为正确身份。
raw A整段PIT是诊断对照，index固定槽是工程对照。hard匹配可作评估，不作为shape→E主梯度路径。
不同音色形状一致时允许歧义，不强制E永远排斥。

## 5. 损失草案与单变量顺序

以固定训练集参考活跃RMS的P95为尺度 r（全数据共同尺度、train-only、写入manifest；
若r=0停止数据验收）。训练可在a=A/r上计算，报告还原full-scale。
静音审计门限1e-3是数值约定；严格零语义仍按V=0。

| 项 | 建议起点 | 定义/风险 |
|---|---|---|
| 主shape基线 | L1(a,b) | 同时间点误差；偏移会惩罚正确但错时的形状 |
| 主shape对照 | Huber(delta=.1)，除以delta使大残差斜率约1 | 原单位delta=.1r，显式torch Huber而非混称SmoothL1；逐点时移风险同上 |
| 小权重shape组合 | 上项 + .1×(1-areaIoU) | 积分min/积分max，同t对齐，**不是平移不变** |
| 静音 | .1×均值a（已匹配参考为零处） | 独立记录mask数量、贡献与主项重复区域 |
| 未用候选 | .1×均值a（整段PIT未分配候选） | K8余槽独立报告；不按全体均值掩盖活动误差 |
| TimeCycle | 无 | 不实现/接入；主关联照常工作 |

areaIoU仅在参考面积>0的来源段计算；全零参考另列静音误报，
不以epsilon使全零“完美IoU”拉高均值。整段PIT代价与训练shape使用同一组合，
一次固定一对一分配；不逐点/逐帧改来源。记录近并列：次优与最优代价差
<=1e-4×max(1,abs(best))（归一化代价约定）。

先P1选shape稳定配置，再固定该loss进入P2；对照顺序为：
时间序列头 vs 全窗均值池化（同骨干/RMS旁路/预算）；
有/无绝对RMS旁路；frame stride1 vs2；最后才比较patch。
池化对照仍在同中心输出V，不用最终分类head，明确它丢失相对时间结构。
骨干解冻、直接局部对应辅助、防塌缩辅助只作**后续另列可选消融**，
不默认全局instrument-ID contrastive，不将多变项收益归给backbone。
若shape路径不向E传梯度，先诊断/修复主链，不以辅助loss掩盖。

## 6. 阶段、预算与可否证门槛（建议，非结果）

| 阶段 | 执行/预算上限 | 通过条件与失败行动 |
|---|---|---|
| P0 | 一次frame权重核验；8个2秒窗口覆盖真实音频、静音、缩放、脉冲/边沿、裁剪补零；batch1/4暖机5次、计时20次；最多30分钟计算 | shape、位置、有效掩码与坐标一致，推导误差<=1个采样点（按约定），相同输入可复现，全部有限；报告峰值RAM/VRAM、每窗耗时、整曲估算。加载错误/坐标错/超预算即停止，不进入训练 |
| P1 | 100步接线；1首train过拟合诊断上限1000步；正式每配置每seed至多5000更新或60分钟，先到者停；batch4完整样本、AdamW lr1e-3、weight_decay1e-4、clip1起点 | 100步只算接线。train归一化MAE<=.05且活跃areaIoU>=.90才认为小样本可拟合；val MAE<=.10、areaIoU>=.80，门后静音/余槽误报中心<=1%，onset/offset中位绝对误差<=40 ms作为进入P2建议门槛；失败报告幅值/边界/塌缩，不认定骨干上限 |
| P2 | 固定P1配置，C1-local每配置每seed至多10000更新或90分钟；同batch/optimizer起点；关联温度逐项改变 | test transported每源areaIoU>=.70、归一化MAE<=.15；无歧义局部端点对应>=90%，相对index错配率减半；额外活动面积<=5%、来源数MAE<=.1、复制窗<=5%；三seed报告范围，不满足即明确失败 |

门槛是小固定音色课程的可审阅工程/研究标准，用户可调整；不是广泛泛化承诺。
P1/2每100更新检查val，最少1000更新；val主shape相对改善<1%持续10次检查才早停，
并记录实际更新、数据暴露与墙钟。预算耗尽称“未达门槛”，不能称收敛。
test只在每配置停止选定后评估，不据test调参。

首轮只跑一个frame序列头+RMS旁路：P0 .5h + P1三seed至多3h +
P2三seed至多4.5h，总计算上限8h。若获反馈执行对照，再增加一个池化对照（至多7.5h）；
其他loss/旁路/patch不是一次全网格，按失败证据另定预算。下载、渲染、实现时间另记。
这是墙钟硬预算，不是假设现有GPU已可用；P0若CPU速度不合适需重估再反馈。
缓存按中心窗数×token数×D×dtype字节估算并实测，建议磁盘上限10GB，OOM先减batch；
禁止悄悄改变输入宽度、换tiny或重采样精度仍比较同预算结果。

## 7. 报告指标与不能成立的证明

所有图回原音频时间轴，分别给train/val/test及三seed：
- raw A与transported A的full-scale MAE、活跃归一化MAE、areaIoU、面积偏差；
  单列静音与未用候选，门前/后、onset/offset误差（以相同审计门限产生边界）。
- 混音RMS oracle：C0将实测混音RMS放一个候选，余槽0，复用相同测量窗/坐标；
  同时列全零基线。C0 oracle即一源标签，故C0成功只能证明幅值接线/拟合，不能证明分离。
  P2另列“复制混音RMS到多候选”反例，监督源oracle仅作上界，不作模型输入。
- 多/漏报活动面积、每中心输出来源数误差。复制窗可先定义为两个活动候选曲线
  cosine>.99且二者面积均>参考显著活动面积5%；阈值敏感性及raw/transport分开报告。
- 整段PIT最佳/次佳代价差与并列率；真值形状本身并列与模型塌缩造成的并列分开。
- 单独对shape backward记录E/关联logits梯度范数与方向，检查matched active部分；
  只有非零梯度不够，还要量化错配后更新是否改善端点与shape。
- 每窗独立打乱候选后重新关联，映射回匿名编号比较输出/指标差；
  首锚编号允许整体置换，门槛建议MAE变化<1e-5（float32容差）。
  该不变性只证明不依赖槽索引，不证明身份正确。
- 无歧义端点真值对应、错误关联、拒配/歧义保留，匹配熵、空匹配率。
  端点覆盖是数据验收，须与正确对应分开；同形状可辨别前不强迫唯一答案。
- 内部区间与音频边缘单列成本/误差：真实证据长度、补零比例、预测中心偏差、
  RMS测量支持、未来上下文（中心窗约1秒延迟需求）、CPU/GPU/存储成本。

只靠事件两端覆盖、排列不变、低cycle或梯度存在均不能证明正确身份。
P2无歧义正确端点+整段shape才是核心，不以门控掩盖raw塌缩或候选复制。

## 8. 所需研究代码（尚未实现）

建议独立 `research/issue_51_ssast/`，按阶段实现：
1. `sources.json`/`checkpoint_manifest.json`：官方commit、下载URL、SHA256、依赖、
   shape/stride、加载key核验；锁定环境，官方timm==0.4.5与main Python3.12/torch>=2.5的
   兼容性须P0解决，不改main通用依赖，不执行官方一键训练脚本。
2. `features.py` + `probe.py`：显式fbank、未池化提取、位置核查、坐标/有效掩码、
   绝对RMS旁路、增益探针与缓存manifest。
3. `dataset_adapter.py`：仅调用现有数据/标签接口，按实际窗/中心重审共同证据；
   train-only尺度、完整样本批次、时间映射保留。
4. `head.py`、`local_association.py`、`shape_loss.py`：时序V头、
   首有效锚定/局部容量约束搬运、整段PIT与分项loss；无TimeCycle、无长期原型、无回环。
5. `train.py`/`evaluate.py`：显式启动、预算停止、seed、raw/transport/gate、
   梯度/打乱/复制/端点诊断；报告与机器证据分别置于docs/reports及evidence研究子目录。

后续必要测试只覆盖真实风险：token展平/时间映射、补零掩码、strict零E无效、
首锚A守恒、cycle=0主关联、同权重搬运、整段PIT、候选打乱和shape→E梯度。
本轮只写文档，不建立空实现或伪造测试通过。

## 9. 历史证据与本轮核查

[#46归档日志](https://github.com/guajun/anonymous-audio-tracks/issues/46#issuecomment-5977045906)，
固定提交 `05fc4de` 的
[C0](https://github.com/guajun/anonymous-audio-tracks/blob/05fc4de/docs/reports/issue-46-c0.md)、
[association](https://github.com/guajun/anonymous-audio-tracks/blob/05fc4de/docs/reports/issue-46-association.md)、
[local-context](https://github.com/guajun/anonymous-audio-tracks/blob/05fc4de/docs/reports/issue-46-local-context.md)、
[endpoint](https://github.com/guajun/anonymous-audio-tracks/blob/05fc4de/docs/reports/issue-46-endpoint.md)
与该提交evidence/*.json为既有数值依据；不在此复制资产。
委托上下文给出的旧v3 seed46/100步：L1三组A全零；Huber+IoU index ValIoU .426381/
Test .384871，soft无cycle .213178/.186358；E cosine约.99994，
soft多轨复制混合形状、35.46%额外活动、PIT并列。这里是历史上下文，不是本轮重跑结果，
也不是新SSAST基线。短预算不足以推断骨干能力上限。
Demucs预训练是4类别stem重建，旧提取点是冻结cross-transformer瓶颈，不是分类模型。

本轮实际完成：核实PR #50合入与main SHA、创建独立managed worktree、
读取规范与通用管线接口、阅读官方论文/固定版本源码并推导时间网格、
建立研究issue与仅文档draft PR。核查diff、文档相对链接和提交范围。
未验证：权重下载/加载、真实特征shape、成本、数据重新审计、任何训练/指标。
下一步是用户对checkpoint、阶段门槛/预算、头/匹配候选的草案反馈。
