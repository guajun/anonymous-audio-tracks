"""Build the reviewable report from completed, independently verified evidence."""
import gzip
import json
from pathlib import Path

def main():
    out=Path('evidence');load=lambda name:json.loads((out/name).read_text())
    summary=load('issue-51-split-id-summary.json');init=load('issue-51-split-initialization.json')
    selected=load('issue-51-split-selected-verification.json');audit=load('issue-51-split-execution-audit.json')
    small=load('issue-51-split-small.json')
    f=lambda xs:'/'.join(f'{x:.4f}' for x in xs)
    lines=['# #51 响度/身份拆头：匹配课程与初始化限制','',
        '本轮完成耦合结构 vs 标量响度拆头的匹配联合教师+ID.2对照。旧纯包络教师、联合教师fit-v1/fit-v2与原C0失败均完整保留。以下主结果都是预测-only相邻Hungarian推理，teacher仅为诊断，不用于checkpoint选择。','',
        '## 输出参数化与固定变量','',
        '共享769→128投影、4Conv3/GELU及center98.25+full197 mean/std不变。耦合为A=norm(V)、E=normalize(V)，689,920参数；拆头为A=abs(Linear384→8)、Z=原Linear384→8×128、E=normalize(Z)，693,000参数，增加3,080（0.4464%）。所有身份仍128维。独立A直接进入幅度损失、联合教师、推理连续可靠度与硬排列；Z的norm只用于方向归一化，从未重新充当响度。','',
        '两臂均修正版joint-teacher fit-v2：F=L_amp+.2L_ID，完整片段目标、block16单遍坐标、全段amp上界+.02和逐GT非零节点误差保护+.02。冻结GT包络可辨度×direction spread可靠度不变；可靠度不是正确标签posterior。GT0身份未定义、余轨/GT0独立A线性监督、双向异源margin.2+.05正拉近、offset1/2/5/15/30/45、radius45均不变。无.001训练硬gate，无向量抵消。单遍坐标并非全局最优，原穷举3/24非最优、maxgap .007797证据仍保留。','',
        '完整197×768冻结缓存、固定线性r=.2591032087802887、4train/2val/2test、pad/pluck、C3重触发不变。seed4650..4653、50/50 replay、AdamW lr.001 decay.0001 clip1、每课1000更新、每50步val及7200秒安全budget不变。所有已见val等样本评分：MAE/r+.1(1-meanIoU)+.1unusedmeanA/r+.01countMAE；test和teacher均不选checkpoint。推理保持原相邻cosine/连续可靠度/Hungarian，A守恒。','',
        '## 公平初始化：已复制的部分与无法精确匹配的部分','',
        '共享层和128维Z输出层逐tensor完整复制历史C1-raw初始化，train上的Z最大差为0。新响度层只用C0/C1共8个训练缓存、以旧head A为目标，直接旧A ridge1e-6后1000次cached-feature Adam MSE；不读取val/test，不拟合GT，不更新共享/Z。耦合臂做同训练特征1000次readout零残差控制，保持已有参数。这是相同pass预算，优化工作与墙钟成本不同；不能把校准额外成本隐藏进正式课程。','',
        f"初始化耗时拆头{init['split_seconds']:.3f}s、耦合控制{init['coupled_seconds']:.3f}s。标量abs affine无法精确表示128维affine向量norm：平均幅度差、峰值差和活动数量差见下表。这些起点差异是实际混杂，不能将观察差异全部归因于梯度解耦。这里记录参数化而不是宣称精确同初始预测。",'',
        '| Train clip | A/r MAE | A/r max差 | 相对mean A | 活动数量MAE |',
        '|---|---:|---:|---:|---:|']
    for r in init['differences']:lines.append(f"| {r['sample_id']} | {r['amplitude_mae']:.6f} | {r['amplitude_max_error']:.6f} | {r['relative_mae']:.2%} | {r['count_mae']:.4f} |")
    lines+=['',f"原初始化SHA256 `{init['original_sha256']}`；耦合归档 `{init['coupled_sha256']}`；拆头 `{init['split_sha256']}`。归档序列化SHA不同，不代表复制tensor不同；独立执行审计逐tensor验证共同部分与原权重相等。用户SSAST权重SHA保持 `{audit['user_weights_sha256']}`，原来源未认证。",'',
        '初始train-only完整actual/raw/teacher诊断也归档。正式训练不是从pilot末权重继续，pilot每臂20更新后丢弃，分别重新加载已校准初始化。','',
        '## 每课程当前test：原线性轴','',
        '| Arm | Course | Actual IoU A/B | Raw IoU A/B | Teacher IoU A/B | Actual MAE/r | Actual原轴MAE A/B |',
        '|---|---|---|---|---|---:|---|']
    for arm in summary['arms']:
        for c in arm['courses']:
            m=c['current']['test'];a=m['actual']
            lines.append(f"| {arm['name']} | {c['stage']} | {f(a['per_source_iou'])} | {f(m['raw']['per_source_iou'])} | {f(m['teacher']['per_source_iou'])} | {a['normalized_mae']:.6f} | {f(a['per_source_mae'])} |")
    lines+=['','Teacher固定GT源行，不再PIT掩盖对应错误；Actual/Raw使用原整段PIT评价。完整val/test、gate前后与输出gain诊断见JSON，输出缩放不代表输入增益泛化。','',
        '| Arm | Course | 空轨FP | 静音FP A/B | 额外面积 | Count MAE | 复制窗口率 | 全8 MAE/r |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for arm in summary['arms']:
        for c in arm['courses']:
            a=c['current']['test']['actual']
            lines.append(f"| {arm['name']} | {c['stage']} | {a['unused_false_positive_fraction']:.2%} | {'/'.join(format(x,'.2%') for x in a['per_source_silence_fp'])} | {a['extra_candidate_area_fraction']:.4f} | {a['source_count_mae']:.4f} | {a['candidate_copy_window_fraction']:.2%} | {a['full_eight_normalized_mae']:.6f} |")
    lines+=['','## 跨静音与cos：两种端点标准','',
        '| Arm | Course | 同任意预测行 | 两端都在整段GT对齐源行 | 正cos | 负cos | margin违反 | 可靠度 |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for arm in summary['arms']:
        for c in arm['courses']:
            rr=[r for r in selected['records'] if r['arm']==arm['name'] and r['checkpoint_stage']==c['stage'] and r['evaluation_stage']==c['stage'] and r['split']=='test']
            e=[p for r in rr for p in r['endpoint_labels']];n=sum(p['pairs'] for p in e)
            same=sum(p['same_any_predicted_row'] for p in e);strict=sum(p['both_in_whole_sequence_source_row'] for p in e)
            r=c['current']['test']['relations']
            lines.append(f"| {arm['name']} | {c['stage']} | {same}/{n} ({same/n:.2%}) | {strict}/{n} ({strict/n:.2%}) | {r['weighted_positive_cosine']:.4f} | {r['weighted_negative_cosine']:.4f} | {r['weighted_margin_violation']:.2%} | {r['teacher_active_confidence_mean']:.4f} |")
    lines+=['','同预测行比严格GT源行标准弱。GT0未要求稳定身份；检查的是间隙前后真实有声端点。即使正cos高、负cos低，也不能替代整段来源保持。标签来自GT RMS拟合，包络相同的真实物理源仍不能认证。','',
        '## 梯度与教师目标审计','',
        '| Arm | Course | Amp对Z径向norm | ID切向norm | ID径向norm | 共享投影梯度cos |',
        '|---|---|---:|---:|---:|---:|']
    for arm in summary['arms']:
        for c in arm['courses']:
            g=c['gradient_diagnostics'];lines.append(f"| {arm['name']} | {c['stage']} | {g['amplitude_radial_norm_mean']:.6f} | {g['identity_tangent_norm_mean']:.6f} | {g['identity_radial_norm_mean']:.2e} | {g['shared_projection_gradient_cosine_mean']:.4f} |")
    lines+=['','拆头Amp对Z梯度为0说明输出路径独立，不说明A/共享幅度梯度为0。ID不更新独立A输出层，但两项仍通过共享特征冲突；共享梯度cos为记录步等样本均值，不是完整轨迹，也不能证明易学。','',
        '| Arm | Course | 对旧包络有声对应变化 | F before/after | Amp before/after | ID before/after | 教师/torch最大误差 |',
        '|---|---|---:|---|---|---|---:|']
    for arm in summary['arms']:
        for c in arm['courses']:
            d=c['joint_diagnostics']['test'];lines.append(f"| {arm['name']} | {c['stage']} | {d['mean_assignment_change']:.2%} | {d['objective_before']:.6f}/{d['objective_after']:.6f} | {d['amplitude_before']:.6f}/{d['amplitude_after']:.6f} | {d['identity_before']:.6f}/{d['identity_after']:.6f} | {d['max_teacher_training_cost_error']:.2e} |")
    lines+=['','真实C3 train-only额外搜索与方向干预（不用test改规则）：','',
        '| Arm | Block | Sweeps | F | Amp | ID | 秒 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in selected['train_only_search_sensitivity']:
        m=r['metadata'];lines.append(f"| {r['arm']} | {r['block']} | {r['sweeps']} | {m['objective_after']:.8f} | {m['amplitude_after']:.8f} | {m['identity_after']:.8f} | {r['seconds']:.3f} |")
    for r in selected['real_train_identity_evidence_reactions']:lines.append(f"\n{r['arm']} `{r['name']}` 有声对应变化 {r['active_assignment_change']:.2%}。")
    lines+=[f"\n独立A额外24个四帧3P2穷举：{small['cases_with_nonzero_gap']}个坐标非最优，maxgap {small['max_coordinate_optimality_gap']:.8f}。此有限样本零差不提供全局最优保证；原耦合随机例3/24非最优仍保留。"]
    lines+=['','block变化也改变初始化/可行集合，因此跨block成本差不是自由排列全局误差界。同block额外sweep才在相同可行集合审计剩余局部改善。不得用合成身份交换例夸大真实缓存的ID选择空间。','',
        '## 最终C3 selected：所有已见课程遗忘','',
        '| Arm | Split | Course | Actual IoU | 空轨FP | Count MAE |','|---|---|---|---|---:|---:|']
    for arm in summary['arms']:
        for split in ('validation','test'):
            for stage,m in arm['courses'][-1]['forgetting'][split].items():
                a=m['actual'];lines.append(f"| {arm['name']} | {split} | {stage} | {f(a['per_source_iou'])} | {a['unused_false_positive_fraction']:.2%} | {a['source_count_mae']:.4f} |")
    lines+=['','## 执行预算与可复查证据','',
        '| Arm | Course | Updates | Selected update | Stop | 秒 |','|---|---|---:|---:|---|---:|']
    for arm in summary['arms']:
        for c in arm['courses']:lines.append(f"| {arm['name']} | {c['stage']} | {c['updates']} | {c['selected_update']} | {c['stop_reason']} | {c['seconds']:.2f} |")
    lines+=['',f"正式课程共{sum(c['updates'] for arm in summary['arms'] for c in arm['courses'])}次更新。durable runner墙钟{summary['status']['elapsed_seconds']:.2f}s，含并发加载/完整评估及轮询，pilot单独记录；初始化另计。所有阶段达到更新上限，不称收敛。两臂并发，各阶段wall秒可重叠；仅缓存后head训练成本，不冒充SSAST特征提取成本；无云账单，不虚构金额。只使用GPU0，不管理GPU1及外部进程。",'',
        f"41 tests passed。重载selected全部已见val/test共{selected['npz_rechecks']} NPZ，Z最大差{selected['max_v_error']:.2e}、A最大差{selected['max_a_error']:.2e}、数量差{selected['count_conservation_max_error']}，教师/torch完整F最大差{selected['max_objective_error']:.2e}。NPZ的v字段沿用旧兼容命名：拆头存Z*r，raw_a才是独立A*r，不能从norm(v)重建响度。完整预测ZIP SHA256 `{audit['zip_sha256']}`；每文件SHA、执行源快照、共同初始化tensor、抽样schedule重建hash已保存。抽样hash是根据相同manifest/seed/代码重建，不冒充逐步现场轨迹。",'',
        '[具体协议](../research/issue-51-split-head.md) · [汇总](../../evidence/issue-51-split-id-summary.json) · [selected验证](../../evidence/issue-51-split-selected-verification.json) · [初始化](../../evidence/issue-51-split-initialization.json) · [执行SHA审计](../../evidence/issue-51-split-execution-audit.json) · [耦合完整结果](../../evidence/issue-51-split-coupled-full.json.gz) · [拆头完整结果](../../evidence/issue-51-split-split-full.json.gz)。', '',
        '![当前课程](../../evidence/issue-51-split-comparison.png)','',
        '![C3原轴两test](../../evidence/issue-51-split-c3-curves.png)','',
        '![val曲线](../../evidence/issue-51-split-validation.png)','',
        '## 限制与验收','',
        '本轮固定小样本单seed及固定音色，且标量响度初始化有非零残差。不声称统计显著性、未见音色/输入增益泛化、SSAST相对RMS收益、全局最优标签或分离成功。旧C0 seed46 val空轨4.87%>1%失败与joint fit-v2的最终C0 val84.76%失败完整保留。Issue开放，PR继续draft，不merge。']
    for arm in summary['arms']:
        final=arm['courses'][-1]['forgetting'];c0=final['validation']['C0']['actual']
        lines.append(f"\n本轮{arm['name']}最终C0 val空轨{c0['unused_false_positive_fraction']:.2%}，test IoU {f(final['test']['C0']['actual']['per_source_iou'])}。{'未通过1%空轨验收。' if c0['unused_false_positive_fraction']>.01 else '仅此项通过，不能代替全部课程来源保持验收。'}")
    c3=[arm['courses'][-1]['current']['test'] for arm in summary['arms']]
    conclusion=f"新匹配两臂C3 test：耦合Actual {f(c3[0]['actual']['per_source_iou'])}，拆头Actual {f(c3[1]['actual']['per_source_iou'])}；拆头Raw {f(c3[1]['raw']['per_source_iou'])}、Teacher {f(c3[1]['teacher']['per_source_iou'])}。拆头空轨FP {c3[1]['actual']['unused_false_positive_fraction']:.2%}，数量MAE {c3[1]['actual']['source_count_mae']:.4f}。"
    lines.insert(4,conclusion)
    lines.insert(5,'')
    lines.insert(6,'C3连接改善，但整体验收仍失败：拆头最终C0 val空轨57.03%远超1%，C0 test IoU从耦合.7714退到.5370；C2-low主源IoU从.9319退到.3780。单seed、固定小样本及非零初始化残差不允许普遍改善或统计显著性结论。')
    lines.insert(7,'')
    attempts=load('issue-51-split-preserved-attempts.json')
    lines += ['', '## 保留的初始化失败与额外成本', '', f"fit-v1 softplus的弱ridge校准A层权重norm4943，共享训练后A塌缩；完整两臂额外8000更新保留，runner {attempts['fit_v1_status']['elapsed_seconds']:.2f}s。不能用其低空轨FP宣称拆头成功，也不能将病态初始化失败归因于所有拆头结构。", '', 'fit-v2只有训练集更强正则校准与两臂各20次pilot，未运行正式课程；初始相对幅度误差约32%–67%，因此最终选用train匹配更好的abs标量非负参数化。此选择依据为train-only数值条件/初始化匹配，未使用val/test指标选择最终参数化。早期正式fit-v1提前启动是执行缺陷，额外成本明确保留。', '', f"最终fit-v3初始化A层权重norm {init['amplitude_weight_norm']:.4f}。abs在精确零处取零次梯度（原向量norm在精确零处同样没有径向恢复梯度），非零近零A有直接梯度，不增加训练gate。两臂pilot各20更新，runner {summary['pilot_status']['elapsed_seconds']:.2f}s，独立于正式runner；每job墙钟包括加载/评估与至多10秒轮询误差。", '', '[首版完整失败](../../evidence/issue-51-split-v1-preserved-full.json.gz) · [初始化探索记录](../../evidence/issue-51-split-preserved-attempts.json) · [标量参数化审计](../../evidence/issue-51-split-parameterization.json)。']
    repair=load('issue-51-split-initialization-repair.json')
    rec=repair['recomputed_initialization']
    lines += ['', f"执行审计曾因路径变量遮蔽将审计JSON误写到split-initial.pth。训练/selected/112个NPZ均未改变；修正路径命名和写后SHA保护后，按同一确定性初始化恢复，耦合/拆头初始化SHA均与训练前逐字节一致。原初始化记录保留，恢复额外缓存校准每臂1000pass，拆头{rec['split_seconds']:.3f}s、耦合控制{rec['coupled_seconds']:.3f}s，不计作新的正式训练。见[修复记录](../../evidence/issue-51-split-initialization-repair.json)。"]
    Path('docs/reports/issue-51-split-head.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
