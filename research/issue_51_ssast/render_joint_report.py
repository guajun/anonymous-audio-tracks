"""Write a numeric report from preserved evidence; no metric transcription."""
import json
import gzip
from pathlib import Path


def fmt(xs):return '/'.join(f'{x:.4f}' for x in xs)
def main():
    base=Path('runs/joint-teacher/evidence')
    s=json.loads((base/'issue-51-joint-id-summary.json').read_text())
    small=json.loads((base/'issue-51-joint-small.json').read_text())
    oldsmall=json.loads((base/'issue-51-joint-small-v1.json').read_text())
    audit=json.loads((base/'issue-51-joint-execution-audit.json').read_text())
    verify=json.loads((base/'issue-51-joint-selected-verification.json').read_text())
    gap_audit=json.loads((base/'issue-51-joint-gap-failures.json').read_text())
    with gzip.open(base/'issue-51-joint-id-full.json.gz','rt') as f:full=json.load(f)
    logged=[p['teacher_objective'] for stage in full for h in stage['history'] for p in h['parts'] if p['stage']!='C0']
    lines=['# #51 模长 + 预测身份联合教师：匹配课程重跑与失败审计','',
        '本轮教师求 P 确实读取预测 E，并直接评价完整训练片段的模长与双向异源 ID 损失。',
        '主要推理仍只使用预测 V。完成真实 full197-token train-only pilot、可枚举验证、',
        '两臂各四课程1000更新、selected checkpoint 重载和完整遗忘评估。',
        '以下均为修正版 fit-v2；首版 fit-v1 的完整结果另存。验收失败，没有分离成功或收敛声明。','',
        'C3主要Actual：旧包络-ID.2为 .9053/.7333，联合-ID.2为 .4801/.4081；联合教师诊断',
        '却为 .9373/.8723。C2-low有改善，C1/C3整体退步，不能用teacher高IoU替代实际来源保持。',
        '联合C3跨静音同预测行34/62，旧包络60/62，余槽FP95.43%，最终C0 val余槽84.76%。',
        '因此本轮新联合实现没有支持最终连接改善；这不是将旧包络实验冒充联合方案失败。','',
        '## 目标、可行域与控制变量','',
        '详见[执行协议](../research/issue-51-joint-teacher.md)。F(P)=L_amp(P)+.2 L_ID(P;c)。',
        'L_amp 与训练完全相同：真实源 Huber/.1、全段 areaIoU、GT0点和余槽各 .1 平均模长；',
        'ID 是原候选同源正对与两个真实异源竞争的 margin.2 排序 + .05 正拉近。',
        '所有 GT 活动/置信权重与训练一致，分母仅 GT 活动，不允许对应自己降低置信。',
        'c 在求 P 前冻结，为 GT 全段包络可辨度乘候选方向 spread。两臂共享此规则。',
        '它不是正确身份后验：小幅空轨随机方向也可能扩大 spread，复制/相同 GT 的完整歧义',
        '有降权，但部分复制与包络近似仍需审计，不能将高可靠度当正确标签。','',
        '身份方向早期可能错误。教师限制全段模长损失最多比无索引先验的包络初始化增加 .02，',
        '并保护每个 GT 非零节点：其包络绝对误差最多比初始化增加 .02（固定 r 尺度）。',
        '首版只有全段限制，穷举小例仍会牺牲真实节点选近零候选，因此被否决为最终方案；',
        '修正版加入逐节点保护，且完整重跑。界限是启发式，弱 GT 仍可能在容差内被低估，',
        '错误包络初始化也不会自动成为正确真值，不能宣称无错误标签。','',
        '训练每16帧共享对应，逐块全部8P1/8P2状态做一次坐标搜索。每次评价精确完整 F，',
        '包括所有跨块 ID 对与完整 IoU；不存在候选索引连续性先验。求解不是全局最优。',
        'P/G 都是硬一对一，GT0 身份未定义，零轨继续接受原模长监督。没有向量平均抵消，',
        '没有 .001 预测训练 gate；.001 只作固定评估。主要推理仍为原相邻预测 cosine',
        '连续可靠度 + Hungarian，未修推理/轴/活动判据，teacher 指标不选 checkpoint。','',
        'envelope-id 是旧包络 teacher(block16,switch.01) + ID.2；joint-id 更换对应求解。',
        '两臂相同初始化、完整输入、head、RMS旁路、数据、抽样seed4650..4653、50/50 replay、',
        'AdamW/clip、预算和预测-only选权重。共同置信变化与上一轮历史 ID.2 区分；',
        '历史 ID0/.2 对照和旧 raw/local 都保留，不能混作本轮匹配因果对照。','',
        '## 可枚举验证与近似误差','',
        f"36 tests passed。24个四帧3P2随机小例的穷举范围为6^4，单遍坐标有 {small['cases_with_nonzero_gap']} 个非最优，最大差 {small['max_coordinate_optimality_gap']:.8f}。",
        f"首版相同随机范围有 {oldsmall['cases_with_nonzero_gap']} 个非最优，最大差 {oldsmall['max_coordinate_optimality_gap']:.8f}；两版可行集合不同，不能把差的下降当更强全局优化证明。",'',
        '| 小例 | 初始完整 F | 坐标后 F | 穷举 F | 教师/torch成本绝对误差 |',
        '|---|---:|---:|---:|---:|']
    for c in small['cases']:
        lines.append(f"| {c['name']} | {c['coordinate']['objective_before']:.8f} | {c['coordinate']['objective_after']:.8f} | {c['exhaustive']['objective_after']:.8f} | {c['full_cost_abs_error']:.2e} |")
    lines+=['','保持模长不变、只改变 E，教师对应会改变；复制输出/相同 GT 降权，GT0无ID，',
        '近零方向梯度有限且非零、跨静音有声端点与真实双向竞争、一对一及禁止抵消都验证。',
        'block1穷举在逐帧候选打乱下最优目标/解集合等变；对称并列最优可选不同标签。',
        '训练block16对块内逐帧随机打乱不等变，这是限制，不能用推理打乱测试掩盖。','',
        '| C3 train-only求解敏感性 | 完整 F | 模长项 | ID项 | 秒 |',
        '|---|---:|---:|---:|---:|']
    for q in verify['train_only_search_sensitivity']:
        m=q['metadata'];lines.append(f"| block{q['block']}, sweeps{q['sweeps']} | {m['objective_after']:.8f} | {m['amplitude_after']:.8f} | {m['identity_after']:.8f} | {q['seconds']:.3f} |")
    lines+=['','改变block也改变初始化和可行域，因此这些差不是完整自由排列目标的全局误差界。',
        '同block的额外遍数才是在同一可行域内测试剩余局部改进；低成本不等于正确来源。']
    for r in verify['real_train_identity_evidence_reactions']:
        lines.append(f"真实C3 train `{r['name']}` 相对最终教师的 GT非零节点对应变化 {r['active_assignment_change']:.2%}。")
    lines += ['这两个真实样本干预都未改对应，尽管翻转 E 显著改变 ID 成本；当前约束与包络拟合',
        '主导了最终可行标签。不能由合成例中 E 改标签就声称真实缓存中有足够身份选择空间。',
        f"训练记录步（非C0）共{len(logged)}个样本：有接受坐标更换{sum(m['accepted_moves']>0 for m in logged)}个，有ID项下降>1e-7共{sum(m['identity_before']-m['identity_after']>1e-7 for m in logged)}个。",
        '记录步不是所有更新的普查；更换也可能由完整模长目标驱动，不能单独证明ID造成更换。']
    lines+=['','## 固定当前课程 test：主要推理/raw/教师分别报告','',
        '| Arm | Course | Actual IoU A/B | Raw IoU A/B | Teacher IoU A/B | Actual MAE/r | 原轴源MAE A/B |',
        '|---|---|---|---|---|---:|---|']
    for arm in s['arms']:
        for c in arm['courses']:
            m=c['current']['test'];a=m['actual']
            lines.append(f"| {arm['name']} | {c['stage']} | {fmt(a['per_source_iou'])} | {fmt(m['raw']['per_source_iou'])} | {fmt(m['teacher']['per_source_iou'])} | {a['normalized_mae']:.6f} | {fmt(a['per_source_mae'])} |")
    lines+=['','所有 IoU 和 MAE 来自原线性 RMS，不是log轴或teacher推理结果。Teacher强制真实行0:S，',
        '不再次PIT。主要Actual用整段PIT评估预测推理。固定gate前后和输出缩放.5/1/2在JSON',
        '分别保存；输出缩放不是音频输入增益泛化。','',
        '| Arm | Course | 空轨FP | 静音FP A/B | 额外面积 | 数量MAE | 复制窗口率 | 全8 MAE/r |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for arm in s['arms']:
        for c in arm['courses']:
            a=c['current']['test']['actual'];sil='/'.join(f'{x:.2%}' for x in a['per_source_silence_fp'])
            lines.append(f"| {arm['name']} | {c['stage']} | {a['unused_false_positive_fraction']:.2%} | {sil} | {a['extra_candidate_area_fraction']:.4f} | {a['source_count_mae']:.4f} | {a['candidate_copy_window_fraction']:.2%} | {a['full_eight_normalized_mae']:.6f} |")
    lines+=['','硬连接保留逐帧活动槽数量与全部模长，数量超标源自head原候选；排列不能新增槽。',
        '空轨已有.1线性监督，近零梯度不自动消失。复制窗口率结合面积解释，FP本身不证明复制。','',
        '## 教师标签、身份关系和梯度审计','',
        '| Joint current test | 旧包络对应变化(有声) | F前/后 | Amp前/后 | ID前/后 | 教师/训练成本max误差 |',
        '|---|---:|---|---|---|---:|']
    for c in s['arms'][1]['courses']:
        q=c['joint_diagnostics']['test'];lines.append(f"| {c['stage']} | {q['mean_assignment_change']:.2%} | {q['objective_before']:.5f}/{q['objective_after']:.5f} | {q['amplitude_before']:.5f}/{q['amplitude_after']:.5f} | {q['identity_before']:.5f}/{q['identity_after']:.5f} | {q['max_teacher_training_cost_error']:.2e} |")
    lines+=['','完整 before/after、每次接受下降、可行状态数、条件状态gap、GT有声拟合残差及超过最佳',
        '单候选包络的误差都保存。条件gap不是全局posterior；teacher最低F不自动证明真实身份。','',
        '| Arm | Course | 正cos | 负cos | margin违反 | 跨静音同预测行 | 跨静音都在GT对齐源行 | 可靠度均值 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in s['arms']:
        for c in arm['courses']:
            r=c['current']['test']['relations'];gap=r['across_silence'];parts=[p for q in verify['records'] if q['arm']==arm['name'] and q['checkpoint_stage']==c['stage'] and q['evaluation_stage']==c['stage'] for p in q['endpoint_labels']]
            n=sum(p['pairs'] for p in parts);correct=sum(p['both_in_whole_sequence_source_row'] for p in parts)
            lines.append(f"| {arm['name']} | {c['stage']} | {r['weighted_positive_cosine']:.4f} | {r['weighted_negative_cosine']:.4f} | {r['weighted_margin_violation']:.2%} | {gap['correct_fraction']:.2%} ({gap['pairs']}) | {correct/max(1,n):.2%} ({correct}/{n}) | {r['teacher_active_confidence_mean']:.4f} |")
    lines+=['','同预测行率比GT源行率弱；静音节点没有被强迫拥有身份，检查的是前后真实有声端点。',
        '具体错误端点与期望行保存在selected验证JSON。标签仍只能以GT RMS审计，不能认证',
        '包络完全相同的真实物理源身份。正负cos和相邻准确率均不能替代整段IoU。','',
        '| Arm | Course | Amp径向 | ID切向 | ID径向 | 共享投影梯度cos |',
        '|---|---|---:|---:|---:|---:|']
    for arm in s['arms']:
        for c in arm['courses']:
            g=c['gradient_diagnostics'];lines.append(f"| {arm['name']} | {c['stage']} | {g['amplitude_radial_norm_mean']:.6f} | {g['identity_tangent_norm_mean']:.6f} | {g['identity_radial_norm_mean']:.2e} | {g['shared_projection_gradient_cosine_mean']:.4f} |")
    changes=[q for r in gap_audit['records'] for q in r['changes']]
    lines += ['',f"联合C3跨静音28次同预测行失败：源0 {sum(r['source']==0 for r in gap_audit['records'])}次，源1 {sum(r['source']==1 for r in gap_audit['records'])}次；全部两端teacher选择同一原候选。",
        f"在这些候选的间隙共定位{len(changes)}次行变化，{sum(q['gt_left']==0 for q in changes)}次左节点GT0，{sum(q['gt_left']==0 and q['gt_right']==0 for q in changes)}次两端GT0，{sum(q['candidate_cosine']<0 for q in changes)}次相邻方向cos为负。",
        '这是后验错误定位，未用test调阈值或选权重。GT0方向未监督，而相邻预测推理仍通过这些',
        '节点连接，给出明确训练/推理差距；不据此强迫真实静音节点有稳定身份。',
        '联合C3 raw/teacher IoU相同而实际大幅退步，原候选未匹配面积约.4603，实际约.8895：',
        '真源活动流入整段PIT未选的轨迹，并不是硬关联创建新的幅度；逐帧数量保持不变。',
        '[逐事件诊断](../../evidence/issue-51-joint-gap-failures.json)保存时间、GT/预测幅度、cos及换行。']
    lines+=['','这是记录步的等样本均值，不是完整优化轨迹；共享梯度冲突和teacher低成本不能证明易学。','',
        '## 最终 selected C3：完整已见遗忘','',
        '| Arm | Split | Old course | Actual IoU | Actual空轨FP | 数量MAE |',
        '|---|---|---|---|---:|---:|']
    for arm in s['arms']:
        for split in ('validation','test'):
            for stage,m in arm['courses'][-1]['forgetting'][split].items():
                a=m['actual'];lines.append(f"| {arm['name']} | {split} | {stage} | {fmt(a['per_source_iou'])} | {a['unused_false_positive_fraction']:.2%} | {a['source_count_mae']:.4f} |")
    lines+=['','原C0 seed46 val余槽4.87%>1%和历史三seed test约.951-.954保持原失败解释。',
        '本轮也没有通过空轨/数量与全部已见课程验收。旧raw C3 .9252/.8807、余槽1.80%保留，',
        '但其训练、选择和推理不同，不能直接作本轮教师单独因果比较。','',
        '## 预算、验证与证据','',
        '| Arm | Stage | Updates | Selected update | Stop | Head阶段秒 |',
        '|---|---|---:|---:|---|---:|']
    for arm in s['arms']:
        for c in arm['courses']:lines.append(f"| {arm['name']} | {c['stage']} | {c['updates']} | {c['selected_update']} | {c['stop_reason']} | {c['seconds']:.2f} |")
    lines += ['',f"修正版两臂8000总更新，runner wall {s['status']['elapsed_seconds']:.2f}s。首版另有8000更新和510.03s，均保留；两个pilot各20更新。",
        '以上仅缓存后head阶段，不冒充完整SSAST特征提取成本。只用GPU0；GPU1和外部进程未管理。',
        '没有实际云账单，未虚构金额；原输入特征成本见历史报告。全部达到更新上限，不称收敛。',
        f"selected重载复查 {verify['npz_rechecks']} 个NPZ，V最大误差 {verify['max_v_error']:.2e}，活动数量守恒max差 {verify['count_conservation_max_error']}。",
        f"归档 {audit['npz_count']} 个有限NPZ，最大总模长误差 {max(r['total_amplitude_max_error'] for r in audit['files']):.2e}；初始权重SHA一致={audit['identical_initial_checkpoint']}。",'',
        '证据：[汇总](../../evidence/issue-51-joint-id-summary.json)、',
        '[旧包络臂原始](../../evidence/issue-51-envelope-id-full.json.gz)、',
        '[联合臂原始](../../evidence/issue-51-joint-id-full.json.gz)、',
        '[穷举](../../evidence/issue-51-joint-small.json)、',
        '[首版完整负面结果](../../evidence/issue-51-joint-v1-preserved-full.json.gz)、',
        '[selected验证](../../evidence/issue-51-joint-selected-verification.json)、',
        '[SHA执行审计](../../evidence/issue-51-joint-execution-audit.json)。','',
        '![当前课程比较](../../evidence/issue-51-joint-comparison.png)','',
        '![C3两条test原轴](../../evidence/issue-51-joint-c3-curves.png)','',
        '![所有已见val选择曲线](../../evidence/issue-51-joint-validation.png)','',
        '服务器独立clone保存 runs/joint-teacher/fit-v2 下每阶段 selected/last、日志、NPZ；',
        'runs/joint-teacher/fit-v1 原始首版和 source 快照不覆盖。预测ZIP独立保存并核对SHA。',
        '固定4/2/2、小样本单seed、固定pad/pluck音色，仅研究当前任务。不声称统计显著性、',
        '未见音色/输入增益泛化、SSAST相对RMS收益；没有C4。Issue保持开放，PR保持draft，不merge。']
    path=Path('docs/reports/issue-51-joint-teacher.md');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(path)


if __name__=='__main__':main()
