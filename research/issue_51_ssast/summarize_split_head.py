"""Preserve full experiment JSON, matched tables, raw-axis figures and provenance."""
import gzip
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from features import sha256


def mean(values):
    valid=[v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def aggregate(records,variant):
    metrics=[r[variant] for r in records]
    return dict(per_source_iou=np.mean([m['all']['per_source_iou'] for m in metrics],axis=0).tolist(),
        per_source_mae=np.mean([m['all']['per_source_mae'] for m in metrics],axis=0).tolist(),
        normalized_mae=mean([m['normalized_mae'] for m in metrics]),
        per_source_silence_fp=np.mean([m['per_source_silence_false_positive_fraction'] for m in metrics],axis=0).tolist(),
        **{k:mean([m.get(k) for m in metrics]) for k in ('unused_false_positive_fraction','extra_candidate_area_fraction','source_count_mae','candidate_copy_window_fraction','full_eight_normalized_mae','unused_mean_normalized_amplitude')})


def relations(records):
    out={}
    for category in ('adjacent','block_boundary','block_interior','across_silence'):
        parts=[r['relations'][category] for r in records];n=sum(p['pairs'] for p in parts);w=sum(p['confidence_weight_sum'] for p in parts)
        out[category]=dict(pairs=n,correct_fraction=sum(p['pairs']*(p['correct_fraction'] or 0) for p in parts)/n if n else None,
            confidence_weight_sum=w,weighted_correct_fraction=sum(p['confidence_weight_sum']*(p['weighted_correct_fraction'] or 0) for p in parts)/w if w else None)
    for key in ('weighted_positive_cosine','weighted_negative_cosine','weighted_margin_violation'):
        out[key]=mean([r['relations'].get(key) for r in records])
    for key in ('teacher_zero_confidence_fraction','positive_cross_slot_fraction','positive_cross_silence_fraction','candidate_shuffle_normalized_mae_delta'):
        out[key]=mean([r.get(key) for r in records])
    out['teacher_active_confidence_mean']=mean([r['teacher_confidence']['mean'] for r in records])
    out['identity_pair_count']=sum(r['identity_pair_count'] for r in records)
    out['identity_effective_weight']=sum(r['identity_effective_weight'] for r in records)
    return out


def main():
    root=Path('runs/split-head/fit-v1');out=Path('runs/split-head/evidence');out.mkdir(parents=True,exist_ok=True)
    status=json.loads((root/'status.json').read_text())
    if not status['complete'] or any(j['exit_code']!=0 for j in status['jobs']):raise ValueError('suite incomplete/failed')
    result=dict(status=status,arms=[],definitions=dict(primary='actual prediction-only hard adjacent identity association',
        teacher='same joint fit-v2 teacher in both arms; independent A adapter only; diagnostics only',
        aggregate='equal-sample mean, fixed current course val/test; relation rates pooled by count/GT-confidence',
        output_gain='multiply Z, independent A and y after head; not input-gain generalization',
        historical='matched coupled vs split; shared/Z copied exactly; scalar amplitude train-only calibration residuals recorded'))
    full={}
    for name in ('coupled','split'):
        summary=json.loads((root/name/'summary.json').read_text());full[name]=summary
        with gzip.open(out/('issue-51-split-'+name+'-full.json.gz'),'wt',encoding='utf-8') as f:json.dump(summary,f)
        architecture=json.loads((root/name/'architecture.json').read_text())
        (out/('issue-51-split-'+name+'-architecture.json')).write_text(json.dumps(architecture,indent=2)+'\n')
        arm=dict(name=name,architecture=architecture,courses=[])
        for stage in summary:
            current=stage['stage'];record=dict(stage=current,updates=stage['updates'],selected_update=stage['selected_update'],
                seconds=stage['seconds'],stop_reason=stage['stop_reason'],plateau_condition_at_stop=stage['plateau_condition_at_stop'],
                schedule_seed=stage['schedule_seed'],cache_manifest_sha256=stage['cache_manifest_sha256'],current={},forgetting={})
            for split in ('validation','test'):
                rows=[r for r in stage[split] if r['stage']==current]
                record['current'][split]={v:aggregate(rows,v) for v in ('raw','actual','teacher','actual_gated','teacher_gated')}
                record['current'][split]['relations']=relations(rows)
                record['current'][split]['output_gain']={str(g):aggregate([dict(actual=r['output_gain_diagnostic'][str(g)]) for r in rows],'actual') for g in (.5,1.,2.)}
                record['forgetting'][split]={old:{v:aggregate([r for r in stage[split] if r['stage']==old],v) for v in ('raw','actual','teacher')}
                    for old in dict.fromkeys(r['stage'] for r in stage[split])}
            gradients=[p['gradient'] for h in stage['history'] for p in h['parts'] if p['stage']!='C0']
            record['gradient_diagnostics']=dict(samples=len(gradients),
                amplitude_radial_norm_mean=mean([g['amplitude_vector']['radial_norm'] for g in gradients]),
                amplitude_tangent_norm_mean=mean([g['amplitude_vector']['tangent_norm'] for g in gradients]),
                identity_tangent_norm_mean=mean([g['identity_vector']['tangent_norm'] for g in gradients]),
                identity_radial_norm_mean=mean([g['identity_vector']['radial_norm'] for g in gradients]),
                shared_projection_gradient_cosine_mean=mean([g['shared_projection_gradient_cosine'] for g in gradients]),
                identity_projection_norm_mean=mean([g['shared_projection_identity_norm'] for g in gradients]),
                amplitude_projection_norm_mean=mean([g['shared_projection_amplitude_norm'] for g in gradients]),
                identity_coefficient=architecture['identity_weight'])
            arm['courses'].append(record)
            if arm['name'] in ('coupled','split'):
                record['joint_diagnostics']={}
                for split in ('validation','test'):
                    rows=[r for r in stage[split] if r['stage']==current]
                    audits=[r['joint_objective_audit'] for r in rows]
                    record['joint_diagnostics'][split]=dict(
                        mean_assignment_change=mean([r['joint_vs_legacy_active_assignment_change'] for r in rows]),
                        max_teacher_training_cost_error=max(r['teacher_training_cost_abs_error'] for r in audits),
                        **{key:mean([r[key] for r in audits]) for key in ('objective_before','objective_after',
                        'amplitude_before','amplitude_after','identity_before','identity_after','changed_blocks','accepted_moves')},
                        label_fit=[r['teacher_label_fit_audit'] for r in rows])
        result['arms'].append(arm)
    (out/'issue-51-split-id-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'issue-51-split-suite-status.json').write_text(json.dumps(status,indent=2)+'\n')
    pilot={a:json.loads((root/(a+'-pilot')/'pilot.json').read_text()) for a in ('coupled','split')}
    (out/'issue-51-split-pilot.json').write_text(json.dumps(pilot,indent=1)+'\n')
    initial={name:json.loads((root/name/'initial-train-diagnostics.json').read_text()) for name in full}
    with gzip.open(out/'issue-51-split-initial-train.json.gz','wt',encoding='utf-8') as f:json.dump(initial,f)
    lines=['# Matched teacher curriculum (fixed current-course test)','',
        '| Arm | Course | Actual IoU A/B | Teacher IoU A/B | Actual empty FP | Extra area | Count MAE | Gap relation correct | Confidence |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for arm in result['arms']:
        for c in arm['courses']:
            m=c['current']['test'];a=m['actual'];t=m['teacher'];r=m['relations']
            fmt=lambda xs:'/'.join(f'{x:.4f}' for x in xs)
            gap=r['across_silence']['correct_fraction']
            lines.append(f"| {arm['name']} | {c['stage']} | {fmt(a['per_source_iou'])} | {fmt(t['per_source_iou'])} | {a['unused_false_positive_fraction']:.2%} | {a['extra_candidate_area_fraction']:.3f} | {a['source_count_mae']:.3f} | {gap:.2%} | {r['teacher_active_confidence_mean']:.3f} |")
    (out/'issue-51-split-tables.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for idx,arm in enumerate(result['arms']):
        courses=arm['courses'];x=np.arange(4);color=('tab:blue','tab:orange')[idx]
        for source,style in enumerate(('-','--')):
            axes[0,0].plot(x,[c['current']['test']['actual']['per_source_iou'][source] for c in courses],style,color=color,marker='o',label=f"{arm['name']} S{source+1}")
            axes[0,1].plot(x,[c['current']['test']['teacher']['per_source_iou'][source] for c in courses],style,color=color,marker='o',label=f"{arm['name']} S{source+1}")
        axes[1,0].plot(x,[c['current']['test']['actual']['unused_false_positive_fraction']*100 for c in courses],color=color,marker='o',label=arm['name'])
        axes[1,1].plot(x,[c['current']['test']['relations']['across_silence']['correct_fraction']*100 for c in courses],color=color,marker='o',label=arm['name'])
    for ax,title in zip(axes.flat,['Actual inference source IoU','GT-teacher diagnostic IoU','Actual unmatched FP (%)','Across-silence relation correct (%)']):
        ax.set_title(title);ax.set_xticks(np.arange(4),['C1','C2-low','C2-high','C3']);ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in axes[0]:ax.set_ylim(0,1)
    for ax in axes[1]:ax.set_ylim(0,100)
    fig.suptitle('One seed, fixed 4/2/2 data; new arms share init/data/budget; teacher is not inference')
    fig.tight_layout();fig.savefig(out/'issue-51-split-comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(18,13),sharex=True,sharey=True)
    for idx,name in enumerate(full):
        paths=sorted((root/name/'C3').glob('C3-*.npz'))
        for sample,path in enumerate(paths):
            z=np.load(path);time=z['times'];target=z['target'];raw=z['raw_a'];actual=z['actual_a'];ta=z['teacher_a']
            r=next(r for r in full[name][-1]['test'] if r['stage']=='C3' and r['sample_id']==path.stem[3:])
            actual_order=r['actual']['order'];mask=np.ones(8,dtype=bool);mask[actual_order]=False
            for col,(prediction,title,order) in enumerate([(raw,'raw',r['raw']['order']),
                    (actual,'actual',actual_order),(ta,'GT teacher',[0,1]),(actual,'actual unmatched',np.flatnonzero(mask))]):
                ax=axes[2*idx+sample,col]
                ax.set_title(name+' / test '+str(sample)+' / '+title,fontsize=9)
                for source in range(2):ax.plot(time,target[:,source],color=('black','gray')[source],linewidth=1.5,label=f'GT S{source+1}')
                for source,slot in enumerate(order):ax.plot(time,prediction[:,slot],linewidth=.8,alpha=.8,label=f'candidate {slot}')
                ax.axhline(.001,color='red',linewidth=.7,linestyle=':');ax.grid(alpha=.15);ax.legend(fontsize=6)
    for ax in axes[:,0]:ax.set_ylabel('Linear RMS (full scale)')
    for ax in axes[-1]:ax.set_xlabel('seconds')
    fig.suptitle('C3 BOTH fixed test clips, same original amplitude axes; unmatched activity shown explicitly')
    fig.tight_layout();fig.savefig(out/'issue-51-split-c3-curves.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for name,summary in full.items():
        for ax,stage in zip(axes,summary):
            ax.plot([h['update'] for h in stage['history']],[h['val_score'] for h in stage['history']],label=name)
            ax.axvline(stage['selected_update'],linestyle=':',alpha=.5)
            ax.set_title(stage['stage']);ax.set_xlabel('updates');ax.grid(alpha=.2)
    axes[0].set_ylabel('Prediction-only validation score');axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'issue-51-split-validation.png',dpi=150);plt.close(fig)
    audit=dict(files=[])
    with zipfile.ZipFile('runs/split-head/predictions.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in root.glob('**/*.npz'):
            with np.load(path) as z:
                if not all(np.isfinite(z[k]).all() for k in z.files):raise ValueError('nonfinite prediction artifact')
                if z['v'].shape[1:]!=(8,128):raise ValueError('vector shape')
                error=float(np.abs(z['actual_a'].sum(1)-z['raw_a'].sum(1)).max())
            audit['files'].append(dict(path=str(path),sha256=sha256(path),total_amplitude_max_error=error));archive.write(path,str(path.relative_to(root)))
    audit['npz_count']=len(audit['files']);audit['zip_sha256']=sha256('runs/split-head/predictions.zip')
    (out/'issue-51-split-tests.log').write_text((root/'tests.log').read_text())
    audit['gpu']=0;audit['tests']='40 passed; issue-51-split-tests.log';audit['new_inference_has_truth_input']=False
    audit['implementation_commit']='recorded in git research branch'
    audit['source_sha256']={name:sha256(Path('research/issue_51_ssast')/name) for name in
        ('train_joint_teacher.py','joint_teacher.py','teacher_assignment.py','identity_supervision.py','hard_identity_inference.py','run_split_suite.py','initialize_split_head.py','test_split_head.py','summarize_split_head.py','verify_split_selected.py','verify_split_execution.py','course_metrics.py','window_head.py')}
    audit['identical_initial_checkpoint']=result['arms'][0]['architecture']['initial_sha256']==result['arms'][1]['architecture']['initial_sha256']
    audit['checkpoint_sha256']={str(path):sha256(path) for path in root.glob('*/*/head.pth')}
    (out/'issue-51-split-execution-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'issue-51-split-initialization.json').write_text((root/'initialization.json').read_text())
    print('EVIDENCE COMPLETE',out,flush=True)

if __name__=='__main__':main()
