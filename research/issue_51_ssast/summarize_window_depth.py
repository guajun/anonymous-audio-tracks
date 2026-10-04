"""Compact full-window depth evidence and publication-style static plots."""
import argparse
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_courses import compact


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit',type=Path,required=True)
    args=p.parse_args();stages=['C0','C1','C2-low','C2-high','C3']
    all_records=[];histories={};full_records={}
    for depth in (2,4,8):
        d=args.fit/f'depth{depth}'
        full=json.loads((d/'summary.json').read_text());full_records[depth]=full
        if [r['stage'] for r in full]!=stages:raise ValueError('incomplete course sequence')
        (args.fit/f'depth{depth}-full.json.gz').write_bytes(gzip.compress((d/'summary.json').read_bytes(),mtime=0))
        architecture=json.loads((d/'architecture.json').read_text())
        architecture['center_interpolated_local_receptive_tokens']=2+2*depth
        courses=[]
        for record in full:
            current=record['stage'];item=dict(stage=current,actual_overlap=record['actual_overlap'])
            for arm in ('raw','local'):
                if arm not in record:continue
                r=record[arm]
                item[arm]=dict(current=compact(r,current),updates=r['updates'],seconds=r['seconds'],
                    stop_reason=r['stop_reason'],budget=r['budget'],
                    selected_update=min(r['history'],key=lambda h:h['val_score'])['update'],
                    forgetting={old:compact(r,old) for old in stages[:stages.index(current)+1]},
                    raw_active_E_pair_cosine_mean=[x['raw_active_E_pair_cosine_mean']
                        for x in r['test'] if x['stage']==current])
                item[arm]['input_boundary_test']=[dict(sample_id=x['sample_id'],
                    input_context=x['input_context'],
                    variants={variant:x[variant]['input_boundary'] for variant in ('raw','transported','raw_gated','transported_gated')})
                    for x in r['test'] if x['stage']==current]
                histories[depth,current,arm]=r['history']
            courses.append(item)
        all_records.append(dict(depth=depth,architecture=architecture,courses=courses))
    sensitivity=json.loads((args.fit/'context-sensitivity.json').read_text())
    summary=dict(depths=all_records,context_sensitivity=sensitivity,
        original_C0_seed46_collection_failure_retained=True,
        historical_comparison_limit='window readout, numerical execution and update budgets changed; no isolated receptive-field causal claim',
        seed_count=1,unseen_timbre_generalization_claimed=False,C4_started=False)
    (args.fit/'compact.json').write_text(json.dumps(summary,indent=2)+'\n')
    tables=['# Full-window depth result tables','',
        'All rows use the selected fixed-val checkpoint. Test metrics average two current-course samples.',
        'Every head reads197 tokens; intermediate convolution support is5/9/17 tokens. One training seed.', '',
        '|Depth|Parameters|Stage|Arm|Test source IoU|Full-scale source MAE|Unused FP|Copy blocks|Count MAE|Correct endpoints|',
        '|---|---:|---|---|---|---|---:|---:|---:|---|']
    for record in all_records:
        for course in record['courses']:
            for arm in ('raw','local'):
                if arm not in course:continue
                r=course[arm]['current']['test'];m=r['transported']
                fmt=lambda values:'/'.join(f'{v:.4f}' for v in values)
                endpoint=fmt(r['endpoint_correct_fraction_per_source']) if 'endpoint_correct_fraction_per_source' in r else 'n/a'
                tables.append(f"|{record['depth']}|{record['architecture']['parameters']}|{course['stage']}|{arm}|{fmt(m['per_source_iou'])}|{fmt(m['per_source_mae'])}|{m['unused_false_positive_fraction']:.3%}|{m['candidate_copy_window_fraction']:.3%}|{m['source_count_mae']:.4f}|{endpoint}|")
    tables+=['','|Depth|Stage|Arm|Updates|Selected update|Stop|Total seconds|',
        '|---|---|---|---:|---:|---|---:|']
    for record in all_records:
        for course in record['courses']:
            for arm in ('raw','local'):
                if arm not in course:continue
                r=course[arm]
                tables.append(f"|{record['depth']}|{course['stage']}|{arm}|{r['updates']}|{r['selected_update']}|{r['stop_reason']}|{r['seconds']:.2f}|")
    tables+=['','Walltime includes training-time validation/final evaluation and concurrent resource contention; it is not isolated throughput.','']
    (args.fit/'tables.md').write_text('\n'.join(tables))
    colors={2:'#2379bd',4:'#d77820',8:'#3f9653'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for record in all_records:
        depth=record['depth']
        for arm,column in (('raw',0),('local',1)):
            cc=[r for r in record['courses'] if arm in r]
            names=[r['stage'] for r in cc]
            iou=[np.mean(r[arm]['current']['test']['transported']['per_source_iou']) for r in cc]
            count=[r[arm]['current']['test']['transported']['source_count_mae'] for r in cc]
            axes[0,column].plot(names,iou,'o-',color=colors[depth],label=f'{depth} layers')
            axes[1,column].plot(names,count,'o-',color=colors[depth],label=f'{depth} layers')
    for col,arm in enumerate(('Raw supervised line','Local transported line')):
        axes[0,col].set_title(arm);axes[0,col].set_ylim(0,1)
        axes[0,col].set_ylabel('Mean source area IoU');axes[0,col].legend()
        axes[1,col].set_ylabel('Source-count MAE');axes[1,col].set_xlabel('Current course test')
        axes[1,col].set_ylim(0,8)
    fig.savefig(args.fit/'depth-course-comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(18,7),constrained_layout=True)
    for col,stage in enumerate(stages):
        for row,arm in enumerate(('raw','local')):
            ax=axes[row,col];ax.set_title(f'{stage} / {arm}')
            for depth in (2,4,8):
                h=histories.get((depth,stage,arm))
                if h:ax.plot([r['update'] for r in h],[r['val_score'] for r in h],color=colors[depth],label=f'{depth} layers')
            ax.set_xlabel('Updates');ax.set_ylabel('All-seen val selection score')
    axes[0,0].legend();axes[1,0].text(.2,.5,'C0 local not trained',transform=axes[1,0].transAxes)
    fig.savefig(args.fit/'depth-validation-curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    for row,depth in enumerate((2,4,8)):
        r=full_records[depth][-1]
        prediction=next(p for p in r['local']['test'] if p['stage']=='C3')
        z=np.load(args.fit/f'depth{depth}'/'C3-local'/('C3-'+prediction['sample_id']+'.npz'))
        mask=(z['times']>=2)&(z['times']<=4)
        for source in (0,1):
            ax=axes[row,source];time=z['times'][mask]
            ax.plot(time,z['target'][mask,source],color='black',label='Reference')
            ax.plot(time,z['raw_a'][mask,prediction['raw']['order'][source]],color='#2379bd',label='Local checkpoint raw A')
            ax.plot(time,z['transported_a'][mask,prediction['transported']['order'][source]],color='#d77820',label='Transported A')
            ax.set_title(f'{depth} layers / source {source+1}')
            ax.set_xlabel('Original audio time (s)');ax.set_ylabel('Linear RMS')
    axes[0,0].legend();fig.savefig(args.fit/'depth-c3-curves.png',dpi=160)
    print('DEPTH EVIDENCE COMPLETE',flush=True)


if __name__=='__main__':main()
