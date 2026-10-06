"""Readable evidence tables and original-axis figures from completed runs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def avg(rows,key):
    return float(np.mean([key(r) for r in rows]))


def compact(report,stage):
    answer={}
    for split in ('validation','test'):
        rows=[r for r in report[split] if r['stage']==stage]
        variants={}
        for variant in ('raw','transported','raw_gated','transported_gated'):
            rr=[r[variant] for r in rows]
            variants[variant]=dict(
                per_source_iou=np.mean([r['all']['per_source_iou'] for r in rr],axis=0).tolist(),
                per_source_mae=np.mean([r['all']['per_source_mae'] for r in rr],axis=0).tolist(),
                nonoverlap_per_source_iou=np.mean([r['nonoverlap']['per_source_iou'] for r in rr],axis=0).tolist(),
                overlap_per_source_iou=None if rr[0]['overlap']['per_source_iou'] is None else np.mean([r['overlap']['per_source_iou'] for r in rr],axis=0).tolist(),
                overlap_per_source_mae=None if rr[0]['overlap']['per_source_mae'] is None else np.mean([r['overlap']['per_source_mae'] for r in rr],axis=0).tolist(),
                normalized_mae=avg(rr,lambda r:r['normalized_mae']),
                second_source_missed_fraction=None if rr[0]['second_source_missed_fraction'] is None else avg(rr,lambda r:r['second_source_missed_fraction']),
                per_source_silence_false_positive_fraction=np.mean([r['per_source_silence_false_positive_fraction'] for r in rr],axis=0).tolist(),
                unused_false_positive_fraction=avg(rr,lambda r:r['unused_false_positive_fraction']),
                extra_candidate_area_fraction=avg(rr,lambda r:r['extra_candidate_area_fraction']),
                source_count_mae=avg(rr,lambda r:r['source_count_mae']),
                candidate_copy_window_fraction=avg(rr,lambda r:r.get('candidate_copy_window_fraction',0)),
                pit_tied_fraction=avg(rr,lambda r:r['pit_tied']))
        answer[split]=variants
        if 'endpoints' in rows[0]:
            answer[split]['endpoint_correct_fraction_per_source']=np.mean([
                [s['correct_fraction'] for s in r['endpoints']['sources']] for r in rows],axis=0).tolist()
            answer[split]['shuffle_mae_delta']=[r.get('candidate_shuffle_normalized_mae_delta') for r in rows]
            answer[split]['transport_mass_max_error']=[r['amplitude_transport_total_max_error'] for r in rows]
    return answer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fit',type=Path,required=True)
    args=p.parse_args()
    stages=['C1','C2-low','C2-high','C3']
    full=json.loads((args.fit/'summary.json').read_text())
    compact_results=[]
    forgetting={}
    for result in full:
        stage=result['stage']
        record=dict(stage=stage,actual_overlap=result['actual_overlap'],
            raw_updates=result['raw']['updates'],raw_stop=result['raw']['stop_reason'],
            raw_seconds=result['raw']['seconds'],local_updates=result['local']['updates'],
            local_stop=result['local']['stop_reason'],local_seconds=result['local']['seconds'],
            raw_line=compact(result['raw'],stage),local_line=compact(result['local'],stage),
            baselines=result['baselines'])
        compact_results.append(record)
        forgetting[stage]={}
        for old in ['C0']+stages[:stages.index(stage)+1]:
            forgetting[stage][old]=dict(raw_line=compact(result['raw'],old),
                local_line=compact(result['local'],old))
    (args.fit/'compact.json').write_text(json.dumps(dict(courses=compact_results,
        forgetting=forgetting,seed=46,C0_original_collection_gate_passed=False,
        C4_started=False),indent=2)+'\n')
    fig,axes=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
    for i,result in enumerate(full):
        stage=result['stage']
        record=next(r for r in result['local']['test'] if r['stage']==stage)
        z=np.load(args.fit/(stage+'-local')/(stage+'-'+record['sample_id']+'.npz'))
        order=record['transported']['order'];raw_order=record['raw']['order']
        for s in (0,1):
            ax=axes[i,s]
            select=(z['times']>=2)&(z['times']<=4)
            times=z['times'][select]
            ax.plot(times,z['target'][select,s],color='black',label='Reference source')
            ax.plot(times,z['raw_a'][select,raw_order[s]],color='#2379bd',label='Raw A',alpha=.9)
            ax.plot(times,z['transported_a'][select,order[s]],color='#d77820',label='Transported A',alpha=.9)
            ax.set_title(stage+' / source '+str(s+1))
            ax.set_ylabel('Linear RMS')
            ax.set_xlabel('Original audio time (s)')
    axes[0,0].legend()
    fig.savefig(args.fit/'course-curves.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for line,color in [('raw_line','#2379bd'),('local_line','#d77820')]:
        iou=[np.mean(r[line]['test']['transported']['per_source_iou']) for r in compact_results]
        count=[r[line]['test']['transported']['source_count_mae'] for r in compact_results]
        axes[0].plot(stages,iou,'o-',color=color,label=line)
        axes[1].plot(stages,count,'o-',color=color,label=line)
    axes[0].set_ylabel('Mean source area IoU');axes[0].set_ylim(0,1)
    axes[1].set_ylabel('Source-count MAE')
    axes[0].legend();axes[1].legend()
    fig.savefig(args.fit/'course-learning.png',dpi=160)


if __name__=='__main__':main()
