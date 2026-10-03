#!/usr/bin/env python3
"""Collect measured C1-local evidence and read-only descriptor/memory probes.

Run after the six predeclared runs; does not modify data, checkpoints or models.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root,device):
    import torch
    from aat.envelopes.models import EnvelopeVectorHead
    from aat.envelopes.experiment import _predict
    from aat.envelopes.association import AssociationConfig,rollout
    data=load(root/'data-c1-local-v1/index.json')
    cache=load(root/'cache-c1-local-v1/index.json')
    sample=torch.load(root/'cache-c1-local-v1/c1-local-val-00.pt',map_location='cpu',weights_only=True)
    report={'version':'aat-c1-local-evidence-v1','date':'2026-10-04','data_index':data,
            'data_index_sha256':digest(root/'data-c1-local-v1/index.json'),
            'cache_index_sha256':digest(root/'cache-c1-local-v1/index.json'),
            'cache':{k:v for k,v in cache.items() if k!='probes'},'encoder_probe':cache['probes'][0],
            'historical_comparison':'Old C1 includes cross-disjoint-context reconnection beyond local matchability. Different data; do not rank old/new loss performance as a controlled ablation.',
            'runs':[]}
    with np.load(root/'data-c1-local-v1/c1-local-val-00/envelope.npz') as labels:
        report['reference_example']={k:labels[k].tolist() for k in ('center_times','rms')}
    report['window_example']={'center_times':sample['center_times'].tolist(),
                              'input_bounds_seconds':sample['context_bounds'].tolist(),
                              'complete_event_counts':sample['context_event_counts'].tolist()}
    sequences=[]
    for family in ('l1','huber_iou'):
        for mode in ('index','soft0','softcycle'):
            name=f'c1-local-v1-{family}-{mode}-100'
            directory=root/name
            summary=load(directory/'summary.json')
            logs=[json.loads(row) for row in (directory/f'{family}.jsonl').read_text().splitlines()]
            sequences.append([(row['sample_id'],row['segment_start']) for row in logs])
            checkpoint=torch.load(directory/f'{family}.pt',map_location='cpu',weights_only=True)
            model=EnvelopeVectorHead(checkpoint['in_channels'],slots=checkpoint['slots']).to(device)
            model.load_state_dict(checkpoint['model']);model.eval()
            with torch.no_grad():
                e,a=_predict(model,sample,device)
                associated=rollout(e,a,AssociationConfig(**checkpoint['association_config'])) if mode!='index' else None
                slots=e.shape[1]
                off_diagonal=~torch.eye(slots,dtype=torch.bool,device=device)
                within=(e@e.transpose(1,2))[:,off_diagonal]
                adjacent=(e[1:]*e[:-1]).sum(-1)
                probe={'candidate_cosine_mean':float(within.mean()),'candidate_cosine_min':float(within.min()),
                       'adjacent_40ms_same_channel_cosine_mean':float(adjacent.mean()),
                       'adjacent_input_overlap_seconds':1.96,
                       'interpretation':'Candidate channel similarity, not source identity accuracy. Adjacent inputs overlap; stored seeds need not overlap later inputs.'}
                if associated is not None:
                    memory=associated['prototypes']
                    seed_cos=(memory*e[0][None]).sum(-1)
                    probe.update(memory_seed_cosine_mean=float(seed_cos.mean()),memory_seed_cosine_min=float(seed_cos.min()),
                                 memory_seed_cosine_final_mean=float(seed_cos[-1].mean()),
                                 memory_current_same_channel_cosine_mean=float((memory*e).sum(-1).mean()),
                                 birth_mass_mean=float(associated['birth_mass'].mean()),birth_mass_max=float(associated['birth_mass'].max()),
                                 frames_with_any_birth_over_01=int((associated['birth_mass'].max(1).values>.01).sum()),
                                 null_mass_mean=float(associated['null_mass'].mean()),entropy_mean=float(associated['entropy'].mean()),
                                 seed_to_final_input_overlap_seconds=0.,seed_to_final_center_gap_seconds=float(sample['center_times'][-1]-sample['center_times'][0]))
            with np.load(directory/f'{family}-predictions/c1-local-val-00.npz') as curves:
                example={k:curves[k].tolist() for k in curves.files}
            report['runs'].append({'name':name,'summary':summary,'training_log':logs,
                                  'summary_sha256':digest(directory/'summary.json'),
                                  'checkpoint_sha256':digest(directory/f'{family}.pt'),
                                  'validation_example':example,'descriptor_probe':probe,
                                  'tensorboard_run':f'{name}/{family}'})
    report['fair_sampling']={'identical_100_sample_offset_pairs':all(sequence==sequences[0] for sequence in sequences),
                             'seed':46,'steps':100,'segment_centers':180,'same_cache':len({r['summary']['cache_identity_sha256'] for r in report['runs']})==1,
                             'same_git_commit':len({r['summary']['git_commit'] for r in report['runs']})==1}
    if not all(report['fair_sampling'][k] for k in ('identical_100_sample_offset_pairs','same_cache','same_git_commit')):
        raise ValueError('fair comparison contract failed')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    result=collect(args.root,args.device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    for row in result['runs']:
        run=row['summary']['runs'][0];songs=run['val']['songs']
        print(row['name'],'Val',run['val']['mean_area_iou'],'Test',run['test']['mean_area_iou'],
              'shuffled',run['val_shuffled']['mean_area_iou'],'missed',sum(s['events']['missed_events'] for s in songs),
              'PIT', [s['pit_optimal_count'] for s in songs], 'raw',np.mean([s['raw_A_mean'] for s in songs]),
              'transported',np.mean([s['tracked_A_mean'] for s in songs]),'probe',row['descriptor_probe'],flush=True)


if __name__=='__main__':main()
