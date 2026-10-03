#!/usr/bin/env python3
"""Collect measured endpoint experiments, including the v1 numerical-tail probe."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root,device):
    import torch
    from aat.envelopes.models import EnvelopeVectorHead
    from aat.envelopes.experiment import _predict
    from aat.envelopes.local_association import LocalAssociationConfig,rollout
    cache=load(root/'cache-c1-local-v1/index.json')
    sample=torch.load(root/'cache-c1-local-v1/c1-local-val-00.pt',map_location='cpu',weights_only=True)
    result={'version':'aat-endpoint-evidence-v3','date':'2026-10-04','design_issue':48,
            'cache_identity_sha256':digest(root/'cache-c1-local-v1/index.json'),
            'data_index_sha256':digest(root/'data-c1-local-v1/index.json'),
            'context_audits':cache['local_context_audits'],'groups':{},
            'comparability':'Within each six-run group: identical cache/seed/initial weights/sample-offset pairs. Versus prior ungated prototype runs: head gate, association, fragment PIT normalization, cycle pairs and trajectory reconstruction changed together.'}
    for version in ('v1','v2','v3'):
        rows=[];sequences=[]
        for family in ('l1','huber_iou'):
            for mode in ('index','soft0','softcycle'):
                name=f'c1-endpoint-{version}-{family}-{mode}-100';directory=root/name
                summary=load(directory/'summary.json')
                log=[json.loads(line) for line in (directory/f'{family}.jsonl').read_text().splitlines()]
                sequences.append([(x['sample_id'],x['segment_start']) for x in log])
                with np.load(directory/f'{family}-predictions/c1-local-val-00.npz') as curves:
                    example={key:curves[key].tolist() for key in curves.files}
                row={'name':name,'summary':summary,'training_log':log,'validation_example':example,
                     'checkpoint_sha256':digest(directory/f'{family}.pt'),'summary_sha256':digest(directory/'summary.json'),
                     'tensorboard_run':f'{name}/{family}'}
                link_path=directory/f'{family}-predictions/c1-local-val-00-links.json'
                if link_path.exists():
                    links=load(link_path)
                    row['link_file_sha256']=digest(link_path)
                    row['local_link_evidence']={k:v for k,v in links.items() if k!='links'}
                    row['local_link_evidence']['links']=[{k:v for k,v in link.items() if k not in ('forward','backward')} for link in links['links']]
                    distances=[float(sample['center_times'][link['frame']]-sample['center_times'][old])
                               for link in links['links'] for old,eligible in zip(link['endpoint_frames'],link['eligible']) if eligible]
                    row['local_link_evidence']['maximum_actual_endpoint_distance_seconds']=max(distances,default=None)
                    assert all(distance<links['maximum_endpoint_distance_seconds'] for distance in distances)
                if version=='v3':
                    checkpoint=torch.load(directory/f'{family}.pt',map_location='cpu',weights_only=True)
                    model=EnvelopeVectorHead(checkpoint['in_channels'],slots=checkpoint['slots'],magnitude_gate=checkpoint['magnitude_gate']).to(device)
                    model.load_state_dict(checkpoint['model']);model.eval()
                    with torch.no_grad():
                        e,a,pre=_predict(model,sample,device,return_raw=True)
                        valid=a>0;pair_valid=valid[:,:,None]&valid[:,None,:]&~torch.eye(a.shape[1],device=device,dtype=torch.bool)[None]
                        cosine=e@e.transpose(1,2)
                        row['output_probe']={'nonzero_E_norm_min':float(e.norm(dim=-1)[valid].min()) if valid.any() else None,
                                             'zero_output_E_max_abs':float(e[~valid].abs().max()) if (~valid).any() else None,
                                             'within_valid_candidates_cosine_mean':float(cosine[pair_valid].mean()) if pair_valid.any() else None,
                                             'head_gate_zero_fraction':float((~valid).float().mean()),
                                             'head_pregate_mean':float(pre.mean()),'head_postgate_mean':float(a.mean())}
                        if mode!='index':
                            tracked=rollout(e,a,LocalAssociationConfig(**checkpoint['association_config']),center_times=sample['center_times'],context_bounds=sample['context_bounds'])
                            np.testing.assert_allclose(tracked['tracked'].cpu().numpy(),np.array(example['tracked_candidates']),atol=1e-7,rtol=1e-5)
                            if family=='huber_iou' and mode=='soft0':
                                valid_frames=torch.nonzero(valid.any(1)).flatten().cpu().tolist()
                                first=valid_frames[0]
                                later=next(i for i in valid_frames if float(sample['center_times'][i]-sample['center_times'][first])>1.1)
                                indices=torch.tensor([first,later])
                                sparse=rollout(e[indices],a[indices],LocalAssociationConfig(**checkpoint['association_config']),
                                               center_times=sample['center_times'][indices],context_bounds=sample['context_bounds'][indices])
                                row['sparse_observation_break_probe']={
                                    'kind':'actual cached audio/head outputs with omitted intermediate observations; not a physical long-silence benchmark',
                                    'original_frame_indices':[first,later],'center_times':sample['center_times'][indices].tolist(),
                                    'actual_context_bounds':sample['context_bounds'][indices].tolist(),
                                    'candidate_A':a[indices].cpu().tolist(),'candidate_E':e[indices].cpu().tolist(),
                                    'fragment_ids':sparse['fragment_ids'].tolist(),'tracked':sparse['tracked'].cpu().tolist(),
                                    'direct_anchor_frames':sparse['direct_anchor_frames'],'cycle_pair_count':len(sparse['cycle_links'])}
                rows.append(row)
        fair={'identical_sample_offset_pairs':all(s==sequences[0] for s in sequences),
              'same_cache':len({r['summary']['cache_identity_sha256'] for r in rows})==1,
              'same_code':len({r['summary']['git_commit'] for r in rows})==1,
              'same_head_gate':len({r['summary']['magnitude_gate'] for r in rows})==1}
        if not all(fair.values()):raise ValueError('six-run fairness failed')
        result['groups'][version]={'fairness':fair,'runs':rows}
    # Explicit numerical boundary example: this is not an audio performance result.
    times=torch.arange(34,dtype=torch.float64)*.04
    p=torch.zeros(34,2);p[:,1]=.2;p[0,0]=p[15,0]=p[33,0]=.2
    e=torch.eye(2)[None].expand(34,-1,-1).clone();e[p==0]=0
    with torch.no_grad():boundary=rollout(e,p)
    result['boundary_probe']={'kind':'controlled numerical endpoint/zero/break verification; not synthesized audio',
                              'center_times':times.tolist(),'candidate_A':p.tolist(),'fragment_ids':boundary['fragment_ids'].tolist(),
                              'tracked':boundary['tracked'].tolist(),'expired_endpoints':boundary['expired_endpoints'],
                              'short_bridge_same_fragment':bool(boundary['fragment_ids'][0,0]==boundary['fragment_ids'][15,0]),
                              'second_short_bridge_same_fragment':bool(boundary['fragment_ids'][15,0]==boundary['fragment_ids'][33,0])}
    # 15->33 is .72s, so use an independent clearly out-of-range gap for break proof.
    long_p=torch.zeros(34,1);long_p[0]=long_p[-1]=.2
    with torch.no_grad():long_result=rollout(torch.ones(34,1,2),long_p)
    result['boundary_probe']['long_gap_example']={'center_times':times.tolist(),'candidate_A':long_p.tolist(),
        'fragment_ids':long_result['fragment_ids'].tolist(),'tracked':long_result['tracked'].tolist(),
        'different_fragment':bool(long_result['fragment_ids'][0,0]!=long_result['fragment_ids'][-1,0])}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    result=collect(args.root,args.device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    for version,group in result['groups'].items():
        print(version,'fairness',group['fairness'])
        for row in group['runs']:
            run=row['summary']['runs'][0];songs=run['val']['songs']
            mean=lambda key:float(np.mean([s[key] for s in songs if s.get(key) is not None])) if any(s.get(key) is not None for s in songs) else None
            print(row['name'],'val',run['val']['mean_area_iou'],'test',run['test']['mean_area_iou'],
                  'shuffle',run['val_shuffled']['mean_area_iou'],'missed',sum(s['events']['missed_events'] for s in songs),
                  'stitch',mean('short_gap_pit_stitch_fraction'),'extra',mean('extra_active_fraction'),
                  'mass',mean('transported_mass_ratio'),'zero',mean('gate_zero_fraction'),
                  'fragments',[s.get('fragment_count') for s in songs],flush=True)


if __name__=='__main__':main()
