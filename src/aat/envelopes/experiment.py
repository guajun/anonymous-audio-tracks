"""Bounded C0/C1 frozen-Demucs shape and association experiment runner."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import subprocess
import time
import numpy as np
import torch

from aat.contracts.jsonio import dump_json, load_json
from aat.labels.wav import read_wav
from aat.labels.energy import mean_square_at_samples
from aat.windowing import centered_window_bounds
from .demucs import FrozenDemucsFeatures
from .labels import EnvelopeData
from .losses import ShapeLossConfig, segment_shape_components, area_iou_numpy
from .models import EnvelopeVectorHead
from .metrics import event_diagnostics,stitch_diagnostics
from .association import AssociationConfig,rollout,reference_cycle_mask,cycle_loss
from .telemetry import Telemetry
from .local_context import audit_local_context,CONSTRAINT_VERSION,CONTEXT_SECONDS
from .local_association import LocalAssociationConfig,rollout as local_rollout,cycle_loss as local_cycle_loss,VERSION as LOCAL_VERSION


def build_cache(data_root: str | Path, out: str | Path, *, device="cuda:0", hop_seconds=0.04, batch_windows=4):
    root, out = Path(data_root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("cache output must be empty")
    out.mkdir(parents=True,exist_ok=True)
    index = load_json(root/"index.json")
    if index["stage"] not in ("C0","C1","C1-local"):
        raise ValueError("this runner currently supports C0/C1")
    encoder = FrozenDemucsFeatures(device=device,cache_dir=out/"models")
    records, probes = [], []
    local_reports=[]
    cache_version='aat-envelope-cache-local-v2' if index['stage']=='C1-local' else 'aat-envelope-cache-v1'
    start=time.perf_counter()
    try:
        for entry in index["entries"]:
            directory=root/entry["directory"]
            labels=EnvelopeData.load(directory)
            wav=read_wav(directory/"mix.wav")
            origin=load_json(directory/'manifest.json').get('track_start_seconds',0.)
            if wav.sample_rate != encoder.model.samplerate:
                raise ValueError("C0 requires native Demucs sample rate")
            for name,digest in labels.input_sha256.items():
                if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError("cached label stem mismatch")
            if hashlib.sha256((directory/"mix.wav").read_bytes()).hexdigest()!=entry["mix_sha256"]:
                raise ValueError("mix digest mismatch")
            stride=round(hop_seconds/labels.hop_seconds)
            if stride<1 or abs(stride*labels.hop_seconds-hop_seconds)>1e-9:
                raise ValueError("cache hop must be a positive integer multiple of label hop")
            chosen=np.arange(0,len(labels.center_times),stride)
            centers=labels.center_times[chosen]
            width=round(CONTEXT_SECONDS*wav.sample_rate)
            indices=np.floor((centers-origin)*wav.sample_rate+0.5).astype(np.int64)
            valid=np.array([0<=centered_window_bounds(int(i),width)[0] and centered_window_bounds(int(i),width)[1]<=wav.frames for i in indices])
            valid &= labels.valid[chosen]
            if index['stage']=='C1-local':
                if index.get('constraint_version')!=CONSTRAINT_VERSION:
                    raise ValueError('unsupported local context contract')
                valid &= (centers>=entry['scoring_start_seconds']-1e-9)&(centers<=entry['scoring_end_seconds']+1e-9)
            chosen,centers,indices=chosen[valid],centers[valid],indices[valid]
            if not len(chosen):
                raise ValueError("no full-context centers")
            coverage_arrays=None
            if index['stage']=='C1-local':
                audited,coverage_arrays=audit_local_context(labels,centers,audio_frames=wav.frames,origin_seconds=origin)
                if not audited['passed']:
                    raise ValueError(f'actual cache windows violate local evidence contract: {audited}')
                local_reports.append({'sample_id':entry['sample_id'],'split':entry['split'],**audited})
            source_audio=wav.samples
            if source_audio.shape[1]==1:
                source_audio=np.repeat(source_audio,2,axis=1)
            if source_audio.shape[1]!=2:
                raise ValueError("mono or stereo required")
            features=[]
            for offset in range(0,len(indices),batch_windows):
                windows=[]
                for i in indices[offset:offset+batch_windows]:
                    left,right=centered_window_bounds(int(i),width)
                    windows.append(source_audio[left:right].T)
                encoded,times,probe=encoder.encode(torch.from_numpy(np.stack(windows)).float())
                features.append(encoded)
                probes.append({"sample_id":entry["sample_id"],**probe})
            mix_rms=np.sqrt(mean_square_at_samples(source_audio,wav.sample_rate,labels.energy_window_seconds,indices))
            payload={"version":cache_version,"features":torch.cat(features),
                     "relative_times":times-1.0,"target":torch.from_numpy(labels.rms[chosen]),
                     "mix_rms":torch.from_numpy(mix_rms).float(),"center_times":torch.from_numpy(centers),
                     "source_ids":list(labels.source_ids),"encoder":encoder.identity,
                     "hop_seconds":hop_seconds,"mix_sha256":entry["mix_sha256"],
                     "label_sha256":hashlib.sha256((directory/"envelope.npz").read_bytes()).hexdigest()}
            if coverage_arrays is not None:
                payload.update(context_constraint=CONSTRAINT_VERSION,
                               context_bounds=torch.from_numpy(coverage_arrays['input_bounds_seconds']),
                               context_event_counts=torch.from_numpy(coverage_arrays['complete_event_counts']),
                               context_guard_seconds=audited['edge_guard_seconds'])
            path=out/f"{entry['sample_id']}.pt"
            torch.save(payload,path)
            records.append({"sample_id":entry["sample_id"],"split":entry["split"],"file":path.name,
                            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"shape":list(payload["features"].shape)})
            print(f"cached {entry['sample_id']} {payload['features'].shape}",flush=True)
    finally:
        encoder.close()
    meta={"version":cache_version,"stage":index["stage"],"data_index_sha256":hashlib.sha256((root/"index.json").read_bytes()).hexdigest(),
          "encoder":encoder.identity,"hop_seconds":hop_seconds,"context_seconds":2.0,
          "data_policy":index["split_policy"],"entries":records,
          "extraction_seconds":time.perf_counter()-start,"probes":probes,
          "peak_cuda_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.startswith("cuda") else None}
    if local_reports:
        meta.update(context_constraint=CONSTRAINT_VERSION,local_context_audits=local_reports)
    dump_json(out/"index.json",meta)
    return meta


def _predict(model, sample, device, *, batch=32,return_raw=False):
    directions,intensity,pregate=[],[],[]
    for start in range(0,len(sample["target"]),batch):
        features=sample["features"][start:start+batch].to(device)
        n=features.shape[0]
        e,p,raw=model(features,sample["relative_times"].to(device).expand(n,-1),sample["mix_rms"][start:start+batch].to(device),return_raw=True)
        directions.append(e)
        intensity.append(p)
        pregate.append(raw)
    result=(torch.cat(directions),torch.cat(intensity))
    return (*result,torch.cat(pregate)) if return_raw else result


def _candidate_permutation(e,p,seed):
    generator=torch.Generator().manual_seed(seed)
    indices=torch.stack([torch.randperm(p.shape[1],generator=generator) for _ in range(len(p))]).to(p.device)
    return e.gather(1,indices[...,None].expand_as(e)),p.gather(1,indices)


def _track(e,p,association,association_config,sample=None):
    if association=="index":
        return p,None
    if isinstance(association_config,LocalAssociationConfig):
        result=local_rollout(e,p,association_config,center_times=None if sample is None else sample['center_times'],
                            context_bounds=None if sample is None else sample.get('context_bounds'))
    else:result=rollout(e,p,association_config)
    return result["tracked"],result


def _cycle(result,target,assignment,slots):
    if result is not None and result.get('version')==LOCAL_VERSION:
        return local_cycle_loss(result,target,assignment)
    mask=target.new_zeros((len(target)-1,slots))
    value=target.sum()*0
    if result is not None:
        mask=reference_cycle_mask(target,assignment,slots)
        value=cycle_loss(result,mask)
    return value,float(mask.mean()) if mask.numel() else 0.


def gate_diagnostics(raw,selected):
    return {'pregate_A_mean':float(raw.detach().mean()),'pregate_A_max':float(raw.detach().max()),
            'gate_zero_fraction':float((selected.detach()==0).float().mean()),
            'gate_all_zero_frames':int((selected.detach()==0).all(1).sum())}


def fragment_diagnostics(result):
    if result is None or result.get('version')!=LOCAL_VERSION:return {}
    return {'fragment_count':int((result['tracked'].detach()>0).any(0).sum()),
            'fragment_columns':result['tracked'].shape[1],'direct_anchor_frames':len(result['direct_anchor_frames']),
            'expired_endpoints':sum(result['expired_endpoints']),
            'local_link_pairs':sum(int(link['eligible'].sum()) for link in result['cycle_links']),
            'transport_pregate_sum_mean':float(result['pregate_lane_activity'].detach().sum(1).mean()),
            'transport_gate_removed_sum_mean':float((result['pregate_lane_activity'].detach()-result['lane_activity'].detach()).sum(1).mean()),
            'transport_gate_zero_fraction':float((result['lane_activity'].detach()==0).float().mean())}


def eligible_offsets(target,n,*,require_recurrence=False):
    """C1 spans three complete threshold events, rather than merely being long."""
    activity=(target.cpu().numpy()>=.002).any(1)
    edges=np.diff(np.r_[False,activity,False].astype(np.int8))
    starts=np.flatnonzero(edges==1)
    ends=np.flatnonzero(edges==-1)-1
    return [i for i in range(len(target)-n+1) if activity[i:i+n].any() and
            (not require_recurrence or ((starts>=i)&(ends<i+n)).sum()>=3)]


def activity_transport_diagnostics(raw,tracked,target):
    active=(target>=.002).any(1)
    values={'raw_A_mean':float(raw.detach().mean()),'raw_A_max':float(raw.detach().max()),
            'tracked_A_mean':float(tracked.detach().mean()),'tracked_A_max':float(tracked.detach().max()),
            'raw_A_sum_mean':float(raw.detach().sum(1).mean()),
            'tracked_A_sum_mean':float(tracked.detach().sum(1).mean())}
    for role,mask in (('active',active),('silent',~active)):
        values[f'raw_A_{role}_mean']=float(raw[mask].detach().mean()) if mask.any() else None
        values[f'tracked_A_{role}_mean']=float(tracked[mask].detach().mean()) if mask.any() else None
    total=float(raw.detach().sum())
    values['transported_mass_ratio']=float(tracked.detach().sum())/total if total>0 else None
    return values


@torch.no_grad()
def evaluate(model, samples, config, device, *, predictions_out=None,association="index",
             association_config=AssociationConfig(),shuffle_seed=None,cycle_weight=0):
    rows=[]
    model.eval()
    for sample in samples:
        e,p,pregate=_predict(model,sample,device,return_raw=True)
        if shuffle_seed is not None:
            e,p=_candidate_permutation(e,p,shuffle_seed)
        raw=p
        p,associated=_track(e,p,association,association_config,sample)
        target=sample["target"].to(device)
        valid=torch.ones(len(target),dtype=torch.bool,device=device)
        local=associated is not None and associated.get('version')==LOCAL_VERSION
        parts,assignment=segment_shape_components(p,target,valid,config,fragment_capacity=raw.shape[1] if local else None)
        cycle,cycle_fraction=_cycle(associated,target,assignment,p.shape[1])
        loss=sum(parts.values())+cycle_weight*cycle
        selected=list(assignment.assignments[0])
        matched=p[:,selected].cpu().numpy()
        truth=target.cpu().numpy()
        unused=[k for k in range(p.shape[1]) if k not in selected]
        empty_curves=p[:,unused].cpu().numpy()
        by_source=[{"source_id":source_id,**event_diagnostics(matched[:,s],truth[:,s],sample["center_times"].numpy())}
                   for s,source_id in enumerate(sample["source_ids"])]
        diagnostics={"rms_threshold":by_source[0]["rms_threshold"]}
        for key in ("matched_events","missed_events","extra_events"):
            diagnostics[key]=sum(s[key] for s in by_source)
        for key in ("onset_mae_seconds","offset_mae_seconds","onset_signed_seconds","offset_signed_seconds","frame_f1"):
            values=[s[key] for s in by_source if s[key] is not None]
            diagnostics[key]=float(np.mean(values)) if values else None
        if predictions_out is not None:
            predictions_out=Path(predictions_out)
            predictions_out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(predictions_out/f"{sample['sample_id']}.npz",
                                center_times=sample["center_times"].numpy(),target=truth,
                                predicted=matched,unused=empty_curves,
                                raw_candidates=raw.cpu().numpy(),tracked_candidates=p.cpu().numpy(),
                                pregate_candidates=pregate.cpu().numpy(),
                                fragment_ids=associated['fragment_ids'] if local else np.empty((0,0),dtype=int),
                                pregate_lane_activity=associated['pregate_lane_activity'].cpu().numpy() if local else raw.cpu().numpy(),
                                matched_slots=np.array(selected))
            if local:
                links=associated['cycle_links']
                dump_json(predictions_out/f"{sample['sample_id']}-links.json",{
                    'version':LOCAL_VERSION,'direct_anchor_frames':associated['direct_anchor_frames'],
                    'expired_endpoints':associated['expired_endpoints'],
                    'maximum_endpoint_distance_seconds':associated['maximum_endpoint_distance_seconds'],
                    'links':[{'frame':l['frame'],'endpoint_frames':l['endpoint_frames'].tolist(),
                              'fragment_ids':l['fragment_ids'].tolist(),'eligible':l['eligible'].cpu().tolist(),
                              'forward':l['forward'].cpu().tolist(),'backward':l['backward'].cpu().tolist()} for l in links]})
        rows.append({"sample_id":sample["sample_id"],"loss":float(loss),"mae":float(np.abs(matched-truth).mean()),
                     **{key:float(value) for key,value in parts.items()},"cycle":float(cycle),"weighted_cycle":float(cycle_weight*cycle),
                     "cycle_eligible_fraction":cycle_fraction,
                     "relative_l1":float(np.abs(matched-truth).sum()/max(1e-8,truth.sum())),
                     "area_iou":float(np.mean([area_iou_numpy(matched[:,s],truth[:,s]) for s in range(truth.shape[1])])),
                     "empty_mean":float(p[:,unused].mean()) if unused else 0.0,
                     "extra_active_fraction":float((empty_curves>=diagnostics["rms_threshold"]).any(1).mean()),
                     "mean_extra_active_count":float((empty_curves>=diagnostics["rms_threshold"]).sum(1).mean()),
                     "events":diagnostics,
                     "events_by_source":by_source,
                     "source_count_mae":float(np.abs((p.cpu().numpy()>=.002).sum(1)-(truth>=.002).sum(1)).mean()),
                     "assignment_ambiguous":not assignment.unique,
                     "pit_optimal_count":assignment.num_optimal,
                     **activity_transport_diagnostics(raw,p,target),**gate_diagnostics(pregate,raw),**fragment_diagnostics(associated),
                     **stitch_diagnostics(p.cpu().numpy(),truth,sample['center_times'].numpy(),assignment,
                                          maximum_endpoint_distance=association_config.maximum_endpoint_distance if local else .91),
                     "mix_rms_baseline_mae":float(np.abs(sample["mix_rms"].numpy()-truth[:,0]).mean()) if truth.shape[1]==1 else None})
        if associated is not None:
            rows[-1].update(association_entropy=float(associated["entropy"].mean()),
                            association_null_mass=float(associated["null_mass"].mean()),
                            association_capacity_residual=float(associated["capacity_residual"].max()))
    return {"songs":rows,**{f'mean_{key}':float(np.mean([r[key] for r in rows])) for key in ('loss','shape','silence','null','cycle','weighted_cycle')},
            "mean_mae":float(np.mean([r["mae"] for r in rows])),
            "mean_area_iou":float(np.mean([r["area_iou"] for r in rows])),
            "mean_empty":float(np.mean([r["empty_mean"] for r in rows]))}


def train_combinations(cache: str | Path, out: str | Path, *, device="cuda:0", steps=100, seed=46, lr=3e-4,
                       families=("l1","huber","area_iou","huber_iou","multiscale"), segment_centers=40,
                       delta=0.05, iou_weight=0.1, empty_weight=1.0,
                       scales_seconds=(0.0,0.02,0.05,0.10),normalize_huber=True,slots=8,
                       null_weight=None,association="index",cycle_weight=0.0,
                       association_config=None,shuffle_candidates=False,
                       tensorboard_root=None,eval_every=0,require_recurrence=None,
                       association_backend='local',magnitude_gate=.002):
    cache,out=Path(cache),Path(out)
    if steps<1 or segment_centers<2 or lr<=0 or not 1<=slots<=8:
        raise ValueError("invalid training budget")
    if association not in ("index","soft") or cycle_weight<0 or not np.isfinite(cycle_weight) or eval_every<0:
        raise ValueError("invalid association/cycle/evaluation settings")
    if association=="index" and cycle_weight:
        raise ValueError("cycle requires soft association; cycle=0 still enables soft association")
    if out.exists() and any(out.iterdir()):
        raise ValueError("experiment output must be empty")
    out.mkdir(parents=True,exist_ok=True)
    meta=load_json(cache/"index.json")
    if association_backend not in ('local','legacy'):raise ValueError('invalid association backend')
    if association_config is None:
        association_config=LocalAssociationConfig(hop_seconds=meta['hop_seconds'],context_seconds=meta['context_seconds']) if association_backend=='local' else AssociationConfig()
    if isinstance(association_config,LocalAssociationConfig)!=(association_backend=='local'):
        raise ValueError('association configuration/backend mismatch')
    if require_recurrence is None:
        require_recurrence=meta["stage"] in ("C1","C1-local")
    samples={split:[] for split in ("train","val","test")}
    for record in meta["entries"]:
        path=cache/record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            raise ValueError("feature cache digest mismatch")
        sample=torch.load(path,map_location="cpu",weights_only=True)
        if sample["version"]!=meta["version"] or sample["encoder"]!=meta["encoder"]:
            raise ValueError("cache identity mismatch")
        if meta['stage']=='C1-local' and (sample.get('context_constraint')!=CONSTRAINT_VERSION or sample['context_event_counts'].min()<2):
            raise ValueError('local context evidence missing from feature cache')
        sample["sample_id"]=record["sample_id"]
        if not 1<=sample["target"].shape[1]<=slots:
            raise ValueError("source count exceeds candidate capacity")
        samples[record["split"]].append(sample)
    if not all(samples.values()):
        raise ValueError("train, val, test splits required")
    git=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    result={"version":"aat-envelope-local-fragments-sweep-v3" if association_backend=='local' else "aat-envelope-association-sweep-v2","result_kind":f"frozen-pretrained-demucs-{meta['stage'].lower()}-envelope-experiment",
            "seed":seed,"steps":steps,"lr":lr,"segment_centers":segment_centers,"device":device,"slots":slots,
            "git_commit":git,"torch":torch.__version__,"cache_identity_sha256":hashlib.sha256((cache/"index.json").read_bytes()).hexdigest(),
            "encoder":meta["encoder"],"data_policy":meta["data_policy"],
            "association":association,"stage":meta["stage"],"association_config":asdict(association_config),
            "shuffle_candidates":shuffle_candidates,
            "require_recurrence":require_recurrence,
            "head_version":'aat-envelope-vector-gated-v3' if magnitude_gate else EnvelopeVectorHead.VERSION,
            "association_backend":association_backend,"association_version":LOCAL_VERSION if association_backend=='local' else 'aat-persistent-prototype-v1',
            "magnitude_gate":magnitude_gate,"gate_training_gradient":"identity straight-through surrogate" if magnitude_gate else 'ungated',
            "cycle_weight":cycle_weight,"runs":[]}
    if meta.get('context_constraint'):
        result.update(context_constraint=meta['context_constraint'],context_seconds=meta['context_seconds'])
    for family in families:
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        shuffle_rng=np.random.default_rng(seed+460000)
        model=EnvelopeVectorHead(samples["train"][0]["features"].shape[1],slots=slots,magnitude_gate=magnitude_gate).to(device)
        config=ShapeLossConfig(family=family,hop_seconds=meta["hop_seconds"],delta=delta,
                               iou_weight=iou_weight,empty_weight=empty_weight,scales_seconds=tuple(scales_seconds),
                               normalize_huber=normalize_huber,null_weight=null_weight)
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        evaluation_settings={"association":association,"association_config":association_config,"cycle_weight":cycle_weight}
        telemetry=Telemetry(tensorboard_root,f"{out.name}/{family}",{**{k:v for k,v in result.items() if k!='runs'},"loss_config":asdict(config),"walltime":"actual measurement time"})
        initial=evaluate(model,samples["val"],config,device,**evaluation_settings)
        telemetry.evaluation('val',initial,0)
        validation_history=[{"step":0,"metrics":initial}]
        logs=[]
        start_time=time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(steps):
            model.train()
            sample=samples["train"][int(rng.integers(len(samples["train"])))]
            n=min(segment_centers,len(sample["target"]))
            candidates=eligible_offsets(sample["target"],n,require_recurrence=require_recurrence)
            if not candidates:
                raise ValueError("no eligible training segments; C1 requires three full activity events; increase segment-centers")
            offset=int(rng.choice(candidates))
            features=sample["features"][offset:offset+n].to(device)
            e,p,pregate=model(features,sample["relative_times"].to(device).expand(n,-1),sample["mix_rms"][offset:offset+n].to(device),return_raw=True)
            if shuffle_candidates:
                e,p=_candidate_permutation(e,p,int(shuffle_rng.integers(2**31)))
            context={'center_times':sample['center_times'][offset:offset+n]}
            if 'context_bounds' in sample:context['context_bounds']=sample['context_bounds'][offset:offset+n]
            tracked,associated=_track(e,p,association,association_config,context)
            target=sample["target"][offset:offset+n].to(device)
            local=associated is not None and associated.get('version')==LOCAL_VERSION
            parts,assignment=segment_shape_components(tracked,target,torch.ones(n,dtype=torch.bool,device=device),config,fragment_capacity=slots if local else None)
            shape_total=sum(parts.values())
            cycle,cycle_fraction=_cycle(associated,target,assignment,tracked.shape[1])
            loss=shape_total+cycle_weight*cycle
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optim.zero_grad(set_to_none=True)
            descriptor_gradient=None
            if association=="soft" and (step==0 or (step+1)%25==0):
                gradient=torch.autograd.grad(parts["shape"],e,retain_graph=True,allow_unused=True)[0]
                descriptor_gradient=float(gradient.norm()) if gradient is not None else 0.0
            amplitude_gradient=None
            if step==0 or (step+1)%25==0:
                gate_gradient=torch.autograd.grad(shape_total,pregate,retain_graph=True,allow_unused=True)[0]
                if gate_gradient is not None:
                    amplitude_gradient=float(gate_gradient[pregate.detach()<magnitude_gate].norm())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optim.step()
            row={"step":step+1,"loss":float(loss.detach()),"sample_id":sample["sample_id"],"segment_start":offset,
                 **{key:float(value.detach()) for key,value in parts.items()},"cycle":float(cycle.detach()),
                 "weighted_cycle":float((cycle_weight*cycle).detach()),"shape_descriptor_grad_norm":descriptor_gradient}
            row['gate_closed_amplitude_grad_norm']=amplitude_gradient
            row.update(cycle_eligible_fraction=cycle_fraction,pit_ambiguous=float(not assignment.unique))
            row.update(activity_transport_diagnostics(p,tracked,target),pit_optimal_count=assignment.num_optimal)
            row.update(gate_diagnostics(pregate,p),**fragment_diagnostics(associated))
            if associated is not None:
                row.update(association_entropy=float(associated["entropy"].mean().detach()),
                           association_null_mass=float(associated["null_mass"].mean().detach()),
                           association_birth_mass=float(associated["birth_mass"].mean().detach()),
                           association_capacity_residual=float(associated["capacity_residual"].max().detach()))
            logs.append(row)
            telemetry.scalars('train',row,step+1)
            if eval_every and (step+1)%eval_every==0 and step+1<steps:
                measured=evaluate(model,samples["val"],config,device,**evaluation_settings)
                validation_history.append({"step":step+1,"metrics":measured})
                telemetry.evaluation('val',measured,step+1)
            if (step+1)%25==0:
                print(f"{family} step={step+1} loss={float(loss.detach()):.6f}",flush=True)
        val=evaluate(model,samples["val"],config,device,predictions_out=out/f"{family}-predictions",**evaluation_settings)
        # No selection/tuning on test; all predeclared loss families are evaluated.
        test=evaluate(model,samples["test"],config,device,predictions_out=out/f"{family}-predictions",**evaluation_settings)
        shuffled=evaluate(model,samples["val"],config,device,shuffle_seed=seed+460001,**evaluation_settings)
        telemetry.evaluation('val',val,steps)
        validation_history.append({"step":steps,"metrics":val})
        telemetry.evaluation('test',test,steps)
        telemetry.evaluation('val_shuffled',shuffled,steps)
        telemetry.curves(out/f"{family}-predictions",steps)
        run={"family":family,"config":asdict(config),"parameters":sum(p.numel() for p in model.parameters()),
             "initial_val":initial,"val":val,"test":test,"val_shuffled":shuffled,"validation_history":validation_history,"seconds":time.perf_counter()-start_time,
             "peak_cuda_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.startswith("cuda") else None}
        torch.save({"version":result["version"],"model":model.state_dict(),"config":asdict(config),
                    "head_version":model.version,"magnitude_gate":magnitude_gate,"association_backend":association_backend,
                    "association_version":result['association_version'],
                    "slots":slots,
                    "association":association,"association_config":asdict(association_config),"cycle_weight":cycle_weight,
                    "stage":meta["stage"],"label_unit":"linear_rms_full_scale","cache_identity_sha256":result["cache_identity_sha256"],
                    "require_recurrence":require_recurrence,
                    "context_constraint":meta.get('context_constraint'),
                    "in_channels":samples["train"][0]["features"].shape[1],"encoder":meta["encoder"],
                    "seed":seed,"steps":steps},out/f"{family}.pt")
        dump_json(out/f"{family}.json",run)
        (out/f"{family}.jsonl").write_text('\n'.join(json.dumps(row) for row in logs)+'\n',encoding="utf-8")
        result["runs"].append(run)
        dump_json(out/"summary.json",result)
        telemetry.scalars('resources',{k:run[k] for k in ('parameters','seconds','peak_cuda_mib')},steps)
        telemetry.close()
    return result
