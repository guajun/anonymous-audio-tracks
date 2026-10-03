"""Bounded C0 frozen-Demucs cache and loss-combination experiment runner."""
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
from .metrics import event_diagnostics
from .association import AssociationConfig,rollout,reference_cycle_mask,cycle_loss
from .telemetry import Telemetry


def build_cache(data_root: str | Path, out: str | Path, *, device="cuda:0", hop_seconds=0.04, batch_windows=4):
    root, out = Path(data_root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("cache output must be empty")
    out.mkdir(parents=True,exist_ok=True)
    index = load_json(root/"index.json")
    if index["stage"] not in ("C0","C1"):
        raise ValueError("this runner currently supports C0/C1")
    encoder = FrozenDemucsFeatures(device=device,cache_dir=out/"models")
    records, probes = [], []
    start=time.perf_counter()
    try:
        for entry in index["entries"]:
            directory=root/entry["directory"]
            labels=EnvelopeData.load(directory)
            wav=read_wav(directory/"mix.wav")
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
            width=round(2.0*wav.sample_rate)
            indices=np.floor(centers*wav.sample_rate+0.5).astype(np.int64)
            valid=np.array([0<=centered_window_bounds(int(i),width)[0] and centered_window_bounds(int(i),width)[1]<=wav.frames for i in indices])
            chosen,centers,indices=chosen[valid],centers[valid],indices[valid]
            if not len(chosen):
                raise ValueError("no full-context centers")
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
            payload={"version":"aat-envelope-cache-v1","features":torch.cat(features),
                     "relative_times":times-1.0,"target":torch.from_numpy(labels.rms[chosen]),
                     "mix_rms":torch.from_numpy(mix_rms).float(),"center_times":torch.from_numpy(centers),
                     "source_ids":list(labels.source_ids),"encoder":encoder.identity,
                     "hop_seconds":hop_seconds,"mix_sha256":entry["mix_sha256"],
                     "label_sha256":hashlib.sha256((directory/"envelope.npz").read_bytes()).hexdigest()}
            path=out/f"{entry['sample_id']}.pt"
            torch.save(payload,path)
            records.append({"sample_id":entry["sample_id"],"split":entry["split"],"file":path.name,
                            "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"shape":list(payload["features"].shape)})
            print(f"cached {entry['sample_id']} {payload['features'].shape}",flush=True)
    finally:
        encoder.close()
    meta={"version":"aat-envelope-cache-v1","stage":index["stage"],"data_index_sha256":hashlib.sha256((root/"index.json").read_bytes()).hexdigest(),
          "encoder":encoder.identity,"hop_seconds":hop_seconds,"context_seconds":2.0,
          "data_policy":index["split_policy"],"entries":records,
          "extraction_seconds":time.perf_counter()-start,"probes":probes,
          "peak_cuda_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.startswith("cuda") else None}
    dump_json(out/"index.json",meta)
    return meta


def _predict(model, sample, device, *, batch=32):
    directions,intensity=[],[]
    for start in range(0,len(sample["target"]),batch):
        features=sample["features"][start:start+batch].to(device)
        n=features.shape[0]
        e,p=model(features,sample["relative_times"].to(device).expand(n,-1),sample["mix_rms"][start:start+batch].to(device))
        directions.append(e)
        intensity.append(p)
    return torch.cat(directions),torch.cat(intensity)


def _candidate_permutation(e,p,seed):
    generator=torch.Generator().manual_seed(seed)
    indices=torch.stack([torch.randperm(p.shape[1],generator=generator) for _ in range(len(p))]).to(p.device)
    return e.gather(1,indices[...,None].expand_as(e)),p.gather(1,indices)


def _track(e,p,association,association_config):
    if association=="index":
        return p,None
    result=rollout(e,p,association_config)
    return result["tracked"],result


@torch.no_grad()
def evaluate(model, samples, config, device, *, predictions_out=None,association="index",
             association_config=AssociationConfig(),shuffle_seed=None,cycle_weight=0):
    rows=[]
    model.eval()
    for sample in samples:
        e,p=_predict(model,sample,device)
        if shuffle_seed is not None:
            e,p=_candidate_permutation(e,p,shuffle_seed)
        p,associated=_track(e,p,association,association_config)
        target=sample["target"].to(device)
        valid=torch.ones(len(target),dtype=torch.bool,device=device)
        parts,assignment=segment_shape_components(p,target,valid,config)
        cycle=p.sum()*0
        if associated is not None:
            cycle=cycle_loss(associated,reference_cycle_mask(target,assignment,p.shape[1]))
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
                                matched_slots=np.array(selected))
        rows.append({"sample_id":sample["sample_id"],"loss":float(loss),"mae":float(np.abs(matched-truth).mean()),
                     **{key:float(value) for key,value in parts.items()},"cycle":float(cycle),"weighted_cycle":float(cycle_weight*cycle),
                     "relative_l1":float(np.abs(matched-truth).sum()/max(1e-8,truth.sum())),
                     "area_iou":float(np.mean([area_iou_numpy(matched[:,s],truth[:,s]) for s in range(truth.shape[1])])),
                     "empty_mean":float(p[:,unused].mean()) if unused else 0.0,
                     "extra_active_fraction":float((empty_curves>=diagnostics["rms_threshold"]).any(1).mean()),
                     "mean_extra_active_count":float((empty_curves>=diagnostics["rms_threshold"]).sum(1).mean()),
                     "events":diagnostics,
                     "events_by_source":by_source,
                     "source_count_mae":float(np.abs((p.cpu().numpy()>=.002).sum(1)-(truth>=.002).sum(1)).mean()),
                     "assignment_ambiguous":not assignment.unique,
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
                       association_config=AssociationConfig(),shuffle_candidates=False,
                       tensorboard_root=None,eval_every=0):
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
    samples={split:[] for split in ("train","val","test")}
    for record in meta["entries"]:
        path=cache/record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            raise ValueError("feature cache digest mismatch")
        sample=torch.load(path,map_location="cpu",weights_only=True)
        if sample["version"]!=meta["version"] or sample["encoder"]!=meta["encoder"]:
            raise ValueError("cache identity mismatch")
        sample["sample_id"]=record["sample_id"]
        if not 1<=sample["target"].shape[1]<=slots:
            raise ValueError("source count exceeds candidate capacity")
        samples[record["split"]].append(sample)
    if not all(samples.values()):
        raise ValueError("train, val, test splits required")
    git=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    result={"version":"aat-c0-loss-sweep-v1","result_kind":"frozen-pretrained-demucs-c0-envelope-experiment",
            "seed":seed,"steps":steps,"lr":lr,"segment_centers":segment_centers,"device":device,"slots":slots,
            "git_commit":git,"torch":torch.__version__,"cache_identity_sha256":hashlib.sha256((cache/"index.json").read_bytes()).hexdigest(),
            "encoder":meta["encoder"],"data_policy":meta["data_policy"],
            "association":association,"stage":meta["stage"],"association_config":asdict(association_config),
            "shuffle_candidates":shuffle_candidates,
            "head_version":EnvelopeVectorHead.VERSION,
            "cycle_weight":cycle_weight,"runs":[]}
    for family in families:
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        shuffle_rng=np.random.default_rng(seed+460000)
        model=EnvelopeVectorHead(samples["train"][0]["features"].shape[1],slots=slots).to(device)
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
            candidates=[i for i in range(len(sample["target"])-n+1) if sample["target"][i:i+n].max()>0.002]
            if not candidates:
                raise ValueError("no active training segments")
            offset=int(rng.choice(candidates))
            features=sample["features"][offset:offset+n].to(device)
            e,p=model(features,sample["relative_times"].to(device).expand(n,-1),sample["mix_rms"][offset:offset+n].to(device))
            if shuffle_candidates:
                e,p=_candidate_permutation(e,p,int(shuffle_rng.integers(2**31)))
            tracked,associated=_track(e,p,association,association_config)
            target=sample["target"][offset:offset+n].to(device)
            parts,assignment=segment_shape_components(tracked,target,torch.ones(n,dtype=torch.bool,device=device),config)
            shape_total=sum(parts.values())
            cycle=shape_total*0
            if associated is not None:
                cycle=cycle_loss(associated,reference_cycle_mask(target,assignment,slots))
            loss=shape_total+cycle_weight*cycle
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optim.zero_grad(set_to_none=True)
            descriptor_gradient=None
            if association=="soft" and (step==0 or (step+1)%25==0):
                gradient=torch.autograd.grad(parts["shape"],e,retain_graph=True,allow_unused=True)[0]
                descriptor_gradient=float(gradient.norm()) if gradient is not None else 0.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optim.step()
            row={"step":step+1,"loss":float(loss.detach()),"sample_id":sample["sample_id"],"segment_start":offset,
                 **{key:float(value.detach()) for key,value in parts.items()},"cycle":float(cycle.detach()),
                 "weighted_cycle":float((cycle_weight*cycle).detach()),"shape_descriptor_grad_norm":descriptor_gradient}
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
                    "head_version":EnvelopeVectorHead.VERSION,
                    "slots":slots,
                    "association":association,"association_config":asdict(association_config),"cycle_weight":cycle_weight,
                    "stage":meta["stage"],"label_unit":"linear_rms_full_scale","cache_identity_sha256":result["cache_identity_sha256"],
                    "in_channels":samples["train"][0]["features"].shape[1],"encoder":meta["encoder"],
                    "seed":seed,"steps":steps},out/f"{family}.pt")
        dump_json(out/f"{family}.json",run)
        (out/f"{family}.jsonl").write_text('\n'.join(json.dumps(row) for row in logs)+'\n',encoding="utf-8")
        result["runs"].append(run)
        dump_json(out/"summary.json",result)
        telemetry.scalars('resources',{k:run[k] for k in ('parameters','seconds','peak_cuda_mib')},steps)
        telemetry.close()
    return result
