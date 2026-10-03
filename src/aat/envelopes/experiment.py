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
from .losses import ShapeLossConfig, segment_shape_loss, area_iou_numpy
from .models import EnvelopeVectorHead
from .metrics import event_diagnostics


def build_cache(data_root: str | Path, out: str | Path, *, device="cuda:0", hop_seconds=0.04, batch_windows=4):
    root, out = Path(data_root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("cache output must be empty")
    out.mkdir(parents=True,exist_ok=True)
    index = load_json(root/"index.json")
    if index["stage"] != "C0":
        raise ValueError("this runner currently supports C0")
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
    meta={"version":"aat-envelope-cache-v1","stage":"C0","data_index_sha256":hashlib.sha256((root/"index.json").read_bytes()).hexdigest(),
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


@torch.no_grad()
def evaluate(model, samples, config, device, *, predictions_out=None):
    rows=[]
    model.eval()
    for sample in samples:
        _,p=_predict(model,sample,device)
        target=sample["target"].to(device)
        valid=torch.ones(len(target),dtype=torch.bool,device=device)
        loss,assignment=segment_shape_loss(p,target,valid,config)
        matched=p[:,assignment.assignments[0][0]].cpu().numpy()
        truth=target[:,0].cpu().numpy()
        unused=[k for k in range(p.shape[1]) if k!=assignment.assignments[0][0]]
        empty_curves=p[:,unused].cpu().numpy()
        diagnostics=event_diagnostics(matched,truth,sample["center_times"].numpy())
        if predictions_out is not None:
            predictions_out=Path(predictions_out)
            predictions_out.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(predictions_out/f"{sample['sample_id']}.npz",
                                center_times=sample["center_times"].numpy(),target=truth,
                                predicted=matched,unused=empty_curves,
                                matched_slot=np.array(assignment.assignments[0][0]))
        rows.append({"sample_id":sample["sample_id"],"loss":float(loss),"mae":float(np.abs(matched-truth).mean()),
                     "relative_l1":float(np.abs(matched-truth).sum()/max(1e-8,truth.sum())),
                     "area_iou":area_iou_numpy(matched,truth),
                     "empty_mean":float(p[:,unused].mean()) if unused else 0.0,
                     "extra_active_fraction":float((empty_curves>=diagnostics["rms_threshold"]).any(1).mean()),
                     "mean_extra_active_count":float((empty_curves>=diagnostics["rms_threshold"]).sum(1).mean()),
                     "events":diagnostics,
                     "assignment_ambiguous":not assignment.unique,
                     "mix_rms_baseline_mae":float(np.abs(sample["mix_rms"].numpy()-truth).mean())})
    return {"songs":rows,"mean_mae":float(np.mean([r["mae"] for r in rows])),
            "mean_area_iou":float(np.mean([r["area_iou"] for r in rows])),
            "mean_empty":float(np.mean([r["empty_mean"] for r in rows]))}


def train_combinations(cache: str | Path, out: str | Path, *, device="cuda:0", steps=100, seed=46, lr=3e-4,
                       families=("l1","huber","area_iou","huber_iou","multiscale"), segment_centers=40,
                       delta=0.05, iou_weight=0.1, empty_weight=1.0,
                       scales_seconds=(0.0,0.02,0.05,0.10),normalize_huber=True):
    cache,out=Path(cache),Path(out)
    if steps<1 or segment_centers<2 or lr<=0:
        raise ValueError("invalid training budget")
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
        if sample["target"].shape[1]!=1:
            raise ValueError("C0 requires exactly one source")
        samples[record["split"]].append(sample)
    if not all(samples.values()):
        raise ValueError("train, val, test splits required")
    git=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    result={"version":"aat-c0-loss-sweep-v1","result_kind":"frozen-pretrained-demucs-c0-envelope-experiment",
            "seed":seed,"steps":steps,"lr":lr,"segment_centers":segment_centers,"device":device,
            "git_commit":git,"torch":torch.__version__,"cache_identity_sha256":hashlib.sha256((cache/"index.json").read_bytes()).hexdigest(),
            "encoder":meta["encoder"],"data_policy":meta["data_policy"],
            "association":"C0 fixed segment slots; E discrimination/association not validated",
            "head_version":EnvelopeVectorHead.VERSION,
            "cycle_weight":0,"runs":[]}
    for family in families:
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        model=EnvelopeVectorHead(samples["train"][0]["features"].shape[1]).to(device)
        config=ShapeLossConfig(family=family,hop_seconds=meta["hop_seconds"],delta=delta,
                               iou_weight=iou_weight,empty_weight=empty_weight,scales_seconds=tuple(scales_seconds),
                               normalize_huber=normalize_huber)
        optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        initial=evaluate(model,samples["val"],config,device)
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
            target=sample["target"][offset:offset+n].to(device)
            loss,_=segment_shape_loss(p,target,torch.ones(n,dtype=torch.bool,device=device),config)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optim.step()
            logs.append({"step":step+1,"loss":float(loss.detach()),"sample_id":sample["sample_id"],"segment_start":offset})
            if (step+1)%25==0:
                print(f"{family} step={step+1} loss={float(loss.detach()):.6f}",flush=True)
        val=evaluate(model,samples["val"],config,device,predictions_out=out/f"{family}-predictions")
        # No selection/tuning on test; all predeclared loss families are evaluated.
        test=evaluate(model,samples["test"],config,device,predictions_out=out/f"{family}-predictions")
        run={"family":family,"config":asdict(config),"parameters":sum(p.numel() for p in model.parameters()),
             "initial_val":initial,"val":val,"test":test,"seconds":time.perf_counter()-start_time,
             "peak_cuda_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.startswith("cuda") else None}
        torch.save({"version":result["version"],"model":model.state_dict(),"config":asdict(config),
                    "head_version":EnvelopeVectorHead.VERSION,
                    "in_channels":samples["train"][0]["features"].shape[1],"encoder":meta["encoder"],
                    "seed":seed,"steps":steps},out/f"{family}.pt")
        dump_json(out/f"{family}.json",run)
        (out/f"{family}.jsonl").write_text('\n'.join(json.dumps(row) for row in logs)+'\n',encoding="utf-8")
        result["runs"].append(run)
        dump_json(out/"summary.json",result)
    return result
