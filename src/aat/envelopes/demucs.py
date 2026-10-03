"""Frozen official HTDemucs feature probe using its unmodified eval forward.

The first research baseline intentionally executes the decoder as well: hooks
avoid accidentally changing normalization, padding, or transformer behavior.
This is a reference extraction path, not an encoder-only performance claim.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
import torch
from torch.nn import functional as F


class FrozenDemucsFeatures:
    def __init__(self, *, device="cuda:0", cache_dir: str | Path, model_name="htdemucs"):
        from demucs.pretrained import get_model
        torch.hub.set_dir(str(Path(cache_dir)/"torch-hub"))
        loaded = get_model(model_name)
        models = getattr(loaded,"models",[loaded])
        if len(models) != 1:
            raise ValueError("first experiment requires one model, not an ensemble")
        self.model = models[0].eval().to(device)
        self.model.requires_grad_(False)
        if self.model.crosstransformer is None:
            raise ValueError("HTDemucs cross-transformer required")
        self.device = device
        self.captured = None
        self.handle = self.model.crosstransformer.register_forward_hook(self._capture)
        weights = sorted((Path(cache_dir)/"torch-hub"/"checkpoints").glob("*"))
        self.identity = {"backend":"demucs", "model":model_name,"package":"demucs==4.0.1",
                         "extraction":"official-full-forward.cross-transformer-output",
                         "sample_rate":self.model.samplerate,"audio_channels":self.model.audio_channels,
                         "segment_seconds":float(self.model.segment),"use_train_segment":self.model.use_train_segment,
                         "hop_samples":self.model.hop_length,
                         "time_mapping":"nominal STFT bin centers; not an impulse-certified receptive-field boundary",
                         "frozen_parameters":sum(p.numel() for p in self.model.parameters()),
                         "weights":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights if p.is_file()}}

    def _capture(self, module, args, output):
        self.captured = tuple(x.detach() for x in output)

    @torch.no_grad()
    def encode(self, audio: torch.Tensor):
        if audio.ndim != 3 or audio.shape[1] != self.model.audio_channels:
            raise ValueError("expected [batch,channels,samples] at Demucs sample rate")
        duration = audio.shape[-1]/self.model.samplerate
        start = time.perf_counter()
        self.model(audio.to(self.device))
        if self.captured is None:
            raise ValueError("feature hook did not execute")
        frequency, temporal = self.captured
        # Transformer time branch is resampled onto the frequency time grid.
        aligned = F.interpolate(temporal,size=frequency.shape[-1],mode="linear",align_corners=False)
        aligned = aligned[:,:,None].expand(-1,-1,frequency.shape[2],-1)
        features = torch.cat((frequency,aligned),dim=1)
        times = (torch.arange(features.shape[-1],device=self.device)+0.5)*self.model.hop_length/self.model.samplerate
        keep = times < duration
        result = features[...,keep].contiguous()
        self.captured = None
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return result.cpu().half(), times[keep].cpu(), {
            "seconds":time.perf_counter()-start,"input_shape":list(audio.shape),
            "frequency_shape":list(frequency.shape),"temporal_shape":list(temporal.shape),
            "retained_shape":list(result.shape),
            "executed_seconds":max(duration,float(self.model.segment)) if self.model.use_train_segment else duration,
        }

    def close(self):
        self.handle.remove()
