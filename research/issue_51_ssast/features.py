"""Frozen, unpooled official SSAST features; no project data-module changes."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torchaudio.compliance.kaldi import fbank

OFFICIAL_COMMIT = "a1a3eecb94731e226308a6812f2fbf268d789caf"
FBANK = dict(htk_compat=True, sample_frequency=16000, use_energy=False,
             window_type="hanning", num_mel_bins=128, dither=0.0,
             frame_length=25., frame_shift=10., snip_edges=True,
             preemphasis_coefficient=.97, remove_dc_offset=True,
             low_freq=20., high_freq=0., use_log_fbank=True,
             use_power=True, round_to_power_of_two=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(frames, fshape, tshape, fstride, tstride, origin=0.):
    f = (128 - fshape) // fstride + 1
    t = (frames - tshape) // tstride + 1
    centers = origin + .0125 + .01 * (np.arange(t) * tstride + (tshape - 1) / 2)
    starts = origin + .01 * np.arange(t) * tstride
    ends = starts + .025 + .01 * (tshape - 1)
    return f, t, centers, starts, ends


def log_fbank(waveform):
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("explicit 16kHz mono [1,N] waveform required")
    x = fbank(waveform - waveform.mean(), **FBANK)
    return (x + 4.2677393) / (4.5689974 * 2)


class FrozenSSAST:
    def __init__(self, source, checkpoint, *, model_type, device="cuda"):
        if model_type not in ("frame", "patch"):
            raise ValueError("explicit model type required")
        spec = importlib.util.spec_from_file_location("official_ssast", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        kernel = tuple(sd["module.v.patch_embed.proj.weight"].shape[-2:])
        expected = (128, 2) if model_type == "frame" else (16, 16)
        if kernel != expected:
            raise ValueError(f"checkpoint type mismatch: {kernel} != {expected}")
        pf = int(sd["module.p_input_fdim"].item())
        pt = int(sd["module.p_input_tdim"].item())
        # Verify every pretraining key before the official downstream loader's
        # strict=False can conceal missing/random feature weights.
        original = module.ASTModel(fshape=kernel[0], tshape=kernel[1],
            fstride=kernel[0], tstride=kernel[1], input_fdim=pf,
            input_tdim=pt, model_size="base", pretrain_stage=True)
        load = torch.nn.DataParallel(original).load_state_dict(sd, strict=True)
        del original
        fs, ts = (128, 1) if model_type == "frame" else (10, 10)
        model = module.ASTModel(fshape=kernel[0], tshape=kernel[1],
            fstride=fs, tstride=ts, input_fdim=128, input_tdim=198,
            model_size="base", pretrain_stage=False,
            load_pretrained_mdl_path=str(checkpoint))
        self.v = model.v.eval().to(device)
        self.v.requires_grad_(False)
        self.device = device
        self.grid = grid(198, *kernel, fs, ts)
        f, t, *_ = self.grid
        self.f, self.t = f, t
        self.manifest = dict(official_commit=OFFICIAL_COMMIT,
            source_sha256=sha256(source), checkpoint_sha256=sha256(checkpoint),
            kernel=list(kernel), stride=[fs, ts], pretrain_dims=[pf, pt],
            checkpoint_keys=len(sd), missing_keys=list(load.missing_keys),
            unexpected_keys=list(load.unexpected_keys),
            output_grid=[f, t], output_dim=self.v.pos_embed.shape[-1],
            adapted_position_shape=list(self.v.pos_embed.shape),
            position_time_crop_start=pt // kernel[1] // 2 - t // 2,
            special_tokens=2, fbank=FBANK,
            normalization=dict(mean=-4.2677393, std=4.5689974, std_multiplier=2),
            padding="none at spectrogram level; waveform padding separately audited")
        if self.v.pos_embed.shape[1] != f * t + 2:
            raise ValueError("position/token shape mismatch")

    @torch.inference_mode()
    def __call__(self, x):
        # x[B,time,mel] -> token sequence in official F-major/T-minor order.
        x = x.to(self.device).unsqueeze(1).transpose(2, 3)
        v = self.v
        x = v.patch_embed(x)
        b = x.shape[0]
        x = torch.cat((v.cls_token.expand(b, -1, -1),
                       v.dist_token.expand(b, -1, -1), x), dim=1)
        x = v.pos_drop(x + v.pos_embed)
        for block in v.blocks:
            x = block(x)
        x = v.norm(x)[:, 2:]
        return x.reshape(b, self.f, self.t, -1)
