"""Versioned continuous RMS labels on the original-track center grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

from aat.contracts.jsonio import dump_json, load_json
from aat.labels.energy import mean_square_at_samples
from aat.labels.wav import read_wav

VERSION = "aat-envelope-v1"


@dataclass(frozen=True)
class EnvelopeData:
    center_times: np.ndarray
    rms: np.ndarray
    valid: np.ndarray
    source_ids: tuple[str, ...]
    sample_rate: int
    energy_window_seconds: float
    hop_seconds: float
    input_sha256: dict[str, str]

    def __post_init__(self):
        if self.center_times.ndim != 1 or self.rms.shape != (len(self.center_times), len(self.source_ids)):
            raise ValueError("envelope: inconsistent time/source shapes")
        if self.valid.dtype != np.bool_ or self.valid.shape != self.center_times.shape:
            raise ValueError("envelope: expected boolean time validity")
        if not np.isfinite(self.center_times).all() or not np.isfinite(self.rms).all() or (self.rms < 0).any():
            raise ValueError("envelope: non-finite or negative values")
        if (self.center_times < 0).any() or (np.diff(self.center_times) <= 0).any():
            raise ValueError("envelope: times must be nonnegative and increasing")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("envelope: duplicate source IDs")
        if self.sample_rate <= 0 or self.energy_window_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("envelope: invalid sampling/window settings")

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "envelope.npz"
        np.savez_compressed(path, center_times=self.center_times, rms=self.rms, valid=self.valid)
        dump_json(directory / "envelope.json", {
            "version": VERSION, "kind": "source_envelope", "unit": "linear_rms_full_scale",
            "source_ids": list(self.source_ids), "sample_rate": self.sample_rate,
            "energy_window_seconds": self.energy_window_seconds, "hop_seconds": self.hop_seconds,
            "measurement": "centered rectangular mean-square; equal-power channel average",
            "normalization": "none", "input_sha256": self.input_sha256,
            "arrays_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(self.rms.shape),
        })

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        meta = load_json(directory / "envelope.json")
        if meta.get("version") != VERSION or meta.get("kind") != "source_envelope" or meta.get("unit") != "linear_rms_full_scale":
            raise ValueError("unsupported envelope semantics")
        path = directory / "envelope.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["arrays_sha256"]:
            raise ValueError("envelope array digest mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            result = cls(arrays["center_times"], arrays["rms"], arrays["valid"],
                         tuple(meta["source_ids"]), meta["sample_rate"],
                         meta["energy_window_seconds"], meta["hop_seconds"], meta["input_sha256"])
        if list(result.rms.shape) != meta["shape"]:
            raise ValueError("envelope metadata shape mismatch")
        return result


def label_sample(directory: str | Path, *, hop_seconds=0.02, energy_window_seconds=0.02):
    directory = Path(directory).resolve()
    if not np.isfinite(hop_seconds) or not np.isfinite(energy_window_seconds) or min(hop_seconds, energy_window_seconds) <= 0:
        raise ValueError("positive finite envelope windows required")
    manifest = load_json(directory / "manifest.json")
    source_ids = tuple(row["source_id"] for row in load_json(directory / "sources.json")["sources"])
    if set(source_ids) != set(manifest["stem_paths"]):
        raise ValueError("manifest/source registry mismatch")
    stems, digests, frames, rate = [], {}, None, manifest["sample_rate"]
    for source_id in source_ids:
        relative = manifest["stem_paths"][source_id]
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("stem path escapes sample directory")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["content_sha256"][relative]:
            raise ValueError("stem digest mismatch")
        wav = read_wav(path)
        if wav.sample_rate != rate or (frames is not None and wav.frames != frames):
            raise ValueError("stems must share a sample rate and duration")
        frames = wav.frames
        stems.append(wav.samples)
        digests[relative] = digest
    if frames is None:
        raise ValueError("at least one source required")
    origin = manifest.get("track_start_seconds", 0.0)
    local = np.arange(int(np.floor(frames / rate / hop_seconds + 1e-9)) + 1) * hop_seconds
    indices = np.floor(local * rate + 0.5).astype(np.int64)
    rms = np.stack([np.sqrt(mean_square_at_samples(stem, rate, energy_window_seconds, indices))
                    for stem in stems], axis=1).astype(np.float32)
    # This validity describes the envelope measurement center, not a model context window.
    result = EnvelopeData(origin + local, rms, indices < frames, source_ids, rate,
                          energy_window_seconds, hop_seconds, digests)
    result.save(directory)
    return result


def overlap_ratio(rms: np.ndarray, *, threshold=1e-3):
    """Acoustic overlap: >=2 active sources / >=1 active source, fixed RMS gate."""
    counts = (np.asarray(rms) > threshold).sum(axis=1)
    return float(np.sum(counts >= 2) / max(1, np.sum(counts >= 1)))
