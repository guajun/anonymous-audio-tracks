"""Small programmatic waveforms for label tests (never real audio)."""

from __future__ import annotations

import numpy as np


def time_axis(duration_seconds: float, sample_rate: int) -> np.ndarray:
    return np.arange(round(duration_seconds * sample_rate), dtype=np.float64) / sample_rate


def sine(
    freq_hz: float,
    duration_seconds: float,
    sample_rate: int,
    *,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    return amplitude * np.sin(
        2.0 * np.pi * freq_hz * time_axis(duration_seconds, sample_rate) + phase
    )


def add_tone(
    buffer: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    stop_seconds: float,
    *,
    freq_hz: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Add a rectangular sine burst in ``[start, stop)`` to a 1-D buffer."""

    samples = np.asarray(buffer, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("add_tone expects a 1-D buffer")
    t = np.arange(samples.size, dtype=np.float64) / sample_rate
    mask = (t >= start_seconds) & (t < stop_seconds)
    samples[mask] += (amplitude * np.sin(2.0 * np.pi * freq_hz * t))[mask]
    return samples


def add_ramp_tone(
    buffer: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    attack_seconds: float,
    hold_until_seconds: float,
    *,
    freq_hz: float = 440.0,
    peak_amplitude: float = 1.0,
) -> np.ndarray:
    """Sine with a linear amplitude attack, then held until ``hold_until``."""

    samples = np.asarray(buffer, dtype=np.float64)
    t = np.arange(samples.size, dtype=np.float64) / sample_rate
    mask = (t >= start_seconds) & (t < hold_until_seconds)
    gain = np.clip((t - start_seconds) / attack_seconds, 0.0, 1.0)
    samples[mask] += (gain * peak_amplitude * np.sin(2.0 * np.pi * freq_hz * t))[mask]
    return samples


def add_decaying_tone(
    buffer: np.ndarray,
    sample_rate: int,
    note_on_seconds: float,
    note_off_seconds: float,
    end_seconds: float,
    *,
    freq_hz: float = 440.0,
    peak_amplitude: float = 1.0,
    tau_seconds: float = 0.2,
) -> np.ndarray:
    """Sustained tone plus an exponential release tail after note-off."""

    samples = np.asarray(buffer, dtype=np.float64)
    t = np.arange(samples.size, dtype=np.float64) / sample_rate
    sustain = (t >= note_on_seconds) & (t < note_off_seconds)
    samples[sustain] += (peak_amplitude * np.sin(2.0 * np.pi * freq_hz * t))[sustain]
    decay = (t >= note_off_seconds) & (t < end_seconds)
    envelope = peak_amplitude * np.exp(-(t - note_off_seconds) / tau_seconds)
    samples[decay] += (envelope * np.sin(2.0 * np.pi * freq_hz * t))[decay]
    return samples


def add_white_noise(
    buffer: np.ndarray, amplitude: float, *, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    buffer += rng.normal(0.0, amplitude, size=np.asarray(buffer).shape)
    return buffer
