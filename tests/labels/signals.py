"""Small programmatic waveforms shared by the labeler tests.

Everything here is generated with numpy; no audio files, models or datasets are
involved.  The same continuous waveform can be rendered at different sample
rates to check rate invariance.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
HOP_SECONDS = 0.02


def seconds_to_samples(seconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    return int(np.floor(seconds * sample_rate + 0.5))


def silence(duration_seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(seconds_to_samples(duration_seconds, sample_rate), dtype=np.float64)


def tone(
    duration_seconds: float,
    *,
    freq: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
    phase: float = 0.0,
) -> np.ndarray:
    samples = seconds_to_samples(duration_seconds, sample_rate)
    time = np.arange(samples, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq * time + phase)


def add_at(
    buffer: np.ndarray,
    signal: np.ndarray,
    start_seconds: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    start = seconds_to_samples(start_seconds, sample_rate)
    stop = min(start + signal.shape[0], buffer.shape[0])
    if stop > start:
        buffer[start:stop] += signal[: stop - start]
    return buffer


def pulse_buffer(
    duration_seconds: float,
    onset_seconds: float,
    *,
    pulse_seconds: float = 0.05,
    freq: float = 1000.0,
    amplitude: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    buffer = silence(duration_seconds, sample_rate)
    return add_at(
        buffer,
        tone(pulse_seconds, freq=freq, amplitude=amplitude, sample_rate=sample_rate),
        onset_seconds,
        sample_rate,
    )


def db_ramp_tone(
    duration_seconds: float,
    onset_seconds: float,
    *,
    attack_seconds: float = 0.4,
    hold_seconds: float = 0.6,
    db_start: float = -70.0,
    db_end: float = -9.0309,
    freq: float = 440.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Tone whose amplitude rises exponentially (linear in dB) over the attack."""

    attack_samples = max(2, seconds_to_samples(attack_seconds, sample_rate))
    hold_samples = seconds_to_samples(hold_seconds, sample_rate)
    frac = np.arange(attack_samples, dtype=np.float64) / (attack_samples - 1)
    attack_db = db_start + (db_end - db_start) * frac
    hold_db = np.full(hold_samples, db_end, dtype=np.float64)
    amp = np.power(10.0, np.concatenate((attack_db, hold_db)) / 20.0)
    time = np.arange(amp.size, dtype=np.float64) / sample_rate
    signal = amp * np.sin(2.0 * np.pi * freq * time)
    buffer = silence(duration_seconds, sample_rate)
    return add_at(buffer, signal, onset_seconds, sample_rate)


def decaying_tone(
    duration_seconds: float,
    onset_seconds: float,
    *,
    tone_seconds: float = 1.0,
    freq: float = 440.0,
    amplitude: float = 0.5,
    tau_seconds: float = 0.1,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Note with an exponential tail: keeps sounding (decaying) after note-off."""

    samples = seconds_to_samples(tone_seconds, sample_rate)
    time = np.arange(samples, dtype=np.float64) / sample_rate
    signal = amplitude * np.exp(-time / tau_seconds) * np.sin(2.0 * np.pi * freq * time)
    buffer = silence(duration_seconds, sample_rate)
    return add_at(buffer, signal, onset_seconds, sample_rate)


def sustain_then_tail(
    duration_seconds: float,
    onset_seconds: float,
    note_off_seconds: float,
    *,
    freq: float = 440.0,
    amplitude: float = 0.5,
    tau_seconds: float = 0.15,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Sustained note, then an exponential acoustic tail after note-off.

    The tail amplitude starts where the sustain ends, so the rendered audio
    keeps sounding above the off threshold well after ``note_off_seconds``.
    """

    sustain_samples = seconds_to_samples(note_off_seconds - onset_seconds, sample_rate)
    tail_seconds = max(0.0, duration_seconds - note_off_seconds)
    tail_samples = seconds_to_samples(tail_seconds, sample_rate)
    time = np.arange(sustain_samples + tail_samples, dtype=np.float64) / sample_rate
    envelope = np.ones(time.size, dtype=np.float64)
    tail_time = time[sustain_samples:] - (note_off_seconds - onset_seconds)
    envelope[sustain_samples:] = np.exp(-tail_time / tau_seconds)
    signal = amplitude * envelope * np.sin(2.0 * np.pi * freq * time)
    buffer = silence(duration_seconds, sample_rate)
    return add_at(buffer, signal, onset_seconds, sample_rate)


def constant_tone_with_dip(
    duration_seconds: float,
    *,
    first_on_seconds: float,
    first_off_seconds: float,
    second_on_seconds: float,
    dip_seconds: float = 0.04,
    freq: float = 440.0,
    amplitude: float = 0.5,
    dip_amplitude: float = 0.002,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Loud - short dip below the off threshold - loud, for hysteresis tests."""

    buffer = silence(duration_seconds, sample_rate)
    add_at(
        buffer,
        tone(
            first_off_seconds - first_on_seconds,
            freq=freq,
            amplitude=amplitude,
            sample_rate=sample_rate,
        ),
        first_on_seconds,
        sample_rate,
    )
    add_at(
        buffer,
        tone(dip_seconds, freq=freq, amplitude=dip_amplitude, sample_rate=sample_rate),
        first_off_seconds,
        sample_rate,
    )
    second_seconds = duration_seconds - second_on_seconds
    add_at(
        buffer,
        tone(second_seconds, freq=freq, amplitude=amplitude, sample_rate=sample_rate),
        second_on_seconds,
        sample_rate,
    )
    return buffer
