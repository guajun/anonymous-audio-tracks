"""Deterministic, self-generated sample material.

The smoke renderer never downloads or references commercial material.  Every
source plays a small PCM buffer synthesized here from an explicit seed, so the
same configuration always produces the same samples on every machine.  Buffers
are returned as float32 arrays shaped ``(channels, samples)`` in ``[-1, 1]``,
which is exactly what ``DawDreamer.SamplerProcessor`` expects.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import RenderConfigError

#: Supported sample-generator names (no instrument semantics, just shapes).
SAMPLE_TYPES = ("kick", "snare", "hat", "pluck", "pad")

#: Default parameter sets per generator.  Unknown keys are rejected by
#: :func:`render_sample` so a typo cannot silently change the render.
_DEFAULTS: dict[str, dict[str, float | bool]] = {
    "kick": {
        "duration_seconds": 0.6,
        "freq_start_hz": 120.0,
        "freq_end_hz": 45.0,
        "decay_seconds": 0.35,
        "click": 0.4,
    },
    "snare": {
        "duration_seconds": 0.5,
        "tone_hz": 190.0,
        "tone_decay_seconds": 0.12,
        "noise_decay_seconds": 0.2,
        "noise_level": 0.7,
    },
    "hat": {
        "duration_seconds": 0.25,
        "decay_seconds": 0.05,
        "brightness": 0.9,
    },
    "pluck": {
        "duration_seconds": 1.2,
        "freq_hz": 110.0,
        "decay_seconds": 0.8,
        "damping": 0.5,
    },
    "pad": {
        "duration_seconds": 2.0,
        "freq_hz": 220.0,
        "attack_seconds": 0.3,
        "release_seconds": 0.8,
        "detune_cents": 6.0,
        "stereo": True,
    },
}

_POSITIVE_PARAMS = (
    "duration_seconds",
    "freq_start_hz",
    "freq_end_hz",
    "decay_seconds",
    "tone_hz",
    "tone_decay_seconds",
    "noise_decay_seconds",
    "freq_hz",
    "attack_seconds",
    "release_seconds",
)


def default_params(sample_type: str) -> dict[str, float | bool]:
    """Return a copy of the default parameters for ``sample_type``."""

    if sample_type not in _DEFAULTS:
        raise RenderConfigError(
            f"sample.type: unsupported sample type {sample_type!r}; "
            f"expected one of {list(SAMPLE_TYPES)}"
        )
    return dict(_DEFAULTS[sample_type])


def _coerce_params(sample_type: str, params: dict[str, object] | None) -> dict[str, float | bool]:
    merged = default_params(sample_type)
    for key, value in (params or {}).items():
        if key not in merged:
            raise RenderConfigError(
                f"sample.params: unknown key {key!r} for sample type {sample_type!r}; "
                f"allowed keys: {sorted(merged)}"
            )
        merged[key] = value  # type: ignore[assignment]
    for key in _POSITIVE_PARAMS:
        if key in merged:
            number = merged[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise RenderConfigError(f"sample.params.{key}: expected a number")
            number = float(number)
            if not math.isfinite(number) or number <= 0.0:
                raise RenderConfigError(f"sample.params.{key}: must be > 0, got {number}")
            merged[key] = number
    return merged


def _normalize(signal: np.ndarray, peak: float = 0.95) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    maximum = float(np.max(np.abs(signal))) if signal.size else 0.0
    if maximum > 0.0:
        signal = signal * (peak / maximum)
    return np.ascontiguousarray(signal, dtype=np.float32)


def _mono(signal: np.ndarray) -> np.ndarray:
    return signal.reshape(1, -1)


def _seed_offset(seed: int, channel: int) -> int:
    return int(seed) + 1009 * (int(channel) + 1)


def _synthesize(
    sample_type: str, params: dict[str, float | bool], sample_rate: int, seed: int
) -> np.ndarray:
    if sample_type == "kick":
        duration = float(params["duration_seconds"])
        n = max(1, int(round(duration * sample_rate)))
        t = np.arange(n, dtype=np.float64) / sample_rate
        tau = max(float(params["decay_seconds"]) / 3.0, 1.0 / sample_rate)
        freq = float(params["freq_end_hz"]) + (
            float(params["freq_start_hz"]) - float(params["freq_end_hz"])
        ) * np.exp(-t / tau)
        phase = 2.0 * math.pi * np.cumsum(freq) / sample_rate
        tone = np.sin(phase) * np.exp(-t / tau)
        click_level = float(params["click"])
        if click_level > 0.0:
            click_len = min(n, max(1, int(round(0.004 * sample_rate))))
            rng = np.random.default_rng(_seed_offset(seed, 0))
            click = np.zeros(n, dtype=np.float64)
            click[:click_len] = rng.uniform(-1.0, 1.0, click_len) * (
                1.0 - np.arange(click_len, dtype=np.float64) / click_len
            )
            tone = tone + click * click_level
        return _mono(_normalize(tone))

    if sample_type == "snare":
        duration = float(params["duration_seconds"])
        n = max(1, int(round(duration * sample_rate)))
        t = np.arange(n, dtype=np.float64) / sample_rate
        rng = np.random.default_rng(_seed_offset(seed, 0))
        noise = rng.uniform(-1.0, 1.0, n) * np.exp(
            -t / max(float(params["noise_decay_seconds"]), 1.0 / sample_rate)
        )
        tone = np.sin(2.0 * math.pi * float(params["tone_hz"]) * t) * np.exp(
            -t / max(float(params["tone_decay_seconds"]), 1.0 / sample_rate)
        )
        signal = noise * float(params["noise_level"]) + tone * (1.0 - float(params["noise_level"]))
        return _mono(_normalize(signal))

    if sample_type == "hat":
        duration = float(params["duration_seconds"])
        n = max(1, int(round(duration * sample_rate)))
        t = np.arange(n, dtype=np.float64) / sample_rate
        rng = np.random.default_rng(_seed_offset(seed, 0))
        noise = rng.uniform(-1.0, 1.0, n)
        bright = np.diff(noise, prepend=0.0) * float(params["brightness"]) + noise * (
            1.0 - float(params["brightness"])
        )
        signal = bright * np.exp(-t / max(float(params["decay_seconds"]), 1.0 / sample_rate))
        return _mono(_normalize(signal))

    if sample_type == "pluck":
        duration = float(params["duration_seconds"])
        n = max(1, int(round(duration * sample_rate)))
        delay_length = max(2, int(round(sample_rate / float(params["freq_hz"]))))
        rng = np.random.default_rng(_seed_offset(seed, 0))
        delay = rng.uniform(-1.0, 1.0, delay_length)
        out = np.zeros(n, dtype=np.float64)
        damping = 0.5 * (1.0 - float(params["damping"]))
        for index in range(n):
            current = delay[index % delay_length]
            out[index] = current
            following = delay[(index + 1) % delay_length]
            delay[index % delay_length] = damping * (current + following)
        return _mono(_normalize(out))

    if sample_type == "pad":
        duration = float(params["duration_seconds"])
        attack = float(params["attack_seconds"])
        release = float(params["release_seconds"])
        if attack + release >= duration:
            raise RenderConfigError(
                "sample.params: pad attack_seconds + release_seconds must be "
                f"< duration_seconds ({attack} + {release} >= {duration})"
            )
        n = max(1, int(round(duration * sample_rate)))
        t = np.arange(n, dtype=np.float64) / sample_rate
        envelope = np.ones(n, dtype=np.float64)
        attack_samples = max(1, int(round(attack * sample_rate)))
        release_samples = max(1, int(round(release * sample_rate)))
        envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
        envelope[-release_samples:] = np.linspace(
            1.0, 0.0, release_samples, endpoint=True
        )

        def voice(frequency: float) -> np.ndarray:
            signal = np.zeros(n, dtype=np.float64)
            for harmonic in range(1, 7):
                signal += (
                    np.sin(2.0 * math.pi * frequency * harmonic * t)
                    / float(harmonic)
                    * (0.9 ** (harmonic - 1))
                )
            return signal * envelope

        left = voice(float(params["freq_hz"]))
        if bool(params["stereo"]):
            detune = 2.0 ** (float(params["detune_cents"]) / 1200.0)
            right = voice(float(params["freq_hz"]) * detune)
            stereo = np.stack([_normalize(left), _normalize(right)])
            return np.ascontiguousarray(stereo, dtype=np.float32)
        return _mono(_normalize(left))

    raise RenderConfigError(f"sample.type: unsupported sample type {sample_type!r}")


def resolve_sample_params(
    sample_type: str, params: dict[str, object] | None
) -> dict[str, float | bool]:
    """Validate user parameters against the generator defaults and return them."""

    return _coerce_params(sample_type, params)


def render_sample(
    sample_type: str,
    params: dict[str, object] | None,
    sample_rate: int,
    seed: int,
) -> np.ndarray:
    """Synthesize one deterministic ``(channels, samples)`` float32 buffer."""

    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise RenderConfigError(f"sample_rate: expected a positive integer, got {sample_rate!r}")
    if not isinstance(seed, int) or seed < 0:
        raise RenderConfigError(f"sample seed: expected a non-negative integer, got {seed!r}")
    resolved = resolve_sample_params(sample_type, params)
    return _synthesize(sample_type, resolved, sample_rate, seed)


def sample_duration_seconds(sample_type: str, params: dict[str, object] | None) -> float:
    """Duration of the synthesized buffer without rendering it."""

    resolved = _coerce_params(sample_type, params)
    return float(resolved["duration_seconds"])


__all__ = [
    "SAMPLE_TYPES",
    "default_params",
    "render_sample",
    "resolve_sample_params",
    "sample_duration_seconds",
]
