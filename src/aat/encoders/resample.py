"""Anti-aliased 16 kHz resampling for the AuT input (NumPy only).

The AuT checkpoint expects mono, 16 kHz audio (``preprocessor_config.json``:
``sampling_rate=16000``).  Renders produced by the project renderer are 44.1
kHz, so the feature path has to resample.  This module implements a
windowed-sinc (Blackman window) resampler with no SciPy dependency so the
base CPU test environment stays lightweight.  The resampler is deterministic
and tested for DC gain and alias rejection in ``tests/encoders/test_resample.py``.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import EncoderInputError
from .grid import AUT_SAMPLE_RATE


#: Default windowed-sinc zero crossings per side used by :func:`resample_audio`.
DEFAULT_RESAMPLE_TAPS = 16


def resample_kernel_half_width(
    sample_rate: int,
    target_rate: int = AUT_SAMPLE_RATE,
    *,
    taps: int = DEFAULT_RESAMPLE_TAPS,
) -> int:
    """Half width (in *input* samples) of the resampling kernel support.

    Returns ``0`` when no resampling happens (same rate), so callers can use
    the same interval arithmetic for both cases.  The width matches the kernel
    in :func:`resample_audio` (``ceil(taps / cutoff)`` input samples per side).
    """

    if sample_rate == target_rate:
        return 0
    ratio = target_rate / sample_rate
    cutoff = min(1.0, ratio)
    return max(1, int(math.ceil(taps / cutoff)))


def real_region_after_resample(
    valid_samples: np.ndarray,
    sample_rate: int,
    target_rate: int,
    n_out: int,
    *,
    taps: int = DEFAULT_RESAMPLE_TAPS,
) -> tuple[int, int] | None:
    """Map the contiguous real-sample span of a window onto the resampled axis.

    ``valid_samples`` marks the input samples that carry real audio (the rest is
    artificial zero padding).  It must be one contiguous ``True`` span;
    internal holes are rejected instead of silently filled, because a hole
    would mean a token can be computed from padding while the mask says real.

    The returned half-open interval ``[start, stop)`` contains exactly the
    output samples whose **full resampling kernel support** stays inside the
    real input interval (``taps`` are widened by the downsampling cutoff, same
    as :func:`resample_audio`).  At the outer buffer edges the resampler uses
    edge padding of real samples, so the region is not shrunk there unless the
    real span ends inside the buffer (i.e. artificial padding exists).

    Integer arithmetic is used for the interval conversion so equal-rate
    windows keep their exact sample bounds (``stop`` is exclusive and is never
    turned into ``stop + 1``).
    """

    mask = np.asarray(valid_samples)
    if mask.ndim != 1:
        raise EncoderInputError(
            f"valid_samples: expected a 1-D array, got shape {mask.shape}"
        )
    if mask.dtype != np.bool_:
        raise EncoderInputError(
            f"valid_samples: expected a boolean array, got dtype {mask.dtype}"
        )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    start, stop = int(indices[0]), int(indices[-1]) + 1
    if indices.size != stop - start:
        raise EncoderInputError(
            "valid_samples must be one contiguous True span (no internal holes); "
            f"got {indices.size} true samples in [{start}, {stop})"
        )

    if sample_rate == target_rate:
        return max(0, min(start, n_out)), max(0, min(stop, n_out))

    half_width = resample_kernel_half_width(sample_rate, target_rate, taps=taps)
    n_in = int(mask.shape[0])
    # Left/right buffer borders use edge replication of real audio, so only an
    # interior real-span border needs the kernel margin.
    start_in = start + half_width if start > 0 else 0
    stop_in = stop - half_width if stop < n_in else n_in
    if stop_in <= start_in:
        start_out = max(0, min(n_out, (start_in * target_rate + sample_rate - 1) // sample_rate))
        return start_out, start_out
    start_out = 0 if start_in <= 0 else (start_in * target_rate + sample_rate - 1) // sample_rate
    stop_out = (stop_in * target_rate + sample_rate - 1) // sample_rate
    start_out = max(0, min(start_out, n_out))
    stop_out = max(start_out, min(stop_out, n_out))
    return start_out, stop_out


def resample_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_rate: int = AUT_SAMPLE_RATE,
    *,
    taps: int = DEFAULT_RESAMPLE_TAPS,
) -> np.ndarray:
    """Resample a 1-D mono signal with a Blackman-windowed sinc kernel.

    ``taps`` is the number of sinc zero crossings kept on each side; for
    downsampling the kernel is widened by ``1 / cutoff`` input samples so the
    anti-alias filter has enough taps at the lower cutoff.  Edges are handled
    by edge-padding the input.  Returns ``float32`` and a length of
    ``round(n_in * target / sample_rate)`` (half-up, matching
    :func:`aat.windowing.seconds_to_samples` conventions).
    """

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise EncoderInputError(f"audio: expected a 1-D mono array, got shape {samples.shape}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise EncoderInputError(f"sample_rate: expected an integer, got {type(sample_rate).__name__}")
    sample_rate = int(sample_rate)
    if sample_rate < 1:
        raise EncoderInputError(f"sample_rate: must be >= 1, got {sample_rate}")
    if isinstance(target_rate, bool) or not isinstance(target_rate, (int, np.integer)):
        raise EncoderInputError(f"target_rate: expected an integer, got {type(target_rate).__name__}")
    target_rate = int(target_rate)
    if target_rate < 1:
        raise EncoderInputError(f"target_rate: must be >= 1, got {target_rate}")
    if isinstance(taps, bool) or not isinstance(taps, (int, np.integer)) or int(taps) < 1:
        raise EncoderInputError(f"taps: expected an integer >= 1, got {taps!r}")
    taps = int(taps)

    values = samples.astype(np.float64, copy=False)
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise EncoderInputError("audio: contains NaN/Inf")
    if sample_rate == target_rate:
        return values.astype(np.float32)

    ratio = target_rate / sample_rate
    n_out = max(1, math.floor(values.size * ratio + 0.5))
    positions = np.arange(n_out, dtype=np.float64) / ratio
    cutoff = min(1.0, ratio)
    half_width = max(1, int(math.ceil(taps / cutoff)))

    padded = np.pad(values, (half_width, half_width), mode="edge")
    base = np.floor(positions).astype(np.int64)
    frac = positions - base

    acc = np.zeros(n_out, dtype=np.float64)
    weight_sum = np.zeros(n_out, dtype=np.float64)
    for offset in range(-half_width, half_width + 1):
        distance = frac - offset
        outside = np.abs(distance) > half_width
        window = np.where(
            outside,
            0.0,
            0.42 + 0.5 * np.cos(np.pi * distance / half_width) + 0.08 * np.cos(2.0 * np.pi * distance / half_width),
        )
        weight = np.sinc(distance * cutoff) * window
        acc += weight * padded[base + offset + half_width]
        weight_sum += weight
    # Normalising the kernel keeps DC gain exactly 1 and tames edge effects.
    denominator = np.where(np.abs(weight_sum) < 1e-12, 1.0, weight_sum)
    return (acc / denominator).astype(np.float32)


def prepare_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    taps: int = DEFAULT_RESAMPLE_TAPS,
) -> np.ndarray:
    """Validate and convert input audio to the AuT's 16 kHz mono float32.

    Only resampling to the model's own rate is performed here; no amplitude
    normalisation, channel mixing or trimming is applied (the official
    ``WhisperFeatureExtractor`` does the log-mel computation).  ``taps`` must
    be passed identically to :func:`real_region_after_resample` when mapping a
    validity mask through the same resampling step.
    """

    samples = np.asarray(audio)
    if samples.ndim != 1:
        raise EncoderInputError(f"audio: expected a 1-D mono array, got shape {samples.shape}")
    converted = resample_audio(samples, sample_rate, AUT_SAMPLE_RATE, taps=taps)
    if converted.size and not np.all(np.isfinite(converted)):
        raise EncoderInputError("audio: resampling produced NaN/Inf")
    return converted
