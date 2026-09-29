"""Token grid of the Qwen3-Omni AuT audio encoder (pure NumPy port).

The grid is derived from the pinned upstream source, not from memory:

* ``transformers`` 4.57.0
  ``src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py``
* class ``Qwen3OmniMoeAudioEncoder`` (``forward``) and the module-level
  ``_get_feat_extract_output_lengths``.
* checkpoint ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` revision
  ``26291f793822fb6be9555850f06dfe95f2d7e695``: ``n_window=50``,
  ``n_window_infer=800``, 128 mel bins, 16 kHz, mel hop 160 samples
  (10 ms), ``conv2d1..3`` are kernel-3 stride-2 with padding 1.

Facts that drive the formula (all verified against the upstream source and the
installed class in ``tests/encoders/test_grid.py`` and the real probe):

* Mel spectrogram frames are produced at 10 ms (hop 160 at 16 kHz), so a
  *full* mel chunk of ``n_window * 2 = 100`` frames is one second of audio.
* Each chunk is convolved independently (``pad_sequence`` + ``conv2d1..3``),
  so no convolution context crosses a chunk boundary.  A 100-frame chunk
  yields ``13`` CNN tokens: ``100 -> 50 -> 25 -> 13``.
* Three kernel-3/stride-2/padding-1 convolutions give token ``k`` inside its
  chunk a receptive field of mel frames ``[8k - 7, 8k + 7]`` (centre
  ``8k``), i.e. a nominal *centre* step of 80 ms and no drift inside a chunk.
* The residual tail chunk of ``r = mel_len % 100`` frames yields
  ``f(r)`` tokens with the same 8-frame centre step; the last token's field is
  clipped by the chunk.  ``mel_len -> tokens`` is therefore::

      tokens(mel_len) = 13 * (mel_len // 100) + f(mel_len % 100)
      f(r) = (( ((r-1)//2 + 1 - 1)//2 + 1 - 1)//2 + 1)   (r > 0), f(0) = 0

  The practical consequence is a 40 ms gap between the last token of a chunk
  (centre 0.96 s inside the chunk) and the first token of the next chunk
  (centre +0.00 s of the next second).  Token times are therefore not a
  uniform 80 ms grid and must never be presented as 10 ms-precision timings.

Everything here is deterministic, dependency-free (NumPy only) and shared by
the real encoder path and the fake test double so the time axis and validity
mask are tested identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import EncoderInputError, EncoderError

#: Sampling rate the AuT encoder expects (checkpoint ``preprocessor_config``).
AUT_SAMPLE_RATE = 16000
#: Mel hop of the official Whisper feature extractor (160 / 16000 = 10 ms).
AUT_MEL_HOP_SAMPLES = 160
#: STFT window of the official Whisper feature extractor.
AUT_N_FFT = 400
#: Number of mel bins of the official checkpoint.
AUT_MEL_BINS = 128
#: Upstream ``n_window``; the convolution chunk is ``n_window * 2`` mel frames.
AUT_N_WINDOW = 50
#: Mel frames per AuT convolution chunk (``n_window * 2`` with ``n_window=50``).
AUT_CHUNK_MEL_FRAMES = 100
#: Tokens produced by one full 100-frame chunk (100 -> 50 -> 25 -> 13).
AUT_TOKENS_PER_FULL_CHUNK = 13
#: Total stride of the three stride-2 convolutions.
AUT_CONV_STRIDE = 8
#: Symmetric receptive-field half width in mel frames (kernel 3 x 3 layers).
AUT_FIELD_RADIUS_MEL = 7
#: Token-centre step inside one chunk: 8 mel frames = 80 ms.
AUT_NOMINAL_TOKEN_STEP_SECONDS = 0.08
#: Minimum samples accepted by ``torch.stft(center=True)`` with
#: ``n_fft=400`` (``n_fft // 2 < length``); shorter audio yields no mel frame.
AUT_MIN_AUDIO_SAMPLES = AUT_N_FFT // 2 + 1


def _tokens_in_partial_chunk(frames: int) -> int:
    """Port of the non-``100`` part of ``_get_feat_extract_output_lengths``."""

    if frames < 0 or frames >= AUT_CHUNK_MEL_FRAMES:
        raise EncoderInputError(
            f"partial chunk must have 0 <= frames < {AUT_CHUNK_MEL_FRAMES}, got {frames}"
        )
    if frames == 0:
        return 0
    feat_lengths = (frames - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1


def aut_output_length(mel_len: int) -> int:
    """Number of AuT tokens for ``mel_len`` mel frames.

    Mirrors ``transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.
    _get_feat_extract_output_lengths`` for a single sample.
    """

    if isinstance(mel_len, bool) or not isinstance(mel_len, (int, np.integer)):
        raise EncoderInputError(f"mel_len: expected an integer, got {type(mel_len).__name__}")
    mel_len = int(mel_len)
    if mel_len < 0:
        raise EncoderInputError(f"mel_len: must be >= 0, got {mel_len}")
    if mel_len == 0:
        return 0
    return (mel_len // AUT_CHUNK_MEL_FRAMES) * AUT_TOKENS_PER_FULL_CHUNK + _tokens_in_partial_chunk(
        mel_len % AUT_CHUNK_MEL_FRAMES
    )


def aut_chunk_lengths(mel_len: int) -> list[int]:
    """Per-convolution-chunk mel lengths exactly as the encoder splits them."""

    total = aut_output_length(mel_len)  # validation + parity with the formula
    del total
    full, tail = divmod(int(mel_len), AUT_CHUNK_MEL_FRAMES)
    lengths = [AUT_CHUNK_MEL_FRAMES] * full
    if tail:
        lengths.append(tail)
    return lengths


@dataclass(frozen=True)
class TokenGrid:
    """Per-token layout of one encoded buffer.

    Arrays are parallel and have shape ``(token_count,)``:

    ``chunk_index``
        Index of the 100-frame convolution chunk the token belongs to.
    ``token_in_chunk``
        Token position inside its chunk (0-based).
    ``mel_centers``
        Centre mel frame of the token inside the whole mel input
        (``100 * chunk_index + 8 * token_in_chunk``).
    ``field_start`` / ``field_stop``
        Inclusive/exclusive mel frames that actually contribute to the token.
        The nominal field is ``[mel_center-7, mel_center+7]``; it is clipped to
        the token's own chunk because chunks are convolved independently.
    """

    mel_len: int
    chunk_index: np.ndarray
    token_in_chunk: np.ndarray
    mel_centers: np.ndarray
    field_start: np.ndarray
    field_stop: np.ndarray

    def __post_init__(self) -> None:
        if self.mel_len < 0:
            raise EncoderError(f"mel_len: must be >= 0, got {self.mel_len}")
        expected = self.mel_centers.shape
        for name in (
            "chunk_index",
            "token_in_chunk",
            "mel_centers",
            "field_start",
            "field_stop",
        ):
            array = np.asarray(getattr(self, name))
            if array.shape != expected:
                raise EncoderError(
                    f"grid.{name}: shape {array.shape} != expected {expected}; a grid is a "
                    "set of parallel per-token arrays and may be a row subset of the "
                    "full-buffer grid"
                )
        if self.token_count and not bool(np.all(self.field_stop > self.field_start)):
            raise EncoderError("grid: field_stop must be > field_start for every token")
        if self.token_count and not bool(np.all(np.diff(self.mel_centers) > 0)):
            raise EncoderError("grid: mel_centers must be strictly increasing")

    @property
    def token_count(self) -> int:
        return int(self.mel_centers.shape[0])

    def frame_times(self, origin_seconds: float, *, mel_hop_seconds: float = 0.01) -> np.ndarray:
        """Absolute frame times (float64 seconds) of the token centres.

        ``origin_seconds`` may be negative for windows that start before the
        original-track origin; the protocol requires non-negative times only
        when building a :class:`aat.contracts.FeatureData` document.
        """

        return float(origin_seconds) + self.mel_centers.astype(np.float64) * float(
            mel_hop_seconds
        )

    def sample_support(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample-level support ``[start, stop)`` of every token's mel field.

        A mel frame ``t`` covers samples ``[t * hop - n_fft // 2,
        t * hop + n_fft // 2)``; the support of a field ``[a, b]`` is the union,
        clipped to ``[0, n_samples]``.
        """

        start = self.field_start.astype(np.int64) * AUT_MEL_HOP_SAMPLES - AUT_N_FFT // 2
        stop = (self.field_stop.astype(np.int64) - 1) * AUT_MEL_HOP_SAMPLES + AUT_N_FFT // 2
        start = np.maximum(start, 0)
        stop = np.minimum(stop, int(n_samples))
        return start, stop


def aut_token_grid(mel_len: int) -> TokenGrid:
    """Build the :class:`TokenGrid` for ``mel_len`` mel frames."""

    token_count = aut_output_length(mel_len)
    chunk_index = np.empty(token_count, dtype=np.int64)
    token_in_chunk = np.empty(token_count, dtype=np.int64)
    mel_centers = np.empty(token_count, dtype=np.int64)
    field_start = np.empty(token_count, dtype=np.int64)
    field_stop = np.empty(token_count, dtype=np.int64)

    cursor = 0
    for chunk, length in enumerate(aut_chunk_lengths(mel_len)):
        tokens_here = _tokens_in_partial_chunk(length) if length < AUT_CHUNK_MEL_FRAMES else 13
        for k in range(tokens_here):
            chunk_index[cursor] = chunk
            token_in_chunk[cursor] = k
            center = chunk * AUT_CHUNK_MEL_FRAMES + k * AUT_CONV_STRIDE
            mel_centers[cursor] = center
            local_start = max(0, k * AUT_CONV_STRIDE - AUT_FIELD_RADIUS_MEL)
            local_stop = min(length - 1, k * AUT_CONV_STRIDE + AUT_FIELD_RADIUS_MEL) + 1
            field_start[cursor] = chunk * AUT_CHUNK_MEL_FRAMES + local_start
            field_stop[cursor] = chunk * AUT_CHUNK_MEL_FRAMES + local_stop
            cursor += 1
    if cursor != token_count:  # pragma: no cover - defensive parity check
        raise EncoderError(
            f"grid construction produced {cursor} tokens, formula says {token_count}"
        )
    if token_count != aut_output_length(mel_len):  # pragma: no cover - parity guard
        raise EncoderError(
            f"grid construction produced {token_count} tokens, formula says "
            f"{aut_output_length(mel_len)}"
        )
    return TokenGrid(
        mel_len=mel_len,
        chunk_index=chunk_index,
        token_in_chunk=token_in_chunk,
        mel_centers=mel_centers,
        field_start=field_start,
        field_stop=field_stop,
    )


def token_valid_mask(
    grid: TokenGrid,
    n_samples: int,
    *,
    real_start: int = 0,
    real_stop: int | None = None,
) -> np.ndarray:
    """Validity mask for tokens whose mel field uses zero-padded samples.

    ``real_start``/``real_stop`` delimit the sample range inside the encoded
    buffer that contains real audio; samples outside are zero padding inserted
    by centered-window extraction.  A token is valid exactly when its
    sample-level support is fully inside that real range.  For a plain buffer
    (no artificial padding) the whole buffer is real and every token is valid;
    the encoder's own convolution zero-padding at the buffer border is normal
    model behaviour, not pipeline padding.
    """

    if real_stop is None:
        real_stop = int(n_samples)
    if not (0 <= real_start <= real_stop <= int(n_samples)):
        raise EncoderInputError(
            f"real sample range [{real_start}, {real_stop}) outside buffer of {n_samples} samples"
        )
    start, stop = grid.sample_support(n_samples)
    return (start >= real_start) & (stop <= real_stop)
