"""Surge XT plugin probe.

The probe loads a user-supplied Surge XT VST3 through DawDreamer, plays one MIDI
note, and decides pass/fail from the rendered audio.  It never pretends the
built-in sampler smoke render is Surge validation: without a plugin path the
probe reports ``not_configured`` and the caller must treat Surge as unverified.
No commercial plugin is downloaded or bundled, and the plugin location is
injected at runtime (CLI/config), never committed.

Failure reasons are stable identifiers (``not_configured``, ``path_not_found``,
``plugin_load_failed``, ``graph_rejected``, ``render_failed``,
``midi_rejected``, ``empty_output``, ``silent_output``, ``non_finite_output``)
so tests and reports can distinguish "no plugin" from "plugin is broken".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import analysis
from .errors import SurgeProbeError
from .graph import require_dawdreamer
from .version import REPORT_SCHEMA_VERSION

#: Amplitude below which a render is considered silent (about -80 dBFS).
SILENCE_AMPLITUDE = 1e-4


@dataclass(frozen=True)
class SurgeProbeResult:
    """Outcome of one Surge XT probe attempt."""

    status: str
    reason: str
    detail: str | None
    sample_rate: int
    block_size: int
    duration_seconds: float
    note: int
    velocity: int
    input_channels: int | None = None
    output_channels: int | None = None
    latency_samples: int | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    non_finite_samples: int | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_json_dict(self) -> dict:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "kind": "surge_probe",
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "plugin_ref": "surge_xt",
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "duration_seconds": self.duration_seconds,
            "note": self.note,
            "velocity": self.velocity,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "latency_samples": self.latency_samples,
            "peak_dbfs": self.peak_dbfs,
            "rms_dbfs": self.rms_dbfs,
            "non_finite_samples": self.non_finite_samples,
        }


def _failed(reason: str, detail: str | None, **common) -> SurgeProbeResult:
    return SurgeProbeResult(status="failed", reason=reason, detail=detail, **common)


def probe_surge(
    plugin_path: str | Path | None,
    *,
    sample_rate: int = 44100,
    block_size: int = 512,
    duration_seconds: float = 1.0,
    note: int = 60,
    velocity: int = 100,
) -> SurgeProbeResult:
    """Probe a Surge XT VST3 without ever silently passing.

    Returns a result with ``status == "failed"`` for every environmental
    problem.  Only invalid arguments raise :class:`SurgeProbeError`.
    """

    if not isinstance(sample_rate, int) or sample_rate < 1:
        raise SurgeProbeError(f"sample_rate: expected a positive integer, got {sample_rate!r}")
    if not isinstance(block_size, int) or block_size < 1:
        raise SurgeProbeError(f"block_size: expected a positive integer, got {block_size!r}")
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise SurgeProbeError("duration_seconds: expected a number")
    if not 0.1 <= float(duration_seconds) <= 30.0:
        raise SurgeProbeError(f"duration_seconds: must be in [0.1, 30], got {duration_seconds}")
    if not isinstance(note, int) or not 0 <= note <= 127:
        raise SurgeProbeError(f"note: expected an integer in [0, 127], got {note!r}")
    if not isinstance(velocity, int) or not 1 <= velocity <= 127:
        raise SurgeProbeError(f"velocity: expected an integer in [1, 127], got {velocity!r}")

    common = {
        "sample_rate": int(sample_rate),
        "block_size": int(block_size),
        "duration_seconds": float(duration_seconds),
        "note": int(note),
        "velocity": int(velocity),
    }
    if plugin_path is None or not str(plugin_path).strip():
        return _failed(
            "not_configured",
            "no Surge XT plugin path supplied; pass --surge-plugin-path or set [surge].plugin_path "
            "in a local, uncommitted config",
            **common,
        )
    path = Path(plugin_path)
    if not path.exists():
        return _failed(
            "path_not_found",
            "the configured Surge XT plugin path does not exist on this machine",
            **common,
        )

    dawdreamer = require_dawdreamer()
    try:
        engine = dawdreamer.RenderEngine(int(sample_rate), int(block_size))
        processor = engine.make_plugin_processor("surge_xt", str(path))
    except Exception as exc:  # noqa: BLE001 - plugin hosts raise arbitrary types
        return _failed("plugin_load_failed", type(exc).__name__, **common)

    input_channels = None
    output_channels = None
    latency_samples = None
    try:
        input_channels = int(processor.get_num_input_channels())
        output_channels = int(processor.get_num_output_channels())
    except Exception:  # noqa: BLE001 - optional introspection
        pass
    try:
        latency_samples = int(processor.get_latency_samples())
    except Exception:  # noqa: BLE001 - optional introspection
        pass

    try:
        if not engine.load_graph([(processor, [])]):
            return _failed(
                "graph_rejected",
                "DawDreamer load_graph returned false",
                input_channels=input_channels,
                output_channels=output_channels,
                latency_samples=latency_samples,
                **common,
            )
    except Exception as exc:  # noqa: BLE001
        return _failed(
            "graph_rejected",
            type(exc).__name__,
            input_channels=input_channels,
            output_channels=output_channels,
            latency_samples=latency_samples,
            **common,
        )

    try:
        if not processor.add_midi_note(int(note), int(velocity), 0.0, float(duration_seconds)):
            return _failed(
                "midi_rejected",
                "add_midi_note returned false",
                input_channels=input_channels,
                output_channels=output_channels,
                latency_samples=latency_samples,
                **common,
            )
    except Exception as exc:  # noqa: BLE001
        return _failed(
            "midi_rejected",
            type(exc).__name__,
            input_channels=input_channels,
            output_channels=output_channels,
            latency_samples=latency_samples,
            **common,
        )

    try:
        if not engine.render(float(duration_seconds)):
            return _failed(
                "render_failed",
                "DawDreamer render returned false",
                input_channels=input_channels,
                output_channels=output_channels,
                latency_samples=latency_samples,
                **common,
            )
    except Exception as exc:  # noqa: BLE001
        return _failed(
            "render_failed",
            type(exc).__name__,
            input_channels=input_channels,
            output_channels=output_channels,
            latency_samples=latency_samples,
            **common,
        )

    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.size == 0:
        return _failed(
            "empty_output",
            "plugin rendered zero samples",
            input_channels=input_channels,
            output_channels=output_channels,
            latency_samples=latency_samples,
            **common,
        )
    non_finite = analysis.count_non_finite(audio)
    if non_finite:
        return _failed(
            "non_finite_output",
            f"{non_finite} non-finite sample(s)",
            input_channels=input_channels,
            output_channels=output_channels,
            latency_samples=latency_samples,
            non_finite_samples=non_finite,
            **common,
        )
    peak = analysis.peak(audio)
    if peak <= SILENCE_AMPLITUDE:
        return _failed(
            "silent_output",
            f"peak amplitude {peak:.3e} is below the silence floor",
            input_channels=input_channels,
            output_channels=output_channels,
            latency_samples=latency_samples,
            peak_dbfs=float(analysis.peak_dbfs(audio)),
            rms_dbfs=float(analysis.rms_dbfs(audio)),
            non_finite_samples=0,
            **common,
        )
    return SurgeProbeResult(
        status="passed",
        reason="rendered_audio",
        detail=None,
        input_channels=input_channels,
        output_channels=output_channels,
        latency_samples=latency_samples,
        peak_dbfs=float(analysis.peak_dbfs(audio)),
        rms_dbfs=float(analysis.rms_dbfs(audio)),
        non_finite_samples=0,
        **common,
    )


__all__ = ["SILENCE_AMPLITUDE", "SurgeProbeResult", "probe_surge"]
