"""Deterministic DawDreamer renderer for issue #3.

Public entry points:

* :func:`aat.render.pipeline.render_sample` — render a config to an artifact
  directory (mix/stems/dry/sources/controls/manifest/report).
* :func:`aat.render.synth.render_sample` — synthesize one self-generated
  sample buffer (pure NumPy, no DawDreamer).
* :func:`aat.render.probe_surge.probe_surge` — Surge XT pass/fail probe.

The package never imports ``dawdreamer`` at module import time, so the base
install (and CI) can run the contract and unit suites without the optional
render extra.
"""

from __future__ import annotations

from .analysis import (
    check_stem_sum,
    count_clipped,
    count_non_finite,
    first_active_sample,
    last_active_sample,
    stem_sum_tolerance_lsb,
    tail_decay_ratio,
)
from .config import (
    AmpSpec,
    FilterEffect,
    GainEffect,
    NoteSpec,
    PatternSpec,
    RenderConfig,
    SampleSpec,
    SourceSpec,
    SurgeConfig,
    config_from_dict,
    load_config,
)
from .errors import (
    RenderConfigError,
    RenderDependencyError,
    RenderError,
    RenderValidationError,
    StemSumError,
    SurgeProbeError,
)
from .pipeline import RenderResult, render_sample
from .probe_surge import SurgeProbeResult, probe_surge
from .synth import render_sample as synthesize_sample
from .timeline import ScoreEvent, build_controls, expand_score
from .version import RENDER_VERSION, REPORT_SCHEMA_VERSION
from .wavio import quantize_int16, read_pcm16_wav, write_pcm16_wav

__all__ = [
    "AmpSpec",
    "FilterEffect",
    "GainEffect",
    "NoteSpec",
    "PatternSpec",
    "RENDER_VERSION",
    "REPORT_SCHEMA_VERSION",
    "RenderConfig",
    "RenderConfigError",
    "RenderDependencyError",
    "RenderError",
    "RenderResult",
    "RenderValidationError",
    "SampleSpec",
    "ScoreEvent",
    "SourceSpec",
    "StemSumError",
    "SurgeConfig",
    "SurgeProbeError",
    "SurgeProbeResult",
    "build_controls",
    "check_stem_sum",
    "config_from_dict",
    "count_clipped",
    "count_non_finite",
    "expand_score",
    "first_active_sample",
    "last_active_sample",
    "load_config",
    "probe_surge",
    "quantize_int16",
    "read_pcm16_wav",
    "render_sample",
    "stem_sum_tolerance_lsb",
    "synthesize_sample",
    "tail_decay_ratio",
    "write_pcm16_wav",
]
