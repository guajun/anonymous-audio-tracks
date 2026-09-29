"""DawDreamer graph construction and offline rendering.

The graph is deliberately linear: every source is a self-generated sample
played by a ``SamplerProcessor`` through an ordered filter/gain chain, and a
single ``AddProcessor`` sums the per-source chains.  Because the master is a
plain sum, the exported PCM stems add up to the exported mix (quadratic/cubic
master processing is explicitly out of scope for this issue).  Dry and
post-effect buffers are recorded from the same render pass, and a tiny impulse
probe measures the effect-chain latency for every source.

``dawdreamer`` is imported lazily so the rest of the package, its unit tests and
the Windows/Linux unit-test suite work without the optional render extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GainEffect, RenderConfig, SourceSpec
from .errors import RenderDependencyError, RenderError, RenderValidationError
from .synth import render_sample
from .timeline import ScoreEvent
from .version import RENDERER_ENGINE

#: Smallest non-zero amplitude considered a real output sample during probing.
_LATENCY_THRESHOLD = 1e-7


def require_dawdreamer():
    try:
        import dawdreamer  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - exercised on machines without the extra
        raise RenderDependencyError(
            "DawDreamer is required for rendering; install it with "
            "`uv sync --locked --extra render` (or `uv pip install '.[render]'`)"
        ) from exc
    return dawdreamer


@dataclass
class RenderedGraph:
    """Raw float audio recorded from one render pass."""

    mix: np.ndarray
    stems: dict[str, np.ndarray]
    dry: dict[str, np.ndarray]
    renderer_version: str
    effect_latency_samples: dict[str, int | None] = field(default_factory=dict)
    effect_latency_seconds: dict[str, float | None] = field(default_factory=dict)


def _source_sample(source: SourceSpec, config: RenderConfig) -> np.ndarray:
    data = render_sample(source.sample.type, source.sample.params, config.sample_rate, source.sample.seed)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 1:
        data = np.vstack([data, data])
    return np.ascontiguousarray(data, dtype=np.float32)


def _sampler_parameter_indexes(sampler) -> dict[str, int]:
    return {entry["name"]: int(entry["index"]) for entry in sampler.get_parameters_description()}


def _configure_sampler(sampler, source: SourceSpec) -> None:
    indexes = _sampler_parameter_indexes(sampler)
    required = (
        "Center Note",
        "Amp Active",
        "Amp Env Attack",
        "Amp Env Decay",
        "Amp Env Sustain",
        "Amp Env Release",
    )
    missing = [name for name in required if name not in indexes]
    if missing:
        raise RenderError(
            f"DawDreamer SamplerProcessor is missing expected parameter(s) {missing}; "
            "the renderer requires the documented sampler interface"
        )
    sampler.set_parameter(indexes["Center Note"], float(source.center_note))
    sampler.set_parameter(indexes["Amp Active"], 1.0)
    sampler.set_parameter(indexes["Amp Env Attack"], float(source.amp.attack_ms))
    sampler.set_parameter(indexes["Amp Env Decay"], float(source.amp.decay_ms))
    sampler.set_parameter(indexes["Amp Env Sustain"], float(source.amp.sustain))
    sampler.set_parameter(indexes["Amp Env Release"], float(source.amp.release_ms))


def _make_effect(engine, name: str, effect):
    if isinstance(effect, GainEffect):
        return engine.make_add_processor(name, [float(effect.gain)])
    return engine.make_filter_processor(
        name,
        mode=effect.mode,
        freq=float(effect.frequency_hz),
        q=float(effect.q),
        gain=float(effect.gain),
    )


def _effect_chain_nodes(engine, source: SourceSpec):
    nodes = []
    for index, effect in enumerate(source.effects):
        nodes.append(_make_effect(engine, f"{source.source_id}__fx{index}", effect))
    return nodes


def _dry_name(source_id: str) -> str:
    return f"{source_id}__dry"


def _stem_name(source_id: str) -> str:
    return f"{source_id}__stem"


def render_graph(config: RenderConfig, score: tuple[ScoreEvent, ...]) -> RenderedGraph:
    """Render mix/stems/dry buffers with DawDreamer from an expanded score."""

    dawdreamer = require_dawdreamer()
    engine = dawdreamer.RenderEngine(config.sample_rate, config.block_size)
    engine.set_bpm(config.bpm)

    by_source: dict[str, list[ScoreEvent]] = {source_id: [] for source_id in config.source_ids}
    for event in score:
        by_source[event.source_id].append(event)

    graph: list[tuple[object, list[str]]] = []
    stems: dict[str, np.ndarray] = {}
    dry: dict[str, np.ndarray] = {}
    dry_names: dict[str, str] = {}
    stem_names: dict[str, str] = {}
    # DawDreamer timelines always start at the beginning of the rendered
    # buffer, while controls/report times are absolute original-track seconds.
    # Convert to render-local time here and keep the metadata untouched.
    offset = config.track_start_seconds

    for source in config.sources:
        sampler = engine.make_sampler_processor(_dry_name(source.source_id), _source_sample(source, config))
        _configure_sampler(sampler, source)
        for event in by_source[source.source_id]:
            local_start = event.start_seconds - offset
            if local_start < 0.0:
                raise RenderValidationError(
                    f"source {source.source_id!r}: note at absolute {event.start_seconds}s "
                    f"precedes the render window start {offset}s"
                )
            sampler.add_midi_note(event.note, event.velocity, local_start, event.duration_seconds)
        sampler.record = True

        graph.append((sampler, []))
        previous = sampler.get_name()
        for effect_node in _effect_chain_nodes(engine, source):
            graph.append((effect_node, [previous]))
            previous = effect_node.get_name()
        gain = engine.make_add_processor(f"{source.source_id}__gain", [float(source.gain)])
        graph.append((gain, [previous]))
        gain.record = True

        dry_names[source.source_id] = sampler.get_name()
        stem_names[source.source_id] = gain.get_name()

    master = engine.make_add_processor("master", [1.0] * len(config.sources))
    graph.append((master, [stem_names[source_id] for source_id in config.source_ids]))
    master.record = True

    if not engine.load_graph(graph):
        raise RenderError("DawDreamer rejected the render graph (load_graph returned false)")
    if not engine.render(config.total_seconds):
        raise RenderError("DawDreamer render() returned false")

    mix = np.ascontiguousarray(engine.get_audio(), dtype=np.float32)
    for source in config.sources:
        stems[source.source_id] = np.ascontiguousarray(
            engine.get_audio(stem_names[source.source_id]), dtype=np.float32
        )
        dry[source.source_id] = np.ascontiguousarray(
            engine.get_audio(dry_names[source.source_id]), dtype=np.float32
        )

    _validate_shapes(mix, stems, dry)
    latency_samples: dict[str, int | None] = {}
    latency_seconds: dict[str, float | None] = {}
    for source in config.sources:
        samples = measure_effect_chain_latency(config, source)
        latency_samples[source.source_id] = samples
        latency_seconds[source.source_id] = (
            None if samples is None else samples / float(config.sample_rate)
        )

    version = getattr(dawdreamer, "__version__", "unknown")
    return RenderedGraph(
        mix=mix,
        stems=stems,
        dry=dry,
        renderer_version=f"{RENDERER_ENGINE.lower()}-{version}",
        effect_latency_samples=latency_samples,
        effect_latency_seconds=latency_seconds,
    )


def _validate_shapes(
    mix: np.ndarray, stems: dict[str, np.ndarray], dry: dict[str, np.ndarray]
) -> None:
    if mix.ndim != 2 or mix.size == 0:
        raise RenderValidationError(f"render produced empty mix audio with shape {mix.shape}")
    samples = mix.shape[1]
    for name, buffers in (("stem", stems), ("dry", dry)):
        for source_id, buffer in buffers.items():
            if buffer.ndim != 2 or buffer.size == 0:
                raise RenderValidationError(
                    f"{name} buffer for source {source_id!r} is empty (shape {buffer.shape})"
                )
            if buffer.shape[1] != samples:
                raise RenderValidationError(
                    f"{name} buffer for source {source_id!r} has {buffer.shape[1]} samples "
                    f"but the mix has {samples}"
                )


def measure_effect_chain_latency(config: RenderConfig, source: SourceSpec) -> int | None:
    """Samples of delay introduced by a source's effect/gain chain.

    A single-sample impulse is played through a copy of the chain (source ->
    effects -> gain) and the first non-zero output sample is reported.  The
    built-in processors used by the smoke samples are zero-latency, but the
    measurement is kept so a future effect cannot change latency silently.
    """

    dawdreamer = require_dawdreamer()
    engine = dawdreamer.RenderEngine(config.sample_rate, config.block_size)
    impulse = np.zeros((2, 16), dtype=np.float32)
    impulse[:, 0] = 1.0
    playback = engine.make_playback_processor(f"{source.source_id}__latency", impulse)
    graph: list[tuple[object, list[str]]] = [(playback, [])]
    previous = playback.get_name()
    for index, effect in enumerate(source.effects):
        node = _make_effect(engine, f"{source.source_id}__latency_fx{index}", effect)
        graph.append((node, [previous]))
        previous = node.get_name()
    gain = engine.make_add_processor(
        f"{source.source_id}__latency_gain", [float(source.gain)]
    )
    graph.append((gain, [previous]))
    if not engine.load_graph(graph):
        raise RenderError(
            f"DawDreamer rejected the latency-probe graph for source {source.source_id!r}"
        )
    if not engine.render(0.2):
        raise RenderError(f"DawDreamer latency probe failed for source {source.source_id!r}")
    audio = np.asarray(engine.get_audio(), dtype=np.float32)
    if audio.size == 0:
        return None
    active = np.flatnonzero(np.max(np.abs(audio), axis=0) > _LATENCY_THRESHOLD)
    return int(active[0]) if active.size else None


__all__ = ["RenderedGraph", "measure_effect_chain_latency", "render_graph", "require_dawdreamer"]
