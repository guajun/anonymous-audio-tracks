"""Render configuration loading and validation.

Configuration is plain TOML (Python's :mod:`tomllib`, no extra dependency) and
is mapped onto frozen dataclasses.  Validation is strict: unknown keys, wrong
types, out-of-range values and non-finite numbers are rejected with a message
that names the JSON-ish location, so a typo cannot silently produce a different
render.  Nothing in a committed config is machine-specific; Surge XT plugin
locations are injected through the CLI (or a local, ignored override file).
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import RenderConfigError
from .synth import SAMPLE_TYPES, resolve_sample_params

#: Upper bound for a single render.  Bigger jobs belong to the dataset issue.
MAX_TOTAL_SECONDS = 3600.0

_SOURCE_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_FILTER_MODES = ("low", "high")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RenderConfigError(f"{path}: expected a table, got {type(value).__name__}")
    return value


def _check_keys(table: Mapping[str, Any], allowed: tuple[str, ...], path: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise RenderConfigError(
            f"{path}: unknown key(s) {unknown}; allowed keys: {list(allowed)}"
        )


def _require_key(table: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in table:
        raise RenderConfigError(f"{path}: missing required key {key!r}")
    return table[key]


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderConfigError(f"{path}: expected a non-empty string")
    return value


def _require_number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RenderConfigError(f"{path}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise RenderConfigError(f"{path}: must be finite, got {value!r}")
    if minimum is not None:
        if number < minimum or (strict_minimum and number == minimum):
            comparator = ">" if strict_minimum else ">="
            raise RenderConfigError(f"{path}: must be {comparator} {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise RenderConfigError(f"{path}: must be <= {maximum}, got {number}")
    return number


def _require_int(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RenderConfigError(f"{path}: expected an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise RenderConfigError(f"{path}: must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise RenderConfigError(f"{path}: must be <= {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class NoteSpec:
    """One MIDI note inside a source pattern (times are pattern-local steps)."""

    step: int
    note: int
    velocity: int = 100
    length_steps: int = 1

    def to_dict(self) -> dict[str, int]:
        return {
            "step": self.step,
            "note": self.note,
            "velocity": self.velocity,
            "length_steps": self.length_steps,
        }


@dataclass(frozen=True)
class PatternSpec:
    """A repeating MIDI pattern; ``loop_steps=None`` plays the notes once."""

    step_seconds: float
    notes: tuple[NoteSpec, ...]
    loop_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "step_seconds": self.step_seconds,
            "notes": [note.to_dict() for note in self.notes],
        }
        if self.loop_steps is not None:
            data["loop_steps"] = self.loop_steps
        return data


@dataclass(frozen=True)
class SampleSpec:
    """A self-generated sample buffer selected by a seed and parameters."""

    type: str
    seed: int
    params: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "seed": self.seed, "params": dict(self.params)}


@dataclass(frozen=True)
class AmpSpec:
    """Sampler ADSR parameters (milliseconds, except ``sustain``)."""

    attack_ms: float = 2.0
    decay_ms: float = 0.0
    sustain: float = 1.0
    release_ms: float = 120.0

    def to_dict(self) -> dict[str, float]:
        return {
            "attack_ms": self.attack_ms,
            "decay_ms": self.decay_ms,
            "sustain": self.sustain,
            "release_ms": self.release_ms,
        }


@dataclass(frozen=True)
class FilterEffect:
    mode: str
    frequency_hz: float
    q: float = 0.7071067811865476
    gain: float = 1.0

    @property
    def type(self) -> str:
        return "filter"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "mode": self.mode,
            "frequency_hz": self.frequency_hz,
            "q": self.q,
            "gain": self.gain,
        }


@dataclass(frozen=True)
class GainEffect:
    gain: float = 1.0

    @property
    def type(self) -> str:
        return "gain"

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "gain": self.gain}


EffectSpec = FilterEffect | GainEffect


@dataclass(frozen=True)
class SourceSpec:
    """One independently controlled sound layer inside the composition."""

    source_id: str
    index: int
    sample: SampleSpec
    pattern: PatternSpec
    gain: float = 1.0
    center_note: int = 60
    amp: AmpSpec = field(default_factory=AmpSpec)
    effects: tuple[EffectSpec, ...] = ()
    preset_ref: str | None = None
    sample_ref: str | None = None
    notes: str | None = None

    @property
    def resolved_preset_ref(self) -> str:
        return self.preset_ref or f"aat/sampler/{self.sample.type}-v1"

    @property
    def resolved_sample_ref(self) -> str:
        return self.sample_ref or f"aat/generated/{self.sample.type}-v1"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source_id": self.source_id,
            "index": self.index,
            "sample": self.sample.to_dict(),
            "pattern": self.pattern.to_dict(),
            "gain": self.gain,
            "center_note": self.center_note,
            "amp": self.amp.to_dict(),
            "effects": [effect.to_dict() for effect in self.effects],
            "preset_ref": self.resolved_preset_ref,
            "sample_ref": self.resolved_sample_ref,
        }
        if self.notes is not None:
            data["notes"] = self.notes
        return data


@dataclass(frozen=True)
class SurgeConfig:
    """Surge XT probe settings; the plugin path is never committed."""

    plugin_path: str | None = None
    probe_duration_seconds: float = 1.0
    note: int = 60
    velocity: int = 100


@dataclass(frozen=True)
class RenderConfig:
    """A fully resolved render configuration."""

    sample_id: str
    composition: str
    seed: int
    sample_rate: int
    block_size: int
    bpm: float
    duration_seconds: float
    tail_seconds: float
    sources: tuple[SourceSpec, ...]
    track_start_seconds: float = 0.0
    notes: str | None = None
    surge: SurgeConfig = field(default_factory=SurgeConfig)

    @property
    def total_seconds(self) -> float:
        return self.duration_seconds + self.tail_seconds

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)

    @property
    def presets(self) -> list[str]:
        seen: list[str] = []
        for source in self.sources:
            reference = source.resolved_preset_ref
            if reference not in seen:
                seen.append(reference)
        return seen

    @property
    def sample_origins(self) -> list[str]:
        seen: list[str] = []
        for source in self.sources:
            reference = source.resolved_sample_ref
            if reference not in seen:
                seen.append(reference)
        return seen

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sample_id": self.sample_id,
            "composition": self.composition,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "block_size": self.block_size,
            "bpm": self.bpm,
            "duration_seconds": self.duration_seconds,
            "tail_seconds": self.tail_seconds,
            "track_start_seconds": self.track_start_seconds,
            "sources": [source.to_dict() for source in self.sources],
            "surge": {
                "plugin_path_configured": bool(self.surge.plugin_path),
                "probe_duration_seconds": self.surge.probe_duration_seconds,
                "note": self.surge.note,
                "velocity": self.surge.velocity,
            },
        }
        if self.notes is not None:
            data["notes"] = self.notes
        return data


def _parse_note(raw: Any, path: str, *, pattern_end_step: int | None) -> NoteSpec:
    table = _require_mapping(raw, path)
    _check_keys(table, ("step", "note", "velocity", "length_steps"), path)
    step = _require_int(_require_key(table, "step", path), f"{path}.step", minimum=0)
    note = _require_int(_require_key(table, "note", path), f"{path}.note", minimum=0, maximum=127)
    velocity = _require_int(
        table.get("velocity", 100), f"{path}.velocity", minimum=1, maximum=127
    )
    length_steps = _require_int(
        table.get("length_steps", 1), f"{path}.length_steps", minimum=1
    )
    if pattern_end_step is not None and step >= pattern_end_step:
        raise RenderConfigError(
            f"{path}.step: {step} must be < loop_steps ({pattern_end_step})"
        )
    return NoteSpec(step=step, note=note, velocity=velocity, length_steps=length_steps)


def _parse_pattern(raw: Any, path: str, *, duration_seconds: float) -> PatternSpec:
    table = _require_mapping(raw, path)
    _check_keys(table, ("step_seconds", "loop_steps", "notes"), path)
    step_seconds = _require_number(
        _require_key(table, "step_seconds", path),
        f"{path}.step_seconds",
        minimum=0.0,
        strict_minimum=True,
    )
    loop_steps: int | None = None
    if "loop_steps" in table:
        loop_steps = _require_int(table["loop_steps"], f"{path}.loop_steps", minimum=1)
    raw_notes = _require_key(table, "notes", path)
    if not isinstance(raw_notes, list) or not raw_notes:
        raise RenderConfigError(f"{path}.notes: expected a non-empty array of note tables")
    notes = tuple(
        _parse_note(item, f"{path}.notes[{position}]", pattern_end_step=loop_steps)
        for position, item in enumerate(raw_notes)
    )
    if loop_steps is None:
        for position, note in enumerate(notes):
            start = note.step * step_seconds
            if start >= duration_seconds:
                raise RenderConfigError(
                    f"{path}.notes[{position}].step: starts at {start:g}s which is outside "
                    f"the musical duration ({duration_seconds:g}s)"
                )
    else:
        loop_seconds = loop_steps * step_seconds
        if loop_seconds <= 0.0:
            raise RenderConfigError(f"{path}.loop_steps: loop duration must be > 0")
    return PatternSpec(step_seconds=step_seconds, notes=notes, loop_steps=loop_steps)


def _parse_effects(raw: Any, path: str, *, sample_rate: int) -> tuple[EffectSpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RenderConfigError(f"{path}: expected an array of effect tables")
    effects: list[EffectSpec] = []
    for position, item in enumerate(raw):
        item_path = f"{path}[{position}]"
        table = _require_mapping(item, item_path)
        effect_type = _require_str(_require_key(table, "type", item_path), f"{item_path}.type")
        if effect_type == "filter":
            _check_keys(table, ("type", "mode", "frequency_hz", "q", "gain"), item_path)
            mode = _require_str(table.get("mode", "low"), f"{item_path}.mode")
            if mode not in _FILTER_MODES:
                raise RenderConfigError(
                    f"{item_path}.mode: expected one of {list(_FILTER_MODES)}, got {mode!r}"
                )
            frequency = _require_number(
                _require_key(table, "frequency_hz", item_path),
                f"{item_path}.frequency_hz",
                minimum=20.0,
                maximum=sample_rate / 2.0,
            )
            q = _require_number(
                table.get("q", 0.7071067811865476), f"{item_path}.q", minimum=0.01, maximum=100.0
            )
            gain = _require_number(
                table.get("gain", 1.0), f"{item_path}.gain", minimum=0.0, maximum=4.0
            )
            effects.append(FilterEffect(mode=mode, frequency_hz=frequency, q=q, gain=gain))
        elif effect_type == "gain":
            _check_keys(table, ("type", "gain"), item_path)
            gain = _require_number(
                _require_key(table, "gain", item_path),
                f"{item_path}.gain",
                minimum=0.0,
                strict_minimum=True,
                maximum=4.0,
            )
            effects.append(GainEffect(gain=gain))
        else:
            raise RenderConfigError(
                f"{item_path}.type: unsupported effect {effect_type!r}; "
                "supported effects: ['filter', 'gain']"
            )
    return tuple(effects)


def _parse_source(
    raw: Any,
    index: int,
    path: str,
    *,
    config_seed: int,
    sample_rate: int,
    duration_seconds: float,
) -> SourceSpec:
    table = _require_mapping(raw, path)
    _check_keys(
        table,
        (
            "id",
            "sample",
            "pattern",
            "gain",
            "center_note",
            "amp",
            "effects",
            "preset_ref",
            "sample_ref",
            "seed",
            "notes",
        ),
        path,
    )
    source_id = _require_str(_require_key(table, "id", path), f"{path}.id")
    if not set(source_id) <= _SOURCE_ID_CHARS:
        raise RenderConfigError(
            f"{path}.id: must contain only letters, digits, '_' or '-', got {source_id!r}"
        )
    if len(source_id) > 63:
        raise RenderConfigError(f"{path}.id: must be at most 63 characters")

    sample_table = _require_mapping(_require_key(table, "sample", path), f"{path}.sample")
    _check_keys(sample_table, ("type", "seed", "params"), f"{path}.sample")
    sample_type = _require_str(
        _require_key(sample_table, "type", f"{path}.sample"), f"{path}.sample.type"
    )
    if sample_type not in SAMPLE_TYPES:
        raise RenderConfigError(
            f"{path}.sample.type: unsupported sample type {sample_type!r}; "
            f"expected one of {list(SAMPLE_TYPES)}"
        )
    sample_seed = _require_int(
        sample_table.get("seed", config_seed + index), f"{path}.sample.seed", minimum=0
    )
    raw_params = sample_table.get("params", {})
    params = _require_mapping(raw_params, f"{path}.sample.params")
    resolved_params = resolve_sample_params(sample_type, dict(params))

    gain = _require_number(table.get("gain", 1.0), f"{path}.gain", minimum=0.0, maximum=4.0)
    center_note = _require_int(
        table.get("center_note", 60), f"{path}.center_note", minimum=0, maximum=127
    )

    amp_table = _require_mapping(table.get("amp", {}), f"{path}.amp")
    _check_keys(amp_table, ("attack_ms", "decay_ms", "sustain", "release_ms"), f"{path}.amp")
    amp = AmpSpec(
        attack_ms=_require_number(
            amp_table.get("attack_ms", 2.0), f"{path}.amp.attack_ms", minimum=0.0, maximum=10000.0
        ),
        decay_ms=_require_number(
            amp_table.get("decay_ms", 0.0), f"{path}.amp.decay_ms", minimum=0.0, maximum=10000.0
        ),
        sustain=_require_number(
            amp_table.get("sustain", 1.0), f"{path}.amp.sustain", minimum=0.0, maximum=1.0
        ),
        release_ms=_require_number(
            amp_table.get("release_ms", 120.0),
            f"{path}.amp.release_ms",
            minimum=0.0,
            strict_minimum=True,
            maximum=60000.0,
        ),
    )

    effects = _parse_effects(table.get("effects"), f"{path}.effects", sample_rate=sample_rate)
    pattern = _parse_pattern(
        _require_key(table, "pattern", path),
        f"{path}.pattern",
        duration_seconds=duration_seconds,
    )
    preset_ref = table.get("preset_ref")
    sample_ref = table.get("sample_ref")
    if preset_ref is not None:
        preset_ref = _require_str(preset_ref, f"{path}.preset_ref")
    if sample_ref is not None:
        sample_ref = _require_str(sample_ref, f"{path}.sample_ref")
    notes = table.get("notes")
    if notes is not None:
        notes = _require_str(notes, f"{path}.notes")

    return SourceSpec(
        source_id=source_id,
        index=index,
        sample=SampleSpec(type=sample_type, seed=sample_seed, params=resolved_params),
        pattern=pattern,
        gain=gain,
        center_note=center_note,
        amp=amp,
        effects=effects,
        preset_ref=preset_ref,
        sample_ref=sample_ref,
        notes=notes,
    )


def load_config(path: str | Path) -> RenderConfig:
    """Load and validate a render config from a TOML file."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise RenderConfigError(f"config: file not found: {source}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RenderConfigError(f"config: invalid TOML in {source.name}: {exc}") from exc
    return config_from_dict(data, origin=source.name)


def config_from_dict(data: Mapping[str, Any], *, origin: str = "<config>") -> RenderConfig:
    """Validate an already-parsed TOML dictionary."""

    table = _require_mapping(data, origin)
    _check_keys(table, ("render", "sources", "surge"), origin)

    render = _require_mapping(_require_key(table, "render", origin), "render")
    _check_keys(
        render,
        (
            "sample_id",
            "composition",
            "seed",
            "sample_rate",
            "block_size",
            "bpm",
            "duration_seconds",
            "tail_seconds",
            "track_start_seconds",
            "notes",
        ),
        "render",
    )
    sample_id = _require_str(_require_key(render, "sample_id", "render"), "render.sample_id")
    composition = _require_str(
        _require_key(render, "composition", "render"), "render.composition"
    )
    seed = _require_int(_require_key(render, "seed", "render"), "render.seed", minimum=0)
    sample_rate = _require_int(
        _require_key(render, "sample_rate", "render"),
        "render.sample_rate",
        minimum=8000,
        maximum=384000,
    )
    block_size = _require_int(
        _require_key(render, "block_size", "render"),
        "render.block_size",
        minimum=1,
        maximum=8192,
    )
    bpm = _require_number(
        _require_key(render, "bpm", "render"), "render.bpm", minimum=0.0, strict_minimum=True
    )
    duration_seconds = _require_number(
        _require_key(render, "duration_seconds", "render"),
        "render.duration_seconds",
        minimum=1.0,
        strict_minimum=True,
        maximum=MAX_TOTAL_SECONDS,
    )
    tail_seconds = _require_number(
        _require_key(render, "tail_seconds", "render"),
        "render.tail_seconds",
        minimum=0.0,
        maximum=MAX_TOTAL_SECONDS,
    )
    track_start_seconds = _require_number(
        render.get("track_start_seconds", 0.0),
        "render.track_start_seconds",
        minimum=0.0,
    )
    if duration_seconds + tail_seconds > MAX_TOTAL_SECONDS:
        raise RenderConfigError(
            f"render: duration_seconds + tail_seconds must be <= {MAX_TOTAL_SECONDS:g}"
        )
    notes = render.get("notes")
    if notes is not None:
        notes = _require_str(notes, "render.notes")

    raw_sources = _require_key(table, "sources", origin)
    if not isinstance(raw_sources, list):
        raise RenderConfigError("sources: expected an array of source tables")
    if not 1 <= len(raw_sources) <= 8:
        raise RenderConfigError(
            f"sources: expected 1..8 independently controlled sources, got {len(raw_sources)}"
        )
    sources = tuple(
        _parse_source(
            raw,
            index,
            f"sources[{index}]",
            config_seed=seed,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )
        for index, raw in enumerate(raw_sources)
    )
    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            raise RenderConfigError(f"sources: duplicate id {source.source_id!r}")
        seen.add(source.source_id)

    surge_table = _require_mapping(table.get("surge", {}), "surge")
    _check_keys(surge_table, ("plugin_path", "probe_duration_seconds", "note", "velocity"), "surge")
    plugin_path = surge_table.get("plugin_path")
    if plugin_path is not None:
        plugin_path = _require_str(plugin_path, "surge.plugin_path")
    surge = SurgeConfig(
        plugin_path=plugin_path,
        probe_duration_seconds=_require_number(
            surge_table.get("probe_duration_seconds", 1.0),
            "surge.probe_duration_seconds",
            minimum=0.1,
            maximum=30.0,
        ),
        note=_require_int(surge_table.get("note", 60), "surge.note", minimum=0, maximum=127),
        velocity=_require_int(
            surge_table.get("velocity", 100), "surge.velocity", minimum=1, maximum=127
        ),
    )

    return RenderConfig(
        sample_id=sample_id,
        composition=composition,
        seed=seed,
        sample_rate=sample_rate,
        block_size=block_size,
        bpm=bpm,
        duration_seconds=duration_seconds,
        tail_seconds=tail_seconds,
        sources=sources,
        track_start_seconds=track_start_seconds,
        notes=notes,
        surge=surge,
    )


__all__ = [
    "AmpSpec",
    "EffectSpec",
    "FilterEffect",
    "GainEffect",
    "NoteSpec",
    "PatternSpec",
    "RenderConfig",
    "SampleSpec",
    "SourceSpec",
    "SurgeConfig",
    "config_from_dict",
    "load_config",
]
