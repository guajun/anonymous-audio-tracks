"""Versioned JSON document protocols.

Documents in this module are pure data.  They never load audio, models or run
outputs; binary payloads (activity/feature/prediction arrays) live in
:mod:`aat.contracts.arrays`.

Shared conventions
------------------
* Every document starts with ``schema_version`` and ``kind``.
* All times are absolute seconds on the original-track axis: ``t = 0`` is the
  start of the original composition.  A clip that starts later records
  ``track_start_seconds``; times are never re-based to the clip.
* Unknown extra keys are tolerated for forward compatibility, but any key
  documented here must have the documented type/units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .checks import (
    check_non_decreasing,
    check_schema_header,
    check_schema_version,
    check_strictly_increasing,
    require_git_commit,
    require_int,
    require_keys,
    require_mapping,
    require_number,
    require_probability,
    require_relative_posix_path,
    require_sequence,
    require_sha256_hex,
    require_str,
    require_unique,
)
from .errors import ContractError
from .jsonio import dump_json, load_json
from .version import (
    KIND_CONTROLS,
    KIND_SAMPLE_MANIFEST,
    KIND_SOURCES,
    KIND_TRAJECTORY,
    SCHEMA_VERSION,
)

_GROUP_KEYS = ("composition", "preset", "sample_origin")
_LIST_GROUP_KEYS = ("preset", "sample_origin")

#: Explicit manifest production stages: a renderer emits ``rendered`` before any
#: labeler has run; ``labeled`` adds the activity-label pair and its digests.
MANIFEST_STAGES = ("rendered", "labeled")

#: Kind of data behind a trajectory/prediction provenance record.
DATA_KINDS = ("model", "annotation", "mock")

#: Tolerance used when checking trajectory points against the recorded audio span.
TRAJECTORY_TIME_TOLERANCE_SECONDS = 1e-6


def _check_schema_version(value: Any) -> None:
    # Constructed documents may override the header; the header is the contract.
    check_schema_version(value)


def _check_string_list(value: Any, path: str) -> tuple[str, ...]:
    values = require_sequence(value, path)
    for position, item in enumerate(values):
        require_str(item, f"{path}[{position}]")
    require_unique(values, path)
    return tuple(values)


@dataclass(frozen=True)
class SampleManifest:
    """``manifest.json`` (``kind = sample_manifest``).

    Describes one rendered sample and is explicit about its production stage:

    ``rendered``
        The renderer has produced mix/stems/sources/controls only.  Activity
        label paths must be absent; no placeholder hashes are allowed.
    ``labeled``
        The labeler has additionally produced the activity sidecar pair; both
        paths and both digests are required.

    ``composition`` is one composition id.  ``preset`` and ``sample_origin`` are
    deduplicated, possibly empty lists of asset ids so that one composition can
    reference several presets/assets and split logic can detect shared assets
    without a shared "none" sentinel.
    """

    sample_id: str
    stage: str
    seed: int
    sample_rate: int
    duration_seconds: float
    track_start_seconds: float
    groups: Mapping[str, Any]
    mix_path: str
    stem_paths: Mapping[str, str]
    sources_path: str
    controls_path: str
    versions: Mapping[str, str]
    content_sha256: Mapping[str, str]
    activity_metadata_path: str | None = None
    activity_arrays_path: str | None = None
    render_latency_seconds: float | None = None
    tail_seconds: float | None = None
    notes: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_mapping(self.groups, "manifest.groups")
        require_mapping(self.stem_paths, "manifest.stem_paths")
        require_mapping(self.versions, "manifest.versions")
        require_mapping(self.content_sha256, "manifest.content_sha256")
        groups = dict(self.groups)
        for key in _LIST_GROUP_KEYS:
            if isinstance(groups.get(key), (list, tuple)):
                groups[key] = list(groups[key])
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "stem_paths", dict(self.stem_paths))
        object.__setattr__(self, "versions", dict(self.versions))
        object.__setattr__(self, "content_sha256", dict(self.content_sha256))
        self.validate()

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        require_str(self.sample_id, "manifest.sample_id")
        require_str(self.stage, "manifest.stage")
        if self.stage not in MANIFEST_STAGES:
            raise ContractError(
                f"manifest.stage: expected one of {list(MANIFEST_STAGES)}, got {self.stage!r}"
            )
        require_int(self.seed, "manifest.seed", minimum=0)
        require_int(self.sample_rate, "manifest.sample_rate", minimum=1)
        require_number(
            self.duration_seconds,
            "manifest.duration_seconds",
            minimum=0.0,
            strict_minimum=True,
        )
        require_number(self.track_start_seconds, "manifest.track_start_seconds", minimum=0.0)

        groups = require_mapping(self.groups, "manifest.groups")
        require_keys(groups, _GROUP_KEYS, "manifest.groups")
        require_str(groups["composition"], "manifest.groups.composition")
        for key in _LIST_GROUP_KEYS:
            _check_string_list(groups[key], f"manifest.groups.{key}")

        mix_path = require_relative_posix_path(self.mix_path, "manifest.mix_path")
        stems = require_mapping(self.stem_paths, "manifest.stem_paths")
        if not stems:
            raise ContractError("manifest.stem_paths: at least one stem is required")
        for source_id, stem_path in stems.items():
            require_str(source_id, "manifest.stem_paths key")
            require_relative_posix_path(stem_path, f"manifest.stem_paths[{source_id!r}]")

        sources_path = require_relative_posix_path(self.sources_path, "manifest.sources_path")
        controls_path = require_relative_posix_path(self.controls_path, "manifest.controls_path")

        label_paths: tuple[str, ...]
        if self.stage == "rendered":
            if self.activity_metadata_path is not None or self.activity_arrays_path is not None:
                raise ContractError(
                    "manifest.activity_metadata_path/manifest.activity_arrays_path: must be "
                    "omitted in the 'rendered' stage; label artifacts do not exist yet"
                )
            label_paths = ()
        else:
            if self.activity_metadata_path is None or self.activity_arrays_path is None:
                raise ContractError(
                    "manifest: 'labeled' stage requires both activity_metadata_path and "
                    "activity_arrays_path"
                )
            label_paths = (
                require_relative_posix_path(
                    self.activity_metadata_path, "manifest.activity_metadata_path"
                ),
                require_relative_posix_path(
                    self.activity_arrays_path, "manifest.activity_arrays_path"
                ),
            )

        versions = require_mapping(self.versions, "manifest.versions")
        require_keys(versions, ("renderer",), "manifest.versions")
        for key, value in versions.items():
            require_str(value, f"manifest.versions[{key!r}]")

        hashes = require_mapping(self.content_sha256, "manifest.content_sha256")
        required_hash_paths = sorted(
            {mix_path, *stems.values(), sources_path, controls_path, *label_paths}
        )
        require_keys(hashes, required_hash_paths, "manifest.content_sha256")
        for artifact_path, digest in hashes.items():
            require_str(artifact_path, "manifest.content_sha256 key")
            require_sha256_hex(digest, f"manifest.content_sha256[{artifact_path!r}]")

        if self.render_latency_seconds is not None:
            require_number(
                self.render_latency_seconds,
                "manifest.render_latency_seconds",
                minimum=0.0,
            )
        if self.tail_seconds is not None:
            require_number(self.tail_seconds, "manifest.tail_seconds", minimum=0.0)
        if self.notes is not None:
            require_str(self.notes, "manifest.notes", allow_empty=True)

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        groups = require_mapping(self.groups, "manifest.groups")
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": KIND_SAMPLE_MANIFEST,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "seed": self.seed,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "track_start_seconds": self.track_start_seconds,
            "groups": {
                "composition": groups["composition"],
                "preset": list(groups["preset"]),
                "sample_origin": list(groups["sample_origin"]),
            },
            "mix_path": self.mix_path,
            "stem_paths": dict(self.stem_paths),
            "sources_path": self.sources_path,
            "controls_path": self.controls_path,
            "versions": dict(self.versions),
            "content_sha256": dict(self.content_sha256),
        }
        if self.activity_metadata_path is not None:
            data["activity_metadata_path"] = self.activity_metadata_path
        if self.activity_arrays_path is not None:
            data["activity_arrays_path"] = self.activity_arrays_path
        if self.render_latency_seconds is not None:
            data["render_latency_seconds"] = self.render_latency_seconds
        if self.tail_seconds is not None:
            data["tail_seconds"] = self.tail_seconds
        if self.notes is not None:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> SampleManifest:
        check_schema_header(data, KIND_SAMPLE_MANIFEST)
        require_keys(
            data,
            (
                "sample_id",
                "stage",
                "seed",
                "sample_rate",
                "duration_seconds",
                "track_start_seconds",
                "groups",
                "mix_path",
                "stem_paths",
                "sources_path",
                "controls_path",
                "versions",
                "content_sha256",
            ),
            "manifest",
        )
        return cls(
            sample_id=data["sample_id"],
            stage=data["stage"],
            seed=data["seed"],
            sample_rate=data["sample_rate"],
            duration_seconds=data["duration_seconds"],
            track_start_seconds=data["track_start_seconds"],
            groups=data["groups"],
            mix_path=data["mix_path"],
            stem_paths=data["stem_paths"],
            sources_path=data["sources_path"],
            controls_path=data["controls_path"],
            versions=data["versions"],
            content_sha256=data["content_sha256"],
            activity_metadata_path=data.get("activity_metadata_path"),
            activity_arrays_path=data.get("activity_arrays_path"),
            render_latency_seconds=data.get("render_latency_seconds"),
            tail_seconds=data.get("tail_seconds"),
            notes=data.get("notes"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path: str | Path) -> Path:
        return dump_json(path, self.to_json_dict())

    @classmethod
    def load(cls, path: str | Path) -> SampleManifest:
        return cls.from_json_dict(load_json(path))


@dataclass(frozen=True)
class SourceEntry:
    """One generation-time raw source identity (``sources[].source_id``)."""

    source_id: str
    index: int
    renderer: str | None = None
    preset_ref: str | None = None
    sample_ref: str | None = None
    seed: int | None = None
    notes: str | None = None

    def validate(self) -> None:
        require_str(self.source_id, "sources[].source_id")
        require_int(self.index, "sources[].index", minimum=0)
        for name in ("renderer", "preset_ref", "sample_ref"):
            value = getattr(self, name)
            if value is not None:
                require_str(value, f"sources[].{name}")
        if self.seed is not None:
            require_int(self.seed, "sources[].seed", minimum=0)
        if self.notes is not None:
            require_str(self.notes, "sources[].notes", allow_empty=True)

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {"source_id": self.source_id, "index": self.index}
        for name in ("renderer", "preset_ref", "sample_ref", "seed", "notes"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any], path: str = "sources[]") -> SourceEntry:
        require_mapping(data, path)
        require_keys(data, ("source_id", "index"), path)
        return cls(
            source_id=data["source_id"],
            index=data["index"],
            renderer=data.get("renderer"),
            preset_ref=data.get("preset_ref"),
            sample_ref=data.get("sample_ref"),
            seed=data.get("seed"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class SourceRegistry:
    """``sources.json`` (``kind = sources``).

    ``source_ids`` order defines the column order of activity/feature arrays and
    the generation-time identity of each rendered stem.  It is *not* a stable
    track identity: see :class:`Track`.
    """

    sources: tuple[SourceEntry, ...]
    sample_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        self.validate()

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(entry.source_id for entry in self.sources)

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        if self.sample_id is not None:
            require_str(self.sample_id, "sources.sample_id")
        if not isinstance(self.sources, tuple):
            raise ContractError("sources.sources: expected an array of source objects")
        for position, entry in enumerate(self.sources):
            if not isinstance(entry, SourceEntry):
                raise ContractError(f"sources.sources[{position}]: expected a source object")
            entry.validate()
            if entry.index != position:
                raise ContractError(
                    f"sources.sources[{position}].index: must equal its 0-based position "
                    f"{position}, got {entry.index}"
                )
        require_unique(self.source_ids, "sources.sources[].source_id")

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": KIND_SOURCES,
            "sources": [entry.to_json_dict() for entry in self.sources],
        }
        if self.sample_id is not None:
            data["sample_id"] = self.sample_id
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> SourceRegistry:
        check_schema_header(data, KIND_SOURCES)
        require_keys(data, ("sources",), "sources")
        raw_sources = require_sequence(data["sources"], "sources.sources")
        entries = tuple(
            SourceEntry.from_json_dict(raw, f"sources.sources[{position}]")
            for position, raw in enumerate(raw_sources)
        )
        return cls(
            sources=entries,
            sample_id=data.get("sample_id"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path: str | Path) -> Path:
        return dump_json(path, self.to_json_dict())

    @classmethod
    def load(cls, path: str | Path) -> SourceRegistry:
        return cls.from_json_dict(load_json(path))


@dataclass(frozen=True)
class ControlEvent:
    """One generation-time control event in ``controls.json``."""

    time_seconds: float
    source_id: str
    event_type: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_mapping(self.data, "controls.events[].data")
        object.__setattr__(self, "data", dict(self.data))
        self.validate()

    def validate(self) -> None:
        require_number(self.time_seconds, "controls.events[].time_seconds", minimum=0.0)
        require_str(self.source_id, "controls.events[].source_id")
        require_str(self.event_type, "controls.events[].event_type")
        require_mapping(self.data, "controls.events[].data")

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "time_seconds": self.time_seconds,
            "source_id": self.source_id,
            "event_type": self.event_type,
            "data": dict(self.data),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any], path: str = "controls.events[]") -> ControlEvent:
        require_mapping(data, path)
        require_keys(data, ("time_seconds", "source_id", "event_type"), path)
        return cls(
            time_seconds=data["time_seconds"],
            source_id=data["source_id"],
            event_type=data["event_type"],
            data=data.get("data", {}),
        )


@dataclass(frozen=True)
class Controls:
    """``controls.json`` (``kind = controls``).

    MIDI/parameter/trigger events are stored separately from acoustic activity
    labels.  Several events may share one timestamp (simultaneous onsets), so
    times are non-decreasing, not strictly increasing.  All times are absolute
    seconds on the original-track axis.
    """

    events: tuple[ControlEvent, ...]
    sample_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        self.validate()

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        if self.sample_id is not None:
            require_str(self.sample_id, "controls.sample_id")
        if not isinstance(self.events, tuple):
            raise ContractError("controls.events: expected an array of event objects")
        for position, event in enumerate(self.events):
            if not isinstance(event, ControlEvent):
                raise ContractError(f"controls.events[{position}]: expected an event object")
            event.validate()
        check_non_decreasing(
            [event.time_seconds for event in self.events],
            "controls.events[].time_seconds",
        )

    def referenced_source_ids(self) -> set[str]:
        return {event.source_id for event in self.events}

    def validate_source_references(self, source_ids: Sequence[str]) -> None:
        known = set(source_ids)
        unknown = sorted(self.referenced_source_ids() - known)
        if unknown:
            raise ContractError(
                "controls.events[].source_id: references unknown source id(s): "
                + ", ".join(unknown)
            )

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": KIND_CONTROLS,
            "events": [event.to_json_dict() for event in self.events],
        }
        if self.sample_id is not None:
            data["sample_id"] = self.sample_id
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> Controls:
        check_schema_header(data, KIND_CONTROLS)
        require_keys(data, ("events",), "controls")
        raw_events = require_sequence(data["events"], "controls.events")
        events = tuple(
            ControlEvent.from_json_dict(raw, f"controls.events[{position}]")
            for position, raw in enumerate(raw_events)
        )
        return cls(
            events=events,
            sample_id=data.get("sample_id"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path: str | Path) -> Path:
        return dump_json(path, self.to_json_dict())

    @classmethod
    def load(cls, path: str | Path) -> Controls:
        return cls.from_json_dict(load_json(path))


@dataclass(frozen=True)
class RunProvenance:
    """Provenance of one run that produced a document.

    ``data_kind`` distinguishes real model inference, human/algorithmic
    annotation and mock/placeholder data so that viewers and evaluations cannot
    silently treat a mock as a model result.
    """

    run_id: str
    data_kind: str
    model_id: str | None = None
    git_commit: str | None = None
    config_hash: str | None = None
    created_at_utc: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        require_str(self.run_id, "provenance.run_id")
        require_str(self.data_kind, "provenance.data_kind")
        if self.data_kind not in DATA_KINDS:
            raise ContractError(
                f"provenance.data_kind: expected one of {list(DATA_KINDS)}, got {self.data_kind!r}"
            )
        for name in ("model_id", "config_hash"):
            value = getattr(self, name)
            if value is not None:
                require_str(value, f"provenance.{name}")
        if self.git_commit is not None:
            require_git_commit(self.git_commit, "provenance.git_commit")
        if self.created_at_utc is not None:
            require_str(self.created_at_utc, "provenance.created_at_utc")
            try:
                parsed = datetime.fromisoformat(self.created_at_utc)
            except ValueError as exc:
                raise ContractError(
                    f"provenance.created_at_utc: expected ISO-8601 UTC timestamp, "
                    f"got {self.created_at_utc!r}"
                ) from exc
            if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
                raise ContractError(
                    "provenance.created_at_utc: must carry a UTC timezone "
                    f"(e.g. '2026-09-29T12:00:00Z'), got {self.created_at_utc!r}"
                )
        if self.notes is not None:
            require_str(self.notes, "provenance.notes", allow_empty=True)

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {"run_id": self.run_id, "data_kind": self.data_kind}
        for name in ("model_id", "git_commit", "config_hash", "created_at_utc", "notes"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any], path: str = "provenance") -> RunProvenance:
        require_mapping(data, path)
        require_keys(data, ("run_id", "data_kind"), path)
        return cls(
            run_id=data["run_id"],
            data_kind=data["data_kind"],
            model_id=data.get("model_id"),
            git_commit=data.get("git_commit"),
            config_hash=data.get("config_hash"),
            created_at_utc=data.get("created_at_utc"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class AudioSpan:
    """Original-track audio extent of the analyzed clip.

    ``duration_seconds`` may be 0 for an empty track.  ``track_start_seconds``
    is the absolute original-track time of the clip's first sample; trajectory
    times are never re-based to it.
    """

    duration_seconds: float
    track_start_seconds: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        require_number(
            self.duration_seconds,
            "trajectory.audio.duration_seconds",
            minimum=0.0,
        )
        require_number(
            self.track_start_seconds,
            "trajectory.audio.track_start_seconds",
            minimum=0.0,
        )

    @property
    def end_seconds(self) -> float:
        return self.track_start_seconds + self.duration_seconds

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "duration_seconds": self.duration_seconds,
            "track_start_seconds": self.track_start_seconds,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any], path: str = "trajectory.audio") -> AudioSpan:
        require_mapping(data, path)
        require_keys(data, ("duration_seconds", "track_start_seconds"), path)
        return cls(
            duration_seconds=data["duration_seconds"],
            track_start_seconds=data["track_start_seconds"],
        )


@dataclass(frozen=True)
class Track:
    """One anonymous source trajectory.

    ``track_id`` is a stable identity assigned by cross-window association.  It
    persists across windows and short silences and is never a slot index; raw
    slot provenance is carried separately in ``slot_indices``.  A ``None`` entry
    in ``slot_indices`` marks a silence-memory point with no matched candidate
    instead of fabricating a slot number.
    """

    track_id: str
    center_times: tuple[float, ...]
    activity: tuple[float, ...]
    confidence: tuple[float, ...] | None = None
    slot_indices: tuple[int | None, ...] | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        require_sequence(self.center_times, "trajectory.tracks[].center_times")
        require_sequence(self.activity, "trajectory.tracks[].activity")
        object.__setattr__(self, "center_times", tuple(self.center_times))
        object.__setattr__(self, "activity", tuple(self.activity))
        if self.confidence is not None:
            require_sequence(self.confidence, "trajectory.tracks[].confidence")
            object.__setattr__(self, "confidence", tuple(self.confidence))
        if self.slot_indices is not None:
            require_sequence(self.slot_indices, "trajectory.tracks[].slot_indices")
            object.__setattr__(self, "slot_indices", tuple(self.slot_indices))
        self.validate()

    def validate(self, *, slots: int | None = None) -> None:
        require_str(self.track_id, "trajectory.tracks[].track_id")
        times = require_sequence(self.center_times, "trajectory.tracks[].center_times")
        if len(times) == 0:
            raise ContractError(
                f"trajectory.tracks[{self.track_id!r}]: an empty curve must be omitted, "
                "not stored as a zero-length track"
            )
        for position, value in enumerate(times):
            require_number(
                value,
                f"trajectory.tracks[{self.track_id!r}].center_times[{position}]",
                minimum=0.0,
            )
        check_strictly_increasing(
            list(times), f"trajectory.tracks[{self.track_id!r}].center_times"
        )

        activity = require_sequence(
            self.activity, f"trajectory.tracks[{self.track_id!r}].activity"
        )
        if len(activity) != len(times):
            raise ContractError(
                f"trajectory.tracks[{self.track_id!r}]: activity length ({len(activity)}) "
                f"must match center_times length ({len(times)})"
            )
        for position, value in enumerate(activity):
            require_probability(
                value, f"trajectory.tracks[{self.track_id!r}].activity[{position}]"
            )

        if self.confidence is not None:
            confidence = require_sequence(
                self.confidence, f"trajectory.tracks[{self.track_id!r}].confidence"
            )
            if len(confidence) != len(times):
                raise ContractError(
                    f"trajectory.tracks[{self.track_id!r}]: confidence length "
                    f"({len(confidence)}) must match center_times length ({len(times)})"
                )
            for position, value in enumerate(confidence):
                require_probability(
                    value, f"trajectory.tracks[{self.track_id!r}].confidence[{position}]"
                )

        if self.slot_indices is not None:
            indices = require_sequence(
                self.slot_indices, f"trajectory.tracks[{self.track_id!r}].slot_indices"
            )
            if len(indices) != len(times):
                raise ContractError(
                    f"trajectory.tracks[{self.track_id!r}]: slot_indices length "
                    f"({len(indices)}) must match center_times length ({len(times)})"
                )
            for position, value in enumerate(indices):
                if value is None:
                    continue  # silence-memory point without a matched candidate
                require_int(
                    value,
                    f"trajectory.tracks[{self.track_id!r}].slot_indices[{position}]",
                    minimum=0,
                )
                if slots is not None and value >= slots:
                    raise ContractError(
                        f"trajectory.tracks[{self.track_id!r}].slot_indices[{position}]: "
                        f"slot {value} is outside 0..{slots - 1}"
                    )

        if self.notes is not None:
            require_str(self.notes, "trajectory.tracks[].notes", allow_empty=True)

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "track_id": self.track_id,
            "center_times": list(self.center_times),
            "activity": list(self.activity),
        }
        if self.confidence is not None:
            data["confidence"] = list(self.confidence)
        if self.slot_indices is not None:
            data["slot_indices"] = list(self.slot_indices)
        if self.notes is not None:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any], path: str = "trajectory.tracks[]") -> Track:
        require_mapping(data, path)
        require_keys(data, ("track_id", "center_times", "activity"), path)
        return cls(
            track_id=data["track_id"],
            center_times=data["center_times"],
            activity=data["activity"],
            confidence=data.get("confidence"),
            slot_indices=data.get("slot_indices"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class Trajectory:
    """``trajectory.json`` (``kind = trajectory``).

    A whole-song association output.  ``audio`` records the analyzed clip's
    original-track span so viewers can reject mismatched audio or misaligned
    clips; every curve point must lie inside that span (tolerance
    :data:`TRAJECTORY_TIME_TOLERANCE_SECONDS`).  An absent curve is represented
    by omitting the track; a track that is currently silent keeps its identity
    and stores an all-zero activity series.  Empty audio must not contain tracks.
    """

    audio: AudioSpan
    provenance: RunProvenance
    tracks: tuple[Track, ...]
    sample_id: str | None = None
    slots: int | None = None
    params: Mapping[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_sequence(self.tracks, "trajectory.tracks")
        object.__setattr__(self, "tracks", tuple(self.tracks))
        if isinstance(self.audio, Mapping):
            object.__setattr__(self, "audio", AudioSpan.from_json_dict(self.audio))
        if isinstance(self.provenance, Mapping):
            object.__setattr__(
                self, "provenance", RunProvenance.from_json_dict(self.provenance)
            )
        if self.params is not None:
            require_mapping(self.params, "trajectory.params")
            object.__setattr__(self, "params", dict(self.params))
        self.validate()

    def validate(self) -> None:
        _check_schema_version(self.schema_version)
        if self.sample_id is not None:
            require_str(self.sample_id, "trajectory.sample_id")
        if not isinstance(self.audio, AudioSpan):
            raise ContractError("trajectory.audio: expected an audio span object")
        self.audio.validate()
        if not isinstance(self.provenance, RunProvenance):
            raise ContractError("trajectory.provenance: expected a provenance object")
        self.provenance.validate()
        slots: int | None = None
        if self.slots is not None:
            slots = require_int(self.slots, "trajectory.slots", minimum=1)
        if self.params is not None:
            require_mapping(self.params, "trajectory.params")

        if self.audio.duration_seconds == 0.0 and self.tracks:
            raise ContractError(
                "trajectory: audio.duration_seconds is 0 (empty audio), so no track may exist"
            )
        start = self.audio.track_start_seconds
        end = self.audio.end_seconds
        for position, track in enumerate(self.tracks):
            if not isinstance(track, Track):
                raise ContractError(f"trajectory.tracks[{position}]: expected a track object")
            track.validate(slots=slots)
            for point, value in enumerate(track.center_times):
                if value < start - TRAJECTORY_TIME_TOLERANCE_SECONDS or (
                    value > end + TRAJECTORY_TIME_TOLERANCE_SECONDS
                ):
                    raise ContractError(
                        f"trajectory.tracks[{track.track_id!r}].center_times[{point}]: "
                        f"{value} lies outside the audio span [{start}, {end}] "
                        f"(tolerance {TRAJECTORY_TIME_TOLERANCE_SECONDS})"
                    )
        require_unique((track.track_id for track in self.tracks), "trajectory.tracks[].track_id")

    def to_json_dict(self) -> dict[str, Any]:
        self.validate()
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": KIND_TRAJECTORY,
            "audio": self.audio.to_json_dict(),
            "provenance": self.provenance.to_json_dict(),
            "tracks": [track.to_json_dict() for track in self.tracks],
        }
        if self.sample_id is not None:
            data["sample_id"] = self.sample_id
        if self.slots is not None:
            data["slots"] = self.slots
        if self.params is not None:
            data["params"] = dict(self.params)
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> Trajectory:
        check_schema_header(data, KIND_TRAJECTORY)
        require_keys(data, ("audio", "provenance", "tracks"), "trajectory")
        tracks = require_sequence(data["tracks"], "trajectory.tracks")
        return cls(
            audio=AudioSpan.from_json_dict(data["audio"]),
            provenance=RunProvenance.from_json_dict(data["provenance"]),
            tracks=tuple(
                Track.from_json_dict(raw, f"trajectory.tracks[{position}]")
                for position, raw in enumerate(tracks)
            ),
            sample_id=data.get("sample_id"),
            slots=data.get("slots"),
            params=data.get("params"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path: str | Path) -> Path:
        return dump_json(path, self.to_json_dict())

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        return cls.from_json_dict(load_json(path))
