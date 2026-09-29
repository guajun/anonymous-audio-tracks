"""Dataset index: scan, validate, group and split labeled samples.

The index is a JSON document (``dataset-index-v1``) that:

* discovers every sample directory under a data root (a directory containing
  ``manifest.json``) in a deterministic sorted order;
* validates the frozen protocol documents (manifest/sources/controls/activity)
  and every recorded sha256 digest, with every error locating its sample;
* records the original artifact digests plus a digest of the record itself, so
  a moved or edited input is detected instead of being consumed silently;
* groups samples by ``composition``/``preset``/``sample_origin`` in independent
  namespaces with a transitive connected-component union, so a whole group
  always lands in exactly one split;
* stores the explicit split seed, requested ratios, actual ratios and every
  oversized component so a rebuild with the same seed is reproducible.

Paths stored in the index are relative to the data root (POSIX separators).
The ``layout`` block optionally stores the data root relative to the index
file, so an index and its data root can be moved together without absolute
paths entering the repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

from ..contracts import (
    DEFAULT_SLOTS,
    DEFAULT_WINDOW_SECONDS,
    ActivityData,
    ContractError,
    Controls,
    SampleManifest,
    SourceRegistry,
    dump_json,
    load_json,
)
from ..contracts.checks import (
    require_int,
    require_keys,
    require_mapping,
    require_number,
    require_relative_posix_path,
    require_sequence,
    require_sha256_hex,
    require_str,
)
from ..labels import LabelConfig, LabelError, read_wav
from .errors import DatasetError
from .split import (
    GROUP_NAMES,
    SPLITS,
    assign_splits,
    connected_components,
    cross_split_assets,
    mark_oversized,
    normalize_ratios,
)

#: Version of the dataset index document; unknown versions are rejected.
INDEX_VERSION = "dataset-index-v1"

MANIFEST_FILENAME = "manifest.json"

#: Default split ratios.  Groups are never broken up to reach them.
DEFAULT_RATIOS: Mapping[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}

DEFAULT_DURATION_TOLERANCE_SECONDS = 0.01

_SAMPLE_KEYS = (
    "sample_id",
    "path",
    "split",
    "sample_rate",
    "duration_seconds",
    "track_start_seconds",
    "groups",
    "source_ids",
    "labels",
    "controls",
    "audio",
    "content_sha256",
    "usable",
    "warnings",
    "sample_sha256",
)

_LABEL_KEYS = (
    "metadata_path",
    "arrays_path",
    "center_count",
    "valid_count",
    "hop_seconds",
    "center_window_seconds",
    "label_version",
    "label_config_sha256",
)


def sha256_file(path: str | Path) -> str:
    """Lower-case sha256 hex digest of a file's bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_record_sha256(payload: Mapping[str, Any]) -> str:
    """Canonical sha256 digest of one index record payload."""

    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def portable_data_root(data_root: str | Path, index_dir: str | Path | None) -> str | None:
    """Data root as a POSIX path relative to the index directory, or ``None``.

    ``None`` means the relationship cannot be expressed portably (different
    drives on Windows, or no index directory known); loading then requires an
    explicit data root instead of guessing an absolute path.
    """

    if index_dir is None:
        return None
    root = Path(data_root).resolve()
    base = Path(index_dir).resolve()
    try:
        relative = os.path.relpath(root, base)
    except ValueError:
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive:
        return None
    return candidate.as_posix()


def discover_samples(data_root: str | Path) -> tuple[str, ...]:
    """Sorted data-root-relative POSIX paths of sample directories.

    A sample directory is any directory containing ``manifest.json``.  The data
    root itself must not be a sample directory: relative sample paths are what
    keep the index portable.
    """

    root = Path(data_root)
    if not root.is_dir():
        raise DatasetError(f"data root {root}: expected a directory")
    if (root / MANIFEST_FILENAME).is_file():
        raise DatasetError(
            f"data root {root} is itself a sample directory; pass the parent directory "
            "that contains sample directories"
        )
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        if MANIFEST_FILENAME in filenames:
            found.append(Path(dirpath).relative_to(root).as_posix())
    return tuple(sorted(found))


@dataclass(frozen=True)
class SampleEntry:
    """One validated labeled sample inside the index.

    ``content_sha256`` holds the manifest's recorded digests (mix, stems,
    sources, controls and both label files).  ``sample_sha256`` covers the whole
    record, so a modified index is detected at parse time; comparing the record
    against freshly scanned inputs detects changed files.
    """

    sample_id: str
    path: str
    split: str
    sample_rate: int
    duration_seconds: float
    track_start_seconds: float
    groups: Mapping[str, Any]
    source_ids: tuple[str, ...]
    labels: Mapping[str, Any]
    controls: Mapping[str, Any]
    audio: Mapping[str, Any]
    content_sha256: Mapping[str, str]
    usable: bool
    warnings: tuple[str, ...] = ()
    sample_sha256: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        groups = require_mapping(self.groups, "sample.groups")
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "split": self.split,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "track_start_seconds": self.track_start_seconds,
            "groups": {
                "composition": groups["composition"],
                "preset": list(groups["preset"]),
                "sample_origin": list(groups["sample_origin"]),
            },
            "source_ids": list(self.source_ids),
            "labels": _plain(self.labels),
            "controls": _plain(self.controls),
            "audio": _plain(self.audio),
            "content_sha256": {key: self.content_sha256[key] for key in sorted(self.content_sha256)},
            "usable": bool(self.usable),
            "warnings": list(self.warnings),
            "sample_sha256": self.sample_sha256,
        }

    def canonical_payload(self) -> dict[str, Any]:
        payload = self.to_json_dict()
        payload.pop("sample_sha256")
        return payload

    def record_sha256(self) -> str:
        return dataset_record_sha256(self.canonical_payload())

    @classmethod
    def from_json_dict(cls, data: Any, path: str = "samples[]") -> SampleEntry:
        try:
            mapping = require_mapping(data, path)
            require_keys(mapping, _SAMPLE_KEYS, path)
            sample_id = require_str(mapping["sample_id"], f"{path}.sample_id")
            sample_path = require_relative_posix_path(mapping["path"], f"{path}.path")
            split = require_str(mapping["split"], f"{path}.split")
            if split not in SPLITS:
                raise ContractError(
                    f"{path}.split: expected one of {list(SPLITS)}, got {split!r}"
                )
            sample_rate = require_int(mapping["sample_rate"], f"{path}.sample_rate", minimum=1)
            duration = require_number(
                mapping["duration_seconds"],
                f"{path}.duration_seconds",
                minimum=0.0,
                strict_minimum=True,
            )
            track_start = require_number(
                mapping["track_start_seconds"], f"{path}.track_start_seconds", minimum=0.0
            )
            groups = _check_groups(mapping["groups"], f"{path}.groups")
            source_ids = _string_tuple(mapping["source_ids"], f"{path}.source_ids")
            labels = _check_labels(mapping["labels"], f"{path}.labels")
            controls = _check_controls(mapping["controls"], f"{path}.controls")
            audio = _check_audio(mapping["audio"], f"{path}.audio")
            content = _check_content_hashes(mapping["content_sha256"], f"{path}.content_sha256")
            usable = mapping["usable"]
            if not isinstance(usable, bool):
                raise ContractError(
                    f"{path}.usable: expected a boolean, got {type(usable).__name__}"
                )
            warnings_raw = require_sequence(mapping["warnings"], f"{path}.warnings")
            warnings = tuple(
                require_str(value, f"{path}.warnings[{position}]")
                for position, value in enumerate(warnings_raw)
            )
            declared_sha = require_sha256_hex(mapping["sample_sha256"], f"{path}.sample_sha256")
        except ContractError as exc:
            raise DatasetError(f"dataset index {path}: {exc}") from exc

        entry = cls(
            sample_id=sample_id,
            path=sample_path,
            split=split,
            sample_rate=sample_rate,
            duration_seconds=duration,
            track_start_seconds=track_start,
            groups=groups,
            source_ids=source_ids,
            labels=labels,
            controls=controls,
            audio=audio,
            content_sha256=content,
            usable=usable,
            warnings=warnings,
            sample_sha256=declared_sha,
        )
        actual_sha = entry.record_sha256()
        if actual_sha != declared_sha:
            raise DatasetError(
                f"dataset index {path}: sample_sha256 mismatch for sample {sample_id!r} "
                "(the record was modified or is stale)"
            )
        return entry


@dataclass(frozen=True)
class DatasetVerification:
    """Result of re-checking an index against its data root."""

    checked_samples: int
    checked_files: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise DatasetError(
                "dataset verification failed:\n"
                + "\n".join(f"  - {error}" for error in self.errors)
            )


@dataclass(frozen=True)
class DatasetIndex:
    """A validated, split dataset index document.

    ``samples`` follow data-root-relative path order (``discover_samples``
    sorts); each sample's ``split`` is already assigned.  ``summary`` carries
    the requested/actual ratios, empty splits, oversized connected components
    and leak evidence.
    """

    plan: Mapping[str, Any]
    summary: Mapping[str, Any]
    components: tuple[Mapping[str, Any], ...]
    samples: tuple[SampleEntry, ...]
    layout: Mapping[str, Any] = field(
        default_factory=lambda: {"data_root": None, "sample_path_base": "data_root"}
    )
    index_version: str = INDEX_VERSION

    def samples_for_split(self, split: str) -> tuple[SampleEntry, ...]:
        if split not in SPLITS:
            raise DatasetError(f"split: expected one of {list(SPLITS)}, got {split!r}")
        return tuple(
            sorted(
                (entry for entry in self.samples if entry.split == split),
                key=lambda entry: entry.sample_id,
            )
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "index_version": self.index_version,
            "plan": _plain(self.plan),
            "layout": _plain(self.layout),
            "summary": _plain(self.summary),
            "components": [_plain(component) for component in self.components],
            "samples": [entry.to_json_dict() for entry in self.samples],
        }

    def save(self, path: str | Path) -> Path:
        return dump_json(path, self.to_json_dict())

    @classmethod
    def from_json_dict(cls, data: Any, path: str = "dataset index") -> DatasetIndex:
        try:
            mapping = require_mapping(data, path)
            require_keys(
                mapping,
                ("index_version", "plan", "layout", "summary", "components", "samples"),
                path,
            )
            version = require_str(mapping["index_version"], f"{path}.index_version")
            if version != INDEX_VERSION:
                raise DatasetError(
                    f"{path}.index_version: unsupported version {version!r}; "
                    f"expected {INDEX_VERSION!r}. Rebuild the index."
                )
            plan_raw = require_mapping(mapping["plan"], f"{path}.plan")
            require_keys(plan_raw, ("seed", "ratios", "slots", "window_seconds"), f"{path}.plan")
            plan = {
                "seed": require_int(plan_raw["seed"], f"{path}.plan.seed", minimum=0),
                "ratios": _check_stored_ratios(plan_raw["ratios"], f"{path}.plan.ratios"),
                "slots": require_int(plan_raw["slots"], f"{path}.plan.slots", minimum=1),
                "window_seconds": require_number(
                    plan_raw["window_seconds"],
                    f"{path}.plan.window_seconds",
                    minimum=0.0,
                    strict_minimum=True,
                ),
            }
            layout = _check_layout(mapping["layout"], f"{path}.layout")
            summary = _check_summary(mapping["summary"], f"{path}.summary")
            components_raw = require_sequence(mapping["components"], f"{path}.components")
            components = tuple(
                _check_component(raw, f"{path}.components[{position}]")
                for position, raw in enumerate(components_raw)
            )
            samples_raw = require_sequence(mapping["samples"], f"{path}.samples")
            samples = tuple(
                SampleEntry.from_json_dict(raw, f"{path}.samples[{position}]")
                for position, raw in enumerate(samples_raw)
            )
            if summary["sample_count"] != len(samples):
                raise ContractError(
                    f"{path}.summary.sample_count: {summary['sample_count']} does not "
                    f"match {len(samples)} sample record(s)"
                )
            actual_counts = {split: 0 for split in SPLITS}
            for sample in samples:
                actual_counts[sample.split] += 1
            if actual_counts != summary["split_counts"]:
                raise ContractError(
                    f"{path}.summary.split_counts: {summary['split_counts']} does not match "
                    f"the sample records {actual_counts}"
                )
            sample_ids = {sample.sample_id for sample in samples}
            component_size = 0
            for component in components:
                component_size += component["size"]
                unknown = [
                    sample_id
                    for sample_id in component["sample_ids"]
                    if sample_id not in sample_ids
                ]
                if unknown:
                    raise ContractError(
                        f"{path}.components[{component['component_id']!r}]: unknown "
                        f"sample_id(s): {', '.join(unknown)}"
                    )
            if component_size != len(samples):
                raise ContractError(
                    f"{path}.components: sizes sum to {component_size} but there are "
                    f"{len(samples)} sample record(s)"
                )
        except ContractError as exc:
            raise DatasetError(f"{path}: {exc}") from exc
        return cls(
            plan=plan,
            summary=summary,
            components=components,
            samples=samples,
            layout=layout,
            index_version=version,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        data_root: str | Path | None = None,
        verify_files: bool | None = None,
        duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    ) -> DatasetIndex:
        """Load an index; by default verify files when a data root is resolvable.

        * ``verify_files=None`` (default): verify digests and inputs when the
          layout or ``data_root`` provides a root, otherwise only parse/record
          checks run.
        * ``verify_files=True``: require a resolvable data root and verify.
        * ``verify_files=False``: parse/record checks only (``sample_sha256`` is
          still enforced).
        """

        source = Path(path)
        try:
            data = load_json(source)
        except ContractError as exc:
            raise DatasetError(f"{source}: invalid dataset index JSON: {exc}") from exc
        index = cls.from_json_dict(data, source.name)

        resolved: Path | None = None
        if data_root is not None:
            resolved = Path(data_root)
        else:
            try:
                resolved = index.resolve_data_root(index_path=source)
            except DatasetError:
                resolved = None
        if verify_files is True and resolved is None:
            raise DatasetError(
                f"{source}: verify_files=True but the index records no portable data root; "
                "pass data_root explicitly"
            )
        if verify_files is not False and resolved is not None:
            verification = verify_dataset(
                index,
                resolved,
                duration_tolerance_seconds=duration_tolerance_seconds,
            )
            if not verification.ok:
                raise DatasetError(
                    f"{source}: dataset verification failed:\n"
                    + "\n".join(f"  - {error}" for error in verification.errors)
                )
        return index

    def resolve_data_root(
        self, *, index_path: str | Path | None = None, data_root: str | Path | None = None
    ) -> Path:
        """Resolve the data root from an explicit path or the portable layout."""

        if data_root is not None:
            return Path(data_root)
        recorded = self.layout.get("data_root")
        if not recorded:
            raise DatasetError(
                "dataset index does not record a portable data root; pass data_root explicitly"
            )
        if index_path is None:
            raise DatasetError(
                "dataset index needs the index path to resolve its recorded data root"
            )
        base = Path(index_path).resolve().parent
        return (base / PurePosixPath(str(recorded))).resolve()


def scan_sample(
    sample_dir: str | Path,
    *,
    data_root: str | Path,
    slots: int = DEFAULT_SLOTS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    verify_digests: bool = True,
    split: str = "",
) -> SampleEntry:
    """Validate one labeled sample directory and build its index record.

    Checks, in order: manifest stage/schema, every recorded sha256, label
    metadata/arrays, source column order, label config version and center
    window, audio sample rate/duration, center-time range and slot capacity.
    Every failure names the sample.
    """

    root = Path(data_root)
    directory = Path(sample_dir)
    try:
        relative = directory.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DatasetError(
            f"{directory}: sample directory is not inside data root {root}"
        ) from exc
    label = f"sample '{relative}'"

    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DatasetError(f"{label}: missing {MANIFEST_FILENAME}")
    try:
        manifest = SampleManifest.load(manifest_path)
    except (ContractError, OSError) as exc:
        raise DatasetError(f"{label}: invalid {MANIFEST_FILENAME}: {exc}") from exc
    if manifest.stage != "labeled":
        raise DatasetError(
            f"{label}: manifest stage is {manifest.stage!r}, expected 'labeled'; "
            "run scripts/label_sample.py first"
        )
    if not manifest.activity_metadata_path or not manifest.activity_arrays_path:
        raise DatasetError(f"{label}: labeled manifest is missing activity label paths")

    if verify_digests:
        for artifact in sorted(manifest.content_sha256):
            _verify_artifact(directory, label, artifact, manifest.content_sha256[artifact])

    metadata_path = _resolve_inside(
        directory, manifest.activity_metadata_path, label, "activity_metadata_path"
    )
    arrays_path = _resolve_inside(
        directory, manifest.activity_arrays_path, label, "activity_arrays_path"
    )
    try:
        metadata_payload = load_json(metadata_path)
    except (ContractError, OSError) as exc:
        raise DatasetError(f"{label}: cannot read {manifest.activity_metadata_path}: {exc}") from exc
    if not isinstance(metadata_payload, Mapping):
        raise DatasetError(
            f"{label}: {manifest.activity_metadata_path} must be a JSON object"
        )
    declared_arrays = metadata_payload.get("arrays_path")
    if not isinstance(declared_arrays, str):
        raise DatasetError(
            f"{label}: {manifest.activity_metadata_path}.arrays_path must be a relative path string"
        )
    declared_path = _resolve_inside(
        metadata_path.parent, declared_arrays, label, "arrays_path"
    )
    if declared_path != arrays_path:
        raise DatasetError(
            f"{label}: manifest.activity_arrays_path {manifest.activity_arrays_path!r} and "
            f"the arrays_path {declared_arrays!r} declared by {manifest.activity_metadata_path} "
            "resolve to different files"
        )
    try:
        activity = ActivityData.load(metadata_path.parent, metadata_filename=metadata_path.name)
    except (ContractError, OSError) as exc:
        raise DatasetError(f"{label}: invalid activity labels: {exc}") from exc

    try:
        sources = SourceRegistry.load(_resolve_inside(directory, manifest.sources_path, label, "sources_path"))
    except (ContractError, OSError) as exc:
        raise DatasetError(f"{label}: invalid sources.json: {exc}") from exc
    if sources.sample_id is not None and sources.sample_id != manifest.sample_id:
        raise DatasetError(
            f"{label}: sources.sample_id {sources.sample_id!r} does not match manifest "
            f"sample_id {manifest.sample_id!r}"
        )
    if set(manifest.stem_paths) != set(sources.source_ids):
        raise DatasetError(
            f"{label}: manifest.stem_paths keys {sorted(manifest.stem_paths)} do not match "
            f"sources.json source_ids {list(sources.source_ids)}"
        )
    source_ids = sources.source_ids
    if not source_ids:
        raise DatasetError(
            f"{label}: sources.json declares no sources; every sample needs at least one stem"
        )
    if len(source_ids) > slots:
        raise DatasetError(
            f"{label}: {len(source_ids)} source(s) exceed the slots capacity K={slots}; "
            "silent sources still count. Increase --slots or exclude this sample."
        )
    if tuple(activity.source_ids) != source_ids:
        raise DatasetError(
            f"{label}: activity source columns {list(activity.source_ids)} do not match the "
            f"sources.json order {list(source_ids)}; labels are inconsistent"
        )
    if activity.sample_rate != manifest.sample_rate:
        raise DatasetError(
            f"{label}: activity sample_rate {activity.sample_rate} does not match manifest "
            f"{manifest.sample_rate}"
        )
    if activity.sample_id is not None and activity.sample_id != manifest.sample_id:
        raise DatasetError(
            f"{label}: activity.sample_id {activity.sample_id!r} does not match manifest "
            f"sample_id {manifest.sample_id!r}"
        )

    label_params = activity.label_params
    if not isinstance(label_params, Mapping):
        raise DatasetError(
            f"{label}: activity.json has no label_params snapshot; relabel with "
            "scripts/label_sample.py so the window and thresholds are auditable"
        )
    try:
        label_config = LabelConfig.from_label_params(label_params)
    except (LabelError, ContractError) as exc:
        raise DatasetError(f"{label}: unsupported activity label config: {exc}") from exc
    center_window = float(label_config.center_window_seconds)
    if abs(center_window - window_seconds) > 1e-9:
        raise DatasetError(
            f"{label}: labels were produced with center_window_seconds={center_window} but the "
            f"dataset plan uses window_seconds={window_seconds}; rebuild the index with "
            f"--window-seconds {center_window}"
        )

    track_start = float(manifest.track_start_seconds)
    duration = float(manifest.duration_seconds)
    end = track_start + duration
    tolerance = 1e-6
    if activity.center_times.size:
        first = float(activity.center_times[0])
        last = float(activity.center_times[-1])
        if first < track_start - tolerance or last > end + tolerance:
            raise DatasetError(
                f"{label}: activity center times [{first}, {last}] lie outside the rendered span "
                f"[{track_start}, {end}]"
            )

    mix_path = _resolve_inside(directory, manifest.mix_path, label, "mix_path")
    try:
        mix = read_wav(mix_path)
    except (LabelError, OSError) as exc:
        raise DatasetError(f"{label}: cannot read {manifest.mix_path}: {exc}") from exc
    if mix.sample_rate != manifest.sample_rate:
        raise DatasetError(
            f"{label}: mix sample_rate {mix.sample_rate} does not match manifest "
            f"{manifest.sample_rate}"
        )
    if abs(mix.duration_seconds - duration) > duration_tolerance_seconds:
        raise DatasetError(
            f"{label}: mix duration {mix.duration_seconds:.6f} s differs from manifest "
            f"{duration:.6f} s by more than {duration_tolerance_seconds} s"
        )

    try:
        controls = Controls.load(_resolve_inside(directory, manifest.controls_path, label, "controls_path"))
    except (ContractError, OSError) as exc:
        raise DatasetError(f"{label}: invalid controls.json: {exc}") from exc
    if controls.sample_id is not None and controls.sample_id != manifest.sample_id:
        raise DatasetError(
            f"{label}: controls.sample_id {controls.sample_id!r} does not match manifest "
            f"sample_id {manifest.sample_id!r}"
        )
    controls.validate_source_references(source_ids)

    valid_count = int(activity.valid.sum())
    warnings: list[str] = []
    if valid_count == 0:
        warnings.append("no valid center windows (activity.valid is all False)")
    if not bool(activity.activity.any()):
        warnings.append("all center activity labels are 0 for every source")

    entry = SampleEntry(
        sample_id=manifest.sample_id,
        path=relative,
        split=split,
        sample_rate=int(manifest.sample_rate),
        duration_seconds=duration,
        track_start_seconds=track_start,
        groups=_check_groups(manifest.groups, f"{label}.groups"),
        source_ids=tuple(source_ids),
        labels={
            "metadata_path": manifest.activity_metadata_path,
            "arrays_path": manifest.activity_arrays_path,
            "center_count": int(activity.center_times.size),
            "valid_count": valid_count,
            "hop_seconds": activity.hop_seconds,
            "center_window_seconds": center_window,
            "label_version": label_params.get("version"),
            "label_config_sha256": label_params.get("sha256"),
        },
        controls=_controls_evidence(controls, source_ids),
        audio={
            "mix_path": manifest.mix_path,
            "sample_rate": int(mix.sample_rate),
            "channels": int(mix.channels),
            "frames": int(mix.frames),
            "duration_seconds": float(mix.duration_seconds),
        },
        content_sha256=dict(manifest.content_sha256),
        usable=bool(valid_count > 0),
        warnings=tuple(warnings),
    )
    return replace(entry, sample_sha256=entry.record_sha256())


def build_dataset_index(
    data_root: str | Path,
    *,
    seed: int,
    ratios: Mapping[str, Any] | Sequence[Any] = DEFAULT_RATIOS,
    slots: int = DEFAULT_SLOTS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    verify_digests: bool = True,
    index_dir: str | Path | None = None,
) -> DatasetIndex:
    """Scan, validate, group, split and index every labeled sample under a root."""

    root = Path(data_root)
    if not root.is_dir():
        raise DatasetError(
            f"data root {root}: expected an existing directory containing labeled samples"
        )
    seed_value = _require_seed(seed)
    slots_value = _require_positive_int(slots, "slots")
    window_value = _require_positive_number(window_seconds, "window_seconds")
    tolerance = _require_non_negative_number(
        duration_tolerance_seconds, "duration_tolerance_seconds"
    )
    ratios_value = normalize_ratios(ratios)

    entries = tuple(
        scan_sample(
            root / relative,
            data_root=root,
            slots=slots_value,
            window_seconds=window_value,
            duration_tolerance_seconds=tolerance,
            verify_digests=verify_digests,
        )
        for relative in discover_samples(root)
    )
    duplicates: dict[str, list[str]] = {}
    for entry in entries:
        duplicates.setdefault(entry.sample_id, []).append(entry.path)
    repeated = {key: value for key, value in duplicates.items() if len(value) > 1}
    if repeated:
        details = "; ".join(
            f"{sample_id!r}: {', '.join(paths)}" for sample_id, paths in sorted(repeated.items())
        )
        raise DatasetError(f"duplicate sample_id in data root: {details}")

    components = mark_oversized(connected_components(entries), ratios_value)
    assignment = assign_splits(components, ratios=ratios_value, seed=seed_value)
    split_entries = tuple(
        replace(entry, split=assignment[entry.sample_id]) for entry in entries
    )
    split_entries = tuple(
        replace(entry, sample_sha256=entry.record_sha256()) for entry in split_entries
    )
    component_payload = tuple(
        component.to_json_dict(split=assignment[component.sample_ids[0]])
        for component in components
    )

    counts = {split: 0 for split in SPLITS}
    for entry in split_entries:
        counts[entry.split] += 1
    total = len(split_entries)
    requested = {split: ratios_value[split] for split in SPLITS}
    actual = {split: (counts[split] / total if total else 0.0) for split in SPLITS}
    empty_splits = [split for split in SPLITS if counts[split] == 0]
    oversized = [component for component in components if component.oversized]
    cross = cross_split_assets(split_entries)
    if any(cross.values()):
        raise DatasetError(
            "internal error: split assignment leaked assets across splits: "
            + ", ".join(f"{name}={count}" for name, count in cross.items())
        )

    warnings: list[str] = []
    if total == 0:
        warnings.append(
            "data root contains no labeled samples (manifest.json with stage='labeled')"
        )
    for component in oversized:
        warnings.append(
            f"connected component {component.component_id} has {component.size} sample(s) and "
            f"cannot fit any split target; it was kept whole in one split "
            f"(largest component target {total * max(requested.values()):.3f})"
        )
    for split in empty_splits:
        if total:
            warnings.append(
                f"split {split!r} is empty; requested ratio {requested[split]:.3f} cannot be "
                "met without splitting a connected group"
            )
    for entry in split_entries:
        for warning in entry.warnings:
            warnings.append(f"sample {entry.sample_id!r}: {warning}")

    summary = {
        "sample_count": total,
        "split_counts": counts,
        "split_ratios_requested": requested,
        "split_ratios_actual": actual,
        "component_count": len(components),
        "largest_component_size": max((component.size for component in components), default=0),
        "empty_splits": empty_splits,
        "oversized_components": [
            {
                "component_id": component.component_id,
                "size": component.size,
                "sample_ids": list(component.sample_ids),
                "assets": {name: list(component.assets[name]) for name in GROUP_NAMES},
            }
            for component in oversized
        ],
        "warnings": warnings,
        "cross_split_assets": cross,
        "leak_free": True,
    }
    plan = {
        "seed": seed_value,
        "ratios": requested,
        "slots": slots_value,
        "window_seconds": window_value,
    }
    layout = {
        "data_root": portable_data_root(root, index_dir),
        "sample_path_base": "data_root",
    }
    return DatasetIndex(
        plan=plan,
        summary=summary,
        components=component_payload,
        samples=split_entries,
        layout=layout,
    )


def verify_dataset(
    index: DatasetIndex,
    data_root: str | Path,
    *,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    verify_digests: bool = True,
) -> DatasetVerification:
    """Re-scan inputs and re-check the index without changing it.

    Errors are collected (not raised one by one) so a broken dataset can be
    inspected in one pass; each error names its sample.  ``verify_digests``
    re-hashes every recorded artifact, which is the only reliable way to detect
    a changed mixed/stem file.
    """

    errors: list[str] = []
    root = Path(data_root)
    if not root.is_dir():
        errors.append(f"data root {root} is not a directory")
    checked_files = 0
    seen_ids: dict[str, str] = {}
    seen_paths: dict[str, str] = {}
    for entry in index.samples:
        previous = seen_ids.get(entry.sample_id)
        if previous is not None:
            errors.append(
                f"duplicate sample_id {entry.sample_id!r} in {previous} and {entry.path}"
            )
        else:
            seen_ids[entry.sample_id] = entry.path
        if entry.path in seen_paths:
            errors.append(
                f"duplicate sample path {entry.path!r} ({seen_paths[entry.path]!r} and "
                f"{entry.sample_id!r})"
            )
        else:
            seen_paths[entry.path] = entry.sample_id
        if entry.split not in SPLITS:
            errors.append(
                f"sample '{entry.sample_id}' ({entry.path}): unknown split {entry.split!r}"
            )
        if entry.record_sha256() != entry.sample_sha256:
            errors.append(
                f"sample '{entry.sample_id}' ({entry.path}): sample_sha256 mismatch; "
                "the index record was modified"
            )
        if not root.is_dir():
            continue
        try:
            fresh = scan_sample(
                root / PurePosixPath(entry.path),
                data_root=root,
                slots=index.plan["slots"],
                window_seconds=index.plan["window_seconds"],
                duration_tolerance_seconds=duration_tolerance_seconds,
                verify_digests=verify_digests,
            )
        except DatasetError as exc:
            errors.append(str(exc))
            continue
        checked_files += len(entry.content_sha256)
        fresh_payload = _json_normalize(replace(fresh, split=entry.split).canonical_payload())
        recorded = _json_normalize(entry.canonical_payload())
        if fresh_payload != recorded:
            differing = sorted(
                key
                for key in set(fresh_payload) | set(recorded)
                if fresh_payload.get(key) != recorded.get(key)
            )
            errors.append(
                f"sample '{entry.sample_id}' ({entry.path}): index record no longer matches "
                f"the inputs; differing field(s): {', '.join(differing)}"
            )

    cross = cross_split_assets(index.samples)
    for name, count in cross.items():
        if count:
            errors.append(
                f"leakage: {count} {name} asset(s) appear in more than one split"
            )
    return DatasetVerification(
        checked_samples=len(index.samples),
        checked_files=checked_files,
        errors=tuple(errors),
    )


def _verify_artifact(directory: Path, label: str, relative: str, expected: str) -> None:
    path = _resolve_inside(directory, relative, label, f"content_sha256[{relative!r}]")
    if not path.is_file():
        raise DatasetError(f"{label}: content_sha256[{relative!r}]: file not found")
    actual = sha256_file(path)
    if actual != expected:
        raise DatasetError(
            f"{label}: content_sha256[{relative!r}]: expected {expected}, got {actual}"
        )


def _resolve_inside(directory: Path, relative: Any, label: str, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise DatasetError(f"{label}: {field} must be a non-empty relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive:
        raise DatasetError(f"{label}: {field} must be relative, got {relative!r}")
    base = directory.resolve()
    resolved = (directory / candidate).resolve()
    if not resolved.is_relative_to(base):
        raise DatasetError(f"{label}: {field} escapes the sample directory: {relative!r}")
    return resolved


def _controls_evidence(controls: Controls, source_ids: Sequence[str]) -> dict[str, Any]:
    """Note/event counts used as non-label evidence (MIDI never changes labels)."""

    per_source: dict[str, dict[str, Any]] = {
        source_id: {"note_on_count": 0, "note_min": None, "note_max": None}
        for source_id in source_ids
    }
    note_on_count = 0
    note_min: int | None = None
    note_max: int | None = None
    for event in controls.events:
        if event.event_type != "note_on":
            continue
        note = event.data.get("note")
        if isinstance(note, bool) or not isinstance(note, int):
            continue
        note_on_count += 1
        note_min = note if note_min is None else min(note_min, note)
        note_max = note if note_max is None else max(note_max, note)
        row = per_source[event.source_id]
        row["note_on_count"] += 1
        row["note_min"] = note if row["note_min"] is None else min(row["note_min"], note)
        row["note_max"] = note if row["note_max"] is None else max(row["note_max"], note)
    return {
        "event_count": len(controls.events),
        "note_on_count": note_on_count,
        "note_min": note_min,
        "note_max": note_max,
        "sources": per_source,
    }


def _plain(value: Any) -> Any:
    """Recursively convert tuples/mappings into JSON document values."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(_plain(value), sort_keys=True, separators=(",", ":")))


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    raw = require_sequence(value, path)
    values = tuple(require_str(item, f"{path}[{position}]") for position, item in enumerate(raw))
    if len(set(values)) != len(values):
        raise ContractError(f"{path}: values must be unique")
    return values


def _check_groups(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(mapping, GROUP_NAMES, path)
    composition = require_str(mapping["composition"], f"{path}.composition")
    preset = _string_tuple(mapping["preset"], f"{path}.preset")
    origin = _string_tuple(mapping["sample_origin"], f"{path}.sample_origin")
    return {
        "composition": composition,
        "preset": tuple(sorted(preset)),
        "sample_origin": tuple(sorted(origin)),
    }


def _check_content_hashes(value: Any, path: str) -> dict[str, str]:
    mapping = require_mapping(value, path)
    if not mapping:
        raise ContractError(f"{path}: at least one artifact digest is required")
    result: dict[str, str] = {}
    for key, digest in mapping.items():
        relative = require_relative_posix_path(key, f"{path} key")
        result[relative] = require_sha256_hex(digest, f"{path}[{relative!r}]")
    return result


def _check_labels(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(mapping, _LABEL_KEYS, path)
    center_count = require_int(mapping["center_count"], f"{path}.center_count", minimum=0)
    valid_count = require_int(mapping["valid_count"], f"{path}.valid_count", minimum=0)
    if valid_count > center_count:
        raise ContractError(
            f"{path}.valid_count: {valid_count} exceeds center_count {center_count}"
        )
    hop = mapping["hop_seconds"]
    if hop is not None:
        hop = require_number(hop, f"{path}.hop_seconds", minimum=0.0, strict_minimum=True)
    config_sha = mapping["label_config_sha256"]
    if config_sha is not None:
        config_sha = require_sha256_hex(config_sha, f"{path}.label_config_sha256")
    return {
        "metadata_path": require_relative_posix_path(mapping["metadata_path"], f"{path}.metadata_path"),
        "arrays_path": require_relative_posix_path(mapping["arrays_path"], f"{path}.arrays_path"),
        "center_count": center_count,
        "valid_count": valid_count,
        "hop_seconds": hop,
        "center_window_seconds": require_number(
            mapping["center_window_seconds"],
            f"{path}.center_window_seconds",
            minimum=0.0,
            strict_minimum=True,
        ),
        "label_version": require_str(mapping["label_version"], f"{path}.label_version"),
        "label_config_sha256": config_sha,
    }


def _check_controls(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(mapping, ("event_count", "note_on_count", "note_min", "note_max", "sources"), path)
    event_count = require_int(mapping["event_count"], f"{path}.event_count", minimum=0)
    note_on_count = require_int(mapping["note_on_count"], f"{path}.note_on_count", minimum=0)
    if note_on_count > event_count:
        raise ContractError(
            f"{path}.note_on_count: {note_on_count} exceeds event_count {event_count}"
        )
    for name in ("note_min", "note_max"):
        if mapping[name] is not None:
            require_int(mapping[name], f"{path}.{name}")
    sources_raw = require_mapping(mapping["sources"], f"{path}.sources")
    sources: dict[str, Any] = {}
    for source_id, row in sources_raw.items():
        require_str(source_id, f"{path}.sources key")
        row_mapping = require_mapping(row, f"{path}.sources[{source_id!r}]")
        require_keys(row_mapping, ("note_on_count", "note_min", "note_max"), f"{path}.sources[{source_id!r}]")
        require_int(row_mapping["note_on_count"], f"{path}.sources[{source_id!r}].note_on_count", minimum=0)
        for name in ("note_min", "note_max"):
            if row_mapping[name] is not None:
                require_int(row_mapping[name], f"{path}.sources[{source_id!r}].{name}")
        sources[source_id] = {
            "note_on_count": row_mapping["note_on_count"],
            "note_min": row_mapping["note_min"],
            "note_max": row_mapping["note_max"],
        }
    return {
        "event_count": event_count,
        "note_on_count": note_on_count,
        "note_min": mapping["note_min"],
        "note_max": mapping["note_max"],
        "sources": sources,
    }


def _check_audio(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(
        mapping,
        ("mix_path", "sample_rate", "channels", "frames", "duration_seconds"),
        path,
    )
    return {
        "mix_path": require_relative_posix_path(mapping["mix_path"], f"{path}.mix_path"),
        "sample_rate": require_int(mapping["sample_rate"], f"{path}.sample_rate", minimum=1),
        "channels": require_int(mapping["channels"], f"{path}.channels", minimum=1),
        "frames": require_int(mapping["frames"], f"{path}.frames", minimum=1),
        "duration_seconds": require_number(
            mapping["duration_seconds"], f"{path}.duration_seconds", minimum=0.0, strict_minimum=True
        ),
    }


def _check_stored_ratios(value: Any, path: str) -> dict[str, float]:
    mapping = require_mapping(value, path)
    require_keys(mapping, SPLITS, path)
    ratios: dict[str, float] = {}
    for split in SPLITS:
        ratios[split] = require_number(mapping[split], f"{path}.{split}", minimum=0.0)
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ContractError(f"{path}: values must sum to 1, got {total}")
    return ratios


def _check_layout(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(mapping, ("data_root", "sample_path_base"), path)
    data_root = mapping["data_root"]
    if data_root is not None:
        if not isinstance(data_root, str) or not data_root:
            raise ContractError(
                f"{path}.data_root: expected a non-empty string or null, got {data_root!r}"
            )
        if "\\" in data_root or data_root.startswith("/") or Path(data_root).drive:
            raise ContractError(
                f"{path}.data_root: expected a relative POSIX path, got {data_root!r}"
            )
    base = require_str(mapping["sample_path_base"], f"{path}.sample_path_base")
    if base != "data_root":
        raise ContractError(
            f"{path}.sample_path_base: expected 'data_root', got {base!r}"
        )
    return {"data_root": data_root, "sample_path_base": base}


def _check_summary(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(
        mapping,
        (
            "sample_count",
            "split_counts",
            "split_ratios_requested",
            "split_ratios_actual",
            "component_count",
            "largest_component_size",
            "empty_splits",
            "oversized_components",
            "warnings",
            "cross_split_assets",
            "leak_free",
        ),
        path,
    )
    sample_count = require_int(mapping["sample_count"], f"{path}.sample_count", minimum=0)
    counts_raw = require_mapping(mapping["split_counts"], f"{path}.split_counts")
    require_keys(counts_raw, SPLITS, f"{path}.split_counts")
    split_counts = {
        split: require_int(counts_raw[split], f"{path}.split_counts.{split}", minimum=0)
        for split in SPLITS
    }
    if sum(split_counts.values()) != sample_count:
        raise ContractError(
            f"{path}.split_counts: values must sum to sample_count {sample_count}"
        )
    for name in ("split_ratios_requested", "split_ratios_actual"):
        ratios_raw = require_mapping(mapping[name], f"{path}.{name}")
        require_keys(ratios_raw, SPLITS, f"{path}.{name}")
        for split in SPLITS:
            require_number(ratios_raw[split], f"{path}.{name}.{split}", minimum=0.0)
    empty_raw = require_sequence(mapping["empty_splits"], f"{path}.empty_splits")
    empty_splits = list(_string_tuple(empty_raw, f"{path}.empty_splits"))
    for split in empty_splits:
        if split not in SPLITS:
            raise ContractError(f"{path}.empty_splits: unknown split {split!r}")
    oversized_raw = require_sequence(mapping["oversized_components"], f"{path}.oversized_components")
    oversized = [
        _check_oversized_component(raw, f"{path}.oversized_components[{position}]")
        for position, raw in enumerate(oversized_raw)
    ]
    warnings_raw = require_sequence(mapping["warnings"], f"{path}.warnings")
    warnings = [
        require_str(item, f"{path}.warnings[{position}]")
        for position, item in enumerate(warnings_raw)
    ]
    cross_raw = require_mapping(mapping["cross_split_assets"], f"{path}.cross_split_assets")
    require_keys(cross_raw, GROUP_NAMES, f"{path}.cross_split_assets")
    cross = {
        name: require_int(cross_raw[name], f"{path}.cross_split_assets.{name}", minimum=0)
        for name in GROUP_NAMES
    }
    leak_free = mapping["leak_free"]
    if not isinstance(leak_free, bool):
        raise ContractError(
            f"{path}.leak_free: expected a boolean, got {type(leak_free).__name__}"
        )
    return {
        "sample_count": sample_count,
        "split_counts": split_counts,
        "split_ratios_requested": {split: float(mapping["split_ratios_requested"][split]) for split in SPLITS},
        "split_ratios_actual": {split: float(mapping["split_ratios_actual"][split]) for split in SPLITS},
        "component_count": require_int(mapping["component_count"], f"{path}.component_count", minimum=0),
        "largest_component_size": require_int(
            mapping["largest_component_size"], f"{path}.largest_component_size", minimum=0
        ),
        "empty_splits": empty_splits,
        "oversized_components": oversized,
        "warnings": warnings,
        "cross_split_assets": cross,
        "leak_free": leak_free,
    }


def _check_oversized_component(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(mapping, ("component_id", "size", "sample_ids", "assets"), path)
    size = require_int(mapping["size"], f"{path}.size", minimum=0)
    sample_ids = _string_tuple(mapping["sample_ids"], f"{path}.sample_ids")
    if len(sample_ids) != size:
        raise ContractError(
            f"{path}.size: {size} does not match {len(sample_ids)} sample_ids"
        )
    assets_raw = require_mapping(mapping["assets"], f"{path}.assets")
    require_keys(assets_raw, GROUP_NAMES, f"{path}.assets")
    assets = {
        name: list(_string_tuple(assets_raw[name], f"{path}.assets.{name}"))
        for name in GROUP_NAMES
    }
    return {
        "component_id": require_str(mapping["component_id"], f"{path}.component_id"),
        "size": size,
        "sample_ids": list(sample_ids),
        "assets": assets,
    }


def _check_component(value: Any, path: str) -> dict[str, Any]:
    mapping = require_mapping(value, path)
    require_keys(mapping, ("component_id", "size", "sample_ids", "assets", "oversized", "split"), path)
    component = _check_oversized_component(mapping, path)
    oversized = mapping["oversized"]
    if not isinstance(oversized, bool):
        raise ContractError(
            f"{path}.oversized: expected a boolean, got {type(oversized).__name__}"
        )
    split = require_str(mapping["split"], f"{path}.split")
    if split not in SPLITS:
        raise ContractError(f"{path}.split: expected one of {list(SPLITS)}, got {split!r}")
    return {
        "component_id": component["component_id"],
        "size": component["size"],
        "sample_ids": component["sample_ids"],
        "assets": component["assets"],
        "oversized": oversized,
        "split": split,
    }


def _require_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(f"seed: expected an integer, got {type(value).__name__}")
    if value < 0:
        raise DatasetError(f"seed: must be >= 0, got {value}")
    return int(value)


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetError(f"{name}: expected an integer, got {type(value).__name__}")
    if value < 1:
        raise DatasetError(f"{name}: must be >= 1, got {value}")
    return int(value)


def _require_positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{name}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise DatasetError(f"{name}: must be finite and > 0, got {value!r}")
    return number


def _require_non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"{name}: expected a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise DatasetError(f"{name}: must be finite and >= 0, got {value!r}")
    return number


__all__ = [
    "DEFAULT_DURATION_TOLERANCE_SECONDS",
    "DEFAULT_RATIOS",
    "INDEX_VERSION",
    "MANIFEST_FILENAME",
    "DatasetIndex",
    "DatasetVerification",
    "SampleEntry",
    "build_dataset_index",
    "dataset_record_sha256",
    "discover_samples",
    "portable_data_root",
    "scan_sample",
    "sha256_file",
    "verify_dataset",
]
