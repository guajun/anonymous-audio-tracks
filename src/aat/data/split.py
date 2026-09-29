"""Group-connected split assignment with transitive leakage prevention.

Samples are linked when they share an asset inside one *independent* namespace:

``composition``
    A scalar composition id; one namespace value per sample.
``preset``
    A list of preset/timbre asset ids; an empty list contributes nothing.
``sample_origin``
    A list of sample-material ids; an empty list contributes nothing.

The namespaces never collide: ``preset`` id ``x`` and ``sample_origin`` id
``x`` are different assets.  Linking samples transitively (A-B by preset,
B-C by sample_origin) yields a connected component; a whole component is
assigned to exactly one split, so no shared asset can cross train/val/test.
Components that cannot fit into any split target are reported as oversized
instead of being split to hit a ratio.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .errors import DatasetError

#: The three evaluation splits, always emitted in this order.
SPLITS = ("train", "val", "test")

#: Group keys used for leakage-free grouping, in manifest order.
GROUP_NAMES = ("composition", "preset", "sample_origin")


@dataclass(frozen=True)
class Component:
    """A connected group of samples that must stay in one split."""

    component_id: str
    sample_ids: tuple[str, ...]
    size: int
    assets: Mapping[str, tuple[str, ...]]
    oversized: bool = False

    def to_json_dict(self, *, split: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "component_id": self.component_id,
            "size": self.size,
            "sample_ids": list(self.sample_ids),
            "assets": {name: list(self.assets[name]) for name in GROUP_NAMES},
            "oversized": self.oversized,
        }
        if split is not None:
            data["split"] = split
        return data


def asset_keys(groups: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Namespaced assets shared by a sample, ignoring empty asset lists."""

    keys: list[tuple[str, str]] = [("composition", groups["composition"])]
    for name in ("preset", "sample_origin"):
        keys.extend((name, value) for value in groups[name])
    return tuple(keys)


def connected_components(entries: Sequence[Any]) -> tuple[Component, ...]:
    """Union groups of samples that share any namespaced asset.

    ``entries`` only need ``sample_id`` and ``groups`` attributes.  The result
    is deterministic: samples are sorted by ``sample_id`` first, components by
    descending size and then first ``sample_id``, and ids are ``c0000``...
    """

    items = sorted(entries, key=lambda entry: entry.sample_id)
    parent = list(range(len(items)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[tuple[str, str], int] = {}
    for index, entry in enumerate(items):
        for key in asset_keys(entry.groups):
            previous = owner.get(key)
            if previous is None:
                owner[key] = index
            else:
                union(index, previous)

    buckets: dict[int, list[int]] = {}
    for index in range(len(items)):
        buckets.setdefault(find(index), []).append(index)

    components: list[Component] = []
    for members in buckets.values():
        sample_ids = tuple(sorted(items[index].sample_id for index in members))
        assets: dict[str, tuple[str, ...]] = {}
        for name in GROUP_NAMES:
            values: set[str] = set()
            for index in members:
                raw = items[index].groups[name]
                if name == "composition":
                    values.add(raw)
                else:
                    values.update(raw)
            assets[name] = tuple(sorted(values))
        components.append(
            Component(
                component_id="",
                sample_ids=sample_ids,
                size=len(members),
                assets=assets,
            )
        )
    components.sort(key=lambda component: (-component.size, component.sample_ids[0]))
    return tuple(
        replace(component, component_id=f"c{position:04d}")
        for position, component in enumerate(components)
    )


def mark_oversized(
    components: Sequence[Component], ratios: Mapping[str, float]
) -> tuple[Component, ...]:
    """Flag components larger than the largest split target.

    Such a component forces the actual split ratios away from the requested
    ones; it is kept whole and reported instead of being broken up.
    """

    total = sum(component.size for component in components)
    if total == 0:
        return tuple(components)
    max_target = total * max(ratios.values())
    return tuple(
        replace(component, oversized=component.size > max_target + 1e-12)
        for component in components
    )


def normalize_ratios(ratios: Mapping[str, Any] | Sequence[Any]) -> dict[str, float]:
    """Validate and normalize train/val/test ratios to a positive sum of 1."""

    if isinstance(ratios, Mapping):
        unknown = sorted(set(ratios) - set(SPLITS))
        missing = [split for split in SPLITS if split not in ratios]
        if unknown:
            raise DatasetError(f"ratios: unknown split(s): {', '.join(unknown)}")
        if missing:
            raise DatasetError(f"ratios: missing split(s): {', '.join(missing)}")
        raw = [ratios[split] for split in SPLITS]
    elif isinstance(ratios, Sequence) and not isinstance(ratios, (str, bytes)):
        if len(ratios) != len(SPLITS):
            raise DatasetError(
                "ratios: expected exactly three values for train, val, test; "
                f"got {len(ratios)}"
            )
        raw = list(ratios)
    else:
        raise DatasetError("ratios: expected a mapping or a sequence of three numbers")

    values: list[float] = []
    for split, value in zip(SPLITS, raw):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DatasetError(
                f"ratios.{split}: expected a number, got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise DatasetError(f"ratios.{split}: must be finite and >= 0, got {value!r}")
        values.append(number)
    total = sum(values)
    if total <= 0.0:
        raise DatasetError("ratios: at least one value must be > 0")
    return {split: value / total for split, value in zip(SPLITS, values)}


def assign_splits(
    components: Sequence[Component],
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole components to splits, returning ``sample_id -> split``.

    Components are processed largest first (ties by first ``sample_id``), each
    going to the split with the largest remaining deficit.  The explicit
    ``seed`` only breaks ties between splits with equal deficits, so the result
    is reproducible while still seed-dependent.
    """

    total = sum(component.size for component in components)
    targets = {split: total * ratios[split] for split in SPLITS}
    assigned = {split: 0 for split in SPLITS}
    rng = random.Random(seed)
    order = sorted(components, key=lambda component: (-component.size, component.sample_ids[0]))
    assignment: dict[str, str] = {}
    for component in order:
        deficits = {split: targets[split] - assigned[split] for split in SPLITS}
        best = max(deficits.values())
        tied = [
            split
            for split in SPLITS
            if abs(deficits[split] - best) <= 1e-9 * max(1.0, abs(best))
        ]
        split = rng.choice(tied) if len(tied) > 1 else tied[0]
        for sample_id in component.sample_ids:
            assignment[sample_id] = split
        assigned[split] += component.size
    return assignment


def cross_split_assets(entries: Sequence[Any]) -> dict[str, int]:
    """Count assets per namespace that appear in more than one split.

    A correct index must report zero everywhere; the function is used both to
    build the index evidence and to detect a tampered/stale index at load time.
    """

    counts: dict[str, int] = {}
    for name in GROUP_NAMES:
        splits_by_asset: dict[str, set[str]] = {}
        for entry in entries:
            raw = entry.groups[name]
            values = (raw,) if name == "composition" else tuple(raw)
            for value in values:
                splits_by_asset.setdefault(value, set()).add(entry.split)
        counts[name] = sum(1 for splits in splits_by_asset.values() if len(splits) > 1)
    return counts


__all__ = [
    "GROUP_NAMES",
    "SPLITS",
    "Component",
    "asset_keys",
    "assign_splits",
    "connected_components",
    "cross_split_assets",
    "mark_oversized",
    "normalize_ratios",
]
