"""Grouping, transitive leakage prevention and seeded split assignment."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aat.data import (
    DatasetError,
    asset_keys,
    assign_splits,
    connected_components,
    cross_split_assets,
    mark_oversized,
    normalize_ratios,
)


def entry(sample_id: str, composition: str, presets=(), origins=(), split: str = "train"):
    return SimpleNamespace(
        sample_id=sample_id,
        groups={
            "composition": composition,
            "preset": tuple(sorted(presets)),
            "sample_origin": tuple(sorted(origins)),
        },
        split=split,
    )


@pytest.mark.parametrize(
    ("ratios", "expected"),
    [
        ([1, 1, 2], {"train": 0.25, "val": 0.25, "test": 0.5}),
        ({"train": 0.8, "val": 0.1, "test": 0.1}, {"train": 0.8, "val": 0.1, "test": 0.1}),
        ((0, 1, 0), {"train": 0.0, "val": 1.0, "test": 0.0}),
    ],
)
def test_normalize_ratios_accepts_mapping_and_sequence(ratios, expected) -> None:
    assert normalize_ratios(ratios) == pytest.approx(expected)


@pytest.mark.parametrize(
    "ratios",
    [
        [0.5, -0.1, 0.6],
        [0.0, 0.0, 0.0],
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        {"train": 0.5, "val": 0.5},
        {"train": 0.5, "val": 0.5, "test": 0.5, "extra": 0.0},
        {"train": True, "val": 0.0, "test": 0.0},
        "abc",
    ],
)
def test_normalize_ratios_rejects_invalid(ratios) -> None:
    with pytest.raises(DatasetError):
        normalize_ratios(ratios)


def test_asset_keys_use_independent_namespaces() -> None:
    keys = asset_keys(
        {"composition": "c1", "preset": ["x"], "sample_origin": ["x"]}
    )
    assert keys == (("composition", "c1"), ("preset", "x"), ("sample_origin", "x"))


def test_connected_components_handle_transitive_links() -> None:
    # A shares preset P with B; B shares sample_origin Q with C.  A and C must
    # end up in the same component through B even though they share nothing.
    entries = [
        entry("a", "comp-a", presets=["p"]),
        entry("b", "comp-b", presets=["p"], origins=["q"]),
        entry("c", "comp-c", origins=["q"]),
    ]
    components = connected_components(entries)
    assert len(components) == 1
    component = components[0]
    assert component.size == 3
    assert component.sample_ids == ("a", "b", "c")
    assert component.assets["composition"] == ("comp-a", "comp-b", "comp-c")
    assert component.assets["preset"] == ("p",)
    assert component.assets["sample_origin"] == ("q",)


def test_empty_asset_lists_do_not_group_samples() -> None:
    entries = [
        entry("a", "comp-a"),
        entry("b", "comp-b"),
    ]
    components = connected_components(entries)
    assert len(components) == 2
    assert all(component.size == 1 for component in components)


def test_same_value_in_different_namespaces_does_not_group() -> None:
    entries = [
        entry("a", "comp-a", presets=["x"]),
        entry("b", "comp-b", origins=["x"]),
    ]
    components = connected_components(entries)
    assert len(components) == 2


def test_composition_sharing_groups_samples() -> None:
    entries = [
        entry("a", "comp-a", presets=["p1"], origins=["o1"]),
        entry("b", "comp-a", presets=["p2"], origins=["o2"]),
    ]
    components = connected_components(entries)
    assert len(components) == 1
    assert components[0].assets["preset"] == ("p1", "p2")


def test_components_are_deterministic_and_numbered() -> None:
    entries = [entry("b", "comp-b"), entry("a", "comp-a"), entry("c", "comp-c")]
    first = connected_components(entries)
    second = connected_components(list(reversed(entries)))
    assert [component.sample_ids for component in first] == [
        component.sample_ids for component in second
    ]
    assert [component.component_id for component in first] == ["c0000", "c0001", "c0002"]


def test_mark_oversized_flags_components_above_largest_target() -> None:
    entries = [entry("lonely", "comp-lonely")]
    entries.extend(entry(f"linked-{index}", f"comp-linked-{index}", presets=["shared"]) for index in range(5))
    components = mark_oversized(
        connected_components(entries), normalize_ratios([0.8, 0.1, 0.1])
    )
    oversized = [component for component in components if component.oversized]
    assert len(oversized) == 1
    assert set(oversized[0].sample_ids) == {f"linked-{index}" for index in range(5)}
    assert all(component.size == 1 for component in components if component not in oversized)


def test_assign_splits_reproducible_and_seed_sensitive() -> None:
    entries = [entry(f"s{index:02d}", f"comp-{index}") for index in range(9)]
    components = connected_components(entries)
    ratios = normalize_ratios([1, 1, 1])
    first = assign_splits(components, ratios=ratios, seed=7)
    second = assign_splits(components, ratios=ratios, seed=7)
    assert first == second
    variants = {
        tuple(sorted(assign_splits(components, ratios=ratios, seed=seed).items()))
        for seed in range(8)
    }
    assert len(variants) > 1


def test_assign_splits_keeps_whole_components_together() -> None:
    entries = [
        entry("a", "comp-a", presets=["p"]),
        entry("b", "comp-b", presets=["p"], origins=["q"]),
        entry("c", "comp-c", origins=["q"]),
        entry("d", "comp-d"),
        entry("e", "comp-e"),
    ]
    components = connected_components(entries)
    assignment = assign_splits(components, ratios=normalize_ratios([1, 1, 1]), seed=3)
    component_splits = {
        component.component_id: {assignment[sample_id] for sample_id in component.sample_ids}
        for component in components
    }
    assert all(len(splits) == 1 for splits in component_splits.values())
    assert assignment["a"] == assignment["b"] == assignment["c"]


def test_cross_split_assets_detects_leaks_per_namespace() -> None:
    clean = [
        entry("a", "comp-a", presets=["p"], origins=["o"], split="train"),
        entry("b", "comp-b", presets=["p"], origins=["o"], split="train"),
        entry("c", "comp-c", split="val"),
    ]
    assert cross_split_assets(clean) == {
        "composition": 0,
        "preset": 0,
        "sample_origin": 0,
    }
    leaked = [
        entry("a", "comp-a", presets=["p"], origins=["o"], split="train"),
        entry("b", "comp-b", presets=["p"], origins=["o"], split="val"),
        entry("c", "comp-c", origins=["o"], split="test"),
    ]
    counts = cross_split_assets(leaked)
    assert counts["preset"] == 1
    assert counts["sample_origin"] == 1
    assert counts["composition"] == 0
