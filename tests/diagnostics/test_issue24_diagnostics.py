"""Base-environment issue #24 diagnostics: context, sampler, tracker, plan.

These tests never import ``aat.losses``/torch, so they run in the base and
render CI jobs.  They assert the measured structural facts, not model quality.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aat.diagnostics import (
    context_boundary_report,
    context_overlap_report,
    load_phase_b_plan,
    sampling_coverage_report,
    tracker_postprocess_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPO_ROOT / "configs" / "research" / "issue24_phase_b_plan.toml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_issue24.py"


def test_context_boundary_shows_bare_edges_and_context_padding_fix():
    report = context_boundary_report()
    assert report["status"] == "ok"
    assert report["target_interval_seconds"] == [0.0, 8.0]

    bare = report["bare_clip"]
    assert bare["centers_total"] == 81
    assert bare["centers_valid"] == 61
    assert bare["centers_invalid"] == 20
    assert bare["first_valid_center_seconds"] == pytest.approx(1.0)
    assert bare["last_valid_center_seconds"] == pytest.approx(7.0)
    # The two short drums of the issue #11 listening clip sit in the first
    # second: in a bare clip they appear in the context of valid windows but
    # no valid center can carry them as a center label.
    for event in bare["events"]:
        assert event["target_relative_onset_seconds"] in (pytest.approx(0.1), pytest.approx(0.6))
        assert event["valid_centers_at_onset"] == 0
        assert event["valid_centers_with_onset_in_window"] > 0
        assert event["visible_as_context_only"] is True

    # Reading W/2 extra context from the original source keeps the target
    # interval and its time axis unchanged while the same target-relative
    # events become labelable.
    padded = report["context_padded"]
    assert padded["context_seconds_each_side"] == pytest.approx(1.0)
    assert padded["absolute_target_interval_seconds"] == pytest.approx([1.0, 9.0])
    assert padded["target_relative_axis_preserved"] is True
    assert padded["first_valid_center_target_relative_seconds"] == pytest.approx(0.0)
    assert padded["last_valid_center_target_relative_seconds"] == pytest.approx(8.0)
    for event in padded["events"]:
        assert event["target_relative_onset_seconds"] in (pytest.approx(0.1), pytest.approx(0.6))
        assert event["absolute_onset_seconds"] == pytest.approx(
            1.0 + event["target_relative_onset_seconds"]
        )
        assert event["valid_centers_at_onset"] == 1
        assert event["onset_labelable_as_center"] is True
    assert "do not shrink" in report["recommendation"]


def test_context_overlap_is_an_upper_bound_for_the_closest_allowed_pair():
    report = context_overlap_report()
    assert report["is_upper_bound_only"] is True
    assert report["closest_allowed_center_spacing_seconds"] == pytest.approx(0.1)
    assert report["max_possible_context_overlap_seconds"] == pytest.approx(1.9)
    assert report["max_possible_context_overlap_fraction"] == pytest.approx(0.95)
    assert "min_center_gap" in report["note"]


def test_tracker_postprocess_separates_slots_from_tracks():
    report = tracker_postprocess_report()
    scenarios = {item["scenario"]: item for item in report["scenarios"]}

    permutation = scenarios["slot-permutation"]
    assert permutation["track_count"] == 2
    assert permutation["raw_slot_identity_changes"] == 1
    assert permutation["single_stable_track_ids"] is True

    retention = scenarios["silence-retention"]
    # A silent but still-matchable candidate keeps the track alive; retention
    # only expires tracks when no candidate passes the gate.
    assert retention["short_gap_with_embedding"]["track_count"] == 1
    assert retention["long_gap_with_embedding"]["track_count"] == 1
    assert retention["short_gap_without_candidate"]["track_count"] == 1
    assert retention["long_gap_without_candidate"]["track_count"] == 2

    gate = scenarios["gate-sensitivity"]
    assert gate["track_count_at_gate_0_7"] == 2
    assert gate["track_count_at_gate_0_6"] == 1
    assert gate["forced_link_at_loose_gate"] is True


def test_sampling_coverage_is_deterministic_and_bounded(smoke_corpus):
    from aat.data import DatasetIndex

    index = DatasetIndex.load(
        smoke_corpus["index_path"], data_root=smoke_corpus["data_root"], verify_files=True
    )
    first = sampling_coverage_report(index, smoke_corpus["data_root"], steps=4)
    second = sampling_coverage_report(index, smoke_corpus["data_root"], steps=4)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] == "ok"
    assert first["songs"]
    closest = first["closest_allowed_center_spacing_seconds"]
    assert closest == pytest.approx(0.1)
    for song in first["songs"]:
        assert 0.0 <= song["valid_row_coverage"] <= 1.0
        for coverage in song["active_row_coverage_per_source"]:
            assert coverage is None or 0.0 <= coverage <= 1.0
        gaps = song["sampled_pair_gap_seconds"]
        assert gaps is not None
        # min_center_gap is a lower bound: the actual sampled gaps are >= it.
        assert gaps["min"] >= closest - 1e-9
        assert gaps["min"] <= gaps["median"] <= gaps["max"]
        overlap = song["sampled_group_max_context_overlap_fraction"]
        assert 0.0 <= overlap["min"] <= overlap["max"] <= 1.0
    overall_gaps = first["sampled_pair_gap_seconds"]
    assert overall_gaps["min"] >= closest - 1e-9
    overall_overlap = first["sampled_group_max_context_overlap_fraction"]
    assert (
        overall_overlap["max"]
        <= first["max_possible_context_overlap_fraction_for_closest_pair"] + 1e-9
    )


def test_committed_phase_b_plan_is_valid_and_not_executable():
    import tomllib

    result = load_phase_b_plan(PLAN_PATH)
    assert result["valid"] is True, result["errors"]
    assert result["plan"]["stage"] == "design-only"
    assert result["plan"]["authorization"] == "pending-main-session-approval"
    assert result["plan"]["not_executable"] is True
    assert len(result["experiments"]) >= 12
    ids = [experiment["id"] for experiment in result["experiments"]]
    assert len(ids) == len(set(ids))
    for expected in (
        "b0-overfit-gate",
        "b3a-supervision-negative-scope",
        "b3b-supervision-negative-off",
        "b8-endpoint-splice-postproc",
        "b10-frozen-holdout-check",
    ):
        assert expected in ids

    payload = tomllib.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert "never used for candidate" in payload["data"]["historically_observed_test"]
    assert "NOT generated in Phase A" in payload["data"]["frozen_holdout"]
    assert "only after the final configuration is frozen" in payload["data"]["frozen_holdout"]
    experiment_ids = [experiment["id"] for experiment in payload["experiments"]]
    assert "b10-frozen-holdout-check" in experiment_ids


def test_plan_validation_rejects_unsafe_mutations(tmp_path):
    original = PLAN_PATH.read_text(encoding="utf-8")

    def write(name: str, content: str) -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    assert not load_phase_b_plan(write("bad-auth.toml", original.replace(
        'authorization = "pending-main-session-approval"',
        'authorization = "approved"',
    )))["valid"]
    assert not load_phase_b_plan(write("bad-runnable.toml", original.replace(
        'not_executable = true',
        'not_executable = false',
    )))["valid"]
    assert not load_phase_b_plan(write("bad-fake.toml", original.replace(
        "fake_use = ",
        "# fake_use = ",
    )))["valid"]
    duplicated = original.replace(
        'id = "b1-baseline-current"',
        'id = "b0-overfit-gate"',
    )
    assert not load_phase_b_plan(write("bad-duplicate.toml", duplicated))["valid"]
    missing_stop = original.replace(
        'stop_condition = "If the re-run differs materially from the recorded #8 baseline, diagnose the difference first; do not compare later candidates against an unreproduced baseline"',
        'stop_condition = ""',
    )
    assert not load_phase_b_plan(write("bad-stop.toml", missing_stop))["valid"]
    undecided = original.replace(
        'single_variable = "negative term scope (batch-global -> same-composition-only); all else as b1"',
        'single_variable = "negative scope: same-composition-only or disabled, chosen before the run"',
    )
    assert not load_phase_b_plan(write("bad-undecided.toml", undecided))["valid"]
    no_holdout = original.replace(
        "frozen_holdout = ",
        "# frozen_holdout = ",
    )
    assert not load_phase_b_plan(write("bad-holdout.toml", no_holdout))["valid"]


def test_diagnose_cli_validates_plan_and_skips_sections_without_torch():
    validate = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "validate-plan", "--plan", str(PLAN_PATH)],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert validate.returncode == 0, validate.stderr
    assert "not executable" in validate.stdout

    report = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "report",
            "--skip-torch",
            "--skip-matching",
            "--skip-corpus",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    assert report.returncode == 0, report.stderr
    payload = json.loads(report.stdout)
    assert payload["diagnostic_version"] == "issue24-stage-a-v1"
    assert payload["sections"]["context_boundaries"]["status"] == "ok"
    assert payload["sections"]["matching_ambiguity"]["status"] == "skipped"
    assert payload["sections"]["sampling_coverage"]["status"] == "skipped"
    assert "not model evidence" in payload["data_kind"] or "not model evidence" in payload["sections"]["tracker_postprocess"]["evidence_kind"]
