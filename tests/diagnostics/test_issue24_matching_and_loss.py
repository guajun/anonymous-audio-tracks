"""ML issue #24 diagnostics: exact matching and identity-supervision counts.

These tests import the ``aat.losses`` package, so they need the ``ml`` extra
and are marked ``ml`` like the other head/loss tests.  They measure the
*structure* of supervision on synthetic fixtures; they are not training runs.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from aat.diagnostics import (
    identity_supervision_report,
    matching_ambiguity_report,
)
from aat.diagnostics.issue24 import _step_seed

pytestmark = pytest.mark.ml


def _by_name(report):
    assert report["status"] == "ok"
    return {case["name"]: case for case in report["cases"]}


def test_matching_ambiguity_follows_activity_columns_not_timbres():
    cases = _by_name(matching_ambiguity_report())

    unique = [
        "single-source",
        "piano-two-short-drums",
        "alternating-activity",
        "reappear-after-silence",
    ]
    for name in unique:
        assert cases[name]["ambiguous"] is False, name
        assert cases[name]["identity_masked"] is False, name

    tied = cases["same-on-off-two-timbres"]
    assert tied["tied_activity_source_pairs"] == [[0, 1]]
    assert tied["ambiguous"] is True
    assert tied["identity_masked"] is True
    assert tied["num_optimal"] == 2

    duplicates = cases["indistinguishable-duplicates"]
    assert duplicates["ambiguous"] is True
    assert duplicates["num_optimal"] == 6

    silent = cases["all-silent"]
    assert silent["ambiguous"] is True
    assert silent["num_optimal"] == 12

    truncated = cases["silent-duplicates-truncated"]
    assert truncated["truncated"] is True
    assert truncated["identity_masked"] is True


def test_identity_supervision_counts_are_exposed_and_masked_correctly():
    cases = _by_name(identity_supervision_report())

    single = cases["single-source"]
    assert single["positive_terms"] == 3
    assert single["negative_terms"] == 0
    assert single["identity_masked_groups"] == 0

    piano = cases["piano-two-short-drums"]
    assert piano["positive_terms"] == 7
    assert piano["negative_terms"] == 7

    alternating = cases["alternating-activity"]
    assert alternating["positive_terms"] == 4
    assert alternating["negative_terms"] == 4

    reappear = cases["reappear-after-silence"]
    assert reappear["positive_terms"] == 4
    assert reappear["negative_terms"] == 0

    same_on_off = cases["same-on-off-two-timbres"]
    assert same_on_off["ambiguous_groups"] == 1
    assert same_on_off["identity_masked_groups"] == 1
    assert same_on_off["positive_terms"] == 0
    assert same_on_off["negative_terms"] == 0
    assert same_on_off["activity_terms"] > 0

    silent = cases["all-silent"]
    assert silent["ambiguous_groups"] == 1
    assert silent["positive_terms"] == 0
    assert silent["activity_terms"] > 0

    truncated = cases["silent-duplicates-truncated"]
    assert truncated["truncated_groups"] == 1
    assert truncated["supervision_masked_groups"] == 1
    assert truncated["activity_terms"] == 0
    assert truncated["positive_terms"] == 0


def test_identity_supervision_report_is_deterministic():
    first = identity_supervision_report()
    second = identity_supervision_report()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_local_step_seed_mirror_matches_training_step_seed():
    from aat.training.checkpoint import step_seed

    for base in (0, 20260929, 2**32 - 1):
        for step in (0, 1, 17, 100, 999):
            assert _step_seed(base, step) == step_seed(base, step)
