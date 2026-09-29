"""Deterministic Phase A diagnostics for the issue #24 DETR research question.

This package is **diagnostics only**.  It does not add a model, a loss, a
tracking algorithm or a public protocol: it measures the behaviour of the
already-merged label / matching / head-loss / tracker code on small,
programmatically constructed cases and returns JSON-safe dictionaries.

The entry point is :func:`aat.diagnostics.issue24.build_issue24_report` (see
``scripts/diagnose_issue24.py``).  Every section carries an explicit
``evidence_kind`` so a synthetic label/matching counterexample is never read as
real model evidence.
"""

from __future__ import annotations

from .issue24 import (
    DIAGNOSTIC_DATA_KIND,
    DIAGNOSTIC_VERSION,
    build_issue24_report,
    context_boundary_report,
    context_overlap_report,
    identity_supervision_report,
    load_phase_b_plan,
    matching_ambiguity_report,
    sampling_coverage_report,
    tracker_postprocess_report,
)

__all__ = [
    "DIAGNOSTIC_DATA_KIND",
    "DIAGNOSTIC_VERSION",
    "build_issue24_report",
    "context_boundary_report",
    "context_overlap_report",
    "identity_supervision_report",
    "load_phase_b_plan",
    "matching_ambiguity_report",
    "sampling_coverage_report",
    "tracker_postprocess_report",
]
