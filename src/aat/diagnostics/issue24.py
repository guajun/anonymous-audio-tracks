"""Deterministic Phase A diagnostics for issue #24 (DETR-style research).

What this module is
-------------------
A measurement harness over the **existing** code only:

* label/window boundary semantics (:func:`context_boundary_report`), using
  ``aat.windowing``;
* sampling coverage of the real multi-center sampler
  (:func:`sampling_coverage_report`), using ``aat.data.sample_batch`` on the
  synthetic smoke corpus;
* group-matching ambiguity (:func:`matching_ambiguity_report`), using
  ``aat.losses.matching`` (requires the ``ml`` extra because the parent
  ``aat.losses`` package imports torch);
* effective identity supervision counts
  (:func:`identity_supervision_report`), using ``aat.losses.head_loss``
  (requires torch);
* tracker/post-processing effects separated from raw E/P
  (:func:`tracker_postprocess_report`), using ``aat.tracking``;
* Phase B plan validation (:func:`load_phase_b_plan`), descriptive only.

What this module is not
-----------------------
It is **not** a model, a DETR decoder, a temporal-mask protocol or an endpoint
splicing implementation.  It never reads private, external or real audio, AuT
weights or checkpoints.  The sampler section writes and reads a tiny
temporary **synthetic smoke WAV corpus** (``make_smoke_dataset``) inside a
caller-provided temporary directory; everything else is constructed in memory
and the temporary corpus is deleted by the caller.

Every section is labelled with ``evidence_kind``.  Values measured from the
synthetic label/matching fixtures are *minimal counterexamples*, not model
quality statements; values from ``head_loss`` are supervision *counts* on
synthetic tensors, not training results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aat.contracts import AudioSpan, PredictionData, RunProvenance
from aat.tracking import TrackingConfig, associate_sequence
from aat.windowing import (
    center_times,
    centered_window_bounds,
    extract_windows_at_times,
    window_sample_count,
)

__all__ = [
    "DIAGNOSTIC_DATA_KIND",
    "DIAGNOSTIC_VERSION",
    "MATCHING_CASES",
    "build_issue24_report",
    "context_boundary_report",
    "context_overlap_report",
    "identity_supervision_report",
    "load_phase_b_plan",
    "matching_ambiguity_report",
    "sampling_coverage_report",
    "tracker_postprocess_report",
]

DIAGNOSTIC_VERSION = "issue24-stage-a-v1"

#: Machine-readable marker so no consumer treats the output as model evidence.
DIAGNOSTIC_DATA_KIND = (
    "synthetic-label-matching-postprocess-diagnostic "
    "(no private/external/real audio, weights, checkpoints or real model forward; "
    "the sampler section only writes and reads a temporary synthetic smoke WAV "
    "corpus; not model evidence)"
)

_MATCHING_EVIDENCE_KIND = (
    "label/matching minimal counterexample (deterministic synthetic fixtures; "
    "not model evidence)"
)
_SUPERVISION_EVIDENCE_KIND = (
    "loss-term counts on synthetic tensors (supervision structure only; "
    "not a training or quality result)"
)
_TRACKER_EVIDENCE_KIND = (
    "tracker/post-processing behaviour on synthetic E/P fixtures "
    "(post-process effect only; not model evidence)"
)
_CONTEXT_EVIDENCE_KIND = (
    "label/window boundary arithmetic on synthetic timelines "
    "(no acoustic claim)"
)
_SAMPLING_EVIDENCE_KIND = (
    "sampler coverage on the synthetic smoke corpus "
    "(engineering fixture, not the rendered local corpus)"
)

_FIXTURE_SEED = 20260924
_EMBEDDING_DIM = 128

# ---------------------------------------------------------------------------
# Minimal matching cases (labels only, no model weights)
# ---------------------------------------------------------------------------
#
# Each case is ``activity [N, S]`` for one composition-local group.  The
# current matching cost is built only from these activity sequences (embedding
# values never enter the matcher), so identical columns are structurally
# ambiguous regardless of the logits a model would emit.

MATCHING_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "single-source",
        "description": (
            "One composition-local source active on 3 of 4 center windows. "
            "Expected: unique matching."
        ),
        "activity": [[1.0], [0.0], [1.0], [1.0]],
        "slots": 4,
    },
    {
        "name": "piano-two-short-drums",
        "description": (
            "Piano active in 5 of 6 windows plus one drum source with two "
            "short isolated hits (windows 2 and 5). Expected: unique matching "
            "(the activity sequences differ)."
        ),
        "activity": [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        "slots": 4,
    },
    {
        "name": "same-on-off-two-timbres",
        "description": (
            "Two sources with identical activity sequences (1,1,0,0) but "
            "acoustically distinguishable timbres. Activity-only matching is "
            "ambiguous: the labels cannot name which column is which."
        ),
        "activity": [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "slots": 4,
    },
    {
        "name": "alternating-activity",
        "description": (
            "Two sources that never overlap (1,1,0,0) vs (0,0,1,1). Expected: "
            "unique matching; identity supervision can run."
        ),
        "activity": [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        "slots": 4,
    },
    {
        "name": "reappear-after-silence",
        "description": (
            "One source active, then silent for two windows, then active again "
            "inside the same group. Expected: unique matching; silent centers "
            "are supervised to P=0 while identity anchors only come from "
            "active windows."
        ),
        "activity": [[1.0], [1.0], [0.0], [0.0], [1.0], [1.0]],
        "slots": 4,
    },
    {
        "name": "all-silent",
        "description": (
            "Two valid but silent sources and two empty slots. Expected: "
            "identical all-zero rows tie several optimal assignments; activity "
            "is still supervised to 0, identity is masked."
        ),
        "activity": [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        "slots": 4,
    },
    {
        "name": "indistinguishable-duplicates",
        "description": (
            "Three identical activity sequences (1,0,1,1): an indistinguishable "
            "repeated layer. Expected: 3! = 6 tied assignments (ambiguous), so "
            "identity supervision is masked and activity is symmetric."
        ),
        "activity": [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        "slots": 4,
    },
    {
        "name": "silent-duplicates-truncated",
        "description": (
            "Three identical all-zero sources with K=8: the optimal set has "
            "P(8,3)=336 assignments, above the enumeration cap (64). Expected: "
            "truncated -> every supervised term of the group is skipped, not "
            "just identity."
        ),
        "activity": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "slots": 8,
        "max_optimal": 64,
    },
)


def _bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise BCE, matching ``torch`` semantics."""

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))


def _oracle_logits(activity: np.ndarray, slots: int) -> np.ndarray:
    """Deterministic "well-fit model" logits for a matching fixture.

    Slot ``k < S`` copies source ``k``'s activity pattern (``+2`` active,
    ``-2`` inactive); extra slots always emit ``-2``.  The matching cost is
    built from these logits exactly the way ``head_loss`` builds its group
    cost, but the ambiguity reported for identical activity columns is
    independent of this choice.
    """

    activity = np.asarray(activity, dtype=np.float64)
    windows, sources = activity.shape
    logits = np.full((windows, slots), -2.0, dtype=np.float64)
    for slot in range(min(slots, sources)):
        active = activity[:, slot] >= 0.5
        logits[:, slot] = np.where(active, 2.0, -2.0)
    return logits


def group_cost_matrix(
    activity: Any,
    logits: Any,
    *,
    center_valid: Any | None = None,
    slot_valid: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate the group-level matching cost from ``aat.losses.head_loss``.

    Returns ``(cost [S, K], source_valid [S], slot_valid [K])``.  ``cost`` is
    the mean BCE over valid centers, i.e. ``head_loss``'s
    ``cost_num / cost_den`` for one group.
    """

    activities = np.asarray(activity, dtype=np.float64)
    if activities.ndim != 2:
        raise ValueError("activity must be [N, S]")
    windows, sources = activities.shape
    logits_array = np.asarray(logits, dtype=np.float64)
    if logits_array.shape[0] != windows or logits_array.ndim != 2:
        raise ValueError("logits must be [N, K] with the same N as activity")
    slots = int(logits_array.shape[1])

    if center_valid is None:
        center_mask = np.ones(windows, dtype=bool)
    else:
        center_mask = np.asarray(center_valid, dtype=bool)
        if center_mask.shape != (windows,):
            raise ValueError("center_valid must be [N]")
    source_mask = np.ones(sources, dtype=bool)
    slot_mask = np.ones(slots, dtype=bool)
    if slot_valid is not None:
        slot_mask = np.asarray(slot_valid, dtype=bool)
        if slot_mask.shape != (slots,):
            raise ValueError("slot_valid must be [K]")

    cost = np.full((sources, slots), np.inf, dtype=np.float64)
    for source in range(sources):
        if not bool(source_mask[source]):
            continue
        for slot in range(slots):
            if not bool(slot_mask[slot]):
                continue
            column = _bce_with_logits(logits_array[:, slot], activities[:, source])
            denominator = int(center_mask.sum())
            cost[source, slot] = float(column[center_mask].sum()) / max(denominator, 1)
    return cost, source_mask, slot_mask


def matching_ambiguity_report(*, max_optimal: int | None = None) -> dict[str, Any]:
    """Run the minimal cases through the real exact matcher.

    Requires the ``ml`` extra only because ``aat.losses.__init__`` imports
    torch; the matcher itself is NumPy.  Without torch the section is reported
    as ``unavailable`` instead of failing the whole diagnostic run.
    """

    try:
        from aat.losses.matching import match_sources
    except ImportError as exc:  # pragma: no cover - depends on environment
        return {
            "status": "unavailable",
            "reason": "aat.losses.matching requires the ml extra (torch import in aat.losses)",
            "error": str(exc),
            "evidence_kind": _MATCHING_EVIDENCE_KIND,
        }

    rows: list[dict[str, Any]] = []
    for case in MATCHING_CASES:
        activity = np.asarray(case["activity"], dtype=np.float64)
        slots = int(case["slots"])
        cap = int(case.get("max_optimal", max_optimal if max_optimal is not None else 64))
        logits = _oracle_logits(activity, slots)
        cost, source_mask, slot_mask = group_cost_matrix(activity, logits)
        result = match_sources(
            cost[None, :, :], source_mask[None, :], slot_mask[None, :], max_optimal=cap
        )
        group = result.groups[0]

        tied_pairs: list[list[int]] = []
        for left in range(activity.shape[1]):
            for right in range(left + 1, activity.shape[1]):
                if np.array_equal(activity[:, left], activity[:, right]):
                    tied_pairs.append([left, right])

        rows.append(
            {
                "name": case["name"],
                "description": case["description"],
                "windows": int(activity.shape[0]),
                "sources": int(activity.shape[1]),
                "slots": slots,
                "max_optimal": cap,
                "tied_activity_source_pairs": tied_pairs,
                "cost_rows_identical": (
                    bool(np.allclose(cost[tied_pairs[0][0]], cost[tied_pairs[0][1]]))
                    if tied_pairs
                    else False
                ),
                "num_optimal": int(group.optimal.num_optimal),
                "enumerated": len(group.optimal.assignments),
                "truncated": bool(group.truncated),
                "ambiguous": bool(group.ambiguous),
                "identity_masked": bool(group.identity_masked),
                "optimal_cost": float(group.optimal.optimal_cost),
            }
        )

    return {
        "status": "ok",
        "evidence_kind": _MATCHING_EVIDENCE_KIND,
        "matcher": "aat.losses.matching.match_sources (exact bitmask DP)",
        "cost": (
            "group mean BCE over centers, embedding never enters the cost; "
            "identical activity columns therefore tie regardless of model logits"
        ),
        "cases": rows,
    }


# ---------------------------------------------------------------------------
# Effective identity supervision (head_loss term counts)
# ---------------------------------------------------------------------------


def _synthetic_embeddings(windows: int, slots: int):
    """Deterministic unit vectors [1, N, K, 128] with slot-distinct prototypes."""

    import math

    import torch

    dim = _EMBEDDING_DIM
    index = torch.arange(dim, dtype=torch.float32)
    prototypes = []
    for slot in range(slots):
        base = torch.cos(2.0 * math.pi * (slot + 1) * (index + 0.5) / dim)
        base = base - base.mean()
        prototypes.append(base / base.norm())
    rows = []
    for window in range(windows):
        per_slot = []
        for slot in range(slots):
            drift = 0.03 * torch.sin((window + 1) * (index + 1) * (slot + 1) / dim * math.pi)
            vector = prototypes[slot] + drift
            per_slot.append(vector / vector.norm())
        rows.append(torch.stack(per_slot))
    return torch.stack(rows).unsqueeze(0)


def identity_supervision_report() -> dict[str, Any]:
    """Measure identity-term counts for the same minimal cases.

    The embeddings/logits are synthetic; only the *counts* (and the masking
    structure) are meaningful.  Counts answer "how much identity supervision
    would this label structure actually execute", not "how good is identity".
    """

    try:
        import torch

        from aat.losses.head_loss import head_loss
    except ImportError as exc:  # pragma: no cover - depends on environment
        return {
            "status": "unavailable",
            "reason": "head_loss requires the ml extra (torch)",
            "error": str(exc),
            "evidence_kind": _SUPERVISION_EVIDENCE_KIND,
        }

    rows: list[dict[str, Any]] = []
    for case in MATCHING_CASES:
        activity = np.asarray(case["activity"], dtype=np.float64)
        slots = int(case["slots"])
        windows, sources = activity.shape
        logits = _oracle_logits(activity, slots)

        embeddings = _synthetic_embeddings(windows, slots)
        activity_tensor = torch.tensor(activity, dtype=torch.float32).unsqueeze(0)
        logits_tensor = torch.tensor(logits, dtype=torch.float32).unsqueeze(0)
        result = head_loss(
            embeddings,
            logits_tensor,
            activity_target=activity_tensor,
            composition_ids=("comp-diagnostic",),
            source_ids=(tuple(f"src-{index}" for index in range(sources)),),
            max_optimal_assignments=int(case.get("max_optimal", 64)),
        )
        stats = result.stats
        components = result.summary()
        activity_terms = max(int(stats.activity_terms), 0)
        positive_terms = int(stats.positive_terms)
        negative_terms = int(stats.negative_terms)
        matched_groups = int(stats.matched_groups)
        rows.append(
            {
                "name": case["name"],
                "windows": int(windows),
                "sources": int(sources),
                "slots": slots,
                "matched_groups": matched_groups,
                "groups": int(stats.groups),
                "ambiguous_groups": int(stats.ambiguous_groups),
                "truncated_groups": int(stats.truncated_groups),
                "identity_masked_groups": int(stats.identity_masked_groups),
                "supervision_masked_groups": int(stats.supervision_masked_groups),
                "activity_terms": activity_terms,
                "empty_terms": int(stats.empty_terms),
                "positive_terms": positive_terms,
                "negative_terms": negative_terms,
                "identity_masked_group_ratio": (
                    float(stats.identity_masked_groups / matched_groups)
                    if matched_groups
                    else None
                ),
                "positive_per_activity_term": (
                    float(positive_terms / activity_terms) if activity_terms else None
                ),
                "negative_per_activity_term": (
                    float(negative_terms / activity_terms) if activity_terms else None
                ),
                "loss_components": {key: float(value) for key, value in components.items()},
            }
        )

    return {
        "status": "ok",
        "evidence_kind": _SUPERVISION_EVIDENCE_KIND,
        "loss": "aat.losses.head_loss with default weights and synthetic tensors",
        "note": (
            "identity terms run only for reliable (unique-optimum) groups; "
            "ambiguous groups mask identity while activity/empty are averaged; "
            "truncated groups skip every supervised term"
        ),
        "cases": rows,
    }


# ---------------------------------------------------------------------------
# Context / window boundary arithmetic
# ---------------------------------------------------------------------------


def _round_trip_int(value: Any) -> int:
    return int(value)


def context_overlap_report(
    *,
    window_seconds: float = 2.0,
    label_hop_seconds: float = 0.02,
    min_center_gap: int = 5,
    centers_per_item: int = 4,
) -> dict[str, Any]:
    """Upper bound of context sharing between the closest allowed pair.

    ``min_center_gap`` is a **minimum** distance in grid steps, not a fixed
    spacing: ``aat.data.batch._select_centers`` randomly picks centers whose
    pairwise distance is at least this large, so actual pairs are usually
    farther apart.  This report therefore names the numbers as the
    worst-case/upper bound of a single ``centers_per_item``-wide group; the
    actual sampled distribution is measured in :func:`sampling_coverage_report`.
    """

    closest_spacing = float(label_hop_seconds) * int(min_center_gap)
    max_overlap = float(window_seconds) - closest_spacing
    return {
        "status": "ok",
        "evidence_kind": _CONTEXT_EVIDENCE_KIND,
        "window_seconds": float(window_seconds),
        "label_hop_seconds": float(label_hop_seconds),
        "min_center_gap_grid_steps": int(min_center_gap),
        "centers_per_item": int(centers_per_item),
        "closest_allowed_center_spacing_seconds": closest_spacing,
        "max_possible_context_overlap_seconds": max_overlap,
        "max_possible_context_overlap_fraction": max_overlap / float(window_seconds),
        "is_upper_bound_only": True,
        "note": (
            "min_center_gap=5 only forbids pairs closer than 0.1 s; sampled pairs "
            "are >= that distance, so 1.9 s / 95% is the worst-case overlap of the "
            "closest allowed pair, not the group's actual spacing. Actual sampled "
            "gap/overlap min/median/max are reported by sampling_coverage_report."
        ),
    }


def _event_center_stats(
    centers: np.ndarray,
    window_valid: np.ndarray,
    *,
    width: int,
    rate: int,
    hop_seconds: float,
    onset_seconds: float,
) -> dict[str, Any]:
    """Per-event center statistics on one center grid (absolute seconds)."""

    onset_sample = int(np.floor(onset_seconds * rate + 0.5))
    centers_with_onset = 0
    valid_centers_with_onset = 0
    valid_centers_at_onset = 0
    for index, center in enumerate(centers):
        center_sample = int(np.floor(float(center) * rate + 0.5))
        start, stop = centered_window_bounds(center_sample, width)
        if start <= onset_sample < stop:
            centers_with_onset += 1
            if bool(window_valid[index]):
                valid_centers_with_onset += 1
        if (
            abs(float(center) - onset_seconds) <= float(hop_seconds) / 2.0 + 1e-12
            and bool(window_valid[index])
        ):
            valid_centers_at_onset += 1
    return {
        "centers_with_onset_in_window": centers_with_onset,
        "valid_centers_with_onset_in_window": valid_centers_with_onset,
        "valid_centers_at_onset": valid_centers_at_onset,
        "onset_labelable_as_center": bool(valid_centers_at_onset),
        "visible_as_context_only": bool(
            valid_centers_with_onset and not valid_centers_at_onset
        ),
    }


def context_boundary_report(
    *,
    duration_seconds: float = 8.0,
    window_seconds: float = 2.0,
    hop_seconds: float = 0.1,
    sample_rate: int = 16000,
    events: Sequence[float] = (0.1, 0.6),
) -> dict[str, Any]:
    """Window/valid-mask arithmetic for a fixed-length target interval.

    This is the issue #11 listening-clip situation: two short events at
    target-relative 0.1 s and 0.6 s of an 8 s interval.  Two views are compared:

    ``bare_clip``
        the 8 s interval is the whole audio; its first/last W/2 are zero-padded,
        so those events have no valid center;
    ``context_padded``
        an extra W/2 is read from the original source on both sides (the audio
        is longer), while the target interval and its absolute original-track
        time axis are unchanged.  The same events stay at their target-relative
        times and now have fully covered centers.

    The report's point is that the correct handling is context padding, not
    shrinking or excluding the target edges.
    """

    rate = int(sample_rate)
    duration = float(duration_seconds)
    window = float(window_seconds)
    width = window_sample_count(window, rate)
    context = window / 2.0

    # --- bare clip: the target interval is the entire audio -----------------
    bare_samples = np.zeros(int(round(duration * rate)), dtype=np.float32)
    bare_centers = center_times(duration, hop_seconds)
    _, bare_valid_matrix = extract_windows_at_times(
        bare_samples, bare_centers, rate, window
    )
    bare_window_valid = (
        bare_valid_matrix.all(axis=1)
        if bare_valid_matrix.size
        else np.zeros(0, dtype=bool)
    )
    bare_events: list[dict[str, Any]] = []
    for onset in events:
        stats = _event_center_stats(
            bare_centers,
            bare_window_valid,
            width=width,
            rate=rate,
            hop_seconds=float(hop_seconds),
            onset_seconds=float(onset),
        )
        bare_events.append(
            {
                "target_relative_onset_seconds": float(onset),
                **stats,
                "finding": (
                    "no valid center has this onset at its center: the event is "
                    "only inside the context of later valid windows and is not a "
                    "center label"
                    if stats["visible_as_context_only"]
                    else "a valid center exists at this onset"
                ),
            }
        )
    bare_valid_centers = bare_centers[bare_window_valid]
    bare_first = float(bare_valid_centers[0]) if bare_valid_centers.size else None
    bare_last = float(bare_valid_centers[-1]) if bare_valid_centers.size else None

    # --- context padded: target interval unchanged, extra W/2 each side -----
    padded_samples = np.zeros(int(round((duration + window) * rate)), dtype=np.float32)
    padded_centers = center_times(duration + window, hop_seconds)
    _, padded_valid_matrix = extract_windows_at_times(
        padded_samples, padded_centers, rate, window
    )
    padded_window_valid = (
        padded_valid_matrix.all(axis=1)
        if padded_valid_matrix.size
        else np.zeros(0, dtype=bool)
    )
    padded_events: list[dict[str, Any]] = []
    for onset in events:
        absolute_onset = context + float(onset)
        stats = _event_center_stats(
            padded_centers,
            padded_window_valid,
            width=width,
            rate=rate,
            hop_seconds=float(hop_seconds),
            onset_seconds=absolute_onset,
        )
        padded_events.append(
            {
                "target_relative_onset_seconds": float(onset),
                "absolute_onset_seconds": absolute_onset,
                **stats,
                "finding": (
                    "a valid center exists at the onset; the event is labelable "
                    "at its unchanged target-relative time"
                    if stats["onset_labelable_as_center"]
                    else "the event is still not labelable on this hop grid"
                ),
            }
        )
    padded_valid_centers = padded_centers[padded_window_valid]
    padded_first = float(padded_valid_centers[0]) if padded_valid_centers.size else None
    padded_last = float(padded_valid_centers[-1]) if padded_valid_centers.size else None

    return {
        "status": "ok",
        "evidence_kind": _CONTEXT_EVIDENCE_KIND,
        "target_duration_seconds": duration,
        "target_interval_seconds": [0.0, duration],
        "window_seconds": window,
        "hop_seconds": float(hop_seconds),
        "sample_rate": rate,
        "left_context_seconds": context,
        "right_context_seconds": context,
        "bare_clip": {
            "audio_seconds_read": duration,
            "centers_total": _round_trip_int(bare_centers.size),
            "centers_valid": _round_trip_int(bare_valid_centers.size),
            "centers_invalid": _round_trip_int(
                bare_centers.size - bare_valid_centers.size
            ),
            "first_valid_center_seconds": bare_first,
            "last_valid_center_seconds": bare_last,
            "addressable_interval_seconds": [bare_first, bare_last],
            "events": bare_events,
        },
        "context_padded": {
            "context_seconds_each_side": context,
            "audio_seconds_read": duration + window,
            "absolute_target_interval_seconds": [context, context + duration],
            "target_relative_axis_preserved": True,
            "centers_total": _round_trip_int(padded_centers.size),
            "centers_valid": _round_trip_int(padded_valid_centers.size),
            "centers_invalid": _round_trip_int(
                padded_centers.size - padded_valid_centers.size
            ),
            "first_valid_center_absolute_seconds": padded_first,
            "last_valid_center_absolute_seconds": padded_last,
            "first_valid_center_target_relative_seconds": (
                None if padded_first is None else padded_first - context
            ),
            "last_valid_center_target_relative_seconds": (
                None if padded_last is None else padded_last - context
            ),
            "events": padded_events,
        },
        "note": (
            "center_valid follows the labeler rule 'the model center window fits "
            "fully inside the audio'. A bare target clip cannot label its first/last "
            "W/2 seconds; reading W/2 extra context from the original source keeps "
            "the target interval and its original-track time axis unchanged while "
            "giving the same target-relative events fully covered centers."
        ),
        "recommendation": (
            "pad context from the original song; do not shrink, exclude or re-base "
            "the target interval"
        ),
    }


# ---------------------------------------------------------------------------
# Sampling coverage (real sampler on the synthetic smoke corpus)
# ---------------------------------------------------------------------------


def _step_seed(base_seed: int, step: int) -> int:
    """Mirror of ``aat.training.checkpoint.step_seed`` without importing torch.

    ``aat.training.checkpoint`` imports torch at module scope, so the base
    environment cannot import it.  The formula is identical (checked by an
    ``ml`` test when torch is available).
    """

    return int(
        np.random.SeedSequence([int(base_seed), int(step), 0xA17]).generate_state(1)[0]
    )


def _load_activity_arrays(entry: Any, data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``activity.activity`` / ``activity.valid`` for an index entry."""

    from pathlib import PurePosixPath

    from aat.contracts import ActivityData

    directory = Path(data_root) / PurePosixPath(entry.path)
    metadata = PurePosixPath(str(entry.labels["metadata_path"]))
    activity = ActivityData.load(directory / metadata.parent, metadata_filename=metadata.name)
    return (
        np.asarray(activity.activity, dtype=np.float64),
        np.asarray(activity.valid, dtype=bool),
    )


def _numeric_stats(values: Sequence[float]) -> dict[str, Any] | None:
    """min/median/max/count summary for a possibly empty sample list."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def sampling_coverage_report(
    index: Any,
    data_root: str | Path,
    *,
    seed: int = 20260929,
    split: str = "train",
    steps: int = 10,
    groups_per_step: int = 2,
    centers_per_item: int = 4,
    min_center_gap: int = 5,
    activity_threshold: float = 0.5,
) -> dict[str, Any]:
    """Measure what the real ``sample_batch`` scheme actually touches.

    For a fixed number of simulated training steps (the same
    ``seed_sequence`` batch seeds as training), report per song: share of
    valid rows sampled, share of active label rows sampled, and how often each
    source has at least two active anchors inside one group (the prerequisite
    for the leave-one-out positive term).
    """

    from aat.data import sample_batch

    entries = list(index.samples_for_split(split))
    totals: dict[str, dict[str, Any]] = {}
    for entry in entries:
        activity, valid = _load_activity_arrays(entry, data_root)
        sources = len(entry.source_ids)
        active_counts = []
        for source in range(sources):
            active = (activity[:, source] >= activity_threshold) & valid
            active_counts.append(int(active.sum()))
        totals[entry.sample_id] = {
            "path": str(entry.path),
            "sources": sources,
            "valid_rows": int(valid.sum()),
            "total_rows": int(valid.size),
            "active_rows_per_source": active_counts,
            "sampled_rows": set(),
            "sampled_active_rows_per_source": [set() for _ in range(sources)],
            "groups_with_two_anchors_per_source": [0] * sources,
            "groups_sampled": 0,
            "pair_gaps": [],
            "group_min_gaps": [],
            "group_max_overlap_fractions": [],
            "window_seconds": None,
            "hop_seconds": None,
        }

    skipped: list[dict[str, str]] = []
    for step in range(int(steps)):
        batch = sample_batch(
            index,
            data_root,
            seed=_step_seed(seed, step),
            split=split,
            items=int(groups_per_step),
            centers_per_item=int(centers_per_item),
            min_center_gap=int(min_center_gap),
            valid_only=True,
            activity_threshold=float(activity_threshold),
            on_unusable="skip",
        )
        skipped.extend(
            {"sample_id": sample_id, "reason": reason} for sample_id, reason in batch.skipped
        )
        for block in batch.blocks:
            record = totals[block.sample_id]
            record["groups_sampled"] += 1
            record["window_seconds"] = float(block.window_seconds)
            record["hop_seconds"] = block.hop_seconds
            sampled_times = np.sort(np.asarray(block.center_times, dtype=np.float64))
            if sampled_times.size >= 2:
                gaps = np.diff(sampled_times)
                record["pair_gaps"].extend(float(value) for value in gaps)
                record["group_min_gaps"].append(float(gaps.min()))
                record["group_max_overlap_fractions"].append(
                    float(
                        max(
                            0.0,
                            1.0 - float(gaps.min()) / float(block.window_seconds),
                        )
                    )
                )
            for row, center_index in enumerate(np.asarray(block.center_indices).tolist()):
                record["sampled_rows"].add(int(center_index))
            for source in range(len(block.source_ids)):
                active_row_mask = (
                    np.asarray(block.activity[:, source]) >= activity_threshold
                ) & np.asarray(block.center_valid)
                active_rows = np.nonzero(active_row_mask)[0]
                if len(active_rows) >= 2:
                    record["groups_with_two_anchors_per_source"][source] += 1
                for row in active_rows.tolist():
                    center_index = int(block.center_indices[row])
                    record["sampled_active_rows_per_source"][source].add(center_index)

    songs: list[dict[str, Any]] = []
    for sample_id, record in sorted(totals.items()):
        valid_rows = max(int(record["valid_rows"]), 1)
        active_per_source = record["active_rows_per_source"]
        sampled_active = [len(values) for values in record["sampled_active_rows_per_source"]]
        coverage = [
            (sampled / available if available else None)
            for sampled, available in zip(sampled_active, active_per_source)
        ]
        songs.append(
            {
                "sample_id": sample_id,
                "path": record["path"],
                "sources": int(record["sources"]),
                "groups_sampled": int(record["groups_sampled"]),
                "valid_rows": int(record["valid_rows"]),
                "total_rows": int(record["total_rows"]),
                "sampled_valid_rows": len(record["sampled_rows"]),
                "valid_row_coverage": len(record["sampled_rows"]) / valid_rows,
                "active_rows_per_source": active_per_source,
                "sampled_active_rows_per_source": sampled_active,
                "active_row_coverage_per_source": coverage,
                "groups_with_two_anchors_per_source": record[
                    "groups_with_two_anchors_per_source"
                ],
                "sampled_pair_gap_seconds": _numeric_stats(record["pair_gaps"]),
                "sampled_group_min_gap_seconds": _numeric_stats(record["group_min_gaps"]),
                "sampled_group_max_context_overlap_fraction": _numeric_stats(
                    record["group_max_overlap_fractions"]
                ),
                "sampling_window_seconds": record["window_seconds"],
                "sources_never_activated_in_sampled_windows": [
                    index
                    for index, available in enumerate(active_per_source)
                    if available > 0 and sampled_active[index] == 0
                ],
            }
        )

    all_pair_gaps = [
        gap for record in totals.values() for gap in record["pair_gaps"]
    ]
    all_group_overlaps = [
        value
        for record in totals.values()
        for value in record["group_max_overlap_fractions"]
    ]
    sample_window = next(
        (record["window_seconds"] for record in totals.values() if record["window_seconds"]),
        None,
    )
    sample_hop = next(
        (record["hop_seconds"] for record in totals.values() if record["hop_seconds"]),
        None,
    )
    closest_allowed_spacing = (
        None if sample_hop is None else float(min_center_gap) * float(sample_hop)
    )
    closest_allowed_overlap = (
        None
        if sample_window is None or closest_allowed_spacing is None
        else max(0.0, float(sample_window) - closest_allowed_spacing)
    )
    closest_allowed_overlap_fraction = (
        None
        if closest_allowed_overlap is None or not sample_window
        else closest_allowed_overlap / float(sample_window)
    )

    return {
        "status": "ok",
        "evidence_kind": _SAMPLING_EVIDENCE_KIND,
        "split": split,
        "steps": int(steps),
        "groups_per_step": int(groups_per_step),
        "centers_per_item": int(centers_per_item),
        "min_center_gap": int(min_center_gap),
        "activity_threshold": float(activity_threshold),
        "seed": int(seed),
        "sampling_window_seconds": sample_window,
        "sampling_hop_seconds": sample_hop,
        "closest_allowed_center_spacing_seconds": closest_allowed_spacing,
        "max_possible_context_overlap_fraction_for_closest_pair": (
            closest_allowed_overlap_fraction
        ),
        "sampled_pair_gap_seconds": _numeric_stats(all_pair_gaps),
        "sampled_group_max_context_overlap_fraction": _numeric_stats(
            all_group_overlaps
        ),
        "songs": songs,
        "skipped": skipped,
        "note": (
            "sampled windows are the actual per-step training draws; min_center_gap "
            "only forbids pairs closer than the closest allowed spacing, so the "
            "measured pair gaps here are the real sampling behaviour. The synthetic "
            "smoke corpus has short sustained bursts, so absolute coverage numbers "
            "do not transfer to the rendered corpus."
        ),
    }


# ---------------------------------------------------------------------------
# Tracker / post-processing effects
# ---------------------------------------------------------------------------


def _unit_blend(cosine: float, dim: int = _EMBEDDING_DIM) -> np.ndarray:
    value = float(np.clip(cosine, -1.0, 1.0))
    vector = np.zeros(dim, dtype=np.float64)
    vector[0] = value
    vector[1] = float(np.sqrt(max(0.0, 1.0 - value * value)))
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _basis(index: int, dim: int = _EMBEDDING_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _make_prediction(
    times: Sequence[float],
    embeddings: np.ndarray,
    activity: np.ndarray,
    slot_valid: np.ndarray,
    center_valid: np.ndarray,
    *,
    hop_seconds: float,
) -> PredictionData:
    return PredictionData(
        center_times=np.asarray(times, dtype=np.float64),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        activity=np.asarray(activity, dtype=np.float32),
        slot_valid=np.asarray(slot_valid, dtype=bool),
        center_valid=np.asarray(center_valid, dtype=bool),
        hop_seconds=hop_seconds,
    )


def _run_tracker(
    prediction: PredictionData,
    *,
    duration_seconds: float,
    config: TrackingConfig | None = None,
):
    return associate_sequence(
        prediction,
        AudioSpan(duration_seconds=float(duration_seconds), track_start_seconds=0.0),
        RunProvenance(run_id="issue24-diagnostic", data_kind="mock"),
        config=config,
    )


def _active_tracks_per_window(trajectory: Any, times: Sequence[float]) -> list[list[str]]:
    index = {round(float(value), 9): position for position, value in enumerate(times)}
    per_window: list[list[str]] = [[] for _ in times]
    for track in trajectory.tracks:
        for point_time, activity in zip(track.center_times, track.activity):
            key = round(float(point_time), 9)
            position = index.get(key)
            if position is None:
                continue
            if float(activity) > 0.5:
                per_window[position].append(track.track_id)
    return per_window


def tracker_postprocess_report() -> dict[str, Any]:
    """Separate raw E/P fixtures from tracker/post-processing decisions.

    Three controlled fixtures:

    ``slot-permutation``
        The same two identities are carried by different slots after a
        permutation; stable track ids must not follow slot numbers.
    ``silence-retention``
        A silent gap with a still-matchable embedding vs. a gap with no valid
        candidate at all (the only case where ``retention_seconds`` expires).
    ``gate-sensitivity``
        A pair at cosine 0.68: rejected at the default gate 0.7 (new track),
        accepted at 0.6 (forced link).  The test for "no loose threshold" in
        the endpoint-splicing design.
    """

    windows = 12
    times = np.arange(windows, dtype=np.float64) * 0.1
    slots = 3
    identity_a = _basis(0)
    identity_b = _basis(1)
    embeddings = np.zeros((windows, slots, _EMBEDDING_DIM), dtype=np.float32)
    activity = np.zeros((windows, slots), dtype=np.float32)
    slot_valid = np.zeros((windows, slots), dtype=bool)
    for window in range(windows):
        if window < windows // 2:
            embeddings[window, 0] = identity_a
            embeddings[window, 1] = identity_b
        else:
            embeddings[window, 0] = identity_b
            embeddings[window, 1] = identity_a
        activity[window, 0] = 1.0
        activity[window, 1] = 1.0
        slot_valid[window, 0] = True
        slot_valid[window, 1] = True
    permutation_prediction = _make_prediction(
        times, embeddings, activity, slot_valid, np.ones(windows, dtype=bool), hop_seconds=0.1
    )
    permutation_trajectory = _run_tracker(permutation_prediction, duration_seconds=1.2)
    active = _active_tracks_per_window(permutation_trajectory, times)
    permutation = {
        "scenario": "slot-permutation",
        "raw_slot_identity_sequence": [
            ("A", "B") if window < windows // 2 else ("B", "A") for window in range(windows)
        ],
        "track_ids_per_window": active,
        "raw_slot_identity_changes": 1,
        "track_count": len(permutation_trajectory.tracks),
        "single_stable_track_ids": (
            len(active[0]) == 2
            and all(len(ids) == 2 for ids in active)
            and len({tuple(sorted(ids)) for ids in active}) == 1
        ),
        "note": "slot numbers change at window 6; identity tracking must not follow them",
    }

    def retention_case(gap_windows: int, *, candidate_present: bool) -> dict[str, Any]:
        total = 2 + gap_windows + 2
        case_times = np.arange(total, dtype=np.float64) * 0.1
        case_embeddings = np.zeros((total, 1, _EMBEDDING_DIM), dtype=np.float32)
        case_activity = np.zeros((total, 1), dtype=np.float32)
        case_slot_valid = np.zeros((total, 1), dtype=bool)
        for window in range(total):
            if window < 2 or window >= total - 2:
                case_slot_valid[window, 0] = True
                case_embeddings[window, 0] = identity_a
                case_activity[window, 0] = 1.0
            elif candidate_present:
                case_slot_valid[window, 0] = True
                case_embeddings[window, 0] = identity_a
                case_activity[window, 0] = 0.0
        prediction = _make_prediction(
            case_times,
            case_embeddings,
            case_activity,
            case_slot_valid,
            np.ones(total, dtype=bool),
            hop_seconds=0.1,
        )
        trajectory = _run_tracker(prediction, duration_seconds=float(total) * 0.1)
        return {
            "gap_seconds": gap_windows * 0.1,
            "candidate_present_during_gap": candidate_present,
            "track_count": len(trajectory.tracks),
            "track_active_windows": [
                int(sum(1 for value in track.activity if value > 0.5))
                for track in trajectory.tracks
            ],
        }

    retention = {
        "scenario": "silence-retention",
        "retention_seconds": TrackingConfig().retention_seconds,
        "short_gap_with_embedding": retention_case(3, candidate_present=True),
        "long_gap_with_embedding": retention_case(20, candidate_present=True),
        "short_gap_without_candidate": retention_case(3, candidate_present=False),
        "long_gap_without_candidate": retention_case(20, candidate_present=False),
        "note": (
            "a P=0 but still-matchable candidate refreshes retention, so a stable "
            "identity memory keeps the same track across arbitrarily long silence; "
            "retention_seconds only expires tracks when no candidate passes the gate"
        ),
    }

    blend = _unit_blend(0.68)
    gate_windows = 6
    gate_times = np.arange(gate_windows, dtype=np.float64) * 0.1
    gate_embeddings = np.zeros((gate_windows, 1, _EMBEDDING_DIM), dtype=np.float32)
    gate_activity = np.ones((gate_windows, 1), dtype=np.float32)
    gate_slot_valid = np.ones((gate_windows, 1), dtype=bool)
    for window in range(gate_windows):
        gate_embeddings[window, 0] = identity_a if window < 3 else blend
    gate_prediction = _make_prediction(
        gate_times,
        gate_embeddings,
        gate_activity,
        gate_slot_valid,
        np.ones(gate_windows, dtype=bool),
        hop_seconds=0.1,
    )
    strict = _run_tracker(
        gate_prediction,
        duration_seconds=0.6,
        config=TrackingConfig(match_threshold=0.7),
    )
    loose = _run_tracker(
        gate_prediction,
        duration_seconds=0.6,
        config=TrackingConfig(match_threshold=0.6),
    )
    gate = {
        "scenario": "gate-sensitivity",
        "cosine_between_halves": 0.68,
        "track_count_at_gate_0_7": len(strict.tracks),
        "track_count_at_gate_0_6": len(loose.tracks),
        "forced_link_at_loose_gate": len(loose.tracks) == 1,
        "note": (
            "the same raw E/P pair is split or merged purely by the gate; the "
            "endpoint-splicing confirmation must not rely on a loose gate"
        ),
    }

    return {
        "status": "ok",
        "evidence_kind": _TRACKER_EVIDENCE_KIND,
        "tracker": "aat.tracking.associate_sequence",
        "default_config": TrackingConfig().as_params(),
        "scenarios": [permutation, retention, gate],
    }


# ---------------------------------------------------------------------------
# Phase B plan validation (descriptive; never runs an experiment)
# ---------------------------------------------------------------------------

_PLAN_ALLOWED_STATUS = {"proposed", "gated"}
_PLAN_ALLOWED_KINDS = {"gate", "training", "engineering-postprocess", "evaluation"}
_UNDECIDED_MARKERS = ("chosen before", "to be decided", "tbd", "to decide")
_UNDECIDED_DECISION_WORDS = (
    "choose",
    "chosen",
    "decide",
    "decided",
    "select",
    "selected",
)
_PLAN_REQUIRED_PLAN_KEYS = {
    "id",
    "issue",
    "stage",
    "authorization",
    "not_executable",
    "seed",
    "baseline_commit",
}
_PLAN_REQUIRED_EXPERIMENT_KEYS = {
    "id",
    "kind",
    "status",
    "question",
    "single_variable",
    "baseline",
    "data",
    "budget",
    "seed",
    "metrics",
    "stop_condition",
}


def _is_undecided_design(text: str) -> bool:
    """Reject a single-variable field that still offers an ``or`` choice."""

    lowered = text.lower()
    if any(marker in lowered for marker in _UNDECIDED_MARKERS):
        return True
    return " or " in lowered and any(
        word in lowered for word in _UNDECIDED_DECISION_WORDS
    )


def load_phase_b_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate the Phase B design checklist.

    The file is a **description**, not an authorization: ``not_executable``
    must be true, ``authorization`` must be pending, and every experiment must
    declare exactly one single-variable change, a budget, a seed, metrics and a
    stop condition.  Validation never starts a run.
    """

    import tomllib

    errors: list[str] = []
    try:
        with open(path, "rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {"valid": False, "errors": [f"cannot read plan: {exc}"], "experiments": []}

    plan = payload.get("plan")
    if not isinstance(plan, dict):
        errors.append("missing [plan] table")
        plan = {}
    missing = sorted(_PLAN_REQUIRED_PLAN_KEYS - set(plan))
    if missing:
        errors.append(f"[plan] missing keys: {missing}")
    if plan.get("stage") != "design-only":
        errors.append("[plan].stage must be 'design-only'")
    if plan.get("authorization") != "pending-main-session-approval":
        errors.append("[plan].authorization must be 'pending-main-session-approval'")
    if plan.get("not_executable") is not True:
        errors.append("[plan].not_executable must be true")
    if plan.get("issue") != 24:
        errors.append("[plan].issue must be 24")

    constraints = payload.get("constraints", {})
    if not isinstance(constraints, dict) or not constraints.get("fake_use"):
        errors.append("[constraints].fake_use must state the fake/mock policy")
    if constraints.get("not_model_evidence") is not True:
        errors.append("[constraints].not_model_evidence must be true")

    data_table = payload.get("data", {})
    if not isinstance(data_table, dict):
        errors.append("[data] must be a table")
    else:
        for key in ("development_validation", "historically_observed_test", "frozen_holdout"):
            value = data_table.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"[data].{key} must be a non-empty string describing the split policy"
                )

    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list) or not experiments:
        errors.append("[[experiments]] must contain at least one entry")
        experiments = []

    seen: set[str] = set()
    summary: list[dict[str, Any]] = []
    for index, experiment in enumerate(experiments):
        label = f"experiments[{index}]"
        if not isinstance(experiment, dict):
            errors.append(f"{label}: expected a table")
            continue
        missing = sorted(_PLAN_REQUIRED_EXPERIMENT_KEYS - set(experiment))
        if missing:
            errors.append(f"{label}: missing keys {missing}")
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not experiment_id:
            errors.append(f"{label}: id must be a non-empty string")
        elif experiment_id in seen:
            errors.append(f"{label}: duplicate id {experiment_id!r}")
        else:
            seen.add(experiment_id)
        status = experiment.get("status")
        if status not in _PLAN_ALLOWED_STATUS:
            errors.append(f"{label}: status must be one of {sorted(_PLAN_ALLOWED_STATUS)}")
        kind = experiment.get("kind")
        if kind not in _PLAN_ALLOWED_KINDS:
            errors.append(f"{label}: kind must be one of {sorted(_PLAN_ALLOWED_KINDS)}")
        budget = experiment.get("budget")
        if not isinstance(budget, dict) or not budget:
            errors.append(f"{label}: budget must be a non-empty table")
        else:
            upper = budget.get("gpu_minutes_upper")
            if not isinstance(upper, (int, float)) or isinstance(upper, bool) or upper < 0:
                errors.append(f"{label}: budget.gpu_minutes_upper must be a non-negative number")
            steps = budget.get("steps")
            if steps is not None and (
                not isinstance(steps, int) or isinstance(steps, bool) or steps < 0
            ):
                errors.append(f"{label}: budget.steps must be a non-negative integer")
        seed = experiment.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            errors.append(f"{label}: seed must be an integer")
        for field in ("question", "single_variable", "baseline", "data", "stop_condition"):
            value = experiment.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label}: {field} must be a non-empty string")
        single_variable = experiment.get("single_variable")
        if isinstance(single_variable, str) and _is_undecided_design(single_variable):
            errors.append(
                f"{label}: single_variable must fix one level, not an undecided "
                "choice/alternative"
            )
        baseline = experiment.get("baseline")
        if isinstance(baseline, str) and _is_undecided_design(baseline):
            errors.append(f"{label}: baseline must be fixed, not an undecided choice")
        metrics = experiment.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            errors.append(f"{label}: metrics must be a non-empty list")
        if experiment.get("executable") is True or experiment.get("run") is True:
            errors.append(f"{label}: plan must not authorize execution")
        summary.append(
            {
                "id": experiment_id,
                "kind": kind,
                "status": status,
                "single_variable": experiment.get("single_variable"),
                "baseline": experiment.get("baseline"),
                "budget": budget,
                "seed": seed,
                "metrics": metrics,
            }
        )

    return {
        "valid": not errors,
        "errors": errors,
        "plan": {
            "id": plan.get("id"),
            "issue": plan.get("issue"),
            "stage": plan.get("stage"),
            "authorization": plan.get("authorization"),
            "not_executable": plan.get("not_executable"),
            "baseline_commit": plan.get("baseline_commit"),
            "seed": plan.get("seed"),
        },
        "experiments": summary,
    }


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------


def build_issue24_report(
    *,
    include_matching: bool = True,
    include_supervision: bool = True,
    corpus_root: str | Path | None = None,
    sampling_seed: int = 20260929,
    sampling_steps: int = 10,
) -> dict[str, Any]:
    """Build the deterministic issue #24 Phase A diagnostic report.

    ``corpus_root`` is required for the sampling section (the tiny synthetic
    smoke corpus is written there, then read back by ``sample_batch``); pass
    ``None`` to omit that section.  No private, external or real audio is
    read; the only audio files are the temporary synthetic smoke WAVs created
    under ``corpus_root``, which the caller owns and is expected to delete.
    Sections that need torch report ``status = 'unavailable'`` in a base
    environment instead of raising.
    """

    sections: dict[str, Any] = {}
    sections["context_boundaries"] = context_boundary_report()
    sections["context_overlap"] = context_overlap_report()
    sections["tracker_postprocess"] = tracker_postprocess_report()
    if include_matching:
        sections["matching_ambiguity"] = matching_ambiguity_report()
    else:
        sections["matching_ambiguity"] = {
            "status": "skipped",
            "reason": "include_matching=False",
            "evidence_kind": _MATCHING_EVIDENCE_KIND,
        }
    if include_supervision:
        sections["identity_supervision"] = identity_supervision_report()
    else:
        sections["identity_supervision"] = {
            "status": "skipped",
            "reason": "include_supervision=False",
            "evidence_kind": _SUPERVISION_EVIDENCE_KIND,
        }
    if corpus_root is not None:
        from aat.training.dataset import make_smoke_dataset

        summary = make_smoke_dataset(Path(corpus_root), seed=sampling_seed, force=True)
        from aat.data import DatasetIndex

        index = DatasetIndex.load(
            summary["index_path"],
            data_root=summary["data_root"],
            verify_files=True,
        )
        sections["sampling_coverage"] = sampling_coverage_report(
            index,
            summary["data_root"],
            seed=sampling_seed,
            steps=sampling_steps,
        )
        sections["sampling_coverage"]["corpus"] = {
            "kind": "aat.training.dataset.make_smoke_dataset (synthetic)",
            "index_sha256": summary.get("fingerprint", {}).get("index_sha256"),
            "split_counts": summary.get("split_counts"),
            "leak_free": summary.get("leak_free"),
        }
    else:
        sections["sampling_coverage"] = {
            "status": "skipped",
            "reason": "corpus_root=None",
            "evidence_kind": _SAMPLING_EVIDENCE_KIND,
        }

    return {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "data_kind": DIAGNOSTIC_DATA_KIND,
        "sections": sections,
        "limitations": [
            "every fixture is synthetic; this is label/matching/post-processing "
            "structure evidence, not real model quality",
            "the sampler section writes and reads a temporary synthetic smoke WAV "
            "corpus (NumPy PCM, not a DawDreamer render); no private, external or "
            "real audio is read, and the temporary corpus is deleted by the caller",
            "head_loss values on synthetic tensors are not training results",
            "no AuT weights, checkpoints or real model forward are used",
        ],
    }
