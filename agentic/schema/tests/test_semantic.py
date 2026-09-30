"""跨字段语义测试：时间边界、排序、ID、路径安全、tempo 策略、范围兜底。"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
sys.path.insert(0, str(BASE))

from agentic_schema.semantic import (  # noqa: E402
    SEMANTIC_CODES,
    unsafe_path_reason,
    validate_semantic,
)


def codes(issues):
    return {(i.code, i.pointer) for i in issues}


@pytest.fixture()
def doc():
    return json.loads((BASE / "fixtures" / "valid_minimal.json").read_text(encoding="utf-8"))


def test_valid_doc_has_no_semantic_issues():
    for name in ("valid_minimal.json", "valid_full.json", "valid_multi_instrument.json"):
        data = json.loads((BASE / "fixtures" / name).read_text(encoding="utf-8"))
        assert validate_semantic(data) == [], name


# ---- 时间边界 ---------------------------------------------------------


def test_time_boundaries_zero_end_and_equal_allowed(doc):
    doc["instruments"][0]["events"] = [
        {"id": "ev-1", "onset_seconds": 0.0},
        {"id": "ev-2", "onset_seconds": 11.0, "duration_seconds": 1.0},  # 11+1 == 12，允许相等
        {"id": "ev-3", "onset_seconds": 12.0},  # 恰好落在文件末尾，允许
    ]
    assert validate_semantic(doc) == []


def test_onset_after_end_rejected(doc):
    doc["instruments"][0]["events"] = [{"id": "ev-1", "onset_seconds": 12.0000001}]
    assert ("E_TIME_RANGE", "/instruments/0/events/0/onset_seconds") in codes(validate_semantic(doc))


def test_negative_onset_rejected_semantically(doc):
    doc["instruments"][0]["events"] = [{"id": "ev-1", "onset_seconds": -0.001}]
    assert ("E_TIME_RANGE", "/instruments/0/events/0/onset_seconds") in codes(validate_semantic(doc))


def test_duration_overflow_rejected(doc):
    doc["instruments"][0]["events"] = [{"id": "ev-1", "onset_seconds": 11.5, "duration_seconds": 0.6}]
    assert ("E_TIME_DURATION", "/instruments/0/events/0/duration_seconds") in codes(validate_semantic(doc))


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_duration_rejected(doc, value):
    doc["instruments"][0]["events"] = [{"id": "ev-1", "onset_seconds": 1.0, "duration_seconds": value}]
    assert ("E_TIME_DURATION", "/instruments/0/events/0/duration_seconds") in codes(validate_semantic(doc))


def test_tolerance_1e9_on_boundary(doc):
    doc["instruments"][0]["events"] = [{"id": "ev-1", "onset_seconds": 11.0, "duration_seconds": 1.0 + 5e-10}]
    assert validate_semantic(doc) == []


def test_unsorted_events_rejected(doc):
    doc["instruments"][0]["events"] = [
        {"id": "ev-1", "onset_seconds": 2.0},
        {"id": "ev-2", "onset_seconds": 1.0},
    ]
    assert ("E_TIME_ORDER", "/instruments/0/events/1") in codes(validate_semantic(doc))


def test_simultaneous_events_sorted_by_id(doc):
    doc["instruments"][0]["events"] = [
        {"id": "ev-a", "onset_seconds": 1.0},
        {"id": "ev-b", "onset_seconds": 1.0},
        {"id": "ev-c", "onset_seconds": 1.0},
    ]
    assert validate_semantic(doc) == []


def test_simultaneous_events_wrong_id_order_rejected(doc):
    doc["instruments"][0]["events"] = [
        {"id": "ev-b", "onset_seconds": 1.0},
        {"id": "ev-a", "onset_seconds": 1.0},
    ]
    assert ("E_TIME_ORDER", "/instruments/0/events/1") in codes(validate_semantic(doc))


# ---- ID 唯一 ----------------------------------------------------------


def test_duplicate_instrument_id_rejected(doc):
    second = copy.deepcopy(doc["instruments"][0])
    second["id"] = "inst-1"
    second["events"] = []
    doc["instruments"].append(second)
    assert ("E_ID_UNIQUE", "/instruments/1/id") in codes(validate_semantic(doc))


def test_duplicate_event_id_across_instruments_rejected(doc):
    second = copy.deepcopy(doc["instruments"][0])
    second["id"] = "inst-2"
    second["events"] = [{"id": "inst-1-ev-1", "onset_seconds": 9.0}]
    doc["instruments"].append(second)
    assert ("E_ID_UNIQUE", "/instruments/1/events/0/id") in codes(validate_semantic(doc))


def test_same_label_different_ids_allowed(doc):
    second = copy.deepcopy(doc["instruments"][0])
    second["id"] = "inst-2"
    second["events"] = [{"id": "inst-2-ev-1", "onset_seconds": 9.0}]
    doc["instruments"].append(second)
    assert validate_semantic(doc) == []


# ---- tempo 未知/已知互斥 -----------------------------------------------


def test_unknown_tempo_policy():
    data = json.loads((BASE / "fixtures" / "valid_unknown_tempo.json").read_text(encoding="utf-8"))
    assert validate_semantic(data) == []


def test_null_bpm_with_known_source_rejected(doc):
    doc["tempo"] = {"bpm": None, "source": "dsp", "confidence": 0.5}
    got = codes(validate_semantic(doc))
    assert ("E_TEMPO", "/tempo/source") in got
    assert ("E_TEMPO", "/tempo/confidence") in got


def test_null_bpm_with_beat_origin_rejected(doc):
    doc["tempo"] = {"bpm": None, "source": "unknown", "confidence": 0, "beat_origin_seconds": 0.5}
    assert ("E_TEMPO", "/tempo/beat_origin_seconds") in codes(validate_semantic(doc))


def test_known_bpm_with_unknown_source_rejected(doc):
    doc["tempo"] = {"bpm": 120.0, "source": "unknown", "confidence": 0.5}
    assert ("E_TEMPO", "/tempo/source") in codes(validate_semantic(doc))


def test_beat_origin_must_stay_in_audio(doc):
    doc["tempo"] = {"bpm": 120.0, "source": "mock", "confidence": 0.5, "beat_origin_seconds": 12.5}
    assert ("E_TIME_RANGE", "/tempo/beat_origin_seconds") in codes(validate_semantic(doc))
    doc["tempo"]["beat_origin_seconds"] = 12.0
    assert validate_semantic(doc) == []


# ---- 路径安全 ---------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/music/a.wav",
        "C:/music/a.wav",
        "C:a.wav",
        "\\\\server\\share\\a.wav",
        "//server/share/a.wav",
        "https://example.com/a.wav",
        "file://a.wav",
        "../secret/a.wav",
        "stems/../../etc/a.wav",
        "./a.wav",
        "stems//a.wav",
        "stems/a.wav/",
        "stems\\a.wav",
        "a ? b.wav",
        "a\x00b.wav",
        "con.wav",
        "stems/NUL.wav",
        "ends-with-dot./a.wav",
        "ends-with-space /a.wav",
        "x" * 513,
    ],
)
def test_unsafe_paths_rejected(value):
    assert unsafe_path_reason(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "a.wav",
        "stems/mock-trio.piano.wav",
        "dir1/dir2/a-b_c.1.wav",
        "COM10.wav",
        "console.wav",
    ],
)
def test_safe_paths_allowed(value):
    assert unsafe_path_reason(value) is None


def test_path_check_hits_both_filename_slots(doc):
    doc["audio"]["filename"] = "../a.wav"
    doc["instruments"][0]["stem"] = {"filename": "C:/stems/a.wav", "sha256": "0" * 64}
    got = codes(validate_semantic(doc))
    assert ("E_PATH", "/audio/filename") in got
    assert ("E_PATH", "/instruments/0/stem/filename") in got


# ---- 范围/布尔/有限数兜底 ----------------------------------------------


def test_confidence_out_of_range_rejected_semantically(doc):
    doc["instruments"][0]["confidence"] = 1.2
    assert ("E_RANGE", "/instruments/0/confidence") in codes(validate_semantic(doc))
    doc["tempo"]["confidence"] = -0.1
    assert ("E_RANGE", "/tempo/confidence") in codes(validate_semantic(doc))


def test_pitch_midi_range_and_int(doc):
    doc["instruments"][0]["events"][0]["pitch"] = {"midi": 200, "confidence": 0.5, "source": "llm"}
    assert ("E_RANGE", "/instruments/0/events/0/pitch/midi") in codes(validate_semantic(doc))
    doc["instruments"][0]["events"][0]["pitch"] = {"midi": 60.5, "confidence": 0.5, "source": "llm"}
    assert ("E_RANGE", "/instruments/0/events/0/pitch/midi") in codes(validate_semantic(doc))


def test_bool_masquerading_number_rejected(doc):
    doc["instruments"][0]["events"][0]["onset_seconds"] = True
    assert ("E_BOOL_NUMBER", "/instruments/0/events/0/onset_seconds") in codes(validate_semantic(doc))


def test_nonfinite_rejected_semantically(doc):
    doc["audio"]["duration_seconds"] = math.inf
    assert ("E_FINITE", "/audio/duration_seconds") in codes(validate_semantic(doc))


def test_empty_instruments_semantically_ok():
    data = json.loads((BASE / "fixtures" / "valid_empty_instruments.json").read_text(encoding="utf-8"))
    assert validate_semantic(data) == []


def test_semantic_codes_are_frozen():
    assert SEMANTIC_CODES == tuple(sorted(SEMANTIC_CODES))
    assert "E_PATH" in SEMANTIC_CODES and "E_TIME_ORDER" in SEMANTIC_CODES
