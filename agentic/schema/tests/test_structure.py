"""结构校验测试（零依赖 stdlib 引擎）+ schema 便携子集契约。"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
sys.path.insert(0, str(BASE))

from agentic_schema.pipeline import SCHEMA_PATH, SCHEMA_VERSION, load_schema  # noqa: E402
from agentic_schema.structure import validate_structure  # noqa: E402

#: schema 允许使用的关键字（便携子集；出现别的关键字=浏览器兼容风险）
PORTABLE_KEYWORDS = {
    "$schema",
    "$id",
    "$defs",
    "$ref",
    "title",
    "description",
    "type",
    "const",
    "enum",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "pattern",
    "minimum",
    "exclusiveMinimum",
    "maximum",
    "minLength",
    "maxLength",
    "minItems",
}


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture()
def doc():
    return json.loads((BASE / "fixtures" / "valid_minimal.json").read_text(encoding="utf-8"))


def struct_codes(issues):
    return {(i.code, i.pointer) for i in issues}


def test_valid_minimal_passes(schema, doc):
    assert validate_structure(doc, schema) == []


def test_schema_uses_portable_subset_only(schema):
    used: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$defs", "properties"):
                    for sub in value.values():
                        walk(sub)
                elif not key.startswith("$") and key not in ("title", "description", "examples"):
                    used.add(key)
                    walk(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)
    assert used <= PORTABLE_KEYWORDS, f"schema 用到了不可移植关键字: {sorted(used - PORTABLE_KEYWORDS)}"


def test_version_const_is_frozen(schema):
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_type_error_bool_is_not_number(schema, doc):
    doc["tempo"]["bpm"] = True
    issues = validate_structure(doc, schema)
    assert ("type", "/tempo/bpm") in struct_codes(issues)


def test_type_error_string_for_number(schema, doc):
    doc["audio"]["duration_seconds"] = "12.0"
    assert ("type", "/audio/duration_seconds") in struct_codes(validate_structure(doc, schema))


def test_integer_accepts_integral_float_but_not_fraction(schema, doc):
    doc["audio"]["sample_rate"] = 44100.0
    assert validate_structure(doc, schema) == []
    doc["audio"]["sample_rate"] = 44100.5
    assert ("type", "/audio/sample_rate") in struct_codes(validate_structure(doc, schema))


def test_missing_required_field(schema, doc):
    del doc["audio"]["sha256"]
    assert ("required", "/audio") in struct_codes(validate_structure(doc, schema))


def test_unknown_field_rejected_everywhere(schema, doc):
    doc["instruments"][0]["stem"] = {"filename": "stems/a.wav", "sha256": "0" * 64, "extra": 1}
    assert ("additionalProperties", "/instruments/0/stem") in struct_codes(validate_structure(doc, schema))


def test_pattern_rejects_bad_id(schema, doc):
    doc["instruments"][0]["id"] = "钢琴"
    assert ("pattern", "/instruments/0/id") in struct_codes(validate_structure(doc, schema))


def test_ranges_checked(schema, doc):
    doc["instruments"][0]["confidence"] = 1.0001
    assert ("maximum", "/instruments/0/confidence") in struct_codes(validate_structure(doc, schema))
    doc["instruments"][0]["confidence"] = -0.1
    assert ("minimum", "/instruments/0/confidence") in struct_codes(validate_structure(doc, schema))
    doc["instruments"][0]["confidence"] = 0.5
    doc["tempo"]["bpm"] = 0
    assert ("exclusiveMinimum", "/tempo/bpm") in struct_codes(validate_structure(doc, schema))
    doc["tempo"]["bpm"] = 1001
    assert ("maximum", "/tempo/bpm") in struct_codes(validate_structure(doc, schema))


def test_nullable_bpm_accepts_null_but_not_string(schema, doc):
    doc["tempo"]["bpm"] = None
    assert validate_structure(doc, schema) == []
    doc["tempo"]["bpm"] = "120"
    assert ("type", "/tempo/bpm") in struct_codes(validate_structure(doc, schema))


def test_min_items_and_required_in_provenance(schema, doc):
    doc["provenance"]["steps"] = []
    assert ("minItems", "/provenance/steps") in struct_codes(validate_structure(doc, schema))
    doc["provenance"] = {}
    assert ("required", "/provenance") in struct_codes(validate_structure(doc, schema))


def test_pitch_requires_all_evidence_fields(schema, doc):
    doc["instruments"][0]["events"][0]["pitch"] = {"midi": 60, "source": "llm"}
    assert ("required", "/instruments/0/events/0/pitch") in struct_codes(validate_structure(doc, schema))


def test_sha256_pattern(schema, doc):
    doc["audio"]["sha256"] = "XYZ"
    assert ("pattern", "/audio/sha256") in struct_codes(validate_structure(doc, schema))


def test_const_rejects_wrong_version(schema, doc):
    doc["schema_version"] = "agentic-audio-tracks/v2"
    assert ("const", "/schema_version") in struct_codes(validate_structure(doc, schema))


def test_empty_instruments_and_events_allowed(schema):
    doc = json.loads((BASE / "fixtures" / "valid_empty_instruments.json").read_text(encoding="utf-8"))
    assert validate_structure(doc, schema) == []
    doc2 = copy.deepcopy(doc)
    doc2["instruments"] = [
        {
            "id": "inst-1",
            "label": "unknown",
            "description": "无事件来源",
            "source": "unknown",
            "confidence": 0.1,
            "events": [],
        }
    ]
    assert validate_structure(doc2, schema) == []


def test_schema_file_is_itself_parseable_draft2020():
    raw = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert raw["$schema"] == "https://json-schema.org/draft/2020-12/schema"
