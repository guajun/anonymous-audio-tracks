"""双引擎一致性：零依赖 stdlib 结构校验 与 jsonschema（若安装）必须同进退。

jsonschema 未安装时本文件跳过（显式 skip，不是通过）；安装 pin 的依赖后应 0 skip。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
sys.path.insert(0, str(BASE))

jsonschema = pytest.importorskip("jsonschema", reason="jsonschema 未安装（见 requirements.txt pin）")

from agentic_schema.pipeline import load_schema, validate_file  # noqa: E402
from agentic_schema.schema_check import validate_with_jsonschema  # noqa: E402
from agentic_schema.structure import validate_structure  # noqa: E402

FIXTURES = BASE / "fixtures"


def all_fixture_files():
    return sorted(FIXTURES.glob("*.json")) + sorted(
        p for p in (FIXTURES / "negative").glob("*.json") if p.name != "expected.json"
    )


@pytest.mark.parametrize("path", all_fixture_files(), ids=lambda p: p.name)
def test_engines_agree_on_accept_reject(path: Path):
    stdlib = validate_file(path, engine="stdlib")
    official = validate_file(path, engine="jsonschema")
    assert stdlib.ok == official.ok, (
        f"{path.name}: stdlib={'OK' if stdlib.ok else 'FAIL'} vs "
        f"jsonschema={'OK' if official.ok else 'FAIL'}"
    )


@pytest.mark.parametrize(
    "fixture",
    [
        "neg_unknown_field.json",
        "neg_bad_id_pattern.json",
        "neg_bool_number.json",
        "neg_confidence_range.json",
        "neg_negative_onset.json",
        "neg_missing_events.json",
        "neg_pitch_missing_confidence.json",
    ],
)
def test_engines_report_same_structure_errors(fixture: str):
    data = json.loads((FIXTURES / "negative" / fixture).read_text(encoding="utf-8"))
    schema = load_schema()
    mine = {(i.code, i.pointer) for i in validate_structure(data, schema)}
    theirs = {(i.code, i.pointer) for i in validate_with_jsonschema(data, schema)}
    assert mine == theirs, f"{fixture}: stdlib {sorted(mine)} vs jsonschema {sorted(theirs)}"
