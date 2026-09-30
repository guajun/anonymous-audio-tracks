"""严格 JSON 载入器测试：NaN/Inf 字面量、重复键、溢出数字、编码问题全部拒绝。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
sys.path.insert(0, str(BASE))

from agentic_schema.loader import load_path, load_text  # noqa: E402


def codes(issues):
    return [(i.layer, i.code, i.pointer) for i in issues]


def test_good_text_parses():
    data, issues = load_text('{"a": [1, 2.5, null, true], "b": "x"}')
    assert issues == []
    assert data == {"a": [1, 2.5, None, True], "b": "x"}


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_constants_rejected(token):
    data, issues = load_text(f'{{"a": {token}}}')
    assert data is None
    assert codes(issues) == [("parse", "E_PARSE", "")]
    assert token in issues[0].message


def test_duplicate_key_rejected_not_last_wins():
    data, issues = load_text('{"a": 1, "a": 2}')
    assert data is None
    assert codes(issues) == [("parse", "E_DUPLICATE_KEY", "")]
    assert "'a'" in issues[0].message


def test_duplicate_key_inside_nested_object_rejected():
    data, issues = load_text('{"outer": {"b": 1, "b": 2}}')
    assert data is None
    assert issues[0].code == "E_DUPLICATE_KEY"


@pytest.mark.parametrize(
    ("snippet", "pointer"),
    [
        ('{"audio": {"duration_seconds": 1e999}}', "/audio/duration_seconds"),
        ('{"tempo": {"bpm": -1e999}}', "/tempo/bpm"),
        ('{"instruments": [{"events": [{"onset_seconds": 1e400}]}]}', "/instruments/0/events/0/onset_seconds"),
    ],
)
def test_overflow_float_rejected_with_pointer(snippet, pointer):
    """1e999 语法合法，但解析成 inf —— 必须拒绝并给出精确 JSON Pointer。"""
    data, issues = load_text(snippet)
    assert data is None
    assert codes(issues) == [("parse", "E_NONFINITE", pointer)]


def test_malformed_json_rejected_with_position():
    data, issues = load_text('{"a": }')
    assert data is None
    assert issues[0].code == "E_PARSE"
    assert "第 1 行" in issues[0].message


def test_bom_rejected():
    data, issues = load_text('\ufeff{"schema_version": "agentic-audio-tracks/v1"}')
    assert data is None
    assert issues[0].code == "E_PARSE"


def test_non_utf8_file_rejected(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b'{"a": "\xff\xfe"}')
    data, issues = load_path(bad)
    assert data is None
    assert issues[0].code == "E_PARSE"
    assert "UTF-8" in issues[0].message


def test_missing_file_reports_parse_error(tmp_path: Path):
    data, issues = load_path(tmp_path / "nope.json")
    assert data is None
    assert issues[0].code == "E_PARSE"
