"""fixture 契约测试：正例全通过、负例命中 expected 错误、纯 JSON、无私有信息。"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
sys.path.insert(0, str(BASE))

from agentic_schema.pipeline import validate_file  # noqa: E402

FIXTURES = BASE / "fixtures"
NEGATIVE = FIXTURES / "negative"

#: 公开 fixture 不允许出现的私有信息
SECRET_PATTERNS = (
    (re.compile(r"[A-Za-z]:\\+Users\\+"), "personal home path"),
    (re.compile(r"MSI" + "-NB"), "local username"),
    (re.compile(r"F:[\\\\/]LED"), "local drive path"),
    (re.compile(r"sk-" + r"[A-Za-z0-9_-]{20,}"), "API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
)


def valid_files():
    return sorted(p for p in FIXTURES.glob("*.json"))


def negative_files():
    return sorted(p for p in NEGATIVE.glob("*.json") if p.name != "expected.json")


@pytest.fixture(scope="module")
def expected_cases():
    data = json.loads((NEGATIVE / "expected.json").read_text(encoding="utf-8"))
    return data["cases"]


def test_expected_index_matches_negative_files(expected_cases):
    indexed = sorted(case["file"] for case in expected_cases)
    on_disk = sorted(p.name for p in negative_files())
    assert indexed == on_disk, "expected.json 与 negative/ 目录不一致"


@pytest.mark.parametrize("path", valid_files(), ids=lambda p: p.name)
def test_valid_fixtures_pass(path: Path):
    report = validate_file(path, engine="stdlib")
    assert report.ok, f"{path.name} 应通过: {[i.format() for i in report.issues]}"
    report2 = validate_file(path, engine="jsonschema") if _has_jsonschema() else None
    if report2 is not None:
        assert report2.ok, f"{path.name} 应通过（jsonschema 引擎）: {[i.format() for i in report2.issues]}"


def _has_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
    except Exception:
        return False
    return True


def test_negative_fixtures_rejected_with_expected_error(expected_cases):
    by_file = {case["file"]: case for case in expected_cases}
    for path in negative_files():
        case = by_file[path.name]
        report = validate_file(path, engine="stdlib")
        assert not report.ok, f"{path.name} 必须被拒绝"
        got = {(i.layer, i.code, i.pointer) for i in report.issues}
        want = (case["layer"], case["code"], case["pointer"])
        assert want in got, f"{path.name} 应命中 {want}，实际 {sorted(got)}（{case['why']}）"


@pytest.mark.parametrize("path", valid_files() + negative_files(), ids=lambda p: p.name)
def test_fixtures_are_strict_json_syntax(path: Path):
    """fixture 必须是纯 JSON 语法（无 NaN/Infinity 字面量；溢出用 1e999 表达）。"""
    text = path.read_text(encoding="utf-8")

    def reject(name):
        raise AssertionError(f"{path.name} 含非标准常量 {name}")

    json.loads(text, parse_constant=reject)


@pytest.mark.parametrize("path", valid_files() + negative_files(), ids=lambda p: p.name)
def test_fixtures_carry_no_private_information(path: Path):
    text = path.read_text(encoding="utf-8")
    for pattern, label in SECRET_PATTERNS:
        assert pattern.search(text) is None, f"{path.name} 含 {label}"


@pytest.mark.parametrize("path", valid_files(), ids=lambda p: p.name)
def test_valid_fixture_filenames_are_relative(path: Path):
    """正例的 audio.filename 必须是安全相对路径（负例故意写坏，不在此检查）。"""
    from agentic_schema.semantic import unsafe_path_reason

    data = json.loads(path.read_text(encoding="utf-8"))
    assert unsafe_path_reason(data["audio"]["filename"]) is None


def test_fixture_lists_are_nonempty():
    assert len(valid_files()) >= 4
    assert len(negative_files()) >= 15
