"""鲁棒性回归（PR #37 首轮 review 复现项）。

1. 超大整数/溢出数字：三入口（validate_text/validate_file/validate_document）× 双引擎
   都必须给出**稳定带 Pointer 的拒绝**，绝不静默转 Infinity、绝不抛异常。
2. id/sha256/method 尾随换行：必须被双引擎拒绝（绝对结尾约束，含 CRLF）。
3. 超深嵌套：受控拒绝（E_PARSE/退出码 1），无 traceback。
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
sys.path.insert(0, str(BASE))

from agentic_schema.pipeline import (  # noqa: E402
    validate_document,
    validate_file,
    validate_text,
)
from agentic_schema.semantic import validate_semantic  # noqa: E402

CLI = BASE / "validate.py"
VALID = BASE / "fixtures" / "valid_minimal.json"
HUGE_INT = 10**400
MAX_FLOAT = 1.7976931348623157e308


def _has_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
    except Exception:
        return False
    return True


ENGINES = [
    pytest.param("stdlib", id="engine-stdlib"),
    pytest.param(
        "jsonschema",
        id="engine-jsonschema",
        marks=pytest.mark.skipif(not _has_jsonschema(), reason="jsonschema 未安装（见 requirements.txt pin）"),
    ),
]


def valid_doc():
    return json.loads(VALID.read_text(encoding="utf-8"))


def issues_of(report):
    return {(i.layer, i.code, i.pointer) for i in report.issues}


# ---- 1. 超大整数 / 溢出数字 -------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_huge_integer_rejected_via_text(engine: str):
    doc = valid_doc()
    doc["audio"]["sample_rate"] = HUGE_INT
    report = validate_text(json.dumps(doc), engine=engine)
    assert report.ok is False
    assert ("parse", "E_NONFINITE", "/audio/sample_rate") in issues_of(report)
    # 不静默转 Infinity：错误说明里是原值/范围，而不是 inf
    issue = [i for i in report.issues if i.pointer == "/audio/sample_rate"][0]
    assert "float64" in issue.message
    assert "得到 inf" not in issue.message


@pytest.mark.parametrize("engine", ENGINES)
def test_huge_integer_rejected_via_file(engine: str, tmp_path: Path):
    doc = valid_doc()
    doc["audio"]["sample_rate"] = HUGE_INT
    path = tmp_path / "huge.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    report = validate_file(path, engine=engine)
    assert report.ok is False
    assert ("parse", "E_NONFINITE", "/audio/sample_rate") in issues_of(report)


@pytest.mark.parametrize("engine", ENGINES)
def test_huge_integer_rejected_via_document_api(engine: str):
    doc = valid_doc()
    doc["tempo"]["bpm"] = HUGE_INT
    report = validate_document(doc, engine=engine)
    assert report.ok is False
    assert ("parse", "E_NONFINITE", "/tempo/bpm") in issues_of(report)


@pytest.mark.parametrize("engine", ENGINES)
def test_huge_integer_in_otherwise_invalid_field(engine: str):
    """字段本身另有错误（未知字段 + 缺必填）时，超大数仍稳定报 Pointer，不崩。"""
    doc = valid_doc()
    doc["audio"]["sample_rate"] = HUGE_INT
    doc["extra_note"] = "未定义字段"
    del doc["instruments"][0]["events"]
    report = validate_document(doc, engine=engine)
    assert report.ok is False
    assert ("parse", "E_NONFINITE", "/audio/sample_rate") in issues_of(report)


@pytest.mark.parametrize("engine", ENGINES)
def test_overflow_float_rejected_via_document_api(engine: str):
    doc = valid_doc()
    doc["audio"]["duration_seconds"] = float("inf")
    report = validate_document(doc, engine=engine)
    assert report.ok is False
    assert ("parse", "E_NONFINITE", "/audio/duration_seconds") in issues_of(report)


def test_semantic_layer_never_crashes_on_huge_numbers():
    """validate_semantic 直接调用（绕过 parse 层）也不得抛异常。"""
    doc = valid_doc()
    doc["audio"]["duration_seconds"] = HUGE_INT
    doc["instruments"][0]["events"][0]["onset_seconds"] = HUGE_INT
    doc["instruments"][0]["events"][0]["duration_seconds"] = -HUGE_INT
    doc["instruments"][0]["confidence"] = HUGE_INT
    doc["tempo"]["bpm"] = HUGE_INT
    issues = validate_semantic(doc)
    pointers = {i.pointer for i in issues}
    assert "/audio/duration_seconds" in pointers
    assert "/instruments/0/events/0/onset_seconds" in pointers
    assert all(i.code in ("E_FINITE", "E_RANGE", "E_TIME_RANGE", "E_TIME_DURATION", "E_TEMPO") for i in issues)


@pytest.mark.parametrize("engine", ENGINES)
def test_semantic_checks_do_not_crash_after_structure_failure(engine: str):
    """结构失败（类型错/缺字段/未知字段）后语义层照常运行且不崩。"""
    doc = valid_doc()
    doc["audio"] = "not-an-object"  # 结构失败
    doc["instruments"][0]["events"] = {"not": "a list"}  # 结构失败
    doc["tempo"]["confidence"] = True  # 布尔冒充数字
    doc["unknown_field"] = 1
    report = validate_document(doc, engine=engine)
    assert report.ok is False
    assert any(i.layer == "structure" for i in report.issues)
    assert ("semantic", "E_BOOL_NUMBER", "/tempo/confidence") in issues_of(report)


def test_numeric_range_policy_boundary():
    """float64 上限内的大数合法，越界拒绝（范围规范回归）。"""
    doc = valid_doc()
    doc["audio"]["duration_seconds"] = MAX_FLOAT
    assert validate_text(json.dumps(doc), engine="stdlib").ok is True
    doc["audio"]["duration_seconds"] = 2**1024  # 整数越界（略超 float64 上限）
    report = validate_text(json.dumps(doc), engine="stdlib")
    assert report.ok is False
    assert ("parse", "E_NONFINITE", "/audio/duration_seconds") in issues_of(report)


# ---- 2. id/sha256/method 尾随换行（$ 漏洞） ----------------------------


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("suffix", ["\n", "\r\n", "\n\n"])
def test_identifier_hash_method_trailing_newline_rejected(engine: str, suffix: str):
    base = valid_doc()
    cases = [
        ("instruments", suffix, "/instruments/0/id", lambda d: d["instruments"][0].__setitem__("id", "inst-1" + suffix)),
        ("audio", suffix, "/audio/sha256", lambda d: d["audio"].__setitem__("sha256", "0" * 64 + suffix)),
        (
            "method",
            suffix,
            "/instruments/0/events/0/method",
            lambda d: d["instruments"][0]["events"][0].__setitem__("method", "spectral-flux" + suffix),
        ),
    ]
    for _name, _sfx, pointer, mutate in cases:
        doc = copy.deepcopy(base)
        mutate(doc)
        report = validate_text(json.dumps(doc), engine=engine)
        assert report.ok is False, f"{pointer!r} 尾随 {suffix!r} 必须被拒绝（engine={engine}）"
        assert any(
            i.layer == "structure" and i.code == "pattern" and i.pointer == pointer for i in report.issues
        ), f"{pointer}: 期望 pattern 错误，实际 {[i.format() for i in report.issues]}"


@pytest.mark.parametrize("engine", ENGINES)
def test_identifier_embedded_newline_rejected(engine: str):
    doc = valid_doc()
    doc["instruments"][0]["id"] = "inst\n-1"
    report = validate_text(json.dumps(doc), engine=engine)
    assert report.ok is False
    assert ("structure", "pattern", "/instruments/0/id") in issues_of(report)


@pytest.mark.parametrize("engine", ENGINES)
def test_clean_identifier_hash_method_still_accepted(engine: str):
    doc = valid_doc()
    doc["instruments"][0]["events"][0]["method"] = "spectral-flux"
    assert validate_text(json.dumps(doc), engine=engine).ok is True


def test_patterns_use_absolute_end_not_dollar():
    schema = json.loads((BASE / "schema" / "agentic-audio-tracks-v1.schema.json").read_text(encoding="utf-8"))
    patterns = [
        schema["$defs"]["identifier"]["pattern"],
        schema["$defs"]["sha256"]["pattern"],
        schema["properties"]["instruments"]["items"]["$ref"],  # 触达 $defs 引用链
    ]
    assert schema["$defs"]["identifier"]["pattern"].endswith(r"(?![\s\S])")
    assert schema["$defs"]["sha256"]["pattern"].endswith(r"(?![\s\S])")
    event = schema["$defs"]["event"]["properties"]["method"]["pattern"]
    assert event.endswith(r"(?![\s\S])")
    for pattern in (schema["$defs"]["identifier"]["pattern"], schema["$defs"]["sha256"]["pattern"], event):
        assert not pattern.endswith("$"), "$ 在 Python/JS 会放过末尾换行"
    assert patterns[2] == "#/$defs/instrument"


# ---- 3. 超深嵌套 ------------------------------------------------------


def test_extreme_nesting_text_is_controlled():
    report = validate_text("[" * 1200 + "0" + "]" * 1200, engine="stdlib")
    assert report.ok is False
    assert any(i.layer == "parse" and i.code == "E_PARSE" for i in report.issues)


def test_over_depth_document_rejected_with_pointer():
    deep = "[" * 70 + "0" + "]" * 70  # 合法 JSON，但深度 > 64
    report = validate_text(deep, engine="stdlib")
    assert report.ok is False
    issue = [i for i in report.issues if i.code == "E_PARSE"][0]
    assert "嵌套过深" in issue.message


def test_deep_document_api_is_controlled():
    node = 0
    for _ in range(70):
        node = [node]
    report = validate_document(node, engine="stdlib")
    assert report.ok is False
    assert any(i.code == "E_PARSE" for i in report.issues)


def test_deep_object_document_api_is_controlled():
    node: object = {"leaf": 1}
    for _ in range(70):
        node = {"next": node}
    report = validate_document(node, engine="stdlib")
    assert report.ok is False
    assert any(i.code == "E_PARSE" for i in report.issues)


def test_depth_64_is_the_documented_boundary():
    from agentic_schema.loader import load_text

    node = 0
    for _ in range(63):  # 根=1，数字落在第 64 层：允许
        node = [node]
    data, issues = load_text(json.dumps(node))
    assert issues == [] and data is not None
    node = [node]  # 数字落在第 65 层：拒绝
    data, issues = load_text(json.dumps(node))
    assert data is None
    assert issues[0].code == "E_PARSE" and "嵌套过深" in issues[0].message


def test_cli_deep_input_exits_nonzero_without_traceback(tmp_path: Path):
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 1200 + "0" + "]" * 1200, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), str(deep)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(tmp_path),
        timeout=120,
    )
    assert result.returncode == 1, (result.returncode, result.stdout, result.stderr)
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined
    assert "E_PARSE" in combined


def test_cli_huge_integer_file_exits_nonzero_without_traceback(tmp_path: Path):
    doc = valid_doc()
    doc["audio"]["sample_rate"] = HUGE_INT
    path = tmp_path / "huge.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CLI), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(tmp_path),
        timeout=120,
    )
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined
    assert "/audio/sample_rate" in combined and "E_NONFINITE" in combined
