"""文档/规则契约测试：README 与 semantic_rules.json 必须和代码保持一致。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
REPO = BASE.parents[1]
sys.path.insert(0, str(BASE))

from agentic_schema.pipeline import SCHEMA_PATH, SCHEMA_VERSION, SEMANTIC_RULES_PATH  # noqa: E402
from agentic_schema.semantic import SEMANTIC_CODES  # noqa: E402


def test_semantic_rules_match_code():
    rules = json.loads(SEMANTIC_RULES_PATH.read_text(encoding="utf-8"))
    documented = sorted({rule["code"] for rule in rules["rules"] if rule["layer"] == "semantic"})
    assert documented == list(SEMANTIC_CODES)
    for rule in rules["rules"]:
        assert rule["rule"].strip(), rule["code"]
        assert rule["pointers"], rule["code"]


def test_semantic_rules_cover_parse_and_version_layers():
    rules = json.loads(SEMANTIC_RULES_PATH.read_text(encoding="utf-8"))
    layers = {rule["layer"] for rule in rules["rules"]}
    assert {"parse", "version", "semantic"} <= layers


def test_semantic_rules_document_policies():
    rules = json.loads(SEMANTIC_RULES_PATH.read_text(encoding="utf-8"))
    assert rules["max_depth"] == 64
    assert "float64" in rules["numeric_policy"]
    assert "(?![" in rules["pattern_policy"]
    layers = {rule["layer"] for rule in rules["rules"]}
    assert "structure" in layers  # pattern 绝对结尾契约也机器可读


def test_schema_id_and_version_agree():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["title"] == SCHEMA_VERSION
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert SCHEMA_PATH.name.endswith("v1.schema.json")


def test_readme_documents_frozen_interface():
    readme = (BASE / "README.md").read_text(encoding="utf-8")
    for token in (
        SCHEMA_VERSION,
        "python agentic/schema/validate.py",
        "agentic-audio-tracks-v1.schema.json",
        "semantic_rules.json",
        "outputs/result.json",
        "E_PATH",
        "E_TIME_ORDER",
        "E_TEMPO",
        "uv run",
        "退出码",
    ):
        assert token in readme, f"README.md 缺少冻结接口说明: {token}"


def test_readme_documents_numeric_depth_pattern_policy():
    readme = (BASE / "README.md").read_text(encoding="utf-8")
    for token in ("float64", "1.7976931348623157e308", "嵌套过深", "64", r"(?![\s\S])", "E_NONFINITE"):
        assert token in readme, f"README.md 缺少数值/深度/pattern 政策说明: {token}"


def test_readme_states_limits():
    readme = (BASE / "README.md").read_text(encoding="utf-8")
    for token in ("不是 MIDI", "不承诺", "秒"):
        assert token in readme, f"README.md 缺少限制说明: {token}"


def test_no_old_protocol_touched():
    """不改旧 anonymous E/P 协议：本目录不得出现旧 schema 文件的重定义。"""
    old_docs = REPO / "docs" / "SCHEMAS.md"
    assert old_docs.is_file()  # 旧文档仍在，且本目录只新增自己的协议
    new_files = {p.name for p in (BASE / "schema").glob("*")}
    assert new_files == {"agentic-audio-tracks-v1.schema.json", "semantic_rules.json"}
