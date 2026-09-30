"""CLI 测试：退出码、错误定位、--json、cwd 无关（在临时目录里运行）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
CLI = BASE / "validate.py"
VALID = BASE / "fixtures" / "valid_minimal.json"
NEG_ONSET = BASE / "fixtures" / "negative" / "neg_onset_after_end.json"
NEG_PATH = BASE / "fixtures" / "negative" / "neg_path_traversal.json"


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
        timeout=120,
    )


def test_valid_fixture_exits_zero(tmp_path: Path):
    result = run_cli(str(VALID), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    assert "engine=" in result.stdout


def test_negative_fixture_exits_one_and_points_to_field(tmp_path: Path):
    result = run_cli(str(NEG_ONSET), cwd=tmp_path)
    assert result.returncode == 1
    assert "FAIL" in result.stdout
    assert "/instruments/0/events/0/onset_seconds" in result.stdout
    assert "E_TIME_RANGE" in result.stdout


def test_path_error_mentions_pointer(tmp_path: Path):
    result = run_cli(str(NEG_PATH), cwd=tmp_path)
    assert result.returncode == 1
    assert "/instruments/0/stem/filename" in result.stdout
    assert "E_PATH" in result.stdout


def test_multiple_files_report_each(tmp_path: Path):
    result = run_cli(str(VALID), str(NEG_ONSET), cwd=tmp_path)
    assert result.returncode == 1
    assert "OK" in result.stdout and "FAIL" in result.stdout


def test_json_output_is_machine_readable(tmp_path: Path):
    result = run_cli("--json", str(NEG_ONSET), cwd=tmp_path)
    assert result.returncode == 1
    payload = json.loads(result.stdout.splitlines()[0])
    assert payload["ok"] is False
    assert payload["schema_version"] == "agentic-audio-tracks/v1"
    assert payload["engine"] in ("stdlib", "jsonschema")
    issue = payload["issues"][0]
    assert set(issue) == {"layer", "code", "pointer", "message"}
    assert issue["pointer"] == "/instruments/0/events/0/onset_seconds"


def test_missing_file_exits_two(tmp_path: Path):
    result = run_cli(str(tmp_path / "nope.json"), cwd=tmp_path)
    assert result.returncode == 2
    assert "error" in result.stderr.lower() or "error" in result.stdout.lower()


def test_no_arguments_exits_two(tmp_path: Path):
    result = run_cli(cwd=tmp_path)
    assert result.returncode == 2


def test_explicit_stdlib_engine(tmp_path: Path):
    result = run_cli("--engine", "stdlib", str(VALID), cwd=tmp_path)
    assert result.returncode == 0
    assert "engine=stdlib" in result.stdout


def test_cwd_independence(tmp_path: Path):
    """换任意目录运行结果一致（不写死 cwd，不依赖私有路径）。"""
    here = run_cli(str(NEG_ONSET), cwd=tmp_path)
    elsewhere = run_cli(str(NEG_ONSET), cwd=BASE)
    assert here.returncode == elsewhere.returncode == 1
    assert here.stdout.splitlines()[1:] == elsewhere.stdout.splitlines()[1:]


def test_jsonschema_engine_request():
    try:
        import jsonschema  # noqa: F401

        available = True
    except Exception:
        available = False
    if not available:
        pytest.skip("jsonschema 未安装（见 requirements.txt pin）")
    result = run_cli("--engine", "jsonschema", "--json", str(VALID), cwd=BASE)
    assert result.returncode == 0
    payload = json.loads(result.stdout.splitlines()[0])
    assert payload["engine"] == "jsonschema"
    assert payload["engine_detail"] not in ("", None)
