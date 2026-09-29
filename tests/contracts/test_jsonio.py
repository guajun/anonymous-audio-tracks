"""Strict JSON I/O tests: no NaN/Infinity in or out, nested payloads included."""

from __future__ import annotations

import pytest

from aat.contracts import ContractError, dump_json, dumps_json, load_json


def test_dump_json_writes_stable_text_and_roundtrips(tmp_path):
    path = dump_json(tmp_path / "doc.json", {"b": 1, "a": [True, "x"]})
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert load_json(path) == {"b": 1, "a": [True, "x"]}


def test_dump_json_rejects_nested_nan_and_infinity(tmp_path):
    with pytest.raises(ContractError, match="JSON-serialisable"):
        dump_json(tmp_path / "doc.json", {"a": [1, {"b": float("nan")}]})
    with pytest.raises(ContractError, match="JSON-serialisable"):
        dump_json(tmp_path / "doc.json", {"a": [float("inf")]})
    assert not (tmp_path / "doc.json").exists()


def test_dumps_json_rejects_nan():
    with pytest.raises(ContractError, match="JSON-serialisable"):
        dumps_json({"x": float("-inf")})


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_load_json_rejects_nonstandard_constants(tmp_path, literal):
    path = tmp_path / "doc.json"
    path.write_text(f'{{"value": {literal}}}', encoding="utf-8")
    with pytest.raises(ContractError, match="non-finite"):
        load_json(path)


def test_load_json_rejects_overflowing_exponent(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text('{"value": 1e999, "nested": [{"deep": -1e999}]}', encoding="utf-8")
    with pytest.raises(ContractError, match="non-finite"):
        load_json(path)


def test_load_json_rejects_invalid_syntax(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ContractError, match="invalid JSON"):
        load_json(path)
