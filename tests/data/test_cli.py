"""CLI: build / verify / summary / batch over a fixture corpus."""

from __future__ import annotations

import functools
import json

from . import fixtures

from aat.data import DatasetIndex

RATE = fixtures.RATE


@functools.lru_cache(maxsize=1)
def build_cli():
    return fixtures.load_cli(fixtures.BUILD_CLI, "aat_build_dataset_index_cli")


def _corpus(tmp_path):
    data = tmp_path / "data"
    for number in range(3):
        fixtures.make_sample(
            data,
            f"s{number}",
            sample_id=f"sample-{number}",
            composition=f"comp-{number}",
            stems={"s01": fixtures.tone_bursts(4.0, RATE, [(0.0, 4.0)])},
        )
    return data


def _build(tmp_path, capsys):
    module = build_cli()
    data = _corpus(tmp_path)
    out = tmp_path / "index" / "index.json"
    assert (
        module.main(
            [
                "build",
                "--data-root",
                str(data),
                "--out",
                str(out),
                "--seed",
                "5",
                "--ratios",
                "0.8,0.1,0.1",
                "--slots",
                "3",
                "--window-seconds",
                "2.0",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    return data, out, report


def test_build_verify_summary_batch_roundtrip(tmp_path, capsys) -> None:
    module = build_cli()
    data, out, report = _build(tmp_path, capsys)
    assert report["sample_count"] == 3
    assert report["leak_free"] is True
    assert report["plan"]["seed"] == 5
    assert out.is_file()
    index = DatasetIndex.load(out)
    assert index.plan["slots"] == 3

    assert module.main(["verify", "--index", str(out)]) == 0
    assert "verify: ok" in capsys.readouterr().out

    assert module.main(["summary", "--index", str(out), "--samples"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["plan"]["slots"] == 3
    assert len(summary["samples"]) == 3
    assert all("split" in sample for sample in summary["samples"])

    assert (
        module.main(
            [
                "batch",
                "--index",
                str(out),
                "--split",
                "train",
                "--items",
                "1",
                "--centers",
                "3",
                "--min-gap",
                "2",
                "--seed",
                "4",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["returned_items"] == 1
    assert payload["skipped"] == []
    block = payload["blocks"][0]
    assert len(block["center_times"]) == 3
    assert block["slots"] == 3
    assert block["source_ids"] == ["s01"]
    assert block["slot_ids"] == ["s01", None, None]
    assert all(isinstance(value, bool) for row in block["source_present"] for value in row)
    assert block["audio_valid_all"] is True


def test_verify_fails_with_located_error(tmp_path, capsys) -> None:
    module = build_cli()
    data, out, _ = _build(tmp_path, capsys)
    with (data / "s1" / "mix.wav").open("ab") as handle:
        handle.write(b"\x00")
    assert module.main(["verify", "--index", str(out)]) == 1
    captured = capsys.readouterr()
    assert "sample 's1'" in captured.err
    assert "content_sha256['mix.wav']" in captured.err
    assert "verify: failed" in captured.err


def test_build_rejects_invalid_ratios(tmp_path, capsys) -> None:
    module = build_cli()
    data = _corpus(tmp_path)
    out = tmp_path / "index" / "index.json"
    assert (
        module.main(
            [
                "build",
                "--data-root",
                str(data),
                "--out",
                str(out),
                "--ratios",
                "0.5,-0.5,1.0",
            ]
        )
        == 2
    )
    assert "ratios.val" in capsys.readouterr().err


def test_verify_rejects_unknown_index_version(tmp_path, capsys) -> None:
    module = build_cli()
    _, out, _ = _build(tmp_path, capsys)
    fixtures.rewrite_json(out, lambda payload: payload.__setitem__("index_version", "v99"))
    assert module.main(["verify", "--index", str(out)]) == 2
    assert "unsupported version" in capsys.readouterr().err


def test_batch_on_empty_split_fails_by_default(tmp_path, capsys) -> None:
    module = build_cli()
    _, out, _ = _build(tmp_path, capsys)
    assert module.main(["batch", "--index", str(out), "--split", "test", "--items", "1"]) == 2
    assert "only 0 sample" in capsys.readouterr().err
