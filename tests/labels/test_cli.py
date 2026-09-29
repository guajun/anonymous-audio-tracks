"""End-to-end CLI tests on synthetic sample directories."""

from __future__ import annotations

import json

import numpy as np

from aat.contracts import ActivityData, SampleManifest, load_json
from aat.labels import LabelConfig

from . import signals, support


def pulse_buffer(rate: int = support.RATE, seconds: float = 2.0):
    buffer = np.zeros(round(rate * seconds), dtype=np.float64)
    signals.add_tone(buffer, rate, 0.5, 1.0, amplitude=0.5)
    return buffer


def run_cli(argv):
    return support.load_cli().main(argv)


def activity_of(root):
    return ActivityData.load(root)


def test_end_to_end_updates_manifest_and_reports(tmp_path, capsys):
    root = support.make_sample_dir(
        tmp_path / "sample",
        stems={"s01": pulse_buffer(), "s02": np.zeros(support.RATE * 2)},
    )
    assert run_cli([str(root)]) == 0
    out = capsys.readouterr().out
    assert "labeled 2 stem(s)" in out
    assert "activity.json" in out and "activity.npz" in out
    assert "stage=labeled" in out

    for name in ("activity.json", "activity.npz", "activity_summary.json"):
        assert (root / name).is_file()
    manifest = SampleManifest.load(root / "manifest.json")
    assert manifest.stage == "labeled"
    assert manifest.activity_metadata_path == "activity.json"
    assert manifest.activity_arrays_path == "activity.npz"
    assert manifest.content_sha256["activity.json"] == support.sha256_file(
        root / "activity.json"
    )
    assert manifest.content_sha256["activity.npz"] == support.sha256_file(
        root / "activity.npz"
    )
    data = activity_of(root)
    assert data.source_ids == ("s01", "s02")
    summary = load_json(root / "activity_summary.json")
    assert summary["source_order"] == ["s01", "s02"]
    assert summary["control_events_reference"]["used_for_labels"] is False
    assert summary["config_sha256"] == LabelConfig().config_sha256()
    assert summary["sources"][0]["source_id"] == "s01"


def test_hash_mismatch_is_rejected_before_any_write(tmp_path, capsys):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    (root / "stems" / "s01.wav").write_bytes(b"corrupted")
    assert run_cli([str(root)]) == 2
    err = capsys.readouterr().err
    assert "sha256" in err
    assert not (root / "activity.json").exists()
    assert not (root / "activity.npz").exists()
    assert SampleManifest.load(root / "manifest.json").stage == "rendered"


def test_sample_rate_mismatch_is_rejected(tmp_path, capsys):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    support.write_pcm16(
        root / "stems" / "s01.wav", pulse_buffer(rate=22050), 22050
    )
    support.update_manifest_hash(root, "stems/s01.wav")
    assert run_cli([str(root)]) == 2
    assert "sample rate" in capsys.readouterr().err


def test_duration_mismatch_is_rejected(tmp_path, capsys):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    support.write_pcm16(
        root / "stems" / "s01.wav", pulse_buffer(seconds=1.0), support.RATE
    )
    support.update_manifest_hash(root, "stems/s01.wav")
    assert run_cli([str(root)]) == 2
    assert "duration" in capsys.readouterr().err


def test_controls_are_reference_only(tmp_path):
    tone = np.zeros(support.RATE * 2)
    signals.add_tone(tone, support.RATE, 1.0, 1.5, amplitude=0.5)
    with_controls = support.make_sample_dir(
        tmp_path / "with",
        stems={"s01": tone},
        controls_events=(
            dict(
                time_seconds=0.0,
                source_id="s01",
                event_type="note_on",
                data={"note": 60},
            ),
        ),
    )
    without_controls = support.make_sample_dir(
        tmp_path / "without", stems={"s01": tone}
    )
    assert run_cli([str(with_controls)]) == 0
    assert run_cli([str(without_controls)]) == 0
    np.testing.assert_array_equal(
        activity_of(with_controls).activity, activity_of(without_controls).activity
    )
    summary = load_json(with_controls / "activity_summary.json")
    assert summary["control_events_reference"]["total"] == 1
    data = activity_of(with_controls)
    at_zero = int(np.argmin(np.abs(data.center_times - 0.0)))
    assert data.activity[at_zero, 0] == 0.0


def test_no_manifest_update_flag(tmp_path):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    assert run_cli([str(root), "--no-manifest-update"]) == 0
    manifest = SampleManifest.load(root / "manifest.json")
    assert manifest.stage == "rendered"
    assert manifest.activity_metadata_path is None
    assert (root / "activity.json").is_file()
    assert (root / "activity.npz").is_file()
    assert "activity.json" not in manifest.content_sha256


def test_rerun_on_labeled_sample_is_idempotent(tmp_path):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    assert run_cli([str(root)]) == 0
    first = activity_of(root).activity
    assert run_cli([str(root)]) == 0
    second = activity_of(root).activity
    np.testing.assert_array_equal(first, second)
    manifest = SampleManifest.load(root / "manifest.json")
    assert manifest.stage == "labeled"
    assert manifest.content_sha256["activity.json"] == support.sha256_file(
        root / "activity.json"
    )


def test_source_order_from_sources_json(tmp_path):
    tone = np.zeros(support.RATE)
    signals.add_tone(tone, support.RATE, 0.1, 0.9, amplitude=0.5)
    root = support.make_sample_dir(
        tmp_path / "sample",
        stems={"sA": tone, "sB": np.zeros(support.RATE)},
        source_ids=["sB", "sA"],
    )
    assert run_cli([str(root)]) == 0
    data = activity_of(root)
    assert data.source_ids == ("sB", "sA")
    assert not data.activity[:, 0].any()
    assert data.activity[:, 1].any()


def test_track_start_is_preserved(tmp_path):
    tone = np.zeros(support.RATE * 2)
    signals.add_tone(tone, support.RATE, 1.0, 1.3, amplitude=0.5)
    root = support.make_sample_dir(
        tmp_path / "sample", stems={"s01": tone}, track_start_seconds=12.0
    )
    assert run_cli([str(root)]) == 0
    data = activity_of(root)
    assert data.center_times[0] == 12.0
    at_thirteen = int(np.argmin(np.abs(data.center_times - 13.0)))
    assert data.activity[at_thirteen, 0] == 1.0
    assert data.valid[at_thirteen]


def test_config_file_overrides_are_recorded(tmp_path):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    config_path = tmp_path / "label_config.json"
    config_path.write_text(
        json.dumps({"hop_seconds": 0.04, "absolute_threshold_dbfs": -70.0}),
        encoding="utf-8",
    )
    assert run_cli([str(root), "--config", str(config_path)]) == 0
    metadata = load_json(root / "activity.json")
    assert metadata["hop_seconds"] == 0.04
    assert metadata["label_params"]["config"]["absolute_threshold_dbfs"] == -70.0
    rebuilt = LabelConfig.from_label_params(metadata["label_params"])
    assert rebuilt.hop_seconds == 0.04


def test_toml_config_and_experiment_defaults(tmp_path):
    cli = support.load_cli()
    experiment = cli.load_label_config(support.REPO_ROOT / "configs" / "experiment.toml")
    assert experiment.center_window_seconds == 2.0
    assert experiment.hop_seconds == 0.02

    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    config_path = tmp_path / "label_config.toml"
    config_path.write_text(
        "hop_seconds = 0.04\n\n[task]\nwindow_seconds = 1.0\nhop_seconds = 0.02\n",
        encoding="utf-8",
    )
    assert run_cli([str(root), "--config", str(config_path)]) == 0
    metadata = load_json(root / "activity.json")
    assert metadata["hop_seconds"] == 0.04  # flat key wins over [task]
    assert metadata["label_params"]["config"]["center_window_seconds"] == 1.0


def test_summary_name_override(tmp_path):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    assert run_cli([str(root), "--summary-name", "summary_alt.json"]) == 0
    assert (root / "summary_alt.json").is_file()
    assert SampleManifest.load(root / "manifest.json").stage == "labeled"


def test_missing_stem_file_is_rejected(tmp_path, capsys):
    root = support.make_sample_dir(tmp_path / "sample", stems={"s01": pulse_buffer()})
    (root / "stems" / "s01.wav").unlink()
    assert run_cli([str(root)]) == 2
    assert "not found" in capsys.readouterr().err
