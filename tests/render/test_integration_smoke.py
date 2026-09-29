"""Real DawDreamer integration renders.

These tests exercise the actual optional render stack.  Without the render
extra (``uv sync --locked`` only) they are skipped, so the base CI job stays
fast and dependency-free.  With ``uv sync --locked --extra render`` they render
the committed smoke configs and verify the exported PCM evidence end to end.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dawdreamer")

from aat.contracts.documents import Controls, SampleManifest, SourceRegistry  # noqa: E402
from aat.render import analysis  # noqa: E402
from aat.render.artifacts import sha256_file  # noqa: E402
from aat.render.config import config_from_dict, load_config  # noqa: E402
from aat.render.pipeline import render_sample  # noqa: E402
from aat.render.wavio import read_pcm16_wav  # noqa: E402

pytestmark = pytest.mark.integration

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "render"


def _render_ci(tmp_path: Path):
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    result = render_sample(config, tmp_path / "sample")
    return config, result, tmp_path / "sample"


def _ci_config_with_track_start(delta: float):
    with (CONFIG_DIR / "smoke_ci.toml").open("rb") as handle:
        data = tomllib.load(handle)
    data["render"]["track_start_seconds"] = delta
    data["render"]["sample_id"] = f"smoke-ci-origin-{delta:g}"
    return config_from_dict(data, origin="smoke_ci.toml")


def test_smoke_ci_render_produces_valid_artifacts(tmp_path: Path) -> None:
    config, result, out = _render_ci(tmp_path)

    expected = [
        out / "mix.wav",
        out / "sources.json",
        out / "controls.json",
        out / "manifest.json",
        out / "render_report.json",
    ]
    for source_id in config.source_ids:
        expected.extend([out / "stems" / f"{source_id}.wav", out / "dry" / f"{source_id}.wav"])
    for path in expected:
        assert path.is_file(), path

    mix, mix_meta = read_pcm16_wav(out / "mix.wav")
    stems = [read_pcm16_wav(out / "stems" / f"{source_id}.wav")[0] for source_id in config.source_ids]
    assert mix_meta == {"channels": 2, "sample_rate": config.sample_rate, "frames": mix.shape[1]}
    assert mix.dtype == np.int16
    evidence = analysis.check_stem_sum(stems, mix)
    assert evidence["max_abs_error_lsb"] <= evidence["tolerance_lsb"]

    manifest = SampleManifest.load(out / "manifest.json")
    assert manifest.stage == "rendered"
    assert manifest.sample_rate == config.sample_rate
    assert manifest.duration_seconds == pytest.approx(mix.shape[1] / config.sample_rate)
    assert manifest.seed == config.seed
    for relative, digest in manifest.content_sha256.items():
        assert sha256_file(out / relative) == digest

    sources = SourceRegistry.load(out / "sources.json")
    controls = Controls.load(out / "controls.json")
    controls.validate_source_references(sources.source_ids)
    assert sources.source_ids == config.source_ids

    report = result.report
    assert report["stem_sum"]["max_abs_error_lsb"] <= report["stem_sum"]["tolerance_lsb"]
    assert report["tail"]["decayed"] is True
    assert report["tail"]["margin_seconds"] > 0.0
    assert report["render_latency"]["complete"] is True
    assert report["render_latency"]["value_seconds"] == 0.0
    assert manifest.render_latency_seconds == 0.0
    assert report["mix_stats"]["clipped_samples"] == 0
    assert report["mix_stats"]["non_finite_samples"] == 0
    for row in report["sources"]:
        assert row["stem_stats"]["clipped_samples"] == 0
        assert row["stem_stats"]["non_finite_samples"] == 0
        assert row["dry_stats"]["non_finite_samples"] == 0
        assert row["effect_latency_samples"] == 0
        assert abs(row["onset_offset_seconds"]) <= 0.01

    assert result.surge is None
    assert report["surge_probe"]["status"] == "not_run"


def test_same_seed_render_is_byte_identical(tmp_path: Path) -> None:
    config, _, first = _render_ci(tmp_path)
    second = tmp_path / "sample-again"
    render_sample(config, second)
    for relative in ["mix.wav", *[f"stems/{source_id}.wav" for source_id in config.source_ids]]:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_full_smoke_config_renders_a_15_to_30_second_sample(tmp_path: Path) -> None:
    config = load_config(CONFIG_DIR / "smoke.toml")
    assert 15.0 <= config.duration_seconds <= 30.0
    assert len(config.source_ids) == 4
    result = render_sample(config, tmp_path / "smoke")
    out = tmp_path / "smoke"
    mix, metadata = read_pcm16_wav(out / "mix.wav")
    assert metadata["sample_rate"] == config.sample_rate
    assert 15.0 <= metadata["frames"] / config.sample_rate <= 30.0 + config.tail_seconds + 0.1
    stems = [read_pcm16_wav(out / "stems" / f"{source_id}.wav")[0] for source_id in config.source_ids]
    evidence = analysis.check_stem_sum(stems, mix)
    assert evidence["max_abs_error_lsb"] <= evidence["tolerance_lsb"]
    assert result.report["tail"]["decayed"] is True
    assert result.report["tail"]["margin_seconds"] > 0.0

    # Regression: the slow pad attack is acoustic evidence, not engine latency.
    manifest = SampleManifest.load(out / "manifest.json")
    assert manifest.render_latency_seconds == 0.0
    assert result.report["render_latency"]["value_seconds"] == 0.0
    onsets = [row["onset_offset_seconds"] for row in result.report["sources"]]
    assert max(onsets) > 0.005


def test_track_start_is_metadata_only(tmp_path: Path) -> None:
    renders: dict[float, tuple] = {}
    for delta in (0.0, 0.1, 12.0):
        config = _ci_config_with_track_start(delta)
        out = tmp_path / f"origin-{delta:g}"
        result = render_sample(config, out)
        renders[delta] = (config, result, out)

    base_config, base_result, base_dir = renders[0.0]
    relatives = [
        "mix.wav",
        *[f"stems/{source_id}.wav" for source_id in base_config.source_ids],
        *[f"dry/{source_id}.wav" for source_id in base_config.source_ids],
    ]
    for delta in (0.1, 12.0):
        for relative in relatives:
            assert (renders[delta][2] / relative).read_bytes() == (base_dir / relative).read_bytes()

    base_controls = Controls.load(base_dir / "controls.json")
    base_times = [event.time_seconds for event in base_controls.events]
    base_rows = {row["source_id"]: row for row in base_result.report["sources"]}
    for delta in (0.1, 12.0):
        report = renders[delta][1].report
        controls = Controls.load(renders[delta][2] / "controls.json")
        times = [event.time_seconds for event in controls.events]
        assert times == pytest.approx([time + delta for time in base_times], abs=1e-9)
        assert report["tail"]["margin_seconds"] == pytest.approx(
            base_result.report["tail"]["margin_seconds"]
        )
        assert report["tail"]["required_seconds"] == pytest.approx(
            base_result.report["tail"]["required_seconds"]
        )
        assert report["tail"]["required_absolute_seconds"] == pytest.approx(
            delta + base_result.report["tail"]["required_seconds"]
        )
        assert report["render_latency"]["value_seconds"] == base_result.report["render_latency"]["value_seconds"]
        manifest = SampleManifest.load(renders[delta][2] / "manifest.json")
        assert manifest.track_start_seconds == pytest.approx(delta)
        assert manifest.duration_seconds == pytest.approx(base_result.manifest.duration_seconds)
        assert manifest.render_latency_seconds == base_result.manifest.render_latency_seconds
        rows = {row["source_id"]: row for row in report["sources"]}
        for source_id, base_row in base_rows.items():
            row = rows[source_id]
            assert row["onset_offset_seconds"] == pytest.approx(
                base_row["onset_offset_seconds"], abs=1e-9
            )
            assert row["stem_onset_seconds"] == pytest.approx(
                base_row["stem_onset_seconds"] + delta, abs=1e-6
            )
