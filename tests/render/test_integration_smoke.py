"""Real DawDreamer integration renders.

These tests exercise the actual optional render stack.  Without the render
extra (``uv sync --locked`` only) they are skipped, so the base CI job stays
fast and dependency-free.  With ``uv sync --locked --extra render`` they render
the committed smoke configs and verify the exported PCM evidence end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dawdreamer")

from aat.contracts.documents import Controls, SampleManifest, SourceRegistry  # noqa: E402
from aat.render import analysis  # noqa: E402
from aat.render.artifacts import sha256_file  # noqa: E402
from aat.render.config import load_config  # noqa: E402
from aat.render.pipeline import render_sample  # noqa: E402
from aat.render.wavio import read_pcm16_wav  # noqa: E402

pytestmark = pytest.mark.integration

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "render"


def _render_ci(tmp_path: Path):
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    result = render_sample(config, tmp_path / "sample")
    return config, result, tmp_path / "sample"


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
