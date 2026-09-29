"""Protocol document and digest construction for a rendered sample."""

from __future__ import annotations

import hashlib
from pathlib import Path

from aat.contracts.documents import SampleManifest
from aat.render.artifacts import (
    build_manifest,
    build_sources,
    dry_relative_path,
    mix_relative_path,
    required_digest_paths,
    sha256_file,
    sha256_json,
    stem_relative_path,
)
from aat.render.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "render"


def test_sources_document_matches_config() -> None:
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    sources = build_sources(config)
    assert sources.source_ids == config.source_ids
    assert sources.sample_id == config.sample_id
    for entry in sources.sources:
        assert entry.renderer == "dawdreamer-sampler"
        assert entry.preset_ref == config.sources[entry.index].resolved_preset_ref
        assert entry.sample_ref == config.sources[entry.index].resolved_sample_ref
        assert entry.seed == config.sources[entry.index].sample.seed


def test_sources_round_trip(tmp_path) -> None:
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    path = build_sources(config).save(tmp_path / "sources.json")
    assert path.read_text(encoding="utf-8").endswith("\n")
    loaded = build_sources(config).load(path)
    assert loaded.source_ids == config.source_ids


def test_relative_paths_for_artifacts() -> None:
    assert mix_relative_path() == "mix.wav"
    assert stem_relative_path("s01") == "stems/s01.wav"
    assert dry_relative_path("s01") == "dry/s01.wav"
    required = required_digest_paths(("s01", "s02"))
    assert required == ("mix.wav", "sources.json", "controls.json", "stems/s01.wav", "stems/s02.wav")


def test_manifest_validates_with_digests() -> None:
    config = load_config(CONFIG_DIR / "smoke_ci.toml")
    digest = "ab" * 32
    digest_paths = {path: digest for path in required_digest_paths(config.source_ids)}
    manifest = build_manifest(
        config,
        digest_paths=digest_paths,
        renderer_version="dawdreamer-test",
        render_latency_seconds=0.001,
        tail_seconds=config.tail_seconds,
    )
    assert isinstance(manifest, SampleManifest)
    payload = manifest.to_json_dict()
    assert payload["stage"] == "rendered"
    assert payload["groups"]["preset"] == config.presets
    assert payload["groups"]["sample_origin"] == config.sample_origins
    assert set(payload["content_sha256"]) == set(digest_paths)
    assert payload["versions"]["renderer"] == "dawdreamer-test"


def test_file_and_json_digests_are_stable(tmp_path) -> None:
    target = tmp_path / "data.bin"
    target.write_bytes(b"anonymous-audio-tracks")
    assert sha256_file(target) == hashlib.sha256(b"anonymous-audio-tracks").hexdigest()
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})
