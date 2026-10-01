"""tests/test_validate_result.py — 冻结 schema + e2e 一致性校验（issue #33，全离线）。

覆盖：real 级全绿路径；mock/native/bridge 区别（mock 结果在 real 级必须被拒、
native 冒充必须被拒、bridge 才是真实音频通路）；hash/时轴/来源标注/隐私/路径/stem 的负例。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import common
import validate_result
from helpers import build_result, make_clip_fixture, make_stem, write_result


@pytest.fixture()
def env(tmp_path):
    clip = make_clip_fixture(tmp_path)
    run_dir = tmp_path / "outputs" / "e2e" / "run-1"
    run_dir.mkdir(parents=True)
    return {"root": tmp_path, "clip": clip, "run_dir": run_dir,
            "clip_json": tmp_path / "local" / "e2e" / f"clip-{clip['name']}.json",
            "audio_root": tmp_path / "audio" / "inputs"}


def _validate(env, doc, level="real", with_files=True, stem=None):
    path = write_result(env["run_dir"], doc)
    return validate_result.validate(
        path, level=level,
        clip_path=env["clip_json"] if with_files else None,
        audio_root=env["audio_root"] if with_files else None,
        run_dir=env["run_dir"] if (with_files and stem) else None)


def _by_name(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


class TestRealLevelHappyPath:
    def test_real_doc_passes_with_stem_files(self, env):
        stem = make_stem(env["run_dir"])
        report = _validate(env, build_result(env["clip"], stem=stem), stem=stem)
        assert report["ok"], json.dumps(report, ensure_ascii=False, indent=2)
        assert report["level"] == "real"
        assert report["schema_validation"]["ok"]

    def test_real_doc_passes_without_stems(self, env):
        report = _validate(env, build_result(env["clip"]))
        assert report["ok"], json.dumps(report, ensure_ascii=False, indent=2)


class TestMockNativeBridgeBoundary:
    def test_mock_result_rejected_at_real_level(self, env):
        """mock/fixture 结果不得在 real 级通过（真实 / mock 边界可机器区分）。"""
        doc = build_result(env["clip"], kind="mock")
        real = _validate(env, doc, level="real")
        assert not real["ok"]
        assert not _by_name(real, "provenance_chain")["pass"] or \
            not _by_name(real, "provenance_pins")["pass"]
        fixture = _validate(env, doc, level="fixture")
        assert fixture["ok"], json.dumps(fixture, ensure_ascii=False, indent=2)
        assert fixture["level"] == "fixture"

    def test_native_audio_claim_rejected(self, env):
        """Pi 原生音频是 unsupported；真实音频通路必须是 bridge，冒充 native 直接拒。"""
        steps = [
            {"tool": "pi-native-audio", "source": "native", "note": "冒充原生音频输入"},
            {"tool": "gemini-3.8-flash listening", "source": "llm", "note": "假设"},
            {"tool": "audio-toolbox.sam pin dfbc40a9541f (sam c603de8794cc)", "source": "sam", "note": "分离"},
            {"tool": "e2e dsp_onset energy-flux", "source": "dsp", "note": "onset"},
        ]
        report = _validate(env, build_result(env["clip"], provenance_steps=steps))
        assert not report["ok"]
        assert not _by_name(report, "no_native_audio_claim")["pass"]

    def test_bridge_pathway_documented_in_result(self, env):
        report = _validate(env, build_result(env["clip"]))
        blob = json.dumps(build_result(env["clip"]), ensure_ascii=False)
        assert "bridge" in blob and "gemini-3.8-flash" in blob


class TestNegativeCases:
    def test_hash_mismatch_fails(self, env):
        doc = build_result(env["clip"])
        doc["audio"]["sha256"] = "0" * 64
        report = _validate(env, doc)
        assert not report["ok"]
        assert not _by_name(report, "clip_sha256")["pass"]

    def test_duration_mismatch_fails(self, env):
        doc = build_result(env["clip"])
        doc["audio"]["duration_seconds"] = env["clip"]["duration_s"] + 1.0
        report = _validate(env, doc)
        assert not report["schema_validation"]["ok"] or not report["ok"]

    def test_onset_out_of_range_is_schema_failure(self, env):
        events = [{"id": "inst-1-ev-1", "onset_seconds": 99.0, "source": "dsp", "method": "spectral-flux"}]
        report = _validate(env, build_result(env["clip"], events=events))
        assert not report["schema_validation"]["ok"]
        assert not report["ok"]

    def test_missing_source_method_fails(self, env):
        events = [{"id": "inst-1-ev-1", "onset_seconds": 1.0},
                  {"id": "inst-1-ev-2", "onset_seconds": 2.0, "source": "dsp", "method": "spectral-flux"}]
        report = _validate(env, build_result(env["clip"], events=events))
        assert not _by_name(report, "event_provenance")["pass"]

    def test_dsp_method_with_llm_source_fails(self, env):
        events = [{"id": "inst-1-ev-1", "onset_seconds": 1.0, "source": "llm", "method": "spectral-flux"}]
        report = _validate(env, build_result(env["clip"], events=events))
        assert not _by_name(report, "event_provenance")["pass"]
        assert "spectral-flux" in _by_name(report, "event_provenance")["detail"]

    def test_no_dsp_event_fails_real_level(self, env):
        events = [{"id": "inst-1-ev-1", "onset_seconds": 1.0, "source": "llm", "method": "listening-estimate"}]
        report = _validate(env, build_result(env["clip"], events=events))
        assert not _by_name(report, "event_provenance")["pass"]

    def test_secret_in_text_fails_privacy(self, env):
        doc = build_result(env["clip"], description="token: AIzaSyFakeFakeFakeFakeFakeFake123456")
        report = _validate(env, doc)
        assert not _by_name(report, "privacy")["pass"]

    def test_absolute_path_in_stem_fails(self, env):
        stem = {"filename": "C:/private/stems/target.wav", "sha256": "0" * 64}
        report = _validate(env, build_result(env["clip"], stem=stem))
        assert not report["schema_validation"]["ok"]

    def test_stem_hash_mismatch_fails(self, env):
        stem = make_stem(env["run_dir"])
        bad = dict(stem, sha256="0" * 64)
        report = _validate(env, build_result(env["clip"], stem=bad), stem=bad)
        assert not _by_name(report, "stem_files")["pass"]

    def test_missing_provenance_pins_fail_real_level(self, env):
        steps = [
            {"tool": "some tool", "source": "llm", "note": "假设"},
            {"tool": "some dsp", "source": "dsp", "note": "onset"},
            {"tool": "some sam", "source": "sam", "note": "分离"},
        ]
        report = _validate(env, build_result(env["clip"], provenance_steps=steps))
        assert not _by_name(report, "provenance_pins")["pass"]

    def test_unparseable_result_reports_readable_error(self, env):
        path = env["run_dir"] / "result.json"
        path.write_text("{not json", encoding="utf-8")
        report = validate_result.validate(path, level="real")
        assert not report["ok"]
        assert report["checks"][0]["name"] == "parse"


class TestStrictProvenanceR1:
    """R1：pin 必须精确 SHA/前缀；mock/native 冒称在任何字段都拒。"""

    def test_word_only_tool_reference_fails_pins(self, env):
        steps = [
            {"tool": "some sam-audio wrapper", "source": "sam", "note": "只提到 sam-audio 字样"},
            {"tool": "listening", "source": "llm", "note": "假设"},
            {"tool": "bridge", "source": "bridge", "note": "附件"},
            {"tool": "dsp", "source": "dsp", "note": "onset"},
        ]
        report = _validate(env, build_result(env["clip"], provenance_steps=steps))
        assert not _by_name(report, "provenance_pins")["pass"]
        assert common.TOOLBOX_PIN[:12] in _by_name(report, "provenance_pins")["detail"]

    def test_exact_pin_prefixes_pass(self, env):
        report = _validate(env, build_result(env["clip"]))
        assert _by_name(report, "provenance_pins")["pass"]

    def test_mock_claim_anywhere_fails_real_level(self, env):
        doc = build_result(env["clip"])
        doc["tempo"] = {"bpm": 120.0, "source": "mock", "confidence": 1.0}
        report = _validate(env, doc)
        assert not report["ok"]
        assert not _by_name(report, "no_mock_or_native_claims")["pass"]

    def test_native_event_claim_fails_real_level(self, env):
        doc = build_result(env["clip"])
        ev = doc["instruments"][0]["events"][0]
        ev["source"], ev["method"] = "native", "listening-estimate"
        report = _validate(env, doc)
        assert not _by_name(report, "no_mock_or_native_claims")["pass"]

    def test_fixture_level_still_accepts_mock(self, env):
        doc = build_result(env["clip"], kind="mock")
        report = _validate(env, doc, level="fixture")
        assert report["ok"], json.dumps(report, ensure_ascii=False, indent=2)


class TestCliContract:
    def test_exit_codes(self, env, capsys):
        stem = make_stem(env["run_dir"])
        path = write_result(env["run_dir"], build_result(env["clip"], stem=stem))
        argv = ["--result", str(path), "--level", "real", "--clip", str(env["clip_json"]),
                "--audio-root", str(env["audio_root"]), "--run-dir", str(env["run_dir"])]
        assert validate_result.main(argv) == 0
        assert validate_result.main(argv + ["--json"]) == 0
        bad = write_result(env["run_dir"], build_result(env["clip"], kind="mock"))
        assert validate_result.main(["--result", str(bad), "--level", "real"]) == 1
        assert validate_result.main(["--result", str(env["run_dir"] / "missing.json")]) == 2

    def test_out_file_written(self, env):
        path = write_result(env["run_dir"], build_result(env["clip"]))
        out = env["run_dir"] / "validation.json"
        assert validate_result.main(["--result", str(path), "--level", "real",
                                     "--out", str(out)]) == 0
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["schema"] == "agentic-e2e-result-validation/v1"
