"""CPU-only preflight regression tests; no model imports or external requests."""
import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_inference", ROOT / "scripts/run_inference.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class PreflightTests(unittest.TestCase):
    def test_dry_run_audio_bounds_before_model_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "clip.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(8000)
                stream.writeframes(b"\0\0" * 8000 * 30)
            model = root / "model"
            model.mkdir()
            for name in ("checkpoint.pt", "config.json"):
                (model / name).write_text("{}")
            args = ["--audio", str(audio), "--description", "strings", "--model-dir", str(model), "--text-encoder-dir", str(model), "--dry-run"]
            with patch.object(run, "check_environment"), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(args + ["--anchor", "11.18,11.50"]), 0)
                self.assertEqual(run.main(args + ["--anchor", "0,30"]), 0)
                for anchor in ("40,41", "nan,1", "0,inf", "-1,2", "2,1", "1,1", "0,30.01"):
                    with self.subTest(anchor=anchor), self.assertRaises(SystemExit) as error:
                        run.main(args + ["--anchor=" + anchor])
                    self.assertEqual(error.exception.code, 2)
            self.assertNotIn("sam_audio", sys.modules)
            self.assertNotIn("torch", sys.modules)

    def test_missing_ffmpeg(self):
        with patch.object(run.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "full-shared.*PATH"):
                run.check_environment()

    def test_windows_missing_dll_and_handle_lifetime(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(run.shutil, "which", return_value=str(directory / "ffmpeg.exe")), patch.object(run.os, "name", "nt"):
                # pathlib selects WindowsPath after os.name is patched; use an existing concrete path type.
                with patch.object(run, "Path", type(directory)):
                    with self.assertRaisesRegex(ValueError, "full-shared"):
                        run.check_environment()
                    for name in ("avcodec", "avformat", "avutil", "swresample"):
                        (directory / f"{name}-60.dll").touch()
                    handle = object()
                    with patch.object(run.os, "add_dll_directory", return_value=handle, create=True), patch.object(run, "_DLL_HANDLES", []):
                        run.check_environment()
                        self.assertIs(run._DLL_HANDLES[0], handle)

    def test_ffprobe_fallback(self):
        with patch.object(run.sf, "info", side_effect=RuntimeError), patch.object(run.shutil, "which", return_value="ffprobe"), patch.object(run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "60.0\n")):
            self.assertEqual(run.audio_duration(Path("custom.m4a")), 60)

    def test_git_ignore_boundary(self):
        repo = ROOT.parents[2]
        ignored = ["data/private-metadata.json", "data/audio-analysis/private-note.txt", "data/audio-analysis/breeze/report.json", "data/audio-analysis/breeze/sample.m4a"]
        allowed = ["data/README.md", "data/audio-analysis/README.md", "src/aat/data/example.py", "tests/data/example.py"]
        for path in ignored + allowed:
            result = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=repo, check=False)
            self.assertEqual(result.returncode, 0 if path in ignored else 1, path)


if __name__ == "__main__":
    unittest.main()
