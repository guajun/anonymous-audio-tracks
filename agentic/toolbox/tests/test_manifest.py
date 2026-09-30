"""Validate the agentic/toolbox integration manifest and git-tracked hygiene.

Hygiene checks run over the **git-tracked** file list (`git ls-files`), never
over raw disk content. Later issues (#30/#31) legitimately create fixtures,
audio and session logs in gitignored directories (`.local/`, `outputs/`, ...);
that local data must never be reported as "committed". Conversely, anything
that *is* tracked must not be an audio/weight payload or contain secrets.

Run from the repository worktree root:

    python -m unittest discover -s agentic/toolbox/tests -v
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLBOX = ROOT / "agentic" / "toolbox"
MANIFEST = TOOLBOX / "manifest.json"

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SECRET_PATTERNS = (
    (re.compile(r"[A-Za-z]:\\+Users\\+"), "personal home path"),
    (re.compile(r"MSI" + "-NB"), "local username"),
    (re.compile(r"F:[\\\\/]LED"), "local drive path"),
    (re.compile(r"sk-" + r"[A-Za-z0-9_-]{20,}"), "API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]+['\"]"), "api key assignment"),
)
FORBIDDEN_BINARIES = (".pt", ".safetensors", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".onnx")
TEXT_SUFFIXES = (".md", ".json", ".py", ".toml", ".txt", ".yml", ".yaml", ".cfg", ".ini", ".cmd", ".sh", ".ps1")
MAX_TRACKED_SIZE = 512 * 1024


def tracked_files(repo_root: Path) -> list[str]:
    """Relative paths of git-tracked files in ``repo_root`` (index contents)."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True,
        shell=False,
        check=True,
    )
    return sorted(item for item in result.stdout.decode("utf-8").split("\0") if item)


def hygiene_violations(repo_root: Path, *, under: str = "", exclude: tuple[str, ...] = ()) -> list[str]:
    """Violations among tracked files only (gitignored local data is out of scope)."""
    violations: list[str] = []
    for rel in tracked_files(repo_root):
        normalized = rel.replace("\\", "/")
        if under and not normalized.startswith(under):
            continue
        if normalized in exclude:
            continue
        path = repo_root / rel
        suffix = Path(rel).suffix.lower()
        if suffix in FORBIDDEN_BINARIES:
            violations.append(f"{normalized}: forbidden tracked payload type {suffix}")
            continue
        if path.is_file() and path.stat().st_size >= MAX_TRACKED_SIZE:
            violations.append(f"{normalized}: unexpectedly large tracked file")
            continue
        if suffix in TEXT_SUFFIXES and path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(text):
                    violations.append(f"{normalized}: {label}")
    return violations


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_pins_are_full_sha_and_install_command_matches(self):
        toolbox = self.manifest["toolbox"]
        skill = self.manifest["skill"]
        self.assertRegex(toolbox["pin"]["sha"], FULL_SHA)
        self.assertRegex(skill["install"]["pin"], FULL_SHA)
        self.assertEqual(skill["install"]["pin"], toolbox["pin"]["sha"])
        self.assertIn(toolbox["pin"]["sha"], skill["install"]["command"])
        self.assertIn("gh skill install guajun/agentic-audio-toolbox sam-audio", skill["install"]["command"])
        self.assertIn("--pin", skill["install"]["command"])
        self.assertEqual(skill["install"]["agent"], "pi")
        self.assertEqual(skill["install"]["scope"], "project")
        self.assertTrue(skill["self_contained"])
        self.assertEqual(skill["path"], "skills/sam-audio")

    def test_sam_source_pin_license_and_weights_policy(self):
        source = self.manifest["sam_source"]
        self.assertRegex(source["commit"], FULL_SHA)
        self.assertEqual(source["path"], "infrastructure/audio-analysis/sam-audio")
        self.assertEqual(source["branch"], "infra/audio-analysis-migration")
        self.assertIn("Meta SAM License", source["license"])
        self.assertFalse(source["weights"]["distributed"])
        self.assertIn("model-manifest.json", source["weights"]["integrity"])
        self.assertIn("verify_models.py", source["weights"]["integrity"])
        self.assertEqual(self.manifest["constraints"]["network"], "disabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 for upstream calls)")
        self.assertEqual(self.manifest["constraints"]["gpu_inference"], "deferred to guajun/anonymous-audio-tracks#33")
        self.assertTrue(self.manifest["constraints"]["no_weights_audio_keys_paths_in_git"])
        self.assertFalse(self.manifest["constraints"]["root_lockfiles_touched"])

    def test_exit_code_map_and_referenced_docs_exist(self):
        cli = self.manifest["toolbox"]["cli"]
        self.assertEqual(cli["schema"], "audio-toolbox.sam/v1")
        self.assertEqual(set(cli["exit_codes"]), {"0", "2", "3", "4", "5", "6", "7"})
        for relative in self.manifest["integration"]["docs"]:
            self.assertTrue((ROOT / relative).is_file(), f"missing doc: {relative}")
        self.assertTrue((ROOT / self.manifest["integration"]["tests"]).is_file())


class TrackedHygieneTests(unittest.TestCase):
    """Only git-tracked files are judged; gitignored local data is never flagged."""

    def test_no_tracked_payloads_secrets_or_personal_paths_in_repo(self):
        # This module embeds the detection patterns themselves, so it is excluded
        # from its own scan; every other git-tracked file in the repo is checked.
        # Gitignored local data (fixtures/audio/sessions of #30/#31) is invisible
        # to this check by construction.
        self_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
        self.assertIn("agentic/toolbox/manifest.json", tracked_files(ROOT))
        violations = hygiene_violations(ROOT, exclude=(self_rel,))
        self.assertEqual(violations, [], f"tracked hygiene violations: {violations}")

    def test_ignored_local_fixture_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".gitignore").write_text(".local/\noutputs/\n*.wav\n*.pt\n", encoding="utf-8")
            (repo / ".local").mkdir()
            (repo / ".local" / "session.jsonl").write_text('{"session": true}', encoding="utf-8")
            (repo / "outputs").mkdir()
            (repo / "outputs" / "clip.wav").write_bytes(b"RIFF0000WAVEfake")
            (repo / "model-cache").mkdir()
            (repo / "model-cache" / "checkpoint.pt").write_bytes(b"\0" * 64)
            (repo / "docs").mkdir()
            (repo / "docs" / "README.md").write_text("# fine\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, shell=False)
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, shell=False)
            tracked = tracked_files(repo)
            self.assertIn("docs/README.md", tracked)
            self.assertNotIn("outputs/clip.wav", tracked)
            self.assertNotIn(".local/session.jsonl", tracked)
            self.assertNotIn("model-cache/checkpoint.pt", tracked)
            self.assertEqual(hygiene_violations(repo), [])

    def test_tracked_binary_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "clip.wav").write_bytes(b"RIFF0000WAVEfake")
            (repo / "checkpoint.pt").write_bytes(b"\0" * 64)
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, shell=False)
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, shell=False)
            violations = hygiene_violations(repo)
            self.assertEqual(len(violations), 2, violations)
            self.assertTrue(any("clip.wav" in item for item in violations))
            self.assertTrue(any("checkpoint.pt" in item for item in violations))

    def test_tracked_secret_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "config.json").write_text(
                r'{"api_key": "sk-abcdefghijklmnopqrstuvwxyz1234", "home": "C:\\Users\\someone\\x"}',
                encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, shell=False)
            subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, shell=False)
            violations = hygiene_violations(repo)
            self.assertTrue(any("API key" in item for item in violations), violations)
            self.assertTrue(any("personal home path" in item for item in violations), violations)


if __name__ == "__main__":
    unittest.main()