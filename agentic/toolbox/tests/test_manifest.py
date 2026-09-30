"""Validate the agentic/toolbox integration manifest (stdlib only, CPU-only).

Run from the repository worktree root:

    python -m unittest discover -s agentic/toolbox/tests -v
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLBOX = ROOT / "agentic" / "toolbox"
MANIFEST = TOOLBOX / "manifest.json"

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SECRET_PATTERNS = (
    (re.compile(r"[A-Za-z]:\\Users\\|/home/"), "personal home path"),
    (re.compile(r"MSI-NB"), "local username"),
    (re.compile(r"F:[\\/]LED"), "local drive path"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]+['\"]"), "api key assignment"),
)
FORBIDDEN_BINARIES = (".pt", ".safetensors", ".wav", ".mp3", ".flac", ".ogg", ".m4a")


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

    def test_no_secrets_or_personal_paths_in_toolbox_files(self):
        self_file = Path(__file__).resolve()
        for path in sorted(TOOLBOX.rglob("*")):
            if not path.is_file() or path.suffix not in (".md", ".json", ".py", ".toml", ".txt"):
                continue
            if path.resolve() == self_file:
                continue  # this file embeds the detection patterns themselves
            text = path.read_text(encoding="utf-8")
            for pattern, label in SECRET_PATTERNS:
                with self.subTest(file=path.name, pattern=label):
                    self.assertIsNone(pattern.search(text), f"{label} found in {path.name}")

    def test_no_weights_audio_or_binary_payloads_under_agentic(self):
        for path in (ROOT / "agentic").rglob("*"):
            if path.is_file():
                with self.subTest(file=path.name):
                    self.assertNotIn(path.suffix.lower(), FORBIDDEN_BINARIES,
                                     f"binary payload committed: {path.relative_to(ROOT)}")
                    self.assertLess(path.stat().st_size, 512 * 1024,
                                    f"unexpectedly large file: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    unittest.main()