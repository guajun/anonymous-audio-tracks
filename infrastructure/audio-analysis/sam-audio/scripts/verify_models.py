"""Verify local SAM Audio/T5 files against model-manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "model-cache")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "model-manifest.json")
    args = parser.parse_args()
    models_root = args.models_root if args.models_root.is_absolute() else (PROJECT_ROOT / args.models_root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = 0
    for entry in manifest:
        folder = entry.get("folder") or (
            "sam-audio-small" if entry["repo"] == "facebook/sam-audio-small" else "t5-base"
        )
        path = models_root / folder / entry["file"]
        if not path.is_file():
            print(f"MISSING {path}")
            failures += 1
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
            print(f"FAIL {path}: bytes={actual_size} sha256={actual_hash}")
            failures += 1
            continue
        print(f"OK {path}")
    if failures:
        print(f"{failures} model file(s) failed verification")
        return 1
    print("All model files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
