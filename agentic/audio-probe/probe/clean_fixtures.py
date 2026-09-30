#!/usr/bin/env python3
"""Remove locally generated audio fixtures and run scratch files (issue #30).

Cleanup policy for the audio bridge:
  * audio is sent to the Gemini API as request-inline bytes (`inlineData`), so
    the Gemini Files API is never used and no server-side temporary file
    exists to delete;
  * this script removes the LOCAL artifacts (gitignored fixtures/ and tmp/).

Usage:
    python probe/clean_fixtures.py --dry-run
    python probe/clean_fixtures.py --yes
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TARGETS = [BASE / "fixtures", BASE / "tmp"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="list what would be removed")
    group.add_argument("--yes", action="store_true", help="actually remove fixtures/ and tmp/")
    args = parser.parse_args()

    for target in TARGETS:
        if not target.exists():
            print(f"absent  {target}")
            continue
        if args.dry_run:
            print(f"would remove {target} ({sum(1 for _ in target.rglob('*') if _.is_file())} files)")
        else:
            shutil.rmtree(target, ignore_errors=False)
            print(f"removed {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
