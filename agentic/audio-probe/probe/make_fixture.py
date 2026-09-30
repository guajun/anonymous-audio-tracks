#!/usr/bin/env python3
"""Generate small, deterministic audio fixtures for the issue-30 probes.

Stdlib only (no dependency / lockfile changes). Fixtures are gitignored; the
ground truth of each fixture is defined here by construction, which is what the
real-model runs are checked against (qualitatively — issue #30 does not claim
timestamp-level ground-truth accuracy).

Default fixture: 16-bit PCM mono WAV, 8000 Hz, three ascending sine tones
(440 -> 660 -> 880 Hz), 0.30 s tone + 0.20 s silence each.

Usage:
    python probe/make_fixture.py                 # writes fixtures/fixture-a.wav
    python probe/make_fixture.py --all           # + fixture-b.wav (descending)
    python probe/make_fixture.py --out /tmp/x.wav
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import wave
from pathlib import Path

RATE = 8000
TONE_SECONDS = 0.30
GAP_SECONDS = 0.20
AMPLITUDE = 12000

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# Ground truth, by construction (see reports/40-real-bridge-run.md):
#   fixture-a.wav : 3 tones, ascending pitch (440, 660, 880 Hz)
#   fixture-b.wav : 3 tones, descending pitch (880, 660, 440 Hz)
FIXTURES = {
    "fixture-a.wav": [440.0, 660.0, 880.0],
    "fixture-b.wav": [880.0, 660.0, 440.0],
}


def render_tones(freqs: list[float]) -> bytes:
    samples: list[int] = []
    for freq in freqs:
        n_tone = int(RATE * TONE_SECONDS)
        samples.extend(
            int(AMPLITUDE * math.sin(2.0 * math.pi * freq * i / RATE)) for i in range(n_tone)
        )
        samples.extend([0] * int(RATE * GAP_SECONDS))
    return struct.pack(f"<{len(samples)}h", *samples)


def write_wav(path: Path, freqs: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = render_tones(freqs)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(frames)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write a single ascending fixture to this path")
    parser.add_argument("--all", action="store_true", help="write fixture-a.wav and fixture-b.wav")
    args = parser.parse_args()

    if args.out:
        path = write_wav(Path(args.out), FIXTURES["fixture-a.wav"])
        print(f"wrote {path} ({path.stat().st_size} bytes)")
        return 0

    names = list(FIXTURES) if args.all else ["fixture-a.wav"]
    for name in names:
        path = write_wav(FIXTURE_DIR / name, FIXTURES[name])
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
