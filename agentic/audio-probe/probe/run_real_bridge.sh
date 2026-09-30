#!/usr/bin/env bash
# run_real_bridge.sh — REAL Gemini run through the project-local audio bridge
# (issue #30). Model is fixed to google/gemini-3.8-flash; no fallback.
#
# REAL vs MOCK: this makes real API calls (paid). Everything in tests/ and the
# offline probes is mock/offline; only this script and run_real_native.sh are
# real-model evidence.
#
# Usage (from agentic/audio-probe/):
#   bash probe/run_real_bridge.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p tmp/sessions

echo "== auth check (status only; no key printed) =="
pi auth check --provider google --json

echo "== fixtures =="
python probe/make_fixture.py --all

echo "== real run: audio bridge + google/gemini-3.8-flash =="
PI_AUDIO_BRIDGE_ROOT="$(pwd)/fixtures" \
pi --provider google --model gemini-3.8-flash \
  --extension ./bridge/audio-bridge.ts \
  --tools audio_attach \
  --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files \
  --session-dir ./tmp/sessions --name audio-probe-bridge \
  --mode json --print \
  -- "Call the audio_attach tool once for each of these files: fixture-a.wav and fixture-b.wav (paths relative to the audio root). After both are attached, describe each recording separately: (1) how many distinct sound events you hear in it, (2) whether its pitch goes up, down, or stays level across the recording. Base your answer only on what you actually hear from the attached audio." \
  > tmp/bridge-events.jsonl 2> tmp/bridge-stderr.txt || echo "pi exited non-zero; see tmp/bridge-stderr.txt"

echo "== done =="
echo "raw events : tmp/bridge-events.jsonl"
echo "stderr     : tmp/bridge-stderr.txt"
echo "sanitize with: python probe/extract_evidence.py tmp/bridge-events.jsonl"
