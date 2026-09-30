#!/usr/bin/env bash
# run_real_native.sh — REAL attempt to feed audio through Pi's NATIVE input
# paths (`@file` attachment + the built-in `read` tool), no bridge extension.
# This is the failure-evidence run for issue #30. Model is fixed to
# google/gemini-3.8-flash; no fallback.
#
# Expected (see reports/20-native-source-audit.md): the .wav reaches the model
# as UTF-8 mojibake text (or a read error), never as audio.
#
# Usage (from agentic/audio-probe/):
#   bash probe/run_real_native.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p tmp/sessions

echo "== auth check (status only; no key printed) =="
pi auth check --provider google --json

echo "== fixtures =="
python probe/make_fixture.py --all

echo "== real run: native @file + read tool + google/gemini-3.8-flash =="
pi --provider google --model gemini-3.8-flash \
  --tools read \
  --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files \
  --session-dir ./tmp/sessions --name audio-probe-native \
  --mode json --print \
  -- "@fixtures/fixture-a.wav" "Use the read tool on fixtures/fixture-a.wav and tell me exactly what you perceive from it: how many distinct sound events, and whether the pitch goes up, down, or stays level. If you cannot actually perceive audio, say so explicitly instead of guessing." \
  > tmp/native-events.jsonl 2> tmp/native-stderr.txt || echo "pi exited non-zero; see tmp/native-stderr.txt"

echo "== done =="
echo "raw events : tmp/native-events.jsonl"
echo "stderr     : tmp/native-stderr.txt"
echo "sanitize with: python probe/extract_evidence.py tmp/native-events.jsonl"
