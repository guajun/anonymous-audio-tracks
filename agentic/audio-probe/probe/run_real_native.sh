#!/usr/bin/env bash
# run_real_native.sh — REAL attempt to feed audio through Pi's NATIVE input
# paths (`@file` attachment + the built-in `read` tool), no bridge extension.
# This is the failure-evidence run for issue #30. Model is fixed to
# google/gemini-3.8-flash; no fallback.
#
# Expected (see reports/20-native-source-audit.md): the .wav reaches the model
# as UTF-8 mojibake text (or a read error), never as audio.
#
# NOTE on counts: one Pi run is NOT one model/API call — each of the existing
# runs contains 2 assistant requests (tool-use turn + final turn). Measured
# usage lives in reports/00-environment.md.
#
# Failure semantics (review item 1):
#   - hard timeout   : AUDIO_PROBE_TIMEOUT (default 300 s), enforced by
#                      probe/run_bounded.py which kills the run and exits 4;
#   - command errors : exit code propagates (3), artifacts are preserved;
#   - provider error : a Pi exit code of 0 is still a FAILURE (exit 5) when the
#                      event stream contains stopReason error/aborted or an
#                      assistant errorMessage.
#
# Env overrides (used by offline fake-pi regression tests):
#   PI_BIN (default "pi"), AUDIO_PROBE_TMP (default "tmp"),
#   AUDIO_PROBE_TIMEOUT (default 300)
#
# Usage (from agentic/audio-probe/):
#   bash probe/run_real_native.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PI_BIN="${PI_BIN:-pi}"
OUT_DIR="${AUDIO_PROBE_TMP:-tmp}"
TIMEOUT_SECS="${AUDIO_PROBE_TIMEOUT:-300}"
mkdir -p "$OUT_DIR/sessions"

echo "== auth check (status only; no key printed) =="
$PI_BIN auth check --provider google --json

echo "== fixtures =="
python probe/make_fixture.py --all

echo "== real run: native @file + read tool + google/gemini-3.8-flash =="
echo "   artifacts (preserved on failure/timeout): $OUT_DIR/native-events.jsonl, $OUT_DIR/native-stderr.txt"
python probe/run_bounded.py --timeout "$TIMEOUT_SECS" \
  --stdout "$OUT_DIR/native-events.jsonl" \
  --stderr "$OUT_DIR/native-stderr.txt" \
  --events "$OUT_DIR/native-events.jsonl" \
  -- $PI_BIN --provider google --model gemini-3.8-flash \
    --tools read \
    --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files \
    --session-dir "$OUT_DIR/sessions" --name audio-probe-native \
    --mode json --print \
    -- "@fixtures/fixture-a.wav" "Use the read tool on fixtures/fixture-a.wav and tell me exactly what you perceive from it: how many distinct sound events, and whether the pitch goes up, down, or stays level. If you cannot actually perceive audio, say so explicitly instead of guessing."

echo "== done =="
echo "raw events : $OUT_DIR/native-events.jsonl"
echo "stderr     : $OUT_DIR/native-stderr.txt"
echo "sanitize with: python probe/extract_evidence.py $OUT_DIR/native-events.jsonl"
