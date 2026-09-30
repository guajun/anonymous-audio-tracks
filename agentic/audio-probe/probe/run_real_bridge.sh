#!/usr/bin/env bash
# run_real_bridge.sh — REAL Gemini run through the project-local audio bridge
# (issue #30). Model is fixed to google/gemini-3.8-flash; no fallback.
#
# REAL vs MOCK: this makes real API calls (paid). Everything in tests/ and the
# offline probes is mock/offline; only this script and run_real_native.sh are
# real-model evidence.
#
# NOTE on counts: one Pi run is NOT one model/API call — the existing run
# contains 2 assistant requests (tool-use turn + final turn) and 2 audio_attach
# tool calls. Measured usage lives in reports/00-environment.md.
#
# Failure semantics (review item 1):
#   - hard timeout   : AUDIO_PROBE_TIMEOUT (default 300 s) via probe/run_bounded.py (exit 4);
#   - command errors : exit code propagates (3), artifacts preserved;
#   - provider error : Pi exit 0 is still a FAILURE (exit 5) when the event
#                      stream has stopReason error/aborted or errorMessage.
#
# Env overrides (offline fake-pi regression tests): PI_BIN, AUDIO_PROBE_TMP,
# AUDIO_PROBE_TIMEOUT.
#
# Usage (from agentic/audio-probe/):
#   bash probe/run_real_bridge.sh
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

echo "== real run: audio bridge + google/gemini-3.8-flash =="
echo "   artifacts (preserved on failure/timeout): $OUT_DIR/bridge-events.jsonl, $OUT_DIR/bridge-stderr.txt"
PI_AUDIO_BRIDGE_ROOT="$(pwd)/fixtures" \
python probe/run_bounded.py --timeout "$TIMEOUT_SECS" \
  --stdout "$OUT_DIR/bridge-events.jsonl" \
  --stderr "$OUT_DIR/bridge-stderr.txt" \
  --events "$OUT_DIR/bridge-events.jsonl" \
  -- $PI_BIN --provider google --model gemini-3.8-flash \
    --extension ./bridge/audio-bridge.ts \
    --tools audio_attach \
    --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files \
    --session-dir "$OUT_DIR/sessions" --name audio-probe-bridge \
    --mode json --print \
    -- "Call the audio_attach tool once for each of these files: fixture-a.wav and fixture-b.wav (paths relative to the audio root). After both are attached, describe each recording separately: (1) how many distinct sound events you hear in it, (2) whether its pitch goes up, down, or stays level across the recording. Base your answer only on what you actually hear from the attached audio."

echo "== done =="
echo "raw events : $OUT_DIR/bridge-events.jsonl"
echo "stderr     : $OUT_DIR/bridge-stderr.txt"
echo "sanitize with: python probe/extract_evidence.py $OUT_DIR/bridge-events.jsonl"
