// Shared helpers for the agentic/viewer Node test suite.
// Everything resolves relative to this file so tests pass from any checkout.

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

/** agentic/viewer */
export const VIEWER_ROOT = resolve(HERE, "..");
/** repository worktree root */
export const REPO_ROOT = resolve(HERE, "..", "..", "..");
/** frozen contract copies shipped with the viewer */
export const SCHEMA_PATH = join(VIEWER_ROOT, "schema", "agentic-audio-tracks-v1.schema.json");
export const SEMANTIC_RULES_PATH = join(VIEWER_ROOT, "schema", "semantic_rules.json");
/** upstream frozen contract (issue #32) */
export const UPSTREAM_SCHEMA_DIR = join(REPO_ROOT, "agentic", "schema", "schema");
export const UPSTREAM_FIXTURES = join(REPO_ROOT, "agentic", "schema", "fixtures");

export function readSchema() {
  return JSON.parse(readFileSync(SCHEMA_PATH, "utf8"));
}

export function readText(path) {
  return readFileSync(path, "utf8");
}

/** Triplet list of issues: [layer, code, pointer]. */
export function triples(issues) {
  return issues.map((issue) => [issue.layer, issue.code, issue.pointer]);
}

export function hasTriple(issues, [layer, code, pointer]) {
  return issues.some((issue) => issue.layer === layer && issue.code === code && issue.pointer === pointer);
}

/** A minimal valid document, freely mutated by tests. */
export function minimalDoc() {
  return {
    schema_version: "agentic-audio-tracks/v1",
    audio: {
      filename: "song/track.wav",
      sha256: "a".repeat(64),
      duration_seconds: 30,
      sample_rate: 44100,
    },
    tempo: { bpm: 120, source: "dsp", confidence: 0.5, beat_origin_seconds: 0 },
    instruments: [
      {
        id: "inst-1",
        label: "piano",
        description: "test row",
        source: "llm",
        confidence: 0.5,
        events: [
          { id: "ev-1", onset_seconds: 0.5 },
          { id: "ev-2", onset_seconds: 1.5, duration_seconds: 0.4 },
        ],
      },
    ],
    provenance: { steps: [{ tool: "test", source: "mock" }] },
    limitations: ["test document"],
  };
}
