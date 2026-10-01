// Semantic layer — cross-field rules of the frozen contract.

import test from "node:test";
import assert from "node:assert/strict";

import { unsafePathReason, validateSemantics } from "../js/semantic.js";
import { minimalDoc, triples } from "./test-helpers.js";

const codes = (doc) => triples(validateSemantics(doc));

test("minimal valid document has no semantic issues", () => {
  assert.deepEqual(codes(minimalDoc()), []);
});

test("event ids must be globally unique across instruments (E_ID_UNIQUE)", () => {
  const doc = minimalDoc();
  doc.instruments.push({
    id: "inst-2",
    label: "bass",
    description: "second row",
    source: "llm",
    confidence: 0.5,
    events: [{ id: "ev-1", onset_seconds: 2 }],
  });
  assert.deepEqual(codes(doc), [["semantic", "E_ID_UNIQUE", "/instruments/1/events/0/id"]]);
});

test("instrument ids must be unique (E_ID_UNIQUE)", () => {
  const doc = minimalDoc();
  doc.instruments[0].id = "inst-1";
  doc.instruments.push({ ...doc.instruments[0], events: [] });
  assert.deepEqual(codes(doc), [["semantic", "E_ID_UNIQUE", "/instruments/1/id"]]);
});

test("events must be sorted by (onset, id) (E_TIME_ORDER)", () => {
  const doc = minimalDoc();
  doc.instruments[0].events = [
    { id: "ev-2", onset_seconds: 1.5 },
    { id: "ev-1", onset_seconds: 0.5 },
  ];
  assert.deepEqual(codes(doc), [["semantic", "E_TIME_ORDER", "/instruments/0/events/1"]]);
});

test("simultaneous onsets are legal and ordered by id", () => {
  const doc = minimalDoc();
  doc.instruments[0].events = [
    { id: "ev-a", onset_seconds: 1 },
    { id: "ev-b", onset_seconds: 1 },
  ];
  assert.deepEqual(codes(doc), []);
});

test("time bounds: onset within [0, duration] with 1e-9 tolerance (E_TIME_RANGE)", () => {
  const doc = minimalDoc();
  doc.instruments[0].events = [{ id: "ev-1", onset_seconds: 30.000001 }]; // 1e-6 s past the end > 1e-9 tolerance
  assert.deepEqual(codes(doc), [["semantic", "E_TIME_RANGE", "/instruments/0/events/0/onset_seconds"]]);

  const ok = minimalDoc();
  ok.instruments[0].events = [{ id: "ev-1", onset_seconds: 30 }]; // exactly the end is allowed
  assert.deepEqual(codes(ok), []);
});

test("duration must be > 0 and fit the file (E_TIME_DURATION)", () => {
  const doc = minimalDoc();
  doc.instruments[0].events[1].duration_seconds = 29; // 1.5 + 29 > 30
  assert.deepEqual(codes(doc), [["semantic", "E_TIME_DURATION", "/instruments/0/events/1/duration_seconds"]]);

  const zero = minimalDoc();
  zero.instruments[0].events[1].duration_seconds = 0;
  assert.deepEqual(codes(zero), [["semantic", "E_TIME_DURATION", "/instruments/0/events/1/duration_seconds"]]);
});

test("tempo known/unknown are mutually exclusive (E_TEMPO)", () => {
  const a = minimalDoc();
  a.tempo = { bpm: null, source: "dsp", confidence: 0.5, beat_origin_seconds: 0 };
  const found = codes(a);
  assert.ok(found.some(([layer, code]) => layer === "semantic" && code === "E_TEMPO"));

  const b = minimalDoc();
  b.tempo = { bpm: 120, source: "unknown", confidence: 0 };
  assert.deepEqual(codes(b), [["semantic", "E_TEMPO", "/tempo/source"]]);

  const c = minimalDoc();
  c.tempo = { bpm: null, source: "unknown", confidence: 0 };
  assert.deepEqual(codes(c), []);
});

test("unknown tempo must not carry beat_origin_seconds (E_TEMPO)", () => {
  const doc = minimalDoc();
  doc.tempo = { bpm: null, source: "unknown", confidence: 0, beat_origin_seconds: 1 };
  assert.deepEqual(codes(doc), [["semantic", "E_TEMPO", "/tempo/beat_origin_seconds"]]);
});

test("beat origin must be inside the file (E_TIME_RANGE)", () => {
  const doc = minimalDoc();
  doc.tempo.beat_origin_seconds = 45;
  assert.deepEqual(codes(doc), [["semantic", "E_TIME_RANGE", "/tempo/beat_origin_seconds"]]);
});

test("confidence in [0,1]; midi integer 0..127 (E_RANGE)", () => {
  const doc = minimalDoc();
  doc.instruments[0].events[1].confidence = 2;
  doc.instruments[0].events[0].pitch = { midi: 200, confidence: 0.5, source: "mock" };
  const found = codes(doc);
  assert.deepEqual(found, [
    ["semantic", "E_RANGE", "/instruments/0/events/0/pitch/midi"],
    ["semantic", "E_RANGE", "/instruments/0/events/1/confidence"],
  ]);
});

test("booleans must not impersonate numbers (E_BOOL_NUMBER)", () => {
  const doc = minimalDoc();
  doc.tempo.confidence = true;
  assert.deepEqual(codes(doc), [["semantic", "E_BOOL_NUMBER", "/tempo/confidence"]]);
});

test("path safety (E_PATH) rejects absolute / UNC / URL / traversal / reserved names", () => {
  const bad = [
    "../secret/a.wav",
    "/etc/passwd",
    "C:/music/a.wav",
    "\\\\server\\share\\a.wav",
    "//server/share/a.wav",
    "https://host/a.wav",
    "song//a.wav",
    "song/./a.wav",
    "song/CON.wav",
    "song/a?.wav",
    "song/ a.wav",
    "song/a.wav ",
    "song/a.",
  ];
  for (const filename of bad) {
    const doc = minimalDoc();
    doc.audio.filename = filename;
    assert.deepEqual(codes(doc), [["semantic", "E_PATH", "/audio/filename"]], filename);
    assert.equal(typeof unsafePathReason(filename), "string");
  }
  const good = ["song/track.wav", "a", "stems/piano-1.flac", "x/y/z.wav", "demo-track.wav"];
  for (const filename of good) {
    assert.equal(unsafePathReason(filename), null, filename);
  }
});

test("control characters and overlong paths are rejected", () => {
  assert.equal(typeof unsafePathReason("a\u0007.wav"), "string");
  assert.equal(typeof unsafePathReason("a".repeat(513)), "string");
  assert.equal(unsafePathReason("a".repeat(512)), null);
});

test("stem filenames get the same path checks", () => {
  const doc = minimalDoc();
  doc.instruments[0].stem = { filename: "../x.wav", sha256: "b".repeat(64) };
  assert.deepEqual(codes(doc), [["semantic", "E_PATH", "/instruments/0/stem/filename"]]);
});

test("non-finite numbers in programmatic documents are refused (E_FINITE)", () => {
  const doc = minimalDoc();
  doc.audio.duration_seconds = Number.NaN;
  const found = codes(doc);
  assert.ok(found.some(([layer, code]) => layer === "semantic" && code === "E_FINITE"));
});
