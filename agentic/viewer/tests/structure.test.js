// Structural layer (Draft 2020-12 portable subset) — frozen behaviour.

import test from "node:test";
import assert from "node:assert/strict";

import { validateStructure } from "../js/structure.js";
import { minimalDoc, readSchema, triples } from "./test-helpers.js";

const schema = readSchema();
const codes = (doc) => triples(validateStructure(doc, schema));

test("minimal valid document has no structural issues", () => {
  assert.deepEqual(codes(minimalDoc()), []);
});

test("unknown top-level field is rejected (additionalProperties)", () => {
  const doc = minimalDoc();
  doc.extra = 1;
  assert.deepEqual(codes(doc), [["structure", "additionalProperties", ""]]);
});

test("unknown nested fields are rejected too", () => {
  const doc = minimalDoc();
  doc.audio.extra = 1;
  doc.instruments[0].events[0].extra = 2;
  const found = codes(doc);
  assert.deepEqual(found, [
    ["structure", "additionalProperties", "/audio"],
    ["structure", "additionalProperties", "/instruments/0/events/0"],
  ]);
});

test("missing required fields are reported with pointers", () => {
  const doc = minimalDoc();
  delete doc.audio.sha256;
  delete doc.instruments[0].events;
  assert.deepEqual(codes(doc), [
    ["structure", "required", "/audio"],
    ["structure", "required", "/instruments/0"],
  ]);
});

test("boolean is not a number (type)", () => {
  const doc = minimalDoc();
  doc.tempo.bpm = true;
  assert.deepEqual(codes(doc), [["structure", "type", "/tempo/bpm"]]);
});

test("integer fields accept integral floats, reject non-integral", () => {
  const doc = minimalDoc();
  doc.audio.sample_rate = 2.0;
  assert.deepEqual(codes(doc), []);
  doc.audio.sample_rate = 2.5;
  assert.deepEqual(codes(doc), [["structure", "type", "/audio/sample_rate"]]);
});

test("schema_version must be the exact constant", () => {
  const doc = minimalDoc();
  doc.schema_version = "agentic-audio-tracks/v2";
  assert.deepEqual(codes(doc), [["structure", "const", "/schema_version"]]);
});

test("pattern is absolute-anchored: trailing newlines are rejected", () => {
  for (const suffix of ["\n", "\r\n"]) {
    const doc = minimalDoc();
    doc.instruments[0].id = `inst-1${suffix}`;
    assert.deepEqual(codes(doc), [["structure", "pattern", "/instruments/0/id"]], JSON.stringify(suffix));
  }
  const doc = minimalDoc();
  doc.audio.sha256 = `${"a".repeat(64)}\n`;
  assert.deepEqual(codes(doc), [["structure", "pattern", "/audio/sha256"]]);
});

test("ranges: confidence [0,1], bpm maximum 1000, onset minimum 0", () => {
  const doc = minimalDoc();
  doc.instruments[0].confidence = 1.5;
  doc.tempo.bpm = 1000.5;
  doc.instruments[0].events[0].onset_seconds = -0.1;
  const found = codes(doc);
  assert.deepEqual(found, [
    ["structure", "maximum", "/instruments/0/confidence"],
    ["structure", "minimum", "/instruments/0/events/0/onset_seconds"],
    ["structure", "maximum", "/tempo/bpm"],
  ]);
});

test("string length limits are enforced", () => {
  const doc = minimalDoc();
  doc.instruments[0].label = "";
  doc.limitations = [""];
  const found = codes(doc);
  assert.deepEqual(found, [
    ["structure", "minLength", "/instruments/0/label"],
    ["structure", "minLength", "/limitations/0"],
  ]);
});

test("provenance.steps needs at least one entry (minItems)", () => {
  const doc = minimalDoc();
  doc.provenance.steps = [];
  assert.deepEqual(codes(doc), [["structure", "minItems", "/provenance/steps"]]);
});

test("tempo.bpm allows null (unknown tempo)", () => {
  const doc = minimalDoc();
  doc.tempo = { bpm: null, source: "unknown", confidence: 0 };
  assert.deepEqual(codes(doc), []);
});

test("pitch is all-or-nothing (required trio)", () => {
  const doc = minimalDoc();
  doc.instruments[0].events[0].pitch = { midi: 60 };
  const found = codes(doc);
  assert.deepEqual(found, [["structure", "required", "/instruments/0/events/0/pitch"]]);
});

test("errors are sorted deterministically by (pointer, code)", () => {
  const doc = minimalDoc();
  doc.instruments[0].events[1].onset_seconds = -1;
  doc.instruments[0].events[0].onset_seconds = -1;
  const found = validateStructure(doc, schema).map((issue) => issue.pointer);
  assert.deepEqual(found, [...found].sort());
});
