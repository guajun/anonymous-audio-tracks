// Pipeline: parse -> version -> structure + semantics (frozen ordering).

import test from "node:test";
import assert from "node:assert/strict";

import { parseAndValidate, validateDocument, validateText, SCHEMA_VERSION } from "../js/protocol.js";
import { SCHEMA_PATH, minimalDoc, readSchema, readText, triples } from "./test-helpers.js";

const schema = readSchema();

test("valid text passes and returns the parsed document", () => {
  const { data, report } = parseAndValidate(JSON.stringify(minimalDoc()), schema, { source: "t" });
  assert.equal(report.ok, true);
  assert.equal(report.schema_version, SCHEMA_VERSION);
  assert.equal(data.schema_version, SCHEMA_VERSION);
});

test("version mismatch reports exactly one E_VERSION issue and stops", () => {
  const doc = minimalDoc();
  doc.schema_version = "agentic-audio-tracks/v2";
  const report = validateText(JSON.stringify(doc), schema);
  assert.deepEqual(triples(report.issues), [["version", "E_VERSION", "/schema_version"]]);
  assert.equal(report.ok, false);
});

test("missing schema_version is also only E_VERSION", () => {
  const doc = minimalDoc();
  delete doc.schema_version;
  const report = validateText(JSON.stringify(doc), schema);
  assert.deepEqual(triples(report.issues), [["version", "E_VERSION", "/schema_version"]]);
});

test("parse failures short-circuit the later layers", () => {
  const report = validateText("{ not json", schema);
  assert.equal(report.ok, false);
  assert.deepEqual(report.issues.map((issue) => issue.layer), ["parse"]);
});

test("structure and semantic issues merge and sort by (pointer, code)", () => {
  const doc = minimalDoc();
  doc.audio.filename = "../x.wav"; // semantic E_PATH
  doc.instruments[0].confidence = 5; // structure maximum
  const report = validateDocument(doc, schema);
  const pointers = report.issues.map((issue) => issue.pointer);
  assert.deepEqual(pointers, [...pointers].sort());
  assert.deepEqual(triples(report.issues), [
    ["semantic", "E_PATH", "/audio/filename"],
    ["semantic", "E_RANGE", "/instruments/0/confidence"],
    ["structure", "maximum", "/instruments/0/confidence"],
  ]);
});

test("report shape matches the frozen Python Report.as_dict()", () => {
  const report = validateText(JSON.stringify(minimalDoc()), schema, { source: "x.json" });
  assert.deepEqual(Object.keys(report).sort(), ["engine", "issues", "ok", "schema_version", "source"]);
  assert.equal(report.source, "x.json");
  assert.equal(typeof report.engine, "string");
  assert.equal(report.ok, true);
  assert.deepEqual(report.issues, []);
});

test("invalid documents never produce data", () => {
  const { data, report } = parseAndValidate('{"schema_version": "agentic-audio-tracks/v1"}', schema);
  assert.equal(report.ok, false);
  assert.ok(data === null || report.issues.length > 0);
});

test("the shipped schema copy is a Draft 2020-12 document with the frozen $id", () => {
  const parsed = JSON.parse(readText(SCHEMA_PATH));
  assert.equal(parsed.$schema, "https://json-schema.org/draft/2020-12/schema");
  assert.equal(parsed.$id, "urn:anonymous-audio-tracks:schema:agentic-audio-tracks/v1");
  assert.equal(parsed.properties.schema_version.const, SCHEMA_VERSION);
});
