// Schema-parity regressions for prototype-named keys and Unicode lengths
// (first-review findings): the JS validator must agree with the frozen Python
// reference even for `__proto__` / `constructor` / `toString` keys, inherited
// programmatic fields, and astral (non-BMP) strings.

import test from "node:test";
import assert from "node:assert/strict";

import { parseAndValidate, validateDocument } from "../js/protocol.js";
import { validateStructure, stringLength, hasOwn } from "../js/structure.js";
import { loadText } from "../js/strict-json.js";
import { minimalDoc, readSchema, triples } from "./test-helpers.js";

const schema = readSchema();

function docText(extraKey, extraValue) {
  const doc = minimalDoc();
  return `${JSON.stringify(doc).slice(0, -1)}, ${JSON.stringify(extraKey)}: ${JSON.stringify(extraValue)}}`;
}

test("raw __proto__ key is an own key and rejected as additionalProperties (not a prototype set)", () => {
  const raw = docText("__proto__", { unexpected: true });
  const { data, report } = parseAndValidate(raw, schema, { source: "proto" });
  assert.equal(report.ok, false);
  assert.deepEqual(triples(report.issues), [["structure", "additionalProperties", ""]]);
  // own key survives parsing; the prototype was NOT mutated
  assert.equal(hasOwn(data, "__proto__"), true);
  assert.equal(Object.prototype.unexpected, undefined);
  assert.equal({}.unexpected, undefined);
});

test("raw constructor / toString keys are rejected like Python (own-property checks)", () => {
  for (const key of ["constructor", "toString", "valueOf", "hasOwnProperty", "__proto__"]) {
    const raw = docText(key, { unexpected: true });
    const { report } = parseAndValidate(raw, schema, { source: key });
    assert.equal(report.ok, false, key);
    assert.deepEqual(triples(report.issues), [["structure", "additionalProperties", ""]], key);
  }
});

test("prototype-named keys nested inside instruments/events are rejected", () => {
  const doc = minimalDoc();
  const raw = JSON.stringify(doc).replace('"events":', '"toString": {}, "events":');
  const { report } = parseAndValidate(raw, schema, { source: "nested" });
  assert.equal(report.ok, false);
  assert.deepEqual(triples(report.issues), [["structure", "additionalProperties", "/instruments/0"]]);
});

test("programmatic prototype-named keys are rejected too (no schema-property inheritance)", () => {
  const doc = minimalDoc();
  Object.defineProperty(doc, "constructor", { value: 1, enumerable: true });
  const issues = validateStructure(doc, schema);
  assert.deepEqual(triples(issues), [["structure", "additionalProperties", ""]]);

  const nested = minimalDoc();
  Object.defineProperty(nested.audio, "toString", { value: "x", enumerable: true });
  assert.deepEqual(triples(validateStructure(nested, schema)), [["structure", "additionalProperties", "/audio"]]);
});

test("inherited fields never satisfy required/properties (Python dict own-key parity)", () => {
  const doc = minimalDoc();
  delete doc.audio.sha256;
  // a prototype carrying sha256 must NOT count as present
  const withProto = Object.create({ sha256: "f".repeat(64) });
  Object.assign(withProto, doc.audio);
  const mutated = minimalDoc();
  mutated.audio = withProto;
  assert.deepEqual(triples(validateStructure(mutated, schema)), [["structure", "required", "/audio"]]);

  const semanticFree = validateDocument(mutated, schema, { source: "t" });
  assert.equal(semanticFree.ok, false);
});

test("programmatic NaN still refused on objects with weird prototypes", () => {
  const doc = minimalDoc();
  doc.audio.duration_seconds = Number.NaN;
  const report = validateDocument(doc, schema, { source: "t" });
  // pipeline numeric-policy layer reports E_NONFINITE (same as the Python
  // validate_document early return); the semantic fallback is E_FINITE.
  assert.ok(report.issues.some((issue) => issue.code === "E_NONFINITE"), JSON.stringify(report.issues));
});

test("string lengths count Unicode code points, not UTF-16 units", () => {
  assert.equal(stringLength("🎹"), 1);
  assert.equal(stringLength("🎹".repeat(100)), 100);
  assert.equal(stringLength("a🎹b"), 3);

  const ok = minimalDoc();
  ok.instruments[0].label = "🎹".repeat(100); // 100 code points <= 128
  assert.deepEqual(validateStructure(ok, schema), []);

  const boundary = minimalDoc();
  boundary.instruments[0].label = "🎹".repeat(128);
  assert.deepEqual(validateStructure(boundary, schema), []);

  const tooLong = minimalDoc();
  tooLong.instruments[0].label = "🎹".repeat(129);
  assert.deepEqual(triples(validateStructure(tooLong, schema)), [["structure", "maxLength", "/instruments/0/label"]]);

  const ascii = minimalDoc();
  ascii.instruments[0].label = "x".repeat(129);
  assert.deepEqual(triples(validateStructure(ascii, schema)), [["structure", "maxLength", "/instruments/0/label"]]);
});

test("astral strings pass the semantic layer and the pipeline", () => {
  const doc = minimalDoc();
  doc.instruments[0].label = "🎹🎸🥁".repeat(20);
  doc.instruments[0].description = "合成演示：\u{1F3B9}\u{1F3B8}\u{1F389}";
  const { report } = parseAndValidate(JSON.stringify(doc), schema, { source: "astral" });
  assert.deepEqual(report.issues, []);
});

test("strict parser keeps astral escapes and duplicate-key policy intact", () => {
  const { data, issues } = loadText('{"s": "\\uD83C\\uDFB9"}');
  assert.deepEqual(issues, []);
  assert.equal(data.s, "🎹");
  const dup = loadText('{"a": 1, "a": 2}');
  assert.equal(dup.issues[0].code, "E_DUPLICATE_KEY");
});
