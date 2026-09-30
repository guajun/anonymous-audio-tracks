// Strict JSON loader policy (frozen, mirrors agentic/schema loader.py):
// duplicate keys, non-standard constants, non-finite numbers, depth 64, BOM.

import test from "node:test";
import assert from "node:assert/strict";

import { MAX_DEPTH, depthIssues, loadText, numericIssues } from "../js/strict-json.js";

const codes = (result) => result.issues.map((issue) => `${issue.layer}/${issue.code}@${issue.pointer}`);

test("valid document parses and produces no issues", () => {
  const { data, issues } = loadText('{"a": [1, 2.5, -3e2], "b": {"c": true, "d": null}, "e": "x\\u0041"}');
  assert.deepEqual(issues, []);
  assert.deepEqual(data, { a: [1, 2.5, -300], b: { c: true, d: null }, e: "xA" });
});

test("duplicate object keys are rejected, not last-wins (E_DUPLICATE_KEY)", () => {
  const result = loadText('{"label": "a", "label": "b"}');
  assert.equal(result.data, null);
  assert.deepEqual(codes(result), ["parse/E_DUPLICATE_KEY@"]);
  assert.match(result.issues[0].message, /label/);
});

test("duplicate keys nested also rejected", () => {
  const result = loadText('{"instruments": [{"id": "a", "id": "b"}]}');
  assert.deepEqual(codes(result), ["parse/E_DUPLICATE_KEY@"]);
});

test("NaN / Infinity literals are rejected (E_PARSE)", () => {
  for (const text of ['{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}']) {
    const result = loadText(text);
    assert.deepEqual(codes(result), ["parse/E_PARSE@"], text);
    assert.match(result.issues[0].message, /NaN|Infinity/);
  }
});

test("1e999 (legal syntax, infinite value) is rejected with exact pointer (E_NONFINITE)", () => {
  const result = loadText('{"audio": {"duration_seconds": 1e999}}');
  assert.equal(result.data, null);
  assert.deepEqual(codes(result), ["parse/E_NONFINITE@/audio/duration_seconds"]);
});

test("huge integers beyond float64 interop range are rejected (E_NONFINITE)", () => {
  const result = loadText(`{"audio": {"sample_rate": 1${"0".repeat(400)}}}`);
  assert.deepEqual(codes(result), ["parse/E_NONFINITE@/audio/sample_rate"]);
});

test("multiple non-finite numbers all reported with pointers", () => {
  const result = loadText('{"a": 1e999, "b": [2e999]}');
  assert.deepEqual(codes(result).sort(), ["parse/E_NONFINITE@/a", "parse/E_NONFINITE@/b/0"]);
});

test("depth policy: 64 levels accepted, 65 rejected (E_PARSE)", () => {
  const depth64 = `${"[".repeat(63)}0${"]".repeat(63)}`;
  assert.deepEqual(loadText(depth64).issues, []);
  assert.equal(loadText(depth64).data instanceof Array, true);

  const depth65 = `${"[".repeat(64)}0${"]".repeat(64)}`;
  const result = loadText(depth65);
  const expectedPointer = `/${new Array(64).fill("0").join("/")}`;
  assert.deepEqual(codes(result), [`parse/E_PARSE@${expectedPointer}`]);
});

test("very deep nesting is a controlled rejection, not a crash (E_PARSE)", () => {
  const deep = `${"[".repeat(5000)}0${"]".repeat(5000)}`;
  const result = loadText(deep);
  assert.equal(result.data, null);
  assert.equal(result.issues.length, 1);
  assert.equal(result.issues[0].code, "E_PARSE");
});

test("BOM is rejected (E_PARSE)", () => {
  const result = loadText('\uFEFF{"a": 1}');
  assert.deepEqual(codes(result), ["parse/E_PARSE@"]);
});

test("syntax errors report line/column and stay controlled", () => {
  const result = loadText('{\n  "a": 1,\n  "b" 2\n}');
  assert.deepEqual(codes(result), ["parse/E_PARSE@"]);
  assert.match(result.issues[0].message, /第 3 行/);
});

test("trailing data after the top-level value is rejected", () => {
  assert.deepEqual(codes(loadText('{"a": 1} extra')), ["parse/E_PARSE@"]);
});

test("empty text is rejected", () => {
  assert.deepEqual(codes(loadText("   ")), ["parse/E_PARSE@"]);
});

test("strings: escapes and unescaped control characters", () => {
  assert.deepEqual(loadText('{"s": "a\\n\\t\\u0041\\\\\\""}').data, { s: 'a\n\tA\\"' });
  assert.deepEqual(codes(loadText('{"s": "a\nb"}')), ["parse/E_PARSE@"]); // raw newline in string
  assert.deepEqual(codes(loadText('{"s": "\\q"}')), ["parse/E_PARSE@"]); // illegal escape
});

test("ids with trailing newlines are kept verbatim for the pattern layer", () => {
  const { data, issues } = loadText('{"id": "inst-1\\n"}');
  assert.deepEqual(issues, []);
  assert.equal(data.id, "inst-1\n");
});

test("depthIssues / numericIssues mirrors for programmatic documents", () => {
  assert.deepEqual(numericIssues({ a: NaN })[0].code, "E_NONFINITE");
  assert.deepEqual(numericIssues({ a: 1e308 }), []);
  assert.deepEqual(depthIssues({ a: 1 }), []);
  let node = 0;
  for (let index = 0; index < MAX_DEPTH - 1; index += 1) node = [node];
  assert.equal(depthIssues(node).length, 0);
  node = [node];
  assert.equal(depthIssues(node).length, 1);
});
