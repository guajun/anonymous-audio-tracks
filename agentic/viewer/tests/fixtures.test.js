// Regression over the FROZEN issue #32 fixtures (read-only):
//   - every negative fixture must be rejected with its expected
//     (layer, code, pointer) triple,
//   - every valid fixture must pass cleanly.

import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { parseAndValidate } from "../js/protocol.js";
import { UPSTREAM_FIXTURES, hasTriple, readSchema, triples } from "./test-helpers.js";

const schema = readSchema();
const negativeDir = join(UPSTREAM_FIXTURES, "negative");
const expected = JSON.parse(readFileSync(join(negativeDir, "expected.json"), "utf8"));

test("frozen negative fixtures: every case hits its expected issue triple", () => {
  for (const caseEntry of expected.cases) {
    const path = join(negativeDir, caseEntry.file);
    const text = readFileSync(path, "utf8");
    const { report } = parseAndValidate(text, schema, { source: caseEntry.file });
    assert.equal(report.ok, false, `${caseEntry.file} must be rejected`);
    assert.ok(
      hasTriple(report.issues, [caseEntry.layer, caseEntry.code, caseEntry.pointer]),
      `${caseEntry.file}: expected ${caseEntry.layer}/${caseEntry.code}@${caseEntry.pointer}, got ${JSON.stringify(
        triples(report.issues),
      )}`,
    );
  }
});

test("frozen negative fixture files and the index stay in sync", () => {
  const files = readdirSync(negativeDir)
    .filter((name) => name.endsWith(".json") && name !== "expected.json")
    .sort();
  const indexed = expected.cases.map((entry) => entry.file).sort();
  assert.deepEqual(files, indexed);
});

test("frozen valid fixtures pass the browser-side validator", () => {
  const files = readdirSync(UPSTREAM_FIXTURES)
    .filter((name) => name.startsWith("valid_") && name.endsWith(".json"))
    .sort();
  assert.ok(files.length >= 4, "expected the frozen valid_* fixtures");
  for (const name of files) {
    const text = readFileSync(join(UPSTREAM_FIXTURES, name), "utf8");
    const { report } = parseAndValidate(text, schema, { source: name });
    assert.deepEqual(triples(report.issues), [], `${name} must validate cleanly`);
    assert.equal(report.ok, true);
  }
});
