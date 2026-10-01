// Differential regression: the browser-side validator must agree with the
// frozen Python reference (`python agentic/schema/validate.py --json`) on the
// (layer, code, pointer) issue sets for every fixture.
//
// Skipped (loudly) only when no Python interpreter is available on PATH.

import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { parseAndValidate } from "../js/protocol.js";
import { REPO_ROOT, UPSTREAM_FIXTURES, minimalDoc, readSchema, triples } from "./test-helpers.js";

const schema = readSchema();
const negativeDir = join(UPSTREAM_FIXTURES, "negative");

function findPython() {
  for (const candidate of ["python", "python3", "py"]) {
    const probe = spawnSync(candidate, ["--version"], { shell: false });
    if (!probe.error && probe.status === 0) return candidate;
  }
  return null;
}

function pythonTriples(python, files) {
  const result = spawnSync(python, [join(REPO_ROOT, "agentic", "schema", "validate.py"), "--json", ...files], {
    shell: false,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  const out = new Map();
  for (const line of result.stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const report = JSON.parse(line);
    out.set(basename(report.source), triples(report.issues).map((t) => t.join("|")).sort());
  }
  return out;
}

test("differential vs Python reference on the frozen fixtures", (t) => {
  const python = findPython();
  if (!python) {
    t.skip("no python interpreter on PATH; differential check skipped (other tests still cover the JS validator)");
    return;
  }
  const files = [
    ...readdirSync(negativeDir)
      .filter((name) => name.endsWith(".json") && name !== "expected.json")
      .map((name) => join(negativeDir, name)),
    ...readdirSync(UPSTREAM_FIXTURES)
      .filter((name) => name.startsWith("valid_") && name.endsWith(".json"))
      .map((name) => join(UPSTREAM_FIXTURES, name)),
  ];
  const py = pythonTriples(python, files);
  for (const file of files) {
    const text = readFileSync(file, "utf8");
    const { report } = parseAndValidate(text, schema, { source: file });
    const jsTriples = triples(report.issues).map((t) => t.join("|")).sort();
    assert.deepEqual(jsTriples, py.get(basename(file)) || [], basename(file));
  }
});

test("differential vs Python reference: prototype keys and astral strings", (t) => {
  const python = findPython();
  if (!python) {
    t.skip("no python interpreter on PATH; differential check skipped");
    return;
  }
  const dir = mkdtempSync(join(tmpdir(), "aat-viewer-diff-proto-"));
  try {
    const files = [];
    const write = (name, text) => {
      const path = join(dir, name);
      writeFileSync(path, text, "utf8");
      files.push(path);
    };
    const base = minimalDoc();
    // raw `__proto__` top-level key (Python rejects via additionalProperties)
    write("raw-proto.json", `${JSON.stringify(base).slice(0, -1)}, "__proto__": {"unexpected": true}}`);
    write("raw-constructor.json", `${JSON.stringify(base).slice(0, -1)}, "constructor": {"unexpected": true}}`);
    write("raw-tostring.json", `${JSON.stringify(base).slice(0, -1)}, "toString": {"unexpected": true}}`);
    // nested prototype-named key inside an instrument
    write("nested-tostring.json", JSON.stringify(base).replace('"events":', '"toString": {}, "events":'));
    // astral label: 100 code points is valid, 129 is not
    const astralOk = minimalDoc();
    astralOk.instruments[0].label = "\u{1F3B9}".repeat(100);
    write("astral-ok.json", JSON.stringify(astralOk));
    const astralBad = minimalDoc();
    astralBad.instruments[0].label = "\u{1F3B9}".repeat(129);
    write("astral-too-long.json", JSON.stringify(astralBad));

    const py = pythonTriples(python, files);
    for (const file of files) {
      const text = readFileSync(file, "utf8");
      const { report } = parseAndValidate(text, schema, { source: file });
      const jsTriples = triples(report.issues).map((item) => item.join("|")).sort();
      assert.deepEqual(jsTriples, py.get(basename(file)) || [], basename(file));
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("differential vs Python reference on mutated documents", (t) => {
  const python = findPython();
  if (!python) {
    t.skip("no python interpreter on PATH; differential check skipped");
    return;
  }
  const dir = mkdtempSync(join(tmpdir(), "aat-viewer-diff-"));
  try {
    const cases = {};
    const badOrder = minimalDoc();
    badOrder.instruments[0].events = [
      { id: "ev-2", onset_seconds: 2 },
      { id: "ev-1", onset_seconds: 1 },
    ];
    cases["mut-unsorted.json"] = badOrder;
    const badTempo = minimalDoc();
    badTempo.tempo = { bpm: 128, source: "unknown", confidence: 0 };
    cases["mut-tempo.json"] = badTempo;
    const badPath = minimalDoc();
    badPath.instruments[0].stem = { filename: "C:/x.wav", sha256: "c".repeat(64) };
    cases["mut-path.json"] = badPath;
    const badNested = minimalDoc();
    badNested.instruments[0].events[0].pitch = { midi: 60.5, confidence: 2, source: "mock" };
    cases["mut-pitch.json"] = badNested;
    const deep = minimalDoc();
    deep.instruments[0].events[0].method = "ok-slug";
    cases["mut-clean.json"] = deep;

    const files = [];
    for (const [name, doc] of Object.entries(cases)) {
      const path = join(dir, name);
      writeFileSync(path, JSON.stringify(doc), "utf8");
      files.push(path);
    }
    const py = pythonTriples(python, files);
    for (const file of files) {
      const text = readFileSync(file, "utf8");
      const { report } = parseAndValidate(text, schema, { source: file });
      const jsTriples = triples(report.issues).map((t) => t.join("|")).sort();
      assert.deepEqual(jsTriples, py.get(basename(file)) || [], basename(file));
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
