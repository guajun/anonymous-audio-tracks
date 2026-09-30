// Demo/stress generators: deterministic output, hash- and timeline-consistent
// with the committed fixtures, and clearly labelled as mock data.

import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildDemoWav, demoDoc, demoEvents, writeDemo } from "../tools/make-demo.mjs";
import { buildStressDoc, mulberry32 } from "../tools/make-stress.mjs";
import { parseAndValidate } from "../js/protocol.js";
import { VIEWER_ROOT, readSchema } from "./test-helpers.js";

const schema = readSchema();
const fixturesDir = join(VIEWER_ROOT, "fixtures", "demo");

const sha256 = (buffer) => createHash("sha256").update(buffer).digest("hex");

test("committed demo JSON records the real SHA-256 of the generated WAV", () => {
  const wav = buildDemoWav();
  const hash = sha256(wav);
  const doc = JSON.parse(readFileSync(join(fixturesDir, "demo-track.json"), "utf8"));
  assert.equal(doc.audio.sha256, hash);
  assert.equal(doc.audio.duration_seconds, 12);
  assert.equal(wav.length, 44 + 12 * 44100 * 2);
});

test("demo generator is deterministic (byte-identical WAV and JSON)", async () => {
  const dirA = mkdtempSync(join(tmpdir(), "aat-demo-a-"));
  const dirB = mkdtempSync(join(tmpdir(), "aat-demo-b-"));
  try {
    const resultA = await writeDemo({ outDir: dirA, fixturesDir: join(dirA, "docs") });
    const resultB = await writeDemo({ outDir: dirB, fixturesDir: join(dirB, "docs") });
    assert.equal(resultA.wavSha256, resultB.wavSha256);
    for (const name of ["demo-track.json", "demo-unknown-tempo.json", "demo-empty-events.json", "demo-malicious-labels.json"]) {
      const textA = readFileSync(join(dirA, "docs", name), "utf8");
      const textB = readFileSync(join(dirB, "docs", name), "utf8");
      assert.equal(textA, textB, `${name} must be deterministic`);
      const committed = readFileSync(join(fixturesDir, name), "utf8");
      assert.equal(textA, committed, `${name} in fixtures/demo must match the generator output`);
    }
  } finally {
    rmSync(dirA, { recursive: true, force: true });
    rmSync(dirB, { recursive: true, force: true });
  }
});

test("demo documents validate and are labelled as mock, not research results", () => {
  for (const name of ["demo-track.json", "demo-unknown-tempo.json", "demo-empty-events.json", "demo-malicious-labels.json"]) {
    const text = readFileSync(join(fixturesDir, name), "utf8");
    const { report, data } = parseAndValidate(text, schema, { source: name });
    assert.deepEqual(report.issues, [], name);
    assert.equal(data.provenance.steps[0].source, "mock", name);
    assert.ok(data.limitations.some((line) => line.includes("mock") || line.includes("不是研究结果")), name);
  }
});

test("demo events align with the synthesized audio timeline", () => {
  const events = demoEvents();
  assert.ok(events.length >= 40);
  for (const event of events) {
    assert.ok(event.onset_seconds >= 0 && event.onset_seconds <= 12, JSON.stringify(event));
    if (event.duration_seconds !== null) {
      assert.ok(event.onset_seconds + event.duration_seconds <= 12.000001);
    }
    assert.ok(event.frequency > 0 && event.amplitude > 0);
  }
});

test("same label / different id rows exist in the demo document", () => {
  const doc = JSON.parse(readFileSync(join(fixturesDir, "demo-track.json"), "utf8"));
  const pianos = doc.instruments.filter((instrument) => instrument.label === "piano");
  assert.equal(pianos.length, 2);
  assert.notEqual(pianos[0].id, pianos[1].id);
});

test("unknown-tempo demo has tempo.bpm=null with the frozen unknown shape", () => {
  const doc = JSON.parse(readFileSync(join(fixturesDir, "demo-unknown-tempo.json"), "utf8"));
  assert.deepEqual(doc.tempo, { bpm: null, source: "unknown", confidence: 0 });
});

test("malicious-label demo keeps HTML in strings only (no markup structure)", () => {
  const doc = JSON.parse(readFileSync(join(fixturesDir, "demo-malicious-labels.json"), "utf8"));
  assert.ok(doc.instruments[0].label.includes("<img src=x"));
  assert.ok(doc.instruments[0].description.includes("<script>"));
});

test("stress document builder: sorted, in-bounds, deterministic", () => {
  const docA = buildStressDoc({ events: 500, seconds: 60, seed: 7 });
  const docB = buildStressDoc({ events: 500, seconds: 60, seed: 7 });
  assert.deepEqual(docA, docB);
  const { report, data } = parseAndValidate(JSON.stringify(docA), schema, { source: "stress" });
  assert.deepEqual(report.issues, []);
  const total = data.instruments.reduce((sum, instrument) => sum + instrument.events.length, 0);
  assert.equal(total, 500 - (500 % 8));
  assert.equal(data.provenance.steps[0].source, "mock");
  for (const instrument of data.instruments) {
    for (const event of instrument.events) {
      assert.ok(event.onset_seconds >= 0 && event.onset_seconds <= 60);
      if (event.duration_seconds !== undefined) {
        assert.ok(event.onset_seconds + event.duration_seconds <= 60.000001);
      }
    }
  }
});

test("stress PRNG is deterministic", () => {
  const a = mulberry32(34034);
  const b = mulberry32(34034);
  for (let index = 0; index < 100; index += 1) assert.equal(a(), b());
});
