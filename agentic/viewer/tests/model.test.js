// View model: typed arrays, row identity, BPM state and the frozen guarantee
// that BPM edits never touch onset seconds.

import test from "node:test";
import assert from "node:assert/strict";

import { applyBpm, beatGrid, buildModel, createBpmState, rowVisibleSlice, validateBpm } from "../js/model.js";
import { parseAndValidate } from "../js/protocol.js";
import { minimalDoc, readSchema } from "./test-helpers.js";

const schema = readSchema();

function modelOf(doc) {
  const { data, report } = parseAndValidate(JSON.stringify(doc), schema, { source: "t" });
  assert.equal(report.ok, true, JSON.stringify(report.issues));
  return buildModel(data);
}

test("rows keep identity: same label, different id are never merged", () => {
  const doc = minimalDoc();
  doc.instruments.push({
    id: "inst-2",
    label: "piano", // same label as inst-1
    description: "second piano row",
    source: "llm",
    confidence: 0.5,
    events: [{ id: "ev-9", onset_seconds: 3 }],
  });
  const model = modelOf(doc);
  assert.equal(model.rows.length, 2);
  assert.equal(model.rows[0].label, "piano");
  assert.equal(model.rows[1].label, "piano");
  assert.equal(model.rows[0].id, "inst-1");
  assert.equal(model.rows[1].id, "inst-2");
});

test("onsets land in typed arrays; absent durations stay NaN (no invented lengths)", () => {
  const model = modelOf(minimalDoc());
  const row = model.rows[0];
  assert.ok(row.onset instanceof Float64Array);
  assert.deepEqual(Array.from(row.onset), [0.5, 1.5]);
  assert.equal(Number.isNaN(row.duration[0]), true); // no duration in the fixture
  assert.equal(row.duration[1], 0.4);
  assert.equal(row.withDuration, 1);
  assert.deepEqual(row.ids, ["ev-1", "ev-2"]);
});

test("BPM state: JSON prefill, unknown prefill, manual edit", () => {
  const known = createBpmState(modelOf(minimalDoc()));
  assert.equal(known.value, 120);
  assert.equal(known.origin, "json");
  assert.equal(known.unknown, false);

  const doc = minimalDoc();
  doc.tempo = { bpm: null, source: "unknown", confidence: 0 };
  const unknown = createBpmState(modelOf(doc));
  assert.equal(unknown.value, null);
  assert.equal(unknown.unknown, true);
  assert.match(unknown.message, /未知/);

  const edited = applyBpm(known, "128.5");
  assert.equal(edited.changed, true);
  assert.equal(edited.state.value, 128.5);
  assert.equal(edited.state.origin, "manual");
});

test("BPM validation: finite positive <= 1000, everything else rejected", () => {
  for (const bad of ["", "abc", "-5", "0", "NaN", "Infinity", "1e999", "1000.5", null, undefined, {}, true]) {
    const result = validateBpm(bad);
    assert.equal(result.ok, false, JSON.stringify(bad));
  }
  for (const good of ["0.001", "1", "120", "999.9", "1000", 240]) {
    assert.equal(validateBpm(good).ok, true, JSON.stringify(good));
  }
});

test("invalid BPM input keeps the previous grid (no change applied)", () => {
  const state = createBpmState(modelOf(minimalDoc()));
  const { state: next, changed } = applyBpm(state, "0");
  assert.equal(changed, false);
  assert.equal(next.value, 120);
  assert.match(next.lastError, /正数/);
});

test("changing BPM never changes onset seconds (frozen guarantee)", () => {
  const model = modelOf(minimalDoc());
  const before = model.rows.map((row) => Array.from(row.onset));
  let state = createBpmState(model);
  for (const value of ["240", "60", "333.33", "0.5"]) {
    const applied = applyBpm(state, value);
    assert.equal(applied.changed, true);
    state = applied.state;
  }
  const after = model.rows.map((row) => Array.from(row.onset));
  assert.deepEqual(after, before);
});

test("beat grid derives from BPM + origin and can be dense-limited", () => {
  const state = { valid: true, value: 120 }; // 0.5s per beat
  const grid = beatGrid(state, 0, 0, 4, 512);
  assert.equal(grid.interval, 0.5);
  assert.deepEqual(grid.beats.slice(0, 3), [0, 0.5, 1]);

  const shifted = beatGrid(state, 0.25, 0, 2, 512);
  assert.deepEqual(shifted.beats.slice(0, 3), [0.25, 0.75, 1.25]);

  const dense = beatGrid({ valid: true, value: 1000 }, 0, 0, 600, 50); // ~16ms per beat
  assert.ok(dense.beats.length <= 50);
  assert.ok(dense.skipped >= 2);

  const none = beatGrid({ valid: false, value: null }, 0, 0, 4, 512);
  assert.deepEqual(none.beats, []);
});

test("beat grid is bounded for extreme finite metadata (no RangeError / hang)", () => {
  // review finding: Number.MAX_VALUE BPM used to throw RangeError: Invalid array length
  for (const value of [Number.MAX_VALUE, Number.MIN_VALUE, 1e300, 1e-300, 0.001, 1000]) {
    const grid = beatGrid({ valid: true, value }, 0, 0, 300, 512);
    assert.ok(grid.beats.length <= 512, `beats bounded for bpm=${value}`);
    assert.ok(grid.interval === null || Number.isFinite(grid.interval), `interval finite for bpm=${value}`);
    if (grid.unsupported) assert.equal(typeof grid.reason, "string");
  }
  // non-finite origin / window must stay controlled too
  const weird = beatGrid({ valid: true, value: 120 }, Number.MAX_VALUE, 0, 300, 512);
  assert.ok(weird.beats.length <= 512);
  const reversed = beatGrid({ valid: true, value: 120 }, 0, 10, 5, 512);
  assert.deepEqual(reversed.beats, []);
});

test("rowVisibleSlice culls events outside the window", () => {
  const model = modelOf(minimalDoc());
  const row = model.rows[0];
  assert.deepEqual(rowVisibleSlice(row, 0, 1), { from: 0, to: 1, count: 1 });
  assert.deepEqual(rowVisibleSlice(row, 10, 20), { from: 2, to: 2, count: 0 });
});

test("empty events row stays valid and reports zero counts", () => {
  const doc = minimalDoc();
  doc.instruments[0].events = [];
  const model = modelOf(doc);
  assert.equal(model.rows[0].count, 0);
  assert.equal(model.totalEvents, 0);
  assert.deepEqual(rowVisibleSlice(model.rows[0], 0, 30), { from: 0, to: 0, count: 0 });
});
