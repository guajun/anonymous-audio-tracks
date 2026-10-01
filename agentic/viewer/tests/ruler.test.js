// Bounded ruler for ALL finite accepted metadata (review item 1):
// `audio.duration_seconds = Number.MAX_VALUE` is schema-valid, so renderRuler
// must terminate quickly and keep the page responsive before any grid warning.

import test from "node:test";
import assert from "node:assert/strict";

import { MAX_RULER_TICKS, formatSeconds, niceTimeStep, renderRuler, renderTracks } from "../js/render.js";
import { beatGrid, buildModel, createBpmState } from "../js/model.js";
import { validateDocument } from "../js/protocol.js";
import { minimalDoc, readSchema } from "./test-helpers.js";

const schema = readSchema();

function fakeCtx() {
  const ops = { fillText: 0, fillRect: 0, strokes: 0 };
  return {
    ops,
    save() {},
    restore() {},
    clearRect() {},
    fillRect() {
      ops.fillRect += 1;
    },
    fillText() {
      ops.fillText += 1;
    },
    beginPath() {},
    moveTo() {},
    lineTo() {},
    stroke() {
      ops.strokes += 1;
    },
    set fillStyle(_v) {},
    get fillStyle() {
      return "";
    },
    set strokeStyle(_v) {},
    get strokeStyle() {
      return "";
    },
    set font(_v) {},
    get font() {
      return "";
    },
    set textBaseline(_v) {},
    get textBaseline() {
      return "";
    },
    set lineWidth(_v) {},
    get lineWidth() {
      return 1;
    },
    set globalAlpha(_v) {},
    get globalAlpha() {
      return 1;
    },
  };
}

test("renderRuler terminates quickly for Number.MAX_VALUE ranges (no 2s hang)", () => {
  const windows = [
    { start: 0, end: Number.MAX_VALUE },
    { start: 0, end: 1e18 },
    { start: Number.MAX_VALUE / 2, end: Number.MAX_VALUE },
    { start: 0, end: 12 },
    { start: 5, end: 5.001 },
    { start: Number.MIN_VALUE, end: 1 },
  ];
  for (const view of windows) {
    const ctx = fakeCtx();
    const started = Date.now();
    renderRuler(ctx, view, { width: 1200, height: 26 });
    const elapsed = Date.now() - started;
    assert.ok(elapsed < 500, `renderRuler(${view.start}, ${view.end}) took ${elapsed}ms`);
    assert.ok(ctx.ops.fillText <= MAX_RULER_TICKS, `tick labels bounded: ${ctx.ops.fillText}`);
    assert.ok(ctx.ops.strokes <= MAX_RULER_TICKS, `tick strokes bounded: ${ctx.ops.strokes}`);
  }
});

test("renderTracks with a Number.MAX_VALUE duration document stays bounded too", () => {
  const doc = minimalDoc();
  doc.audio.duration_seconds = Number.MAX_VALUE;
  doc.instruments[0].events = [{ id: "ev-1", onset_seconds: 0 }, { id: "ev-2", onset_seconds: 1e300 }];
  const report = validateDocument(doc, schema, { source: "huge" });
  assert.deepEqual(report.issues, [], "the huge-duration document itself must be schema-valid");

  const model = buildModel(doc);
  const ctx = fakeCtx();
  const started = Date.now();
  const stats = renderTracks(ctx, model, { start: 0, end: Number.MAX_VALUE }, {
    width: 1200,
    height: 200,
    rulerHeight: 0,
    rowHeight: 44,
    bpmState: createBpmState(model),
    now: () => Date.now(),
  });
  const elapsed = Date.now() - started;
  assert.ok(elapsed < 1000, `renderTracks took ${elapsed}ms`);
  assert.ok(stats.rows.length === 1);
  // grid reports a readable unsupported state instead of hanging
  const grid = beatGrid({ valid: true, value: 120 }, 0, 0, Number.MAX_VALUE, 512);
  assert.equal(grid.unsupported, true);
  assert.equal(typeof grid.reason, "string");
});

test("niceTimeStep scales up for huge spans so ticks stay bounded", () => {
  for (const span of [Number.MAX_VALUE, 1e18, 1e12, 1e6, 300, 12]) {
    const step = niceTimeStep({ start: 0, end: span }, 1200);
    const ticks = span / step;
    assert.ok(ticks <= MAX_RULER_TICKS * 2, `span=${span} -> ticks=${ticks} (step=${step})`);
    assert.ok(step > 0 && Number.isFinite(step));
  }
});

test("formatSeconds renders huge finite values compactly", () => {
  assert.equal(formatSeconds(12.5, 1), "12.5s");
  assert.equal(formatSeconds(0.25, 0.001), "0.250s");
  assert.match(formatSeconds(Number.MAX_VALUE, 1), /e\+/i);
  assert.equal(formatSeconds(Number.NaN, 1), "—");
});
