// Timeline math: anchored wheel zoom, pan bounds, fit/reset constraints.

import test from "node:test";
import assert from "node:assert/strict";

import {
  MIN_WINDOW_SECONDS,
  clampWindow,
  fitWindow,
  lowerBound,
  panBy,
  secondsPerPixel,
  timeToX,
  upperBound,
  visibleSlice,
  xToTime,
  zoomAt,
} from "../js/timeline.js";

const TOTAL = 30;

test("clampWindow keeps the window inside [0, total] and honours the minimum duration", () => {
  assert.deepEqual(clampWindow(-5, 10, TOTAL), { start: 0, end: 15 }); // duration is preserved when clamping
  assert.deepEqual(clampWindow(25, 40, TOTAL), { start: 15, end: 30 });
  assert.deepEqual(clampWindow(0, 0, TOTAL), { start: 0, end: MIN_WINDOW_SECONDS });
  assert.deepEqual(clampWindow(10, 5, TOTAL), { start: 5, end: 10 });
  const tiny = clampWindow(0, 1e-12, TOTAL);
  assert.ok(tiny.end - tiny.start >= MIN_WINDOW_SECONDS * 0.999);
  const huge = clampWindow(-100, 1000, TOTAL);
  assert.deepEqual(huge, { start: 0, end: TOTAL });
});

test("zoomAt keeps the time under the pointer fixed (anchor)", () => {
  const win = { start: 10, end: 20 };
  const anchorX = 250;
  const width = 1000;
  const anchorTime = xToTime(anchorX, win, width); // 12.5s
  for (const factor of [0.5, 0.8, 1.25, 2, 0.1]) {
    const next = zoomAt(win, TOTAL, anchorTime, anchorX / width, factor);
    const after = xToTime(anchorX, next, width);
    assert.ok(Math.abs(after - anchorTime) < 1e-9, `factor=${factor}: ${after} != ${anchorTime}`);
  }
});

test("zoomAt respects the total bounds and the minimum window", () => {
  const win = { start: 0, end: 1 };
  let next = zoomAt(win, TOTAL, 0, 0, 0.0001); // zoom way in at the left edge
  assert.ok(next.end - next.start >= MIN_WINDOW_SECONDS * 0.999);
  assert.ok(next.start >= 0);

  next = zoomAt({ start: 0, end: 1 }, TOTAL, 0, 0, 1000); // zoom way out
  assert.deepEqual(next, { start: 0, end: TOTAL });

  next = zoomAt({ start: 29, end: 30 }, TOTAL, 30, 1, 1000);
  assert.ok(next.end <= TOTAL + 1e-12);
  assert.ok(next.start >= -1e-12);
});

test("repeated zoom in/out at the same anchor returns to the same window", () => {
  const win = { start: 10, end: 20 };
  const anchorTime = 15;
  let next = win;
  for (let index = 0; index < 6; index += 1) next = zoomAt(next, TOTAL, anchorTime, 0.5, 0.5);
  for (let index = 0; index < 6; index += 1) next = zoomAt(next, TOTAL, anchorTime, 0.5, 2);
  assert.ok(Math.abs(next.start - 10) < 1e-9);
  assert.ok(Math.abs(next.end - 20) < 1e-9);
});

test("panBy translates and clamps at the file bounds", () => {
  const win = { start: 10, end: 20 };
  assert.deepEqual(panBy(win, TOTAL, 5), { start: 15, end: 25 });
  assert.deepEqual(panBy(win, TOTAL, -100), { start: 0, end: 10 });
  assert.deepEqual(panBy(win, TOTAL, 100), { start: 20, end: 30 });
});

test("fitWindow covers the whole file", () => {
  assert.deepEqual(fitWindow(TOTAL), { start: 0, end: TOTAL });
  assert.deepEqual(fitWindow(0), { start: 0, end: 1 });
});

test("timeToX / xToTime are inverse and secondsPerPixel is consistent", () => {
  const win = { start: 5, end: 15 };
  const width = 800;
  for (const x of [0, 1, 200, 799, 800]) {
    assert.ok(Math.abs(timeToX(xToTime(x, win, width), win, width) - x) < 1e-9);
  }
  assert.equal(secondsPerPixel(win, width), 10 / 800);
});

test("binary search: lowerBound / upperBound / visibleSlice", () => {
  const onsets = new Float64Array([0, 1, 1, 1, 2, 5]);
  assert.equal(lowerBound(onsets, 1), 1);
  assert.equal(upperBound(onsets, 1), 4);
  assert.deepEqual(visibleSlice(onsets, 1, 2), { from: 1, to: 5, count: 4 });
  assert.deepEqual(visibleSlice(onsets, 3, 4), { from: 5, to: 5, count: 0 });
  // events that start before the window but still sound inside it:
  assert.deepEqual(visibleSlice(onsets, 0.5, 0.7, 1), { from: 0, to: 1, count: 1 });
});

test("culling with 100k sorted onsets stays O(log n) per query", () => {
  const count = 100000;
  const onsets = new Float64Array(count);
  for (let index = 0; index < count; index += 1) onsets[index] = (index / count) * 300;
  const started = performance.now();
  let total = 0;
  for (let query = 0; query < 10000; query += 1) {
    const t0 = (query / 10000) * 299;
    total += visibleSlice(onsets, t0, t0 + 0.1).count;
  }
  const elapsed = performance.now() - started;
  assert.ok(total > 0);
  assert.ok(elapsed < 2000, `binary search culling too slow: ${elapsed.toFixed(1)}ms`);
});
