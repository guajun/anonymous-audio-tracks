// Event pixel geometry: sustained events must be clipped to the viewport,
// never dropped when the onset is off-screen (review finding #4).

import test from "node:test";
import assert from "node:assert/strict";

import { eventPixelRect } from "../js/render.js";

const VIEW = { start: 5, end: 6 };
const WIDTH = 1000;

test("sustained event crossing the LEFT edge is visible and clipped (not dropped)", () => {
  // onset=0, duration=10 (ends at 10) with view [5,6]: review reproduction
  const rect = eventPixelRect(0, 10, VIEW, WIDTH);
  assert.equal(rect.visible, true);
  assert.equal(rect.clipped, true);
  assert.equal(rect.left, 0);
  assert.ok(rect.width > 0, "the clipped bar must still have width");
  // exactly the viewport width: the event covers the whole window
  assert.equal(rect.right, WIDTH);
});

test("fully expired event (ended before the window) is invisible", () => {
  const rect = eventPixelRect(0, 1, VIEW, WIDTH); // ends at 1 < 5
  assert.equal(rect.visible, false);
  assert.equal(rect.width, 0);
});

test("boundary: event that ends exactly at view.start counts as touching (visible)", () => {
  const rect = eventPixelRect(0, 5, VIEW, WIDTH); // ends exactly at 5
  assert.equal(rect.visible, true);
});

test("event crossing the RIGHT edge is clipped to the viewport", () => {
  const rect = eventPixelRect(5.5, 10, VIEW, WIDTH); // 5.5..15.5 vs view 5..6
  assert.equal(rect.visible, true);
  assert.equal(rect.clipped, true);
  assert.equal(rect.right, WIDTH);
  assert.ok(rect.left > 0 && rect.left < WIDTH);
});

test("event fully inside the window is drawn at full extent (no clip)", () => {
  const rect = eventPixelRect(5.2, 0.5, VIEW, WIDTH);
  assert.equal(rect.visible, true);
  assert.equal(rect.clipped, false);
  assert.ok(rect.width > 0);
});

test("onset-only events: visible only when the onset is inside the window", () => {
  assert.equal(eventPixelRect(5.5, NaN, VIEW, WIDTH).visible, true);
  assert.equal(eventPixelRect(5, NaN, VIEW, WIDTH).visible, true); // left boundary
  assert.equal(eventPixelRect(6, NaN, VIEW, WIDTH).visible, true); // right boundary
  assert.equal(eventPixelRect(4.9, NaN, VIEW, WIDTH).visible, false);
  assert.equal(eventPixelRect(6.1, NaN, VIEW, WIDTH).visible, false);
  // a missing duration never fabricates a rectangle
  assert.equal(eventPixelRect(5.5, NaN, VIEW, WIDTH).width, 0);
});

test("zero/negative durations behave like onset-only (no invented lengths)", () => {
  assert.equal(eventPixelRect(5.5, 0, VIEW, WIDTH).visible, true);
  assert.equal(eventPixelRect(5.5, -1, VIEW, WIDTH).width, 0);
  assert.equal(eventPixelRect(4, 0, VIEW, WIDTH).visible, false);
});

test("onset at the far right of a long window maps to the right edge", () => {
  const rect = eventPixelRect(300, NaN, { start: 0, end: 300 }, 1200);
  assert.equal(rect.visible, true);
  assert.equal(rect.x0, 1200);
});
