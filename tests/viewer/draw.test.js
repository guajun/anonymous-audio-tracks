import test from "node:test";
import assert from "node:assert/strict";

import {
  niceTimeStep,
  overviewTimeFromX,
  trackColor,
} from "../../viewer/js/draw.js";

test("niceTimeStep picks readable grid steps", () => {
  assert.equal(niceTimeStep(0), 1);
  assert.equal(niceTimeStep(Number.NaN), 1);
  assert.equal(niceTimeStep(8), 1);
  assert.equal(niceTimeStep(16), 2);
  assert.equal(niceTimeStep(40), 5);
  assert.equal(niceTimeStep(80), 10);
  assert.equal(niceTimeStep(4), 0.5);
});

test("overview click maps x-position to local seek time", () => {
  assert.equal(overviewTimeFromX(-10, 600, 600), 0);
  assert.equal(overviewTimeFromX(300, 600, 600), 300);
  assert.equal(overviewTimeFromX(600, 600, 600), 600);
  assert.equal(overviewTimeFromX(900, 600, 600), 600);
  assert.equal(overviewTimeFromX(100, 0, 600), 0);
  assert.equal(overviewTimeFromX(100, 600, Number.NaN), 0);
});

test("track colors cycle deterministically by protocol track order", () => {
  assert.equal(trackColor(0), trackColor(12));
  assert.notEqual(trackColor(0), trackColor(1));
});
