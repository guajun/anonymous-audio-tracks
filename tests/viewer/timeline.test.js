import test from "node:test";
import assert from "node:assert/strict";

import {
  absoluteToLocalTime,
  activityAtAbsoluteTime,
  applyLoopCorrection,
  clamp,
  clampIntervals,
  formatTime,
  localToAbsoluteTime,
  nearestSample,
  pageStartFor,
  shiftWindow,
  stepAdvance,
  thresholdIntervals,
  totalActiveSeconds,
  validateLoopRange,
  windowBounds,
} from "../../viewer/js/timeline.js";
import { parseTrajectoryText } from "../../viewer/js/protocol.js";
import { buildSnapshot } from "../../viewer/js/snapshot.js";
import { approx, fixtureText } from "./test-helpers.js";

const known = parseTrajectoryText(fixtureText("trajectory.known-times.json"));
const clip = parseTrajectoryText(fixtureText("trajectory.clip.json"));
const trackA = known.tracks[0];
const trackB = known.tracks[1];
const trackC = known.tracks[2];

test("local/absolute conversions apply track_start_seconds", () => {
  assert.equal(localToAbsoluteTime(0, 12), 12);
  assert.equal(localToAbsoluteTime(0.24, 12), 12.24);
  assert.ok(approx(absoluteToLocalTime(12.24, 12), 0.24, 1e-12));
  assert.equal(absoluteToLocalTime(12, 12), 0);
});

test("stepAdvance scales elapsed wall clock by playback rate", () => {
  assert.equal(stepAdvance(1, 0.5, 1), 1.5);
  assert.equal(stepAdvance(1, 0.5, 0.5), 1.25);
  assert.equal(stepAdvance(1, 0.5, 2), 2);
  assert.equal(clamp(-3, 0, 10), 0);
  assert.equal(clamp(30, 0, 10), 10);
  assert.equal(clamp(Number.NaN, 0, 10), 0);
});

test("activity interpolation on the known-time fixture", () => {
  assert.equal(activityAtAbsoluteTime(trackA, 0), 0);
  assert.equal(activityAtAbsoluteTime(trackA, 0.5), 0.8);
  assert.equal(activityAtAbsoluteTime(trackA, 1.0), 0.9);
  assert.equal(activityAtAbsoluteTime(trackA, 3.5), 0);
  assert.ok(approx(activityAtAbsoluteTime(trackA, 0.75), 0.85));
  assert.ok(approx(activityAtAbsoluteTime(trackA, 0.3125), 0.5));
  assert.ok(approx(activityAtAbsoluteTime(trackA, 3.4), 0.12));
  assert.equal(activityAtAbsoluteTime(trackA, -0.001), null);
  assert.equal(activityAtAbsoluteTime(trackA, 3.501), null);
  assert.equal(activityAtAbsoluteTime(trackB, 1.1), 0);
});

test("nearestSample reports probability and raw slot provenance", () => {
  const sample = nearestSample(trackA, 0.6, 0);
  assert.equal(sample.index, 1);
  assert.equal(sample.absoluteSeconds, 0.5);
  assert.equal(sample.localSeconds, 0.5);
  assert.equal(sample.activity, 0.8);
  assert.equal(sample.slotIndex, 0);
  const later = nearestSample(trackA, 3.05, 0);
  assert.equal(later.index, 6);
  assert.equal(later.slotIndex, 2);
  assert.equal(nearestSample(trackA, 9, 0), null);
});

test("threshold intervals use interpolated crossings and merge neighbours", () => {
  const intervals = thresholdIntervals(trackA, 0.5, 0);
  assert.equal(intervals.length, 2);
  assert.ok(approx(intervals[0][0], 0.3125));
  assert.ok(approx(intervals[0][1], 1.25));
  assert.ok(approx(intervals[1][0], 2.875));
  assert.ok(approx(intervals[1][1], 3.0833333333333335));
  assert.ok(approx(totalActiveSeconds(intervals), 1.1458333333333335));

  assert.deepEqual(thresholdIntervals(trackB, 0.5, 0), []);
  assert.deepEqual(thresholdIntervals(trackC, 0.5, 0), [[1, 2]]);
  assert.deepEqual(thresholdIntervals(trackC, 0, 0), [[1, 2]]);
  // threshold 0 keeps an all-zero track fully active by definition.
  assert.deepEqual(thresholdIntervals(trackB, 0, 0), [[0.1, 3.1]]);
});

test("clip intervals are shifted to the local timeline", () => {
  const intervals = thresholdIntervals(
    clip.tracks[0],
    0.5,
    clip.audio.trackStartSeconds,
  );
  assert.ok(approx(intervals[0][0], 0.17142857142857143));
  assert.ok(approx(intervals[0][1], 0.48));
});

test("clampIntervals keeps only the visible part of a window", () => {
  assert.deepEqual(clampIntervals([[0, 1], [2, 3], [5, 6]], 0.5, 2.5), [
    [0.5, 1],
    [2, 2.5],
  ]);
  assert.deepEqual(clampIntervals([[9, 10]], 0, 1), []);
});

test("loop range validation covers pending, invalid and valid endpoints", () => {
  assert.equal(validateLoopRange(1, 2, null).reason, "duration-unavailable");
  assert.equal(validateLoopRange(1, 2, Number.NaN).reason, "duration-unavailable");
  assert.equal(validateLoopRange(1, 2, 0).reason, "duration-unavailable");
  assert.equal(validateLoopRange(Number.NaN, 2, 4).reason, "non-finite");
  assert.equal(validateLoopRange(-1, 2, 4).reason, "negative");
  assert.equal(validateLoopRange(1, 4.5, 4).reason, "out-of-range");
  assert.equal(validateLoopRange(1, 1.01, 4).reason, "too-short");
  assert.deepEqual(validateLoopRange(1, 2, 4), { ok: true, start: 1, end: 2 });
  assert.deepEqual(validateLoopRange(0, 4, 4), { ok: true, start: 0, end: 4 });
});

test("applyLoopCorrection wraps at B and is a no-op without a valid loop", () => {
  const loop = { start: 1, end: 2 };
  assert.equal(applyLoopCorrection(1.5, loop), 1.5);
  assert.equal(applyLoopCorrection(2, loop), 1);
  assert.ok(approx(applyLoopCorrection(2.25, loop), 1.25));
  assert.ok(approx(applyLoopCorrection(10.1, loop), 1.1));
  assert.equal(applyLoopCorrection(1.5, null), 1.5);
  assert.equal(applyLoopCorrection(1.5, { start: 2, end: 2 }), 1.5);
});

test("window bounds and page navigation clamp to the timeline", () => {
  assert.deepEqual(windowBounds(5, 30, 20), { start: 5, end: 25 });
  assert.deepEqual(windowBounds(-5, 30, 20), { start: 0, end: 20 });
  assert.deepEqual(windowBounds(25, 30, 20), { start: 10, end: 30 });
  assert.deepEqual(windowBounds(0, 0, 20), { start: 0, end: 0 });
  assert.deepEqual(windowBounds(0, 12, 20), { start: 0, end: 12 });

  assert.equal(pageStartFor(0, 600, 20), 0);
  assert.equal(pageStartFor(25, 600, 20), 20);
  assert.equal(pageStartFor(599, 600, 20), 580);
  assert.equal(pageStartFor(-4, 600, 20), 0);
  assert.equal(pageStartFor(25, 0, 20), 0);

  assert.equal(shiftWindow(0, 1, 600, 20), 20);
  assert.equal(shiftWindow(20, -1, 600, 20), 0);
  assert.equal(shiftWindow(580, 1, 600, 20), 580);
  assert.equal(shiftWindow(0, -1, 600, 20), 0);
});

test("formatTime renders millisecond readouts and handles carries", () => {
  assert.equal(formatTime(0), "00:00.000");
  assert.equal(formatTime(61.5), "01:01.500");
  assert.equal(formatTime(59.9996), "01:00.000");
  assert.equal(formatTime(3661.25), "61:01.250");
  assert.equal(formatTime(-1), "--:--.---");
  assert.equal(formatTime(Number.NaN), "--:--.---");
});

function simulatePlayback({ start, wallSeconds, rate, loop, stepSeconds = 0.05 }) {
  let time = start;
  const trace = [];
  const steps = Math.max(1, Math.round(wallSeconds / stepSeconds));
  for (let step = 0; step < steps; step += 1) {
    time = stepAdvance(time, wallSeconds / steps, rate);
    time = applyLoopCorrection(time, loop);
    trace.push(time);
  }
  return { time, trace };
}

test("known-time fixture stays in sync under play, pause, seek, rate and loop", () => {
  // Play at 1x from 0.5s: at local 1.0s the sampled value is exactly 0.9.
  let simulated = simulatePlayback({ start: 0.5, wallSeconds: 0.5, rate: 1 });
  assert.ok(approx(simulated.time, 1.0, 1e-9));
  let snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: simulated.time,
    threshold: 0.5,
  });
  assert.ok(approx(snapshot.tracks[0].value, 0.9));
  assert.ok(approx(snapshot.absoluteTime, 1.0, 1e-9));

  // Paused: the clock does not move and the readout is unchanged.
  snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: 1.0,
    threshold: 0.5,
  });
  assert.equal(snapshot.tracks[0].value, 0.9);
  assert.equal(snapshot.localTime, 1.0);

  // Seek to 0.75s: interpolation must show 0.85.
  snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: 0.75,
    threshold: 0.5,
  });
  assert.ok(approx(snapshot.tracks[0].value, 0.85));
  assert.equal(snapshot.tracks[0].active, true);

  // Slow motion: 2 wall seconds at 0.5x advance one timeline second.
  simulated = simulatePlayback({ start: 0, wallSeconds: 2, rate: 0.5 });
  assert.ok(approx(simulated.time, 1.0, 1e-9));

  // Double speed: 1 wall second at 2x advances two timeline seconds.
  simulated = simulatePlayback({ start: 1, wallSeconds: 1, rate: 2 });
  assert.ok(approx(simulated.time, 3.0, 1e-9));

  // Loop 0.5-1.0s: never leaves the loop, wraps at least twice in 3s.
  simulated = simulatePlayback({
    start: 0.6,
    wallSeconds: 3,
    rate: 1,
    loop: { start: 0.5, end: 1.0 },
  });
  assert.ok(simulated.trace.every((time) => time >= 0.5 && time <= 1.0 + 1e-9));
  const wraps = simulated.trace.filter(
    (time, index) => index > 0 && time < simulated.trace[index - 1],
  ).length;
  assert.ok(wraps >= 2, `expected at least two wraps, got ${wraps}`);
  const loopSnapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: simulated.time,
    threshold: 0.5,
  });
  assert.ok(loopSnapshot.localTime <= 1.0);
  assert.equal(loopSnapshot.tracks[0].active, true);

  // Rate change mid-playback keeps using the same shared clock value.
  simulated = simulatePlayback({ start: 0.0, wallSeconds: 0.5, rate: 1 });
  simulated = simulatePlayback({
    start: simulated.time,
    wallSeconds: 1,
    rate: 0.5,
  });
  assert.ok(approx(simulated.time, 1.0, 1e-9));
});
