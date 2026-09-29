import test from "node:test";
import assert from "node:assert/strict";

import { parseTrajectoryText } from "../../viewer/js/protocol.js";
import { buildSnapshot, sameInstant } from "../../viewer/js/snapshot.js";
import { approx, fixtureText } from "./test-helpers.js";

const known = parseTrajectoryText(fixtureText("trajectory.known-times.json"));
const clip = parseTrajectoryText(fixtureText("trajectory.clip.json"));
const empty = parseTrajectoryText(fixtureText("trajectory.empty.json"));

test("no trajectory and no audio yields an empty snapshot", () => {
  const snapshot = buildSnapshot({ trajectory: null, currentTime: 5 });
  assert.equal(snapshot.trajectoryLoaded, false);
  assert.equal(snapshot.duration, 0);
  assert.equal(snapshot.durationSource, "none");
  assert.deepEqual(snapshot.tracks, []);
  assert.deepEqual(snapshot.window, { start: 0, end: 0 });
});

test("trajectory span drives the timeline until audio metadata arrives", () => {
  const snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: null,
    currentTime: 1,
    windowLength: 2,
  });
  assert.equal(snapshot.durationSource, "trajectory");
  assert.equal(snapshot.duration, 4);
  assert.deepEqual(snapshot.window, { start: 0, end: 2 });
});

test("loaded audio duration wins over the trajectory span", () => {
  const snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 30,
    currentTime: 1,
    windowLength: 20,
  });
  assert.equal(snapshot.durationSource, "audio");
  assert.equal(snapshot.duration, 30);
  assert.deepEqual(snapshot.window, { start: 0, end: 20 });
});

test("all tracks read from the same shared clock and absolute conversion", () => {
  const snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: 0.75,
    threshold: 0.5,
  });
  assert.equal(snapshot.localTime, 0.75);
  assert.equal(snapshot.absoluteTime, 0.75);
  assert.equal(snapshot.visibleTrackCount, 3);
  const [trackA, trackB, trackC] = snapshot.tracks;
  assert.ok(approx(trackA.value, 0.85));
  assert.equal(trackA.active, true);
  assert.equal(trackB.value, 0);
  assert.equal(trackC.value, null, "trk-c only starts at 1.0s");
  assert.equal(trackA.nearest.localSeconds, 0.5);

  const atOne = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: 1.0,
    threshold: 0.5,
  });
  assert.equal(atOne.tracks[0].value, 0.9);
  assert.equal(atOne.tracks[2].value, 1);
  assert.equal(atOne.tracks[2].active, true);
});

test("clip trajectory: player seconds are local, protocol seconds absolute", () => {
  const snapshot = buildSnapshot({
    trajectory: clip,
    audioDuration: 0.5,
    currentTime: 0.24,
    threshold: 0.5,
  });
  assert.equal(snapshot.trackStart, 12);
  assert.equal(snapshot.localTime, 0.24);
  assert.equal(snapshot.absoluteTime, 12.24);
  assert.ok(approx(snapshot.tracks[0].value, 0.7));
  assert.equal(snapshot.tracks[0].active, true);
  assert.ok(approx(snapshot.tracks[0].nearest.localSeconds, 0.24));
  assert.equal(snapshot.tracks[0].nearest.absoluteSeconds, 12.24);
  // Feeding the absolute value to the player-local readout would yield null.
  const wrong = buildSnapshot({
    trajectory: clip,
    audioDuration: 0.5,
    currentTime: 12.24,
    threshold: 0.5,
  });
  assert.equal(wrong.localTime, 0.5);
});

test("threshold and window clipping change the displayed intervals", () => {
  const snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: 0.9,
    threshold: 0.85,
    windowStart: 0,
    windowLength: 1,
    followPlayhead: false,
  });
  assert.deepEqual(snapshot.window, { start: 0, end: 1 });
  const trackA = snapshot.tracks[0];
  assert.equal(trackA.active, true);
  assert.equal(trackA.intervals.length, 1);
  assert.ok(approx(trackA.intervals[0][0], 0.75));
  assert.ok(approx(trackA.intervals[0][1], 1.03125));
  assert.equal(trackA.windowIntervals.length, 1);
  assert.ok(approx(trackA.windowIntervals[0][0], 0.75, 1e-12));
  assert.equal(trackA.windowIntervals[0][1], 1);
});

test("hidden tracks stay in the snapshot but not in visibleTrackCount", () => {
  const snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: 0.75,
    hiddenTracks: ["trk-a", "trk-b"],
  });
  assert.equal(snapshot.tracks.length, 3);
  assert.equal(snapshot.tracks[0].visible, false);
  assert.equal(snapshot.visibleTrackCount, 1);
});

test("clock beyond the timeline is clamped, not wrapped", () => {
  const snapshot = buildSnapshot({
    trajectory: known,
    audioDuration: 4,
    currentTime: 10,
    threshold: 0.5,
  });
  assert.equal(snapshot.localTime, 4);
  assert.equal(snapshot.absoluteTime, 4);
  assert.equal(snapshot.tracks[0].value, null);
});

test("empty trajectory boundary renders with no tracks and zero duration", () => {
  const snapshot = buildSnapshot({
    trajectory: empty,
    audioDuration: null,
    currentTime: 0,
  });
  assert.equal(snapshot.trajectoryLoaded, true);
  assert.equal(snapshot.duration, 0);
  assert.equal(snapshot.visibleTrackCount, 0);
  assert.deepEqual(snapshot.window, { start: 0, end: 0 });
});

test("sameInstant compares the shared clock position", () => {
  const left = buildSnapshot({ trajectory: known, audioDuration: 4, currentTime: 1 });
  const right = buildSnapshot({ trajectory: known, audioDuration: 4, currentTime: 1 });
  assert.equal(sameInstant(left, right), true);
  const moved = buildSnapshot({ trajectory: known, audioDuration: 4, currentTime: 1.1 });
  assert.equal(sameInstant(left, moved), false);
});
