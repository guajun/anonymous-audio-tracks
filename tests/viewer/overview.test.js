import test from "node:test";
import assert from "node:assert/strict";

import { parseTrajectoryText } from "../../viewer/js/protocol.js";
import { computeOverviewBins } from "../../viewer/js/overview.js";
import { fixtureText } from "./test-helpers.js";

const known = parseTrajectoryText(fixtureText("trajectory.known-times.json"));
const clip = parseTrajectoryText(fixtureText("trajectory.clip.json"));

test("overview bins merge into the requested column count", () => {
  const bins = computeOverviewBins(known.tracks, 0, 4, 8);
  assert.equal(bins.length, 3);
  assert.equal(bins[0].trackId, "trk-a");
  assert.equal(bins[0].max.length, 8);
  assert.equal(bins[0].hasData, true);
  assert.ok(Math.max(...bins[0].max) >= 0.9 - 1e-6);
  // trk-c is active over [1, 2] -> columns 2 and 3 for width 8 / duration 4.
  assert.ok(bins[2].max[2] === 1 && bins[2].max[3] === 1);
  assert.equal(bins[2].max[0], 0);
});

test("gaps between sampled columns are interpolated, including from zero", () => {
  const track = {
    trackId: "ramp",
    centerTimes: [0, 4],
    activity: [0, 1],
  };
  const [bin] = computeOverviewBins([track], 0, 4, 5);
  assert.deepEqual(Array.from(bin.max), [0, 0.25, 0.5, 0.75, 1]);
});

test("track_start_seconds shifts bins back into local time", () => {
  const bins = computeOverviewBins(clip.tracks, clip.audio.trackStartSeconds, 0.5, 5);
  assert.equal(bins[0].max.length, 5);
  // local points: 0, 0.24, 0.48 -> columns 0, 2, 4 with values 0, 0.7, 0.9.
  assert.equal(bins[0].max[0], 0);
  assert.ok(bins[0].max[2] >= 0.7 - 1e-6);
  assert.ok(bins[0].max[4] >= 0.9 - 1e-6);
});

test("empty or zero-duration timelines produce empty envelopes, not crashes", () => {
  const [bin] = computeOverviewBins(known.tracks.slice(0, 1), 0, 0, 10);
  assert.equal(bin.hasData, false);
  assert.deepEqual(Array.from(bin.max), new Array(10).fill(0));
  const empty = computeOverviewBins(known.tracks, 0, 4, 0);
  assert.equal(empty.length, 3);
  assert.equal(empty[0].max.length, 0);
});

test("long-song bins stay bounded for an hour of 20 ms points", () => {
  const points = 180000;
  const centerTimes = Array.from({ length: points }, (_, index) => index * 0.02);
  const activity = Array.from({ length: points }, (_, index) =>
    (Math.sin(index / 50) + 1) / 2,
  );
  const track = { trackId: "long", centerTimes, activity };
  const started = Date.now();
  const [bin] = computeOverviewBins([track], 0, points * 0.02, 1000);
  const elapsed = Date.now() - started;
  assert.equal(bin.max.length, 1000);
  assert.ok(elapsed < 5000, `overview binning took ${elapsed}ms`);
  assert.ok(Math.max(...bin.max) > 0.9);
});
