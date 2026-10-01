// Waveform peak pyramid + LOD selection.

import test from "node:test";
import assert from "node:assert/strict";

import { buildPeakPyramid, peakColumns, pickLevel } from "../js/peaks.js";

function sine(seconds, rate, frequency) {
  const samples = new Float32Array(Math.floor(seconds * rate));
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = Math.sin((2 * Math.PI * frequency * index) / rate);
  }
  return samples;
}

test("pyramid levels halve the bucket count and double the bucket size", () => {
  const samples = new Float64Array(1024);
  for (let index = 0; index < samples.length; index += 1) samples[index] = index / 1024;
  const pyramid = buildPeakPyramid(samples, 64);
  assert.equal(pyramid.levels[0].min.length, 16);
  assert.equal(pyramid.levels[0].bucketSamples, 64);
  assert.equal(pyramid.levels[1].bucketSamples, 128);
  assert.equal(pyramid.levels[1].min.length, 8);
  assert.equal(pyramid.levels.at(-1).min.length, 1);
});

test("level 0 min/max are exact for a known ramp", () => {
  const samples = new Float64Array([1, 2, 3, 4, -5, 0]);
  const pyramid = buildPeakPyramid(samples, 2);
  assert.deepEqual(Array.from(pyramid.levels[0].min), [1, 3, -5]);
  assert.deepEqual(Array.from(pyramid.levels[0].max), [2, 4, 0]);
  assert.deepEqual(Array.from(pyramid.levels[1].min), [1, -5]);
  assert.deepEqual(Array.from(pyramid.levels[1].max), [4, 0]);
  assert.deepEqual(Array.from(pyramid.levels[2].min), [-5]);
  assert.deepEqual(Array.from(pyramid.levels[2].max), [4]);
});

test("empty input still yields a usable (all-zero) pyramid", () => {
  const pyramid = buildPeakPyramid(new Float32Array(0), 256);
  assert.equal(pyramid.levels[0].min.length, 1);
  assert.equal(pyramid.levels[0].min[0], 0);
});

test("pickLevel keeps the finest level when zoomed past its bucket size", () => {
  const samples = sine(2, 44100, 5);
  const pyramid = buildPeakPyramid(samples, 256);
  // very zoomed out: 2s spread over 10px -> 0.2s per px -> coarse level
  assert.ok(pickLevel(pyramid, 44100, 0.2) > 0);
  // zoomed in below the level-0 bucket size -> stay at the finest level
  assert.equal(pickLevel(pyramid, 44100, 0.001), 0);
});

test("peakColumns renders exact column count and bounded envelopes", () => {
  const samples = sine(1, 8000, 10);
  const pyramid = buildPeakPyramid(samples, 16);
  const columns = peakColumns(pyramid, 8000, 0, 1, 100);
  assert.equal(columns.min.length, 100);
  assert.equal(columns.max.length, 100);
  for (let index = 0; index < 100; index += 1) {
    assert.ok(columns.min[index] >= -1.0001 && columns.max[index] <= 1.0001);
    assert.ok(columns.min[index] <= columns.max[index]);
  }
});

test("peakColumns over a windowed range covers the signal amplitude", () => {
  const samples = sine(1, 8000, 50);
  const pyramid = buildPeakPyramid(samples, 8);
  const columns = peakColumns(pyramid, 8000, 0.1, 0.2, 20);
  let maxSeen = -Infinity;
  for (const value of columns.max) maxSeen = Math.max(maxSeen, value);
  assert.ok(maxSeen > 0.9, `expected near-full amplitude, got ${maxSeen}`);
});

test("peakColumns outside the audio returns flat zero columns", () => {
  const samples = sine(0.5, 8000, 50);
  const pyramid = buildPeakPyramid(samples, 8);
  const columns = peakColumns(pyramid, 8000, 10, 11, 8);
  for (let index = 0; index < 8; index += 1) {
    assert.equal(columns.min[index], 0);
    assert.equal(columns.max[index], 0);
  }
});

test("drawing a full song is O(width): 10 minutes of audio in one call", () => {
  const rate = 8000;
  const samples = sine(600, rate, 3);
  const pyramid = buildPeakPyramid(samples, 256);
  const started = performance.now();
  const columns = peakColumns(pyramid, rate, 0, 600, 1200);
  const elapsed = performance.now() - started;
  assert.equal(columns.min.length, 1200);
  assert.ok(elapsed < 500, `peakColumns too slow for 600s audio: ${elapsed.toFixed(1)}ms`);
});
