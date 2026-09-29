import test from "node:test";
import assert from "node:assert/strict";

import {
  beginAudioFileLoad,
  captureLoopPoint,
  checkAudioCompatibility,
  clearLoop,
  createSession,
  describeSession,
  loadTrajectoryText,
  playbackRate,
  setAudioError,
  setAudioMetadata,
  setRate,
  setThreshold,
  setWindowLength,
  setWindowStart,
  toggleTrack,
} from "../../viewer/js/session.js";
import { fixtureText } from "./test-helpers.js";

const knownText = fixtureText("trajectory.known-times.json");
const clipText = fixtureText("trajectory.clip.json");
const emptyText = fixtureText("trajectory.empty.json");

test("a fresh session has safe defaults and no trajectory", () => {
  const session = createSession();
  assert.equal(session.trajectory, null);
  assert.equal(session.trajectoryError, null);
  assert.equal(session.threshold, 0.5);
  assert.equal(session.windowLength, 20);
  assert.equal(session.followPlayhead, true);
  assert.equal(session.rate, 1);
  assert.equal(session.loop, null);
  assert.deepEqual(session.hiddenTracks, []);
});

test("loading a valid trajectory installs curves and clears old selection", () => {
  let session = createSession();
  session = loadTrajectoryText(session, knownText, "known.json");
  assert.equal(session.trajectoryError, null);
  assert.equal(session.trajectory.tracks.length, 3);
  assert.equal(session.trajectoryFileName, "known.json");

  // Pretend the user had hidden a track and set a loop on the previous file.
  session = toggleTrack(session, "trk-a");
  session = captureLoopPoint(session, "start", 0.1);
  session = captureLoopPoint(session, "end", 0.4);
  session = loadTrajectoryText(session, clipText, "clip.json");
  assert.equal(session.trajectory.tracks.length, 1);
  assert.deepEqual(session.hiddenTracks, []);
  assert.equal(session.loop, null);
  assert.deepEqual(session.loopDraft, { start: null, end: null });
});

test("a failing load leaves no stale trajectory, loop or playback state", () => {
  let session = createSession();
  session = beginAudioFileLoad(session, "known.wav");
  session = setAudioMetadata(session, 4);
  session = loadTrajectoryText(session, knownText, "known.json");
  session = captureLoopPoint(session, "start", 0.1);
  session = captureLoopPoint(session, "end", 0.4);
  assert.notEqual(session.loop, null);

  session = loadTrajectoryText(session, "{ definitely not json", "broken.json");
  assert.equal(session.trajectory, null);
  assert.equal(session.loop, null);
  assert.equal(session.loopError, null);
  assert.equal(session.trajectoryError.fileName, "broken.json");
  assert.ok(session.trajectoryError.issues.length >= 1);
});

test("switching audio drops the old duration immediately, then installs metadata", () => {
  let session = createSession();
  session = beginAudioFileLoad(session, "song.wav");
  assert.deepEqual(session.audio, {
    fileName: "song.wav",
    duration: null,
    error: null,
  });
  session = setAudioMetadata(session, 30.5);
  assert.equal(session.audio.duration, 30.5);
  // Infinity (streaming) is not a usable duration.
  session = setAudioMetadata(session, Number.POSITIVE_INFINITY);
  assert.equal(session.audio.duration, null);
  session = setAudioError(session, "decode failed");
  assert.equal(session.audio.error, "decode failed");
  assert.equal(session.audio.duration, null);
});

test("duration compatibility distinguishes pending metadata from mismatch", () => {
  const knownSession = loadTrajectoryText(createSession(), knownText, "known.json");
  const trajectory = knownSession.trajectory;
  assert.equal(checkAudioCompatibility(null, null).status, "no-trajectory");
  assert.equal(checkAudioCompatibility(trajectory, null).status, "pending");
  assert.equal(checkAudioCompatibility(trajectory, Number.NaN).status, "pending");
  assert.equal(checkAudioCompatibility(trajectory, 0).status, "pending");
  assert.equal(checkAudioCompatibility(trajectory, 4).status, "ok");
  assert.equal(checkAudioCompatibility(trajectory, 300).status, "ok");

  const clipSession = loadTrajectoryText(createSession(), clipText, "clip.json");
  const mismatch = checkAudioCompatibility(clipSession.trajectory, 4);
  assert.equal(mismatch.status, "mismatch");
  assert.match(mismatch.message, /只有/);

  const emptySession = loadTrajectoryText(createSession(), emptyText, "empty.json");
  assert.equal(checkAudioCompatibility(emptySession.trajectory, 4).status, "empty");
});

test("threshold, rate and window setters enforce their domains", () => {
  let session = createSession();
  session = setThreshold(session, 0.75);
  assert.equal(session.threshold, 0.75);
  session = setThreshold(session, 5);
  assert.equal(session.threshold, 1);
  session = setThreshold(session, -1);
  assert.equal(session.threshold, 0);
  session = setThreshold(session, Number.NaN);
  assert.equal(session.threshold, 0);

  session = setRate(session, 0.5);
  assert.equal(session.rate, 0.5);
  session = setRate(session, 3);
  assert.equal(session.rate, 0.5, "unsupported rate is ignored");
  assert.equal(playbackRate(session), 0.5);

  session = setWindowStart(session, 120, false);
  session = setWindowLength(session, 60);
  assert.equal(session.windowLength, 60);
  assert.equal(session.windowStart, 0, "changing the window resets the page");
  assert.equal(session.followPlayhead, true);
  session = setWindowStart(session, 40, false);
  assert.equal(session.windowStart, 40);
  assert.equal(session.followPlayhead, false);
});

test("loop capture validates against the loaded audio duration", () => {
  let session = createSession();
  session = loadTrajectoryText(session, knownText, "known.json");
  session = beginAudioFileLoad(session, "known.wav");
  session = setAudioMetadata(session, 4);

  session = captureLoopPoint(session, "start", 1.0);
  assert.equal(session.loop, null);
  assert.match(session.loopError, /B 点/);
  session = captureLoopPoint(session, "end", 2.0);
  assert.deepEqual(session.loop, { start: 1.0, end: 2.0 });
  assert.equal(session.loopError, null);

  session = captureLoopPoint(session, "end", 20);
  assert.equal(session.loop, null);
  assert.match(session.loopError, /超出音频时长/);

  session = captureLoopPoint(session, "end", 1.02);
  assert.equal(session.loop, null);
  assert.match(session.loopError, /至少/);

  session = clearLoop(session);
  assert.equal(session.loop, null);
  assert.deepEqual(session.loopDraft, { start: null, end: null });
  assert.equal(session.loopError, null);
});

test("loop capture before audio metadata is pending, not silently accepted", () => {
  let session = loadTrajectoryText(createSession(), knownText, "known.json");
  session = captureLoopPoint(session, "start", 1);
  session = captureLoopPoint(session, "end", 2);
  assert.equal(session.loop, null);
  assert.match(session.loopError, /时长尚未就绪/);
});

test("track visibility toggling keeps other tracks untouched", () => {
  let session = loadTrajectoryText(createSession(), knownText, "known.json");
  session = toggleTrack(session, "trk-a");
  session = toggleTrack(session, "trk-c");
  assert.deepEqual(session.hiddenTracks.sort(), ["trk-a", "trk-c"]);
  session = toggleTrack(session, "trk-a");
  assert.deepEqual(session.hiddenTracks, ["trk-c"]);
});

test("describeSession exposes a plain-data debug view", () => {
  let session = loadTrajectoryText(createSession(), knownText, "known.json");
  session = beginAudioFileLoad(session, "known.wav");
  session = setAudioMetadata(session, 4);
  const described = describeSession(session);
  assert.equal(described.trackCount, 3);
  assert.equal(described.provenance.dataKind, "model");
  assert.equal(described.audio.duration, 4);
  assert.equal(described.loop, null);
  assert.deepEqual(described.trackIds, ["trk-a", "trk-b", "trk-c"]);
});
