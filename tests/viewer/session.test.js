import test from "node:test";
import assert from "node:assert/strict";

import {
  beginAudioFileLoad,
  beginTrajectoryFileLoad,
  captureLoopPoint,
  checkAudioCompatibility,
  clearLoop,
  COMPATIBILITY_TOLERANCE_SECONDS,
  createSession,
  describeSession,
  failTrajectoryLoad,
  loadTrajectoryText,
  playbackRate,
  setAudioError,
  setAudioMetadata,
  setRate,
  setThreshold,
  setWindowLength,
  setWindowStart,
  syncPlaybackAllowed,
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

test("duration compatibility compares the declared clip length in both directions", () => {
  const knownSession = loadTrajectoryText(createSession(), knownText, "known.json");
  const trajectory = knownSession.trajectory;
  assert.equal(checkAudioCompatibility(null, null).status, "no-trajectory");
  assert.equal(checkAudioCompatibility(trajectory, null).status, "pending");
  assert.equal(checkAudioCompatibility(trajectory, Number.NaN).status, "pending");
  assert.equal(checkAudioCompatibility(trajectory, 0).status, "pending");
  assert.equal(checkAudioCompatibility(trajectory, 4).status, "ok");
  assert.equal(checkAudioCompatibility(trajectory, 4.005).status, "ok");
  assert.equal(checkAudioCompatibility(trajectory, 3.995).status, "ok");
  assert.equal(checkAudioCompatibility(trajectory, 4.02).status, "mismatch");
  assert.equal(checkAudioCompatibility(trajectory, 3.98).status, "mismatch");
  // The old bug: a whole 600s song passed because it merely covered the span.
  const tooLong = checkAudioCompatibility(trajectory, 600);
  assert.equal(tooLong.status, "mismatch");
  assert.match(tooLong.message, /长于/);

  const clipSession = loadTrajectoryText(createSession(), clipText, "clip.json");
  // origin 12s / duration 0.5s requires a 0.5s clip, not "at least 12.5s".
  const clipOk = checkAudioCompatibility(clipSession.trajectory, 0.5);
  assert.equal(clipOk.status, "ok");
  assert.equal(syncPlaybackAllowed(clipSession.trajectory, 0.5), true);
  assert.equal(checkAudioCompatibility(clipSession.trajectory, 0.48).status, "mismatch");
  assert.equal(checkAudioCompatibility(clipSession.trajectory, 0.52).status, "mismatch");
  assert.equal(checkAudioCompatibility(clipSession.trajectory, 4).status, "mismatch");
  assert.equal(syncPlaybackAllowed(clipSession.trajectory, 4), false);
  assert.equal(syncPlaybackAllowed(clipSession.trajectory, null), false);
  assert.equal(syncPlaybackAllowed(null, 4), false);

  // Tolerance is symmetric: exactly at the boundary is still accepted.
  assert.equal(
    checkAudioCompatibility(trajectory, 4 + COMPATIBILITY_TOLERANCE_SECONDS).status,
    "ok",
  );

  const emptySession = loadTrajectoryText(createSession(), emptyText, "empty.json");
  assert.equal(checkAudioCompatibility(emptySession.trajectory, 4).status, "empty");
  assert.equal(syncPlaybackAllowed(emptySession.trajectory, 4), true);
});

test("a new trajectory selection immediately drops the old synchronized data", () => {
  let session = loadTrajectoryText(createSession(), knownText, "known.json");
  session = beginTrajectoryFileLoad(session, "next.json");
  assert.equal(session.trajectory, null);
  assert.equal(session.trajectoryPending, "next.json");
  assert.equal(session.trajectoryFileName, "next.json");
  assert.equal(session.trajectoryError, null);
  assert.equal(session.loop, null);

  // The newest read commits and clears pending.
  session = loadTrajectoryText(session, clipText, "next.json");
  assert.equal(session.trajectory.tracks[0].trackId, "trk-clip-0001");
  assert.equal(session.trajectoryPending, null);

  // A newer pending state also clears a previous error.
  session = beginTrajectoryFileLoad(session, "third.json");
  assert.equal(session.trajectoryPending, "third.json");
  assert.equal(session.trajectoryError, null);
  session = failTrajectoryLoad(session, "third.json", "读取轨迹文件失败：boom");
  assert.equal(session.trajectoryPending, null);
  assert.equal(session.trajectory, null);
  assert.equal(session.trajectoryError.fileName, "third.json");
  assert.match(session.trajectoryError.message, /boom/);
  assert.equal(session.trajectoryError.issues.length, 1);
});

test("pending trajectory reads are observable through describeSession", () => {
  let session = beginTrajectoryFileLoad(createSession(), "pending.json");
  const described = describeSession(session);
  assert.equal(described.trajectoryPending, "pending.json");
  assert.equal(described.trajectoryLoaded, false);
  assert.equal(described.trackCount, 0);
  session = loadTrajectoryText(session, knownText, "pending.json");
  assert.equal(describeSession(session).trajectoryPending, null);
  assert.equal(describeSession(session).trajectoryLoaded, true);
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
