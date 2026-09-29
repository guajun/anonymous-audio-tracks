// Snapshot of everything the UI needs for one render pass.
//
// The clock is a single `currentTime` in player-local seconds, read from the
// shared audio element.  Everything else (absolute original-track time,
// per-track probability, threshold intervals, window) is derived here so the
// "all tracks share one clock" rule is testable without a DOM.

import {
  activityAtAbsoluteTime,
  clampIntervals,
  localToAbsoluteTime,
  nearestSample,
  pageStartFor,
  thresholdIntervals,
  totalActiveSeconds,
  windowBounds,
} from "./timeline.js";

const SNAPSHOT_TOLERANCE_SECONDS = 1e-9;

/**
 * @param {object} options
 * @param {object|null} options.trajectory normalized trajectory or null
 * @param {number|null} options.audioDuration loaded audio duration, if known
 * @param {number} options.currentTime player-local seconds (audio.currentTime)
 * @param {number} options.threshold activity threshold in [0, 1]
 * @param {number} options.windowLength local window length in seconds
 * @param {number} options.windowStart local window start (ignored when
 *   `followPlayhead` is true)
 * @param {boolean} options.followPlayhead
 * @param {string[]} options.hiddenTracks track ids switched off
 */
export function buildSnapshot(options) {
  const {
    trajectory,
    audioDuration = null,
    currentTime = 0,
    threshold = 0.5,
    windowLength = 20,
    windowStart = 0,
    followPlayhead = true,
    hiddenTracks = [],
  } = options;

  const spanDuration = trajectory ? trajectory.audio.durationSeconds : 0;
  const hasAudioDuration =
    Number.isFinite(audioDuration) && audioDuration > 0;
  const duration = hasAudioDuration ? audioDuration : spanDuration;
  const durationSource = hasAudioDuration
    ? "audio"
    : trajectory
      ? "trajectory"
      : "none";
  const localTime = Math.min(Math.max(currentTime, 0), duration);
  const trackStart = trajectory ? trajectory.audio.trackStartSeconds : 0;
  const absoluteTime = localToAbsoluteTime(localTime, trackStart);
  const hidden = new Set(hiddenTracks);

  const resolvedWindowStart = followPlayhead
    ? pageStartFor(localTime, duration, windowLength)
    : windowStart;
  const window = windowBounds(resolvedWindowStart, duration, windowLength);

  const tracks = trajectory
    ? trajectory.tracks.map((track, index) => {
        const value = activityAtAbsoluteTime(track, absoluteTime);
        const intervals = thresholdIntervals(track, threshold, trackStart);
        return {
          trackId: track.trackId,
          index,
          visible: !hidden.has(track.trackId),
          value,
          active: value !== null && value >= threshold,
          intervals,
          windowIntervals: clampIntervals(intervals, window.start, window.end),
          activeSeconds: totalActiveSeconds(intervals),
          pointCount: track.centerTimes.length,
          hasConfidence: track.confidence !== null,
          hasSlotIndices: track.slotIndices !== null,
          nearest: nearestSample(track, absoluteTime, trackStart),
          notes: track.notes,
          // Raw arrays are exposed (by reference, not copied) so the canvas
          // renderer does not need a second lookup path.
          centerTimes: track.centerTimes,
          activity: track.activity,
        };
      })
    : [];

  return {
    trajectoryLoaded: trajectory !== null,
    duration,
    durationSource,
    localTime,
    absoluteTime,
    trackStart,
    threshold,
    window,
    followPlayhead,
    tracks,
    visibleTrackCount: tracks.filter((track) => track.visible).length,
  };
}

/** True when two snapshot reads describe the same clock position. */
export function sameInstant(left, right) {
  return (
    Math.abs(left.localTime - right.localTime) <=
      SNAPSHOT_TOLERANCE_SECONDS &&
    left.absoluteTime === right.absoluteTime
  );
}
