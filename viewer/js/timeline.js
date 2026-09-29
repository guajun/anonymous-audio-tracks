// Time-axis math for the viewer.
//
// Protocol 0.1.0 stores every trajectory center time on the *original track*
// absolute axis, while `HTMLMediaElement.currentTime` is relative to the start
// of the loaded audio file.  For an analyzed clip whose first sample is
// `track_start_seconds` on the original track, the conversion is
//
//     local (player seconds) = absolute (protocol seconds) - track_start_seconds
//
// Never feed an absolute protocol time directly into `audio.currentTime`.
// Every function here is pure so the known-time fixtures can be tested with
// Node's built-in test runner.

import { TRAJECTORY_TIME_TOLERANCE_SECONDS } from "./protocol.js";

export const MIN_LOOP_SECONDS = 0.05;
const MERGE_EPSILON_SECONDS = 1e-9;

export function clamp(value, minimum, maximum) {
  if (!Number.isFinite(value)) {
    return minimum;
  }
  if (value < minimum) {
    return minimum;
  }
  if (value > maximum) {
    return maximum;
  }
  return value;
}

export function localToAbsoluteTime(localSeconds, trackStartSeconds) {
  return localSeconds + trackStartSeconds;
}

export function absoluteToLocalTime(absoluteSeconds, trackStartSeconds) {
  return absoluteSeconds - trackStartSeconds;
}

/** Position after playing `elapsedSeconds` of wall clock at `rate`. */
export function stepAdvance(currentSeconds, elapsedSeconds, rate) {
  return currentSeconds + elapsedSeconds * rate;
}

/**
 * Validate a loop selection in the player-local timeline.
 *
 * @returns {{ok: true, start: number, end: number} | {ok: false, reason: string, message: string}}
 */
export function validateLoopRange(start, end, duration) {
  if (!Number.isFinite(duration) || duration <= 0) {
    return {
      ok: false,
      reason: "duration-unavailable",
      message: "音频时长尚未就绪，无法校验循环端点。",
    };
  }
  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    return {
      ok: false,
      reason: "non-finite",
      message: "循环端点必须是有限秒数。",
    };
  }
  if (start < 0 || end < 0) {
    return {
      ok: false,
      reason: "negative",
      message: `循环端点不能为负：A=${start}s，B=${end}s。`,
    };
  }
  if (end > duration + TRAJECTORY_TIME_TOLERANCE_SECONDS) {
    return {
      ok: false,
      reason: "out-of-range",
      message: `循环端点 B=${end}s 超出音频时长 ${duration}s。`,
    };
  }
  if (end - start < MIN_LOOP_SECONDS) {
    return {
      ok: false,
      reason: "too-short",
      message: `循环区间至少需要 ${MIN_LOOP_SECONDS}s（当前 A=${start}s，B=${end}s）。`,
    };
  }
  return { ok: true, start, end };
}

/**
 * Given the current player time, return the corrected time when the loop end
 * has been reached; otherwise return the input unchanged.  A disabled or
 * invalid loop is a no-op.
 */
export function applyLoopCorrection(currentSeconds, loop) {
  if (
    !loop ||
    !Number.isFinite(loop.start) ||
    !Number.isFinite(loop.end) ||
    loop.end <= loop.start
  ) {
    return currentSeconds;
  }
  if (currentSeconds < loop.end) {
    return currentSeconds;
  }
  const span = loop.end - loop.start;
  const offset = ((currentSeconds - loop.start) % span + span) % span;
  return loop.start + offset;
}

/**
 * Linear interpolation of one track's activity at an absolute original-track
 * time.  Returns `null` outside the track's center-time range: the protocol
 * only claims center activity at sampled centers, so nothing is invented
 * beyond the first/last point.
 */
export function activityAtAbsoluteTime(track, absoluteSeconds) {
  const times = track.centerTimes;
  const values = track.activity;
  if (times.length === 0) {
    return null;
  }
  const first = times[0];
  const last = times[times.length - 1];
  if (absoluteSeconds < first || absoluteSeconds > last) {
    return null;
  }
  if (absoluteSeconds === first) {
    return values[0];
  }
  if (absoluteSeconds === last) {
    return values[values.length - 1];
  }
  let low = 0;
  let high = times.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if (times[middle] <= absoluteSeconds) {
      low = middle;
    } else {
      high = middle;
    }
  }
  if (absoluteSeconds === times[low]) {
    return values[low];
  }
  if (absoluteSeconds === times[high]) {
    return values[high];
  }
  const fraction = (absoluteSeconds - times[low]) / (times[high] - times[low]);
  return values[low] + fraction * (values[high] - values[low]);
}

/**
 * Nearest sampled center point at or around an absolute time, for the raw
 * readout (probability + optional slot provenance).  Returns `null` outside
 * the track range.
 */
export function nearestSample(track, absoluteSeconds, trackStartSeconds) {
  const times = track.centerTimes;
  if (times.length === 0) {
    return null;
  }
  if (absoluteSeconds < times[0] || absoluteSeconds > times[times.length - 1]) {
    return null;
  }
  let bestIndex = 0;
  let bestDistance = Math.abs(absoluteSeconds - times[0]);
  for (let index = 1; index < times.length; index += 1) {
    const distance = Math.abs(absoluteSeconds - times[index]);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return {
    index: bestIndex,
    absoluteSeconds: times[bestIndex],
    localSeconds: times[bestIndex] - trackStartSeconds,
    activity: track.activity[bestIndex],
    slotIndex: track.slotIndices ? track.slotIndices[bestIndex] : null,
  };
}

function mergeIntervals(segments) {
  if (segments.length === 0) {
    return [];
  }
  const sorted = segments
    .slice()
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const merged = [[sorted[0][0], sorted[0][1]]];
  for (let index = 1; index < sorted.length; index += 1) {
    const [start, end] = sorted[index];
    const previous = merged[merged.length - 1];
    if (start <= previous[1] + MERGE_EPSILON_SECONDS) {
      if (end > previous[1]) {
        previous[1] = end;
      }
    } else {
      merged.push([start, end]);
    }
  }
  return merged;
}

/**
 * Local-time segments whose interpolated activity is `>= threshold`.
 * Threshold crossings are interpolated between sampled points.  These are
 * activity intervals, not note-on events.
 */
export function thresholdIntervals(track, threshold, trackStartSeconds) {
  const times = track.centerTimes;
  const values = track.activity;
  const segments = [];
  const toLocal = (time) => time - trackStartSeconds;
  for (let index = 0; index < times.length; index += 1) {
    const localTime = toLocal(times[index]);
    const value = values[index];
    if (value >= threshold) {
      segments.push([localTime, localTime]);
    }
    if (index + 1 >= times.length) {
      continue;
    }
    const nextTime = toLocal(times[index + 1]);
    const nextValue = values[index + 1];
    if (value >= threshold && nextValue >= threshold) {
      segments.push([localTime, nextTime]);
    } else if (value < threshold && nextValue >= threshold) {
      const crossing =
        localTime +
        ((threshold - value) / (nextValue - value)) * (nextTime - localTime);
      segments.push([crossing, nextTime]);
    } else if (value >= threshold && nextValue < threshold) {
      const crossing =
        localTime +
        ((threshold - value) / (nextValue - value)) * (nextTime - localTime);
      segments.push([localTime, crossing]);
    }
  }
  return mergeIntervals(segments);
}

export function totalActiveSeconds(intervals) {
  return intervals.reduce((total, [start, end]) => total + (end - start), 0);
}

/** Clip intervals to a local window; drops intervals fully outside it. */
export function clampIntervals(intervals, windowStart, windowEnd) {
  const result = [];
  for (const [start, end] of intervals) {
    const clippedStart = Math.max(start, windowStart);
    const clippedEnd = Math.min(end, windowEnd);
    if (clippedEnd >= clippedStart - MERGE_EPSILON_SECONDS) {
      result.push([clippedStart, Math.max(clippedEnd, clippedStart)]);
    }
  }
  return result;
}

/**
 * Local window bounds clamped to `[0, duration]`.  A window at most
 * `windowLength` seconds long; `duration <= 0` yields an empty window.
 */
export function windowBounds(start, duration, windowLength) {
  const safeDuration =
    Number.isFinite(duration) && duration > 0 ? duration : 0;
  const safeLength =
    Number.isFinite(windowLength) && windowLength > 0 ? windowLength : 20;
  const maxStart = Math.max(0, safeDuration - safeLength);
  const clampedStart = clamp(Number.isFinite(start) ? start : 0, 0, maxStart);
  return {
    start: clampedStart,
    end: Math.min(clampedStart + safeLength, safeDuration),
  };
}

/** Page start (multiple of `windowLength`) that contains `anchorSeconds`. */
export function pageStartFor(anchorSeconds, duration, windowLength) {
  const safeDuration =
    Number.isFinite(duration) && duration > 0 ? duration : 0;
  const safeLength =
    Number.isFinite(windowLength) && windowLength > 0 ? windowLength : 20;
  if (safeDuration <= 0) {
    return 0;
  }
  const maxStart = Math.max(0, safeDuration - safeLength);
  if (!Number.isFinite(anchorSeconds)) {
    return 0;
  }
  return clamp(
    Math.floor(anchorSeconds / safeLength) * safeLength,
    0,
    maxStart,
  );
}

/** Move a window by `steps` window lengths, clamped to the timeline. */
export function shiftWindow(start, steps, duration, windowLength) {
  const safeLength =
    Number.isFinite(windowLength) && windowLength > 0 ? windowLength : 20;
  const bounds = windowBounds(start, duration, windowLength);
  return windowBounds(
    bounds.start + steps * safeLength,
    duration,
    windowLength,
  ).start;
}

/** `mm:ss.mmm` formatting used by the transport readouts. */
export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "--:--.---";
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  const wholeSeconds = Math.floor(remainder);
  const milliseconds = Math.round((remainder - wholeSeconds) * 1000);
  const carry = milliseconds === 1000;
  const secs = carry ? wholeSeconds + 1 : wholeSeconds;
  const millis = carry ? 0 : milliseconds;
  const totalMinutes = minutes + (secs >= 60 ? 1 : 0);
  const displaySeconds = secs >= 60 ? secs - 60 : secs;
  return `${String(totalMinutes).padStart(2, "0")}:${String(displaySeconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}
