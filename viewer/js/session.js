// Pure UI state for the viewer.
//
// Keeping state transitions out of app.js makes the "switch file / load error
// must not leave a stale trajectory or loop" behaviour testable with Node's
// built-in test runner.

import {
  parseTrajectoryText,
  TrajectoryValidationError,
} from "./protocol.js";
import { clamp, validateLoopRange } from "./timeline.js";

export const RATES = [0.25, 0.5, 0.75, 1, 1.5, 2];
export const WINDOW_LENGTHS = [5, 10, 20, 30, 60, 120];
export const DEFAULT_THRESHOLD = 0.5;
export const DEFAULT_WINDOW_LENGTH = 20;
// Loaded audio length must match the declared analyzed-audio length in both
// directions within this tolerance (e.g. codec padding on short files).
export const COMPATIBILITY_TOLERANCE_SECONDS = 0.01;

export function createSession() {
  return {
    trajectory: null,
    trajectoryFileName: null,
    trajectoryPending: null,
    trajectoryError: null,
    audio: { fileName: null, duration: null, error: null },
    threshold: DEFAULT_THRESHOLD,
    windowLength: DEFAULT_WINDOW_LENGTH,
    windowStart: 0,
    followPlayhead: true,
    rate: 1,
    loop: null,
    loopDraft: { start: null, end: null },
    loopError: null,
    hiddenTracks: [],
  };
}

function clearedTrajectoryState(session, fileName) {
  return {
    ...session,
    trajectory: null,
    trajectoryFileName: fileName,
    trajectoryPending: null,
    windowStart: 0,
    hiddenTracks: [],
    loop: null,
    loopDraft: { start: null, end: null },
    loopError: null,
  };
}

/**
 * Begin reading a newly selected trajectory file.
 *
 * The previous document is dropped immediately: while the new text is being
 * read, no stale curves may be treated as the current synchronization data.
 * app.js pairs this with a generation token so that only the newest read may
 * call {@link loadTrajectoryText} / {@link failTrajectoryLoad}.
 */
export function beginTrajectoryFileLoad(session, fileName) {
  return {
    ...clearedTrajectoryState(session, fileName),
    trajectoryPending: fileName,
    trajectoryError: null,
  };
}

/**
 * Parse and install a trajectory document.  On success the previous document,
 * loop and hidden-track selection are replaced; on failure the trajectory is
 * cleared entirely so no stale curves stay on screen.
 */
export function loadTrajectoryText(session, text, fileName) {
  let trajectory;
  try {
    trajectory = parseTrajectoryText(text);
  } catch (error) {
    const issues =
      error instanceof TrajectoryValidationError
        ? error.issues
        : [{ path: "<file>", message: String((error && error.message) || error) }];
    return {
      ...clearedTrajectoryState(session, fileName),
      trajectoryError: {
        fileName,
        message: error instanceof Error ? error.message : String(error),
        issues,
      },
    };
  }
  return {
    ...clearedTrajectoryState(session, fileName),
    trajectory,
    trajectoryError: null,
  };
}

/** Record an I/O failure for the newest trajectory read (generation-guarded). */
export function failTrajectoryLoad(session, fileName, message, issues = []) {
  return {
    ...clearedTrajectoryState(session, fileName),
    trajectoryError: {
      fileName,
      message,
      issues: issues.length > 0 ? issues : [{ path: "<file>", message }],
    },
  };
}

/** Start loading a new audio file: old duration/error are dropped immediately. */
export function beginAudioFileLoad(session, fileName) {
  return {
    ...session,
    audio: { fileName, duration: null, error: null },
    loop: null,
    loopDraft: { start: null, end: null },
    loopError: null,
    windowStart: 0,
  };
}

/** Async `loadedmetadata`: install the duration once the element reports it. */
export function setAudioMetadata(session, duration) {
  const usable = Number.isFinite(duration) && duration > 0;
  return {
    ...session,
    audio: {
      fileName: session.audio.fileName,
      duration: usable ? duration : null,
      error: session.audio.error,
    },
  };
}

export function setAudioError(session, message) {
  return {
    ...session,
    audio: {
      fileName: session.audio.fileName,
      duration: null,
      error: message,
    },
  };
}

/**
 * Compare the declared analyzed-audio length with the loaded audio length.
 *
 * Protocol 0.1.0 documents the analyzed clip (`audio.duration_seconds`), and
 * this viewer requires that the loaded file *is* that same analyzed clip, so
 * the two lengths must agree within {@link COMPATIBILITY_TOLERANCE_SECONDS} in
 * both directions.  `track_start_seconds` only shifts absolute-time readouts;
 * it is never part of the length comparison.  `pending` is the async-metadata
 * case and must not be reported as a duration error.
 */
export function checkAudioCompatibility(trajectory, audioDuration) {
  if (!trajectory) {
    return { status: "no-trajectory", message: null };
  }
  const span = trajectory.audio;
  if (!Number.isFinite(audioDuration) || audioDuration <= 0) {
    return {
      status: "pending",
      message: "音频 metadata 尚未就绪；加载完成后会自动校验音频长度。",
    };
  }
  if (span.durationSeconds === 0) {
    return {
      status: "empty",
      message: "轨迹声明了空音频范围（duration=0，无轨迹）；该边界不应崩溃。",
    };
  }
  const difference = Math.abs(audioDuration - span.durationSeconds);
  if (difference > COMPATIBILITY_TOLERANCE_SECONDS) {
    const direction = audioDuration < span.durationSeconds ? "短于" : "长于";
    return {
      status: "mismatch",
      message:
        `轨迹声明被分析音频时长 ${span.durationSeconds}s，已加载音频 ${audioDuration.toFixed(3)}s ` +
        `（${direction} ${difference.toFixed(3)}s，超出 ±${COMPATIBILITY_TOLERANCE_SECONDS}s 容差）；` +
        `本页要求载入同一分析片段，不自动接受整歌或更长的音频。`,
    };
  }
  return {
    status: "ok",
    message:
      `已加载音频 ${audioDuration.toFixed(3)}s 与轨迹声明的被分析音频时长 ` +
      `${span.durationSeconds}s 相符（容差 ±${COMPATIBILITY_TOLERANCE_SECONDS}s）；` +
      `原曲绝对起点为 ${span.trackStartSeconds}s。`,
  };
}

/**
 * Whether synchronized playback/loop/seek may be enabled for this pair.
 * `pending` is not allowed because the length cannot be verified yet;
 * `mismatch` is explicitly forbidden by review (issue #10 fix).
 */
export function syncPlaybackAllowed(trajectory, audioDuration) {
  const status = checkAudioCompatibility(trajectory, audioDuration).status;
  return status === "ok" || status === "empty";
}

export function setThreshold(session, value) {
  if (!Number.isFinite(value)) {
    return session;
  }
  return { ...session, threshold: clamp(value, 0, 1) };
}

export function setRate(session, rate) {
  if (!RATES.includes(rate)) {
    return session;
  }
  return { ...session, rate };
}

export function setWindowLength(session, windowLength) {
  if (!WINDOW_LENGTHS.includes(windowLength)) {
    return session;
  }
  return { ...session, windowLength, windowStart: 0, followPlayhead: true };
}

export function setWindowStart(session, windowStart, followPlayhead = false) {
  return {
    ...session,
    windowStart: Number.isFinite(windowStart) ? windowStart : 0,
    followPlayhead,
  };
}

export function toggleTrack(session, trackId) {
  const hidden = new Set(session.hiddenTracks);
  if (hidden.has(trackId)) {
    hidden.delete(trackId);
  } else {
    hidden.add(trackId);
  }
  return { ...session, hiddenTracks: [...hidden] };
}

/** Capture the current player-local time as loop endpoint A or B. */
export function captureLoopPoint(session, which, currentSeconds) {
  const loopDraft = { ...session.loopDraft, [which]: currentSeconds };
  return finalizeLoopDraft({ ...session, loopDraft });
}

/** Try to turn the draft endpoints into an active loop; keep the error text. */
export function finalizeLoopDraft(session) {
  const { start, end } = session.loopDraft;
  if (start === null && end === null) {
    return { ...session, loop: null, loopError: null };
  }
  if (start === null || end === null) {
    return {
      ...session,
      loop: null,
      loopError:
        start === null
          ? "已设置 B 点，还需要设置 A 点。"
          : "已设置 A 点，还需要设置 B 点。",
    };
  }
  const result = validateLoopRange(start, end, session.audio.duration);
  if (!result.ok) {
    return { ...session, loop: null, loopError: result.message };
  }
  return { ...session, loop: { start: result.start, end: result.end }, loopError: null };
}

export function clearLoop(session) {
  return {
    ...session,
    loop: null,
    loopDraft: { start: null, end: null },
    loopError: null,
  };
}

/** Playback rate the audio element should adopt. */
export function playbackRate(session) {
  return RATES.includes(session.rate) ? session.rate : 1;
}

/** Small plain-data view of the session for the browser check hook. */
export function describeSession(session) {
  return {
    trajectoryFileName: session.trajectoryFileName,
    trajectoryPending: session.trajectoryPending,
    trajectoryLoaded: session.trajectory !== null,
    trajectoryError: session.trajectoryError
      ? {
          fileName: session.trajectoryError.fileName,
          message: session.trajectoryError.message,
          issues: session.trajectoryError.issues,
        }
      : null,
    sampleId: session.trajectory ? session.trajectory.sampleId : null,
    provenance: session.trajectory ? session.trajectory.provenance : null,
    audioSpan: session.trajectory ? session.trajectory.audio : null,
    trackCount: session.trajectory ? session.trajectory.tracks.length : 0,
    trackIds: session.trajectory
      ? session.trajectory.tracks.map((track) => track.trackId)
      : [],
    audio: session.audio,
    threshold: session.threshold,
    windowLength: session.windowLength,
    windowStart: session.windowStart,
    followPlayhead: session.followPlayhead,
    rate: session.rate,
    loop: session.loop,
    loopDraft: session.loopDraft,
    loopError: session.loopError,
    hiddenTracks: session.hiddenTracks,
  };
}
