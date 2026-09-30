// View model over a validated `agentic-audio-tracks/v1` document.
//
// Frozen contract invariants preserved here:
//   - one row per instrument entry; identical labels are NEVER merged
//     (label is human text, `id` is the stable identity),
//   - event times stay in audio seconds; changing BPM must never touch them
//     (the beat grid is a purely derived overlay),
//   - events without `duration_seconds` only produce an onset tick; no
//     synthetic note length is invented.

import { lowerBound, upperBound } from "./timeline.js";

export const BPM_MAX = 1000; // mirrors the schema: bpm in (0, 1000]
export const BPM_MIN_EXCLUSIVE = 0;

/**
 * Build typed-array rows from a validated document.
 * @param {object} doc parsed + validated document
 */
export function buildModel(doc) {
  const rows = (Array.isArray(doc.instruments) ? doc.instruments : []).map((instrument, index) => {
    const events = Array.isArray(instrument.events) ? instrument.events : [];
    const count = events.length;
    const onset = new Float64Array(count);
    const duration = new Float64Array(count); // NaN = not provided
    const confidence = new Float32Array(count); // NaN = not provided
    const pitchMidi = new Int16Array(count); // -1 = not provided
    const ids = new Array(count);
    let maxDurationSeconds = 0;
    let withDuration = 0;
    events.forEach((event, i) => {
      onset[i] = event.onset_seconds;
      if (typeof event.duration_seconds === "number") {
        duration[i] = event.duration_seconds;
        withDuration += 1;
        if (event.duration_seconds > maxDurationSeconds) maxDurationSeconds = event.duration_seconds;
      } else {
        duration[i] = NaN;
      }
      confidence[i] = typeof event.confidence === "number" ? event.confidence : NaN;
      pitchMidi[i] = event.pitch && typeof event.pitch.midi === "number" ? event.pitch.midi : -1;
      ids[i] = String(event.id);
    });
    return {
      index,
      id: String(instrument.id),
      label: String(instrument.label),
      description: String(instrument.description),
      source: String(instrument.source),
      confidence: typeof instrument.confidence === "number" ? instrument.confidence : NaN,
      stem: instrument.stem ? { filename: String(instrument.stem.filename), sha256: String(instrument.stem.sha256) } : null,
      onset,
      duration,
      confidenceArray: confidence,
      pitchMidi,
      ids,
      count,
      withDuration,
      maxDurationSeconds,
    };
  });

  return {
    schemaVersion: doc.schema_version,
    audio: {
      filename: String(doc.audio.filename),
      sha256: String(doc.audio.sha256),
      durationSeconds: doc.audio.duration_seconds,
      sampleRate: doc.audio.sample_rate,
    },
    tempo: {
      bpm: doc.tempo.bpm === null ? null : doc.tempo.bpm,
      source: String(doc.tempo.source),
      confidence: doc.tempo.confidence,
      beatOriginSeconds: typeof doc.tempo.beat_origin_seconds === "number" ? doc.tempo.beat_origin_seconds : 0,
      hasBeatOrigin: typeof doc.tempo.beat_origin_seconds === "number",
    },
    rows,
    limitations: Array.isArray(doc.limitations) ? doc.limitations.map(String) : [],
    provenance: doc.provenance,
    totalEvents: rows.reduce((sum, row) => sum + row.count, 0),
    maxEventDurationSeconds: rows.reduce((max, row) => Math.max(max, row.maxDurationSeconds), 0),
  };
}

/**
 * BPM editor state. The JSON value is only a prefill: the user may type a
 * manual value, and `tempo.bpm === null` is surfaced as "unknown".
 * Changing the value never re-times events (grid-only).
 */
export function createBpmState(model) {
  const jsonBpm = model && model.tempo && typeof model.tempo.bpm === "number" ? model.tempo.bpm : null;
  return {
    jsonBpm,
    value: jsonBpm,
    origin: jsonBpm === null ? "unknown" : "json",
    unknown: jsonBpm === null,
    valid: jsonBpm !== null,
    message:
      jsonBpm === null
        ? "BPM 未知（JSON tempo.bpm=null）：可手动输入有限正数（0 < bpm ≤ 1000），只影响节拍网格，不重定时事件。"
        : `BPM 来自 JSON（tempo.bpm=${jsonBpm}），可手动调整；只影响节拍网格，不重定时事件。`,
  };
}

/** Validate a human-entered BPM. Returns {ok, value, message}. */
export function validateBpm(input) {
  const value = typeof input === "string" ? Number(input.trim()) : input;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return { ok: false, value: null, message: "BPM 必须是有限数字（拒绝 NaN/Infinity/非数字）。" };
  }
  if (value <= BPM_MIN_EXCLUSIVE) {
    return { ok: false, value: null, message: "BPM 必须为正数（> 0）。" };
  }
  if (value > BPM_MAX) {
    return { ok: false, value: null, message: `BPM 必须 <= ${BPM_MAX}（schema 上限）。` };
  }
  return { ok: true, value, message: `节拍网格已更新为 ${value} BPM；onset 秒时间不变。` };
}

/**
 * Apply a BPM change to the state. On invalid input the grid state is kept
 * untouched (the previous grid keeps rendering) and an error message is set.
 */
export function applyBpm(state, input) {
  const result = validateBpm(input);
  if (!result.ok) {
    return { state: { ...state, lastError: result.message, lastInputRejected: true }, changed: false };
  }
  return {
    state: {
      ...state,
      value: result.value,
      origin: "manual",
      unknown: false,
      valid: true,
      lastError: null,
      lastInputRejected: false,
      message: result.message,
    },
    changed: true,
  };
}

/** Beat grid overlay: purely derived from BPM + origin; never mutates events. */
export function beatGrid(bpmState, beatOriginSeconds, t0, t1, maxLines = 512) {
  const bpm = bpmState && bpmState.valid ? bpmState.value : null;
  if (!(typeof bpm === "number" && Number.isFinite(bpm) && bpm > 0) || !(t1 > t0)) {
    return { beats: [], interval: null, origin: 0 };
  }
  const interval = 60 / bpm;
  const origin = Number.isFinite(beatOriginSeconds) ? beatOriginSeconds : 0;
  const firstIndex = Math.ceil((t0 - origin) / interval);
  const lastIndex = Math.floor((t1 - origin) / interval);
  const total = lastIndex - firstIndex + 1;
  if (total <= 0) return { beats: [], interval, origin };
  if (total > maxLines) {
    // Density guard: draw only every n-th beat instead of a solid black wall.
    const step = Math.ceil(total / maxLines);
    const beats = [];
    for (let index = firstIndex; index <= lastIndex; index += step) {
      beats.push(origin + index * interval);
    }
    return { beats, interval: interval * step, origin, skipped: step };
  }
  const beats = [];
  for (let index = firstIndex; index <= lastIndex; index += 1) {
    beats.push(origin + index * interval);
  }
  return { beats, interval, origin };
}

/** Binary-search the visible event slice of one row (viewport culling). */
export function rowVisibleSlice(row, t0, t1) {
  const from = lowerBound(row.onset, t0 - row.maxDurationSeconds);
  const to = upperBound(row.onset, t1);
  return { from, to, count: Math.max(0, to - from) };
}
