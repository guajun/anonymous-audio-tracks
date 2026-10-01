// Time-window math shared by the canvas renderer and the unit tests.
//
// The view window is [start, end] in **audio seconds** (the frozen protocol's
// time truth). Zooming keeps the time under the pointer fixed (anchor), the
// window is always clamped to [0, totalSeconds] and to a minimum duration.

export const MIN_WINDOW_SECONDS = 0.002;
export const MAX_WINDOW_SECONDS = 1e9;

function finitePositive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * Clamp and normalise a window.
 * @returns {{start: number, end: number}}
 */
export function clampWindow(start, end, totalSeconds, minWindow = MIN_WINDOW_SECONDS) {
  const total = finitePositive(totalSeconds) ? totalSeconds : MAX_WINDOW_SECONDS;
  const minDur = Math.min(minWindow, total);
  let lo = Number.isFinite(start) ? start : 0;
  let hi = Number.isFinite(end) ? end : total;
  if (hi < lo) [lo, hi] = [hi, lo];
  let duration = hi - lo;
  if (!Number.isFinite(duration) || duration <= 0) duration = minDur;
  duration = Math.min(Math.max(duration, minDur), total);
  lo = Math.min(Math.max(lo, 0), total - duration);
  return { start: lo, end: lo + duration };
}

export function fitWindow(totalSeconds) {
  return { start: 0, end: finitePositive(totalSeconds) ? totalSeconds : 1 };
}

/**
 * Zoom around a pointer anchor.
 * @param {{start:number,end:number}} window current view
 * @param {number} totalSeconds full audio length
 * @param {number} anchorTime absolute seconds under the pointer
 * @param {number} anchorRatio pointer position within the window, 0..1
 * @param {number} factor <1 zooms in, >1 zooms out
 */
export function zoomAt(window, totalSeconds, anchorTime, anchorRatio, factor, minWindow = MIN_WINDOW_SECONDS) {
  const total = finitePositive(totalSeconds) ? totalSeconds : MAX_WINDOW_SECONDS;
  const current = clampWindow(window.start, window.end, total, minWindow);
  const ratio = Math.min(Math.max(Number.isFinite(anchorRatio) ? anchorRatio : 0.5, 0), 1);
  const anchor = Math.min(Math.max(anchorTime, 0), total);
  const nextDuration = (current.end - current.start) * (finitePositive(factor) ? factor : 1);
  const clampedDuration = Math.min(Math.max(nextDuration, Math.min(minWindow, total)), total);
  return clampWindow(anchor - ratio * clampedDuration, anchor - ratio * clampedDuration + clampedDuration, total, minWindow);
}

/** Translate the window by `deltaSeconds` (positive = look later). */
export function panBy(window, totalSeconds, deltaSeconds, minWindow = MIN_WINDOW_SECONDS) {
  const total = finitePositive(totalSeconds) ? totalSeconds : MAX_WINDOW_SECONDS;
  const current = clampWindow(window.start, window.end, total, minWindow);
  const delta = Number.isFinite(deltaSeconds) ? deltaSeconds : 0;
  return clampWindow(current.start + delta, current.end + delta, total, minWindow);
}

export function timeToX(timeSeconds, window, widthPx) {
  const span = window.end - window.start;
  if (!(span > 0) || !(widthPx > 0)) return 0;
  return ((timeSeconds - window.start) / span) * widthPx;
}

export function xToTime(xPx, window, widthPx) {
  const span = window.end - window.start;
  if (!(widthPx > 0)) return window.start;
  return window.start + (xPx / widthPx) * span;
}

export function secondsPerPixel(window, widthPx) {
  const span = window.end - window.start;
  return widthPx > 0 ? span / widthPx : span;
}

/**
 * Binary search: index of the first onset >= time (lower bound).
 * `onset` must be sorted ascending (guaranteed by the frozen protocol).
 */
export function lowerBound(onset, time) {
  let lo = 0;
  let hi = onset.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (onset[mid] < time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Binary search: index of the first onset > time (upper bound). */
export function upperBound(onset, time) {
  let lo = 0;
  let hi = onset.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (onset[mid] <= time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/**
 * Visible slice of a sorted onset array for [t0, t1], including events that
 * start before t0 but still sound inside the window (`maxDurationSeconds`).
 * @returns {{from: number, to: number, count: number}}
 */
export function visibleSlice(onset, t0, t1, maxDurationSeconds = 0) {
  const from = lowerBound(onset, t0 - Math.max(0, maxDurationSeconds || 0));
  const to = upperBound(onset, t1);
  return { from, to, count: Math.max(0, to - from) };
}
