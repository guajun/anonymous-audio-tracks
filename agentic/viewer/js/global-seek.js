// Global whole-file seek slider model (issue #41).
//
// The slider belongs to the PLAYER, not to the zoomed time window:
//   * min = 0 (file start), max = the REAL loaded audio duration
//     (decoded AudioBuffer / HTMLMediaElement finite duration),
//   * it never uses the JSON-declared `audio.duration_seconds` as the real
//     file length (a wrong JSON declaration is reported as an explicit
//     mismatch by js/transport.js `checkAudioMatch`, never silently adopted),
//   * it never follows `view.start/end`: zooming/panning must not rescale or
//     move the slider range or its value.
//
// Pure data + math only (no DOM): these rules are unit-testable in Node and
// the UI just projects the returned model onto the range input.

import { formatSeconds } from "./render.js";

/**
 * Keyboard arrow step (seconds): the explicit key handler moves the playhead
 * by this much. The DOM `step` is deliberately NOT 0.1 — see below.
 */
export const GLOBAL_SEEK_STEP_SECONDS = 0.1;
/** Keyboard PageUp/PageDown step (seconds). */
export const GLOBAL_SEEK_PAGE_STEP_SECONDS = 1;
/**
 * DOM `step` for the range input. It MUST stay "any":
 * a numeric step makes the browser sanitize the value onto the step grid, so
 * a file of e.g. 16.037s could never be reached (value snaps to 16.0) and a
 * file shorter than one step would have no usable position at all. With
 * "any" the raw value can sit exactly on the REAL file end; arrows/Home/End
 * are handled explicitly (0.1s step / exact bounds) in app.js.
 */
export const GLOBAL_SEEK_DOM_STEP = "any";

function finitePositive(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * The real loaded-audio duration, or null when there is no usable audio
 * (nothing loaded, decode failure, NaN/Infinity metadata).
 */
export function realDurationOrNull(durationSeconds) {
  return finitePositive(durationSeconds) ? durationSeconds : null;
}

/**
 * Clamp a seek target to the real audio file bounds [0, duration].
 * Without a usable duration only the lower bound is enforced; the value is
 * NEVER allowed above the real audio end once one exists.
 */
export function clampSeekTime(seconds, realDurationSeconds) {
  const duration = realDurationOrNull(realDurationSeconds);
  const value = Number.isFinite(seconds) ? seconds : 0;
  const lo = Math.max(0, value);
  return duration === null ? lo : Math.min(lo, duration);
}

/**
 * Complete slider state. Inputs are ONLY the real audio duration and the
 * current playhead — there is deliberately no view/JSON parameter, so zoom,
 * pan and a wrong JSON duration can never leak into the slider range.
 *
 * @param {{realDurationSeconds?: number|null, currentTimeSeconds?: number}} input
 * @returns {{disabled: boolean, min: number, max: number, value: number, ratio: number,
 *            step: number, domStep: string, elapsedText: string, durationText: string,
 *            label: string, ariaValueText: string}}
 */
export function globalSeekState({ realDurationSeconds = null, currentTimeSeconds = 0 } = {}) {
  const duration = realDurationOrNull(realDurationSeconds);
  const disabled = duration === null;
  const min = 0;
  const max = disabled ? 0 : duration;
  const value = clampSeekTime(currentTimeSeconds, duration);
  const ratio = max > 0 ? value / max : 0;
  const percent = (ratio * 100).toFixed(1);
  const elapsedText = formatSeconds(value, GLOBAL_SEEK_STEP_SECONDS);
  const durationText = formatSeconds(max, GLOBAL_SEEK_STEP_SECONDS);
  return {
    disabled,
    min,
    max,
    value,
    ratio,
    step: GLOBAL_SEEK_STEP_SECONDS,
    domStep: GLOBAL_SEEK_DOM_STEP,
    elapsedText,
    durationText,
    label: disabled
      ? `已播 ${elapsedText} / 全长 ${durationText}（未加载可解码音频：滑条停用）`
      : `已播 ${elapsedText} / 全长 ${durationText}（整曲 ${percent}%）`,
    ariaValueText: disabled
      ? "未加载音频，滑条停用"
      : `已播 ${elapsedText} / 全长 ${durationText}（整曲 ${percent}%）`,
  };
}
