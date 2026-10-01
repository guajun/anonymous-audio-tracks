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

/** Slider step (seconds): keyboard arrows move the playhead by this much. */
export const GLOBAL_SEEK_STEP_SECONDS = 0.1;

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
 *            step: number, elapsedText: string, durationText: string,
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
