// Downsampled per-track activity envelopes for the whole-song overview strip.
//
// Long songs (tens of minutes) must be navigable without iterating hundreds of
// thousands of protocol points on every animation frame, so the envelope is
// computed once per (trajectory, canvas width) and cached by app.js.  The math
// is pure and unit tested.

/**
 * @param {Array} tracks normalized trajectory tracks
 * @param {number} trackStartSeconds original-track time of local 0
 * @param {number} duration local timeline duration in seconds
 * @param {number} width number of columns (canvas pixels)
 * @returns {Array<{trackId: string, max: Float32Array, hasData: boolean}>}
 */
export function computeOverviewBins(tracks, trackStartSeconds, duration, width) {
  const columns = Math.max(0, Math.floor(width));
  return tracks.map((track) => {
    const max = new Float32Array(columns);
    const known = new Uint8Array(columns);
    let hasData = false;
    if (
      columns === 0 ||
      !Number.isFinite(duration) ||
      duration <= 0 ||
      !Number.isFinite(trackStartSeconds)
    ) {
      return { trackId: track.trackId, max, hasData };
    }
    for (let index = 0; index < track.centerTimes.length; index += 1) {
      const localSeconds = track.centerTimes[index] - trackStartSeconds;
      if (localSeconds < 0 || localSeconds > duration) {
        continue;
      }
      const column = Math.min(
        columns - 1,
        Math.max(0, Math.floor(localSeconds / duration * columns)),
      );
      const value = track.activity[index];
      if (Number.isFinite(value)) {
        max[column] = Math.max(max[column], value);
        known[column] = 1;
        hasData = true;
      }
    }
    fillInterpolatedColumns(max, known);
    return { trackId: track.trackId, max, hasData };
  });
}

/**
 * Fill columns between sampled columns by linear interpolation.  Sampled-zero
 * columns count as data, so a genuine ramp from 0 to 1 is drawn as a ramp
 * instead of being flattened into a gap.
 */
function fillInterpolatedColumns(max, known) {
  const columns = max.length;
  let previous = -1;
  for (let index = 0; index < columns; index += 1) {
    if (known[index] === 0) {
      continue;
    }
    if (previous >= 0 && index - previous > 1) {
      const gap = index - previous;
      for (let step = 1; step < gap; step += 1) {
        const ratio = step / gap;
        max[previous + step] = max[previous] * (1 - ratio) + max[index] * ratio;
        known[previous + step] = 1;
      }
    }
    previous = index;
  }
}
