// Waveform peak pyramid (min/max per bucket) with LOD selection.
//
// Level k holds buckets of `samplesPerBucket * 2**k` samples, so a whole song
// can be drawn in O(pixel width) per frame instead of O(samples). Pure module:
// usable from the Web Worker, the main thread and the unit tests alike.

export const DEFAULT_SAMPLES_PER_BUCKET = 256;

/**
 * Build the pyramid for one mono channel.
 * @param {Float32Array|Float64Array} samples
 * @param {number} samplesPerBucket base bucket size (level 0)
 * @returns {{samplesPerBucket: number, totalSamples: number, levels: Array<{min: Float32Array, max: Float32Array, bucketSamples: number}>}}
 */
export function buildPeakPyramid(samples, samplesPerBucket = DEFAULT_SAMPLES_PER_BUCKET) {
  const base = Math.max(1, Math.floor(samplesPerBucket));
  const level0Count = Math.max(1, Math.ceil(samples.length / base));
  const levels = [];
  let min = new Float32Array(level0Count);
  let max = new Float32Array(level0Count);
  for (let bucket = 0; bucket < level0Count; bucket += 1) {
    const from = bucket * base;
    const to = Math.min(samples.length, from + base);
    let lo = Infinity;
    let hi = -Infinity;
    for (let index = from; index < to; index += 1) {
      const value = samples[index];
      if (value < lo) lo = value;
      if (value > hi) hi = value;
    }
    if (!(lo <= hi)) {
      lo = 0;
      hi = 0;
    }
    min[bucket] = lo;
    max[bucket] = hi;
  }
  levels.push({ min, max, bucketSamples: base });
  while (min.length > 1) {
    const nextLength = Math.ceil(min.length / 2);
    const nextMin = new Float32Array(nextLength);
    const nextMax = new Float32Array(nextLength);
    for (let bucket = 0; bucket < nextLength; bucket += 1) {
      const a = bucket * 2;
      const b = Math.min(min.length - 1, a + 1);
      nextMin[bucket] = Math.min(min[a], min[b]);
      nextMax[bucket] = Math.max(max[a], max[b]);
    }
    min = nextMin;
    max = nextMax;
    levels.push({ min, max, bucketSamples: base * 2 ** levels.length });
  }
  return { samplesPerBucket: base, totalSamples: samples.length, levels };
}

/**
 * Pick the finest level whose buckets still cover at least one pixel column
 * (min/max aggregation is associative, so coarser levels stay accurate while
 * keeping the draw cost O(pixel width)).
 */
export function pickLevel(pyramid, sampleRate, secondsPerPx) {
  if (!pyramid || pyramid.levels.length === 0) return 0;
  const rate = sampleRate > 0 ? sampleRate : 1;
  for (let index = 0; index < pyramid.levels.length; index += 1) {
    const bucketSeconds = pyramid.levels[index].bucketSamples / rate;
    if (bucketSeconds >= secondsPerPx) return index;
  }
  return pyramid.levels.length - 1;
}

/**
 * Per-pixel min/max envelopes for [t0, t1] rendered into `width` columns.
 * @returns {{min: Float32Array, max: Float32Array, level: number}}
 */
export function peakColumns(pyramid, sampleRate, t0, t1, width) {
  const columns = Math.max(1, Math.floor(width));
  const min = new Float32Array(columns);
  const max = new Float32Array(columns);
  if (!pyramid || pyramid.levels.length === 0 || !(t1 > t0)) {
    return { min, max, level: 0 };
  }
  const rate = sampleRate > 0 ? sampleRate : 1;
  const secondsPerPx = (t1 - t0) / columns;
  const level = pickLevel(pyramid, rate, secondsPerPx);
  const buckets = pyramid.levels[level];
  const bucketSamples = buckets.bucketSamples;
  for (let column = 0; column < columns; column += 1) {
    const c0 = t0 + column * secondsPerPx;
    const c1 = c0 + secondsPerPx;
    let from = Math.floor((c0 * rate) / bucketSamples);
    let to = Math.ceil((c1 * rate) / bucketSamples);
    if (to <= from) to = from + 1;
    if (to <= 0 || from >= buckets.min.length) {
      // column completely outside the audio: flat zero
      min[column] = 0;
      max[column] = 0;
      continue;
    }
    from = Math.max(0, Math.min(from, buckets.min.length - 1));
    to = Math.max(from + 1, Math.min(to, buckets.min.length));
    let lo = Infinity;
    let hi = -Infinity;
    for (let bucket = from; bucket < to; bucket += 1) {
      if (buckets.min[bucket] < lo) lo = buckets.min[bucket];
      if (buckets.max[bucket] > hi) hi = buckets.max[bucket];
    }
    min[column] = lo;
    max[column] = hi;
  }
  return { min, max, level };
}
