// Web Worker: waveform peak pyramid construction off the UI thread.
// Message in:  { id, samples: Float32Array, samplesPerBucket }
// Message out: { id, pyramid } (typed arrays transferred back)

import { buildPeakPyramid } from "./peaks.js";

self.addEventListener("message", (event) => {
  const { id, samples, samplesPerBucket } = event.data || {};
  try {
    const pyramid = buildPeakPyramid(samples, samplesPerBucket);
    const transfer = pyramid.levels.flatMap((level) => [level.min.buffer, level.max.buffer]);
    self.postMessage({ id, pyramid }, transfer);
  } catch (error) {
    self.postMessage({ id, error: String(error && error.message ? error.message : error) });
  }
});
