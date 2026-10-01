// Peaks worker fallback correctness (review item 3):
//   - samples must survive a worker failure (no transfer-detach of the caller's
//     buffer -> the inline fallback used to build an all-zero pyramid),
//   - unusable/detached input must produce a controlled error, never a fake
//     flat waveform,
//   - outstanding worker promises are settled on invalidate/dispose (no hang).

import test from "node:test";
import assert from "node:assert/strict";

import { MediaEngine } from "../js/transport.js";

/** Fake worker that REALLY detaches transferred buffers and then errors. */
class DetachingErrorWorker {
  constructor() {
    this.listeners = new Map();
    this.terminated = false;
  }
  addEventListener(type, handler) {
    const list = this.listeners.get(type) || [];
    list.push(handler);
    this.listeners.set(type, list);
  }
  removeEventListener(type, handler) {
    const list = this.listeners.get(type) || [];
    this.listeners.set(type, list.filter((entry) => entry !== handler));
  }
  postMessage(message, transfer = []) {
    // real transfer semantics: detach the sender's buffers
    const { port1, port2 } = new MessageChannel();
    port2.postMessage(message, transfer);
    port1.close();
    port2.close();
    setTimeout(() => {
      for (const handler of this.listeners.get("error") || []) {
        handler({ message: "forced worker failure" });
      }
    }, 5);
  }
  terminate() {
    this.terminated = true;
  }
}

/** Fake worker that never answers (promise must still settle on dispose). */
class SilentWorker {
  constructor() {
    this.listeners = new Map();
    this.terminated = false;
  }
  addEventListener(type, handler) {
    const list = this.listeners.get(type) || [];
    list.push(handler);
    this.listeners.set(type, list);
  }
  removeEventListener() {}
  postMessage() {}
  terminate() {
    if (this.terminated) return; // no recursive terminate -> error -> cleanup loop
    this.terminated = true;
    for (const handler of this.listeners.get("error") || []) handler({ message: "terminated" });
  }
}

test("worker failure falls back to the ORIGINAL samples (no all-zero pyramid)", async () => {
  const engine = new MediaEngine({ createWorker: () => new DetachingErrorWorker() });
  const samples = new Float32Array([0.5, -0.5, 1]);
  const pyramid = await engine.buildPeaks(samples, 2, "fake://peaks");
  // the review reproduction: [0.5, -0.5, 1] must NOT become max 0
  const levelMax = pyramid.levels[0].max;
  assert.ok(Math.max(...levelMax) > 0.9, `expected real peaks, got ${Array.from(levelMax)}`);
  assert.ok(Math.min(...pyramid.levels[0].min) < -0.4);
  await engine.dispose();
});

test("detached/empty input raises a controlled error instead of a flat waveform", async () => {
  const engine = new MediaEngine({ createWorker: () => new DetachingErrorWorker() });
  const samples = new Float32Array([1, -1, 0.5]);
  const { port1, port2 } = new MessageChannel();
  port2.postMessage({ samples }, [samples.buffer]); // detach the caller's buffer
  port1.close();
  port2.close();
  assert.equal(samples.buffer.byteLength, 0, "buffer is detached");
  await assert.rejects(() => engine.buildPeaks(samples, 2, "fake://peaks"), /detach|不可用/);
  await engine.dispose();
});

test("without a worker the inline build works and stays correct", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  const samples = new Float32Array([0.25, -0.75]);
  const pyramid = await engine.buildPeaks(samples, 2, "fake://peaks");
  assert.equal(pyramid.levels[0].max[0], 0.25);
  assert.equal(pyramid.levels[0].min[0], -0.75);
  await engine.dispose();
});

test("dispose settles outstanding worker promises (no hanging await)", async () => {
  const engine = new MediaEngine({ createWorker: () => new SilentWorker() });
  const pending = engine.buildPeaks(new Float32Array(8), 2, "fake://peaks");
  await new Promise((resolve) => setTimeout(resolve, 20));
  await engine.dispose(); // must settle the await above
  const outcome = await Promise.race([
    pending.then(() => "settled").catch(() => "settled"),
    new Promise((resolve) => setTimeout(() => resolve("hung"), 1000)),
  ]);
  assert.equal(outcome, "settled", "pending worker promise must settle, not hang");
});

test("invalidatePending aborts in-flight loads and keeps committed media for revalidation", async () => {
  const engine = new MediaEngine({ createWorker: () => new SilentWorker() });
  const pending = engine.buildPeaks(new Float32Array(8), 2, "fake://peaks");
  await new Promise((resolve) => setTimeout(resolve, 10));
  engine.invalidatePending();
  const outcome = await Promise.race([
    pending.then(() => "settled").catch(() => "settled"),
    new Promise((resolve) => setTimeout(() => resolve("hung"), 1000)),
  ]);
  assert.equal(outcome, "settled");
  assert.equal(engine.resourceState().objectUrls, 0, "invalidation must not leak");
  await engine.dispose();
});
