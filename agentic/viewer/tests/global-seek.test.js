// Global whole-file seek slider (issue #41).
//
// Rules under test:
//   * min = 0 / max = the REAL loaded audio duration (decoded AudioBuffer or
//     HTMLMediaElement) — the JSON-declared `audio.duration_seconds` is NEVER
//     the slider max (a wrong declaration is only an explicit mismatch note),
//   * the current value is clamped to the real audio end,
//   * nothing about the slider depends on the zoomed view window,
//   * no audio / decode failure / non-finite duration -> slider disabled,
//   * replacing the audio updates the range to the new real duration.

import test from "node:test";
import assert from "node:assert/strict";

import { MediaEngine } from "../js/transport.js";
import {
  GLOBAL_SEEK_DOM_STEP,
  GLOBAL_SEEK_STEP_SECONDS,
  clampSeekTime,
  globalSeekState,
  realDurationOrNull,
} from "../js/global-seek.js";

// ---- browser API fakes (Node has no AudioContext/Audio) --------------------
class FakeAudioBuffer {
  constructor(data, sampleRate = 8000) {
    this._data = data;
    this._rate = sampleRate;
  }
  get numberOfChannels() {
    return 1;
  }
  get length() {
    return this._data.length;
  }
  get sampleRate() {
    return this._rate;
  }
  get duration() {
    return this._data.length / this._rate;
  }
  getChannelData() {
    return this._data;
  }
}

let decodeShouldFail = false;

class FakeAudioContext {
  constructor() {
    this.state = "running";
  }
  async decodeAudioData(bytes) {
    if (decodeShouldFail) throw new Error("forced decode failure");
    const text = Buffer.from(bytes).toString("utf8");
    // deterministic durations: "tiny" -> 0.001s, "long" -> 20s, else 8s @ 8kHz
    const length = text === "tiny" ? 8 : text === "fractional" ? Math.round(16.037 * 8000) : text === "long" ? 20 * 8000 : 8 * 8000;
    return new FakeAudioBuffer(new Float32Array(length));
  }
  async close() {
    this.state = "closed";
  }
}

class FakeAudioElement {
  constructor() {
    this.src = "";
    this.currentTime = 0;
    this.duration = NaN; // media metadata not loaded yet: buffer duration must win
  }
  pause() {}
  play() {
    return Promise.resolve();
  }
  removeAttribute() {}
  getAttribute(name) {
    return name === "src" ? this.src || null : null;
  }
  load() {}
}

globalThis.AudioContext = FakeAudioContext;
globalThis.Audio = FakeAudioElement;

const docAudio = { filename: "a/track.wav", sha256: "a".repeat(64), durationSeconds: 8, sampleRate: 8000 };
const file = (name, content = "fake-audio-bytes") => new File([Buffer.from(content, "utf8")], name);

// ---- pure slider model -----------------------------------------------------

test("slider range is the real audio file: min=0, max=real duration (0..16 for a 16s file)", () => {
  const model = globalSeekState({ realDurationSeconds: 16, currentTimeSeconds: 8 });
  assert.equal(model.disabled, false);
  assert.equal(model.min, 0);
  assert.equal(model.max, 16);
  assert.equal(model.value, 8);
  assert.equal(model.ratio, 0.5, "8s of a 16s file is the 50% position");
  assert.equal(model.step, GLOBAL_SEEK_STEP_SECONDS);
});

test("slider max never adopts the JSON-declared duration when the real file differs", () => {
  // the model has NO JSON/view input at all: a wrong JSON duration (12) can
  // never become the range of a 16s file, nor extend a 3s file
  assert.equal(globalSeekState({ realDurationSeconds: 16 }).max, 16);
  assert.equal(globalSeekState({ realDurationSeconds: 3 }).max, 3);
  assert.equal(clampSeekTime(14, 16), 14, "whole file is reachable even if a JSON says 12s");
  assert.equal(clampSeekTime(14, 3), 3, "but never past the REAL file end");
});

test("the current value is clamped to the real audio end and never below 0", () => {
  assert.equal(clampSeekTime(20, 16), 16);
  assert.equal(clampSeekTime(-3, 16), 0);
  assert.equal(clampSeekTime(Number.NaN, 16), 0);
  assert.equal(globalSeekState({ realDurationSeconds: 16, currentTimeSeconds: 999 }).value, 16);
  assert.equal(globalSeekState({ realDurationSeconds: 16, currentTimeSeconds: -2 }).value, 0);
});

test("no usable audio (null / NaN / Infinity / 0 / negative) disables the slider", () => {
  for (const bad of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY, 0, -1]) {
    const model = globalSeekState({ realDurationSeconds: bad });
    assert.equal(model.disabled, true, `${String(bad)} must disable`);
    assert.equal(model.max, 0);
    assert.equal(model.value, 0);
    assert.equal(model.ariaValueText, "未加载音频，滑条停用");
  }
  assert.equal(realDurationOrNull(Number.POSITIVE_INFINITY), null);
  assert.equal(realDurationOrNull(16), 16);
});

test("elapsed/duration labels and aria-valuetext are readable", () => {
  const model = globalSeekState({ realDurationSeconds: 16, currentTimeSeconds: 8 });
  assert.equal(model.elapsedText, "8.00s");
  assert.equal(model.durationText, "16.00s");
  assert.ok(model.label.includes("8.00s") && model.label.includes("16.00s") && model.label.includes("50.0%"));
  assert.ok(model.ariaValueText.includes("8.00s") && model.ariaValueText.includes("16.00s"));
});

test("DOM step stays 'any' so fractional file ends are representable (no step-grid sanitization)", () => {
  // a numeric DOM step (e.g. 0.1) would sanitize 16.037 -> 16.0 and would
  // leave a sub-0.1s file with no usable position at all
  assert.equal(globalSeekState({ realDurationSeconds: 16.037 }).domStep, "any");
  assert.equal(GLOBAL_SEEK_DOM_STEP, "any");
});

test("fractional real duration (16.037s): max and clamped value keep the fraction exactly", () => {
  const atEnd = globalSeekState({ realDurationSeconds: 16.037, currentTimeSeconds: 999 });
  assert.equal(atEnd.max, 16.037);
  assert.equal(atEnd.value, 16.037, "the raw value sits exactly on the real file end");
  assert.equal(atEnd.ratio, 1);
  const mid = globalSeekState({ realDurationSeconds: 16.037, currentTimeSeconds: 8 });
  assert.equal(mid.value, 8);
  assert.equal(mid.max, 16.037);
  // display text may round to two decimals; the raw value must not
  assert.equal(mid.durationText, "16.04s");
  assert.equal(atEnd.value, 16.037);
});

test("files shorter than one 0.1s keyboard step still have a usable whole range", () => {
  const model = globalSeekState({ realDurationSeconds: 0.05, currentTimeSeconds: 999 });
  assert.equal(model.disabled, false, "a 0.05s file is real audio: the slider must stay enabled");
  assert.equal(model.max, 0.05);
  assert.equal(model.value, 0.05, "the end of a sub-step file is reachable (not stuck at 0)");
  assert.equal(clampSeekTime(GLOBAL_SEEK_STEP_SECONDS, 0.05), 0.05, "an arrow step clamps exactly onto the real end");
  assert.equal(clampSeekTime(-1, 0.05), 0);
});

test("the slider model is independent of the view window (zoom/pan cannot move it)", () => {
  const view = { start: 0, end: 16 };
  const before = globalSeekState({ realDurationSeconds: 16, currentTimeSeconds: 8 });
  // zoom into 4..6 and pan around: the model has no view input
  view.start = 4;
  view.end = 6;
  view.start = 1.5;
  view.end = 3.5;
  const after = globalSeekState({ realDurationSeconds: 16, currentTimeSeconds: 8 });
  assert.deepEqual(after, before, "zoom/pan must not rescale or shift the slider");
});

// ---- MediaEngine: where the real duration comes from -----------------------

test("MediaEngine.durationSeconds() is the decoded file length, never the JSON declaration", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  assert.equal(engine.durationSeconds(), null, "nothing loaded");
  const result = await engine.loadMain(file("a/track.wav"), { ...docAudio, durationSeconds: 999 });
  assert.equal(result.durationSeconds, 8);
  assert.equal(engine.durationSeconds(), 8, "8s decoded buffer wins over the JSON claim of 999s");
  await engine.dispose();
});

test("a non-finite HTMLMediaElement duration never breaks the real-duration source", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  await engine.loadMain(file("a/track.wav"), docAudio);
  engine.main.audio.duration = Number.POSITIVE_INFINITY; // streaming-style metadata
  assert.equal(engine.durationSeconds(), 8, "finite decoded buffer duration is used");
  engine.main.audio.duration = 8;
  assert.equal(engine.durationSeconds(), 8);
  await engine.dispose();
});

test("replacing the audio updates the real duration (different-length file)", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  await engine.loadMain(file("a/track.wav"), docAudio);
  assert.equal(engine.durationSeconds(), 8);
  await engine.loadMain(file("b/short.wav", "tiny"), docAudio);
  assert.equal(engine.durationSeconds(), 0.001, "new file's own length replaces the old one");
  await engine.loadMain(file("c/long.wav", "long"), docAudio);
  assert.equal(engine.durationSeconds(), 20);
  await engine.dispose();
  assert.equal(engine.durationSeconds(), null, "disposed engine has no duration (slider disables)");
});

test("failed decode leaves no real duration (slider must disable)", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  decodeShouldFail = true;
  const result = await engine.loadMain(file("broken.wav", "not decodable"), docAudio);
  decodeShouldFail = false;
  assert.equal(result.failed, true);
  assert.equal(engine.durationSeconds(), null);
  await engine.dispose();
});

test("MediaEngine.seek() clamps to the real file end, not to any JSON value", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  await engine.loadMain(file("a/track.wav"), { ...docAudio, durationSeconds: 999 });
  engine.seek(100);
  assert.equal(engine.audioElement.currentTime, 8, "never past the real 8s file end");
  engine.seek(-5);
  assert.equal(engine.audioElement.currentTime, 0);
  engine.seek(4);
  assert.equal(engine.audioElement.currentTime, 4);
  await engine.dispose();
});

test("fractional file length survives end-to-end (duration source + seek clamp, no rounding)", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  await engine.loadMain(file("frac/track.wav", "fractional"), { ...docAudio, durationSeconds: 12 });
  assert.equal(engine.durationSeconds(), 16.037, "decoded 16.037s file keeps its fraction");
  engine.seek(999);
  assert.equal(engine.audioElement.currentTime, 16.037, "seek lands exactly on the real file end");
  await engine.dispose();
});
