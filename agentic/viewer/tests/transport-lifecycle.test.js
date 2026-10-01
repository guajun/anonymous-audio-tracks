// MediaEngine load lifecycle (review item 2): per-kind tokens, document
// ownership, failed-decode clearing, concurrent main+stem completion.

import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

import { MediaEngine } from "../js/transport.js";

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

let decodeGate = null; // optional promise gate to delay decode
let decodeShouldFail = false;

class FakeAudioContext {
  constructor() {
    this.state = "running";
  }
  async decodeAudioData(bytes) {
    if (decodeGate) await decodeGate;
    if (decodeShouldFail) throw new Error("forced decode failure");
    const view = new Uint8Array(bytes);
    const text = Buffer.from(bytes).toString("utf8");
    // deterministic duration: "tiny" decodes to 8 samples (0.001s),
    // everything else to 8s @ 8000Hz (the declared doc duration in tests)
    const length = text === "tiny" ? 8 : 8 * 8000;
    const data = new Float32Array(length);
    for (let index = 0; index < data.length; index += 1) data[index] = ((view[index % view.length] || 0) / 255) * 0.5;
    return new FakeAudioBuffer(data);
  }
  async close() {
    this.state = "closed";
  }
}

class FakeAudioElement {
  constructor() {
    this.src = "";
    this.currentTime = 0;
    this.duration = NaN;
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

const sha = (char) => char.repeat(64);
const sha256 = (content) => createHash("sha256").update(Buffer.from(content, "utf8")).digest("hex");
const STEM_BYTES = "stem-bytes";
const STEM_HASH = sha256(STEM_BYTES);
const OTHER_STEM_BYTES = "other-stem-bytes";
const OTHER_STEM_HASH = sha256(OTHER_STEM_BYTES);
const docAudio = { filename: "a/track.wav", sha256: sha("a"), durationSeconds: 8, sampleRate: 8000 };
const row = (id, hash) => ({ id, stem: { filename: `stems/${id}/target.wav`, sha256: hash } });

function file(name, content = "fake-audio-bytes") {
  return new File([Buffer.from(content, "utf8")], name);
}

test("concurrent main audio + stem loads BOTH complete (separate load tokens)", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  let release;
  decodeGate = new Promise((resolve) => {
    release = resolve;
  });
  const mainPromise = engine.loadMain(file("a/track.wav"), docAudio);
  const stemPromise = engine.loadStem(file("stems/inst-1/target.wav", STEM_BYTES), [row("inst-1", STEM_HASH)], {
    durationSeconds: 8,
  });
  decodeGate = null;
  release();
  const [main, stem] = await Promise.all([mainPromise, stemPromise]);
  assert.equal(main.aborted, undefined, "main load must NOT be cancelled by the stem selection");
  assert.ok(main.pyramid, "main committed");
  assert.equal(stem.applied, true, "stem committed");
  assert.equal(engine.resourceState().mainLoaded, true);
  assert.equal(engine.resourceState().stemsLoaded, 1);
  await engine.dispose();
});

test("latest failed decode clears stale playable audio (no prior file under new name)", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  const first = await engine.loadMain(file("a/track.wav"), docAudio);
  assert.ok(first.pyramid);
  assert.equal(engine.resourceState().mainLoaded, true);

  decodeShouldFail = true;
  const second = await engine.loadMain(file("broken.wav", "not decodable"), docAudio);
  decodeShouldFail = false;
  assert.equal(second.failed, true);
  assert.equal(engine.resourceState().mainLoaded, false, "previous audio must not stay playable");
  assert.equal(engine.resourceState().audioSrc, null);
  assert.equal(engine.audioElement, null);
  assert.ok(second.notes.some((note) => note.code === "A_DECODE"));
  await engine.dispose();
});

test("invalidatePending aborts in-flight main loads (no stale commit after doc change)", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  let release;
  decodeGate = new Promise((resolve) => {
    release = resolve;
  });
  const pending = engine.loadMain(file("a/track.wav"), docAudio);
  decodeGate = null;
  engine.invalidatePending(); // document changed while decoding
  release();
  const result = await pending;
  assert.equal(result.aborted, true, "superseded load must abort");
  assert.equal(engine.resourceState().mainLoaded, false, "no stale audio committed");
  await engine.dispose();
});

test("invalidatePending aborts in-flight stem loads without touching committed stems", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  const applied = await engine.loadStem(file("stems/inst-1/target.wav", STEM_BYTES), [row("inst-1", STEM_HASH)], {
    durationSeconds: 8,
  });
  assert.equal(applied.applied, true);

  let release;
  decodeGate = new Promise((resolve) => {
    release = resolve;
  });
  const pending = engine.loadStem(file("stems/inst-2/target.wav", OTHER_STEM_BYTES), [row("inst-2", OTHER_STEM_HASH)], {
    durationSeconds: 8,
  });
  decodeGate = null;
  engine.invalidatePending();
  release();
  const result = await pending;
  assert.equal(result.aborted, true);
  assert.equal(engine.resourceState().stemsLoaded, 1, "committed stem stays for explicit revalidation");
  await engine.dispose();
});

test("stem identity failure leaves committed stems untouched", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  await engine.loadStem(file("stems/inst-1/target.wav", STEM_BYTES), [row("inst-1", STEM_HASH)], {
    durationSeconds: 8,
  });
  const wrong = await engine.loadStem(file("stems/inst-2/target.wav", "different"), [row("inst-2", OTHER_STEM_HASH)], {
    durationSeconds: 8,
  });
  assert.equal(wrong.applied, false);
  assert.ok(wrong.notes.some((note) => note.code === "STEM_IDENTITY"));
  assert.equal(engine.resourceState().stemsLoaded, 1);
  await engine.dispose();
});

test("stem duration mismatch refuses the stem (no auto-shift)", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  // hash matches, but the fake decode duration (tiny) is far from the declared 8s
  const tiny = "tiny";
  const result = await engine.loadStem(file("stems/inst-3/target.wav", tiny), [row("inst-3", sha256(tiny))], {
    durationSeconds: 8,
  });
  assert.equal(result.applied, false);
  assert.ok(result.notes.some((note) => note.code === "STEM_DURATION"));
  assert.equal(engine.resourceState().stemsLoaded, 0);
  await engine.dispose();
});

test("reconcileStems keeps only (rowId, filename, hash) matches on doc change", async () => {
  const engine = new MediaEngine({ createWorker: () => null });
  await engine.loadStem(file("stems/inst-1/target.wav", STEM_BYTES), [row("inst-1", STEM_HASH)], {
    durationSeconds: 8,
  });
  const dropped = engine.reconcileStems({ rows: [row("other-row", STEM_HASH)] });
  assert.deepEqual(dropped, ["inst-1"]);
  assert.equal(engine.resourceState().stemsLoaded, 0);
  await engine.dispose();
});
