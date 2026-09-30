// Local audio handling: file-name matching, SHA-256 verification, decode,
// ObjectURL/AudioContext/Worker lifecycle. Nothing here ever uploads or
// fetches remote data: the only inputs are File objects the human picked.

export function basename(path) {
  const parts = String(path).split(/[\\/]/);
  return parts[parts.length - 1];
}

/** SHA-256 (lowercase hex) of an ArrayBuffer via the Web Crypto API. */
export async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export const DURATION_TOLERANCE_ABS = 0.05;
export const DURATION_TOLERANCE_REL = 0.01;

/**
 * Compare the user-picked audio against the JSON metadata.
 * Returns explicit notes; the viewer never silently re-times or re-scales
 * anything to "make it fit".
 */
export function checkAudioMatch({ docAudio, pickedName, hashHex, decodedDurationSeconds }) {
  const notes = [];
  const expectedName = basename(docAudio.filename);
  const picked = basename(pickedName || "");
  if (picked && expectedName && picked.toLowerCase() !== expectedName.toLowerCase()) {
    notes.push({
      level: "warning",
      code: "A_FILENAME",
      text: `音频文件名不匹配：JSON 声明 ${expectedName}，实际选择 ${picked}（只做本地文件名匹配，不按 JSON 路径任意读取）。`,
    });
  }
  if (hashHex && hashHex !== docAudio.sha256) {
    notes.push({
      level: "error",
      code: "A_HASH",
      text: `音频内容 SHA-256 与 JSON 不一致：JSON ${docAudio.sha256}，实际 ${hashHex}。不一致时仍按 JSON 秒轴显示，绝不重定时/缩放事件。`,
    });
  }
  if (typeof decodedDurationSeconds === "number" && Number.isFinite(decodedDurationSeconds)) {
    const tolerance = Math.max(DURATION_TOLERANCE_ABS, DURATION_TOLERANCE_REL * docAudio.durationSeconds);
    const delta = decodedDurationSeconds - docAudio.durationSeconds;
    if (Math.abs(delta) > tolerance) {
      notes.push({
        level: "error",
        code: "A_DURATION",
        text: `音频时长与 JSON 不一致：JSON ${docAudio.durationSeconds}s，实际 ${decodedDurationSeconds.toFixed(3)}s（差 ${delta.toFixed(3)}s > 容差 ${tolerance.toFixed(3)}s）。时间轴以 JSON 秒为真值，不自动对齐/重定时。`,
      });
    }
  }
  return notes;
}

/**
 * Owns every audio resource so tests can assert full cleanup:
 * ObjectURLs are revoked on replace/dispose, the AudioContext is closed and
 * every worker is terminated.
 */
export class MediaEngine {
  constructor(options = {}) {
    this.createWorker = options.createWorker || null;
    this.audioContext = null;
    this.main = null; // {file, url, audio, buffer, pyramid, sampleRate}
    this.stems = new Map(); // rowId -> {file, url, pyramid, sampleRate}
    this.workers = new Set();
    this.objectUrls = new Set();
    this.notes = [];
    this.closed = false;
  }

  ensureContext() {
    if (this.closed) throw new Error("MediaEngine 已 dispose，不能再次使用");
    if (!this.audioContext) {
      const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!Context) throw new Error("AudioContext 不可用（无法解码音频）");
      this.audioContext = new Context();
    }
    return this.audioContext;
  }

  trackUrl(file) {
    const url = URL.createObjectURL(file);
    this.objectUrls.add(url);
    return url;
  }

  releaseUrl(url) {
    if (url && this.objectUrls.has(url)) {
      URL.revokeObjectURL(url);
      this.objectUrls.delete(url);
    }
  }

  spawnWorker(url) {
    if (!this.createWorker) return null;
    const worker = this.createWorker(url);
    this.workers.add(worker);
    return worker;
  }

  releaseWorker(worker) {
    if (worker && this.workers.has(worker)) {
      worker.terminate();
      this.workers.delete(worker);
    }
  }

  /**
   * Build a peak pyramid, preferring the worker and falling back inline.
   * @returns {Promise<object>} pyramid (see js/peaks.js)
   */
  async buildPeaks(samples, samplesPerBucket, workerUrl) {
    const worker = this.spawnWorker ? this.spawnWorker(workerUrl) : null;
    if (!worker) {
      const { buildPeakPyramid } = await import("./peaks.js");
      return buildPeakPyramid(samples, samplesPerBucket);
    }
    return new Promise((resolve, reject) => {
      const id = `peaks-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const onMessage = (event) => {
        if (event.data && event.data.id === id) {
          worker.removeEventListener("message", onMessage);
          this.releaseWorker(worker);
          resolve(event.data.pyramid);
        }
      };
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", (event) => {
        this.releaseWorker(worker);
        reject(new Error(`peaks worker failed: ${event.message || "unknown"}`));
      });
      worker.postMessage({ id, samples, samplesPerBucket }, [samples.buffer]);
    });
  }

  /**
   * Load the main audio file picked by the human.
   * @returns {Promise<{notes: Array, pyramid: object|null, sampleRate: number|null, durationSeconds: number|null, hashHex: string|null}>}
   */
  async loadMain(file, docAudio, options = {}) {
    this.disposeMain();
    const bytes = await file.arrayBuffer();
    const hashHex = await sha256Hex(bytes.slice(0));
    const context = this.ensureContext();
    let buffer = null;
    try {
      buffer = await context.decodeAudioData(bytes.slice(0));
    } catch (error) {
      const notes = [
        {
          level: "error",
          code: "A_DECODE",
          text: `音频解码失败：${String(error && error.message ? error.message : error)}（请选择浏览器可解码的音频文件）。`,
        },
      ];
      this.notes = notes;
      return { notes, pyramid: null, sampleRate: null, durationSeconds: null, hashHex };
    }
    const channelCount = buffer.numberOfChannels;
    const length = buffer.length;
    const mono = new Float32Array(length);
    for (let channel = 0; channel < channelCount; channel += 1) {
      const data = buffer.getChannelData(channel);
      for (let index = 0; index < length; index += 1) {
        mono[index] += data[index] / channelCount;
      }
    }
    const pyramid = await this.buildPeaks(
      mono,
      options.samplesPerBucket || 256,
      options.peaksWorkerUrl || new URL("./peaks-worker.js", import.meta.url),
    );
    const url = this.trackUrl(file);
    const audio = new Audio();
    audio.preload = "auto";
    audio.src = url;
    const notes = checkAudioMatch({
      docAudio,
      pickedName: file.name,
      hashHex,
      decodedDurationSeconds: buffer.duration,
    });
    this.main = { file, url, audio, buffer, pyramid, sampleRate: buffer.sampleRate };
    this.notes = notes;
    return {
      notes,
      pyramid,
      sampleRate: buffer.sampleRate,
      durationSeconds: buffer.duration,
      hashHex,
      numberOfChannels: channelCount,
      lengthSamples: length,
    };
  }

  /** Load one optional stem file matched against the JSON stem entry. */
  async loadStem(file, stemEntry, options = {}) {
    const key = stemEntry.filename;
    const existing = this.stems.get(key);
    if (existing) {
      this.releaseUrl(existing.url);
      this.stems.delete(key);
    }
    const bytes = await file.arrayBuffer();
    const hashHex = await sha256Hex(bytes.slice(0));
    const context = this.ensureContext();
    let buffer = null;
    try {
      buffer = await context.decodeAudioData(bytes.slice(0));
    } catch (error) {
      return {
        notes: [
          {
            level: "error",
            code: "A_DECODE",
            text: `stem 解码失败（${stemEntry.filename}）：${String(error && error.message ? error.message : error)}`,
          },
        ],
        pyramid: null,
      };
    }
    const mono = buffer.getChannelData(0);
    const pyramid = await this.buildPeaks(
      mono.slice(0),
      options.samplesPerBucket || 256,
      options.peaksWorkerUrl || new URL("./peaks-worker.js", import.meta.url),
    );
    const notes = [];
    if (hashHex !== stemEntry.sha256) {
      notes.push({
        level: "error",
        code: "A_HASH",
        text: `stem 内容 SHA-256 与 JSON 不一致：${stemEntry.filename}（JSON ${stemEntry.sha256}，实际 ${hashHex}）。`,
      });
    }
    const url = this.trackUrl(file);
    this.stems.set(key, { file, url, pyramid, sampleRate: buffer.sampleRate, notes });
    return { notes, pyramid, sampleRate: buffer.sampleRate, durationSeconds: buffer.duration, hashHex };
  }

  get audioElement() {
    return this.main ? this.main.audio : null;
  }

  play() {
    if (this.main) return this.main.audio.play();
    return Promise.resolve();
  }

  pause() {
    if (this.main) this.main.audio.pause();
  }

  seek(seconds) {
    if (this.main && Number.isFinite(seconds)) {
      const duration = this.main.audio.duration || this.main.buffer.duration;
      this.main.audio.currentTime = Math.min(Math.max(0, seconds), duration);
    }
  }

  currentTime() {
    return this.main ? this.main.audio.currentTime : 0;
  }

  disposeMain() {
    if (this.main) {
      this.main.audio.pause();
      this.main.audio.removeAttribute("src");
      this.main.audio.load();
      this.releaseUrl(this.main.url);
      this.main = null;
    }
  }

  resourceState() {
    return {
      objectUrls: this.objectUrls.size,
      workers: this.workers.size,
      audioContextState: this.audioContext ? this.audioContext.state : this.closed ? "closed" : "none",
      stemsLoaded: this.stems.size,
      mainLoaded: Boolean(this.main),
      audioSrc: this.main ? this.main.audio.getAttribute("src") : null,
    };
  }

  /** Release every resource this engine owns (idempotent). */
  dispose() {
    this.disposeMain();
    for (const [key, stem] of this.stems) {
      this.releaseUrl(stem.url);
      this.stems.delete(key);
    }
    for (const worker of [...this.workers]) {
      this.releaseWorker(worker);
    }
    this.closed = true;
    if (this.audioContext) {
      const closing = this.audioContext.close ? this.audioContext.close() : Promise.resolve();
      return closing;
    }
    return Promise.resolve();
  }
}
