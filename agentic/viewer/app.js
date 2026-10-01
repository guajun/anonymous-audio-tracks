// agentic/viewer — high-performance local viewer for
// `agentic-audio-tracks/v1` documents (issue #34).
//
// Design constraints honored here:
//   * local File inputs only — nothing is uploaded or fetched from a CDN,
//   * one canvas per lane + one DOM node per instrument row (never per event),
//   * viewport culling via binary search, waveform via peak pyramid LOD,
//   * BPM edits only the beat-grid overlay; onsets are never re-timed,
//   * labels/notes/errors go through textContent / Canvas fillText only,
//   * every resource (ObjectURL / AudioContext / Worker / rAF) is released on
//     dispose() so tests can assert cleanup.

import {
  applyBpm,
  buildModel,
  createBpmState,
  beatGrid,
} from "./js/model.js";
import { parseAndValidate } from "./js/protocol.js";
import { readFileStrict } from "./js/strict-json.js";
import {
  clampWindow,
  fitWindow,
  panBy,
  timeToX,
  xToTime,
  zoomAt,
} from "./js/timeline.js";
import {
  DEFAULT_ROW_BUDGET,
  DEFAULT_TOTAL_BUDGET,
  envelopeFor,
  formatSeconds,
  renderRuler,
  renderTracks,
  renderWaveformLane,
} from "./js/render.js";
import { MediaEngine, basename, checkAudioMatch } from "./js/transport.js";

const RULER_HEIGHT = 26;
const WAVE_HEIGHT = 84;
const ROW_HEIGHT = 44;
const MIN_TRACK_HEIGHT = 88;
const BIG_FILE_BYTES = 64 * 1024 * 1024;

const elements = {
  jsonInput: document.getElementById("json-input"),
  audioInput: document.getElementById("audio-input"),
  stemInput: document.getElementById("stem-input"),
  status: document.getElementById("status"),
  fileState: document.getElementById("file-state"),
  notes: document.getElementById("notes"),
  issues: document.getElementById("issues"),
  playButton: document.getElementById("play-button"),
  pauseButton: document.getElementById("pause-button"),
  seekInput: document.getElementById("seek-input"),
  fitButton: document.getElementById("fit-button"),
  resetButton: document.getElementById("reset-button"),
  zoomInButton: document.getElementById("zoom-in-button"),
  zoomOutButton: document.getElementById("zoom-out-button"),
  viewWindow: document.getElementById("view-window"),
  bpmInput: document.getElementById("bpm-input"),
  bpmApply: document.getElementById("bpm-apply"),
  bpmJson: document.getElementById("bpm-json"),
  bpmOrigin: document.getElementById("bpm-origin"),
  bpmStatus: document.getElementById("bpm-status"),
  gutterRows: document.getElementById("gutter-rows"),
  canvasStack: document.getElementById("canvas-stack"),
  rulerCanvas: document.getElementById("ruler-canvas"),
  waveCanvas: document.getElementById("wave-canvas"),
  trackCanvas: document.getElementById("track-canvas"),
  metrics: document.getElementById("metrics"),
};

const state = {
  doc: null,
  model: null,
  bpmState: null,
  view: { start: 0, end: 1 },
  peaks: null,
  stemPeaks: new Map(), // row.id -> pyramid (stable id, never row index)
  stemSampleRates: new Map(), // row.id -> sampleRate
  audio: { loaded: false, hashHex: null, durationSeconds: null, sampleRate: null, notes: [] },
  metrics: {
    validateMs: null,
    decodeMs: null,
    peaksMs: null,
    firstRenderMs: null,
    lastRenderMs: null,
    frameIntervalStartToStartMs: [],
    frameFullRenderMs: [],
  },
  lastRender: null,
  statusText: "等待加载 JSON 结果文件。",
  playing: false,
  rafId: null,
  disposed: false,
  schema: null,
  loads: { json: 0, audio: 0, stems: 0 },
  noteSets: { doc: [], audio: [], stems: [] },
  // generation tokens: a slower earlier load must never overwrite a newer one
  tokens: { json: 0, audio: 0, stems: 0 },
};

/** Per-dataset metrics reset: timings must describe the CURRENT dataset only. */
function resetDatasetMetrics(part) {
  state.metrics.firstRenderMs = null;
  state.metrics.lastRenderMs = null;
  // reset the CURRENT (post-rename) frame arrays — the previous dataset's
  // samples must never leak into the new dataset's footer
  state.metrics.frameIntervalStartToStartMs = [];
  state.metrics.frameFullRenderMs = [];
  // an in-flight frame probe would keep writing into the fresh arrays: stop it
  if (state.frameProbeId) {
    cancelAnimationFrame(state.frameProbeId);
    state.frameProbeId = null;
  }
  if (part === "json") {
    state.metrics.validateMs = null;
    state.metrics.decodeMs = null;
    state.metrics.peaksMs = null;
  }
}

/** Document ownership: pending audio/stem loads of the OLD document must never commit. */
function invalidatePendingMedia() {
  state.tokens.audio += 1;
  state.tokens.stems += 1;
  media.invalidatePending();
}

const media = new MediaEngine({
  createWorker: (url) => {
    try {
      return new Worker(url, { type: "module" });
    } catch {
      return null;
    }
  },
});

let validateWorker = null;
const validatePending = new Set(); // outstanding worker awaits, settled on dispose

function terminateValidateWorker() {
  if (validateWorker) {
    for (const entry of [...validatePending]) {
      validatePending.delete(entry);
      entry.reject(new Error("viewer disposed，校验任务作废"));
    }
    validateWorker.terminate();
    validateWorker = null;
  }
}

function getValidateWorker() {
  if (validateWorker) return validateWorker;
  try {
    validateWorker = new Worker(new URL("./js/validate-worker.js", import.meta.url), { type: "module" });
    return validateWorker;
  } catch {
    return null;
  }
}

async function loadSchema() {
  const response = await fetch(new URL("./schema/agentic-audio-tracks-v1.schema.json", import.meta.url));
  if (!response.ok) throw new Error(`无法加载协议 schema（HTTP ${response.status}）`);
  return response.json();
}

function setStatus(text, kind = "") {
  state.statusText = text;
  elements.status.textContent = text;
  elements.status.className = kind;
}

function renderNotes(notes) {
  elements.notes.textContent = "";
  for (const note of notes || []) {
    const item = document.createElement("li");
    item.className = note.level || "info";
    // codes stay visible so states are traceable (STEM_IDENTITY / A_HASH / ...)
    item.textContent = `[${note.code || "note"}] ${note.text}`;
    elements.notes.appendChild(item);
  }
}

// Notes are kept per source so a later audio/stem status never erases the
// document's own notes (e.g. "stale stems dropped").
function setNotes(part, notes) {
  state.noteSets[part] = notes || [];
  renderNotes([...state.noteSets.doc, ...state.noteSets.audio, ...state.noteSets.stems]);
}

function renderIssues(issues) {
  elements.issues.textContent = "";
  for (const issue of issues || []) {
    const item = document.createElement("li");
    item.className = "error";
    item.textContent = `${issue.pointer || "<root>"} [${issue.layer}/${issue.code}] ${issue.message}`;
    elements.issues.appendChild(item);
  }
}

function cssWidth() {
  return Math.max(1, elements.canvasStack.clientWidth);
}

function setupCanvas(canvas, logicalHeight) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const logicalWidth = cssWidth();
  canvas.style.height = `${logicalHeight}px`;
  const targetWidth = Math.round(logicalWidth * dpr);
  const targetHeight = Math.round(logicalHeight * dpr);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function trackCanvasHeight() {
  if (!state.model) return MIN_TRACK_HEIGHT;
  return Math.max(MIN_TRACK_HEIGHT, state.model.rows.length * ROW_HEIGHT);
}

function renderGutterRows() {
  const container = elements.gutterRows;
  container.textContent = "";
  if (!state.model) return;
  state.model.rows.forEach((row, index) => {
    const div = document.createElement("div");
    div.className = "gutter-row";
    div.style.top = `${index * ROW_HEIGHT}px`;
    div.style.height = `${ROW_HEIGHT}px`;
    const label = document.createElement("div");
    label.textContent = row.label;
    const id = document.createElement("div");
    id.className = "row-id";
    id.textContent = `id: ${row.id} · ${row.count} 事件`;
    const meta = document.createElement("div");
    meta.className = "row-meta";
    meta.textContent = `source=${row.source}${row.stem ? ` · stem=${basename(row.stem.filename)}` : ""}`;
    div.appendChild(label);
    div.appendChild(id);
    div.appendChild(meta);
    container.appendChild(div);
  });
}

function render() {
  if (state.disposed) return null;
  const width = cssWidth();
  const rulerCtx = setupCanvas(elements.rulerCanvas, RULER_HEIGHT);
  const waveCtx = setupCanvas(elements.waveCanvas, WAVE_HEIGHT);
  const trackCtx = setupCanvas(elements.trackCanvas, trackCanvasHeight());

  renderRuler(rulerCtx, state.view, { width, height: RULER_HEIGHT });

  const envelope = state.peaks
    ? envelopeFor(state.peaks, state.audio.sampleRate || 44100, state.view, Math.max(1, Math.round(width)))
    : null;
  renderWaveformLane(waveCtx, envelope, state.view, {
    width,
    height: WAVE_HEIGHT,
    playheadTime: playheadTime(),
    label: state.audio.loaded ? "（波形计算中…）" : "（未加载音频波形）",
  });

  if (!state.model) {
    const ctx = trackCtx;
    ctx.save();
    ctx.clearRect(0, 0, width, trackCanvasHeight());
    ctx.fillStyle = "#5b6675";
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("等待加载 JSON 结果文件（协议 agentic-audio-tracks/v1）。", 12, 24);
    ctx.restore();
    state.lastRender = { visibleEvents: 0, drawnEvents: 0, aggregatedRows: 0, rows: [] };
    updateMetricsText();
    return state.lastRender;
  }

  const rowEnvelopes = new Map();
  for (const [rowId, pyramid] of state.stemPeaks) {
    rowEnvelopes.set(
      rowId,
      envelopeFor(pyramid, state.stemSampleRates.get(rowId) || 44100, state.view, Math.max(1, Math.round(width))),
    );
  }

  const stats = renderTracks(trackCtx, state.model, state.view, {
    width,
    height: trackCanvasHeight(),
    rulerHeight: 0,
    rowHeight: ROW_HEIGHT,
    bpmState: state.bpmState,
    playheadTime: playheadTime(),
    rowBudget: DEFAULT_ROW_BUDGET,
    totalBudget: DEFAULT_TOTAL_BUDGET,
    rowEnvelopes,
    now: () => performance.now(),
  });
  state.lastRender = stats;
  state.metrics.lastRenderMs = stats.renderMs;
  if (state.metrics.firstRenderMs === null) state.metrics.firstRenderMs = stats.renderMs;
  elements.viewWindow.textContent = `视图 ${formatSeconds(state.view.start, 0.001)} – ${formatSeconds(
    state.view.end,
    0.001,
  )}（总 ${formatSeconds(state.model.audio.durationSeconds || 0, 0.001)}）`;
  updateMetricsText();
  return stats;
}

function updateMetricsText() {
  const m = state.metrics;
  const frames = m.frameIntervalStartToStartMs;
  const renders = m.frameFullRenderMs;
  const frameInfo = frames.length
    ? `帧采样（每帧整场景重绘）n=${frames.length} start-to-start间隔avg=${(
        frames.reduce((a, b) => a + b, 0) / frames.length
      ).toFixed(2)}ms/最大=${Math.max(...frames).toFixed(2)}ms 整场景重绘耗时avg=${
        renders.length ? (renders.reduce((a, b) => a + b, 0) / renders.length).toFixed(2) : "-"
      }ms/最大=${renders.length ? Math.max(...renders).toFixed(2) : "-"}ms`
    : "帧采样 n=0（未测）";
  const r = state.lastRender || {};
  const grid = currentGrid();
  elements.metrics.textContent =
    `validate=${m.validateMs === null ? "-" : m.validateMs.toFixed(1)}ms · decode=${
      m.decodeMs === null ? "-" : m.decodeMs.toFixed(1)
    }ms · render=${m.lastRenderMs === null ? "-" : m.lastRenderMs.toFixed(2)}ms · visible=${
      r.visibleEvents || 0
    } · drawn=${r.drawnEvents || 0} · candidates=${r.candidateEvents || 0} · aggregatedRows=${r.aggregatedRows || 0} · ${frameInfo} · dpr=${
      window.devicePixelRatio || 1
    }${grid && grid.unsupported ? ` · 网格不支持：${grid.reason}` : ""}`;
}

function currentGrid() {
  if (!state.model) return null;
  return beatGrid(
    state.bpmState,
    state.model.tempo.hasBeatOrigin ? state.model.tempo.beatOriginSeconds : 0,
    state.view.start,
    state.view.end,
    64,
  );
}

function playheadTime() {
  const element = media.audioElement;
  return element ? element.currentTime : null;
}

function setView(next) {
  const total = state.model ? state.model.audio.durationSeconds : 1;
  state.view = clampWindow(next.start, next.end, total);
  render();
}

function fit() {
  const total = state.model ? state.model.audio.durationSeconds : 1;
  state.view = fitWindow(total);
  render();
}

function zoomAtX(factor, xPx) {
  const width = cssWidth();
  const anchorTime = xToTime(xPx, state.view, width);
  const ratio = xPx / width;
  const total = state.model ? state.model.audio.durationSeconds : 1;
  state.view = zoomAt(state.view, total, anchorTime, ratio, factor);
  render();
}

function panPx(deltaPx) {
  const width = cssWidth();
  const seconds = (deltaPx / width) * (state.view.end - state.view.start);
  const total = state.model ? state.model.audio.durationSeconds : 1;
  state.view = panBy(state.view, total, -seconds);
  render();
}

function seekTo(seconds) {
  const total = state.model ? state.model.audio.durationSeconds : 0;
  const clamped = Math.min(Math.max(0, Number.isFinite(seconds) ? seconds : 0), total);
  if (media.audioElement) media.seek(clamped);
  elements.seekInput.value = clamped.toFixed(2);
  render();
}

function startFrameProbe(count = 60) {
  // Honest measurement: every sampled frame performs a FULL scene re-render
  // (ruler + waveform + all rows) and records both the rAF interval and the
  // render duration. Idle scheduling alone is never reported as "fps with
  // repaint".
  state.metrics.frameIntervalStartToStartMs = [];
  state.metrics.frameFullRenderMs = [];
  let lastStart = null;
  let remaining = count;
  const step = () => {
    if (state.disposed) return;
    // start-to-start frame interval (includes the previous frame's render work)
    const frameStart = performance.now();
    if (lastStart !== null) state.metrics.frameIntervalStartToStartMs.push(frameStart - lastStart);
    lastStart = frameStart;
    // full scene render wrapper: ruler + waveform + envelopes + rows + metrics
    const renderStart = performance.now();
    render();
    state.metrics.frameFullRenderMs.push(performance.now() - renderStart);
    remaining -= 1;
    if (remaining > 0) {
      state.frameProbeId = requestAnimationFrame(step);
    } else {
      state.frameProbeId = null;
      updateMetricsText();
    }
  };
  if (state.frameProbeId) cancelAnimationFrame(state.frameProbeId);
  state.frameProbeId = requestAnimationFrame(step);
}

function playbackLoop() {
  if (state.disposed || !state.playing) return;
  render();
  const element = media.audioElement;
  if (element) elements.seekInput.value = element.currentTime.toFixed(2);
  state.rafId = requestAnimationFrame(playbackLoop);
}

async function play() {
  if (!media.audioElement) {
    setStatus("尚未加载音频文件：请先加载 JSON，再选择音频文件。", "error");
    return;
  }
  try {
    await media.play();
    state.playing = true;
    setStatus(`播放中（${state.audio.notes.some((n) => n.level === "error") ? "注意：有音频匹配警告，见下" : "音频秒时间轴对齐"}）`, "ok");
    startFrameProbe(60);
    if (state.rafId) cancelAnimationFrame(state.rafId);
    state.rafId = requestAnimationFrame(playbackLoop);
  } catch (error) {
    setStatus(`播放失败：${String(error && error.message ? error.message : error)}`, "error");
  }
}

function pause() {
  media.pause();
  state.playing = false;
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  render();
  setStatus("已暂停。");
}

async function validateTextAsync(text, source) {
  const started = performance.now();
  const worker = getValidateWorker();
  if (worker) {
    try {
      const result = await new Promise((resolve, reject) => {
        const id = `validate-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const entry = { reject };
        validatePending.add(entry);
        const cleanup = () => {
          worker.removeEventListener("message", onMessage);
          worker.removeEventListener("error", onError);
          worker.removeEventListener("messageerror", onMessageError);
          validatePending.delete(entry);
        };
        const onMessage = (event) => {
          if (event.data && event.data.id === id) {
            cleanup();
            resolve(event.data);
          }
        };
        const onError = (event) => {
          cleanup();
          reject(new Error(`validate worker failed: ${event.message || "worker error"}`));
        };
        const onMessageError = () => {
          cleanup();
          reject(new Error("validate worker messageerror"));
        };
        worker.addEventListener("message", onMessage);
        worker.addEventListener("error", onError);
        worker.addEventListener("messageerror", onMessageError);
        worker.postMessage({ id, text, schema: state.schema, source });
      });
      return { ...result, elapsed: performance.now() - started };
    } catch {
      // worker failed: fall back to inline validation (never hang)
    }
  }
  const result = parseAndValidate(text, state.schema, { source });
  return { report: result.report, data: result.data, elapsed: performance.now() - started };
}

/** Reset the whole loaded state (doc/model/stems/audio/playback). */
function clearLoadedState({ keepAudio = false } = {}) {
  state.doc = null;
  state.model = null;
  state.bpmState = null;
  state.peaks = null;
  for (const rowId of [...state.stemPeaks.keys()]) state.stemPeaks.delete(rowId);
  for (const rowId of [...state.stemSampleRates.keys()]) state.stemSampleRates.delete(rowId);
  media.reconcileStems({ rows: [] });
  invalidatePendingMedia();
  if (!keepAudio) {
    media.disposeMain();
    state.audio = { loaded: false, hashHex: null, durationSeconds: null, sampleRate: null, notes: [] };
  }
  state.playing = false;
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  setNotes("doc", []);
  setNotes("audio", []);
  setNotes("stems", []);
  renderGutterRows();
}

async function loadJsonFile(file) {
  const token = ++state.tokens.json;
  renderIssues([]);
  setNotes("doc", []);
  setNotes("stems", []);
  if (file.size > BIG_FILE_BYTES) {
    setStatus(`JSON 文件较大（${(file.size / 1024 / 1024).toFixed(1)}MB）：读取与校验在 Web Worker 中异步进行…`);
  } else {
    setStatus(`读取 ${file.name}…`);
  }
  elements.fileState.textContent = `JSON 文件：${file.name}（${file.size} bytes）`;
  // Strict byte-level entry: File.text() would silently strip a BOM and replace
  // invalid UTF-8 with U+FFFD — read raw bytes and apply the frozen decode
  // policy instead (BOM / bad encoding -> controlled E_PARSE).
  const { text, issues: decodeIssues } = await readFileStrict(file);
  if (state.disposed || token !== state.tokens.json) return;
  if (decodeIssues.length > 0) {
    clearLoadedState();
    resetDatasetMetrics("json");
    setStatus(`JSON 文件字节不符合严格 UTF-8 策略（${decodeIssues.length} 个问题）：已清除旧数据，不绘制错误数据。`, "error");
    renderIssues(decodeIssues);
    render();
    updateMetricsText();
    return;
  }
  const { report, data, elapsed } = await validateTextAsync(text, file.name);
  // generation guard: a slower earlier load must never overwrite a newer one
  if (state.disposed || token !== state.tokens.json) return;
  state.metrics.validateMs = elapsed;
  if (!report.ok) {
    // never label a stale graph as the newly selected (invalid) JSON
    clearLoadedState();
    resetDatasetMetrics("json");
    state.metrics.validateMs = elapsed;
    setStatus(`JSON 未通过 agentic-audio-tracks/v1 校验（${report.issues.length} 个问题）：已清除旧数据，不绘制错误数据。`, "error");
    renderIssues(report.issues);
    render();
    updateMetricsText();
    return;
  }
  // new document owns the app state: abort pending audio/stem loads of the old one
  invalidatePendingMedia();
  state.doc = data;
  state.model = buildModel(data);
  state.bpmState = createBpmState(state.model);
  state.view = fitWindow(state.model.audio.durationSeconds);
  resetDatasetMetrics("json");
  state.metrics.validateMs = elapsed;
  // document change: re-validate applied stems by (rowId, filename, sha256)
  const droppedStems = media.reconcileStems(state.model);
  for (const rowId of [...state.stemPeaks.keys()]) {
    if (!media.stems.has(rowId)) {
      state.stemPeaks.delete(rowId);
      state.stemSampleRates.delete(rowId);
    }
  }
  syncBpmUi();
  renderGutterRows();
  const emptyRows = state.model.rows.filter((row) => row.count === 0).length;
  setStatus(
    `已加载 ${file.name}：${state.model.rows.length} 个乐器条目（同类不同 id 不合并）、${state.model.totalEvents} 个事件` +
      (emptyRows ? `、${emptyRows} 行无事件` : "") +
      `，音频声明 ${state.model.audio.durationSeconds}s。`,
    "ok",
  );
  setNotes("doc", [
    ...state.model.limitations.map((text2) => ({
      level: "info",
      code: "LIMITATION",
      text: `文档自述限制（非验证结论）：${text2}`,
    })),
    ...(droppedStems.length
      ? [
          {
            level: "warning",
            code: "STEM_DROPPED",
            text: `文档已更换：旧 stem（行 ${droppedStems.join(", ")}）与新文档身份（id/文件名/hash）不一致，已清除，不保留旧波形。`,
          },
        ]
      : []),
  ]);
  render();
  // If audio was loaded before the JSON, re-evaluate the match now.
  if (media.main) {
    await recheckAudio();
    if (state.disposed || token !== state.tokens.json) return;
  }
}

async function onJsonFile(event) {
  const input = event.target;
  const file = input.files && input.files[0];
  input.value = ""; // allow re-selecting the SAME file (change fires again)
  if (!file) return;
  try {
    await loadJsonFile(file);
  } finally {
    state.loads.json += 1;
  }
}

async function onAudioFile(event) {
  const input = event.target;
  const file = input.files && input.files[0];
  input.value = ""; // allow re-selecting the SAME file
  if (!file) return;
  try {
    await loadAudioFile(file);
  } finally {
    state.loads.audio += 1;
  }
}

async function loadAudioFile(file) {
  const token = ++state.tokens.audio;
  if (!state.model) {
    setStatus("请先加载 JSON 结果文件（需要它的 audio 元数据来校验音频）。", "error");
    return;
  }
  setNotes("audio", []);
  setStatus(`解码音频 ${file.name}（本地，不上传）…`);
  elements.fileState.textContent = `音频文件：${file.name}（${file.size} bytes）`;
  const startedDecode = performance.now();
  let result;
  try {
    result = await media.loadMain(file, state.model.audio, {
      samplesPerBucket: 256,
      peaksWorkerUrl: new URL("./js/peaks-worker.js", import.meta.url),
    });
  } catch (error) {
    if (state.disposed || token !== state.tokens.audio) return;
    setStatus(`音频加载失败：${String(error && error.message ? error.message : error)}`, "error");
    return;
  }
  if (state.disposed || token !== state.tokens.audio || result.aborted) return; // superseded by a newer load
  resetDatasetMetrics("audio");
  state.metrics.decodeMs = performance.now() - startedDecode;
  state.peaks = result.pyramid;
  state.audio = {
    loaded: Boolean(result.pyramid),
    hashHex: result.hashHex,
    durationSeconds: result.durationSeconds,
    sampleRate: result.sampleRate,
    notes: result.notes,
  };
  const hasError = result.notes.some((note) => note.level === "error");
  if (result.failed) {
    setStatus("音频解码失败（已清除之前的可播放状态，绝不把旧文件当成新选择的文件播放）：见下方说明。", "error");
  } else {
    setStatus(
      hasError
        ? "音频已加载，但与 JSON 存在不一致（见下方说明）；时间轴仍以 JSON 秒为真值，不重定时。"
        : `音频已加载并校验（SHA-256 匹配 JSON）：${result.durationSeconds.toFixed(3)}s @ ${result.sampleRate}Hz。`,
      hasError ? "error" : "ok",
    );
  }
  setNotes("audio", result.notes);
  render();
}

async function recheckAudio() {
  if (!media.main || !state.model) return;
  const notes = checkAudioMatch({
    docAudio: state.model.audio,
    pickedName: media.main.file.name,
    hashHex: state.audio.hashHex,
    decodedDurationSeconds: state.audio.durationSeconds,
  });
  state.audio = { ...state.audio, notes };
  const hasError = notes.some((note) => note.level === "error");
  setNotes("audio", notes);
  setStatus(
    hasError
      ? "音频与新加载的 JSON 不一致（见下方说明）；时间轴仍以 JSON 秒为真值，不重定时。"
      : "音频与新加载的 JSON 匹配（SHA-256/时长核对）。",
    hasError ? "error" : "ok",
  );
  render();
}

async function onStemFiles(event) {
  const input = event.target;
  const files = [...(input.files || [])];
  input.value = ""; // allow re-selecting the SAME files
  if (files.length === 0) return;
  const token = ++state.tokens.stems;
  if (!state.model) {
    setStatus("请先加载 JSON 结果文件。", "error");
    return;
  }
  const notes = [];
  for (const file of files) {
    if (state.disposed || token !== state.tokens.stems) return;
    const picked = basename(file.name);
    // Candidates = rows whose DECLARED stem filename shares the basename
    // (only matching, never arbitrary reads). Identity is decided by SHA-256:
    // same-basename stems (e.g. SAM `target.wav`) must never grab the first row.
    const candidates = state.model.rows.filter(
      (row) => row.stem && basename(row.stem.filename).toLowerCase() === picked.toLowerCase(),
    );
    if (candidates.length === 0) {
      notes.push({
        level: "warning",
        code: "STEM_UNMATCHED",
        text: `stem 文件 ${picked} 未匹配任何 instrument.stem.filename（只按文件名匹配，不做任意读取），已忽略。`,
      });
      setNotes("stems", notes);
      continue;
    }
    const result = await media.loadStem(file, candidates, {
      samplesPerBucket: 256,
      peaksWorkerUrl: new URL("./js/peaks-worker.js", import.meta.url),
      durationSeconds: state.model.audio.durationSeconds,
    });
    if (state.disposed || token !== state.tokens.stems) return;
    notes.push(...result.notes);
    if (result.applied) {
      for (const rowId of result.matchedRows) {
        const entry = media.stems.get(rowId);
        if (entry && entry.pyramid) {
          state.stemPeaks.set(rowId, entry.pyramid);
          state.stemSampleRates.set(rowId, entry.sampleRate);
        }
      }
    }
  }
  setNotes("stems", notes);
  render();
}

function syncBpmUi() {
  if (!state.bpmState) {
    elements.bpmInput.value = "";
    elements.bpmOrigin.textContent = "";
    elements.bpmStatus.textContent = "";
    return;
  }
  const bpm = state.bpmState;
  elements.bpmInput.value = bpm.value === null ? "" : String(bpm.value);
  elements.bpmInput.placeholder = bpm.unknown ? "unknown" : "";
  elements.bpmOrigin.textContent = bpm.origin === "json" ? "来源：JSON tempo.bpm" : bpm.origin === "manual" ? "来源：手动输入" : "来源：未知（JSON tempo.bpm=null）";
  elements.bpmStatus.textContent = bpm.lastError || bpm.message;
  elements.bpmStatus.className = bpm.lastError ? "error" : "muted";
}

function applyBpmFromInput() {
  if (!state.model) {
    elements.bpmStatus.textContent = "请先加载 JSON 结果文件。";
    elements.bpmStatus.className = "error";
    return;
  }
  const before = snapshotOnsets();
  const { state: next, changed } = applyBpm(state.bpmState, elements.bpmInput.value);
  state.bpmState = next;
  syncBpmUi();
  render();
  const after = snapshotOnsets();
  if (changed && JSON.stringify(before) === JSON.stringify(after)) {
    // onsets untouched — this is the frozen guarantee.
    elements.bpmStatus.textContent = `${next.message}（onset 秒数组未改变，已验证）`;
    elements.bpmStatus.className = "muted";
  }
}

function restoreJsonBpm() {
  if (!state.model) return;
  state.bpmState = createBpmState(state.model);
  syncBpmUi();
  render();
}

function snapshotOnsets() {
  if (!state.model) return [];
  return state.model.rows.map((row) => ({ id: row.id, onsets: Array.from(row.onset) }));
}

function bindEvents() {
  elements.jsonInput.addEventListener("change", (event) => {
    onJsonFile(event).catch((error) => setStatus(`JSON 加载失败：${String(error && error.message ? error.message : error)}`, "error"));
  });
  elements.audioInput.addEventListener("change", (event) => {
    onAudioFile(event).catch((error) => setStatus(`音频加载失败：${String(error && error.message ? error.message : error)}`, "error"));
  });
  elements.stemInput.addEventListener("change", (event) => {
    onStemFiles(event)
      .catch((error) => setStatus(`stem 加载失败：${String(error && error.message ? error.message : error)}`, "error"))
      .finally(() => {
        state.loads.stems += 1;
      });
  });
  elements.playButton.addEventListener("click", () => {
    play().catch(() => {});
  });
  elements.pauseButton.addEventListener("click", pause);
  elements.seekInput.addEventListener("change", () => seekTo(Number(elements.seekInput.value)));
  elements.fitButton.addEventListener("click", fit);
  elements.resetButton.addEventListener("click", () => {
    fit();
    restoreJsonBpm();
    setStatus("已 reset：视图=全部时长，BPM=JSON 值（onset 从未改变）。");
  });
  elements.zoomInButton.addEventListener("click", () => zoomAtX(0.5, cssWidth() / 2));
  elements.zoomOutButton.addEventListener("click", () => zoomAtX(2, cssWidth() / 2));
  elements.bpmApply.addEventListener("click", applyBpmFromInput);
  elements.bpmInput.addEventListener("change", applyBpmFromInput);
  elements.bpmJson.addEventListener("click", restoreJsonBpm);

  const stack = elements.canvasStack;
  stack.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const rect = stack.getBoundingClientRect();
      const x = event.clientX - rect.left;
      if (event.shiftKey) {
        panPx(event.deltaY > 0 ? 80 : -80);
        return;
      }
      const factor = Math.exp(event.deltaY * 0.0016);
      zoomAtX(factor, x);
    },
    { passive: false },
  );

  let drag = null;
  stack.addEventListener("pointerdown", (event) => {
    drag = { x: event.clientX, moved: false, pointerId: event.pointerId };
    stack.setPointerCapture(event.pointerId);
  });
  stack.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const dx = event.clientX - drag.x;
    if (Math.abs(dx) > 3) {
      drag.moved = true;
      panPx(dx);
      drag.x = event.clientX;
    }
  });
  stack.addEventListener("pointerup", (event) => {
    if (!drag) return;
    const wasMoved = drag.moved;
    drag = null;
    if (!wasMoved) {
      const rect = stack.getBoundingClientRect();
      const x = event.clientX - rect.left;
      seekTo(xToTime(x, state.view, cssWidth()));
    }
  });
  stack.addEventListener("pointercancel", () => {
    drag = null;
  });

  window.addEventListener("keydown", (event) => {
    const tag = document.activeElement ? document.activeElement.tagName : "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (event.code === "Space") {
      event.preventDefault();
      if (state.playing) pause();
      else play().catch(() => {});
    } else if (event.code === "ArrowLeft") {
      seekTo((playheadTime() || 0) - 1);
    } else if (event.code === "ArrowRight") {
      seekTo((playheadTime() || 0) + 1);
    } else if (event.key === "+" || event.key === "=") {
      zoomAtX(0.5, cssWidth() / 2);
    } else if (event.key === "-") {
      zoomAtX(2, cssWidth() / 2);
    }
  });

  const observer = new ResizeObserver(() => {
    render();
  });
  observer.observe(stack);
  window.addEventListener("resize", render);
}

function exposeDebugApi() {
  window.__aatViewer = {
    ready: true,
    state: () => ({
      status: state.statusText,
      disposed: state.disposed,
      playing: state.playing,
      view: { ...state.view },
      bpm: state.bpmState ? { ...state.bpmState } : null,
      doc: state.model
        ? {
            rows: state.model.rows.map((row) => ({ id: row.id, label: row.label, count: row.count })),
            totalEvents: state.model.totalEvents,
            durationSeconds: state.model.audio.durationSeconds,
            audioFilename: state.model.audio.filename,
            audioSha256: state.model.audio.sha256,
            tempoBpm: state.model.tempo.bpm,
            tempoSource: state.model.tempo.source,
            tempoHasBeatOrigin: state.model.tempo.hasBeatOrigin,
          }
        : null,
      audio: {
        ...state.audio,
        resource: media.resourceState(),
        stemMapping: media.stemEntries(),
        validateWorkerActive: Boolean(validateWorker),
      },
      metrics: {
        ...state.metrics,
        frameIntervalStartToStartMs: [...state.metrics.frameIntervalStartToStartMs],
        frameFullRenderMs: [...state.metrics.frameFullRenderMs],
      },
      loads: { ...state.loads },
      lastRender: state.lastRender ? { ...state.lastRender, rows: state.lastRender.rows.map((r) => ({ ...r })) } : null,
    }),
    setView: (start, end) => setView({ start, end }),
    renderNow: () => render(),
    zoomAt: (factor, xPx) => zoomAtX(factor, xPx),
    panPx: (deltaPx) => panPx(deltaPx),
    fit: () => fit(),
    reset: () => {
      fit();
      restoreJsonBpm();
    },
    setBpm: (value) => {
      elements.bpmInput.value = String(value);
      applyBpmFromInput();
      return state.bpmState ? { ...state.bpmState } : null;
    },
    restoreJsonBpm: () => restoreJsonBpm(),
    onsetsSnapshot: () => snapshotOnsets(),
    play: () => play(),
    pause: () => pause(),
    seek: (seconds) => seekTo(seconds),
    playheadTime: () => playheadTime(),
    startFrameProbe: (count) => startFrameProbe(count),
    waveformPeakMax: () => {
      if (!state.peaks || !state.peaks.levels.length) return null;
      let max = -Infinity;
      for (const value of state.peaks.levels[0].max) if (value > max) max = value;
      return max;
    },
    pixelOf: (timeSeconds) => timeToX(timeSeconds, state.view, cssWidth()),
    timeAtPx: (xPx) => xToTime(xPx, state.view, cssWidth()),
    eventScreenPositions: (limit = 50) => {
      if (!state.model) return [];
      return state.model.rows.map((row) => {
        const items = [];
        const slice = { from: 0, to: row.count };
        const maxItems = Math.min(slice.to, limit);
        for (let index = 0; index < maxItems; index += 1) {
          items.push({
            id: row.ids[index],
            onset: row.onset[index],
            duration: Number.isFinite(row.duration[index]) ? row.duration[index] : null,
            x: timeToX(row.onset[index], state.view, cssWidth()),
          });
        }
        return {
          rowId: row.id,
          label: row.label,
          count: row.count,
          hasDuration: row.withDuration,
          items,
        };
      });
    },
    canvasMetrics: () => ({
      ruler: { width: elements.rulerCanvas.width, height: elements.rulerCanvas.height, cssWidth: cssWidth() },
      wave: { width: elements.waveCanvas.width, height: elements.waveCanvas.height },
      track: { width: elements.trackCanvas.width, height: elements.trackCanvas.height },
      dpr: window.devicePixelRatio || 1,
    }),
    rowGrid: () => currentGrid(),
    schemaVersion: () => (state.model ? state.model.schemaVersion : null),
    dispose: async () => {
      if (state.disposed) return;
      state.disposed = true;
      state.tokens.json += 1;
      state.tokens.audio += 1;
      state.tokens.stems += 1;
      if (state.rafId) cancelAnimationFrame(state.rafId);
      if (state.frameProbeId) cancelAnimationFrame(state.frameProbeId);
      state.playing = false;
      terminateValidateWorker(); // settles outstanding awaits instead of leaving them pending
      await media.dispose();
      state.peaks = null;
      state.stemPeaks.clear();
      state.stemSampleRates.clear();
    },
    resourceState: () => ({
      ...media.resourceState(),
      validateWorkerActive: Boolean(validateWorker),
      rafActive: Boolean(state.rafId || state.frameProbeId),
      objectUrlsOwnedByApp: 0,
    }),
  };
}

async function init() {
  try {
    state.schema = await loadSchema();
  } catch (error) {
    setStatus(`初始化失败：${String(error && error.message ? error.message : error)}`, "error");
    exposeDebugApi();
    return;
  }
  bindEvents();
  renderGutterRows();
  render();
  setStatus("就绪：请加载 JSON 结果文件（协议 agentic-audio-tracks/v1），然后加载音频文件。");
  exposeDebugApi();
}

init();
