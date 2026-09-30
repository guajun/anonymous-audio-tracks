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
  stemPeaks: new Map(), // row.index -> pyramid
  stemSampleRates: new Map(),
  audio: { loaded: false, hashHex: null, durationSeconds: null, sampleRate: null, notes: [] },
  metrics: {
    validateMs: null,
    decodeMs: null,
    peaksMs: null,
    firstRenderMs: null,
    lastRenderMs: null,
    frameDeltasMs: [],
  },
  lastRender: null,
  statusText: "等待加载 JSON 结果文件。",
  playing: false,
  rafId: null,
  disposed: false,
  schema: null,
  loads: { json: 0, audio: 0 },
};

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
    item.textContent = note.text;
    elements.notes.appendChild(item);
  }
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
  for (const [index, pyramid] of state.stemPeaks) {
    rowEnvelopes.set(index, envelopeFor(pyramid, state.stemSampleRates.get(index) || 44100, state.view, Math.max(1, Math.round(width))));
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
  elements.viewWindow.textContent = `视图 ${state.view.start.toFixed(3)}s – ${state.view.end.toFixed(3)}s（总 ${(
    state.model.audio.durationSeconds || 0
  ).toFixed(3)}s）`;
  updateMetricsText();
  return stats;
}

function updateMetricsText() {
  const m = state.metrics;
  const frames = m.frameDeltasMs;
  const frameInfo = frames.length
    ? `帧间隔 n=${frames.length} avg=${(frames.reduce((a, b) => a + b, 0) / frames.length).toFixed(2)}ms max=${Math.max(
        ...frames,
      ).toFixed(2)}ms`
    : "帧间隔 n=0";
  const r = state.lastRender || {};
  elements.metrics.textContent =
    `validate=${m.validateMs === null ? "-" : m.validateMs.toFixed(1)}ms · decode=${
      m.decodeMs === null ? "-" : m.decodeMs.toFixed(1)
    }ms · peaks=${m.peaksMs === null ? "-" : m.peaksMs.toFixed(1)}ms · render=${
      m.lastRenderMs === null ? "-" : m.lastRenderMs.toFixed(2)
    }ms · visible=${r.visibleEvents || 0} · drawn=${r.drawnEvents || 0} · aggregatedRows=${r.aggregatedRows || 0} · ${frameInfo} · dpr=${
      window.devicePixelRatio || 1
    }`;
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
  state.metrics.frameDeltasMs = [];
  let last = performance.now();
  let remaining = count;
  const step = () => {
    if (state.disposed) return;
    const now = performance.now();
    state.metrics.frameDeltasMs.push(now - last);
    last = now;
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
  const worker = getValidateWorker();
  const started = performance.now();
  if (worker) {
    return new Promise((resolve) => {
      const id = `validate-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const onMessage = (event) => {
        if (event.data && event.data.id === id) {
          worker.removeEventListener("message", onMessage);
          resolve({ ...event.data, elapsed: performance.now() - started });
        }
      };
      worker.addEventListener("message", onMessage);
      worker.postMessage({ id, text, schema: state.schema, source });
    });
  }
  const result = parseAndValidate(text, state.schema, { source });
  return { report: result.report, data: result.data, elapsed: performance.now() - started };
}

async function onJsonFile(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    await loadJsonFile(file);
  } finally {
    state.loads.json += 1;
  }
}

async function loadJsonFile(file) {
  renderIssues([]);
  renderNotes([]);
  if (file.size > BIG_FILE_BYTES) {
    setStatus(`JSON 文件较大（${(file.size / 1024 / 1024).toFixed(1)}MB）：读取与校验在 Web Worker 中异步进行…`);
  } else {
    setStatus(`读取 ${file.name}…`);
  }
  elements.fileState.textContent = `JSON 文件：${file.name}（${file.size} bytes）`;
  let text;
  try {
    text = await file.text();
  } catch (error) {
    setStatus(`读取 JSON 失败：${String(error && error.message ? error.message : error)}`, "error");
    return;
  }
  const { report, data, elapsed } = await validateTextAsync(text, file.name);
  state.metrics.validateMs = elapsed;
  if (!report.ok) {
    setStatus(`JSON 未通过 agentic-audio-tracks/v1 校验（${report.issues.length} 个问题）：不绘制错误数据。`, "error");
    renderIssues(report.issues);
    updateMetricsText();
    return;
  }
  state.doc = data;
  state.model = buildModel(data);
  state.bpmState = createBpmState(state.model);
  state.view = fitWindow(state.model.audio.durationSeconds);
  syncBpmUi();
  renderGutterRows();
  const emptyRows = state.model.rows.filter((row) => row.count === 0).length;
  setStatus(
    `已加载 ${file.name}：${state.model.rows.length} 个乐器条目（同类不同 id 不合并）、${state.model.totalEvents} 个事件` +
      (emptyRows ? `、${emptyRows} 行无事件` : "") +
      `，音频声明 ${state.model.audio.durationSeconds}s。`,
    "ok",
  );
  renderNotes(
    state.model.limitations.map((text2) => ({
      level: "info",
      code: "LIMITATION",
      text: `限制说明：${text2}`,
    })),
  );
  render();
  // If audio was loaded before the JSON, re-evaluate the match now.
  if (media.main) {
    await recheckAudio();
  }
}

async function onAudioFile(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    await loadAudioFile(file);
  } finally {
    state.loads.audio += 1;
  }
}

async function loadAudioFile(file) {
  if (!state.model) {
    setStatus("请先加载 JSON 结果文件（需要它的 audio 元数据来校验音频）。", "error");
    return;
  }
  renderNotes([]);
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
    setStatus(`音频加载失败：${String(error && error.message ? error.message : error)}`, "error");
    return;
  }
  state.metrics.decodeMs = performance.now() - startedDecode;
  state.metrics.peaksMs = null;
  state.peaks = result.pyramid;
  state.audio = {
    loaded: Boolean(result.pyramid),
    hashHex: result.hashHex,
    durationSeconds: result.durationSeconds,
    sampleRate: result.sampleRate,
    notes: result.notes,
  };
  const hasError = result.notes.some((note) => note.level === "error");
  setStatus(
    hasError
      ? "音频已加载，但与 JSON 存在不一致（见下方说明）；时间轴仍以 JSON 秒为真值，不重定时。"
      : `音频已加载并校验（SHA-256 匹配 JSON）：${result.durationSeconds.toFixed(3)}s @ ${result.sampleRate}Hz。`,
    hasError ? "error" : "ok",
  );
  renderNotes(result.notes);
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
  renderNotes(notes);
  setStatus(
    hasError
      ? "音频与新加载的 JSON 不一致（见下方说明）；时间轴仍以 JSON 秒为真值，不重定时。"
      : "音频与新加载的 JSON 匹配（SHA-256/时长核对）。",
    hasError ? "error" : "ok",
  );
  render();
}

async function onStemFiles(event) {
  const files = [...(event.target.files || [])];
  if (files.length === 0) return;
  if (!state.model) {
    setStatus("请先加载 JSON 结果文件。", "error");
    return;
  }
  const notes = [];
  for (const file of files) {
    const picked = basename(file.name);
    const row = state.model.rows.find((entry) => entry.stem && basename(entry.stem.filename).toLowerCase() === picked.toLowerCase());
    if (!row) {
      notes.push({
        level: "warning",
        code: "STEM_UNMATCHED",
        text: `stem 文件 ${picked} 未匹配任何 instrument.stem.filename（只按文件名匹配，不做任意读取），已忽略。`,
      });
      continue;
    }
    const result = await media.loadStem(file, row.stem, {
      samplesPerBucket: 256,
      peaksWorkerUrl: new URL("./js/peaks-worker.js", import.meta.url),
    });
    if (result.pyramid) {
      state.stemPeaks.set(row.index, result.pyramid);
      state.stemSampleRates.set(row.index, result.sampleRate);
    }
    notes.push(...result.notes);
    notes.push({
      level: "info",
      code: "STEM_OK",
      text: `stem ${picked} → 行 id=${row.id}（${row.label}）：已绘制该行波形。`,
    });
  }
  renderNotes(notes);
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
    onStemFiles(event).catch((error) => setStatus(`stem 加载失败：${String(error && error.message ? error.message : error)}`, "error"));
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
        validateWorkerActive: Boolean(validateWorker),
      },
      metrics: { ...state.metrics, frameDeltasMs: [...state.metrics.frameDeltasMs] },
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
    rowGrid: () => {
      if (!state.model) return null;
      return beatGrid(state.bpmState, state.model.tempo.hasBeatOrigin ? state.model.tempo.beatOriginSeconds : 0, state.view.start, state.view.end, 64);
    },
    schemaVersion: () => (state.model ? state.model.schemaVersion : null),
    dispose: async () => {
      if (state.disposed) return;
      state.disposed = true;
      if (state.rafId) cancelAnimationFrame(state.rafId);
      if (state.frameProbeId) cancelAnimationFrame(state.frameProbeId);
      state.playing = false;
      if (validateWorker) {
        validateWorker.terminate();
        validateWorker = null;
      }
      await media.dispose();
      state.peaks = null;
      state.stemPeaks.clear();
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
