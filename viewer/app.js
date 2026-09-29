// Browser entry point for the local activity viewer.
//
// Data flow: <input type=file> -> object URL (audio) / File.text() (JSON) ->
// protocol.js validation -> session.js state -> buildSnapshot() -> DOM text
// nodes + canvas.  There is no fetch, no upload and no remote asset: the
// loaded files never leave this machine.

import { createObjectUrlManager } from "./js/media.js";
import { computeOverviewBins } from "./js/overview.js";
import { describeDataKind } from "./js/protocol.js";
import {
  beginAudioFileLoad,
  captureLoopPoint,
  checkAudioCompatibility,
  clearLoop,
  createSession,
  describeSession,
  loadTrajectoryText,
  playbackRate,
  setAudioError,
  setAudioMetadata,
  setRate,
  setThreshold,
  setWindowLength,
  setWindowStart,
  toggleTrack,
} from "./js/session.js";
import { buildSnapshot } from "./js/snapshot.js";
import {
  applyLoopCorrection,
  clamp,
  formatTime,
  shiftWindow,
  windowBounds,
} from "./js/timeline.js";
import {
  drawOverviewCanvas,
  drawWindowCanvas,
  niceTimeStep,
  overviewTimeFromX,
  trackColor,
} from "./js/draw.js";

const AUDIO_ERROR_MESSAGES = {
  1: "音频加载被中止（MEDIA_ERR_ABORTED）。",
  2: "读取音频文件失败（MEDIA_ERR_NETWORK）；文件可能已被移动。",
  3: "音频解码失败（MEDIA_ERR_DECODE）；浏览器不支持该编码。",
  4: "浏览器不支持该音频格式或源（MEDIA_ERR_SRC_NOT_SUPPORTED）。",
};

const $ = (id) => document.getElementById(id);

const elements = {
  audioFile: $("audio-file"),
  trajectoryFile: $("trajectory-file"),
  audioName: $("audio-name"),
  trajectoryName: $("trajectory-name"),
  audioStatus: $("audio-status"),
  compatMessage: $("compat-message"),
  validationError: $("validation-error"),
  validationErrorMessage: $("validation-error-message"),
  validationErrorList: $("validation-error-list"),
  provenancePanel: $("provenance-panel"),
  provenance: $("provenance"),
  audio: $("audio"),
  play: $("play"),
  seek: $("seek"),
  rate: $("rate"),
  timeLocal: $("time-local"),
  timeAbs: $("time-abs"),
  timeWindow: $("time-window"),
  threshold: $("threshold"),
  thresholdValue: $("threshold-value"),
  loopA: $("loop-a"),
  loopB: $("loop-b"),
  loopClear: $("loop-clear"),
  loopInfo: $("loop-info"),
  loopError: $("loop-error"),
  windowLength: $("window-length"),
  windowFirst: $("window-first"),
  windowPrev: $("window-prev"),
  windowNext: $("window-next"),
  windowLast: $("window-last"),
  follow: $("follow"),
  tracks: $("tracks"),
  overviewCanvas: $("overview-canvas"),
  windowCanvas: $("window-canvas"),
  status: $("status"),
};

const urlManager = createObjectUrlManager({
  create: (blob) => URL.createObjectURL(blob),
  revoke: (url) => URL.revokeObjectURL(url),
});

let session = createSession();
let lastSnapshot = null;
let draggingSeek = false;
let rafHandle = null;
let overviewBins = null;
let overviewCache = { trajectory: null, duration: -1, width: -1 };
let trackRows = new Map();
let trackRowOrder = null;
let lastStatus = { message: "", kind: "" };

const audio = elements.audio;
audio.preservesPitch = true;

/* ------------------------------------------------------------------ status */

function renderStatus(message, kind = "info") {
  lastStatus = { message, kind };
  elements.status.textContent = message;
  elements.status.className = `status ${kind}`.trim();
}

function audioDuration() {
  return Number.isFinite(audio.duration) && audio.duration > 0
    ? audio.duration
    : null;
}

function audioReady() {
  return audioDuration() !== null && audio.readyState >= 1;
}

/* ------------------------------------------------------------------ render */

function render() {
  const snapshot = buildSnapshot({
    trajectory: session.trajectory,
    audioDuration: audioDuration(),
    currentTime: Number.isFinite(audio.currentTime) ? audio.currentTime : 0,
    threshold: session.threshold,
    windowLength: session.windowLength,
    windowStart: session.windowStart,
    followPlayhead: session.followPlayhead,
    hiddenTracks: session.hiddenTracks,
  });
  lastSnapshot = snapshot;

  // Transport readouts: every one derives from the single audio clock.
  elements.timeLocal.textContent = formatTime(snapshot.localTime);
  elements.timeAbs.textContent = formatTime(snapshot.absoluteTime);
  elements.timeWindow.textContent =
    `${formatTime(snapshot.window.start)} – ${formatTime(snapshot.window.end)}` +
    (snapshot.duration > 0
      ? `（第 ${Math.floor(snapshot.window.start / session.windowLength) + 1}/${Math.max(1, Math.ceil(snapshot.duration / session.windowLength))} 窗）`
      : "");

  elements.play.textContent = audio.paused ? "播放" : "暂停";
  const playable = snapshot.trajectoryLoaded && audioReady();
  elements.play.disabled = !playable;
  elements.seek.disabled = !playable;
  elements.loopA.disabled = !playable;
  elements.loopB.disabled = !playable;
  elements.loopClear.disabled = session.loop === null && session.loopDraft.start === null;

  const duration = snapshot.duration;
  elements.seek.max = String(duration > 0 ? duration : 1);
  if (!draggingSeek) {
    elements.seek.value = String(snapshot.localTime);
  }

  elements.threshold.value = String(session.threshold);
  elements.thresholdValue.textContent = session.threshold.toFixed(2);
  elements.follow.checked = session.followPlayhead;
  elements.rate.value = String(session.rate);
  elements.windowLength.value = String(session.windowLength);
  audio.playbackRate = playbackRate(session);

  renderAudioStatus();
  renderCompatibility(snapshot);
  renderLoopInfo();
  ensureTrackRows(snapshot);
  renderTrackValues(snapshot);
  renderValidationError();
  renderProvenance();
  renderCanvases(snapshot);
}

function renderAudioStatus() {
  const { fileName, duration, error } = session.audio;
  if (!fileName) {
    elements.audioStatus.textContent = "尚未选择音频文件。";
    elements.audioStatus.className = "status";
    elements.audioName.textContent = "未选择";
    return;
  }
  elements.audioName.textContent = fileName;
  if (error) {
    elements.audioStatus.textContent = error;
    elements.audioStatus.className = "status error";
    return;
  }
  if (duration === null) {
    elements.audioStatus.textContent =
      "已选择音频，等待浏览器异步读取 metadata（此时不判为时长错误）。";
    elements.audioStatus.className = "status pending";
    return;
  }
  elements.audioStatus.textContent = `音频已加载：${duration.toFixed(3)}s。`;
  elements.audioStatus.className = "status ok";
}

function renderCompatibility(snapshot) {
  if (!snapshot.trajectoryLoaded) {
    elements.compatMessage.textContent = "";
    elements.compatMessage.className = "status";
    return;
  }
  const compatibility = checkAudioCompatibility(
    session.trajectory,
    audioDuration(),
  );
  elements.compatMessage.textContent = compatibility.message || "";
  const kindByStatus = {
    pending: "pending",
    mismatch: "error",
    empty: "warn",
    ok: "ok",
    "no-trajectory": "",
  };
  elements.compatMessage.className =
    `status ${kindByStatus[compatibility.status] || ""}`.trim();
}

function renderLoopInfo() {
  const { start, end } = session.loopDraft;
  const draftText = `A=${start === null ? "–" : formatTime(start)} B=${end === null ? "–" : formatTime(end)}`;
  if (session.loop) {
    elements.loopInfo.textContent = `循环中：${formatTime(session.loop.start)} – ${formatTime(session.loop.end)}（${draftText}）`;
  } else {
    elements.loopInfo.textContent = `未启用循环（${draftText}）`;
  }
  elements.loopError.hidden = !session.loopError;
  elements.loopError.textContent = session.loopError || "";
}

function ensureTrackRows(snapshot) {
  const order = snapshot.tracks.map((track) => track.trackId).join("\u0000");
  if (order === trackRowOrder) {
    return;
  }
  trackRowOrder = order;
  trackRows = new Map();
  elements.tracks.replaceChildren();
  if (snapshot.tracks.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "当前 trajectory 没有轨迹（空数组）。";
    elements.tracks.append(empty);
    return;
  }
  for (const info of snapshot.tracks) {
    const row = document.createElement("div");
    row.className = "track-row";

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "track-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = info.visible;
    checkbox.addEventListener("change", () => {
      session = toggleTrack(session, info.trackId);
      render();
    });

    const chip = document.createElement("span");
    chip.className = "chip";
    chip.style.background = trackColor(info.index);

    const name = document.createElement("span");
    name.className = "track-id";
    name.textContent = info.trackId;

    const value = document.createElement("span");
    value.className = "track-value";

    const badge = document.createElement("span");
    badge.className = "badge";

    const nearest = document.createElement("span");
    nearest.className = "track-nearest";

    toggleLabel.append(checkbox, chip, name);
    row.append(toggleLabel, value, badge, nearest);
    trackRows.set(info.trackId, { row, value, badge, nearest, checkbox });
    elements.tracks.append(row);
  }
}

function renderTrackValues(snapshot) {
  for (const info of snapshot.tracks) {
    const refs = trackRows.get(info.trackId);
    if (!refs) {
      continue;
    }
    refs.value.textContent =
      info.value === null ? "无数据" : `p=${info.value.toFixed(3)}`;
    refs.badge.textContent = info.active ? "≥ 阈值" : "低于阈值";
    refs.badge.className = `badge ${info.active ? "active" : ""}`.trim();
    if (info.nearest) {
      const slot =
        info.nearest.slotIndex === null ? "null" : String(info.nearest.slotIndex);
      refs.nearest.textContent =
        `最近中心点 ${formatTime(info.nearest.localSeconds)}` +
        `（绝对 ${info.nearest.absoluteSeconds.toFixed(3)}s，p=${info.nearest.activity.toFixed(3)}，原始槽位 ${slot}）`;
    } else {
      refs.nearest.textContent = "当前时间不在该轨迹的中心点范围内";
    }
    refs.row.style.opacity = info.visible ? "1" : "0.45";
  }
}

function renderValidationError() {
  const error = session.trajectoryError;
  elements.trajectoryName.textContent = session.trajectoryFileName || "未选择";
  if (!error) {
    elements.validationError.hidden = true;
    elements.validationErrorList.replaceChildren();
    return;
  }
  elements.validationError.hidden = false;
  elements.validationErrorMessage.textContent = `文件 ${error.fileName}：${error.message}`;
  elements.validationErrorList.replaceChildren();
  for (const issue of error.issues) {
    const item = document.createElement("li");
    item.textContent = `${issue.path}: ${issue.message}`;
    elements.validationErrorList.append(item);
  }
}

function renderProvenance() {
  const trajectory = session.trajectory;
  elements.provenancePanel.hidden = trajectory === null;
  elements.provenance.replaceChildren();
  if (!trajectory) {
    return;
  }
  const provenance = trajectory.provenance;
  const entries = [
    ["run_id", provenance.runId],
    ["data_kind", describeDataKind(provenance.dataKind)],
    ["model_id", provenance.modelId ?? "（未提供）"],
    ["git_commit", provenance.gitCommit ?? "（未提供）"],
    ["config_hash", provenance.configHash ?? "（未提供）"],
    ["created_at_utc", provenance.createdAtUtc ?? "（未提供）"],
    ["sample_id", trajectory.sampleId ?? "（未提供）"],
    ["slots (K)", trajectory.slots === null ? "（未提供）" : String(trajectory.slots)],
    [
      "audio span",
      `track_start=${trajectory.audio.trackStartSeconds}s, duration=${trajectory.audio.durationSeconds}s`,
    ],
    [
      "trajectory 数量",
      `${trajectory.tracks.length}（track_id 是匿名身份，不是乐器名）`,
    ],
  ];
  if (trajectory.params) {
    entries.push(["params", JSON.stringify(trajectory.params)]);
  }
  if (provenance.notes) {
    entries.push(["notes", provenance.notes]);
  }
  for (const [label, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    if (label === "data_kind") {
      dd.className = `data-kind ${provenance.dataKind}`;
    }
    elements.provenance.append(dt, dd);
  }
  if (provenance.dataKind === "mock") {
    const warning = document.createElement("p");
    warning.className = "status warn";
    warning.textContent = "提示：该文件是 mock/占位数据，不能当作模型推理结果。";
    elements.provenance.append(warning);
  }
}

function ensureOverviewBins(snapshot) {
  const width = Math.max(1, Math.floor(elements.overviewCanvas.clientWidth || 300));
  const cache = overviewCache;
  if (
    overviewBins !== null &&
    cache.trajectory === session.trajectory &&
    cache.duration === snapshot.duration &&
    cache.width === width
  ) {
    return overviewBins;
  }
  overviewBins = session.trajectory
    ? computeOverviewBins(
        session.trajectory.tracks,
        session.trajectory.audio.trackStartSeconds,
        snapshot.duration,
        width,
      )
    : [];
  overviewCache = { trajectory: session.trajectory, duration: snapshot.duration, width };
  return overviewBins;
}

function renderCanvases(snapshot) {
  const bins = ensureOverviewBins(snapshot);
  const colorIndexById = new Map(
    snapshot.tracks.map((track) => [track.trackId, track.index]),
  );
  drawOverviewCanvas(elements.overviewCanvas, bins, snapshot, {
    loop: session.loop,
    visibleTrackIds: snapshot.tracks
      .filter((track) => track.visible)
      .map((track) => track.trackId),
    colorIndexById,
  });
  drawWindowCanvas(elements.windowCanvas, snapshot, { loop: session.loop });
}

/* ------------------------------------------------------------- playback */

function ensureAnimationLoop() {
  if (rafHandle !== null) {
    return;
  }
  const frame = () => {
    rafHandle = null;
    if (!audio.paused) {
      // Single shared clock: wrap the audio element itself, never a second one.
      const corrected = applyLoopCorrection(audio.currentTime, session.loop);
      if (corrected !== audio.currentTime) {
        audio.currentTime = corrected;
      }
      render();
      rafHandle = requestAnimationFrame(frame);
    } else {
      render();
    }
  };
  rafHandle = requestAnimationFrame(frame);
}

function stopAnimationLoop() {
  if (rafHandle !== null) {
    cancelAnimationFrame(rafHandle);
    rafHandle = null;
  }
  render();
}

function seekTo(seconds) {
  if (!audioReady()) {
    renderStatus("音频 metadata 未就绪，暂时不能定位。", "pending");
    return;
  }
  const duration = audioDuration();
  audio.currentTime = clamp(seconds, 0, duration);
  render();
}

/* --------------------------------------------------------------- events */

elements.audioFile.addEventListener("change", () => {
  const file = elements.audioFile.files && elements.audioFile.files[0];
  if (!file) {
    return;
  }
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  urlManager.clear();
  session = beginAudioFileLoad(session, file.name);
  const url = urlManager.set(file);
  audio.src = url;
  audio.load();
  renderStatus(`已选择音频 ${file.name}，等待浏览器读取 metadata。`, "pending");
  render();
});

audio.addEventListener("loadedmetadata", () => {
  session = setAudioMetadata(session, audio.duration);
  audio.playbackRate = playbackRate(session);
  renderStatus("音频 metadata 已就绪。", "ok");
  render();
});

audio.addEventListener("error", () => {
  const code = audio.error ? audio.error.code : 0;
  const message =
    AUDIO_ERROR_MESSAGES[code] || "音频加载失败（未知错误）。";
  audio.pause();
  urlManager.clear();
  audio.removeAttribute("src");
  session = setAudioError(session, message);
  renderStatus(message, "error");
  render();
});

audio.addEventListener("play", () => {
  ensureAnimationLoop();
});
audio.addEventListener("pause", () => {
  stopAnimationLoop();
});
audio.addEventListener("seeked", () => {
  render();
});

elements.trajectoryFile.addEventListener("change", async () => {
  const file = elements.trajectoryFile.files && elements.trajectoryFile.files[0];
  if (!file) {
    return;
  }
  audio.pause();
  let text;
  try {
    text = await file.text();
  } catch (error) {
    text = "{}";
    renderStatus(`读取轨迹文件失败：${error.message}`, "error");
  }
  session = loadTrajectoryText(session, text, file.name);
  if (session.trajectoryError) {
    renderStatus(
      `轨迹未加载：${session.trajectoryError.message}（旧轨迹已清除，不会残留显示）。`,
      "error",
    );
  } else {
    renderStatus(
      `轨迹已加载：${session.trajectory.tracks.length} 条轨迹。`,
      "ok",
    );
  }
  render();
});

elements.play.addEventListener("click", async () => {
  if (audio.paused) {
    try {
      await audio.play();
    } catch (error) {
      renderStatus(`无法开始播放：${error.message}`, "error");
    }
  } else {
    audio.pause();
  }
});

elements.seek.addEventListener("input", () => {
  draggingSeek = true;
  const value = Number(elements.seek.value);
  if (Number.isFinite(value)) {
    audio.currentTime = value;
  }
  render();
});
elements.seek.addEventListener("change", () => {
  draggingSeek = false;
  render();
});

elements.rate.addEventListener("change", () => {
  session = setRate(session, Number(elements.rate.value));
  audio.playbackRate = playbackRate(session);
  render();
});

elements.threshold.addEventListener("input", () => {
  session = setThreshold(session, Number(elements.threshold.value));
  render();
});

elements.loopA.addEventListener("click", () => {
  session = captureLoopPoint(session, "start", audio.currentTime);
  render();
});
elements.loopB.addEventListener("click", () => {
  session = captureLoopPoint(session, "end", audio.currentTime);
  render();
});
elements.loopClear.addEventListener("click", () => {
  session = clearLoop(session);
  render();
});

elements.windowLength.addEventListener("change", () => {
  session = setWindowLength(session, Number(elements.windowLength.value));
  render();
});
elements.follow.addEventListener("change", () => {
  session = { ...session, followPlayhead: elements.follow.checked };
  render();
});
elements.windowFirst.addEventListener("click", () => {
  session = setWindowStart(session, 0, false);
  render();
});
elements.windowPrev.addEventListener("click", () => {
  const snapshot = lastSnapshot;
  const start = snapshot
    ? shiftWindow(snapshot.window.start, -1, snapshot.duration, session.windowLength)
    : 0;
  session = setWindowStart(session, start, false);
  render();
});
elements.windowNext.addEventListener("click", () => {
  const snapshot = lastSnapshot;
  const start = snapshot
    ? shiftWindow(snapshot.window.start, 1, snapshot.duration, session.windowLength)
    : 0;
  session = setWindowStart(session, start, false);
  render();
});
elements.windowLast.addEventListener("click", () => {
  const snapshot = lastSnapshot;
  const duration = snapshot ? snapshot.duration : 0;
  const start = windowBounds(
    duration > 0 ? duration : 0,
    duration,
    session.windowLength,
  ).start;
  session = setWindowStart(session, start, false);
  render();
});

elements.overviewCanvas.addEventListener("click", (event) => {
  const rect = elements.overviewCanvas.getBoundingClientRect();
  const duration = lastSnapshot ? lastSnapshot.duration : 0;
  seekTo(overviewTimeFromX(event.clientX - rect.left, rect.width, duration));
});
elements.windowCanvas.addEventListener("click", (event) => {
  const snapshot = lastSnapshot;
  if (!snapshot) {
    return;
  }
  const rect = elements.windowCanvas.getBoundingClientRect();
  const ratio = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
  const span = snapshot.window.end - snapshot.window.start;
  seekTo(snapshot.window.start + clamp(ratio, 0, 1) * span);
});

const resizeObserver = new ResizeObserver(() => {
  if (lastSnapshot) {
    renderCanvases(lastSnapshot);
  }
});
resizeObserver.observe(elements.overviewCanvas);
resizeObserver.observe(elements.windowCanvas);

window.addEventListener("beforeunload", () => {
  urlManager.clear();
});

/* ------------------------------------------- automated verification hook */

// Deliberately small, read-mostly surface used by
// viewer/tools/browser-check.mjs so the real-browser check can assert the
// state that the UI displays.  It is not needed for normal use.
window.__aatViewer = {
  describe() {
    return {
      session: describeSession(session),
      audio: {
        currentTime: audio.currentTime,
        duration: audioDuration(),
        paused: audio.paused,
        readyState: audio.readyState,
        playbackRate: audio.playbackRate,
        srcIsObjectUrl:
          typeof audio.src === "string" && audio.src.startsWith("blob:"),
      },
      objectUrlEvents: urlManager.events,
      revokeCount: urlManager.revokeCount,
      status: lastStatus,
      gridStep: lastSnapshot
        ? niceTimeStep(lastSnapshot.window.end - lastSnapshot.window.start)
        : null,
    };
  },
  snapshot() {
    if (!lastSnapshot) {
      return null;
    }
    return {
      trajectoryLoaded: lastSnapshot.trajectoryLoaded,
      duration: lastSnapshot.duration,
      durationSource: lastSnapshot.durationSource,
      localTime: lastSnapshot.localTime,
      absoluteTime: lastSnapshot.absoluteTime,
      threshold: lastSnapshot.threshold,
      window: lastSnapshot.window,
      visibleTrackCount: lastSnapshot.visibleTrackCount,
      tracks: lastSnapshot.tracks.map((track) => ({
        trackId: track.trackId,
        value: track.value,
        active: track.active,
        visible: track.visible,
        intervals: track.intervals,
        windowIntervals: track.windowIntervals,
        activeSeconds: track.activeSeconds,
        nearest: track.nearest,
      })),
    };
  },
  setTime(seconds) {
    seekTo(seconds);
    return {
      localTime: lastSnapshot ? lastSnapshot.localTime : null,
      absoluteTime: lastSnapshot ? lastSnapshot.absoluteTime : null,
    };
  },
  setLoop(start, end) {
    session = captureLoopPoint(session, "start", start);
    session = captureLoopPoint(session, "end", end);
    render();
    return describeSession(session);
  },
  clearLoop() {
    session = clearLoop(session);
    render();
  },
  play() {
    return audio.play();
  },
  pause() {
    audio.pause();
  },
};

renderStatus("请选择本地音频与 trajectory JSON；文件只在本机读取。", "info");
render();
