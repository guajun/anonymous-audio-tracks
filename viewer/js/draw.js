// Canvas rendering for the local activity window and the whole-song overview.
//
// All user-controlled text is drawn with `ctx.fillText` or inserted into the
// DOM as text nodes by app.js, never as HTML.  The drawing code deliberately
// stays thin; all time/interval math lives in timeline.js/snapshot.js where it
// is covered by Node tests.

import { activityAtAbsoluteTime, formatTime } from "./timeline.js";

export const TRACK_COLORS = [
  "#4cc9f0",
  "#f72585",
  "#ffd166",
  "#06d6a0",
  "#b388ff",
  "#ff8fab",
  "#8ecae6",
  "#f4a261",
  "#a3b18a",
  "#e07a5f",
  "#00b4d8",
  "#cdb4db",
];

export function trackColor(index) {
  return TRACK_COLORS[index % TRACK_COLORS.length];
}

const BACKGROUND = "#12151c";
const LANE_BACKGROUND = "#1a1f29";
const GRID = "#2a3140";
const AXIS_TEXT = "#8b95a7";
const CURSOR = "#f8f9fa";
const LOOP_FILL = "rgba(247, 37, 133, 0.12)";

/** Nice grid step (1/2/5 * 10^n) that keeps roughly `targetLines` lines. */
export function niceTimeStep(spanSeconds, targetLines = 8) {
  if (!Number.isFinite(spanSeconds) || spanSeconds <= 0 || targetLines <= 0) {
    return 1;
  }
  const raw = spanSeconds / targetLines;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const factor of [1, 2, 5, 10]) {
    if (magnitude * factor >= raw) {
      return magnitude * factor;
    }
  }
  return magnitude * 10;
}

function setupCanvas(canvas) {
  const cssWidth = Math.max(1, Math.floor(canvas.clientWidth || canvas.width));
  const cssHeight = Math.max(1, Math.floor(canvas.clientHeight || canvas.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const pixelWidth = Math.floor(cssWidth * dpr);
  const pixelHeight = Math.floor(cssHeight * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  return { ctx, width: cssWidth, height: cssHeight };
}

function lowerBound(times, target) {
  let low = 0;
  let high = times.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (times[middle] < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

function drawGrid(ctx, xFor, bounds, height, top) {
  const span = bounds.end - bounds.start;
  const step = niceTimeStep(span);
  ctx.font = "10px system-ui, sans-serif";
  ctx.textBaseline = "top";
  const first = Math.ceil(bounds.start / step) * step;
  for (let time = first; time <= bounds.end + 1e-9; time += step) {
    const x = xFor(time);
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.fillStyle = AXIS_TEXT;
    ctx.fillText(formatTime(time), x + 3, 2);
  }
}

function centeredText(ctx, text, width, height) {
  ctx.fillStyle = AXIS_TEXT;
  ctx.font = "13px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
  ctx.textAlign = "left";
}

/**
 * Draw the local window for every visible track.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object} snapshot from buildSnapshot()
 * @param {{loop?: object|null}} options
 */
export function drawWindowCanvas(canvas, snapshot, options = {}) {
  const { loop = null } = options;
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, width, height);

  const bounds = snapshot.window;
  const pad = 6;
  const top = 16;
  const plotWidth = Math.max(1, width - pad * 2);
  const span = Math.max(bounds.end - bounds.start, 1e-6);
  const xFor = (time) => pad + ((time - bounds.start) / span) * plotWidth;

  drawGrid(ctx, xFor, bounds, height, top);

  if (!snapshot.trajectoryLoaded) {
    centeredText(ctx, "尚未加载 trajectory JSON", width, height);
    return;
  }
  const visible = snapshot.tracks.filter((track) => track.visible);
  if (visible.length === 0) {
    centeredText(ctx, "所有来源都已隐藏", width, height);
    return;
  }

  const laneHeight = Math.max(16, (height - top - 6) / visible.length - 4);
  const trackStart = snapshot.trackStart;

  visible.forEach((trackInfo, laneIndex) => {
    const laneTop = top + laneIndex * (laneHeight + 4) + 2;
    const laneBottom = laneTop + laneHeight;
    const yFor = (value) =>
      laneBottom - Math.min(Math.max(value, 0), 1) * (laneBottom - laneTop);

    ctx.fillStyle = LANE_BACKGROUND;
    ctx.fillRect(pad, laneTop, plotWidth, laneHeight);

    // Threshold intervals (interpolated activity >= threshold, local time).
    ctx.fillStyle = "rgba(76, 201, 240, 0.18)";
    for (const [start, end] of trackInfo.windowIntervals) {
      const x = xFor(start);
      const endX = xFor(end);
      ctx.fillRect(x, laneTop, Math.max(endX - x, 1), laneHeight);
    }

    // Activity curve, clipped to the window plus interpolated edge points.
    const times = trackInfo.centerTimes;
    const startAbsolute = bounds.start + trackStart;
    const endAbsolute = bounds.end + trackStart;
    ctx.beginPath();
    let started = false;
    const edgeStart = activityAtAbsoluteTime(trackInfo, startAbsolute);
    if (edgeStart !== null) {
      ctx.moveTo(xFor(bounds.start), yFor(edgeStart));
      started = true;
    }
    let index = lowerBound(times, startAbsolute);
    for (; index < times.length; index += 1) {
      const absolute = times[index];
      if (absolute > endAbsolute) {
        break;
      }
      const x = xFor(absolute - trackStart);
      const y = yFor(trackInfo.activity[index]);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    const edgeEnd = activityAtAbsoluteTime(trackInfo, endAbsolute);
    if (edgeEnd !== null && started) {
      ctx.lineTo(xFor(bounds.end), yFor(edgeEnd));
    }
    ctx.strokeStyle = trackColor(trackInfo.index);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Threshold line.
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(248, 249, 250, 0.45)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, yFor(snapshot.threshold));
    ctx.lineTo(width - pad, yFor(snapshot.threshold));
    ctx.stroke();
    ctx.restore();

    // Current sample marker.
    if (trackInfo.value !== null) {
      ctx.fillStyle = trackInfo.active
        ? trackColor(trackInfo.index)
        : "rgba(200, 205, 215, 0.8)";
      ctx.beginPath();
      ctx.arc(xFor(snapshot.localTime), yFor(trackInfo.value), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  if (loop) {
    const startX = xFor(loop.start);
    const endX = xFor(loop.end);
    ctx.fillStyle = LOOP_FILL;
    ctx.fillRect(
      Math.max(startX, pad),
      top,
      Math.max(Math.min(endX, width - pad) - Math.max(startX, pad), 0),
      height - top,
    );
    ctx.save();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "#f72585";
    ctx.lineWidth = 1;
    for (const x of [startX, endX]) {
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    ctx.restore();
    ctx.fillStyle = "#f72585";
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText("A", startX + 2, height - 12);
    ctx.fillText("B", endX + 2, height - 12);
  }

  // Shared clock cursor, drawn last so it stays visible.
  const cursorX = xFor(snapshot.localTime);
  ctx.strokeStyle = CURSOR;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cursorX, top);
  ctx.lineTo(cursorX, height);
  ctx.stroke();
}

/**
 * Draw the whole-song overview from cached bins.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{trackId: string, max: Float32Array}>} bins computeOverviewBins()
 * @param {object} snapshot buildSnapshot()
 * @param {{loop?: object|null, visibleTrackIds?: string[]|null, colorIndexById?: Map<string, number>}} options
 */
export function drawOverviewCanvas(canvas, bins, snapshot, options = {}) {
  const { loop = null, visibleTrackIds = null, colorIndexById = null } = options;
  const { ctx, width, height } = setupCanvas(canvas);
  ctx.fillStyle = BACKGROUND;
  ctx.fillRect(0, 0, width, height);

  const duration = snapshot.duration > 0 ? snapshot.duration : 1;
  const visible = visibleTrackIds ? new Set(visibleTrackIds) : null;
  const lanes = bins.filter((bin) => !visible || visible.has(bin.trackId));
  if (lanes.length === 0) {
    centeredText(ctx, "整曲概览（加载轨迹后可导航）", width, height);
    return;
  }

  const laneHeight = Math.max(6, height / lanes.length - 2);
  if (loop) {
    const x0 = (loop.start / duration) * width;
    const x1 = (loop.end / duration) * width;
    ctx.fillStyle = LOOP_FILL;
    ctx.fillRect(x0, 0, Math.max(x1 - x0, 1), height);
  }

  lanes.forEach((bin, laneIndex) => {
    const laneTop = laneIndex * (laneHeight + 2) + 1;
    const laneBottom = laneTop + laneHeight;
    const colorIndex = colorIndexById
      ? (colorIndexById.get(bin.trackId) ?? laneIndex)
      : laneIndex;
    ctx.beginPath();
    for (let column = 0; column < bin.max.length; column += 1) {
      const x = (column / bin.max.length) * width;
      const y = laneBottom - bin.max[column] * (laneBottom - laneTop);
      if (column === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.strokeStyle = trackColor(colorIndex);
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  // Current page window rectangle.
  const windowStartX = (snapshot.window.start / duration) * width;
  const windowEndX = (snapshot.window.end / duration) * width;
  ctx.strokeStyle = "rgba(248, 249, 250, 0.5)";
  ctx.lineWidth = 1;
  ctx.strokeRect(
    windowStartX + 0.5,
    0.5,
    Math.max(windowEndX - windowStartX - 1, 1),
    height - 1,
  );

  // Shared clock cursor.
  const cursorX = (snapshot.localTime / duration) * width;
  ctx.strokeStyle = CURSOR;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cursorX, 0);
  ctx.lineTo(cursorX, height);
  ctx.stroke();
}

/** Convert an overview canvas x-position to a local seek time. */
export function overviewTimeFromX(x, cssWidth, duration) {
  if (!Number.isFinite(cssWidth) || cssWidth <= 0 || !Number.isFinite(duration)) {
    return 0;
  }
  const ratio = Math.min(Math.max(x / cssWidth, 0), 1);
  return ratio * duration;
}
