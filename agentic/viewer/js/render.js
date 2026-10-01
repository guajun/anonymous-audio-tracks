// Canvas renderer with viewport culling + event LOD.
//
// Performance contract (issue #34):
//   - only visible events are drawn (binary search over sorted onset arrays),
//   - rows with too many visible events switch to a per-pixel density
//     aggregation instead of drawing one primitive per event,
//   - waveforms come from the min/max peak pyramid (never per sample),
//   - no DOM node is created per event or per sample (labels are one DOM node
//     per instrument row, drawn/synced by app.js).

import { peakColumns } from "./peaks.js";
import { beatGrid, rowVisibleSlice } from "./model.js";
import { timeToX } from "./timeline.js";

export const DEFAULT_ROW_BUDGET = 4000;
export const DEFAULT_TOTAL_BUDGET = 20000;

export const ROW_COLORS = [
  "#4f8ef7",
  "#f7a24f",
  "#43c59e",
  "#c86bf2",
  "#f2555a",
  "#8d9eff",
  "#d4b14f",
  "#3fb6c9",
];

const TIME_STEPS = [
  0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200,
];

export function niceTimeStep(window, widthPx) {
  const span = window.end - window.start;
  const target = Math.max(1, Math.floor(widthPx / 110)); // ~110px per label
  const raw = span / target;
  for (const step of TIME_STEPS) {
    if (step >= raw) return step;
  }
  return TIME_STEPS[TIME_STEPS.length - 1];
}

function formatSeconds(value, step) {
  if (step >= 1) return `${value.toFixed(value % 1 === 0 ? 0 : 1)}s`;
  return `${value.toFixed(step >= 0.1 ? 2 : 3)}s`;
}

/** Draw the top time ruler (seconds of the audio file). */
export function renderRuler(ctx, view, options) {
  const { width, height } = options;
  ctx.save();
  ctx.fillStyle = "#151a23";
  ctx.fillRect(0, 0, width, height);
  const step = niceTimeStep(view, width);
  const first = Math.ceil(view.start / step) * step;
  ctx.strokeStyle = "#2c3648";
  ctx.fillStyle = "#93a1b5";
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "top";
  ctx.lineWidth = 1;
  for (let time = first; time <= view.end + 1e-9; time += step) {
    const x = Math.round(timeToX(time, view, width)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, height - 8);
    ctx.lineTo(x, height);
    ctx.stroke();
    if (x >= 0 && x <= width - 2) {
      ctx.fillText(formatSeconds(time, step), x + 3, 3);
    }
  }
  ctx.restore();
}

function drawGrid(ctx, model, bpmState, view, top, bottom, width) {
  const origin = model.tempo.hasBeatOrigin ? model.tempo.beatOriginSeconds : 0;
  const grid = beatGrid(bpmState, origin, view.start, view.end, Math.floor(width / 4) + 2);
  if (grid.beats.length === 0) return grid;
  ctx.save();
  ctx.strokeStyle = "rgba(120, 150, 200, 0.20)";
  ctx.lineWidth = 1;
  for (const time of grid.beats) {
    const x = Math.round(timeToX(time, view, width)) + 0.5;
    if (x < -1 || x > width + 1) continue;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
  }
  ctx.restore();
  return grid;
}

/**
 * Pixel geometry of one event inside the viewport.
 *
 * Sustained events (with duration) are CLIPPED to the viewport edges instead of
 * being dropped when the onset is off-screen; onset-only events are single
 * ticks that only count as visible when their onset is inside the window.
 *
 * @returns {{visible: boolean, x0: number, x1: number, left: number, right: number, width: number, clipped: boolean}}
 */
export function eventPixelRect(onset, duration, view, width) {
  const hasDuration = Number.isFinite(duration) && duration > 0;
  const x0 = timeToX(onset, view, width);
  const x1 = hasDuration ? timeToX(onset + duration, view, width) : x0;
  const visible = hasDuration
    ? onset <= view.end && onset + duration >= view.start
    : onset >= view.start && onset <= view.end;
  const rawLeft = hasDuration ? Math.min(x0, x1) : x0;
  const rawRight = hasDuration ? Math.max(x0, x1) : x0;
  const left = Math.min(Math.max(rawLeft, 0), width);
  const right = Math.min(Math.max(rawRight, 0), width);
  const clipped = left !== rawLeft || right !== rawRight;
  return { visible, x0, x1, left, right, width: Math.max(0, right - left), clipped };
}

function drawRowWaveform(ctx, envelope, top, height, width) {
  if (!envelope || !envelope.min || envelope.min.length === 0) return;
  const mid = top + height / 2;
  const scale = height * 0.45;
  ctx.save();
  ctx.fillStyle = "rgba(120, 170, 255, 0.22)";
  const columns = envelope.min.length;
  for (let column = 0; column < columns; column += 1) {
    const y0 = mid - envelope.max[column] * scale;
    const y1 = mid - envelope.min[column] * scale;
    ctx.fillRect(column, Math.min(y0, y1), 1, Math.max(1, Math.abs(y1 - y0)));
  }
  ctx.restore();
}

function drawRowEvents(ctx, row, view, top, height, width, color, budget, stats) {
  const slice = rowVisibleSlice(row, view.start, view.end);
  stats.candidateEvents += slice.count;
  if (slice.count === 0) return;

  if (slice.count > budget) {
    // LOD: per-pixel density histogram (truly visible events only) instead of
    // one primitive per event.
    const columns = new Uint32Array(Math.max(1, Math.ceil(width)));
    for (let index = slice.from; index < slice.to; index += 1) {
      const rect = eventPixelRect(row.onset[index], row.duration[index], view, width);
      if (!rect.visible) continue;
      stats.visibleEvents += 1;
      const column = Math.min(columns.length - 1, Math.max(0, Math.floor(rect.left)));
      columns[column] += 1;
    }
    let maxCount = 1;
    for (let column = 0; column < columns.length; column += 1) {
      if (columns[column] > maxCount) maxCount = columns[column];
    }
    const barHeight = height * 0.72;
    const baseY = top + height * 0.86;
    ctx.save();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.55;
    for (let column = 0; column < columns.length; column += 1) {
      const count = columns[column];
      if (count === 0) continue;
      const h = Math.max(1, (count / maxCount) * barHeight);
      ctx.fillRect(column, baseY - h, 1, h);
    }
    ctx.restore();
    stats.drawnBars += columns.length;
    stats.aggregatedRows += 1;
    return;
  }

  const barHeight = height * 0.62;
  const baseY = top + height * 0.82;
  ctx.save();
  ctx.fillStyle = color;
  for (let index = slice.from; index < slice.to; index += 1) {
    const onset = row.onset[index];
    const duration = row.duration[index];
    const rect = eventPixelRect(onset, duration, view, width);
    if (!rect.visible) continue;
    stats.visibleEvents += 1;
    if (Number.isFinite(duration)) {
      // sustained event: clipped interval, never dropped at the edges
      const drawWidth = Math.max(1.5, rect.width);
      ctx.fillRect(rect.left, baseY - barHeight, drawWidth, barHeight);
      if (rect.clipped) stats.clippedEvents += 1;
    } else {
      ctx.fillRect(rect.x0 - 1, baseY - barHeight, 2, barHeight);
    }
    stats.drawnEvents += 1;
  }
  ctx.restore();
}

/**
 * Render the instrument rows + beat grid + playhead.
 * @returns {object} render statistics (visible vs drawn events)
 */
export function renderTracks(ctx, model, view, options) {
  const started = options.now ? options.now() : performance.now();
  const {
    width,
    height,
    rulerHeight,
    rowHeight,
    bpmState,
    playheadTime = null,
    rowBudget = DEFAULT_ROW_BUDGET,
    totalBudget = DEFAULT_TOTAL_BUDGET,
  } = options;

  const stats = {
    candidateEvents: 0,
    visibleEvents: 0,
    drawnEvents: 0,
    drawnBars: 0,
    clippedEvents: 0,
    aggregatedRows: 0,
    rows: [],
    renderMs: 0,
  };

  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, width, height);
  ctx.restore();

  const rowsTop = rulerHeight;
  const rowsBottom = rowsTop + model.rows.length * rowHeight;

  // Row backgrounds and separators.
  ctx.save();
  model.rows.forEach((row, index) => {
    const y = rowsTop + index * rowHeight;
    ctx.fillStyle = index % 2 === 0 ? "#11161f" : "#0f141c";
    ctx.fillRect(0, y, width, rowHeight);
    ctx.strokeStyle = "#1d2634";
    ctx.beginPath();
    ctx.moveTo(0, y + rowHeight - 0.5);
    ctx.lineTo(width, y + rowHeight - 0.5);
    ctx.stroke();
  });
  ctx.restore();

  const grid = drawGrid(ctx, model, bpmState, view, rowsTop, Math.max(rowsBottom, rowsTop), width);

  const perRowBudget = Math.min(rowBudget, Math.max(1, Math.floor(totalBudget / Math.max(1, model.rows.length))));
  model.rows.forEach((row, index) => {
    const y = rowsTop + index * rowHeight;
    if (options.rowEnvelopes && options.rowEnvelopes.has(row.id)) {
      drawRowWaveform(ctx, options.rowEnvelopes.get(row.id), y, rowHeight, width);
    }
    const before = stats.drawnEvents;
    const beforeVisible = stats.visibleEvents;
    drawRowEvents(ctx, row, view, y, rowHeight, width, ROW_COLORS[index % ROW_COLORS.length], perRowBudget, stats);
    const rowStats = {
      id: row.id,
      label: row.label,
      candidates: rowVisibleSlice(row, view.start, view.end).count,
      visible: stats.visibleEvents - beforeVisible,
      drawn: stats.drawnEvents - before,
    };
    stats.rows.push(rowStats);
    if (row.count === 0) {
      ctx.save();
      ctx.fillStyle = "#5b6675";
      ctx.font = "11px system-ui, sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillText("（无事件）", 8, y + rowHeight / 2);
      ctx.restore();
    }
  });

  if (typeof playheadTime === "number" && Number.isFinite(playheadTime)) {
    const x = Math.round(timeToX(playheadTime, view, width)) + 0.5;
    if (x >= 0 && x <= width) {
      ctx.save();
      ctx.strokeStyle = "#ff5252";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, Math.max(rowsBottom, rulerHeight));
      ctx.stroke();
      ctx.restore();
    }
  }

  stats.gridBeats = grid.beats.length;
  stats.renderMs = (options.now ? options.now() : performance.now()) - started;
  return stats;
}

/**
 * Render one waveform lane from a min/max envelope (`peakColumns` output).
 * @param {{min: Float32Array, max: Float32Array}} envelope
 */
export function renderWaveformLane(ctx, envelope, view, options) {
  const started = options.now ? options.now() : performance.now();
  const { width, height, playheadTime = null, color = "#5fa8ff", label = "" } = options;
  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0b0f15";
  ctx.fillRect(0, 0, width, height);
  const mid = height / 2;
  if (envelope && envelope.min && envelope.min.length > 0) {
    ctx.fillStyle = color;
    const columns = envelope.min.length;
    for (let column = 0; column < columns; column += 1) {
      const lo = envelope.min[column];
      const hi = envelope.max[column];
      const y0 = mid - hi * (mid - 4);
      const y1 = mid - lo * (mid - 4);
      ctx.fillRect(column, Math.min(y0, y1), 1, Math.max(1, Math.abs(y1 - y0)));
    }
  } else {
    ctx.fillStyle = "#5b6675";
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.fillText(label || "（未加载音频波形）", 8, mid);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.beginPath();
  ctx.moveTo(0, mid + 0.5);
  ctx.lineTo(width, mid + 0.5);
  ctx.stroke();
  if (typeof playheadTime === "number" && Number.isFinite(playheadTime)) {
    const x = Math.round(timeToX(playheadTime, view, width)) + 0.5;
    if (x >= 0 && x <= width) {
      ctx.strokeStyle = "#ff5252";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }
  ctx.restore();
  return { renderMs: (options.now ? options.now() : performance.now()) - started };
}

/** Compute a waveform envelope for a lane (thin wrapper used by app.js). */
export function envelopeFor(pyramid, sampleRate, view, width) {
  return peakColumns(pyramid, sampleRate, view.start, view.end, width);
}
