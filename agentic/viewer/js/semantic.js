// Cross-field semantic validation: browser port of
// `agentic/schema/agentic_schema/semantic.py` (issue #32 frozen rules).
//
// Stable error codes (see `schema/semantic_rules.json`):
//   E_BOOL_NUMBER E_FINITE E_ID_UNIQUE E_PATH E_TEMPO E_TIME_DURATION
//   E_TIME_ORDER E_TIME_RANGE
// plus the structural `pattern` code handled in structure.js.

import { MAX_FLOAT, pointerOf } from "./strict-json.js";
import { cmpText, hasOwn, stringLength } from "./structure.js";

export const SEMANTIC_CODES = [
  "E_BOOL_NUMBER",
  "E_FINITE",
  "E_ID_UNIQUE",
  "E_PATH",
  "E_RANGE",
  "E_TEMPO",
  "E_TIME_DURATION",
  "E_TIME_ORDER",
  "E_TIME_RANGE",
];

/** Time comparison tolerance (seconds), frozen. */
export const TIME_TOLERANCE = 1e-9;

const RESERVED_NAMES = new Set(["CON", "PRN", "AUX", "NUL"]);
for (let index = 1; index <= 9; index += 1) {
  RESERVED_NAMES.add(`COM${index}`);
  RESERVED_NAMES.add(`LPT${index}`);
}
const FORBIDDEN_CHARS = new Set(['<', '>', ':', '"', '|', '?', '*']);
const DRIVE_RE = /^[A-Za-z]:/;
const URL_RE = /^[A-Za-z][A-Za-z0-9+.\-]*:\/\//;

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/** Own-property read: inherited Object.prototype members never count (Python dict parity). */
function own(node, key) {
  if (node === null || typeof node !== "object" || !hasOwn(node, key)) return undefined;
  return node[key];
}

function numberRepr(value) {
  const text = String(value);
  return text.length <= 64 ? text : `${text.slice(0, 64)}...`;
}

/**
 * Path safety (E_PATH, frozen rules). Returns the reason when unsafe, else null.
 * Only safe relative paths are allowed; they are matched against a local audio
 * root by the consumer, never fetched.
 */
export function unsafePathReason(value) {
  if (value === "") return "路径为空";
  const length = stringLength(value);
  if (length > 512) return `路径超长（${length} > 512）`;
  for (const ch of value) {
    const code = ch.codePointAt(0);
    if (code < 32 || code === 127) return `包含控制字符 ${JSON.stringify(ch)}`;
  }
  if (value.includes("\\")) return "不允许反斜杠（UNC/Windows 分隔符）";
  if (value.startsWith("/")) return "不允许绝对路径（/ 开头）";
  if (DRIVE_RE.test(value)) return "不允许盘符绝对路径（如 C:）";
  if (value.startsWith("//")) return "不允许 UNC 路径（// 开头）";
  if (URL_RE.test(value)) return "不允许 URL（scheme://）";
  const bad = [...new Set([...value].filter((ch) => FORBIDDEN_CHARS.has(ch)))].sort();
  if (bad.length > 0) return `包含非法字符 ${JSON.stringify(bad)}`;
  for (const segment of value.split("/")) {
    if (segment === "") return "存在空路径段（多余或结尾的 /）";
    if (segment === "." || segment === "..") return `不允许路径逃逸段 ${JSON.stringify(segment)}`;
    if (segment !== segment.trim() || segment.endsWith(".")) {
      return `路径段首尾空格或结尾点不安全：${JSON.stringify(segment)}`;
    }
    const stem = segment.split(".")[0].toUpperCase();
    if (RESERVED_NAMES.has(stem)) return `Windows 保留设备名不安全：${JSON.stringify(segment)}`;
  }
  return null;
}

class Semantic {
  constructor(data) {
    this.data = data;
    this.issues = [];
  }

  add(code, parts, message) {
    this.issues.push({ layer: "semantic", code, pointer: pointerOf(parts), message });
  }

  // ---- fallback: finite numbers / booleans pretending to be numbers ----
  walkFinite(node, parts) {
    if (typeof node === "boolean") return;
    if (typeof node === "number") {
      if (!Number.isFinite(node) || Math.abs(node) > MAX_FLOAT) {
        this.add(
          "E_FINITE",
          parts,
          `数字必须是有限数且在 float64 可互操作范围内（|x| <= ${MAX_FLOAT}），得到 ${numberRepr(node)}`,
        );
      }
      return;
    }
    if (Array.isArray(node)) {
      node.forEach((value, index) => this.walkFinite(value, parts.concat([index])));
      return;
    }
    if (node !== null && typeof node === "object") {
      for (const [key, value] of Object.entries(node)) {
        this.walkFinite(value, parts.concat([key]));
      }
    }
  }

  checkNumber(node, key, parts) {
    if (node === null || typeof node !== "object" || !hasOwn(node, key)) return null;
    const value = node[key];
    if (typeof value === "boolean") {
      this.add("E_BOOL_NUMBER", parts.concat([key]), `${key} 必须是数字，布尔值不得冒充数字`);
      return null;
    }
    if (!isNumber(value)) return null; // type errors are reported by the structural layer
    return value;
  }

  checkConfidence(node, parts) {
    const value = this.checkNumber(node, "confidence", parts);
    if (value === null) return;
    if (!(value >= 0 && value <= 1)) {
      this.add("E_RANGE", parts.concat(["confidence"]), `confidence 必须在 [0,1]，得到 ${value}`);
    }
  }

  checkPath(node, key, parts) {
    if (node === null || typeof node !== "object" || !hasOwn(node, key)) return;
    const value = node[key];
    if (typeof value !== "string") return;
    const reason = unsafePathReason(value);
    if (reason) this.add("E_PATH", parts.concat([key]), `路径不安全：${reason}`);
  }

  // ---- time bounds ----------------------------------------------------
  checkTimes(duration) {
    const data = this.data;
    if (data === null || typeof data !== "object" || Array.isArray(data)) return;

    const tempo = own(data, "tempo");
    if (tempo !== null && typeof tempo === "object" && hasOwn(tempo, "beat_origin_seconds")) {
      const value = this.checkNumber(tempo, "beat_origin_seconds", ["tempo"]);
      if (value === null) return;
      if (value < -TIME_TOLERANCE) {
        this.add("E_TIME_RANGE", ["tempo", "beat_origin_seconds"], `节拍原点不能为负，得到 ${value}`);
      } else if (duration !== null && value > duration + TIME_TOLERANCE) {
        this.add(
          "E_TIME_RANGE",
          ["tempo", "beat_origin_seconds"],
          `节拍原点必须在 [0, ${duration}] 秒内，得到 ${value}`,
        );
      }
    }

    const instruments = own(data, "instruments");
    if (!Array.isArray(instruments)) return;
    instruments.forEach((instrument, i) => {
      if (instrument === null || typeof instrument !== "object") return;
      const events = own(instrument, "events");
      if (!Array.isArray(events)) return;
      const base = ["instruments", i, "events"];
      let previous = null;
      events.forEach((event, j) => {
        if (event === null || typeof event !== "object") return;
        const eventParts = base.concat([j]);
        const onset = this.checkNumber(event, "onset_seconds", eventParts);
        const eventId = typeof own(event, "id") === "string" ? event.id : "";
        if (onset !== null) {
          if (onset < -TIME_TOLERANCE) {
            this.add("E_TIME_RANGE", eventParts.concat(["onset_seconds"]), `onset_seconds 不能为负，得到 ${onset}`);
          } else if (duration !== null && onset > duration + TIME_TOLERANCE) {
            this.add(
              "E_TIME_RANGE",
              eventParts.concat(["onset_seconds"]),
              `onset_seconds 必须在 [0, ${duration}] 秒内（时间边界=音频末尾），得到 ${onset}`,
            );
          }
          const key = [onset, eventId];
          if (previous !== null && (key[0] < previous[0] || (key[0] === previous[0] && cmpText(key[1], previous[1]) < 0))) {
            this.add(
              "E_TIME_ORDER",
              eventParts,
              `事件必须按 (onset_seconds 升序, id 升序) 排序：(${key[0]}, ${JSON.stringify(key[1])}) 排在 (${previous[0]}, ${JSON.stringify(previous[1])}) 之后`,
            );
          }
          previous = key;
        }
        const durationValue = this.checkNumber(event, "duration_seconds", eventParts);
        if (durationValue !== null) {
          if (durationValue <= 0) {
            this.add(
              "E_TIME_DURATION",
              eventParts.concat(["duration_seconds"]),
              `duration_seconds 必须 > 0，得到 ${durationValue}`,
            );
          } else if (duration !== null && onset !== null && onset + durationValue > duration + TIME_TOLERANCE) {
            this.add(
              "E_TIME_DURATION",
              eventParts.concat(["duration_seconds"]),
              `onset_seconds + duration_seconds 必须 <= 音频时长 ${duration}（允许相等），得到 ${onset} + ${durationValue}`,
            );
          }
        }
        this.checkConfidence(event, eventParts);
        const pitch = own(event, "pitch");
        if (pitch !== null && typeof pitch === "object") {
          const pitchParts = eventParts.concat(["pitch"]);
          const midi = this.checkNumber(pitch, "midi", pitchParts);
          if (midi !== null) {
            if (!Number.isInteger(midi)) {
              this.add("E_RANGE", pitchParts.concat(["midi"]), `pitch.midi 必须是整数 0..127，得到 ${midi}`);
            } else if (!(midi >= 0 && midi <= 127)) {
              this.add("E_RANGE", pitchParts.concat(["midi"]), `pitch.midi 必须在 0..127，得到 ${midi}`);
            }
          }
          this.checkConfidence(pitch, pitchParts);
        }
      });
    });
  }

  checkIds() {
    const data = this.data;
    if (data === null || typeof data !== "object" || Array.isArray(data)) return;
    const seenInstruments = new Set();
    const seenEvents = new Set();
    const instruments = own(data, "instruments");
    if (!Array.isArray(instruments)) return;
    instruments.forEach((instrument, i) => {
      if (instrument === null || typeof instrument !== "object") return;
      const instrumentId = own(instrument, "id");
      if (typeof instrumentId === "string") {
        if (seenInstruments.has(instrumentId)) {
          this.add("E_ID_UNIQUE", ["instruments", i, "id"], `instrument.id 必须全文档唯一，重复：${JSON.stringify(instrumentId)}`);
        }
        seenInstruments.add(instrumentId);
      }
      const events = own(instrument, "events");
      if (!Array.isArray(events)) return;
      events.forEach((event, j) => {
        if (event === null || typeof event !== "object") return;
        const eventId = own(event, "id");
        if (typeof eventId === "string") {
          if (seenEvents.has(eventId)) {
            this.add(
              "E_ID_UNIQUE",
              ["instruments", i, "events", j, "id"],
              `event.id 必须全文档唯一（跨乐器），重复：${JSON.stringify(eventId)}`,
            );
          }
          seenEvents.add(eventId);
        }
      });
    });
  }

  checkTempo() {
    const data = this.data;
    if (data === null || typeof data !== "object" || Array.isArray(data)) return;
    const tempo = own(data, "tempo");
    if (tempo === null || typeof tempo !== "object") return;
    this.checkNumber(tempo, "bpm", ["tempo"]); // boolean / huge number fallback
    const bpm = tempo.bpm;
    const source = tempo.source;
    const hasOrigin = hasOwn(tempo, "beat_origin_seconds");
    if (bpm === null) {
      if (source !== "unknown") {
        this.add("E_TEMPO", ["tempo", "source"], 'bpm 为 null 时 source 必须是 "unknown"');
      }
      const confidence = this.checkNumber(tempo, "confidence", ["tempo"]);
      if (confidence !== null && confidence !== 0) {
        this.add("E_TEMPO", ["tempo", "confidence"], "bpm 为 null 时 confidence 必须是 0");
      }
      if (hasOrigin) {
        this.add(
          "E_TEMPO",
          ["tempo", "beat_origin_seconds"],
          "bpm 为 null 时不允许 beat_origin_seconds（没有节拍网格就没有原点）",
        );
      }
    } else if (isNumber(bpm)) {
      if (source === "unknown") {
        this.add("E_TEMPO", ["tempo", "source"], 'bpm 已知时 source 不得为 "unknown"（未知/已知互斥）');
      }
    }
  }
}

/** Cross-field semantic checks; issues sorted by (pointer, code, message). */
export function validateSemantics(data) {
  const ctx = new Semantic(data);
  ctx.walkFinite(data, []);
  ctx.checkIds();
  ctx.checkTempo();

  let duration = null;
  if (data !== null && typeof data === "object" && !Array.isArray(data)) {
    const audio = own(data, "audio");
    if (audio !== null && typeof audio === "object") {
      duration = ctx.checkNumber(audio, "duration_seconds", ["audio"]);
      ctx.checkPath(audio, "filename", ["audio"]);
    }
    const tempo = own(data, "tempo");
    if (tempo !== null && typeof tempo === "object") {
      ctx.checkConfidence(tempo, ["tempo"]);
    }
    const instruments = own(data, "instruments");
    if (Array.isArray(instruments)) {
      instruments.forEach((instrument, i) => {
        if (instrument === null || typeof instrument !== "object") return;
        const parts = ["instruments", i];
        ctx.checkConfidence(instrument, parts);
        const stem = own(instrument, "stem");
        if (stem !== null && typeof stem === "object") {
          ctx.checkPath(stem, "filename", parts.concat(["stem"]));
        }
      });
    }
  }
  ctx.checkTimes(duration);
  return ctx.issues.sort(
    (a, b) => cmpText(a.pointer, b.pointer) || cmpText(a.code, b.code) || cmpText(a.message, b.message),
  );
}
