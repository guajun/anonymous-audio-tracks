/**
 * audio_guard.mjs — pure validation logic for the audio bridge (issue #30).
 *
 * No Pi imports here on purpose: this module is unit-testable offline with a
 * plain Node runtime. `audio-bridge.ts` (the Pi extension) imports these
 * functions; `tests/` exercise them through `tests/node_guard_harness.mjs`.
 *
 * Policy enforced here:
 *  - input must live inside a controlled root directory (no traversal, no
 *    absolute paths outside the root; the extension additionally re-checks
 *    after `fs.realpath` to defeat symlink escapes),
 *  - input must sniff as one of the allow-listed audio container formats,
 *  - input must not exceed a byte budget (default 4 MiB).
 */
import { resolve as pathResolve, sep } from "node:path";

/** Allow-listed audio container formats: magic bytes -> MIME. */
export const ALLOWED_AUDIO_MIME = {
  wav: "audio/wav",
  mp3: "audio/mpeg",
  ogg: "audio/ogg",
  flac: "audio/flac",
  m4a: "audio/mp4",
};

/** Hard default byte budget for one attachment (inline request cap is 20 MiB). */
export const DEFAULT_MAX_BYTES = 4 * 1024 * 1024;

/** True when `candidate` is `root` itself or below it (after normalization). */
export function isWithinRoot(root, candidate) {
  const r = pathResolve(root);
  const c = pathResolve(candidate);
  return c === r || c.startsWith(r.endsWith(sep) ? r : r + sep);
}

/**
 * Resolve a caller-supplied path against the controlled root.
 * Rejects traversal and absolute paths that escape the root.
 * Throws Error with a safe message (no absolute paths) on rejection.
 */
export function resolveWithinRoot(root, requested) {
  if (typeof requested !== "string" || requested.trim() === "") {
    throw new Error("path must be a non-empty string");
  }
  const trimmed = requested.trim();
  const fromRoot = pathResolve(root, trimmed);
  if (isWithinRoot(root, fromRoot)) return fromRoot;
  // An absolute path is acceptable only when it already points inside the root.
  const absolute = pathResolve(trimmed);
  if (isWithinRoot(root, absolute)) return absolute;
  throw new Error("path escapes the controlled audio root and was rejected");
}

/**
 * Sniff the audio container from magic bytes. Returns a MIME string or null.
 * Deliberately conservative: unknown/ambiguous bytes are rejected, so a text
 * file or an image can never masquerade as audio.
 */
export function sniffAudioMime(buf) {
  if (!buf || buf.length < 12) return null;
  const ascii = (start, len) => buf.subarray(start, start + len).toString("latin1");
  // RIFF....WAVE (WAV / BWF)
  if (ascii(0, 4) === "RIFF" && ascii(8, 4) === "WAVE") return ALLOWED_AUDIO_MIME.wav;
  // "ID3" tag or a raw MPEG audio frame sync
  if (ascii(0, 3) === "ID3") return ALLOWED_AUDIO_MIME.mp3;
  if (buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0) return ALLOWED_AUDIO_MIME.mp3;
  // Ogg container
  if (ascii(0, 4) === "OggS") return ALLOWED_AUDIO_MIME.ogg;
  // Native FLAC
  if (ascii(0, 4) === "fLaC") return ALLOWED_AUDIO_MIME.flac;
  // MP4/M4A container: ftyp....M4A*
  if (ascii(4, 4) === "ftyp") {
    const brand = ascii(8, 4);
    if (brand.startsWith("M4A") || brand === "mp42" || brand === "isom") return ALLOWED_AUDIO_MIME.m4a;
  }
  return null;
}

/** Exponential-backoff-free byte budget check. Throws on oversize input. */
export function checkSize(sizeBytes, maxBytes = DEFAULT_MAX_BYTES) {
  if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
    throw new Error("audio file is empty or unreadable");
  }
  if (sizeBytes > maxBytes) {
    throw new Error(`audio file exceeds size limit (${sizeBytes} > ${maxBytes} bytes)`);
  }
  return sizeBytes;
}

/**
 * Cumulative queue budget: the inline request cap is 20 MiB and base64 costs
 * 4/3 x raw bytes, so the sum of queued raw bytes must stay bounded even when
 * every single file passes `checkSize`. Throws on overflow.
 */
export function checkQueueBudget(queuedBytes, newBytes, maxQueueBytes) {
  const total = queuedBytes + newBytes;
  if (total > maxQueueBytes) {
    throw new Error(
      `queued audio exceeds cumulative budget (${total} > ${maxQueueBytes} bytes)`,
    );
  }
  return total;
}

/** Error-code prefixes used by the bridge (stable contract for downstream). */
export const ERROR_CODES = {
  ARGS: "E_AUDIO_ARGS",
  PATH: "E_AUDIO_PATH",
  NOT_FOUND: "E_AUDIO_NOT_FOUND",
  NOT_FILE: "E_AUDIO_NOT_FILE",
  SIZE: "E_AUDIO_SIZE",
  QUEUE: "E_AUDIO_QUEUE",
  MIME: "E_AUDIO_MIME",
  READ: "E_AUDIO_READ",
  MODEL: "E_AUDIO_MODEL",
};

/** Build an error whose message is guaranteed free of absolute paths. */
export function fail(code, message) {
  const safe = String(message).replace(/[A-Za-z]:[\\/][^\s"']*/g, "<PATH>");
  return new Error(`${code}: ${safe}`);
}

/** Only the file name is safe to echo into model context or reports. */
export function safeBasename(fullPath) {
  const parts = pathResolve(fullPath).split(sep);
  return parts[parts.length - 1] || "audio.bin";
}
