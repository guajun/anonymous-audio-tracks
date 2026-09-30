/**
 * node_guard_harness.mjs — offline unit exercises for bridge/audio_guard.mjs.
 * Run: node tests/node_guard_harness.mjs   (exits non-zero on failure)
 */
import assert from "node:assert/strict";
import { sep } from "node:path";
import {
  ALLOWED_AUDIO_MIME,
  DEFAULT_MAX_BYTES,
  ERROR_CODES,
  checkQueueBudget,
  checkSize,
  fail,
  isWithinRoot,
  resolveWithinRoot,
  safeBasename,
  sniffAudioMime,
} from "../bridge/audio_guard.mjs";

const root = ["C:", "work", "audio-probe", "fixtures"].join(sep);

// --- sniffAudioMime: allow-list positives ---------------------------------
const wav = Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WAVEfmt ")]);
assert.equal(sniffAudioMime(wav), ALLOWED_AUDIO_MIME.wav, "wav sniff");
assert.equal(sniffAudioMime(Buffer.from("ID3\0\0\0\0\0\0\0\0\0\0")), ALLOWED_AUDIO_MIME.mp3, "mp3 id3 sniff");
assert.equal(sniffAudioMime(Buffer.from([0xff, 0xfb, 0x90, 0x00, 0, 0, 0, 0, 0, 0, 0, 0])), ALLOWED_AUDIO_MIME.mp3, "mp3 frame sniff");
assert.equal(sniffAudioMime(Buffer.from("OggS\0\0\0\0\0\0\0\0")), ALLOWED_AUDIO_MIME.ogg, "ogg sniff");
assert.equal(sniffAudioMime(Buffer.from("fLaC\0\0\0\0\0\0\0\0")), ALLOWED_AUDIO_MIME.flac, "flac sniff");
assert.equal(sniffAudioMime(Buffer.from("\0\0\0\x20ftypM4A \0\0\0\0")), ALLOWED_AUDIO_MIME.m4a, "m4a sniff");

// --- sniffAudioMime: negatives (nothing may masquerade as audio) ---------
assert.equal(sniffAudioMime(Buffer.from("RIFF\0\0\0\0AVI \0\0\0\0")), null, "riff-but-not-wav rejected");
assert.equal(sniffAudioMime(Buffer.from("\x89PNG\r\n\x1a\n" + "x".repeat(16))), null, "png rejected");
assert.equal(sniffAudioMime(Buffer.from("hello, this is plain text")), null, "text rejected");
assert.equal(sniffAudioMime(Buffer.from("RIFF")), null, "too short rejected");
assert.equal(sniffAudioMime(Buffer.alloc(0)), null, "empty rejected");

// --- resolveWithinRoot: containment --------------------------------------
assert.equal(resolveWithinRoot(root, "a.wav"), root + sep + "a.wav");
assert.equal(resolveWithinRoot(root, "sub" + sep + "a.wav"), root + sep + "sub" + sep + "a.wav");
assert.throws(() => resolveWithinRoot(root, ".." + sep + "secret.wav"), /escapes/, "traversal rejected");
assert.throws(() => resolveWithinRoot(root, sep + "etc" + sep + "passwd"), /escapes/, "absolute outside rejected");
assert.throws(() => resolveWithinRoot(root, "   "), /non-empty/, "empty rejected");
assert.ok(isWithinRoot(root, root + sep + "x"), "root containment positive");
assert.ok(!isWithinRoot(root, root + "-sibling" + sep + "x"), "sibling dir rejected");

// --- checkSize / checkQueueBudget ------------------------------------------
assert.equal(checkSize(1, DEFAULT_MAX_BYTES), 1);
assert.throws(() => checkSize(0), /empty/, "empty size rejected");
assert.throws(() => checkSize(DEFAULT_MAX_BYTES + 1), /size limit/, "oversize rejected");
assert.throws(() => checkSize(11, 10), /size limit/, "custom limit enforced");
assert.equal(checkQueueBudget(0, 100, 200), 100);
assert.equal(checkQueueBudget(50, 150, 200), 200);
assert.throws(() => checkQueueBudget(150, 100, 200), /cumulative/, "queue overflow rejected");

// --- fail(): stable code, absolute paths never in the message ---------------
const leaked = fail(ERROR_CODES.READ, "EACCES: permission denied, open 'C:\\secret\\dir\\a.wav'");
assert.ok(leaked.message.startsWith("E_AUDIO_READ:"), "stable error code prefix");
assert.ok(!leaked.message.includes("C:"), `message must not leak paths: ${leaked.message}`);
assert.ok(leaked.message.includes("<PATH>"), "leaked path replaced by marker");
assert.ok(fail(ERROR_CODES.MIME, "plain reason").message === "E_AUDIO_MIME: plain reason");

// --- safeBasename (no absolute paths leak into model context) ------------
assert.equal(safeBasename(root + sep + "tone.wav"), "tone.wav");

console.log("node_guard_harness: OK");
