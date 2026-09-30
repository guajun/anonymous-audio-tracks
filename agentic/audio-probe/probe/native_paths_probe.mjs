/**
 * native_paths_probe.mjs — offline evidence for the "Pi native audio input"
 * question (issue #30). Runs Pi's REAL input-path code against an audio file:
 *
 *  1. `processImage()` (pi-coding-agent dist/utils/image-process.js) — the
 *     normalizer used by the `read` tool, `@file` CLI args, RPC `prompt.images`
 *     and the SDK `prompt({images})`/`sendUserMessage` image path. Audio bytes
 *     must be REJECTED here.
 *  2. `detectSupportedImageMimeType()` (dist/utils/mime.js) — the `@file`
 *     image-vs-text decision. A WAV container must NOT be detected as an image,
 *     so `@file` falls through to the TEXT branch.
 *  3. the `@file` text branch itself (`buffer.toString("utf-8")`) — shows that
 *     the bytes reach the model as mojibake text, i.e. the model never hears
 *     anything. This is exactly the trap "不能只把文件名文本送过去就称成功".
 *
 * No network, no credentials. Usage:
 *   node probe/native_paths_probe.mjs --file fixtures/fixture-a.wav [--json]
 */
import { readFileSync } from "node:fs";
import { piUrls } from "./pi_install.mjs";

const args = process.argv.slice(2);
const fileIdx = args.indexOf("--file");
const filePath = fileIdx >= 0 ? args[fileIdx + 1] : "fixtures/fixture-a.wav";
const asJson = args.includes("--json");

const urls = piUrls();
if (!urls) {
  console.error("Pi installation not found. Set PI_INSTALL_DIR to the @earendil-works/pi-coding-agent directory.");
  process.exit(2);
}

const { processImage } = await import(urls.codingAgentUtils("image-process.js"));
const { detectSupportedImageMimeType, detectSupportedImageMimeTypeFromFile } = await import(
  urls.codingAgentUtils("mime.js")
);

const raw = readFileSync(filePath);

// 1. The shared image normalizer every documented input path runs.
const processed = await processImage(raw, "audio/wav", { autoResizeImages: true });

// 2. The @file image-vs-text decision (magic-byte sniffing).
const sniffed = detectSupportedImageMimeType(raw.subarray(0, 4100));
const sniffedFile = await detectSupportedImageMimeTypeFromFile(filePath);

// 3. What @file / read then do: decode as UTF-8 text.
const textBranch = raw.toString("utf-8");
const printable = textBranch.replace(/[^\x20-\x7e\n]/g, "·");

const result = {
  file: filePath,
  bytes: raw.length,
  piPackage: urls.dir,
  processImage: {
    ok: processed.ok,
    message: processed.message ?? null,
    interpretation: processed.ok
      ? "would be sent as an image attachment"
      : "audio rejected: every documented input path (@file, read, RPC/SDK prompt images) hits this",
  },
  detectImageMime: {
    fromBuffer: sniffed,
    fromFile: sniffedFile,
    interpretation:
      sniffed === null
        ? "@file treats the .wav as a TEXT file (no image detected)"
        : "unexpected: detected as image",
  },
  atfileTextBranch: {
    decodedAs: "utf-8",
    previewFirst120: printable.slice(0, 120),
    interpretation:
      "the model would receive binary mojibake as text — file bytes as text, NOT audio",
  },
};

if (asJson) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(`file              : ${result.file} (${result.bytes} bytes)`);
  console.log(`pi package        : ${result.piPackage}`);
  console.log(`processImage      : ok=${result.processImage.ok} message=${JSON.stringify(result.processImage.message)}`);
  console.log(`  -> ${result.processImage.interpretation}`);
  console.log(`image sniff       : ${JSON.stringify(result.detectImageMime)}`);
  console.log(`@file text branch : ${JSON.stringify(result.atfileTextBranch)}`);
}
