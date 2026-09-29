// Generate small local WAV fixtures for manual listening and the headless
// browser check.  Generated audio is git-ignored (see the repository
// .gitignore: `*.wav`); the generator itself is committed.
//
// Usage:
//   node viewer/tools/make-fixture-audio.mjs
//   node viewer/tools/make-fixture-audio.mjs --pattern long --seconds 600
//   node viewer/tools/make-fixture-audio.mjs --pattern known-times \
//     --out tests/viewer/fixtures/generated/known-times.wav

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "..", "..");

/**
 * Render one mono 16-bit PCM WAV buffer.
 *
 * Patterns:
 * - `known-times`: tones aligned with the committed `trajectory.known-times`
 *   fixture activity intervals (0.3125–1.25s and 2.875–3.0833s).
 * - `long`: a long song stand-in (default 600s) with a quiet drone and a
 *   short burst every 30s, so the overview strip and window navigation have
 *   something to show.
 * - `tone`: a plain 440 Hz tone for transport checks.
 */
export function buildWav({ pattern = "known-times", seconds = 4, rate = 44100 } = {}) {
  const sampleRate = Math.max(1000, Math.floor(rate));
  const duration = Math.max(0.01, Number(seconds) || 4);
  const total = Math.floor(duration * sampleRate);
  const samples = new Float64Array(total);

  const addTone = (start, end, frequency, amplitude) => {
    const from = Math.max(0, Math.floor(start * sampleRate));
    const to = Math.min(total, Math.ceil(end * sampleRate));
    const fade = Math.min(0.01 * sampleRate, Math.max(1, (to - from) * 0.05));
    for (let index = from; index < to; index += 1) {
      const time = index / sampleRate;
      let envelope = 1;
      if (index - from < fade) {
        envelope = (index - from) / fade;
      }
      if (to - index < fade) {
        envelope = Math.min(envelope, (to - index) / fade);
      }
      samples[index] += amplitude * envelope * Math.sin(2 * Math.PI * frequency * time);
    }
  };

  if (pattern === "known-times") {
    addTone(0.3125, 1.25, 440, 0.7);
    addTone(2.875, 3.083333, 330, 0.7);
    addTone(1.0, 2.0, 220, 0.15); // trk-c activity interval
  } else if (pattern === "long") {
    addTone(0, duration, 180, 0.05);
    for (let start = 0; start < duration; start += 30) {
      const frequency = 220 * (1 + ((Math.floor(start / 30) % 5) * 0.25));
      addTone(start, Math.min(start + 2, duration), frequency, 0.6);
    }
  } else {
    addTone(0, duration, 440, 0.6);
  }

  const dataSize = total * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write("RIFF", 0, "ascii");
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8, "ascii");
  buffer.write("fmt ", 12, "ascii");
  buffer.writeUInt32LE(16, 16); // PCM chunk size
  buffer.writeUInt16LE(1, 20); // PCM
  buffer.writeUInt16LE(1, 22); // mono
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28); // byte rate
  buffer.writeUInt16LE(2, 32); // block align
  buffer.writeUInt16LE(16, 34); // bits per sample
  buffer.write("data", 36, "ascii");
  buffer.writeUInt32LE(dataSize, 40);
  for (let index = 0; index < total; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    buffer.writeInt16LE(Math.round(clamped * 32767), 44 + index * 2);
  }
  return buffer;
}

function parseArgs(argv) {
  const options = { pattern: "known-times", seconds: 4, rate: 44100, out: null };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--pattern") {
      options.pattern = argv[++index];
    } else if (arg === "--seconds") {
      options.seconds = Number(argv[++index]);
    } else if (arg === "--rate") {
      options.rate = Number(argv[++index]);
    } else if (arg === "--out") {
      options.out = argv[++index];
    }
  }
  return options;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.pattern === "long" && options.seconds === 4) {
    options.seconds = 600;
    options.rate = 8000;
  }
  const defaultName = `${options.pattern}${options.seconds >= 60 ? `-${options.seconds}s` : ""}.wav`;
  const out = options.out
    ? (isAbsolute(options.out) ? options.out : resolve(REPO_ROOT, options.out))
    : resolve(REPO_ROOT, "tests", "viewer", "fixtures", "generated", defaultName);
  const buffer = buildWav(options);
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, buffer);
  console.log(
    `wrote ${out} (${options.pattern}, ${options.seconds}s, ${options.rate}Hz mono 16-bit, ${buffer.length} bytes)`,
  );
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
