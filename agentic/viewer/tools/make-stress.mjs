// Stress fixture generator for the performance checks (issue #34).
//
//   node agentic/viewer/tools/make-stress.mjs [--events 100000] [--seconds 300]
//
// Writes a large `agentic-audio-tracks/v1` document with N synthetic events
// across 8 instrument rows into `agentic/viewer/fixtures/generated/`, which is
// gitignored: huge stress JSON/WAV must never be committed. The data is mock
// (provenance.source="mock") and deterministic (seeded PRNG).

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
export const VIEWER_ROOT = resolve(HERE, "..");
export const DEFAULT_OUT_DIR = join(VIEWER_ROOT, "fixtures", "generated");

const ROWS = [
  ["stress-kick", "kick"],
  ["stress-snare", "snare"],
  ["stress-hat", "hi-hat"],
  ["stress-bass", "bass"],
  ["stress-guitar", "guitar"],
  ["stress-keys", "keys"],
  ["stress-vox", "voice"],
  ["stress-noise", "unknown"],
];

/** Deterministic PRNG (mulberry32). */
export function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function buildStressDoc({ events = 100000, seconds = 300, sha256 = null, seed = 34034 } = {}) {
  const random = mulberry32(seed);
  const perRow = Math.floor(events / ROWS.length);
  const instruments = ROWS.map(([id, label], rowIndex) => {
    const list = [];
    for (let index = 0; index < perRow; index += 1) {
      const onset = Math.round(random() * (seconds - 0.05) * 1000) / 1000;
      const withDuration = random() < 0.5;
      const event = {
        id: `${id}-e${String(index).padStart(6, "0")}`,
        onset_seconds: onset,
      };
      if (withDuration) {
        // frozen rule: onset + duration <= audio duration (never overflow)
        const room = Math.floor((seconds - onset) * 1000) / 1000;
        const duration = Math.floor(Math.min(0.05 + random() * 0.4, room) * 1000) / 1000;
        if (duration > 0 && onset + duration <= seconds) event.duration_seconds = duration;
      }
      event.confidence = Math.round(random() * 100) / 100;
      list.push(event);
    }
    // frozen contract: events must be sorted by (onset_seconds, id)
    list.sort((a, b) => a.onset_seconds - b.onset_seconds || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    return {
      id,
      label,
      description: `stress row ${rowIndex + 1}/${ROWS.length}（mock，自生成，非研究结果）`,
      source: "mock",
      confidence: 0.5,
      events: list,
    };
  });

  return {
    schema_version: "agentic-audio-tracks/v1",
    audio: {
      filename: "generated/stress.wav",
      sha256: sha256 || "0".repeat(64),
      duration_seconds: seconds,
      sample_rate: 8000,
    },
    tempo: { bpm: 128, source: "mock", confidence: 0.3, beat_origin_seconds: 0 },
    instruments,
    provenance: {
      steps: [
        {
          tool: "aat-viewer-make-stress",
          source: "mock",
          note: `性能压力夹具：${events} 个随机合成事件（seed=${seed}），mock 数据，非研究结果。`,
        },
      ],
    },
    limitations: ["mock stress fixture：随机合成事件，不代表任何分析结果。"],
  };
}

/** Deterministic synthetic WAV for the stress timeline (mono 8kHz). */
export function buildStressWav({ seconds = 300, rate = 8000 } = {}) {
  const total = Math.floor(seconds * rate);
  const bytes = Buffer.alloc(44 + total * 2);
  bytes.write("RIFF", 0, "ascii");
  bytes.writeUInt32LE(36 + total * 2, 4);
  bytes.write("WAVE", 8, "ascii");
  bytes.write("fmt ", 12, "ascii");
  bytes.writeUInt32LE(16, 16);
  bytes.writeUInt16LE(1, 20);
  bytes.writeUInt16LE(1, 22);
  bytes.writeUInt32LE(rate, 24);
  bytes.writeUInt32LE(rate * 2, 28);
  bytes.writeUInt16LE(2, 32);
  bytes.writeUInt16LE(16, 34);
  bytes.write("data", 36, "ascii");
  bytes.writeUInt32LE(total * 2, 40);
  for (let index = 0; index < total; index += 1) {
    const t = index / rate;
    const burst = Math.floor(t) % 4 === 0 ? 0.5 : 0.08;
    const value = burst * Math.sin(2 * Math.PI * 180 * t) * (0.6 + 0.4 * Math.sin(2 * Math.PI * 0.25 * t));
    bytes.writeInt16LE(Math.round(Math.max(-1, Math.min(1, value)) * 32767), 44 + index * 2);
  }
  return bytes;
}

export async function writeStress({ outDir = DEFAULT_OUT_DIR, events = 100000, seconds = 300, withWav = true } = {}) {
  await mkdir(outDir, { recursive: true });
  let sha256 = "0".repeat(64);
  let wavPath = null;
  if (withWav) {
    const wav = buildStressWav({ seconds });
    sha256 = createHash("sha256").update(wav).digest("hex");
    wavPath = join(outDir, "stress.wav");
    await writeFile(wavPath, wav);
  }
  const doc = buildStressDoc({ events, seconds, sha256 });
  const jsonPath = join(outDir, `stress-${events}.json`);
  await writeFile(jsonPath, JSON.stringify(doc), "utf8");
  return { jsonPath, wavPath, sha256, events, seconds };
}

export async function main(argv = process.argv.slice(2)) {
  let events = 100000;
  let seconds = 300;
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--events") events = Number(argv[++index]);
    if (argv[index] === "--seconds") seconds = Number(argv[++index]);
  }
  const result = await writeStress({ events, seconds });
  process.stdout.write(
    `stress json: ${result.jsonPath}\nstress wav: ${result.wavPath}\nsha256: ${result.sha256}\n` +
      `（mock 数据，${events} events / ${seconds}s；生成物已 gitignore，不提交）\n`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
