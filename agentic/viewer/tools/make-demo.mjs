// Deterministic demo generator: pure-synthesis WAV + matching
// `agentic-audio-tracks/v1` documents.
//
//   node agentic/viewer/tools/make-demo.mjs [--out-dir agentic/viewer/fixtures/generated]
//
// Everything here is MOCK data (provenance.source="mock"): self-generated
// audio and self-invented events, explicitly NOT research results. The audio
// and the JSON share one timeline (onsets coincide with audible tones) and the
// JSON records the real SHA-256 of the generated WAV, so the viewer's
// hash/duration checks pass on a clean checkout.
//
// The generator is deterministic: re-running it reproduces byte-identical
// audio and JSON (tests verify this). Generated audio is gitignored
// (`*.wav`); the JSON documents under `fixtures/demo/` are committed.

import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
export const VIEWER_ROOT = resolve(HERE, "..");
export const DEMO_SECONDS = 12;
export const DEMO_RATE = 44100;
export const DEMO_WAV_NAME = "demo-track.wav";

/** Deterministic mono 16-bit PCM WAV. */
export function buildDemoWav({ seconds = DEMO_SECONDS, rate = DEMO_RATE, events = demoEvents() } = {}) {
  const total = Math.floor(seconds * rate);
  const samples = new Float64Array(total);
  // quiet room tone so the lane is never empty
  for (let index = 0; index < total; index += 1) {
    samples[index] = 0.02 * Math.sin((2 * Math.PI * 55 * index) / rate);
  }
  for (const event of events) {
    const duration = event.duration_seconds || 0.18;
    const from = Math.floor(event.onset_seconds * rate);
    const to = Math.min(total, Math.ceil((event.onset_seconds + duration) * rate));
    const fade = Math.min(Math.floor(0.01 * rate), Math.max(1, Math.floor((to - from) * 0.25)));
    for (let index = from; index < to; index += 1) {
      const local = index - from;
      let envelope = 1;
      if (local < fade) envelope = local / fade;
      if (to - index <= fade) envelope = (to - index) / fade;
      const t = index / rate;
      samples[index] += event.amplitude * envelope * Math.sin(2 * Math.PI * event.frequency * t);
    }
  }
  const bytes = Buffer.alloc(44 + total * 2);
  bytes.write("RIFF", 0, "ascii");
  bytes.writeUInt32LE(36 + total * 2, 4);
  bytes.write("WAVE", 8, "ascii");
  bytes.write("fmt ", 12, "ascii");
  bytes.writeUInt32LE(16, 16);
  bytes.writeUInt16LE(1, 20); // PCM
  bytes.writeUInt16LE(1, 22); // mono
  bytes.writeUInt32LE(rate, 24);
  bytes.writeUInt32LE(rate * 2, 28);
  bytes.writeUInt16LE(2, 32);
  bytes.writeUInt16LE(16, 34);
  bytes.write("data", 36, "ascii");
  bytes.writeUInt32LE(total * 2, 40);
  for (let index = 0; index < total; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    bytes.writeInt16LE(Math.round(clamped * 32767), 44 + index * 2);
  }
  return bytes;
}

/**
 * The demo event table. Times are absolute audio seconds and stay valid
 * (sorted by (onset, id); durations fit inside the 12s file).
 */
export function demoEvents() {
  const events = [];
  const push = (row, onset, duration, frequency, amplitude) => {
    events.push({
      row,
      onset_seconds: onset,
      duration_seconds: duration,
      frequency,
      amplitude,
    });
  };
  // drums-kick: beats 1 and 3 of every bar (120 BPM, bar = 2s)
  for (let bar = 0; bar < 6; bar += 1) {
    push("drums-kick", bar * 2 + 0.0, 0.16, 60, 0.55);
    push("drums-kick", bar * 2 + 1.0, 0.16, 60, 0.45);
  }
  // hats-1: off-beat ticks, NO duration (onset-only display)
  for (let index = 0; index < 24; index += 1) {
    push("hats-1", index * 0.5 + 0.25, null, 6000, 0.12);
  }
  // bass-1: root notes
  for (let bar = 0; bar < 6; bar += 1) {
    push("bass-1", bar * 2, 0.9, 82.41, 0.4);
    push("bass-1", bar * 2 + 1.0, 0.9, 98.0, 0.35);
  }
  // piano-a / piano-b: same label "piano", different stable ids (never merged)
  for (let bar = 0; bar < 6; bar += 1) {
    push("piano-a", bar * 2 + 0.5, 0.6, 261.63, 0.3);
    push("piano-a", bar * 2 + 1.5, 0.4, 329.63, 0.25);
    push("piano-b", bar * 2 + 0.75, 0.5, 392.0, 0.22);
    push("piano-b", bar * 2 + 1.25, 0.35, 440.0, 0.2);
  }
  return events;
}

export function demoDoc(wavSha256, { withTempo = true, emptyEvents = false, maliciousLabels = false } = {}) {
  const eventsByRow = new Map();
  for (const event of demoEvents()) {
    if (!eventsByRow.has(event.row)) eventsByRow.set(event.row, []);
    eventsByRow.get(event.row).push(event);
  }

  const instrument = (id, label, description, row, extra = {}) => {
    const list = eventsByRow.get(row) || [];
    const events = emptyEvents
      ? []
      : list.map((event, index) => {
          const item = {
            id: `${id}-e${String(index).padStart(2, "0")}`,
            onset_seconds: event.onset_seconds,
          };
          if (event.duration_seconds !== null) item.duration_seconds = event.duration_seconds;
          item.confidence = 0.75;
          if (id.startsWith("piano")) {
            item.pitch = { midi: 60 + ((index * 3) % 18), confidence: 0.5, source: "mock" };
          }
          item.method = "synth-demo";
          return item;
        });
    return {
      id,
      label: maliciousLabels ? `<img src=x onerror="window.__xss=1">${label}` : label,
      description: maliciousLabels
        ? `恶意 label 测试行：<script>window.__xss=2</script>（必须按纯文本渲染）— ${description}`
        : description,
      source: "mock",
      confidence: 0.5,
      ...extra,
      events,
    };
  };

  return {
    schema_version: "agentic-audio-tracks/v1",
    audio: {
      filename: `demo/${DEMO_WAV_NAME}`,
      sha256: wavSha256,
      duration_seconds: DEMO_SECONDS,
      sample_rate: DEMO_RATE,
    },
    tempo: withTempo
      ? {
          bpm: 120,
          source: "mock",
          confidence: 0.5,
          beat_origin_seconds: 0,
        }
      : { bpm: null, source: "unknown", confidence: 0 },
    instruments: [
      instrument("drums-kick", "kick", "自生成的底鼓合成脉冲（每小节 1、3 拍）", "drums-kick"),
      instrument("hats-1", "hi-hat", "自生成的高频 tick（仅 onset，无 duration）", "hats-1"),
      instrument("bass-1", "bass", "自生成的低音线条", "bass-1"),
      instrument("piano-a", "piano", "自生成的钢琴和弦（同一 label 的第一个 id）", "piano-a"),
      instrument(
        "piano-b",
        "piano",
        "自生成的钢琴装饰音（与 piano-a 同 label 不同 id，UI 不合并）",
        "piano-b",
        { stem: { filename: `demo/${DEMO_WAV_NAME}`, sha256: wavSha256 } },
      ),
    ],
    provenance: {
      steps: [
        {
          tool: "aat-viewer-make-demo",
          source: "mock",
          note: "纯合成音频 + 合成事件（确定性生成器）；这是 mock 演示/测试数据，不是研究结果、不代表任何模型能力。",
        },
      ],
      notes: "mock demo for issue #34; events align with audible tones at the same absolute seconds.",
    },
    limitations: [
      "本文件是自生成 mock（provenance.source=mock），不是研究结果，也不代表任何分析模型的能力。",
      "pitch/confidence 均为随意填的演示值，不承诺正确性；没有音高证据的真实输出不应填写 pitch。",
      "音频与事件共享同一合成时间轴（onset 秒=可听脉冲秒），用于验证显示对齐，不是真实音乐分析。",
    ],
  };
}

function sha256Hex(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

export async function writeDemo({ outDir = join(VIEWER_ROOT, "fixtures", "generated"), fixturesDir = join(VIEWER_ROOT, "fixtures", "demo") } = {}) {
  const wav = buildDemoWav();
  const wavSha = sha256Hex(wav);
  const docs = {
    "demo-track.json": demoDoc(wavSha),
    "demo-unknown-tempo.json": demoDoc(wavSha, { withTempo: false }),
    "demo-empty-events.json": demoDoc(wavSha, { emptyEvents: true }),
    "demo-malicious-labels.json": demoDoc(wavSha, { maliciousLabels: true }),
  };
  await mkdir(outDir, { recursive: true });
  await mkdir(fixturesDir, { recursive: true });
  await writeFile(join(outDir, DEMO_WAV_NAME), wav);
  const written = [];
  for (const [name, doc] of Object.entries(docs)) {
    const target = join(fixturesDir, name);
    await writeFile(target, `${JSON.stringify(doc, null, 2)}\n`, "utf8");
    written.push(target);
  }
  return { wavPath: join(outDir, DEMO_WAV_NAME), wavSha256: wavSha, docs: written };
}

export async function main() {
  const result = await writeDemo({});
  process.stdout.write(`demo wav: ${result.wavPath}\n  sha256: ${result.wavSha256}\n`);
  for (const doc of result.docs) {
    process.stdout.write(`demo json: ${doc}\n`);
  }
  process.stdout.write("（mock 数据：自生成合成音频/事件，非研究结果）\n");
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
