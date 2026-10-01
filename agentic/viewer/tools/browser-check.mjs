// Real-browser checks for agentic/viewer (issue #34).
//
//   node agentic/viewer/tools/browser-check.mjs
//   node agentic/viewer/tools/browser-check.mjs --chrome "C:/path/to/chrome.exe"
//
// Launches the locally installed Chrome/Edge headless (no downloads), serves
// ONLY `agentic/viewer/` on 127.0.0.1, drives the page through the Chrome
// DevTools Protocol (DOM.setFileInputFiles + Runtime.evaluate + Input events)
// and asserts the state the UI actually computes. It also measures real load /
// render / frame timings for the 100k-event stress fixture and captures
// screenshots as review evidence.
//
// Artifacts (all gitignored, returned to the main agent for reading):
//   agentic/viewer/reports/browser-check.json
//   agentic/viewer/screenshots/*.png
//
// Exit code 0 = all checks pass.

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { startServer } from "./serve.mjs";
import { writeDemo, buildDemoWav, demoDoc, demoEvents, DEMO_WAV_NAME } from "./make-demo.mjs";
import { writeStress } from "./make-stress.mjs";
import { GLOBAL_SEEK_STEP_SECONDS } from "../js/global-seek.js";

const HERE = fileURLToPath(new URL(".", import.meta.url));
export const VIEWER_ROOT = resolve(HERE, "..");
const SCREENSHOT_DIR = join(VIEWER_ROOT, "screenshots");
const REPORT_DIR = join(VIEWER_ROOT, "reports");
const screenshotStates = []; // capture-time evidence for each screenshot

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function parseArgs(argv) {
  const options = { chrome: null, timeoutMs: 120000, stressEvents: 100000, real: { json: null, audio: null, stems: [] } };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--chrome") options.chrome = argv[++index];
    else if (argv[index] === "--timeout-ms") options.timeoutMs = Number(argv[++index]);
    else if (argv[index] === "--stress-events") options.stressEvents = Number(argv[++index]);
    // real #33 integration inputs — always passed via CLI, never hard-coded
    else if (argv[index] === "--real-json") options.real.json = argv[++index];
    else if (argv[index] === "--real-audio") options.real.audio = argv[++index];
    else if (argv[index] === "--real-stem") options.real.stems.push(argv[++index]);
  }
  return options;
}

function findChrome(explicit) {
  const candidates = [];
  if (explicit) candidates.push(explicit);
  if (process.env.CHROME_PATH) candidates.push(process.env.CHROME_PATH);
  if (process.platform === "win32") {
    candidates.push(
      "C:/Program Files/Google/Chrome/Application/chrome.exe",
      "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
      "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
      "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    );
  } else if (process.platform === "darwin") {
    candidates.push(
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    );
  } else {
    candidates.push("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/microsoft-edge");
  }
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) return candidate;
  }
  return null;
}

async function waitForHttp(url, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
    } catch (error) {
      lastError = error;
    }
    await sleep(200);
  }
  throw new Error(`timeout waiting for ${url}${lastError ? `: ${lastError.message}` : ""}`);
}

async function connectCdp(webSocketDebuggerUrl) {
  const socket = new WebSocket(webSocketDebuggerUrl);
  await new Promise((resolvePromise, rejectPromise) => {
    socket.addEventListener("open", resolvePromise, { once: true });
    socket.addEventListener("error", () => rejectPromise(new Error("CDP socket error")), { once: true });
  });
  let nextId = 1;
  const pending = new Map();
  const listeners = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const entry = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) entry.reject(new Error(`${entry.method}: ${JSON.stringify(message.error)}`));
      else entry.resolve(message.result);
    } else if (message.method) {
      for (const handler of listeners.get(message.method) || []) handler(message.params);
    }
  });
  return {
    send(method, params = {}) {
      const id = nextId++;
      return new Promise((resolvePromise, rejectPromise) => {
        pending.set(id, { resolve: resolvePromise, reject: rejectPromise, method });
        socket.send(JSON.stringify({ id, method, params }));
      });
    },
    on(method, handler) {
      const handlers = listeners.get(method) || [];
      handlers.push(handler);
      listeners.set(method, handlers);
    },
    close() {
      socket.close();
    },
  };
}

async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  });
  if (result.exceptionDetails) {
    const description =
      (result.exceptionDetails.exception && result.exceptionDetails.exception.description) || result.exceptionDetails.text;
    throw new Error(`page evaluate failed: ${description}`);
  }
  return result.result ? result.result.value : undefined;
}

async function waitFor(client, description, expression, timeoutMs = 15000, intervalMs = 120) {
  const deadline = Date.now() + timeoutMs;
  let lastValue = null;
  while (Date.now() < deadline) {
    try {
      const value = await evaluate(client, expression);
      lastValue = value;
      if (value) return value;
    } catch {
      // keep polling
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting for ${description} (last value: ${JSON.stringify(lastValue)})`);
}

async function setFileInput(client, selector, files) {
  // Reset first: re-selecting an identical file must still fire `change`.
  await evaluate(client, `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (el) el.value = ""; })()`);
  const document = await client.send("DOM.getDocument", { depth: 1 });
  const { nodeId } = await client.send("DOM.querySelector", { nodeId: document.root.nodeId, selector });
  if (!nodeId) throw new Error(`file input not found: ${selector}`);
  await client.send("DOM.setFileInputFiles", { files, nodeId });
}

/** Load a JSON file and wait for THIS load to complete (load-counter, no stale state). */
async function loadJsonFile(client, path) {
  const before = await evaluate(client, "window.__aatViewer.state().loads.json");
  await setFileInput(client, "#json-input", [path]);
  await waitFor(client, `json load of ${path}`, `window.__aatViewer.state().loads.json > ${before}`, 60000);
}

/** Load an audio file and wait for THIS load to complete. */
async function loadAudioFile(client, path) {
  const before = await evaluate(client, "window.__aatViewer.state().loads.audio");
  await setFileInput(client, "#audio-input", [path]);
  await waitFor(client, `audio load of ${path}`, `window.__aatViewer.state().loads.audio > ${before}`, 60000);
}

async function screenshotFullPage(client, name) {
  // full page (all rows + onsets + stem waveforms below the fold)
  await client.send("Page.bringToFront");
  await evaluate(client, "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))");
  const layout = await client.send("Page.getLayoutMetrics");
  const size = layout.contentSize || layout.cssContentSize;
  const shot = await client.send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width: Math.ceil(size.width), height: Math.ceil(size.height), scale: 1 },
  });
  const target = join(SCREENSHOT_DIR, `${name}.png`);
  await writeFile(target, Buffer.from(shot.data, "base64"));
  screenshotStates.push({ name, atCapture: { fullPage: { width: size.width, height: size.height } } });
  return target;
}

async function screenshot(client, name) {
  // Headless captures can return the LAST committed frame (observed: one step
  // behind the DOM/canvas state). Bumping the emulated metrics forces the
  // compositor to commit a fresh surface; two rAFs then flush the frame before
  // we capture.
  const metrics = await evaluate(
    client,
    "(() => ({ width: window.innerWidth, height: window.innerHeight, dpr: window.devicePixelRatio }))()",
  );
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: Math.max(1, metrics.width - 1),
    height: metrics.height,
    deviceScaleFactor: metrics.dpr,
    mobile: false,
  });
  await sleep(150);
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: metrics.width,
    height: metrics.height,
    deviceScaleFactor: metrics.dpr,
    mobile: false,
  });
  await evaluate(client, "new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(true))))");
  await sleep(80);
  const shot = await client.send("Page.captureScreenshot", { format: "png", fromSurface: true, captureBeyondViewport: false });
  const target = join(SCREENSHOT_DIR, `${name}.png`);
  await writeFile(target, Buffer.from(shot.data, "base64"));
  // evidence: what the page state said at capture time
  const atCapture = await evaluate(
    client,
    `(() => {
      const s = window.__aatViewer ? window.__aatViewer.state() : null;
      return s && s.doc ? { doc: s.doc.rows.map((r) => r.id).join(','), events: s.doc.totalEvents, view: s.view } : { doc: null };
    })()`,
  ).catch(() => ({ doc: "?" }));
  screenshotStates.push({ name, atCapture });
  return target;
}

const CLOSE = (a, b, tolerance) => Math.abs(a - b) <= tolerance;

// The DOM range input's VALUE string is limited to 15 significant digits by
// the browser (e.g. 16.036979166666665 -> 16.0369791666667, ~3e-13 s error);
// max/min/seek keep the full double and the media clock itself is µs-quantized.
// The endpoint contract is therefore asserted at 1e-9 s — orders of magnitude
// below the old 0.1-step truncation (0.037 s) and below any audible precision.
const ENDPOINT_EPS = 1e-9;

// ---- real input helpers for the global seek slider (issue #41) -----------

/** Real CDP key press (native range behaviour: Home/End/arrows step the slider). */
async function pressKey(client, { key, code, windowsVirtualKeyCode }) {
  await client.send("Input.dispatchKeyEvent", { type: "keyDown", key, code, windowsVirtualKeyCode });
  await client.send("Input.dispatchKeyEvent", { type: "keyUp", key, code, windowsVirtualKeyCode });
}

/** Real mouse click on the slider track at `ratio` (0 = file start, 1 = file end). */
async function sliderClick(client, ratio) {
  await evaluate(client, `(() => { document.getElementById('global-seek').scrollIntoView({ block: 'center' }); return true; })()`);
  await sleep(80);
  const rect = await evaluate(
    client,
    `(() => { const r = document.getElementById('global-seek').getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; })()`,
  );
  const clamped = Math.min(Math.max(ratio, 0), 1);
  const x = rect.x + Math.min(Math.max(clamped * rect.width, 1), Math.max(1, rect.width - 1));
  const y = rect.y + rect.height / 2;
  await client.send("Input.dispatchMouseEvent", { type: "mouseMoved", x, y });
  await client.send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await client.send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });
}

/** Real mouse drag on the slider PAST the right edge (must reach the real file end). */
async function sliderDragRightPastEnd(client) {
  await evaluate(client, `(() => { document.getElementById('global-seek').scrollIntoView({ block: 'center' }); return true; })()`);
  await sleep(80);
  const rect = await evaluate(
    client,
    `(() => { const r = document.getElementById('global-seek').getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; })()`,
  );
  const y = rect.y + rect.height / 2;
  const startX = rect.x + rect.width * 0.4;
  await client.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: startX, y });
  await client.send("Input.dispatchMouseEvent", { type: "mousePressed", x: startX, y, button: "left", buttons: 1, clickCount: 1 });
  for (const ratio of [0.6, 0.8, 1.0]) {
    await client.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: rect.x + rect.width * ratio, y, button: "left", buttons: 1 });
    await sleep(40);
  }
  const pastEnd = rect.x + rect.width + 60;
  await client.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: pastEnd, y, button: "left", buttons: 1 });
  await sleep(60);
  await client.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: pastEnd, y, button: "left", buttons: 0, clickCount: 1 });
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const chromePath = findChrome(options.chrome);
  if (!chromePath) {
    process.stderr.write(
      "no Chrome/Edge found; pass --chrome <path> or set CHROME_PATH. Browser checks are then UNVERIFIED.\n",
    );
    process.exitCode = 2;
    return { ok: false, verified: false };
  }

  await mkdir(SCREENSHOT_DIR, { recursive: true });
  await mkdir(REPORT_DIR, { recursive: true });

  // Fixtures: deterministic demo + stress (gitignored generated assets).
  const demo = await writeDemo({});
  const stress = await writeStress({ events: options.stressEvents, seconds: 300 });
  const workDir = await mkdtemp(join(tmpdir(), "aat-agentic-viewer-check-"));
  const wrongWav = join(workDir, "wrong-duration.wav");
  await writeFile(wrongWav, buildDemoWav({ seconds: 3, events: [] }));
  const badJson = join(workDir, "bad.json");
  await writeFile(badJson, '{ "schema_version": "agentic-audio-tracks/v1", ');
  const badPathJson = join(workDir, "bad-path.json");
  const badPathDoc = demoDoc(demo.wavSha256);
  badPathDoc.audio.filename = "../secret/escape.wav";
  await writeFile(badPathJson, JSON.stringify(badPathDoc));

  // --- synthetic fixtures for lifecycle / stem identity / clip regressions ---
  const sha256 = (buffer) => createHash("sha256").update(buffer).digest("hex");
  const stemDir = (name) => {
    const dir = join(workDir, name);
    return { dir, file: join(dir, "target.wav") };
  };
  const stemA = stemDir("stem-a");
  const stemB = stemDir("stem-b");
  const stemC = stemDir("stem-c");
  const stemShort = stemDir("stem-short");
  await mkdir(stemA.dir, { recursive: true });
  await mkdir(stemB.dir, { recursive: true });
  await mkdir(stemC.dir, { recursive: true });
  await mkdir(stemShort.dir, { recursive: true });
  await writeFile(stemA.file, buildDemoWav({ seconds: 12, events: [] }));
  await writeFile(stemB.file, buildDemoWav({ seconds: 12, events: demoEvents().slice(0, 2) }));
  await writeFile(stemC.file, buildDemoWav({ seconds: 12, events: demoEvents().slice(2, 6) }));
  await writeFile(stemShort.file, buildDemoWav({ seconds: 3, events: [] }));
  const hashA = sha256(readFileSync(stemA.file));
  const hashB = sha256(readFileSync(stemB.file));
  const hashShort = sha256(readFileSync(stemShort.file));
  // three rows, SAME basename target.wav, different full paths + hashes
  const stemDoc = demoDoc(demo.wavSha256, { emptyEvents: true });
  stemDoc.instruments = [
    {
      id: "a-row",
      label: "target A",
      description: "same-basename stem a/target.wav (mock)",
      source: "mock",
      confidence: 0.5,
      stem: { filename: "a/target.wav", sha256: hashA },
      events: [{ id: "a-ev-1", onset_seconds: 1 }],
    },
    {
      id: "b-row",
      label: "target B",
      description: "same-basename stem b/target.wav (mock)",
      source: "mock",
      confidence: 0.5,
      stem: { filename: "b/target.wav", sha256: hashB },
      events: [{ id: "b-ev-1", onset_seconds: 2 }],
    },
    {
      id: "c-row",
      label: "target C (short)",
      description: "declared hash is a 3s file: duration mismatch must refuse it (mock)",
      source: "mock",
      confidence: 0.5,
      stem: { filename: "c/target.wav", sha256: hashShort },
      events: [],
    },
  ];
  const stemDocJson = join(workDir, "stem-doc.json");
  await writeFile(stemDocJson, JSON.stringify(stemDoc));

  // sustained event crossing the viewport edge (onset 0, duration 10, audio 12s)
  const clipDoc = demoDoc(demo.wavSha256, { emptyEvents: true });
  clipDoc.instruments = [
    {
      id: "sus-row",
      label: "sustained",
      description: "one long event for clip regression (mock)",
      source: "mock",
      confidence: 0.5,
      events: [{ id: "sus-1", onset_seconds: 0, duration_seconds: 10 }],
    },
  ];
  const clipDocJson = join(workDir, "clip-doc.json");
  await writeFile(clipDocJson, JSON.stringify(clipDoc));

  // byte-level entry fixtures: BOM and invalid UTF-8 (File.text() would hide both)
  const bytesBomJson = join(workDir, "bytes-bom.json");
  await writeFile(bytesBomJson, Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(JSON.stringify(demoDoc(demo.wavSha256)))]));
  const bytesBadUtf8Json = join(workDir, "bytes-bad-utf8.json");
  await writeFile(
    bytesBadUtf8Json,
    Buffer.concat([Buffer.from('{"schema_version": "agentic-audio-tracks/v1", "label": "'), Buffer.from([0xff]), Buffer.from('"}')]),
  );

  const checks = [];
  const screenshots = [];
  let realIntegration = null;
  const record = (name, ok, details) => {
    checks.push({ name, ok: Boolean(ok), details: details === undefined ? null : details });
    process.stdout.write(`${ok ? "PASS" : "FAIL"}  ${name}${details === undefined ? "" : ` — ${JSON.stringify(details)}`}\n`);
  };

  const { server, port } = await startServer({ port: 0 });
  const pageUrl = `http://127.0.0.1:${port}/index.html`;
  let chrome = null;
  let client = null;
  let chromeLog = "";
  const env = { browserBinary: chromePath, pageUrl, node: process.version, platform: process.platform };

  try {
    chrome = spawn(
      chromePath,
      [
        "--headless=new",
        "--remote-debugging-port=0",
        `--user-data-dir=${join(workDir, "chrome-profile")}`,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu",
        "--mute-audio",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1280,900",
        pageUrl,
      ],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    chrome.stderr.on("data", (chunk) => {
      chromeLog += chunk.toString();
    });
    chrome.stdout.on("data", (chunk) => {
      chromeLog += chunk.toString();
    });

    let debuggingPort = null;
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline && debuggingPort === null) {
      const match = /DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//.exec(chromeLog);
      if (match) debuggingPort = Number(match[1]);
      else await sleep(150);
    }
    if (debuggingPort === null) throw new Error(`Chrome DevTools port not found; log tail: ${chromeLog.slice(-500)}`);

    const version = await waitForHttp(`http://127.0.0.1:${debuggingPort}/json/version`);
    const targets = await waitForHttp(`http://127.0.0.1:${debuggingPort}/json/list`);
    const pageTarget = targets.find((target) => target.type === "page") || null;
    if (!pageTarget) throw new Error("no page target from Chrome");
    client = await connectCdp(pageTarget.webSocketDebuggerUrl);
    const pageErrors = [];
    client.on("Runtime.exceptionThrown", (params) => {
      const details = params.exceptionDetails || {};
      pageErrors.push(String(details.text || details.description || "exception"));
    });
    client.on("Runtime.consoleAPICalled", (params) => {
      if (params.type === "error") {
        pageErrors.push(`console.error: ${JSON.stringify((params.args || []).map((arg) => arg.value ?? arg.description))}`);
      }
    });
    await client.send("Runtime.enable");
    await client.send("DOM.enable");
    await client.send("Page.enable");
    await client.send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });

    await waitFor(client, "viewer ready", "window.__aatViewer && window.__aatViewer.ready === true", 20000);
    env.browserVersion = version.Browser || null;
    env.userAgent = await evaluate(client, "navigator.userAgent");
    env.hardwareConcurrency = await evaluate(client, "navigator.hardwareConcurrency");
    env.deviceMemory = await evaluate(client, "navigator.deviceMemory ?? null");
    env.initialDpr = await evaluate(client, "window.devicePixelRatio");
    record("page loads with the debug API ready", true, { userAgent: env.userAgent });

    // ---- load demo JSON -------------------------------------------------
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    await waitFor(client, "demo doc loaded", "window.__aatViewer.state().doc !== null", 20000);
    const docState = await evaluate(client, "window.__aatViewer.state().doc");
    record(
      "demo JSON: rows keep label + stable id (same label not merged)",
      docState.rows.length === 5 &&
        docState.rows[4].label === "piano" &&
        docState.rows[3].label === "piano" &&
        docState.rows[3].id !== docState.rows[4].id &&
        docState.rows[0].id === "drums-kick",
      { rows: docState.rows.map((r) => `${r.id}:${r.label}`), totalEvents: docState.totalEvents },
    );
    const gutterText = await evaluate(client, "document.getElementById('gutter-rows').textContent");
    const gutterNodes = await evaluate(client, "document.querySelectorAll('#gutter-rows .gutter-row').length");
    record(
      "gutter rows are one DOM node per instrument row with textContent labels",
      gutterNodes === 5 && gutterText.includes("drums-kick") && gutterText.includes("hi-hat"),
      { gutterNodes },
    );

    // ---- onset alignment on the seconds axis ----------------------------
    const alignment = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        const s = api.state();
        const rows = api.eventScreenPositions(3);
        const width = document.getElementById('canvas-stack').clientWidth;
        const bad = [];
        for (const row of rows) {
          for (const item of row.items) {
            const expected = ((item.onset - s.view.start) / (s.view.end - s.view.start)) * width;
            if (Math.abs(expected - item.x) > 0.01) bad.push({ id: item.id, x: item.x, expected });
          }
        }
        const at500ms = api.pixelOf(0.5);
        const expected500ms = (0.5 / s.view.end) * width; // view starts at 0 after load
        return { bad, at500ms, expected500ms, totalEvents: s.doc.totalEvents, rows: rows.map(r => r.count) };
      })()`,
    );
    record(
      "onsets map to x = (onset - viewStart) / viewDuration * width (audio seconds are the time truth)",
      alignment.bad.length === 0 && CLOSE(alignment.at500ms, alignment.expected500ms, 0.01),
      { rows: alignment.rows, at500ms: alignment.at500ms },
    );
    const durationState = await evaluate(client, "window.__aatViewer.state().doc.durationSeconds");
    record("JSON duration is the timeline length", durationState === 12, { durationSeconds: durationState });

    // ---- global whole-file slider before any audio (issue #41) ---------
    const seekBeforeAudio = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
    record(
      "global slider is disabled until a decodable audio file is loaded (min 0, no fake range)",
      seekBeforeAudio.disabled === true && seekBeforeAudio.min === 0 && seekBeforeAudio.max === 0 && /停用/.test(seekBeforeAudio.ariaValueText),
      seekBeforeAudio,
    );

    // ---- audio load, hash + waveform ------------------------------------
    await loadAudioFile(client, join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME));
    await waitFor(client, "audio loaded", "window.__aatViewer.state().audio.loaded === true", 30000);
    const audioState = await evaluate(client, "window.__aatViewer.state().audio");
    const hashOk = audioState.hashHex === docState.audioSha256;
    record(
      "audio loads locally: SHA-256 matches the JSON, no upload",
      hashOk && audioState.notes.filter((n) => n.level === "error").length === 0,
      { hashHex: audioState.hashHex, notes: audioState.notes.map((n) => n.code) },
    );
    const wave = await evaluate(
      client,
      `(() => {
        const canvas = document.getElementById('wave-canvas');
        const ctx = canvas.getContext('2d');
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let nonBackground = 0;
        for (let i = 0; i < data.length; i += 4) {
          if (data[i] !== 11 || data[i + 1] !== 15 || data[i + 2] !== 21) nonBackground += 1;
        }
        return { nonBackground, width: canvas.width, height: canvas.height };
      })()`,
    );
    record("waveform lane is drawn from the peak pyramid (visible pixels)", wave.nonBackground > 1000, wave);

    // ---- optional per-stem waveform (matched by stem filename only) -----
    await setFileInput(client, "#stem-input", [join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME)]);
    await waitFor(client, "stem loaded", "window.__aatViewer.state().audio.resource.stemsLoaded === 1", 30000);
    const stemCheck = await evaluate(
      client,
      `(() => {
        const canvas = document.getElementById('track-canvas');
        const ctx = canvas.getContext('2d');
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let stemPixels = 0;
        for (let i = 0; i < data.length; i += 4) {
          // faint stem waveform colour rgba(120,170,255) blended over row bg
          if (data[i] > 40 && data[i] < 120 && data[i + 2] > 90) stemPixels += 1;
        }
        return { stemPixels, notes: document.getElementById('notes').textContent, stems: window.__aatViewer.state().audio.resource.stemsLoaded };
      })()`,
    );
    record(
      "per-stem waveform renders in its row when the stem file matches by name+hash",
      stemCheck.stems === 1 && stemCheck.stemPixels > 100 && /已绘制.*行波形/.test(stemCheck.notes),
      { stems: stemCheck.stems, stemPixels: stemCheck.stemPixels },
    );
    screenshots.push(await screenshot(client, "01-demo-loaded"));

    // ---- BPM prefill / validation / grid-only behaviour -----------------
    const bpmPrefill = await evaluate(client, "document.getElementById('bpm-input').value");
    record("BPM is prefilled from JSON tempo.bpm", bpmPrefill === "120", { bpmPrefill });
    const bpmCheck = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        const before = JSON.stringify(api.onsetsSnapshot());
        api.setBpm(240);
        const grid = api.rowGrid();
        const after = JSON.stringify(api.onsetsSnapshot());
        const stateAfter = api.state().bpm;
        const invalid = api.setBpm(0);
        const invalidText = document.getElementById('bpm-status').textContent;
        const kept = api.state().bpm.value;
        api.setBpm(120);
        return {
          interval: grid.interval, beats: grid.beats.length,
          onsetsUnchanged: before === after,
          origin: stateAfter.origin,
          invalidMessage: invalidText, keptValue: kept,
          invalidRejected: invalid && invalid.lastInputRejected === true,
        };
      })()`,
    );
    record(
      "BPM edit changes only the beat grid (0.25s at 240 BPM), onsets untouched",
      CLOSE(bpmCheck.interval, 0.25, 1e-9) && bpmCheck.onsetsUnchanged && bpmCheck.beats > 0,
      { interval: bpmCheck.interval, beats: bpmCheck.beats, onsetsUnchanged: bpmCheck.onsetsUnchanged },
    );
    record(
      "invalid BPM (0) is rejected with a readable message and keeps the previous grid",
      bpmCheck.invalidRejected && /正数/.test(bpmCheck.invalidMessage) && bpmCheck.keptValue === 240,
      { invalidMessage: bpmCheck.invalidMessage, keptValue: bpmCheck.keptValue },
    );
    screenshots.push(await screenshot(client, "02-bpm-grid-240"));

    // ---- wheel zoom anchored at the pointer -----------------------------
    // scroll the canvas into the viewport first: the CDP wheel event is
    // dispatched at real viewport coordinates (the added global-seek row
    // shifts the timeline down, so this keeps the pointer ON the canvas)
    await evaluate(client, `(() => { document.getElementById('canvas-stack').scrollIntoView({ block: 'center' }); return true; })()`);
    await sleep(120);
    const stackBox = await evaluate(
      client,
      "(() => { const r = document.getElementById('canvas-stack').getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; })()",
    );
    const anchorX = Math.round(stackBox.x + stackBox.width * 0.4);
    const anchorBefore = await evaluate(client, `window.__aatViewer.timeAtPx(${Math.round(stackBox.width * 0.4)})`);
    await client.send("Input.dispatchMouseEvent", {
      type: "mouseWheel",
      x: anchorX,
      y: Math.round(stackBox.y + 60),
      deltaX: 0,
      deltaY: -240,
    });
    await sleep(120);
    const zoomResult = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        const s = api.state();
        return {
          anchorAfter: api.timeAtPx(${Math.round(stackBox.width * 0.4)}),
          start: s.view.start, end: s.view.end,
          anchorExpected: ${anchorBefore},
        };
      })()`,
    );
    record(
      "wheel zoom keeps the time under the pointer as the anchor",
      CLOSE(zoomResult.anchorAfter, zoomResult.anchorExpected, 1e-6) && zoomResult.end - zoomResult.start < 12,
      { before: zoomResult.anchorExpected, after: zoomResult.anchorAfter, window: [zoomResult.start, zoomResult.end] },
    );
    screenshots.push(await screenshot(client, "03-zoom-anchored"));

    // zoom far out -> clamped to [0, duration]; zoom far in -> min window
    const clamped = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        for (let i = 0; i < 40; i += 1) api.zoomAt(2, 10);
        const out = api.state().view;
        for (let i = 0; i < 80; i += 1) api.zoomAt(0.5, 10);
        const inn = api.state().view;
        api.fit();
        return { out, inn };
      })()`,
    );
    record(
      "view window is constrained: never before 0, never after the audio end, min duration kept",
      clamped.out.start === 0 && CLOSE(clamped.out.end, 12, 1e-9) && clamped.inn.end > clamped.inn.start && clamped.inn.start >= 0,
      clamped,
    );

    // pan + fit/reset
    const panCheck = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        api.setView(2, 8);
        api.panPx(200);            // drag right: window looks earlier
        const panned = api.state().view;
        api.panPx(-100000);        // drag left: clamps at the file end
        const atEnd = api.state().view;
        api.panPx(100000);         // drag right: clamps at 0
        const atStart = api.state().view;
        api.reset();
        const reset = api.state().view;
        const bpm = api.state().bpm;
        return { panned, atEnd, atStart, reset, bpm: bpm.value };
      })()`,
    );
    record(
      "pan/fit/reset behave and clamp at the file bounds",
      panCheck.panned.start < 2 &&
        panCheck.panned.start >= 0 &&
        CLOSE(panCheck.atEnd.end, 12, 1e-9) &&
        panCheck.atStart.start === 0 &&
        panCheck.reset.start === 0 &&
        panCheck.bpm === 120,
      panCheck,
    );

    // ---- playback / seek / playhead ------------------------------------
    const seekCheck = await evaluate(
      client,
      `(() => { window.__aatViewer.seek(3); return window.__aatViewer.playheadTime(); })()`,
    );
    await sleep(200);
    const seekAfter = await evaluate(client, "window.__aatViewer.playheadTime()");
    record("seek moves the playhead to the requested audio second", CLOSE(seekAfter, 3, 0.25), { seekAfter });

    await evaluate(client, "window.__aatViewer.play()");
    await sleep(900);
    const playCheck = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        const t = api.playheadTime();
        const x = api.pixelOf(t);
        const width = document.getElementById('canvas-stack').clientWidth;
        const s = api.state();
        const expected = ((t - s.view.start) / (s.view.end - s.view.start)) * width;
        return { t, x, expected, playing: s.playing };
      })()`,
    );
    await evaluate(client, "window.__aatViewer.pause()");
    record(
      "playback advances the playhead and the playhead line follows audio seconds",
      playCheck.playing === true && playCheck.t > 3.3 && CLOSE(playCheck.x, playCheck.expected, 0.01),
      playCheck,
    );

    // ---- malicious labels ----------------------------------------------
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-malicious-labels.json"));
    await sleep(150);
    const xssCheck = await evaluate(
      client,
      `(() => {
        const gutter = document.getElementById('gutter-rows');
        return {
          xss: window.__xss ?? null,
          hasImgElement: gutter.querySelectorAll('img').length,
          hasScriptElement: gutter.querySelectorAll('script').length,
          text: gutter.textContent,
          notesText: document.getElementById('issues').textContent,
        };
      })()`,
    );
    record(
      "malicious labels render as plain text (textContent/Canvas), no HTML execution",
      xssCheck.xss === null && xssCheck.hasImgElement === 0 && xssCheck.hasScriptElement === 0 && xssCheck.text.includes("<img src=x"),
      { xss: xssCheck.xss, imgs: xssCheck.hasImgElement },
    );
    screenshots.push(await screenshot(client, "04-malicious-labels-plain-text"));

    // ---- unknown BPM / empty events ------------------------------------
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-unknown-tempo.json"));
    const unknownBpm = await evaluate(
      client,
      `(() => {
        const s = window.__aatViewer.state();
        return {
          value: document.getElementById('bpm-input').value,
          placeholder: document.getElementById('bpm-input').placeholder,
          text: document.getElementById('bpm-status').textContent,
          unknown: s.bpm.unknown,
          grid: window.__aatViewer.rowGrid(),
        };
      })()`,
    );
    record(
      "tempo.bpm=null is surfaced as explicit unknown, no grid, human can type a BPM",
      unknownBpm.unknown === true && unknownBpm.value === "" && unknownBpm.placeholder === "unknown" && /未知/.test(unknownBpm.text) && unknownBpm.grid.beats.length === 0,
      { text: unknownBpm.text },
    );
    const manualBpm = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        api.setBpm(90);
        const grid = api.rowGrid();
        return { interval: grid.interval, beats: grid.beats.length, origin: api.state().bpm.origin };
      })()`,
    );
    record("manual BPM on an unknown-tempo document builds the grid", CLOSE(manualBpm.interval, 60 / 90, 1e-9) && manualBpm.beats > 0, manualBpm);

    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-empty-events.json"));
    const emptyCheck = await evaluate(
      client,
      `(() => {
        const s = window.__aatViewer.state();
        return { rows: s.doc.rows.length, total: s.doc.totalEvents, visible: s.lastRender.visibleEvents };
      })()`,
    );
    record("empty events are a readable state (rows render, zero events)", emptyCheck.rows === 5 && emptyCheck.total === 0 && emptyCheck.visible === 0, emptyCheck);

    // ---- error states: bad JSON, bad path, audio mismatch ---------------
    await loadJsonFile(client, badJson);
    await waitFor(client, "error state", "document.getElementById('status').className === 'error'");
    const badJsonCheck = await evaluate(
      client,
      `(() => ({
        status: document.getElementById('status').textContent,
        issues: document.getElementById('issues').textContent,
      }))()`,
    );
    record(
      "malformed JSON produces a readable error (code + pointer, textContent)",
      /E_PARSE/.test(badJsonCheck.issues) && /校验/.test(badJsonCheck.status),
      { status: badJsonCheck.status },
    );
    screenshots.push(await screenshot(client, "05-error-state"));

    await loadJsonFile(client, badPathJson);
    await waitFor(client, "path error", "/E_PATH/.test(document.getElementById('issues').textContent)");
    const pathCheck = await evaluate(client, "document.getElementById('issues').textContent");
    record("unsafe audio path (../) is rejected with E_PATH, no fetch is attempted", /E_PATH/.test(pathCheck), { issues: pathCheck.slice(0, 160) });

    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    await loadAudioFile(client, wrongWav);
    await waitFor(client, "mismatch notes", "window.__aatViewer.state().audio.notes.length > 0", 30000);
    const mismatchCheck = await evaluate(
      client,
      `(() => ({
        notes: window.__aatViewer.state().audio.notes,
        status: document.getElementById('status').textContent,
      }))()`,
    );
    const codes = mismatchCheck.notes.map((n) => n.code);
    record(
      "audio mismatch (hash + duration) is an explicit error, never a silent retime",
      codes.includes("A_HASH") && codes.includes("A_DURATION") && /不重定时|不一致/.test(mismatchCheck.status),
      { codes, status: mismatchCheck.status },
    );

    // ---- resize / DPR ---------------------------------------------------
    await client.send("Emulation.setDeviceMetricsOverride", { width: 900, height: 700, deviceScaleFactor: 2, mobile: false });
    await sleep(300);
    const dprCheck = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        api.renderNow ? api.renderNow() : null;
        const m = api.canvasMetrics();
        return { m, width: document.getElementById('canvas-stack').clientWidth };
      })()`,
    );
    record(
      "resize + devicePixelRatio=2: canvas backing store matches CSS size * dpr",
      dprCheck.m.dpr === 2 && dprCheck.m.track.width === Math.round(dprCheck.width * 2),
      dprCheck,
    );
    await client.send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
    await sleep(200);

    // ---- lifecycle / ownership / byte-entry / stem identity / clip -----
    // stale stems must not survive a document change
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    await loadAudioFile(client, join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME));
    await setFileInput(client, "#stem-input", [join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME)]);
    await waitFor(client, "demo stem applied", "window.__aatViewer.state().audio.resource.stemsLoaded === 1", 30000);
    await loadJsonFile(client, stress.jsonPath);
    const staleStems = await evaluate(
      client,
      `(() => ({
        stems: window.__aatViewer.state().audio.resource.stemsLoaded,
        stemPeaks: window.__aatViewer.state().audio.stemMapping.length,
        notes: document.getElementById('notes').textContent,
      }))()`,
    );
    record(
      "document change clears/re-validates stems by (rowId, filename, hash) — no stale waveform on new rows",
      staleStems.stems === 0 && staleStems.stemPeaks === 0 && /STEM_DROPPED|已清除/.test(staleStems.notes),
      staleStems,
    );

    // invalid JSON after a valid one must never keep showing the stale graph
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    await waitFor(client, "demo back", "window.__aatViewer.state().doc !== null");
    await loadJsonFile(client, badJson);
    const cleared = await evaluate(
      client,
      `(() => ({
        doc: window.__aatViewer.state().doc,
        gutter: document.querySelectorAll('#gutter-rows .gutter-row').length,
        status: document.getElementById('status').textContent,
        rows: (window.__aatViewer.state().lastRender || { rows: [] }).rows.length,
      }))()`,
    );
    record(
      "invalid JSON after valid clears the view (stale graph is never labelled as the new file)",
      cleared.doc === null && cleared.gutter === 0 && cleared.rows === 0 && /已清除旧数据/.test(cleared.status),
      cleared,
    );

    // overlapping async loads: the LATER selection must win (generation guard)
    const loadsBefore = await evaluate(client, "window.__aatViewer.state().loads.json");
    await setFileInput(client, "#json-input", [stress.jsonPath]); // slow (~1s validate)
    await sleep(60);
    await setFileInput(client, "#json-input", [join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json")]); // fast
    await waitFor(client, "both json loads finished", `window.__aatViewer.state().loads.json >= ${loadsBefore + 2}`, 60000);
    await sleep(1500); // give the slow load time to (not) overwrite
    const raced = await evaluate(
      client,
      `(() => ({ totalEvents: window.__aatViewer.state().doc ? window.__aatViewer.state().doc.totalEvents : null }))()`,
    );
    record("overlapping async JSON loads: the latest selection wins (no slow-load overwrite)", raced.totalEvents === 72, raced);

    // byte-level entry: BOM / invalid UTF-8 must be controlled E_PARSE
    await loadJsonFile(client, bytesBomJson);
    const bomCheck = await evaluate(
      client,
      "(() => ({ issues: document.getElementById('issues').textContent, status: document.getElementById('status').textContent }))()",
    );
    record(
      "raw BOM bytes are rejected at the file entry (E_PARSE), File.text() masking is bypassed",
      /E_PARSE/.test(bomCheck.issues) && /BOM/.test(bomCheck.issues) && /字节不符合严格 UTF-8/.test(bomCheck.status),
      { issues: bomCheck.issues.slice(0, 120) },
    );
    await loadJsonFile(client, bytesBadUtf8Json);
    const badUtf8Check = await evaluate(client, "(() => document.getElementById('issues').textContent)()");
    record(
      "raw invalid-UTF-8 bytes are rejected at the file entry (E_PARSE, no U+FFFD silent pass)",
      /E_PARSE/.test(badUtf8Check) && /UTF-8/.test(badUtf8Check),
      { issues: badUtf8Check.slice(0, 120) },
    );

    // stem identity: same basename (3× target.wav) disambiguated by SHA-256
    await loadJsonFile(client, stemDocJson);
    await loadAudioFile(client, join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME));
    const stemPick = async (path) => {
      const before = await evaluate(client, "window.__aatViewer.state().loads.stems");
      await setFileInput(client, "#stem-input", [path]);
      await waitFor(client, `stem pick ${path}`, `window.__aatViewer.state().loads.stems > ${before}`, 60000);
      return evaluate(
        client,
        `(() => ({
          mapping: window.__aatViewer.state().audio.stemMapping,
          stems: window.__aatViewer.state().audio.resource.stemsLoaded,
          notes: document.getElementById('notes').textContent,
        }))()`,
      );
    };
    const pickB = await stemPick(stemB.file);
    record(
      "same-basename stems (3× target.wav): SHA-256 identity picks the RIGHT row (b-row, not the first)",
      pickB.stems === 1 && pickB.mapping.length === 1 && pickB.mapping[0].rowId === "b-row" && /STEM_OK/.test(pickB.notes) && /b-row/.test(pickB.notes),
      pickB,
    );
    const pickA = await stemPick(stemA.file);
    record(
      "second same-basename stem maps to a-row (per-row stem mapping)",
      pickA.stems === 2 && pickA.mapping.some((entry) => entry.rowId === "a-row") && pickA.mapping.some((entry) => entry.rowId === "b-row"),
      pickA,
    );
    const pickWrongHash = await stemPick(stemC.file);
    record(
      "wrong-hash same-basename stem is refused (no render, no STEM_OK)",
      pickWrongHash.stems === 2 && /STEM_IDENTITY/.test(pickWrongHash.notes),
      { stems: pickWrongHash.stems, hasIdentityError: /STEM_IDENTITY/.test(pickWrongHash.notes) },
    );
    const pickWrongDuration = await stemPick(stemShort.file);
    record(
      "stem with wrong time-axis duration is refused (STEM_DURATION, never auto-shifted)",
      pickWrongDuration.stems === 2 && /STEM_DURATION/.test(pickWrongDuration.notes),
      { stems: pickWrongDuration.stems, hasDurationError: /STEM_DURATION/.test(pickWrongDuration.notes) },
    );

    // sustained events are clipped to the viewport, not dropped at the edges
    await loadJsonFile(client, clipDocJson);
    await loadAudioFile(client, join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME));
    const clipCheck = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        api.setView(5, 6); const crossing = api.state().lastRender;   // onset 0 + dur 10 crosses the window
        api.setView(10, 11); const boundary = api.state().lastRender; // ends exactly at view start
        api.setView(11, 12); const expired = api.state().lastRender;  // fully expired
        return {
          crossing: { visible: crossing.visibleEvents, drawn: crossing.drawnEvents, clipped: crossing.clippedEvents },
          boundary: { visible: boundary.visibleEvents, drawn: boundary.drawnEvents },
          expired: { visible: expired.visibleEvents, drawn: expired.drawnEvents },
        };
      })()`,
    );
    record(
      "sustained event crossing the left edge is clipped and drawn (not dropped); expired events are not visible",
      clipCheck.crossing.visible === 1 &&
        clipCheck.crossing.drawn === 1 &&
        clipCheck.crossing.clipped === 1 &&
        clipCheck.boundary.visible === 1 &&
        clipCheck.expired.visible === 0,
      clipCheck,
    );
    screenshots.push(await screenshot(client, "08-sustained-clip"));

    // ---- cross-document / cross-kind async ownership --------------------
    // (a) pending audio of doc A must never commit after doc B is accepted
    await loadJsonFile(client, stress.jsonPath);
    const audioBeforeA = await evaluate(client, "window.__aatViewer.state().loads.audio");
    await setFileInput(client, "#audio-input", [stress.wavPath]); // slow decode (~0.4s+)
    await sleep(60);
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json")); // doc changes mid-decode
    await waitFor(client, "audio attempt settled", `window.__aatViewer.state().loads.audio > ${audioBeforeA}`, 60000);
    await sleep(1500);
    const staleAudio = await evaluate(
      client,
      `(() => ({
        hashHex: window.__aatViewer.state().audio.hashHex,
        docHash: window.__aatViewer.state().doc.audioSha256,
        duration: window.__aatViewer.state().audio.durationSeconds,
        mainLoaded: window.__aatViewer.resourceState().mainLoaded,
        status: document.getElementById('status').textContent,
      }))()`,
    );
    record(
      "pending audio of the OLD document is aborted on doc change (no stale commit / no false 'matches JSON')",
      // the pending stress.wav (300s, other hash) must NOT be committed; only a
      // previously committed + revalidated audio may be present
      staleAudio.mainLoaded === true &&
        staleAudio.hashHex === staleAudio.docHash &&
        staleAudio.duration !== 300 &&
        !(staleAudio.hashHex || "").startsWith("36e6571a"),
      staleAudio,
    );

    // (b) concurrent main + stem selections must BOTH complete (separate tokens)
    const concBefore = await evaluate(client, "(() => ({ a: window.__aatViewer.state().loads.audio, s: window.__aatViewer.state().loads.stems }))()");
    await setFileInput(client, "#audio-input", [join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME)]);
    await setFileInput(client, "#stem-input", [join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME)]);
    await waitFor(
      client,
      "both concurrent loads settled",
      `window.__aatViewer.state().loads.audio > ${concBefore.a} && window.__aatViewer.state().loads.stems > ${concBefore.s}`,
      60000,
    );
    await sleep(600);
    const concurrent = await evaluate(
      client,
      `(() => ({ loaded: window.__aatViewer.state().audio.loaded, stems: window.__aatViewer.resourceState().stemsLoaded }))()`,
    );
    record(
      "concurrent main-audio + stem selections both complete (stem no longer cancels the main decode)",
      concurrent.loaded === true && concurrent.stems === 1,
      concurrent,
    );

    // (c) invalid audio after a valid one never keeps the prior file playable
    const notAudio = join(workDir, "not-audio.wav");
    await writeFile(notAudio, "this is not decodable audio\n");
    await loadAudioFile(client, notAudio);
    await sleep(300);
    const badAudio = await evaluate(
      client,
      `(() => ({
        loaded: window.__aatViewer.state().audio.loaded,
        mainLoaded: window.__aatViewer.resourceState().mainLoaded,
        audioSrc: window.__aatViewer.resourceState().audioSrc,
        status: document.getElementById('status').textContent,
      }))()`,
    );
    record(
      "invalid audio after valid clears the playable state (prior file never plays under the new name)",
      badAudio.loaded === false && badAudio.mainLoaded === false && badAudio.audioSrc === null && /解码失败/.test(badAudio.status),
      badAudio,
    );

    // ---- peaks worker failure fallback (no fake flat waveform) ----------
    await evaluate(
      client,
      `(() => {
        const RealWorker = window.Worker;
        window.__realWorker = RealWorker;
        window.Worker = class {
          constructor() { this.listeners = {}; }
          addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); }
          removeEventListener(type, handler) { this.listeners[type] = (this.listeners[type] || []).filter((h) => h !== handler); }
          postMessage(message, transfer = []) {
            // REAL transfer semantics: detach the caller's buffers, then fail
            const channel = new MessageChannel();
            try { channel.port2.postMessage(message, transfer); } catch (error) { /* detached */ }
            channel.port1.close(); channel.port2.close();
            setTimeout(() => { for (const h of this.listeners.error || []) h({ message: 'forced worker failure' }); }, 5);
          }
          terminate() {}
        };
        return true;
      })()`,
    );
    await loadAudioFile(client, join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME));
    await sleep(400);
    const fallbackPeaks = await evaluate(
      client,
      `(() => ({ peakMax: window.__aatViewer.waveformPeakMax(), loaded: window.__aatViewer.state().audio.loaded }))()`,
    );
    await evaluate(client, "(() => { window.Worker = window.__realWorker; return true; })()");
    record(
      "peaks worker transfer+failure falls back to the ORIGINAL samples (not an all-zero flat waveform)",
      fallbackPeaks.loaded === true && typeof fallbackPeaks.peakMax === "number" && fallbackPeaks.peakMax > 0.2,
      fallbackPeaks,
    );

    // ---- huge (but valid) duration stays responsive (bounded ruler) -----
    const hugeDoc = join(workDir, "huge-duration.json");
    const huge = demoDoc(demo.wavSha256, { emptyEvents: true });
    huge.audio.duration_seconds = Number.MAX_VALUE;
    huge.instruments = [
      {
        id: "huge-row",
        label: "huge range",
        description: "schema-valid Number.MAX_VALUE duration (bounded ruler/grid regression)",
        source: "mock",
        confidence: 0.5,
        events: [
          { id: "huge-ev-1", onset_seconds: 0 },
          { id: "huge-ev-2", onset_seconds: 1e300 },
        ],
      },
    ];
    await writeFile(hugeDoc, JSON.stringify(huge));
    const hugeStart = Date.now();
    await loadJsonFile(client, hugeDoc);
    await waitFor(client, "huge doc loaded", "window.__aatViewer.state().doc !== null", 15000);
    const hugeState = await evaluate(
      client,
      `(() => {
        const s = window.__aatViewer.state();
        return { duration: s.doc.durationSeconds, rendered: s.lastRender !== null, metrics: document.getElementById('metrics').textContent };
      })()`,
    );
    const hugeElapsed = Date.now() - hugeStart;
    record(
      "Number.MAX_VALUE duration: page stays responsive (bounded ruler), grid reports readable unsupported",
      hugeState.duration === Number.MAX_VALUE &&
        hugeState.rendered === true &&
        /网格不支持/.test(hugeState.metrics) &&
        hugeElapsed < 10000,
      { elapsedMs: hugeElapsed, duration: hugeState.duration, metrics: hugeState.metrics.slice(0, 160) },
    );

    // ---- 100k stress ----------------------------------------------------
    await loadJsonFile(client, stress.jsonPath);
    await waitFor(client, "stress doc loaded", "window.__aatViewer.state().doc !== null", 60000);
    await loadAudioFile(client, stress.wavPath);
    await waitFor(client, "stress audio loaded", "window.__aatViewer.state().audio.loaded === true", 60000);
    const stressFull = await evaluate(client, `window.__aatViewer.setView(0, 300)`);
    void stressFull;
    await sleep(400);
    await evaluate(client, "window.__aatViewer.startFrameProbe(60)");
    await sleep(1200);
    const stressStats = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        const s = api.state();
        return {
          totalEvents: s.doc.totalEvents,
          metrics: s.metrics,
          render: s.lastRender,
          domNodes: document.querySelectorAll('*').length,
          rows: s.doc.rows.length,
        };
      })()`,
    );
    const domLean = stressStats.domNodes < 400;
    const visibleSum = stressStats.render.rows.reduce((sum, row) => sum + row.visible, 0);
    const frames = stressStats.metrics.frameIntervalStartToStartMs;
    const frameRenders = stressStats.metrics.frameFullRenderMs || [];
    const avgFrame = frames.length ? frames.reduce((a, b) => a + b, 0) / frames.length : null;
    const avgRender = frameRenders.length ? frameRenders.reduce((a, b) => a + b, 0) / frameRenders.length : null;
    record(
      "100k events: viewport culling + LOD keeps DOM lean and visible counts reported",
      stressStats.totalEvents === options.stressEvents && domLean && visibleSum > 0,
      {
        totalEvents: stressStats.totalEvents,
        domNodes: stressStats.domNodes,
        visibleSum,
        candidates: stressStats.render.candidateEvents,
        drawn: stressStats.render.drawnEvents,
        aggregatedRows: stressStats.render.aggregatedRows,
        rows: stressStats.render.rows.map((r) => `${r.id}: visible=${r.visible} drawn=${r.drawn}`),
      },
    );
    record(
      "100k events: per-dataset measured timings (frame probe re-renders the full scene each sample)",
      typeof stressStats.metrics.validateMs === "number" &&
        typeof stressStats.metrics.lastRenderMs === "number" &&
        typeof stressStats.metrics.firstRenderMs === "number" &&
        frameRenders.length === 60,
      {
        validateMs: stressStats.metrics.validateMs,
        decodeMs: stressStats.metrics.decodeMs,
        firstRenderMs_currentDataset: stressStats.metrics.firstRenderMs,
        lastRenderMs: stressStats.metrics.lastRenderMs,
        frameIntervalStartToStartAvgMs: avgFrame,
        frameIntervalStartToStartMaxMs: frames.length ? Math.max(...frames) : null,
        frameFullRenderAvgMs: avgRender,
        frameFullRenderMaxMs: frameRenders.length ? Math.max(...frameRenders) : null,
        frameSamples: frames.length,
        measurement: "each sampled frame performs a FULL scene re-render (ruler+waveform+envelopes+rows+metrics); interval = frame-start to frame-start (includes the previous frame's render work)",
      },
    );
    screenshots.push(await screenshot(client, "06-stress-100k"));

    // zoom into a busy region for a second stress screenshot
    await evaluate(client, `(() => { const api = window.__aatViewer; api.setView(100, 102); })()`);
    await sleep(300);
    const zoomedStress = await evaluate(
      client,
      `(() => {
        const s = window.__aatViewer.state();
        return { render: s.lastRender, view: s.view };
      })()`,
    );
    record(
      "100k events zoomed-in: only the visible subset is drawn (binary search culling)",
      zoomedStress.render.visibleEvents > 0 && zoomedStress.render.visibleEvents < stressStats.totalEvents,
      { view: zoomedStress.view, visible: zoomedStress.render.visibleEvents, drawn: zoomedStress.render.drawnEvents },
    );
    screenshots.push(await screenshot(client, "07-stress-zoomed"));

    // ---- per-dataset frame arrays: no stale probe samples ----------------
    // (regression: the stress probe's 59 samples used to survive into the next
    // dataset's footer because the reset still cleared the pre-rename names)
    const frameCount = "(() => ({ intervals: window.__aatViewer.state().metrics.frameIntervalStartToStartMs.length, renders: window.__aatViewer.state().metrics.frameFullRenderMs.length }))()";
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    const freshDocFrames = await evaluate(client, frameCount);
    record(
      "frame arrays reset on a NEW document (previous dataset's probe samples are gone)",
      freshDocFrames.intervals === 0 && freshDocFrames.renders === 0,
      freshDocFrames,
    );
    await loadAudioFile(client, join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME));
    const freshAudioFrames = await evaluate(client, frameCount);
    record(
      "frame arrays stay empty on NEW audio until a fresh probe runs",
      freshAudioFrames.intervals === 0 && freshAudioFrames.renders === 0,
      freshAudioFrames,
    );

    // ---- global whole-file seek slider (issue #41) ---------------------
    // state here: demo-track.json (12s declared) + demo wav (12s real)
    const seekLoaded = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
    record(
      "global slider: min=0 / max=REAL loaded audio duration (12s file), enabled",
      seekLoaded.disabled === false && seekLoaded.min === 0 && CLOSE(seekLoaded.max, 12, 1e-6) && CLOSE(seekLoaded.realDurationSeconds, 12, 1e-6),
      seekLoaded,
    );

    // zoom / pan must never rescale or move the slider
    const zoomInvariant = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        api.seek(6);
        const before = api.globalSeek();
        api.setView(4, 6);
        api.zoomAt(0.5, 300);
        api.panPx(-120);
        api.panPx(80);
        const after = api.globalSeek();
        return { before, after, view: api.state().view };
      })()`,
    );
    record(
      "zoom/pan never rescales or moves the global slider (range and value stay put)",
      zoomInvariant.before.min === zoomInvariant.after.min &&
        zoomInvariant.before.max === zoomInvariant.after.max &&
        CLOSE(zoomInvariant.after.value, 6, 0.05) &&
        CLOSE(zoomInvariant.before.value, zoomInvariant.after.value, 0.05),
      zoomInvariant,
    );

    // real mouse on the slider track: whole-file seeks leave the view alone
    const viewFrozen = await evaluate(client, "(() => { window.__aatViewer.setView(3, 5); return window.__aatViewer.state().view; })()");
    const seekRows = [];
    for (const target of [
      { ratio: 0, expected: 0 },
      { ratio: 0.5, expected: 6 },
      { ratio: 1, expected: 12 },
    ]) {
      await sliderClick(client, target.ratio);
      await sleep(150);
      const row = await evaluate(
        client,
        `(() => {
          const api = window.__aatViewer;
          return { seek: api.globalSeek(), playhead: api.playheadTime(), view: api.state().view };
        })()`,
      );
      seekRows.push({ ...target, ...row });
    }
    record(
      "global slider seeks the WHOLE file (0 / 6 / 12 of 12s) and never pans/zooms the time window",
      seekRows.every((row) => CLOSE(row.seek.value, row.expected, 0.3) && CLOSE(row.playhead, row.expected, 0.35)) &&
        seekRows.every((row) => CLOSE(row.view.start, viewFrozen.start, 1e-9) && CLOSE(row.view.end, viewFrozen.end, 1e-9)),
      seekRows,
    );

    // keyboard: Home / End / arrows seek the whole file (native range keys)
    await evaluate(client, `(() => { document.getElementById('global-seek').focus(); return true; })()`);
    const keyRows = [];
    const grabSeek = () =>
      evaluate(client, `(() => { const api = window.__aatViewer; return { seek: api.globalSeek(), playhead: api.playheadTime(), view: api.state().view }; })()`);
    await pressKey(client, { key: "End", code: "End", windowsVirtualKeyCode: 35 });
    await sleep(120);
    keyRows.push({ key: "End", ...(await grabSeek()) });
    await pressKey(client, { key: "ArrowLeft", code: "ArrowLeft", windowsVirtualKeyCode: 37 });
    await sleep(120);
    keyRows.push({ key: "ArrowLeft", ...(await grabSeek()) });
    await pressKey(client, { key: "Home", code: "Home", windowsVirtualKeyCode: 36 });
    await sleep(120);
    keyRows.push({ key: "Home", ...(await grabSeek()) });
    await pressKey(client, { key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39 });
    await sleep(120);
    keyRows.push({ key: "ArrowRight", ...(await grabSeek()) });
    record(
      "keyboard Home/End/arrows drive the global slider (End=file end, Home=file start), view untouched",
      CLOSE(keyRows[0].seek.value, 12, 0.05) &&
        CLOSE(keyRows[1].seek.value, keyRows[0].seek.value - GLOBAL_SEEK_STEP_SECONDS, 0.02) &&
        CLOSE(keyRows[2].seek.value, 0, 0.05) &&
        CLOSE(keyRows[3].seek.value, GLOBAL_SEEK_STEP_SECONDS, 0.02) &&
        keyRows.every((row) => CLOSE(row.view.start, viewFrozen.start, 1e-9) && CLOSE(row.view.end, viewFrozen.end, 1e-9)),
      keyRows,
    );

    // playback keeps the slider in sync with the audio clock
    await evaluate(client, "(() => { window.__aatViewer.seek(2); return true; })()");
    await evaluate(client, "window.__aatViewer.play()");
    await sleep(800);
    const playSync = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        const s = api.globalSeek();
        return { value: s.value, playhead: api.playheadTime(), playing: api.state().playing, labelText: s.labelText, aria: s.ariaValueText };
      })()`,
    );
    await evaluate(client, "window.__aatViewer.pause()");
    record(
      "playback updates the global slider value + elapsed/duration label with the audio clock",
      playSync.playing === true && playSync.playhead > 2.2 && CLOSE(playSync.value, playSync.playhead, 0.15) && /已播/.test(playSync.labelText) && /全长/.test(playSync.aria),
      playSync,
    );

    // onsets / BPM must be untouched by any seek
    const seekInvariants = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        const onsetsBefore = JSON.stringify(api.onsetsSnapshot());
        const bpmBefore = JSON.stringify(api.state().bpm);
        api.seek(1);
        api.seek(7);
        api.seek(11);
        api.seek(0);
        return {
          onsetsUnchanged: onsetsBefore === JSON.stringify(api.onsetsSnapshot()),
          bpmUnchanged: bpmBefore === JSON.stringify(api.state().bpm),
        };
      })()`,
    );
    record("global seeks never change onsets or BPM", seekInvariants.onsetsUnchanged && seekInvariants.bpmUnchanged, seekInvariants);

    // different-duration file: the slider follows the REAL file, not the JSON
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    await loadAudioFile(client, wrongWav); // 3s real file vs 12s JSON claim
    await waitFor(client, "mismatch notes for wrong duration", "window.__aatViewer.state().audio.notes.length > 0", 30000);
    const shortFile = await evaluate(
      client,
      `(() => {
        const api = window.__aatViewer;
        api.seek(999); // must clamp to the REAL 3s file end, never to the JSON's 12s
        return { seek: api.globalSeek(), playhead: api.playheadTime(), codes: api.state().audio.notes.map((n) => n.code) };
      })()`,
    );
    record(
      "different-duration file: slider range = REAL 3s file (not the 12s JSON claim), mismatch stays explicit",
      shortFile.seek.disabled === false &&
        CLOSE(shortFile.seek.max, 3, 0.05) &&
        shortFile.seek.value <= shortFile.seek.max + 1e-9 &&
        CLOSE(shortFile.playhead, 3, 0.1) &&
        shortFile.codes.includes("A_DURATION"),
      shortFile,
    );

    // replacing the file updates the range to the new real duration
    await loadAudioFile(client, join(VIEWER_ROOT, "fixtures", "generated", DEMO_WAV_NAME));
    const replaced = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
    record(
      "replacing the audio updates the slider range to the new real duration (3s -> 12s)",
      replaced.disabled === false && CLOSE(replaced.max, 12, 0.05) && replaced.value <= replaced.max + 1e-9,
      replaced,
    );

    // decode failure disables the slider and clears the playable state
    const notAudioSeek = join(workDir, "not-audio-global-seek.wav");
    await writeFile(notAudioSeek, "this is not decodable audio\n");
    await loadAudioFile(client, notAudioSeek);
    await sleep(300);
    const disabledAfterBad = await evaluate(
      client,
      `(() => ({ seek: window.__aatViewer.globalSeek(), resource: window.__aatViewer.resourceState() }))()`,
    );
    record(
      "decode failure disables the global slider (no stale range/value from the previous file)",
      disabledAfterBad.seek.disabled === true && disabledAfterBad.resource.mainLoaded === false && disabledAfterBad.resource.rafActive === false,
      disabledAfterBad,
    );

    // ---- fractional endpoint + sub-step file (PR #43 review contract) --
    // real decodable files: 16.037s (not on the 0.1 grid) and 0.05s (shorter
    // than one arrow step). The DOM step is "any" so the raw value can sit
    // EXACTLY on the real file end; keyboard stepping is explicit (0.1s).
    const fractionalWav = join(workDir, "fractional-16.037.wav");
    await writeFile(fractionalWav, buildDemoWav({ seconds: 16.037, events: [] }));
    const tinyWav = join(workDir, "tiny-0.05.wav");
    await writeFile(tinyWav, buildDemoWav({ seconds: 0.05, events: [] }));
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    await loadAudioFile(client, fractionalWav);
    const fracLoaded = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
    record(
      "fractional file (16.037s): slider max keeps the fraction (no step sanitization to 16.000)",
      fracLoaded.disabled === false &&
        fracLoaded.domStep === "any" &&
        fracLoaded.realDurationSeconds > 16.03 &&
        fracLoaded.max > 16.03 &&
        CLOSE(fracLoaded.max, fracLoaded.realDurationSeconds, 1e-9),
      fracLoaded,
    );
    await evaluate(client, `(() => { document.getElementById('global-seek').focus(); return true; })()`);
    await pressKey(client, { key: "End", code: "End", windowsVirtualKeyCode: 35 });
    await sleep(120);
    const fracEnd = await evaluate(
      client,
      `(() => { const api = window.__aatViewer; const s = api.globalSeek(); return { value: s.value, max: s.max, real: s.realDurationSeconds, playhead: api.playheadTime() }; })()`,
    );
    record(
      "fractional file: keyboard End lands EXACTLY on the real file end (16.037…, not 16.000)",
      fracEnd.value > 16.03 &&
        Math.abs(fracEnd.value - 16) > 0.03 &&
        CLOSE(fracEnd.value, fracEnd.real, ENDPOINT_EPS) &&
        CLOSE(fracEnd.value, fracEnd.max, ENDPOINT_EPS) &&
        CLOSE(fracEnd.playhead, fracEnd.value, 1e-4),
      fracEnd,
    );
    await evaluate(client, "(() => { window.__aatViewer.seek(5); return true; })()");
    await sliderDragRightPastEnd(client);
    await sleep(150);
    const fracDrag = await evaluate(
      client,
      `(() => { const api = window.__aatViewer; const s = api.globalSeek(); return { value: s.value, real: s.realDurationSeconds, playhead: api.playheadTime() }; })()`,
    );
    record(
      "fractional file: dragging past the right edge reaches the exact real file end",
      fracDrag.value > 16.03 &&
        Math.abs(fracDrag.value - 16) > 0.03 &&
        CLOSE(fracDrag.value, fracDrag.real, ENDPOINT_EPS) &&
        CLOSE(fracDrag.playhead, fracDrag.value, 1e-4),
      fracDrag,
    );
    await evaluate(client, "(() => { window.__aatViewer.seek(16.0); return true; })()");
    await pressKey(client, { key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39 });
    await sleep(120);
    const fracArrow = await evaluate(
      client,
      "(() => { const s = window.__aatViewer.globalSeek(); return { value: s.value, real: s.realDurationSeconds }; })()",
    );
    record(
      "fractional file: an arrow step from 16.0 clamps exactly onto the 16.037s end",
      fracArrow.value > 16.03 && Math.abs(fracArrow.value - 16) > 0.03 && CLOSE(fracArrow.value, fracArrow.real, ENDPOINT_EPS),
      fracArrow,
    );
    await evaluate(client, "(() => { window.__aatViewer.seek(6); return true; })()");
    await evaluate(client, "window.__aatViewer.play()");
    await sleep(700);
    const fracPlay = await evaluate(
      client,
      `(() => { const api = window.__aatViewer; const s = api.globalSeek(); return { value: s.value, playhead: api.playheadTime(), playing: api.state().playing, max: s.max }; })()`,
    );
    await evaluate(client, "window.__aatViewer.pause()");
    record(
      "fractional file: playback keeps the slider in sync with the audio clock (max stays 16.037…)",
      fracPlay.playing === true && fracPlay.playhead > 6.2 && CLOSE(fracPlay.value, fracPlay.playhead, 0.05) && fracPlay.max > 16.03,
      fracPlay,
    );

    // tiny decodable clip: shorter than one 0.1s arrow step, still fully usable
    await loadAudioFile(client, tinyWav);
    const tinyLoaded = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
    record(
      "tiny decodable clip (0.05s): slider enabled, max = the real 0.05s (shorter than one arrow step)",
      tinyLoaded.disabled === false &&
        tinyLoaded.realDurationSeconds > 0.04 &&
        tinyLoaded.realDurationSeconds < 0.06 &&
        CLOSE(tinyLoaded.max, tinyLoaded.realDurationSeconds, 1e-9),
      tinyLoaded,
    );
    await evaluate(client, `(() => { document.getElementById('global-seek').focus(); return true; })()`);
    await pressKey(client, { key: "End", code: "End", windowsVirtualKeyCode: 35 });
    await sleep(120);
    const tinyEnd = await evaluate(
      client,
      "(() => { const api = window.__aatViewer; const s = api.globalSeek(); return { value: s.value, real: s.realDurationSeconds, playhead: api.playheadTime() }; })()",
    );
    await pressKey(client, { key: "ArrowLeft", code: "ArrowLeft", windowsVirtualKeyCode: 37 });
    await sleep(120);
    const tinyBack = await evaluate(client, "(() => window.__aatViewer.globalSeek().value)()");
    record(
      "tiny clip: End reaches the exact end (not stuck at 0) and arrows step within [0, real end]",
      tinyEnd.value > 0.04 &&
        CLOSE(tinyEnd.value, tinyEnd.real, ENDPOINT_EPS) &&
        CLOSE(tinyEnd.playhead, tinyEnd.value, 1e-4) &&
        tinyBack === 0,
      { end: tinyEnd, afterArrowLeft: tinyBack },
    );

    // ---- real #33 output integration (CLI-provided paths only) ----------
    if (options.real.json) {
      if (!options.real.audio) throw new Error("--real-json requires --real-audio");
      await loadJsonFile(client, options.real.json);
      const realDoc = await evaluate(
        client,
        `(() => {
          const s = window.__aatViewer.state();
          return s.doc
            ? {
                rows: s.doc.rows.map((r) => ({ id: r.id, label: r.label, count: r.count })),
                totalEvents: s.doc.totalEvents,
                durationSeconds: s.doc.durationSeconds,
                tempoBpm: s.doc.tempoBpm,
                tempoSource: s.doc.tempoSource,
              }
            : null;
        })()`,
      );
      record(
        "real result: 3 instrument rows (drums/bass/synthesizer), 110 raw events, tempo=null/unknown",
        realDoc !== null &&
          realDoc.rows.length === 3 &&
          realDoc.totalEvents === 110 &&
          realDoc.tempoBpm === null &&
          realDoc.tempoSource === "unknown" &&
          realDoc.rows.map((r) => r.label).join(",") === "drums,bass,synthesizer",
        realDoc,
      );
      const realBpm = await evaluate(
        client,
        `(() => {
          const api = window.__aatViewer;
          const before = api.state().bpm;
          const placeholder = document.getElementById('bpm-input').placeholder;
          api.setBpm(100);
          const grid = api.rowGrid();
          return { unknown: before.unknown, placeholder, interval: grid.interval, beats: grid.beats.length };
        })()`,
      );
      record(
        "real result: BPM unknown is surfaced and manual BPM builds the grid only",
        realBpm.unknown === true && realBpm.placeholder === "unknown" && CLOSE(realBpm.interval, 0.6, 1e-9) && realBpm.beats > 0,
        realBpm,
      );
      await loadAudioFile(client, options.real.audio);
      const realAudio = await evaluate(
        client,
        `(() => {
          const s = window.__aatViewer.state();
          return {
            hashMatch: s.audio.hashHex === s.doc.audioSha256,
            duration: s.audio.durationSeconds,
            errors: s.audio.notes.filter((n) => n.level === 'error').map((n) => n.code),
            decodeMs: s.metrics.decodeMs,
          };
        })()`,
      );
      record(
        "real audio: SHA-256 matches the result and the 16s time axis holds (no retime)",
        realAudio.hashMatch === true && realAudio.errors.length === 0 && CLOSE(realAudio.duration, 16, 0.1),
        realAudio,
      );
      const realSeekZoom = await evaluate(
        client,
        `(() => {
          const api = window.__aatViewer;
          api.seek(8);
          const t = api.playheadTime();
          const width = document.getElementById('canvas-stack').clientWidth;
          const anchorBefore = api.timeAtPx(width * 0.5);
          api.zoomAt(0.5, width * 0.5);
          const anchorAfter = api.timeAtPx(width * 0.5);
          const view = api.state().view;
          api.fit();
          return { t, anchorBefore, anchorAfter, view };
        })()`,
      );
      record(
        "real result: seek lands on audio seconds and wheel-anchor zoom keeps the anchor fixed",
        CLOSE(realSeekZoom.t, 8, 0.3) && CLOSE(realSeekZoom.anchorBefore, realSeekZoom.anchorAfter, 1e-6),
        realSeekZoom,
      );

      // ---- issue #41: global whole-file slider on the REAL 16s file -----
      const realSlider = await evaluate(
        client,
        `(() => {
          const api = window.__aatViewer;
          api.fit();
          api.seek(8);
          return api.globalSeek();
        })()`,
      );
      record(
        "real 16s file: global slider spans the REAL file 0..16 and sits at 8s (50%)",
        realSlider.disabled === false &&
          realSlider.min === 0 &&
          CLOSE(realSlider.max, 16, 0.05) &&
          CLOSE(realSlider.value, 8, 0.05) &&
          CLOSE(realSlider.ratio, 0.5, 0.005) &&
          /全长/.test(realSlider.labelText),
        realSlider,
      );
      const realZoomStable = await evaluate(
        client,
        `(() => {
          const api = window.__aatViewer;
          api.setView(4, 6);
          const width = document.getElementById('canvas-stack').clientWidth;
          api.zoomAt(0.5, width * 0.5);
          return { seek: api.globalSeek(), view: api.state().view };
        })()`,
      );
      record(
        "real: zoom into 4..6 leaves the slider range 0..16 and the value at 8 (50%) untouched",
        CLOSE(realZoomStable.seek.min, 0, 1e-9) &&
          CLOSE(realZoomStable.seek.max, 16, 0.05) &&
          CLOSE(realZoomStable.seek.value, 8, 0.05) &&
          CLOSE(realZoomStable.seek.ratio, 0.5, 0.005) &&
          realZoomStable.view.start >= 4 - 1e-6 &&
          realZoomStable.view.end <= 6 + 1e-6,
        realZoomStable,
      );
      // real mouse clicks: whole-file seeks at 0% / 50% / 100% of the 16s file
      const realOnsetsBpmBefore = await evaluate(
        client,
        `(() => ({ onsets: JSON.stringify(window.__aatViewer.onsetsSnapshot()), bpm: JSON.stringify(window.__aatViewer.state().bpm) }))()`,
      );
      const realSeekRows = [];
      for (const target of [
        { ratio: 0, expected: 0 },
        { ratio: 0.5, expected: 8 },
        { ratio: 1, expected: 16 },
      ]) {
        await sliderClick(client, target.ratio);
        await sleep(150);
        const row = await evaluate(
          client,
          `(() => {
            const api = window.__aatViewer;
            return { seek: api.globalSeek(), playhead: api.playheadTime(), view: api.state().view };
          })()`,
        );
        realSeekRows.push({ ...target, ...row });
      }
      const realOnsetsBpmAfter = await evaluate(
        client,
        `(() => ({ onsets: JSON.stringify(window.__aatViewer.onsetsSnapshot()), bpm: JSON.stringify(window.__aatViewer.state().bpm) }))()`,
      );
      record(
        "real: global slider 0 / 8 / 16 seek the whole file and NEVER move the window (onsets/BPM untouched)",
        realSeekRows.every((row) => CLOSE(row.seek.value, row.expected, 0.3) && CLOSE(row.playhead, row.expected, 0.4)) &&
          realSeekRows.every(
            (row) => CLOSE(row.view.start, realZoomStable.view.start, 1e-9) && CLOSE(row.view.end, realZoomStable.view.end, 1e-9),
          ) &&
          realOnsetsBpmBefore.onsets === realOnsetsBpmAfter.onsets &&
          realOnsetsBpmBefore.bpm === realOnsetsBpmAfter.bpm,
        { rows: realSeekRows, onsetsBpmUnchanged: realOnsetsBpmBefore.onsets === realOnsetsBpmAfter.onsets },
      );
      // keyboard on the real file: Home / End jump to the file bounds
      await evaluate(client, `(() => { document.getElementById('global-seek').focus(); return true; })()`);
      await pressKey(client, { key: "End", code: "End", windowsVirtualKeyCode: 35 });
      await sleep(120);
      const realKeyEnd = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
      await pressKey(client, { key: "Home", code: "Home", windowsVirtualKeyCode: 36 });
      await sleep(120);
      const realKeyHome = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
      await pressKey(client, { key: "ArrowRight", code: "ArrowRight", windowsVirtualKeyCode: 39 });
      await sleep(120);
      const realKeyArrow = await evaluate(client, "(() => window.__aatViewer.globalSeek())()");
      record(
        "real: keyboard Home/End/arrows seek to the real file bounds (0 / 16 / step)",
        CLOSE(realKeyEnd.value, 16, 0.05) &&
          CLOSE(realKeyHome.value, 0, 0.05) &&
          CLOSE(realKeyArrow.value, GLOBAL_SEEK_STEP_SECONDS, 0.02),
        { end: realKeyEnd.value, home: realKeyHome.value, arrow: realKeyArrow.value },
      );
      // playback + slider sync on the real file
      await evaluate(client, "(() => { window.__aatViewer.seek(4); return true; })()");
      await evaluate(client, "window.__aatViewer.play()");
      await sleep(900);
      const realPlaySync = await evaluate(
        client,
        `(() => {
          const api = window.__aatViewer;
          const s = api.globalSeek();
          return { value: s.value, playhead: api.playheadTime(), playing: api.state().playing, labelText: s.labelText };
        })()`,
      );
      await evaluate(client, "window.__aatViewer.pause()");
      record(
        "real: playback advances the global slider with the audio clock (no pan/zoom while playing)",
        realPlaySync.playing === true && realPlaySync.playhead > 4.2 && CLOSE(realPlaySync.value, realPlaySync.playhead, 0.15),
        realPlaySync,
      );
      const realStemMapping = [];
      for (const stemPath of options.real.stems) {
        const picked = await stemPick(stemPath);
        realStemMapping.push(picked.mapping);
      }
      const mappedIds = (realStemMapping.at(-1) || []).map((entry) => entry.rowId);
      const realRows = await evaluate(client, "(() => window.__aatViewer.state().doc.rows.map((r) => r.id))()");
      record(
        "real stems (3× same-basename target.wav) map to the 3 rows by SHA-256 identity",
        options.real.stems.length === 3 &&
          mappedIds.length === 3 &&
          new Set(mappedIds).size === 3 &&
          mappedIds.every((rowId) => realRows.includes(rowId)),
        { mappedIds, rows: realRows },
      );
      const realNotes = await evaluate(client, "(() => document.getElementById('notes').textContent)()");
      record(
        "real integration: UI shows raw frozen-schema events + labelled document self-descriptions (no quality metrics)",
        // self-descriptions must be LABELLED as such; numeric quality stats
        // (harness spotcheck isolation scores etc.) must never be surfaced
        realNotes.includes("文档自述") && !/isolation|score|0\.9[0-9]\s*[-–~]/i.test(realNotes),
        {
          note: "UI renders the frozen JSON verbatim (110 raw events); document limitations/provenance are labelled 文档自述（非验证结论）; harness spotcheck statistics (float32 decode / 48k-44.1k window) are pending offline correction and are NOT surfaced as truth",
        },
      );
      screenshots.push(await screenshot(client, "10-real-integration"));
      // chart region + full page: rows with onsets and per-row stem waveforms
      await evaluate(client, "(() => { document.getElementById('timeline-section').scrollIntoView({ block: 'start' }); return true; })()");
      await sleep(250);
      screenshots.push(await screenshot(client, "10b-real-chart"));
      screenshots.push(await screenshotFullPage(client, "10c-real-fullpage"));
      // issue #41 review evidence: the global slider + the real 3-row chart
      await evaluate(client, "(() => { window.__aatViewer.fit(); window.__aatViewer.seek(8); return true; })()");
      await sleep(200);
      screenshots.push(await screenshotFullPage(client, "11-global-seek-slider-and-real-chart"));
      realIntegration = {
        rows: realRows,
        totalEvents: realDoc ? realDoc.totalEvents : null,
        stemMapping: realStemMapping.at(-1) || [],
        note: "local-only evidence; public summaries must be sanitised (no paths/audio/hashes in full)",
      };
    }

    // ---- resource cleanup ----------------------------------------------
    const cleanup = await evaluate(
      client,
      `(async () => {
        const before = window.__aatViewer.resourceState();
        await window.__aatViewer.dispose();
        const after = window.__aatViewer.resourceState();
        return { before, after };
      })()`,
    );
    record(
      "dispose() releases ObjectURLs / AudioContext / Workers",
      cleanup.after.objectUrls === 0 && cleanup.after.workers === 0 && cleanup.after.validateWorkerActive === false && cleanup.after.audioContextState === "closed" && cleanup.after.audioSrc === null,
      cleanup,
    );

    // ---- dispose during an in-flight load (fresh page) ------------------
    await client.send("Page.reload", { ignoreCache: true });
    await waitFor(client, "viewer ready after reload", "window.__aatViewer && window.__aatViewer.ready === true", 20000);
    await setFileInput(client, "#json-input", [stress.jsonPath]); // slow load (~1s)
    await sleep(80);
    await evaluate(client, "window.__aatViewer.dispose()");
    await sleep(2500); // give the in-flight load time to (not) apply
    const disposedMid = await evaluate(
      client,
      `(() => {
        const s = window.__aatViewer.state();
        return { disposed: s.disposed, doc: s.doc, resource: window.__aatViewer.resourceState() };
      })()`,
    );
    record(
      "dispose() during an in-flight load aborts cleanly (nothing applied afterwards, no leaked resources)",
      disposedMid.disposed === true && disposedMid.doc === null && disposedMid.resource.objectUrls === 0 && disposedMid.resource.workers === 0,
      disposedMid,
    );

    // ---- dispose during in-flight peaks build (fresh page) --------------
    await client.send("Page.reload", { ignoreCache: true });
    await waitFor(client, "viewer ready for peaks dispose", "window.__aatViewer && window.__aatViewer.ready === true", 20000);
    await loadJsonFile(client, join(VIEWER_ROOT, "fixtures", "demo", "demo-track.json"));
    const peaksLoadsBefore = await evaluate(client, "window.__aatViewer.state().loads.audio");
    await setFileInput(client, "#audio-input", [stress.wavPath]); // slow decode + peaks build
    await sleep(120);
    await evaluate(client, "window.__aatViewer.dispose()");
    await sleep(3000);
    const disposedPeaks = await evaluate(
      client,
      `(() => ({
        disposed: window.__aatViewer.state().disposed,
        loadsAudio: window.__aatViewer.state().loads.audio,
        resource: window.__aatViewer.resourceState(),
      }))()`,
    );
    record(
      "dispose() during an in-flight peaks build settles cleanly (await resolves, no leaked worker/URL)",
      disposedPeaks.disposed === true &&
        disposedPeaks.loadsAudio > peaksLoadsBefore &&
        disposedPeaks.resource.objectUrls === 0 &&
        disposedPeaks.resource.workers === 0,
      disposedPeaks,
    );

    // ---- unhandled page exceptions across the whole session -------------
    record(
      "no unhandled page exceptions / console errors during the whole session",
      pageErrors.length === 0,
      { errors: pageErrors.slice(0, 10) },
    );
  } finally {
    if (client) client.close();
    if (chrome) {
      chrome.kill();
      await sleep(300);
    }
    server.close();
    await rm(workDir, { recursive: true, force: true }).catch(() => {});
  }

  const failures = checks.filter((check) => !check.ok);
  const report = {
    schema_version: "aat-viewer-browser-check/1",
    ok: failures.length === 0,
    generated_at_utc: new Date().toISOString(),
    environment: env,
    measurement: {
      method: "Chrome DevTools Protocol: DOM.setFileInputFiles + Runtime.evaluate + Input.dispatchMouseEvent; timings from performance.now()/rAF deltas inside the page",
      stressFixture: { events: options.stressEvents, seconds: 300, json: stress.jsonPath, wav: stress.wavPath },
      note: "人工验收未完成：这些是主 Agent 浏览器自动检查结果，不是人类验收。",
    },
    checks,
    screenshots,
    screenshot_states: screenshotStates,
    real_integration: realIntegration,
  };
  await writeFile(join(REPORT_DIR, "browser-check.json"), `${JSON.stringify(report, null, 2)}\n`, "utf8");
  process.stdout.write(
    `\n${checks.length - failures.length}/${checks.length} checks passed. report: ${join(REPORT_DIR, "browser-check.json")}\n`,
  );
  if (failures.length > 0) {
    process.exitCode = 1;
  }
  return report;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}

export { buildDemoWav, demoDoc };
