// Real-browser check for the local viewer.
//
// Launches a locally installed Chrome or Edge in headless mode, serves the
// repository over 127.0.0.1 only, drives the page through the Chrome
// DevTools Protocol (DOM.setFileInputFiles + evaluate) and asserts the state
// that the UI computes. No npm dependency, no remote CDN: Node's built-in
// `fetch` and `WebSocket` are enough.
//
// Usage:
//   node viewer/tools/browser-check.mjs
//   node viewer/tools/browser-check.mjs --chrome "C:/path/to/chrome.exe"
//   node viewer/tools/browser-check.mjs --report tests/viewer/fixtures/generated/browser-check.json \
//     --screenshot tests/viewer/fixtures/generated/viewer-check.png
//
// `tests/viewer/fixtures/generated/` is ignored (see its nested .gitignore);
// generated artifacts are evidence for review, not repository content.
//
// Exit code is non-zero when a check fails. The report lists every check so
// docs/VIEWER.md can quote a real run.

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildWav } from "./make-fixture-audio.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "..", "..");
const VIEWER_URL_PATH = "/viewer/index.html";

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wav": "audio/wav",
  ".txt": "text/plain; charset=utf-8",
};

const sleep = (milliseconds) => new Promise((resolvePromise) => setTimeout(resolvePromise, milliseconds));

function parseArgs(argv) {
  const options = { chrome: null, report: null, screenshot: null, timeoutMs: 90000 };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--chrome") {
      options.chrome = argv[++index];
    } else if (argv[index] === "--report") {
      options.report = argv[++index];
    } else if (argv[index] === "--screenshot") {
      options.screenshot = argv[++index];
    } else if (argv[index] === "--timeout-ms") {
      options.timeoutMs = Number(argv[++index]);
    }
  }
  return options;
}

function findChrome(explicit) {
  const candidates = [];
  if (explicit) {
    candidates.push(explicit);
  }
  if (process.env.CHROME_PATH) {
    candidates.push(process.env.CHROME_PATH);
  }
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
    candidates.push(
      "/usr/bin/google-chrome",
      "/usr/bin/chromium",
      "/usr/bin/chromium-browser",
      "/usr/bin/microsoft-edge",
    );
  }
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

function startStaticServer() {
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, "http://127.0.0.1");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/") {
        pathname = VIEWER_URL_PATH;
      }
      const target = resolve(REPO_ROOT, "." + pathname);
      if (!target.startsWith(REPO_ROOT)) {
        response.writeHead(403).end("forbidden");
        return;
      }
      const body = await readFile(target);
      response.writeHead(200, {
        "content-type": MIME_TYPES[extname(target).toLowerCase()] || "application/octet-stream",
      });
      response.end(body);
    } catch {
      response.writeHead(404).end("not found");
    }
  });
  return new Promise((resolvePromise, rejectPromise) => {
    server.once("error", rejectPromise);
    server.listen(0, "127.0.0.1", () => {
      resolvePromise({ server, port: server.address().port });
    });
  });
}

async function waitForHttp(url, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return await response.json();
      }
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
      if (message.error) {
        entry.reject(new Error(`${entry.method}: ${JSON.stringify(message.error)}`));
      } else {
        entry.resolve(message.result);
      }
    } else if (message.method) {
      for (const handler of listeners.get(message.method) || []) {
        handler(message.params);
      }
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
      (result.exceptionDetails.exception && result.exceptionDetails.exception.description) ||
      result.exceptionDetails.text;
    throw new Error(`page evaluate failed: ${description}`);
  }
  return result.result ? result.result.value : undefined;
}

async function waitFor(description, predicate, timeoutMs = 10000, intervalMs = 120) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const value = await predicate();
      if (value) {
        return value;
      }
    } catch (error) {
      lastError = error;
    }
    await sleep(intervalMs);
  }
  throw new Error(`timeout waiting for ${description}${lastError ? `: ${lastError.message}` : ""}`);
}

async function setFileInput(client, selector, files) {
  const document = await client.send("DOM.getDocument", { depth: 1 });
  const { nodeId } = await client.send("DOM.querySelector", {
    nodeId: document.root.nodeId,
    selector,
  });
  if (!nodeId) {
    throw new Error(`file input not found: ${selector}`);
  }
  await client.send("DOM.setFileInputFiles", { files, nodeId });
}

function buildLongTrajectoryJson(seconds = 600, hop = 1) {
  const points = Math.floor(seconds / hop) + 1;
  const make = (fn) => ({
    center_times: Array.from({ length: points }, (_, index) => index * hop),
    activity: Array.from({ length: points }, (_, index) => fn(index * hop)),
  });
  const tracks = [
    { track_id: "trk-long-a", ...make((time) => (Math.sin(time / 10) + 1) / 2) },
    { track_id: "trk-long-b", ...make((time) => (time % 30 < 3 ? 0.9 : 0.05)) },
    { track_id: "trk-long-c", ...make((time) => (time > 300 ? 0.0 : 0.4)) },
  ];
  return JSON.stringify({
    schema_version: "0.1.0",
    kind: "trajectory",
    sample_id: "fixture-long",
    audio: { duration_seconds: seconds, track_start_seconds: 0.0 },
    provenance: {
      run_id: "fixture-long-01",
      data_kind: "mock",
      created_at_utc: "2026-09-29T12:00:00Z",
    },
    tracks,
  });
}

function buildInvalidTrajectoryJson() {
  return JSON.stringify({
    schema_version: "0.1.0",
    kind: "trajectory",
    audio: { duration_seconds: 1.0, track_start_seconds: 0.0 },
    provenance: { run_id: "fixture-invalid-01", data_kind: "model" },
    tracks: [
      {
        track_id: "trk-invalid",
        center_times: [0.5, 0.5],
        activity: [0.2],
      },
    ],
  });
}

const CLOSE = (left, right, tolerance) => Math.abs(left - right) <= tolerance;

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const chromePath = findChrome(options.chrome);
  if (!chromePath) {
    console.error(
      "no Chrome/Edge found; pass --chrome <path> or set CHROME_PATH. " +
        "This check is documented as unverified when no browser is available.",
    );
    process.exitCode = 2;
    return;
  }

  const checks = [];
  const record = (name, ok, details) => {
    checks.push({ name, ok: Boolean(ok), details: details === undefined ? null : details });
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${details === undefined ? "" : ` — ${JSON.stringify(details)}`}`);
  };
  const failures = [];

  const workDir = await mkdtemp(join(tmpdir(), "aat-viewer-check-"));
  const { server, port: serverPort } = await startStaticServer();
  let chrome = null;
  let client = null;
  let chromeLog = "";

  try {
    const knownWav = join(workDir, "known-times.wav");
    const longWav = join(workDir, "long-600s.wav");
    const longTrajectory = join(workDir, "trajectory.long.json");
    const invalidTrajectory = join(workDir, "trajectory.invalid.json");
    const notAudio = join(workDir, "not-audio.txt");
    await writeFile(knownWav, buildWav({ pattern: "known-times", seconds: 4, rate: 44100 }));
    await writeFile(longWav, buildWav({ pattern: "long", seconds: 600, rate: 8000 }));
    await writeFile(longTrajectory, buildLongTrajectoryJson(600, 1));
    await writeFile(invalidTrajectory, buildInvalidTrajectoryJson());
    await writeFile(notAudio, "this is not an audio file\n");

    const debuggingUrlPrefix = "http://127.0.0.1:";
    const pageUrl = `http://127.0.0.1:${serverPort}${VIEWER_URL_PATH}`;
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

    const debuggingPort = await waitFor("Chrome DevTools port", async () => {
      const match = /DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)\//.exec(chromeLog);
      return match ? Number(match[1]) : null;
    }, 30000, 100);
    const version = await waitForHttp(`${debuggingUrlPrefix}${debuggingPort}/json/version`);
    const targets = await waitForHttp(`${debuggingUrlPrefix}${debuggingPort}/json/list`);
    const pageTarget = targets.find((target) => target.type === "page" && target.url.includes("viewer/index.html")) ||
      targets.find((target) => target.type === "page");
    if (!pageTarget) {
      throw new Error("no page target from Chrome; log tail: " + chromeLog.slice(-500));
    }
    client = await connectCdp(pageTarget.webSocketDebuggerUrl);
    await client.send("Runtime.enable");
    await client.send("DOM.enable");
    await client.send("Page.enable");
    await waitFor("page DOM ready", async () => evaluate(client, "document.readyState === 'complete' && typeof window.__aatViewer === 'object'"));

    record("browser: headless Chrome/Edge launched", true, {
      browser: version.Browser,
      page: VIEWER_URL_PATH,
    });

    const knownTimes = resolve(REPO_ROOT, "tests/viewer/fixtures/trajectory.known-times.json");
    const clip = resolve(REPO_ROOT, "tests/viewer/fixtures/trajectory.clip.json");
    const empty = resolve(REPO_ROOT, "tests/viewer/fixtures/trajectory.empty.json");
    const hostile = resolve(REPO_ROOT, "tests/viewer/fixtures/trajectory.hostile-strings.json");

    // ---- known-times trajectory + audio -------------------------------------
    await setFileInput(client, "#trajectory-file", [knownTimes]);
    const loaded = await waitFor(
      "known-times trajectory",
      async () => {
        const described = await evaluate(client, "__aatViewer.describe()");
        return described.session.trajectoryError === null && described.session.trackCount === 3
          ? described
          : null;
      },
    );
    record("trajectory 0.1.0 fixture validates and lists 3 anonymous tracks", true, {
      dataKind: loaded.session.provenance.dataKind,
      runId: loaded.session.provenance.runId,
    });

    await setFileInput(client, "#audio-file", [knownWav]);
    const audioState = await waitFor("audio metadata", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.audio.duration !== null && Math.abs(described.audio.duration - 4) < 0.05
        ? described
        : null;
    });
    record("audio file loads via local blob: URL (no upload)", audioState.audio.srcIsObjectUrl, {
      duration: audioState.audio.duration,
    });

    // ---- known-time sync ----------------------------------------------------
    await evaluate(client, "__aatViewer.setTime(0.75)");
    let snapshot = await evaluate(client, "__aatViewer.snapshot()");
    const trackA = snapshot.tracks.find((track) => track.trackId === "trk-a");
    const trackB = snapshot.tracks.find((track) => track.trackId === "trk-b");
    const trackC = snapshot.tracks.find((track) => track.trackId === "trk-c");
    record("shared clock: at local 0.75s absolute time stays 0.75s (track_start=0)", CLOSE(snapshot.absoluteTime, 0.75, 1e-9), {
      localTime: snapshot.localTime,
      absoluteTime: snapshot.absoluteTime,
    });
    record("interpolated activity at 0.75s (0.5s:0.8 -> 1.0s:0.9) equals 0.85", CLOSE(trackA.value, 0.85, 1e-9), { value: trackA.value });
    record("threshold 0.5 marks the point active", trackA.active === true);
    record("silent-identity track reports 0.0 and stays present", trackB.value === 0 && trackB.active === false, { value: trackB.value });
    record("third track starts only at 1.0s, so it has no data at 0.75s", trackC.value === null, { value: trackC.value });
    await evaluate(client, "__aatViewer.setTime(1.0)");
    snapshot = await evaluate(client, "__aatViewer.snapshot()");
    record("second track is active at 1.0s", snapshot.tracks.find((track) => track.trackId === "trk-c").value === 1 && snapshot.tracks.find((track) => track.trackId === "trk-c").active === true);
    await evaluate(client, "__aatViewer.setTime(0.75)");
    snapshot = await evaluate(client, "__aatViewer.snapshot()");
    record(
      "threshold intervals: [0.3125, 1.25] and [2.875, 3.08333]",
      CLOSE(trackA.intervals[0][0], 0.3125, 1e-9) &&
        CLOSE(trackA.intervals[0][1], 1.25, 1e-9) &&
        CLOSE(trackA.intervals[1][0], 2.875, 1e-9) &&
        CLOSE(trackA.intervals[1][1], 3.0833333333333335, 1e-9),
      trackA.intervals,
    );

    await evaluate(
      client,
      "(() => { const slider = document.getElementById('threshold'); slider.value = '0.95'; slider.dispatchEvent(new Event('input', { bubbles: true })); return true; })()",
    );
    snapshot = await evaluate(client, "__aatViewer.snapshot()");
    record("adjustable threshold 0.95 flips 0.9 activity to inactive",
      snapshot.threshold === 0.95 && snapshot.tracks.find((track) => track.trackId === "trk-a").active === false,
    );
    await evaluate(
      client,
      "(() => { const slider = document.getElementById('threshold'); slider.value = '0.5'; slider.dispatchEvent(new Event('input', { bubbles: true })); return true; })()",
    );

    // ---- drag / seek via the real slider ------------------------------------
    await evaluate(
      client,
      "(() => { const slider = document.getElementById('seek'); slider.value = '3.4'; slider.dispatchEvent(new Event('input', { bubbles: true })); slider.dispatchEvent(new Event('change', { bubbles: true })); return true; })()",
    );
    const seekState = await evaluate(client, "__aatViewer.describe()");
    record("seek slider drag updates the shared audio clock", CLOSE(seekState.audio.currentTime, 3.4, 0.01), {
      currentTime: seekState.audio.currentTime,
    });

    // ---- play / pause / rate sync -------------------------------------------
    await evaluate(
      client,
      "(async () => { __aatViewer.setTime(0); await __aatViewer.play(); return true; })()",
    );
    const playStarted = await waitFor("playback to start", async () => {
      const value = (await evaluate(client, "__aatViewer.describe()")).audio.currentTime;
      return value > 0.05 ? value : null;
    }, 5000, 50);
    const wallStart = Date.now();
    await sleep(700);
    const playEnded = (await evaluate(client, "__aatViewer.describe()")).audio.currentTime;
    await evaluate(client, "__aatViewer.pause()");
    const wallElapsed = (Date.now() - wallStart) / 1000;
    const advance = playEnded - playStarted;
    record("play at 1x tracks real wall-clock time", Math.abs(advance - wallElapsed) < 0.3, {
      advance,
      wallElapsed,
    });
    const paused = await evaluate(client, "__aatViewer.describe()");
    await sleep(250);
    const stillPaused = await evaluate(client, "__aatViewer.describe()");
    record("pause freezes the shared clock", paused.audio.currentTime === stillPaused.audio.currentTime);

    await evaluate(
      client,
      "(async () => { const select = document.getElementById('rate'); select.value = '0.5'; select.dispatchEvent(new Event('change', { bubbles: true })); __aatViewer.setTime(0); await __aatViewer.play(); return true; })()",
    );
    const slowStarted = await waitFor("slow playback to start", async () => {
      const value = (await evaluate(client, "__aatViewer.describe()")).audio.currentTime;
      return value > 0.02 ? value : null;
    }, 5000, 50);
    const slowWallStart = Date.now();
    await sleep(700);
    const slowEnded = (await evaluate(client, "__aatViewer.describe()")).audio.currentTime;
    await evaluate(client, "__aatViewer.pause()");
    const slowWallElapsed = (Date.now() - slowWallStart) / 1000;
    const slowAdvance = slowEnded - slowStarted;
    record("slow motion 0.5x advances at half wall-clock rate", Math.abs(slowAdvance - slowWallElapsed * 0.5) < 0.2, {
      slowAdvance,
      slowWallElapsed,
      playbackRate: (await evaluate(client, "__aatViewer.describe()")).audio.playbackRate,
    });
    await evaluate(
      client,
      "(() => { const select = document.getElementById('rate'); select.value = '1'; select.dispatchEvent(new Event('change', { bubbles: true })); return true; })()",
    );

    // ---- loop ---------------------------------------------------------------
    let loopState = await evaluate(client, "__aatViewer.setLoop(1.0, 1.5)");
    record("valid loop A=1.0 B=1.5 activates", loopState.loop !== null && loopState.loopError === null, loopState.loop);
    await evaluate(client, "__aatViewer.setTime(1.4); __aatViewer.play()");
    await sleep(900);
    await evaluate(client, "__aatViewer.pause()");
    const looped = await evaluate(client, "__aatViewer.describe()");
    record("playback wraps inside the loop and never runs past B", looped.audio.currentTime >= 0.99 && looped.audio.currentTime <= 1.55, {
      currentTime: looped.audio.currentTime,
    });
    const badLoop = await evaluate(client, "__aatViewer.setLoop(2.0, 1.0)");
    record("loop endpoint validation rejects A >= B", badLoop.loop === null && /至少|循环区间/.test(badLoop.loopError || ""), badLoop.loopError);
    const shortLoop = await evaluate(client, "__aatViewer.setLoop(0.5, 0.52)");
    record("loop endpoint validation rejects intervals shorter than 0.05s", shortLoop.loop === null && /至少/.test(shortLoop.loopError || ""), shortLoop.loopError);
    const rangeLoop = await evaluate(client, "__aatViewer.setLoop(0, 99)");
    record("loop endpoint validation rejects B beyond the audio duration", rangeLoop.loop === null && /超出/.test(rangeLoop.loopError || ""), rangeLoop.loopError);
    await evaluate(client, "__aatViewer.clearLoop()");
    loopState = await evaluate(client, "__aatViewer.describe()");
    record("loop clears cleanly", loopState.session.loop === null && loopState.session.loopError === null);

    // ---- clip fixture: track_start_seconds mapping ---------------------------
    await setFileInput(client, "#trajectory-file", [clip]);
    await waitFor("clip trajectory", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.session.trajectoryError === null && described.session.trackCount === 1 ? described : null;
    });
    await evaluate(client, "__aatViewer.setTime(0.24)");
    snapshot = await evaluate(client, "__aatViewer.snapshot()");
    const clipTrack = snapshot.tracks[0];
    record("non-zero track_start_seconds: local 0.24s maps to absolute 12.24s", CLOSE(snapshot.absoluteTime, 12.24, 1e-9), {
      localTime: snapshot.localTime,
      absoluteTime: snapshot.absoluteTime,
    });
    record("absolute center time 12.24s yields p=0.7 (not the 0.24s player value)", CLOSE(clipTrack.value, 0.7, 1e-9), { value: clipTrack.value });
    const compatText = await evaluate(client, "document.getElementById('compat-message').textContent");
    const compatClass = await evaluate(client, "document.getElementById('compat-message').className");
    record("duration incompatibility is reported for a 0.5s clip trailer vs 4s audio", /只有/.test(compatText) && /error/.test(compatClass), {
      compatText,
    });

    // ---- empty trajectory boundary ------------------------------------------
    await setFileInput(client, "#trajectory-file", [empty]);
    await waitFor("empty trajectory", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.session.trajectoryError === null && described.session.trackCount === 0 ? described : null;
    });
    snapshot = await evaluate(client, "__aatViewer.snapshot()");
    record("empty trajectory (duration 0, tracks []) renders without crashing", snapshot.trajectoryLoaded === true && snapshot.visibleTrackCount === 0, {
      duration: snapshot.duration,
      durationSource: snapshot.durationSource,
    });

    // ---- invalid trajectory must clear stale state ---------------------------
    await setFileInput(client, "#trajectory-file", [invalidTrajectory]);
    const invalidState = await waitFor("invalid trajectory error", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.session.trajectoryError ? described : null;
    });
    const invalidIssues = invalidState.session.trajectoryError.issues.map((issue) => issue.path).join(",");
    record("invalid trajectory rejected with field-level issues", /严格递增/.test(JSON.stringify(invalidState.session.trajectoryError.issues)) && /长度/.test(JSON.stringify(invalidState.session.trajectoryError.issues)), {
      paths: invalidIssues,
    });
    snapshot = await evaluate(client, "__aatViewer.snapshot()");
    const trackListText = await evaluate(client, "document.getElementById('tracks').textContent");
    const errorPanelVisible = await evaluate(client, "!document.getElementById('validation-error').hidden");
    const errorState = await evaluate(client, "__aatViewer.describe()");
    record("load error clears old curves, old playback and shows the error panel",
      snapshot.trajectoryLoaded === false && !/trk-a|trk-clip/.test(trackListText) && errorPanelVisible && errorState.audio.paused === true,
      {
        trajectoryLoaded: snapshot.trajectoryLoaded,
        trackListText,
        errorPanelVisible,
        paused: errorState.audio.paused,
        trackCount: errorState.session.trackCount,
      },
    );

    // ---- hostile strings stay text ------------------------------------------
    await setFileInput(client, "#trajectory-file", [hostile]);
    await waitFor("hostile-strings trajectory", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.session.trajectoryError === null && described.session.trackCount === 1 ? described : null;
    });
    const injection = await evaluate(
      client,
      "({ pwned: [window.__pwned_notes, window.__pwned_sample, window.__pwned_model, window.__pwned_track].filter(Boolean).length, imgs: document.querySelectorAll('img').length, provenance: document.getElementById('provenance').textContent, tracks: document.getElementById('tracks').textContent })",
    );
    record("hostile JSON strings render as text, never as HTML", injection.pwned === 0 && injection.imgs === 0 && /<img/.test(injection.provenance) && /<b>html<\/b>/.test(injection.tracks), {
      pwned: injection.pwned,
      imgs: injection.imgs,
    });

    // ---- object URL cleanup --------------------------------------------------
    const secondKnownWav = join(workDir, "known-times-second.wav");
    await writeFile(secondKnownWav, buildWav({ pattern: "known-times", seconds: 4, rate: 44100 }));
    await evaluate(
      client,
      "(() => { window.__revokeCount = 0; const original = URL.revokeObjectURL.bind(URL); URL.revokeObjectURL = (url) => { window.__revokeCount += 1; return original(url); }; return true; })()",
    );
    await setFileInput(client, "#audio-file", [secondKnownWav]);
    await waitFor("audio reload after object URL patch", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.audio.duration !== null ? described : null;
    });
    const revokeCount = await evaluate(client, "window.__revokeCount");
    record("switching audio revokes the previous blob URL", revokeCount >= 1, { revokeCount });

    await setFileInput(client, "#audio-file", [notAudio]);
    const audioError = await waitFor("audio error state", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.session.audio.error ? described : null;
    });
    record("non-audio file produces a clear load error and keeps the trajectory", /失败|不支持|格式/.test(audioError.session.audio.error), {
      error: audioError.session.audio.error,
      trackCount: audioError.session.trackCount,
    });

    // ---- long song navigation ------------------------------------------------
    await setFileInput(client, "#audio-file", [longWav]);
    await waitFor("long audio metadata", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.audio.duration !== null && Math.abs(described.audio.duration - 600) < 1 ? described : null;
    });
    await setFileInput(client, "#trajectory-file", [longTrajectory]);
    await waitFor("long trajectory", async () => {
      const described = await evaluate(client, "__aatViewer.describe()");
      return described.session.trajectoryError === null && described.session.trackCount === 3 ? described : null;
    });
    snapshot = await evaluate(client, "__aatViewer.snapshot()");
    record("600s trajectory + audio loads (long-song browsing input)", CLOSE(snapshot.duration, 600, 0.5) && snapshot.tracks.length === 3, {
      duration: snapshot.duration,
    });
    await evaluate(client, "document.getElementById('window-next').click()");
    let navState = await evaluate(client, "__aatViewer.describe()");
    record("window next moves the local window by one window length", navState.session.windowStart === 20 && navState.session.followPlayhead === false, navState.session.windowStart);
    for (let click = 0; click < 3; click += 1) {
      await evaluate(client, "document.getElementById('window-next').click()");
    }
    navState = await evaluate(client, "__aatViewer.describe()");
    record("window next reaches 80s after four clicks", navState.session.windowStart === 80, navState.session.windowStart);
    await evaluate(client, "document.getElementById('window-last').click()");
    navState = await evaluate(client, "__aatViewer.describe()");
    record("window last clamps to 580s for a 600s song", navState.session.windowStart === 580, navState.session.windowStart);
    await evaluate(client, "document.getElementById('window-first').click()");
    navState = await evaluate(client, "__aatViewer.describe()");
    record("window first returns to 0s", navState.session.windowStart === 0, navState.session.windowStart);

    const overviewSeek = await evaluate(
      client,
      "(() => { const canvas = document.getElementById('overview-canvas'); const rect = canvas.getBoundingClientRect(); const event = new MouseEvent('click', { bubbles: true, clientX: rect.left + rect.width * 0.5, clientY: rect.top + rect.height / 2 }); canvas.dispatchEvent(event); return true; })()",
    );
    void overviewSeek;
    const middle = await evaluate(client, "__aatViewer.describe()");
    record("clicking the overview seeks the shared clock to the middle of a 600s song", middle.audio.currentTime > 290 && middle.audio.currentTime < 310, {
      currentTime: middle.audio.currentTime,
    });

    const summary = {
      startedAtUtc: new Date().toISOString(),
      chrome: version.Browser,
      page: pageUrl,
      checks,
      failed: checks.filter((check) => !check.ok).length,
    };
    if (options.screenshot) {
      const screenshot = await client.send("Page.captureScreenshot", { format: "png" });
      const screenshotPath = resolve(REPO_ROOT, options.screenshot);
      await mkdir(dirname(screenshotPath), { recursive: true });
      await writeFile(screenshotPath, Buffer.from(screenshot.data, "base64"));
      summary.screenshot = options.screenshot;
      console.log(`screenshot written to ${screenshotPath}`);
    }
    console.log(`\n${checks.length - summary.failed}/${checks.length} browser checks passed.`);
    if (options.report) {
      const reportPath = resolve(REPO_ROOT, options.report);
      await mkdir(dirname(reportPath), { recursive: true });
      await writeFile(reportPath, JSON.stringify(summary, null, 2));
      console.log(`report written to ${reportPath}`);
    }
    if (summary.failed > 0) {
      process.exitCode = 1;
    }
  } catch (error) {
    console.error(`browser check crashed: ${error.message}`);
    if (chromeLog) {
      console.error("chrome log tail:", chromeLog.slice(-800));
    }
    process.exitCode = 1;
  } finally {
    if (client) {
      client.close();
    }
    if (chrome && chrome.exitCode === null) {
      chrome.kill();
    }
    server.close();
    await rm(workDir, { recursive: true, force: true }).catch(() => {});
  }
}

void main();
