/**
 * node_extension_harness.mjs — offline extension-level tests for
 * bridge/audio-bridge.ts (issue #30 review item 2). Loads the REAL extension
 * module with a mock Pi registry/context; no network, no credentials, no API.
 *
 * Covered: happy attach, multiple attachments, one-shot injection, wrong
 * model/provider (attach + injection), failed inputs (missing/escape/non-audio/
 * oversize/queue overflow/not-a-file/permission), lifecycle clear boundaries,
 * and no transcript persistence.
 *
 * Run: node tests/node_extension_harness.mjs   (exits non-zero on failure)
 */
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readdirSync, mkdirSync, rmSync, statSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, sep } from "node:path";

const GOOD_MODEL = { provider: "google", id: "gemini-3.8-flash" };
const BAD_MODEL = { provider: "openrouter", id: "google/gemini-3.8-flash" };

// Conflicting inherited env must NEVER unlock another provider/model
// (review round 2, blocker 2): the model lock is a hard constant.
process.env.PI_AUDIO_BRIDGE_MODEL = "openrouter/evil-model";
process.env.PI_PROVIDER = "openrouter";
process.env.PI_MODEL = "evil-model";

// Real WAV-sniffable bytes (RIFF....WAVE), padded so base64 is long.
const wavBytes = (fill) =>
  Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WAVEfmt "), Buffer.alloc(256, fill)]);

const root = mkdtempSync(join(tmpdir(), "audio-bridge-test-"));
process.env.PI_AUDIO_BRIDGE_ROOT = root;
delete process.env.PI_AUDIO_BRIDGE_MAX_BYTES;
delete process.env.PI_AUDIO_BRIDGE_MAX_QUEUE_BYTES;

const A = wavBytes(1);
const B = wavBytes(2);
writeFileSync(join(root, "a.wav"), A);
writeFileSync(join(root, "b.wav"), B);
writeFileSync(join(root, "notes.txt"), "not audio");
mkdirSync(join(root, "sub"));
writeFileSync(join(root, "big.wav"), wavBytes(3).subarray(0, 200));

// --- mock Pi registry ------------------------------------------------------
function makeMockPi() {
  const tools = {};
  const handlers = {};
  return {
    pi: {
      registerTool(def) {
        tools[def.name] = def;
      },
      on(event, handler) {
        (handlers[event] ??= []).push(handler);
      },
    },
    tools,
    handlers,
  };
}
const { pi, tools, handlers } = makeMockPi();
const { default: audioBridge } = await import("../bridge/audio-bridge.ts");
audioBridge(pi);
const attach = tools.audio_attach.execute;
const ctxFor = (model) => ({ model });

/** Run the context handler; `undefined` return means "no change" (Pi semantics). */
const runContext = async (event, model) => {
  const res = await handlers.context[0](event, ctxFor(model));
  return res?.messages ?? event.messages;
};

async function expectFail(code, fn) {
  await assert.rejects(fn, (err) => {
    assert.ok(err instanceof Error, "must throw Error");
    assert.ok(err.message.startsWith(code + ":"), `expected ${code}, got: ${err.message}`);
    assert.ok(!/[A-Za-z]:[\\/]/.test(err.message), `error message leaks a path: ${err.message}`);
    return true;
  });
}

// --- 1. happy attach: metadata only, no base64 in tool result --------------
const listingBefore = readdirSync(root).sort().join(",");
const result1 = await attach("t1", { path: "a.wav" }, undefined, undefined, ctxFor(GOOD_MODEL));
const summary1 = JSON.parse(result1.content[0].text);
assert.equal(summary1.attached, "a.wav");
assert.equal(summary1.mime, "audio/wav");
assert.equal(summary1.bytes, A.length);
assert.equal(summary1.queueDepth, 1);
assert.equal(result1.content.length, 1, "tool result must not carry an audio block");
assert.ok(!/[A-Za-z0-9+/=]{80,}/.test(result1.content[0].text), "tool result must not contain base64 audio");
console.log("OK happy attach (metadata only)");

// --- 2. multiple attachments + one-shot injection --------------------------
const result2 = await attach("t2", { path: "b.wav" }, undefined, undefined, ctxFor(GOOD_MODEL));
assert.equal(JSON.parse(result2.content[0].text).queueDepth, 2);

const event = { messages: [{ role: "user", content: "q", timestamp: 0 }] };
const injectResult = await runContext(event, GOOD_MODEL);
assert.equal(injectResult.length, 3, "two injected user messages expected");
const injectedA = injectResult[1];
const injectedB = injectResult[2];
assert.equal(injectedA.role, "user");
assert.equal(injectedA.content[1].type, "image", "type label abuse is intentional");
assert.equal(injectedA.content[1].mimeType, "audio/wav", "MIME must be audio");
assert.equal(injectedA.content[1].data, A.toString("base64"), "raw bytes must travel");
assert.equal(injectedB.content[1].data, B.toString("base64"), "order preserved");

const event2 = { messages: [] };
const secondInjection = await runContext(event2, GOOD_MODEL);
assert.equal(secondInjection.length, 0, "injection must be one-shot");
console.log("OK multiple attachments + one-shot injection");

// --- 3. wrong model at attach and at injection -----------------------------
await expectFail("E_AUDIO_MODEL", () =>
  attach("t3", { path: "a.wav" }, undefined, undefined, ctxFor(BAD_MODEL)),
);
await expectFail("E_AUDIO_MODEL", () =>
  attach("t3", { path: "a.wav" }, undefined, undefined, ctxFor(undefined)),
);

await attach("t4", { path: "a.wav" }, undefined, undefined, ctxFor(GOOD_MODEL));
const dropEvent = { messages: [] };
const dropped = await runContext(dropEvent, BAD_MODEL);
assert.equal(dropped.length, 0, "must not inject into another model");
const afterDrop = { messages: [] };
assert.equal((await runContext(afterDrop, GOOD_MODEL)).length, 0, "queue cleared on drop");
console.log("OK model restriction (attach + injection)");

// --- 3b. conflicting env cannot unlock another provider/model -------------
await expectFail("E_AUDIO_MODEL", () =>
  attach("t3b", { path: "a.wav" }, undefined, undefined, ctxFor({ provider: "openrouter", id: "evil-model" })),
);
const stillWorks = await attach("t3c", { path: "b.wav" }, undefined, undefined, ctxFor(GOOD_MODEL));
assert.equal(JSON.parse(stillWorks.content[0].text).attached, "b.wav", "fixed model still works despite env noise");
await runContext({ messages: [] }, GOOD_MODEL); // drain before queue-budget scenarios
console.log("OK conflicting env cannot unlock model lock");

// --- 4. failed inputs ------------------------------------------------------
await expectFail("E_AUDIO_NOT_FOUND", () =>
  attach("t5", { path: "missing.wav" }, undefined, undefined, ctxFor(GOOD_MODEL)),
);
await expectFail("E_AUDIO_PATH", () =>
  attach("t5", { path: ".." + sep + "escape.wav" }, undefined, undefined, ctxFor(GOOD_MODEL)),
);
await expectFail("E_AUDIO_MIME", () =>
  attach("t5", { path: "notes.txt" }, undefined, undefined, ctxFor(GOOD_MODEL)),
);
await expectFail("E_AUDIO_NOT_FILE", () =>
  attach("t5", { path: "sub" }, undefined, undefined, ctxFor(GOOD_MODEL)),
);
await expectFail("E_AUDIO_ARGS", () =>
  attach("t5", {}, undefined, undefined, ctxFor(GOOD_MODEL)),
);

process.env.PI_AUDIO_BRIDGE_MAX_BYTES = "100";
await expectFail("E_AUDIO_SIZE", () =>
  attach("t6", { path: "a.wav" }, undefined, undefined, ctxFor(GOOD_MODEL)),
);
delete process.env.PI_AUDIO_BRIDGE_MAX_BYTES;

process.env.PI_AUDIO_BRIDGE_MAX_QUEUE_BYTES = String(A.length);
await attach("t7", { path: "a.wav" }, undefined, undefined, ctxFor(GOOD_MODEL));
await expectFail("E_AUDIO_QUEUE", () =>
  attach("t7", { path: "b.wav" }, undefined, undefined, ctxFor(GOOD_MODEL)),
);
delete process.env.PI_AUDIO_BRIDGE_MAX_QUEUE_BYTES;
await runContext({ messages: [] }, GOOD_MODEL); // drain
console.log("OK failed inputs (args/path/not-found/not-file/mime/size/queue)");

// permission case: POSIX only (Windows chmod is not enforced)
if (process.platform !== "win32") {
  const permPath = join(root, "perm.wav");
  writeFileSync(permPath, A);
  chmodSync(permPath, 0o000);
  try {
    await expectFail("E_AUDIO_READ", () =>
      attach("t8", { path: "perm.wav" }, undefined, undefined, ctxFor(GOOD_MODEL)),
    );
    console.log("OK permission failure (E_AUDIO_READ)");
  } finally {
    chmodSync(permPath, 0o644);
  }
} else {
  console.log("SKIP permission failure (not enforceable on win32)");
}

// --- 5. lifecycle clear boundaries ----------------------------------------
for (const boundary of ["session_shutdown", "agent_settled", "session_start"]) {
  await attach("t9", { path: "a.wav" }, undefined, undefined, ctxFor(GOOD_MODEL));
  for (const handler of handlers[boundary] ?? []) {
    await handler({ type: boundary }, ctxFor(GOOD_MODEL));
  }
  const ev = { messages: [] };
  const out = await runContext(ev, GOOD_MODEL);
  assert.equal(out.length, 0, `pending must be cleared at ${boundary}`);
}
console.log("OK lifecycle clear boundaries (session_start/session_shutdown/agent_settled)");

// --- 6. no transcript persistence -----------------------------------------
await attach("t10", { path: "a.wav" }, undefined, undefined, ctxFor(GOOD_MODEL));
const ev = { messages: [] };
await runContext(ev, GOOD_MODEL);
assert.equal(readdirSync(root).sort().join(","), listingBefore, "extension must not write files");
assert.equal(statSync(join(root, "a.wav")).size, A.length, "extension must not modify inputs");
assert.ok(!/[A-Za-z0-9+/=]{80,}/.test(JSON.stringify(result1.details)), "details carry no base64");
console.log("OK no transcript persistence (bytes stay in memory, files untouched)");

rmSync(root, { recursive: true, force: true });
console.log("node_extension_harness: OK");
