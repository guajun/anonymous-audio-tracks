// Static hygiene checks for the viewer UI:
// - user JSON must never flow into an HTML sink (textContent only)
// - no remote asset, network call or telemetry primitive in the page code
// - object URLs are managed by the tested revoke-on-replace manager
//
// These are source-level guards that complement the real-browser injection
// check in viewer/tools/browser-check.mjs.

import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const VIEWER_ROOT = resolve(fileURLToPath(import.meta.url), "..", "..", "..", "viewer");

function listFiles(directory) {
  return readdirSync(directory, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath || entry.path, entry.name));
}

const allViewerFiles = listFiles(VIEWER_ROOT);
const uiFiles = allViewerFiles.filter((file) =>
  [".html", ".css", ".js"].includes(extname(file)) && !file.includes(`${join("viewer", "tools")}`),
);

test("viewer contains the expected entry points", () => {
  for (const name of ["index.html", "app.js", "styles.css", join("js", "protocol.js")]) {
    assert.ok(
      allViewerFiles.some((file) => file.endsWith(name)),
      `missing viewer/${name}`,
    );
  }
});

test("no HTML sink is used anywhere in the viewer code", () => {
  const sinks = ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "new Function(", "eval("];
  for (const file of allViewerFiles.filter((entry) => entry.endsWith(".js") || entry.endsWith(".html"))) {
    const text = readFileSync(file, "utf8");
    for (const sink of sinks) {
      assert.ok(
        !text.includes(sink),
        `${file} uses HTML/JS sink ${sink}; user JSON must be rendered as text`,
      );
    }
  }
});

test("page code has no remote asset or network call", () => {
  const networkPrimitives = ["fetch(", "XMLHttpRequest", "sendBeacon", "EventSource", "WebSocket"];
  for (const file of uiFiles) {
    const text = readFileSync(file, "utf8");
    assert.ok(
      !/https?:\/\//.test(text),
      `${file} references a remote URL; the viewer must stay fully local`,
    );
    for (const primitive of networkPrimitives) {
      assert.ok(
        !text.includes(primitive),
        `${file} uses ${primitive}; the viewer must not call the network`,
      );
    }
  }
});

test("index.html only loads local relative assets", () => {
  const html = readFileSync(join(VIEWER_ROOT, "index.html"), "utf8");
  const references = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((match) => match[1]);
  assert.ok(references.length >= 2);
  for (const reference of references) {
    assert.ok(
      !reference.startsWith("/") && !reference.startsWith("//") && !reference.includes(":"),
      `asset reference must be a local relative path: ${reference}`,
    );
  }
  assert.ok(references.includes("app.js"));
  assert.ok(references.includes("styles.css"));
});

test("audio blob URLs are revoked via the tested manager", () => {
  const app = readFileSync(join(VIEWER_ROOT, "app.js"), "utf8");
  assert.ok(app.includes("createObjectUrlManager"), "app.js must use the object URL manager");
  assert.ok(app.includes("URL.revokeObjectURL"), "app.js must pass a real revoke callback");
  assert.ok(app.includes("beforeunload"), "app.js must clean blob URLs before unload");
});

test("profile data kinds stay visibly distinct (mock is never shown as model)", () => {
  const protocol = readFileSync(join(VIEWER_ROOT, "js", "protocol.js"), "utf8");
  assert.ok(protocol.includes('"model"'));
  assert.ok(protocol.includes('"annotation"'));
  assert.ok(protocol.includes('"mock"'));
  assert.ok(protocol.includes("不是模型结果"), "mock label must warn it is not a model result");
});

test("fixtures are plain-text protocol documents, not binaries", () => {
  const fixtureDir = resolve(VIEWER_ROOT, "..", "tests", "viewer", "fixtures");
  const fixtures = readdirSync(fixtureDir).filter((name) => name.endsWith(".json"));
  assert.ok(fixtures.length >= 4, `expected >= 4 fixtures, got ${fixtures.join(", ")}`);
  for (const name of fixtures) {
    const parsed = JSON.parse(readFileSync(join(fixtureDir, name), "utf8"));
    assert.equal(parsed.schema_version, "0.1.0");
    assert.equal(parsed.kind, "trajectory");
  }
});
