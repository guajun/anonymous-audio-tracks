// Static hygiene for the agentic/viewer area (issue #34):
//   - no HTML sink anywhere in the UI code (labels/errors use textContent,
//     tracks use Canvas fillText),
//   - no remote URL / network primitive in the page code (local-only app),
//   - the shipped schema copies stay byte-identical to the frozen #32 files,
//   - tracked files stay small (< 512 KiB) and generated artifacts are ignored,
//   - no personal absolute paths / secrets in committed text.

import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, relative, resolve } from "node:path";

import {
  REPO_ROOT,
  SCHEMA_PATH,
  SEMANTIC_RULES_PATH,
  UPSTREAM_SCHEMA_DIR,
  VIEWER_ROOT,
} from "./test-helpers.js";

const IGNORED_DIRS = new Set(["generated", "screenshots", "reports", "node_modules"]);
const GENERATED_DIRS = ["fixtures/generated", "screenshots", "reports"];

function listFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (IGNORED_DIRS.has(entry.name) && resolve(full) !== resolve(VIEWER_ROOT)) continue;
      out.push(...listFiles(full));
    } else if (entry.isFile()) {
      out.push(full);
    }
  }
  return out;
}

function isPageCode(file) {
  const rel = relative(VIEWER_ROOT, file).replace(/\\/g, "/");
  if (rel.startsWith("tools/") || rel.startsWith("tests/")) return false;
  return [".js", ".html", ".css"].includes(extname(file));
}

const allFiles = listFiles(VIEWER_ROOT);
const pageFiles = allFiles.filter(isPageCode);

test("page code never uses an HTML sink", () => {
  const sinks = ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write(", "createContextualFragment", "srcdoc"];
  for (const file of pageFiles) {
    const text = readFileSync(file, "utf8");
    for (const sink of sinks) {
      assert.equal(text.includes(sink), false, `${relative(VIEWER_ROOT, file)} must not use ${sink}`);
    }
    assert.equal(/\beval\s*\(/.test(text), false, `${relative(VIEWER_ROOT, file)} must not use eval()`);
    assert.equal(/new\s+Function\s*\(/.test(text), false, `${relative(VIEWER_ROOT, file)} must not use new Function()`);
  }
});

test("page code is local-only: no remote URLs, no XHR/WebSocket, CSP is set", () => {
  for (const file of pageFiles) {
    const text = readFileSync(file, "utf8");
    assert.equal(/https?:\/\//.test(text), false, `${relative(VIEWER_ROOT, file)} must not reference remote URLs`);
    assert.equal(/XMLHttpRequest|WebSocket|EventSource|navigator\.sendBeacon/.test(text), false, "no network primitives");
  }
  const html = readFileSync(join(VIEWER_ROOT, "index.html"), "utf8");
  assert.ok(html.includes("Content-Security-Policy"), "index.html must ship a CSP");
  assert.ok(html.includes("default-src 'none'"), "CSP must default to none");
  assert.ok(html.includes("script type=\"module\" src=\"app.js\""), "only local scripts");
});

test("the only fetch() call loads the local schema via import.meta.url", () => {
  const app = readFileSync(join(VIEWER_ROOT, "app.js"), "utf8");
  const fetches = app.match(/fetch\s*\([^)]*\)/g) || [];
  assert.equal(fetches.length, 1, `expected exactly one fetch(), got ${JSON.stringify(fetches)}`);
  assert.ok(fetches[0].includes("import.meta.url"), "schema fetch must resolve from the module URL, not user input");
});

test("schema copies are byte-identical to the frozen issue #32 contract", () => {
  const source = JSON.parse(readFileSync(join(VIEWER_ROOT, "schema", "SOURCE.json"), "utf8"));
  const hash = (file) => createHash("sha256").update(readFileSync(file)).digest("hex");
  for (const entry of source.files) {
    const copy = join(VIEWER_ROOT, "schema", entry.file);
    const upstream = join(UPSTREAM_SCHEMA_DIR, entry.file);
    assert.equal(hash(copy), entry.sha256, `${entry.file} must match its recorded sha256`);
    assert.equal(hash(upstream), entry.sha256, `${entry.file} must match the upstream frozen file`);
  }
  assert.equal(source.schema_version, "agentic-audio-tracks/v1");
  assert.ok(statSync(SCHEMA_PATH).size > 1000);
  assert.ok(statSync(SEMANTIC_RULES_PATH).size > 1000);
});

test("files stay under the 512 KiB tracked-file limit", () => {
  for (const file of allFiles) {
    const rel = relative(VIEWER_ROOT, file).replace(/\\/g, "/");
    if (GENERATED_DIRS.some((dir) => rel.startsWith(dir))) continue;
    assert.ok(statSync(file).size < 512 * 1024, `${rel} is unexpectedly large`);
  }
});

test("generated artifacts are gitignored, never tracked", () => {
  const probe = spawnSync("git", ["-C", REPO_ROOT, "check-ignore", "-q", join(VIEWER_ROOT, "fixtures", "generated", "stress-100000.json")], {
    shell: false,
  });
  if (probe.error) {
    // git unavailable: fall back to the local .gitignore content check
    const ignore = readFileSync(join(VIEWER_ROOT, ".gitignore"), "utf8");
    assert.ok(ignore.includes("fixtures/generated/"));
    return;
  }
  assert.equal(probe.status, 0, "fixtures/generated must be gitignored");
});

test("committed text contains no personal paths or secrets", () => {
  // Patterns are split so this test file cannot match itself.
  const forbidden = [
    [/[A-Za-z]:\\+Users\\+/, "personal home path"],
    [new RegExp("MSI" + "-NB"), "local username"],
    [new RegExp("F:" + "[\\\\/]LED"), "local drive path"],
    [/sk-[A-Za-z0-9_-]{20,}/, "API key"],
    [/-----BEGIN [A-Z ]*PRIVATE KEY-----/, "private key"],
  ];
  for (const file of allFiles) {
    const rel = relative(VIEWER_ROOT, file).replace(/\\/g, "/");
    if (GENERATED_DIRS.some((dir) => rel.startsWith(dir))) continue;
    const suffix = extname(file).toLowerCase();
    if (![".md", ".json", ".js", ".mjs", ".html", ".css", ".txt"].includes(suffix)) continue;
    const text = readFileSync(file, "utf8");
    for (const [pattern, label] of forbidden) {
      assert.equal(pattern.test(text), false, `${rel} must not contain a ${label}`);
    }
  }
});

test("no audio/binary payload is committed under agentic/viewer", () => {
  const forbidden = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".pt", ".safetensors", ".png"];
  for (const file of allFiles) {
    const rel = relative(VIEWER_ROOT, file).replace(/\\/g, "/");
    if (GENERATED_DIRS.some((dir) => rel.startsWith(dir))) continue;
    assert.equal(forbidden.includes(extname(file).toLowerCase()), false, `${rel} must not be committed`);
  }
});
