// Static server containment (review finding #5): the advertised viewer-only
// root must hold after FILESYSTEM resolution — symlink/junction escapes must
// never be served.

import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { startServer } from "../tools/serve.mjs";

async function get(url) {
  const response = await fetch(url);
  return { status: response.status, body: await response.text() };
}

function trySymlink(target, path, type) {
  try {
    symlinkSync(target, path, type);
    return true;
  } catch {
    return false;
  }
}

test("serves files inside the root, rejects traversal and symlink/junction escapes", async (t) => {
  const base = mkdtempSync(join(tmpdir(), "aat-serve-root-"));
  const outside = mkdtempSync(join(tmpdir(), "aat-serve-outside-"));
  const root = join(base, "viewer");
  mkdirSync(root, { recursive: true });
  writeFileSync(join(root, "index.html"), "<html>ok</html>");
  writeFileSync(join(outside, "synthetic-secret.txt"), "synthetic outside content\n");

  // synthetic junction/symlink inside the root pointing OUTSIDE (no personal files touched)
  const escapeLink = join(root, "escape");
  const linked = trySymlink(outside, escapeLink, "junction") || trySymlink(outside, escapeLink, "dir");
  const { server, port } = await startServer({ port: 0, root });
  try {
    const ok = await get(`http://127.0.0.1:${port}/index.html`);
    assert.equal(ok.status, 200, "regular in-root file must be served");
    assert.match(ok.body, /ok/);

    const traversal = await get(`http://127.0.0.1:${port}/..%2f..%2fescape/synthetic-secret.txt`);
    assert.notEqual(traversal.status, 200, "encoded traversal must not be served");

    if (linked) {
      const escaped = await get(`http://127.0.0.1:${port}/escape/synthetic-secret.txt`);
      assert.notEqual(escaped.status, 200, "junction/symlink escape must not be served (realpath containment)");
      assert.equal(escaped.body.includes("synthetic outside content"), false);
    } else {
      t.diagnostic("symlink creation not permitted on this platform; lexical tests still ran");
    }

    const missing = await get(`http://127.0.0.1:${port}/nope.txt`);
    assert.equal(missing.status, 404);

    const method = await fetch(`http://127.0.0.1:${port}/index.html`, { method: "POST" });
    assert.equal(method.status, 405);
  } finally {
    server.close();
    rmSync(base, { recursive: true, force: true });
    rmSync(outside, { recursive: true, force: true });
  }
});

test("binds 127.0.0.1 only and serves only the viewer root", async () => {
  const base = mkdtempSync(join(tmpdir(), "aat-serve-bind-"));
  const root = join(base, "viewer");
  mkdirSync(root, { recursive: true });
  writeFileSync(join(root, "index.html"), "<html>ok</html>");
  writeFileSync(join(base, "outside.txt"), "outside\n");
  const { server, port } = await startServer({ port: 0, root });
  try {
    const outside = await get(`http://127.0.0.1:${port}/../outside.txt`);
    assert.notEqual(outside.status, 200, "files outside the viewer root must not be served");
    const host = server.address();
    assert.equal(host.address, "127.0.0.1", "server must bind loopback only");
  } finally {
    server.close();
    rmSync(base, { recursive: true, force: true });
  }
});
