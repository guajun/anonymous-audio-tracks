// File-entry decode policy (review addition): `File.text()` silently strips a
// UTF-8 BOM and replaces invalid UTF-8 with U+FFFD, so the strict parse layer
// must never rely on it. The raw bytes go through `decodeUtf8Strict`
// (mirroring the frozen Python `loader.load_path`) and BOM / bad encoding are
// controlled `E_PARSE` verdicts — proven here at byte level, with real `File`
// objects, and differentially against the Python reference.

import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";

import { decodeUtf8Strict, loadText, readFileStrict } from "../js/strict-json.js";
import { parseAndValidate } from "../js/protocol.js";
import { REPO_ROOT, minimalDoc, readSchema } from "./test-helpers.js";

const schema = readSchema();
const validBytes = Buffer.from(JSON.stringify(minimalDoc()), "utf8");

function issueCodes(issues) {
  return issues.map((issue) => `${issue.layer}/${issue.code}@${issue.pointer}`);
}

test("leading UTF-8 BOM is rejected at the byte level (E_PARSE)", () => {
  const bytes = Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), validBytes]);
  const { text, issues } = decodeUtf8Strict(bytes, "bom.json");
  assert.equal(text, null);
  assert.deepEqual(issueCodes(issues), ["parse/E_PARSE@"]);
  assert.match(issues[0].message, /BOM/);
});

test("invalid UTF-8 bytes are a controlled E_PARSE, never U+FFFD substitution", () => {
  // 0xFF is never valid UTF-8; a truncated 3-byte sequence is also invalid
  const withFF = Buffer.concat([Buffer.from('{"label": "'), Buffer.from([0xff]), Buffer.from('"}')]);
  const truncated = Buffer.concat([Buffer.from('{"label": "'), Buffer.from([0xe2, 0x82])]);
  for (const bytes of [withFF, truncated]) {
    const { text, issues } = decodeUtf8Strict(bytes, "bad.json");
    assert.equal(text, null, "no substituted text is ever handed to the validator");
    assert.deepEqual(issueCodes(issues), ["parse/E_PARSE@"]);
    assert.match(issues[0].message, /UTF-8/);
  }
});

test("what File.text() would hide: BOM strip + U+FFFD are visible at the byte layer", async () => {
  const bomBytes = Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), validBytes]);
  const bomFile = new File([bomBytes], "bom.json");
  // File.text() would return clean JSON text without the BOM char:
  const masked = await bomFile.text();
  assert.equal(masked.charCodeAt(0) !== 0xfeff, true, "File.text() masks the BOM");
  // the strict entry does NOT:
  const strict = await readFileStrict(bomFile);
  assert.deepEqual(issueCodes(strict.issues), ["parse/E_PARSE@"]);

  const badBytes = Buffer.concat([Buffer.from('{"x": "'), Buffer.from([0xff]), Buffer.from('"}')]);
  const badFile = new File([badBytes], "bad.json");
  const maskedBad = await badFile.text();
  assert.equal(maskedBad.includes("\uFFFD"), true, "File.text() masks invalid UTF-8 as U+FFFD");
  const strictBad = await readFileStrict(badFile);
  assert.deepEqual(issueCodes(strictBad.issues), ["parse/E_PARSE@"]);
});

test("real File entry: valid UTF-8 (incl. astral) decodes and validates", async () => {
  const doc = minimalDoc();
  doc.instruments[0].label = "🎹合成";
  const file = new File([Buffer.from(JSON.stringify(doc), "utf8")], "ok.json");
  const { text, issues } = await readFileStrict(file);
  assert.deepEqual(issues, []);
  const { report } = parseAndValidate(text, schema, { source: file.name });
  assert.deepEqual(report.issues, []);
});

test("real File entry: BOM/invalid byte files never reach the validator as ok", async () => {
  for (const [name, bytes] of [
    ["bom.json", Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), validBytes])],
    ["bad.json", Buffer.concat([Buffer.from('{"label": "'), Buffer.from([0xff]), Buffer.from('"}')])],
  ]) {
    const { text, issues } = await readFileStrict(new File([bytes], name));
    assert.equal(text, null, name);
    assert.equal(issues.length > 0, true, name);
  }
});

test("parser-level BOM rule still applies to decoded text (mid-doc \\uFEFF stays legal in strings)", () => {
  assert.equal(loadText("\uFEFF{}").issues[0].code, "E_PARSE");
  const inside = loadText('{"s": "a\uFEFFb"}');
  assert.deepEqual(inside.issues, []);
  assert.equal(inside.data.s, "a\uFEFFb");
});

test("differential: Python frozen loader judges BOM/bad-UTF-8 files the same (E_PARSE)", (t) => {
  const python = ["python", "python3", "py"].find((candidate) => {
    const probe = spawnSync(candidate, ["--version"], { shell: false });
    return !probe.error && probe.status === 0;
  });
  if (!python) {
    t.skip("no python interpreter on PATH; differential check skipped");
    return;
  }
  const dir = mkdtempSync(join(tmpdir(), "aat-viewer-diff-bytes-"));
  try {
    const cases = {
      "bytes-bom.json": Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), validBytes]),
      "bytes-bad-utf8.json": Buffer.concat([Buffer.from('{"label": "'), Buffer.from([0xff]), Buffer.from('"}')]),
    };
    const files = [];
    for (const [name, bytes] of Object.entries(cases)) {
      const path = join(dir, name);
      writeFileSync(path, bytes);
      files.push(path);
    }
    const result = spawnSync(python, [join(REPO_ROOT, "agentic", "schema", "validate.py"), "--json", ...files], {
      shell: false,
      encoding: "utf8",
    });
    assert.equal(result.error, undefined);
    const byName = new Map();
    for (const line of result.stdout.split(/\r?\n/)) {
      if (!line.trim()) continue;
      const report = JSON.parse(line);
      byName.set(basename(report.source), report.issues.map((issue) => `${issue.layer}/${issue.code}@${issue.pointer}`));
    }
    for (const name of Object.keys(cases)) {
      assert.deepEqual(byName.get(name) || [], ["parse/E_PARSE@"], `python verdict for ${name}`);
      const { issues } = decodeUtf8Strict(cases[name], name);
      assert.deepEqual(issueCodes(issues), ["parse/E_PARSE@"], `js verdict for ${name}`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
