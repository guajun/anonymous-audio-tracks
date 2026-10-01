// Validation pipeline for the frozen `agentic-audio-tracks/v1` protocol:
// strict parse -> version match -> structure -> semantics.
//
// Browser port of `agentic/schema/agentic_schema/pipeline.py` (issue #32).
// Reports use the same shape as the Python reference Report.as_dict():
//   { source, ok, engine, schema_version, issues: [{layer, code, pointer, message}] }

import { depthIssues, loadText, numericIssues } from "./strict-json.js";
import { validateStructure } from "./structure.js";
import { validateSemantics } from "./semantic.js";

export const SCHEMA_VERSION = "agentic-audio-tracks/v1";
export const ENGINE = "js-stdlib";

function makeReport(source, ok, issues) {
  return {
    source,
    ok,
    engine: ENGINE,
    schema_version: SCHEMA_VERSION,
    issues,
  };
}

function compareIssues(a, b) {
  if (a.pointer !== b.pointer) return a.pointer < b.pointer ? -1 : 1;
  if (a.code !== b.code) return a.code < b.code ? -1 : 1;
  if (a.message !== b.message) return a.message < b.message ? -1 : 1;
  return 0;
}

/**
 * Validate an already parsed document (still enforces depth/number policy).
 * @param {unknown} data
 * @param {object} schema  parsed Draft 2020-12 schema (portable subset)
 */
export function validateDocument(data, schema, options = {}) {
  const source = options.source || "<document>";
  const parseIssues = depthIssues(data);
  if (parseIssues.length === 0) {
    parseIssues.push(...numericIssues(data));
  }
  if (parseIssues.length > 0) {
    return makeReport(source, false, parseIssues);
  }

  const version = data !== null && typeof data === "object" && !Array.isArray(data) ? data.schema_version : null;
  if (version !== SCHEMA_VERSION) {
    return makeReport(source, false, [
      {
        layer: "version",
        code: "E_VERSION",
        pointer: "/schema_version",
        message: `schema_version 必须精确等于 ${JSON.stringify(SCHEMA_VERSION)}，得到 ${JSON.stringify(
          version === undefined ? null : version,
        )}（不匹配时只报本错误）`,
      },
    ]);
  }

  let issues = [];
  try {
    issues = issues.concat(validateStructure(data, schema));
    issues = issues.concat(validateSemantics(data));
  } catch (error) {
    return makeReport(source, false, [
      {
        layer: "parse",
        code: "E_PARSE",
        pointer: "",
        message: `文档嵌套过深（递归上限），拒绝（${String(error && error.message ? error.message : error)}）`,
      },
    ]);
  }
  issues.sort(compareIssues);
  return makeReport(source, issues.length === 0, issues);
}

/**
 * Validate JSON text (includes the strict parse layer).
 * @param {string} text
 * @param {object} schema parsed schema document
 */
export function validateText(text, schema, options = {}) {
  const source = options.source || "<text>";
  const { data, issues } = loadText(text, { source });
  if (issues.length > 0) {
    return makeReport(source, false, issues);
  }
  return validateDocument(data, schema, { source });
}

/** Validate text and also return the parsed document when parsing succeeded. */
export function parseAndValidate(text, schema, options = {}) {
  const source = options.source || "<text>";
  const { data, issues } = loadText(text, { source });
  if (issues.length > 0) {
    return { data: null, report: makeReport(source, false, issues) };
  }
  return { data, report: validateDocument(data, schema, { source }) };
}
