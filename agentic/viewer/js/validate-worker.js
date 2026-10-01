// Web Worker: strict JSON parse + full `agentic-audio-tracks/v1` validation off
// the UI thread (large documents must not block the page).
// Message in:  { id, text, schema, source }
// Message out: { id, report, data }

import { parseAndValidate } from "./protocol.js";

self.addEventListener("message", (event) => {
  const { id, text, schema, source } = event.data || {};
  try {
    const { data, report } = parseAndValidate(String(text), schema, { source: source || "<file>" });
    self.postMessage({ id, report, data });
  } catch (error) {
    self.postMessage({
      id,
      report: {
        source: source || "<file>",
        ok: false,
        engine: "js-worker",
        schema_version: "agentic-audio-tracks/v1",
        issues: [
          {
            layer: "parse",
            code: "E_PARSE",
            pointer: "",
            message: `校验器内部错误（受控拒绝）: ${String(error && error.message ? error.message : error)}`,
          },
        ],
      },
      data: null,
    });
  }
});
