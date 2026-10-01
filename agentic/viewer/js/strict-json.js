// Strict JSON parsing for the frozen `agentic-audio-tracks/v1` contract.
//
// Browser port of `agentic/schema/agentic_schema/loader.py` (issue #32 frozen
// policy). `JSON.parse` alone is not enough because it
//   - silently accepts duplicate object keys (last-wins),
//   - turns `1e999` into Infinity without complaint,
//   - gives no JSON Pointer for offending numbers.
//
// Frozen policy mirrored here:
//   - NaN / Infinity / -Infinity literals           -> E_PARSE
//   - syntax errors, BOM, trailing data             -> E_PARSE
//   - nesting deeper than MAX_DEPTH (root = 1)      -> E_PARSE (controlled, single issue)
//   - duplicate object keys                         -> E_DUPLICATE_KEY (reject, not last-wins)
//   - numbers outside finite float64 interop range  -> E_NONFINITE (with exact pointer)

export const MAX_DEPTH = 64;
export const MAX_FLOAT = 1.7976931348623157e308;

/** RFC 6901 pointer escaping (`~` -> `~0`, `/` -> `~1`). */
export function escapeSegment(segment) {
  return String(segment).replace(/~/g, "~0").replace(/\//g, "~1");
}

/** Build a JSON Pointer from path segments; empty path is the root (`""`). */
export function pointerOf(parts) {
  if (!parts || parts.length === 0) return "";
  return parts.map((part) => `/${escapeSegment(part)}`).join("");
}

export function makeIssue(layer, code, pointer, message) {
  return { layer, code, pointer, message };
}

class AbortParse extends Error {
  constructor(issues) {
    super(issues.map((i) => i.message).join("; "));
    this.issues = issues;
  }
}

function syntaxError(source, text, pos, message) {
  let line = 1;
  let col = 1;
  for (let index = 0; index < pos && index < text.length; index += 1) {
    if (text.charCodeAt(index) === 10) {
      line += 1;
      col = 1;
    } else {
      col += 1;
    }
  }
  return makeIssue(
    "parse",
    "E_PARSE",
    "",
    `JSON 语法错误（${source} 第 ${line} 行第 ${col} 列）: ${message}`,
  );
}

function numberRepr(raw) {
  return raw.length <= 64 ? raw : `${raw.slice(0, 64)}...`;
}

class StrictParser {
  constructor(text, source) {
    this.text = text;
    this.source = source;
    this.pos = 0;
    this.nonfinite = [];
  }

  fail(message, pos = this.pos) {
    throw new AbortParse([syntaxError(this.source, this.text, pos, message)]);
  }

  skipWs() {
    const { text } = this;
    while (this.pos < text.length) {
      const code = text.charCodeAt(this.pos);
      if (code === 32 || code === 9 || code === 10 || code === 13) {
        this.pos += 1;
      } else {
        break;
      }
    }
  }

  parseDocument() {
    const { text } = this;
    if (text.length > 0 && text.charCodeAt(0) === 0xfeff) {
      throw new AbortParse([
        makeIssue("parse", "E_PARSE", "", "拒绝 UTF-8 BOM（\\uFEFF 开头不是合法 JSON 文本）"),
      ]);
    }
    this.skipWs();
    if (this.pos >= text.length) {
      this.fail("空文档，不是合法 JSON");
    }
    const data = this.parseValue([], 1);
    this.skipWs();
    if (this.pos < text.length) {
      this.fail("JSON 语法错误：值结束后存在多余数据");
    }
    if (this.nonfinite.length > 0) {
      return { data: null, issues: this.nonfinite };
    }
    return { data, issues: [] };
  }

  parseValue(parts, depth) {
    // Depth policy: the node at `parts` has depth parts.length + 1 (root container = 1).
    const nodeDepth = parts.length + 1;
    if (nodeDepth > MAX_DEPTH) {
      throw new AbortParse([
        makeIssue(
          "parse",
          "E_PARSE",
          pointerOf(parts),
          `文档嵌套过深（深度上限 ${MAX_DEPTH}，此处已到第 ${nodeDepth} 层）；不支持深嵌套 JSON，拒绝`,
        ),
      ]);
    }
    const ch = this.text[this.pos];
    if (ch === undefined) this.fail("JSON 语法错误：文档提前结束");
    if (ch === "{") return this.parseObject(parts, depth);
    if (ch === "[") return this.parseArray(parts, depth);
    if (ch === '"') return this.parseString();
    if (ch === "-" && this.text.startsWith("-Infinity", this.pos)) return this.parseConstant();
    if (ch === "-" || (ch >= "0" && ch <= "9")) return this.parseNumber(parts);
    if (ch === "N" || ch === "I") return this.parseConstant();
    if (ch === "t" || ch === "f" || ch === "n") return this.parseKeyword();
    return this.fail(`JSON 语法错误：意外字符 ${JSON.stringify(ch)}`);
  }

  parseConstant() {
    const rest = this.text.slice(this.pos);
    const match = /^(-?Infinity|NaN)/.exec(rest);
    const name = match ? match[0] : rest.slice(0, 8);
    throw new AbortParse([
      makeIssue(
        "parse",
        "E_PARSE",
        "",
        `拒绝非标准 JSON 常量 ${name}（NaN/Infinity/-Infinity 不是合法 JSON）`,
      ),
    ]);
  }

  parseKeyword() {
    for (const [literal, value] of [["true", true], ["false", false], ["null", null]]) {
      if (this.text.startsWith(literal, this.pos)) {
        this.pos += literal.length;
        return value;
      }
    }
    return this.fail("JSON 语法错误：无法识别的字面量");
  }

  parseNumber(parts) {
    const start = this.pos;
    const { text } = this;
    if (text[this.pos] === "-") this.pos += 1;
    if (text[this.pos] === "0") {
      this.pos += 1;
    } else if (text[this.pos] >= "1" && text[this.pos] <= "9") {
      while (text[this.pos] >= "0" && text[this.pos] <= "9") this.pos += 1;
    } else {
      return this.fail("JSON 语法错误：非法数字");
    }
    if (text[this.pos] === ".") {
      this.pos += 1;
      if (!(text[this.pos] >= "0" && text[this.pos] <= "9")) {
        return this.fail("JSON 语法错误：小数点后缺少数字");
      }
      while (text[this.pos] >= "0" && text[this.pos] <= "9") this.pos += 1;
    }
    if (text[this.pos] === "e" || text[this.pos] === "E") {
      this.pos += 1;
      if (text[this.pos] === "+" || text[this.pos] === "-") this.pos += 1;
      if (!(text[this.pos] >= "0" && text[this.pos] <= "9")) {
        return this.fail("JSON 语法错误：指数部分缺少数字");
      }
      while (text[this.pos] >= "0" && text[this.pos] <= "9") this.pos += 1;
    }
    const raw = text.slice(start, this.pos);
    const value = Number(raw);
    if (!Number.isFinite(value) || Math.abs(value) > MAX_FLOAT) {
      // Frozen numeric policy: record with exact pointer, keep parsing to report all.
      this.nonfinite.push(
        makeIssue(
          "parse",
          "E_NONFINITE",
          pointerOf(parts),
          `数字必须是有限数且在 float64 可互操作范围内（|x| <= ${MAX_FLOAT}），得到 ${numberRepr(raw)}（拒绝 NaN/Inf/超大数）`,
        ),
      );
      return 0;
    }
    return value;
  }

  parseString() {
    const { text } = this;
    this.pos += 1; // opening quote
    let out = "";
    for (;;) {
      const ch = text[this.pos];
      if (ch === undefined) return this.fail("JSON 语法错误：字符串未闭合");
      const code = text.charCodeAt(this.pos);
      if (ch === '"') {
        this.pos += 1;
        return out;
      }
      if (code < 0x20) {
        return this.fail(`JSON 语法错误：字符串包含未转义控制字符 (U+${code.toString(16).padStart(4, "0").toUpperCase()})`);
      }
      if (ch === "\\") {
        this.pos += 1;
        const esc = text[this.pos];
        if (esc === undefined) return this.fail("JSON 语法错误：字符串提前结束（孤立转义符）");
        this.pos += 1;
        if (esc === '"' || esc === "\\" || esc === "/") out += esc;
        else if (esc === "b") out += "\b";
        else if (esc === "f") out += "\f";
        else if (esc === "n") out += "\n";
        else if (esc === "r") out += "\r";
        else if (esc === "t") out += "\t";
        else if (esc === "u") {
          const hex = text.slice(this.pos, this.pos + 4);
          if (!/^[0-9a-fA-F]{4}$/.test(hex)) return this.fail("JSON 语法错误：\\u 转义需要 4 位十六进制");
          out += String.fromCharCode(parseInt(hex, 16));
          this.pos += 4;
        } else {
          return this.fail(`JSON 语法错误：非法转义符 \\${esc}`);
        }
        continue;
      }
      out += ch;
      this.pos += 1;
    }
  }

  parseObject(parts, depth) {
    this.pos += 1; // {
    // Null-prototype-free plain object, but every key is created as an OWN
    // property: raw `"__proto__"` keys must survive as own keys (prototype
    // pollution guard) so the structural layer can reject them like Python.
    const out = {};
    const seen = new Set();
    this.skipWs();
    if (this.text[this.pos] === "}") {
      this.pos += 1;
      return out;
    }
    for (;;) {
      this.skipWs();
      if (this.text[this.pos] !== '"') this.fail("JSON 语法错误：对象键必须是字符串");
      const key = this.parseString();
      if (seen.has(key)) {
        throw new AbortParse([
          makeIssue(
            "parse",
            "E_DUPLICATE_KEY",
            "",
            `对象存在重复键 ${JSON.stringify(key)}（拒绝 last-wins；重复键无法给出完整路径）`,
          ),
        ]);
      }
      seen.add(key);
      this.skipWs();
      if (this.text[this.pos] !== ":") this.fail("JSON 语法错误：对象键后缺少 ':'");
      this.pos += 1;
      this.skipWs();
      Object.defineProperty(out, key, {
        value: this.parseValue(parts.concat([key]), depth + 1),
        enumerable: true,
        writable: true,
        configurable: true,
      });
      this.skipWs();
      const ch = this.text[this.pos];
      if (ch === ",") {
        this.pos += 1;
        continue;
      }
      if (ch === "}") {
        this.pos += 1;
        return out;
      }
      this.fail("JSON 语法错误：对象成员后缺少 ',' 或 '}'");
    }
  }

  parseArray(parts, depth) {
    this.pos += 1; // [
    const out = [];
    this.skipWs();
    if (this.text[this.pos] === "]") {
      this.pos += 1;
      return out;
    }
    for (;;) {
      this.skipWs();
      out.push(this.parseValue(parts.concat([out.length]), depth + 1));
      this.skipWs();
      const ch = this.text[this.pos];
      if (ch === ",") {
        this.pos += 1;
        continue;
      }
      if (ch === "]") {
        this.pos += 1;
        return out;
      }
      this.fail("JSON 语法错误：数组元素后缺少 ',' 或 ']'");
    }
  }
}

/**
 * Strict UTF-8 decoding for FILE ENTRY POINTS (mirrors `loader.load_path`).
 *
 * `File.text()` silently strips a UTF-8 BOM and replaces invalid UTF-8 bytes
 * with U+FFFD, so the strict parse layer would never see either problem. This
 * helper checks the raw bytes first: a leading BOM is rejected explicitly and
 * invalid UTF-8 is a controlled `E_PARSE` — same verdict as the frozen Python
 * loader (`raw.decode("utf-8")` strict + BOM rejection).
 *
 * @param {ArrayBuffer|Uint8Array} bytes
 * @returns {{text: string|null, issues: Array}}
 */
export function decodeUtf8Strict(bytes, source = "<bytes>") {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (view.length >= 3 && view[0] === 0xef && view[1] === 0xbb && view[2] === 0xbf) {
    return {
      text: null,
      issues: [makeIssue("parse", "E_PARSE", "", `拒绝 UTF-8 BOM（\uFEFF 开头不是合法 JSON 文本，${source}）`)],
    };
  }
  try {
    // fatal: invalid byte sequences throw; ignoreBOM: keep \uFEFF visible so
    // the parser's own BOM rule applies to any later occurrence too.
    const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
    return { text: decoder.decode(view), issues: [] };
  } catch {
    return {
      text: null,
      issues: [
        makeIssue(
          "parse",
          "E_PARSE",
          "",
          `文件必须是 UTF-8 文本（${source}）：存在非法 UTF-8 字节序列（受控拒绝，不用替换字符\uFFFD 静默容错）`,
        ),
      ],
    };
  }
}

/** Strict read of a picked File/Blob: raw bytes -> strict UTF-8 -> strict JSON. */
export async function readFileStrict(file) {
  const source = (file && file.name) || "<file>";
  let bytes;
  try {
    bytes = await file.arrayBuffer();
  } catch (error) {
    return {
      text: null,
      issues: [
        makeIssue("parse", "E_PARSE", "", `无法读取文件（${source}）：${String(error && error.message ? error.message : error)}`),
      ],
    };
  }
  return decodeUtf8Strict(bytes, source);
}

/**
 * Strictly parse JSON text.
 * @returns {{data: unknown|null, issues: Array<{layer: string, code: string, pointer: string, message: string}>}}
 */
export function loadText(text, options = {}) {
  const source = options.source || "<text>";
  if (typeof text !== "string") {
    return {
      data: null,
      issues: [makeIssue("parse", "E_PARSE", "", "输入必须是文本（UTF-8 JSON）")],
    };
  }
  try {
    return new StrictParser(text, source).parseDocument();
  } catch (error) {
    if (error instanceof AbortParse) {
      return { data: null, issues: error.issues };
    }
    return {
      data: null,
      issues: [
        makeIssue(
          "parse",
          "E_PARSE",
          "",
          `JSON 解析失败（受控拒绝，不抛出异常）: ${String(error && error.message ? error.message : error)}`,
        ),
      ],
    };
  }
}

/** Depth policy check for programmatically built documents (root container = 1). */
export function depthIssues(data, maxDepth = MAX_DEPTH) {
  const stack = [[data, 1, []]];
  while (stack.length > 0) {
    const [node, depth, parts] = stack.pop();
    if (depth > maxDepth) {
      return [
        makeIssue(
          "parse",
          "E_PARSE",
          pointerOf(parts),
          `文档嵌套过深（深度上限 ${maxDepth}，此处已到第 ${depth} 层）；不支持深嵌套 JSON，拒绝`,
        ),
      ];
    }
    if (Array.isArray(node)) {
      node.forEach((value, index) => stack.push([value, depth + 1, parts.concat([index])]));
    } else if (node !== null && typeof node === "object") {
      for (const [key, value] of Object.entries(node)) {
        stack.push([value, depth + 1, parts.concat([key])]);
      }
    }
  }
  return [];
}

/** Numeric policy check for programmatically built documents (E_NONFINITE). */
export function numericIssues(data) {
  const out = [];
  const walk = (node, parts) => {
    if (typeof node === "number") {
      if (!Number.isFinite(node) || Math.abs(node) > MAX_FLOAT) {
        out.push(
          makeIssue(
            "parse",
            "E_NONFINITE",
            pointerOf(parts),
            `数字必须是有限数且在 float64 可互操作范围内（|x| <= ${MAX_FLOAT}），拒绝 NaN/Inf/超大数`,
          ),
        );
      }
      return;
    }
    if (Array.isArray(node)) {
      node.forEach((value, index) => walk(value, parts.concat([index])));
    } else if (node !== null && typeof node === "object") {
      for (const [key, value] of Object.entries(node)) {
        walk(value, parts.concat([key]));
      }
    }
  };
  walk(data, []);
  return out;
}
