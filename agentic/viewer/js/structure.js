// Structural validation: zero-dependency port of the Draft 2020-12 portable
// subset validator in `agentic/schema/agentic_schema/structure.py` (issue #32).
//
// Supported keywords (anything else is an internal error):
//   type const enum properties required additionalProperties items $defs $ref
//   pattern minimum exclusiveMinimum maximum minLength maxLength minItems
//
// Booleans are not numbers (`true` is not `1`); integer fields accept
// integral floats such as `2.0` (same as the `jsonschema` engine).

import { pointerOf } from "./strict-json.js";

export const SUPPORTED_KEYWORDS = new Set([
  "$schema",
  "$id",
  "$defs",
  "$ref",
  "title",
  "description",
  "type",
  "const",
  "enum",
  "properties",
  "required",
  "additionalProperties",
  "items",
  "pattern",
  "minimum",
  "exclusiveMinimum",
  "maximum",
  "minLength",
  "maxLength",
  "minItems",
]);

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isInteger(value) {
  return typeof value === "number" && Number.isInteger(value);
}

function typeOk(value, name) {
  switch (name) {
    case "object":
      return value !== null && typeof value === "object" && !Array.isArray(value);
    case "array":
      return Array.isArray(value);
    case "string":
      return typeof value === "string";
    case "number":
      return isNumber(value);
    case "integer":
      return isInteger(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      throw new Error(`unsupported type name: ${name}`);
  }
}

function jsonEqual(a, b) {
  if (isNumber(a) && isNumber(b)) return a === b;
  return typeof a === typeof b && a === b;
}

/** Code-point string ordering (must match the Python reference sort). */
export function cmpText(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

function typeLabel(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function repr(value) {
  try {
    const text = JSON.stringify(value);
    return text === undefined ? String(value) : text;
  } catch {
    return String(value);
  }
}

function resolveRef(ref, root) {
  if (!ref.startsWith("#/$defs/")) {
    throw new Error(`only internal #/$defs/ refs are supported, got ${ref}`);
  }
  const name = ref.split("/").pop();
  return root.$defs[name];
}

function validateNode(instance, schema, root, parts, out) {
  const unknown = Object.keys(schema).filter((key) => !SUPPORTED_KEYWORDS.has(key));
  if (unknown.length > 0) {
    throw new Error(`unsupported schema keywords: ${JSON.stringify(unknown)}`);
  }

  if ("$ref" in schema) {
    validateNode(instance, resolveRef(schema.$ref, root), root, parts, out);
    return;
  }

  const pointer = pointerOf(parts);

  if ("const" in schema && !jsonEqual(instance, schema.const)) {
    out.push({
      layer: "structure",
      code: "const",
      pointer,
      message: `必须等于 ${repr(schema.const)}，得到 ${repr(instance)}`,
    });
  }
  if ("enum" in schema && !schema.enum.some((choice) => jsonEqual(instance, choice))) {
    out.push({
      layer: "structure",
      code: "enum",
      pointer,
      message: `必须是 ${repr(schema.enum)} 之一，得到 ${repr(instance)}`,
    });
  }

  if (schema.type !== undefined) {
    const names = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!names.some((name) => typeOk(instance, name))) {
      out.push({
        layer: "structure",
        code: "type",
        pointer,
        message: `类型应为 ${names.join(" 或 ")}，得到 ${typeLabel(instance)} ${repr(instance)}`,
      });
      return; // wrong type: stop descending (same as the reference implementation)
    }
  }

  if (instance !== null && typeof instance === "object" && !Array.isArray(instance)) {
    const required = schema.required || [];
    const missing = required.filter((key) => !(key in instance));
    if (missing.length > 0) {
      out.push({
        layer: "structure",
        code: "required",
        pointer,
        message: `缺少必填字段 ${repr(missing)}`,
      });
    }
    const properties = schema.properties || {};
    for (const [key, value] of Object.entries(instance)) {
      if (key in properties) {
        validateNode(value, properties[key], root, parts.concat([key]), out);
      }
    }
    if (schema.additionalProperties === false) {
      const extras = Object.keys(instance).filter((key) => !(key in properties));
      if (extras.length > 0) {
        out.push({
          layer: "structure",
          code: "additionalProperties",
          pointer,
          message: `不允许出现未定义字段 ${repr(extras)}（未知字段策略=拒绝；加字段=新版本）`,
        });
      }
    }
    return;
  }

  if (Array.isArray(instance)) {
    if (schema.minItems !== undefined && instance.length < schema.minItems) {
      out.push({
        layer: "structure",
        code: "minItems",
        pointer,
        message: `至少需要 ${schema.minItems} 项，得到 ${instance.length}`,
      });
    }
    if (schema.items !== undefined) {
      instance.forEach((value, index) => {
        validateNode(value, schema.items, root, parts.concat([index]), out);
      });
    }
    return;
  }

  if (typeof instance === "string") {
    if (schema.pattern !== undefined) {
      // The frozen patterns end with (?![\s\S]) so they anchor absolutely.
      if (new RegExp(schema.pattern).test(instance) === false) {
        out.push({
          layer: "structure",
          code: "pattern",
          pointer,
          message: `不匹配模式 ${repr(schema.pattern)}，得到 ${repr(instance)}`,
        });
      }
    }
    if (schema.minLength !== undefined && instance.length < schema.minLength) {
      out.push({
        layer: "structure",
        code: "minLength",
        pointer,
        message: `长度至少 ${schema.minLength}，得到 ${instance.length}`,
      });
    }
    if (schema.maxLength !== undefined && instance.length > schema.maxLength) {
      out.push({
        layer: "structure",
        code: "maxLength",
        pointer,
        message: `长度最多 ${schema.maxLength}，得到 ${instance.length}`,
      });
    }
    return;
  }

  if (isNumber(instance)) {
    if (schema.minimum !== undefined && instance < schema.minimum) {
      out.push({
        layer: "structure",
        code: "minimum",
        pointer,
        message: `必须 >= ${schema.minimum}，得到 ${repr(instance)}`,
      });
    }
    if (schema.exclusiveMinimum !== undefined && instance <= schema.exclusiveMinimum) {
      out.push({
        layer: "structure",
        code: "exclusiveMinimum",
        pointer,
        message: `必须 > ${schema.exclusiveMinimum}，得到 ${repr(instance)}`,
      });
    }
    if (schema.maximum !== undefined && instance > schema.maximum) {
      out.push({
        layer: "structure",
        code: "maximum",
        pointer,
        message: `必须 <= ${schema.maximum}，得到 ${repr(instance)}`,
      });
    }
  }
}

/** Validate `instance` against `schema`; issues sorted by (pointer, code, message). */
export function validateStructure(instance, schema) {
  const out = [];
  validateNode(instance, schema, schema, [], out);
  return out.sort(
    (a, b) => cmpText(a.pointer, b.pointer) || cmpText(a.code, b.code) || cmpText(a.message, b.message),
  );
}
