/**
 * audio-bridge.ts — project-local Pi extension, issue #30 (agentic/audio-probe).
 *
 * WHAT THIS IS (read carefully):
 *   Pi 0.87.1 has NO native audio modality. There is no AudioContent block in
 *   the message type system (UserMessage / ToolResultMessage / CustomMessage
 *   accept only TextContent | ImageContent), the Google provider catalog entry
 *   for google/gemini-3.8-flash declares `input: ["text", "image"]`, and every
 *   documented audio-adjacent entry point (`@file` CLI args, the `read` tool,
 *   RPC `prompt.images`, SDK `prompt({images})`, `sendUserMessage`) runs
 *   attachments through `processImage()`, which rejects non-image bytes with
 *   "[Image omitted: could not be converted to a supported inline image format.]".
 *   See reports/20-native-source-audit.md for the evidence.
 *
 *   This extension is therefore a BRIDGE, not native support. It is an
 *   honest, minimal workaround:
 *     1. the agent calls the `audio_attach` tool (agent-triggered);
 *     2. the tool validates the file (controlled root, per-file and cumulative
 *        size budgets, magic-byte MIME sniffing against an allow-list) and
 *        reads the raw bytes (re-checked after reading);
 *     3. a `context` handler injects the audio into the next model request as
 *        a user content block carrying the true audio MIME type and the base64
 *        bytes;
 *     4. Pi's own Google provider converter (pi-ai dist/api/google-shared.js
 *        `convertMessages`) turns that block into
 *        `parts: [{ inlineData: { mimeType: "audio/wav", data: <base64> } }]`
 *        — the Gemini API's native audio payload shape.
 *        Offline proof: `node probe/payload_shape.mjs`.
 *
 *   TYPE-LABEL ABUSE (documented on purpose): the injected block is typed
 *   `"image"` because that is the only non-text content block Pi accepts, but
 *   its `mimeType` is an audio MIME (e.g. "audio/wav") and its `data` is raw
 *   audio bytes. Nothing here is Pi-native audio; the audio only reaches the
 *   model because the Gemini API itself natively accepts audio inlineData.
 *
 * MODEL GUARANTEE:
 *   Attachment and injection only proceed when the active model is exactly
 *   `google/gemini-3.8-flash` (research model fixed by issue #30). The lock is
 *   a hard constant: NO environment override exists (an inherited env var must
 *   never silently retarget audio uploads to another provider/model). Any
 *   other provider/model is refused with E_AUDIO_MODEL; queued audio is
 *   dropped (never injected) if the model changes before injection.
 *
 * CREDENTIALS:
 *   The extension never touches credentials. The model call is made by Pi
 *   itself with Pi's own provider auth resolution; no key is read, printed,
 *   or persisted by this code.
 *
 * UPLOAD / CLEANUP:
 *   Audio is sent as request-inline bytes (`inlineData`). The Gemini Files API
 *   is NOT used, so no server-side temporary file is created and there is
 *   nothing to delete afterwards. Local fixtures live under `fixtures/`
 *   (gitignored) and `probe/clean_fixtures.py` removes them.
 *
 * SIZE BUDGETS:
 *   - per file:        PI_AUDIO_BRIDGE_MAX_BYTES       (default 4 MiB raw)
 *   - queued total:    PI_AUDIO_BRIDGE_MAX_QUEUE_BYTES (default 8 MiB raw;
 *     base64 costs 4/3 x raw, so <= ~10.7 MiB inline, inside the 20 MiB
 *     request cap that the model catalog declares).
 *
 * LIFECYCLE:
 *   Queued audio is cleared on `session_start` / `session_shutdown` /
 *   `agent_settled`, so pending bytes cannot spill into another session or
 *   survive an aborted run. Injection is one-shot per attachment and the audio
 *   is never written to the session transcript (kept in memory only).
 *
 * FAILURE SEMANTICS:
 *   `execute()` THROWS on invalid input (Pi marks a thrown tool result as a
 *   failed tool result; returning a normal object would read as success).
 *   Thrown messages carry a stable E_AUDIO_* code and never contain absolute
 *   paths.
 *
 * LIMITS:
 *   - one-shot injection: each attached file is injected into the next model
 *     request once; later requests in the same run do not re-send the bytes.
 *   - the `path` label of the block is "image" (see above), so any consumer
 *     that sniffs content-block types instead of MIME will misclassify it.
 *   - no resampling/transcoding: whatever bytes the file holds are sent as-is
 *     under the sniffed container MIME.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createHash } from "node:crypto";
import { readFile, realpath, stat } from "node:fs/promises";
import { resolve as resolvePath } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_MAX_BYTES,
  ERROR_CODES,
  checkQueueBudget,
  checkSize,
  fail,
  isWithinRoot,
  resolveWithinRoot,
  safeBasename,
  sniffAudioMime,
} from "./audio_guard.mjs";

interface PendingAudio {
  name: string;
  mime: string;
  data: string; // base64
  bytes: number;
  sha256: string;
}

const AUDIO_ROOT_DEFAULT = fileURLToPath(new URL("../fixtures", import.meta.url));

/** Cumulative queued raw bytes; base64 adds ~33% on the wire. */
const DEFAULT_MAX_QUEUE_BYTES = 8 * 1024 * 1024;

/** Research model fixed by issue #30. Hard constant on purpose: no env override. */
const EXPECTED_MODEL = Object.freeze({ provider: "google", id: "gemini-3.8-flash" });

function audioRoot(): string {
  const override = process.env.PI_AUDIO_BRIDGE_ROOT?.trim();
  return resolvePath(override && override.length > 0 ? override : AUDIO_ROOT_DEFAULT);
}

function envBytes(name: string, fallback: number): number {
  const raw = Number(process.env[name] ?? "");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : fallback;
}

function isExpectedModel(model: { provider?: string; id?: string } | undefined): boolean {
  return model?.provider === EXPECTED_MODEL.provider && model?.id === EXPECTED_MODEL.id;
}

export default function audioBridgeExtension(pi: ExtensionAPI) {
  /** Queue of validated attachments waiting for the next model request. */
  const pending: PendingAudio[] = [];
  let queuedBytes = 0;

  const clearPending = (reason: string) => {
    if (pending.length > 0) {
      pending.splice(0, pending.length);
      queuedBytes = 0;
      void reason; // kept for future diagnostics; intentionally silent
    }
  };

  pi.registerTool({
    name: "audio_attach",
    label: "Audio Attach",
    description:
      "Attach a local audio file so the model can hear it on the next request. " +
      "The file must live inside the controlled audio root and use a known audio " +
      "container (wav, mp3, ogg, flac, m4a). The raw bytes are delivered to the " +
      "model as inline audio data via the project-local audio bridge (not Pi " +
      "native audio). Use `path` relative to the audio root, e.g. \"fixture-a.wav\".",
    promptSnippet: "Attach a local audio file for listening (project audio bridge)",
    promptGuidelines: [
      "Use audio_attach to let the model listen to a local audio file before describing it.",
      "Do not claim to have heard audio that was only provided as text or a file name.",
    ],
    // Hand-written JSON schema (no typebox import) so this module is also
    // loadable by plain Node for offline extension-level tests.
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: 'Audio file path relative to the controlled audio root (e.g. "fixture-a.wav").',
        },
      },
      required: ["path"],
      additionalProperties: false,
    },
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      // 1. Model guarantee: only the fixed research model may receive audio.
      if (!isExpectedModel(ctx?.model)) {
        throw fail(ERROR_CODES.MODEL, `audio bridge is restricted to ${EXPECTED_MODEL.provider}/${EXPECTED_MODEL.id}`);
      }
      if (typeof params?.path !== "string") {
        throw fail(ERROR_CODES.ARGS, "path must be a string");
      }

      const root = audioRoot();
      const limit = envBytes("PI_AUDIO_BRIDGE_MAX_BYTES", DEFAULT_MAX_BYTES);
      const queueLimit = envBytes("PI_AUDIO_BRIDGE_MAX_QUEUE_BYTES", DEFAULT_MAX_QUEUE_BYTES);

      // 2. Controlled path (guard rejects traversal/absolute escapes).
      let candidate: string;
      try {
        candidate = resolveWithinRoot(root, params.path);
      } catch (err) {
        throw fail(ERROR_CODES.PATH, err instanceof Error ? err.message : String(err));
      }

      let resolved: string;
      try {
        resolved = await realpath(candidate);
      } catch {
        throw fail(ERROR_CODES.NOT_FOUND, "audio file not found inside the controlled audio root");
      }
      try {
        const realRoot = await realpath(root).catch(() => root);
        if (!isWithinRoot(realRoot, resolved)) {
          throw fail(ERROR_CODES.PATH, "path escapes the controlled audio root and was rejected");
        }
      } catch (err) {
        if (err instanceof Error && err.message.startsWith("E_AUDIO_")) throw err;
        throw fail(ERROR_CODES.PATH, "path could not be verified against the audio root");
      }

      let size: number;
      try {
        const info = await stat(resolved);
        if (!info.isFile()) throw fail(ERROR_CODES.NOT_FILE, "path is not a regular file");
        size = info.size;
      } catch (err) {
        if (err instanceof Error && err.message.startsWith("E_AUDIO_")) throw err;
        throw fail(ERROR_CODES.READ, "audio file metadata could not be read");
      }

      // 3. Per-file budget (stat) and cumulative queue budget.
      try {
        checkSize(size, limit);
        checkQueueBudget(queuedBytes, size, queueLimit);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        throw fail(message.includes("cumulative") ? ERROR_CODES.QUEUE : ERROR_CODES.SIZE, message);
      }

      // 4. Read, then RE-check the actual byte count (stat can be stale).
      let raw: Buffer;
      try {
        raw = await readFile(resolved);
      } catch {
        throw fail(ERROR_CODES.READ, "audio file could not be read");
      }
      try {
        checkSize(raw.length, limit);
        checkQueueBudget(queuedBytes, raw.length, queueLimit);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        throw fail(message.includes("cumulative") ? ERROR_CODES.QUEUE : ERROR_CODES.SIZE, message);
      }

      // 5. MIME allow-list via magic bytes (text/images can never pass).
      const mime = sniffAudioMime(raw);
      if (!mime) {
        throw fail(ERROR_CODES.MIME, "file is not a recognized audio container (wav/mp3/ogg/flac/m4a)");
      }

      const item: PendingAudio = {
        name: safeBasename(resolved),
        mime,
        data: raw.toString("base64"),
        bytes: raw.length,
        sha256: createHash("sha256").update(raw).digest("hex"),
      };
      pending.push(item);
      queuedBytes += item.bytes;

      const summary = {
        attached: item.name,
        mime: item.mime,
        bytes: item.bytes,
        sha256: item.sha256,
        queueDepth: pending.length,
        queuedBytes,
        note: "Audio bytes are injected into the next model request as inline data by the project audio bridge (not Pi native audio support).",
      };
      // Tool result carries metadata only: no base64 audio is persisted or
      // echoed back; the bytes live in memory until the one-shot injection.
      return {
        content: [{ type: "text", text: JSON.stringify(summary) }],
        details: summary,
      };
    },
  });

  // Inject pending audio as user content blocks for the next model request.
  // `event.messages` excludes system messages; Pi restores its own state after
  // the handler, so the transcript is untouched while the request carries the
  // audio exactly once per attachment (never persisted to the session file).
  pi.on("context", (event, ctx) => {
    if (pending.length === 0) return;
    // Never inject into a different provider/model than the one attachment
    // was validated for: drop rather than leak audio to another model.
    if (!isExpectedModel(ctx?.model)) {
      clearPending("model-changed");
      return;
    }
    const batch = pending.splice(0, pending.length);
    queuedBytes = 0;
    for (const item of batch) {
      event.messages.push({
        role: "user",
        timestamp: Date.now(),
        content: [
          {
            type: "text",
            text:
              `[audio-bridge] attached audio "${item.name}" ` +
              `(${item.mime}, ${item.bytes} bytes, sha256 ${item.sha256.slice(0, 16)}…). ` +
              `The audio data follows as inline content.`,
          },
          // Typed "image" because Pi has no audio content block; the MIME and
          // bytes are audio and travel to the provider unchanged as inlineData.
          {
            type: "image",
            mimeType: item.mime,
            data: item.data,
          },
        ],
      });
    }
    return { messages: event.messages };
  });

  // Lifecycle boundaries: pending audio must not spill into another session or
  // survive an aborted/settled run.
  pi.on("session_start", () => clearPending("session_start"));
  pi.on("session_shutdown", () => clearPending("session_shutdown"));
  pi.on("agent_settled", () => clearPending("agent_settled"));
}
