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
 *     2. the tool validates the file (controlled root, size budget, magic-byte
 *        MIME sniffing against an allow-list) and reads the raw bytes;
 *     3. a `context` handler injects the audio into the next model request as
 *        a user content block carrying the true audio MIME type and the base64
 *        bytes;
 *     4. Pi's own Google provider converter (pi-ai `convertMessages`) turns
 *        that block into `parts: [{ inlineData: { mimeType: "audio/wav",
 *        data: <base64> } }]` — the Gemini API's native audio payload shape.
 *        Offline proof: `node probe/payload_shape.mjs`.
 *
 *   TYPE-LABEL ABUSE (documented on purpose): the injected block is typed
 *   `"image"` because that is the only non-text content block Pi accepts, but
 *   its `mimeType` is an audio MIME (e.g. "audio/wav") and its `data` is raw
 *   audio bytes. Nothing here is Pi-native audio; the audio only reaches the
 *   model because the Gemini API itself natively accepts audio inlineData.
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
 * LIMITS:
 *   - one-shot injection: each attached file is injected into the next model
 *     request once; later requests in the same run do not re-send the bytes.
 *   - the `path` label of the block is "image" (see above), so any consumer
 *     that sniffs content-block types instead of MIME will misclassify it.
 *   - no resampling/transcoding: whatever bytes the file holds are sent as-is
 *     under the sniffed container MIME.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { createHash } from "node:crypto";
import { readFile, realpath, stat } from "node:fs/promises";
import { resolve as resolvePath } from "node:path";
import { fileURLToPath } from "node:url";
import {
  DEFAULT_MAX_BYTES,
  checkSize,
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

function audioRoot(): string {
  const override = process.env.PI_AUDIO_BRIDGE_ROOT?.trim();
  return resolvePath(override && override.length > 0 ? override : AUDIO_ROOT_DEFAULT);
}

function maxBytes(): number {
  const raw = Number(process.env.PI_AUDIO_BRIDGE_MAX_BYTES ?? "");
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : DEFAULT_MAX_BYTES;
}

export default function audioBridgeExtension(pi: ExtensionAPI) {
  /** Queue of validated attachments waiting for the next model request. */
  const pending: PendingAudio[] = [];

  pi.registerTool({
    name: "audio_attach",
    label: "Audio Attach",
    description:
      "Attach a local audio file so the model can hear it on the next request. " +
      "The file must live inside the controlled audio root and use a known audio " +
      "container (wav, mp3, ogg, flac, m4a). The raw bytes are delivered to the " +
      "model as inline audio data via the project-local audio bridge. " +
      "Use `path` relative to the audio root, e.g. \"fixture-a.wav\".",
    promptSnippet: "Attach a local audio file for listening (project audio bridge)",
    promptGuidelines: [
      "Use audio_attach to let the model listen to a local audio file before describing it.",
      "Do not claim to have heard audio that was only provided as text or a file name.",
    ],
    parameters: Type.Object({
      path: Type.String({
        description: 'Audio file path relative to the controlled audio root (e.g. "fixture-a.wav").',
      }),
    }),
    async execute(_toolCallId, params) {
      try {
        const root = audioRoot();
        const limit = maxBytes();
        const candidate = resolveWithinRoot(root, params.path);

        let resolved: string;
        try {
          resolved = await realpath(candidate);
        } catch {
          throw new Error("audio file not found inside the controlled audio root");
        }
        // Defeat symlink escapes: the real path must stay inside the real root.
        const realRoot = await realpath(root).catch(() => root);
        if (!isWithinRoot(realRoot, resolved)) {
          throw new Error("path escapes the controlled audio root and was rejected");
        }

        const info = await stat(resolved);
        if (!info.isFile()) throw new Error("path is not a regular file");
        checkSize(info.size, limit);

        const raw = await readFile(resolved);
        const mime = sniffAudioMime(raw);
        if (!mime) {
          throw new Error("file is not a recognized audio container (wav/mp3/ogg/flac/m4a)");
        }

        const item: PendingAudio = {
          name: safeBasename(resolved),
          mime,
          data: raw.toString("base64"),
          bytes: raw.length,
          sha256: createHash("sha256").update(raw).digest("hex"),
        };
        pending.push(item);

        const summary = {
          attached: item.name,
          mime: item.mime,
          bytes: item.bytes,
          sha256: item.sha256,
          queueDepth: pending.length,
          note: "Audio bytes are injected into the next model request as inline data by the project audio bridge (not Pi native audio support).",
        };
        return {
          content: [{ type: "text", text: JSON.stringify(summary) }],
          details: summary,
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          content: [{ type: "text", text: `audio_attach failed: ${message}` }],
          details: { error: message },
        };
      }
    },
  });

  // Inject pending audio as user content blocks for the next model request.
  // `event.messages` excludes system messages; Pi restores its own state after
  // the handler, so the transcript is untouched while the request carries the
  // audio exactly once per attachment.
  pi.on("context", (event) => {
    if (pending.length === 0) return;
    const batch = pending.splice(0, pending.length);
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
}
