// Minimal local static server for `agentic/viewer` ONLY.
//
//   node agentic/viewer/tools/serve.mjs [--port 8123]
//
// Security constraints (issue #34):
//   - binds 127.0.0.1 only,
//   - serves exclusively the `agentic/viewer/` directory (never the repository
//     root, never `.local/`, never personal files),
//   - GET/HEAD only, no directory listing, path traversal is rejected.
//
// The viewer itself loads JSON/audio through local file inputs, so the server
// only has to deliver the page assets.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { statSync } from "node:fs";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
export const VIEWER_ROOT = resolve(HERE, "..");

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
};

export function startServer({ port = 8123, root = VIEWER_ROOT } = {}) {
  const server = createServer(async (request, response) => {
    try {
      if (request.method !== "GET" && request.method !== "HEAD") {
        response.writeHead(405, { "content-type": "text/plain; charset=utf-8" }).end("method not allowed");
        return;
      }
      const url = new URL(request.url, "http://127.0.0.1");
      let pathname = decodeURIComponent(url.pathname);
      if (pathname === "/") pathname = "/index.html";
      const target = resolve(root, `.${pathname}`);
      if (target !== root && !target.startsWith(root + sep)) {
        response.writeHead(403, { "content-type": "text/plain; charset=utf-8" }).end("forbidden");
        return;
      }
      if (!isFile(target)) {
        response.writeHead(404, { "content-type": "text/plain; charset=utf-8" }).end("not found");
        return;
      }
      const body = await readFile(target);
      response.writeHead(200, {
        "content-type": MIME_TYPES[extname(target).toLowerCase()] || "application/octet-stream",
        "cache-control": "no-store",
      });
      response.end(request.method === "HEAD" ? undefined : body);
    } catch {
      response.writeHead(500, { "content-type": "text/plain; charset=utf-8" }).end("server error");
    }
  });
  return new Promise((resolvePromise, rejectPromise) => {
    server.once("error", rejectPromise);
    server.listen(port, "127.0.0.1", () => {
      resolvePromise({ server, port: server.address().port, url: `http://127.0.0.1:${server.address().port}/index.html` });
    });
  });
}

function isFile(target) {
  try {
    return statSync(target).isFile();
  } catch {
    return false;
  }
}

export async function main(argv = process.argv.slice(2)) {
  let port = 8123;
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--port") port = Number(argv[++index]);
  }
  const { url } = await startServer({ port });
  process.stdout.write(
    `agentic/viewer 本地服务（仅 127.0.0.1，仅 ${VIEWER_ROOT}）:\n  ${url}\n`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
