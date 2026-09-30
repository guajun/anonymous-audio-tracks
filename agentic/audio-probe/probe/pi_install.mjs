/**
 * pi_install.mjs — locate the installed Pi 0.87.1 package tree.
 *
 * Used by the offline probes (`payload_shape.mjs`, `native_paths_probe.mjs`)
 * so they execute Pi's *real* code instead of reimplementing it.
 *
 * Resolution order:
 *   1. PI_INSTALL_DIR  -> directory of @earendil-works/pi-coding-agent
 *   2. PI_PACKAGE_DIR  -> .../node_modules/@earendil-works/pi-coding-agent
 *   3. $PI_CODING_AGENT_DIR (default ~/.pi/agent)/install/releases/<newest>
 *      /node_modules/@earendil-works/pi-coding-agent
 */
import { readdirSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

function newestReleaseDir(installRoot) {
  const releasesDir = join(installRoot, "install", "releases");
  if (!existsSync(releasesDir)) return null;
  const versions = readdirSync(releasesDir).filter((name) => /^\d+\.\d+\.\d+$/.test(name));
  versions.sort((a, b) => {
    const pa = a.split(".").map(Number);
    const pb = b.split(".").map(Number);
    return pa[0] - pb[0] || pa[1] - pb[1] || pa[2] - pb[2];
  });
  return versions.length > 0 ? join(releasesDir, versions[versions.length - 1]) : null;
}

export function findPiCodingAgentDir() {
  const explicit = process.env.PI_INSTALL_DIR?.trim();
  if (explicit && existsSync(explicit)) return resolve(explicit);

  const pkgDir = process.env.PI_PACKAGE_DIR?.trim();
  if (pkgDir) {
    const candidate = join(pkgDir, "node_modules", "@earendil-works", "pi-coding-agent");
    if (existsSync(candidate)) return candidate;
  }

  const agentDir = process.env.PI_CODING_AGENT_DIR?.trim() || join(homedir(), ".pi", "agent");
  const release = newestReleaseDir(agentDir);
  if (release) {
    const candidate = join(release, "node_modules", "@earendil-works", "pi-coding-agent");
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

export function piUrls() {
  const dir = findPiCodingAgentDir();
  if (!dir) return null;
  const piAiDir = resolve(dir, "..", "pi-ai");
  return {
    dir,
    piAiDir,
    codingAgentUtils: (rel) => pathToFileURL(join(dir, "dist", "utils", rel)).href,
    codingAgentCli: (rel) => pathToFileURL(join(dir, "dist", "cli", rel)).href,
    piAiApi: (rel) => pathToFileURL(join(piAiDir, "dist", "api", rel)).href,
    googleCatalog: join(piAiDir, "dist", "providers", "data", "google.json"),
  };
}
