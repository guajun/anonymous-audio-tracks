// Shared helpers for the viewer tests.

import { readFileSync } from "node:fs";

export function fixtureText(name) {
  return readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");
}

export function fixtureJson(name) {
  return JSON.parse(fixtureText(name));
}

export function approx(actual, expected, tolerance = 1e-9) {
  return Math.abs(actual - expected) <= tolerance;
}

export function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

/** Collect issue paths from a thrown TrajectoryValidationError. */
export function issuePaths(error) {
  return error.issues.map((issue) => issue.path);
}

export function issuesText(error) {
  return error.issues
    .map((issue) => `${issue.path}: ${issue.message}`)
    .join("\n");
}
