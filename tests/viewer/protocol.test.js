import test from "node:test";
import assert from "node:assert/strict";

import {
  DATA_KINDS,
  parseTrajectoryText,
  TrajectoryValidationError,
  validateTrajectory,
} from "../../viewer/js/protocol.js";
import { clone, fixtureJson, fixtureText, issuesText } from "./test-helpers.js";

const validDocument = () => fixtureJson("trajectory.known-times.json");

function expectRejected(mutate, expectedPathFragment, expectedMessageFragment) {
  const document = validDocument();
  mutate(document);
  let error = null;
  try {
    validateTrajectory(document);
  } catch (thrown) {
    error = thrown;
  }
  assert.ok(error instanceof TrajectoryValidationError, "expected a TrajectoryValidationError");
  const text = issuesText(error);
  assert.ok(
    text.includes(expectedPathFragment),
    `expected issue path ${expectedPathFragment} in:\n${text}`,
  );
  if (expectedMessageFragment) {
    assert.ok(
      text.includes(expectedMessageFragment),
      `expected message ${expectedMessageFragment} in:\n${text}`,
    );
  }
  return error;
}

test("checked-in fixtures parse through the mirror validator", () => {
  const known = parseTrajectoryText(fixtureText("trajectory.known-times.json"));
  assert.equal(known.schemaVersion, "0.1.0");
  assert.equal(known.sampleId, "fixture-known-times");
  assert.equal(known.audio.trackStartSeconds, 0);
  assert.equal(known.audio.durationSeconds, 4);
  assert.equal(known.provenance.dataKind, "model");
  assert.equal(known.tracks.length, 3);
  assert.deepEqual(known.tracks[0].centerTimes, [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]);

  const clip = parseTrajectoryText(fixtureText("trajectory.clip.json"));
  assert.equal(clip.audio.trackStartSeconds, 12);
  assert.equal(clip.tracks[0].trackId, "trk-clip-0001");

  const empty = parseTrajectoryText(fixtureText("trajectory.empty.json"));
  assert.equal(empty.audio.durationSeconds, 0);
  assert.deepEqual(empty.tracks, []);

  const hostile = parseTrajectoryText(fixtureText("trajectory.hostile-strings.json"));
  assert.equal(hostile.tracks[0].trackId, "trk-<b>html</b>");
});

test("trajectory fixture documents stay valid under the Python contract vocabulary", () => {
  const document = validDocument();
  assert.equal(document.kind, "trajectory");
  assert.ok(DATA_KINDS.includes(document.provenance.data_kind));
  for (const track of document.tracks) {
    assert.equal(track.center_times.length, track.activity.length);
  }
});

test("unknown schema_version is rejected, never guessed", () => {
  const error = expectRejected((document) => {
    document.schema_version = "0.2.0";
  }, "trajectory.schema_version", "未知协议版本");
  assert.match(issuesText(error), /0\.1\.0/);
});

test("wrong kind is rejected", () => {
  expectRejected((document) => {
    document.kind = "prediction";
  }, "trajectory.kind");
});

test("missing required top-level fields are reported per path", () => {
  const error = expectRejected((document) => {
    delete document.audio;
    delete document.provenance;
  }, "trajectory.audio", "缺少必填字段");
  assert.ok(issuesText(error).includes("trajectory.provenance"));
});

test("duplicate track_id is rejected", () => {
  expectRejected((document) => {
    document.tracks[1].track_id = "trk-a";
  }, "track_id", "必须唯一");
});

test("non-increasing center_times are rejected", () => {
  expectRejected((document) => {
    document.tracks[0].center_times[3] = document.tracks[0].center_times[2];
  }, "center_times", "严格递增");
});

test("probabilities outside [0, 1] are rejected", () => {
  const error = expectRejected((document) => {
    document.tracks[0].activity[1] = 1.2;
    document.tracks[0].activity[2] = -0.1;
  }, "activity", "概率");
  assert.ok(issuesText(error).includes("1.2"));
  assert.ok(issuesText(error).includes("-0.1"));
});

test("numeric overflow from JSON (1e999) becomes Infinity and is rejected", () => {
  const text = fixtureText("trajectory.known-times.json").replace("[0.0, 0.5", "[1e999, 0.5");
  assert.throws(
    () => parseTrajectoryText(text),
    (error) => {
      assert.ok(error instanceof TrajectoryValidationError);
      assert.match(issuesText(error), /有限秒数/);
      return true;
    },
  );
});

test("negative or non-finite audio span values are rejected", () => {
  expectRejected((document) => {
    document.audio.track_start_seconds = -1;
  }, "track_start_seconds");
  expectRejected((document) => {
    document.audio.duration_seconds = Number.NaN;
  }, "duration_seconds");
});

test("center times outside the declared audio span are rejected", () => {
  expectRejected((document) => {
    document.tracks[0].center_times[7] = 4.5;
  }, "center_times", "超出 audio 范围");
});

test("empty audio span with tracks is rejected", () => {
  expectRejected((document) => {
    document.audio = { duration_seconds: 0, track_start_seconds: 0 };
  }, "duration_seconds", "必须为空");
});

test("zero-length track curves must be omitted", () => {
  expectRejected((document) => {
    document.tracks[0].center_times = [];
    document.tracks[0].activity = [];
  }, "center_times", "省略整个 track");
});

test("activity/confidence length mismatches are reported", () => {
  expectRejected((document) => {
    document.tracks[0].activity = document.tracks[0].activity.slice(0, 3);
  }, "trk-a", "长度");
  expectRejected((document) => {
    document.tracks[0].confidence = [1, 2];
  }, "confidence", "长度");
});

test("data_kind must be model / annotation / mock", () => {
  expectRejected((document) => {
    document.provenance.data_kind = "model-v2";
  }, "data_kind", "model");
});

test("malformed optional provenance fields are rejected", () => {
  expectRejected((document) => {
    document.provenance.git_commit = "XYZ";
  }, "git_commit");
  expectRejected((document) => {
    document.provenance.created_at_utc = "2026-09-29T12:00:00";
  }, "created_at_utc");
  expectRejected((document) => {
    document.provenance.run_id = "";
  }, "run_id");
});

test("slot_indices must be integers in range or null", () => {
  expectRejected((document) => {
    document.tracks[0].slot_indices[0] = 8;
  }, "slot_indices", "超出");
  expectRejected((document) => {
    document.tracks[0].slot_indices[0] = 1.5;
  }, "slot_indices");
});

test("unknown extra fields are tolerated per protocol reader rules", () => {
  const document = validDocument();
  document.future_top_level = { anything: true };
  document.tracks[0].future_track_field = 42;
  document.provenance.future = "ok";
  const parsed = validateTrajectory(document);
  assert.equal(parsed.tracks.length, 3);
});

test("invalid JSON text produces a parse error, not a silent empty document", () => {
  assert.throws(
    () => parseTrajectoryText("{not json"),
    (error) => {
      assert.match(issuesText(error), /不是有效的 JSON/);
      return true;
    },
  );
});

test("non-object roots are rejected", () => {
  assert.throws(
    () => parseTrajectoryText("[]"),
    (error) => {
      assert.match(issuesText(error), /JSON 对象/);
      return true;
    },
  );
});

test("all issues are collected in one pass for actionable error panels", () => {
  const document = clone(validDocument());
  document.tracks[0].activity[1] = 2;
  document.tracks[0].center_times[3] = 0.5;
  document.tracks[1].track_id = "";
  let error = null;
  try {
    validateTrajectory(document);
  } catch (thrown) {
    error = thrown;
  }
  assert.ok(error);
  assert.ok(error.issues.length >= 3, `expected >= 3 issues, got ${error.issues.length}`);
});
