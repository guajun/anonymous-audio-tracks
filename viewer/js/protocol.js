// Client-side reader/validator for the protocol 0.1.0 `trajectory.json`
// document defined in docs/SCHEMAS.md section 7.
//
// The viewer is a static page: it cannot import the Python contract code, so
// the subset of checks that matters for display and time alignment is mirrored
// here.  The rules (required fields, strict ordering, probability range, span
// containment, unique track ids, unknown schema version rejection) follow the
// Python implementation in src/aat/contracts/documents.py.  Unknown *extra*
// fields are tolerated, as the protocol requires of any reader.

export const SCHEMA_VERSION = "0.1.0";
export const DOCUMENT_KIND = "trajectory";
export const DATA_KINDS = ["model", "annotation", "mock"];
export const TRAJECTORY_TIME_TOLERANCE_SECONDS = 1e-6;
const GIT_COMMIT_PATTERN = /^[0-9a-f]{7,40}$/;
const UTC_TIMESTAMP_PATTERN = /(?:Z|\+00:00)$/;

/** Error carrying one issue per protocol violation found in a document. */
export class TrajectoryValidationError extends Error {
  constructor(issues, message) {
    const first = issues[0];
    super(message ?? `trajectory 校验失败: ${first.path}: ${first.message}`);
    this.name = "TrajectoryValidationError";
    this.issues = issues;
  }
}

class IssueCollector {
  constructor() {
    this.issues = [];
  }

  add(path, message) {
    this.issues.push({ path, message });
  }

  require(condition, path, message) {
    if (!condition) {
      this.add(path, message);
    }
    return condition;
  }

  throwIfAny(summary) {
    if (this.issues.length > 0) {
      throw new TrajectoryValidationError(this.issues, summary);
    }
  }
}

const isMapping = (value) =>
  value !== null && typeof value === "object" && !Array.isArray(value);
const isFiniteNumber = (value) =>
  typeof value === "number" && Number.isFinite(value);
const isInteger = (value) => Number.isInteger(value);
const isNonEmptyString = (value) =>
  typeof value === "string" && value.length > 0;
const isProbability = (value) =>
  isFiniteNumber(value) && value >= 0 && value <= 1;

function requireMapping(value, path, issues) {
  return issues.require(isMapping(value), path, "必须是 JSON 对象（mapping）");
}

function requireKeys(value, keys, path, issues) {
  for (const key of keys) {
    issues.require(
      Object.prototype.hasOwnProperty.call(value, key),
      `${path}.${key}`,
      "缺少必填字段",
    );
  }
}

function requireUnknownFieldsTolerated() {
  // The protocol requires readers to tolerate unknown extra fields; nothing to do.
}

function validateAudioSpan(raw, issues) {
  const path = "trajectory.audio";
  if (!requireMapping(raw, path, issues)) {
    return null;
  }
  requireKeys(raw, ["duration_seconds", "track_start_seconds"], path, issues);
  const duration = raw.duration_seconds;
  const trackStart = raw.track_start_seconds;
  issues.require(
    isFiniteNumber(duration) && duration >= 0,
    `${path}.duration_seconds`,
    `必须是 >= 0 的有限秒数，当前为 ${JSON.stringify(duration)}`,
  );
  issues.require(
    isFiniteNumber(trackStart) && trackStart >= 0,
    `${path}.track_start_seconds`,
    `必须是 >= 0 的有限秒数，当前为 ${JSON.stringify(trackStart)}`,
  );
  if (!isFiniteNumber(duration) || !isFiniteNumber(trackStart)) {
    return null;
  }
  return {
    durationSeconds: duration,
    trackStartSeconds: trackStart,
    endSeconds: trackStart + duration,
  };
}

function validateProvenance(raw, issues) {
  const path = "trajectory.provenance";
  if (!requireMapping(raw, path, issues)) {
    return null;
  }
  requireKeys(raw, ["run_id", "data_kind"], path, issues);
  issues.require(
    isNonEmptyString(raw.run_id),
    `${path}.run_id`,
    "必须是非空字符串",
  );
  issues.require(
    DATA_KINDS.includes(raw.data_kind),
    `${path}.data_kind`,
    `必须是 ${DATA_KINDS.join(" / ")} 之一，当前为 ${JSON.stringify(raw.data_kind)}`,
  );
  for (const name of ["model_id", "config_hash"]) {
    if (raw[name] !== undefined && raw[name] !== null) {
      issues.require(
        isNonEmptyString(raw[name]),
        `${path}.${name}`,
        "给出时必须是非空字符串",
      );
    }
  }
  if (raw.git_commit !== undefined && raw.git_commit !== null) {
    issues.require(
      typeof raw.git_commit === "string" &&
        GIT_COMMIT_PATTERN.test(raw.git_commit),
      `${path}.git_commit`,
      "必须是 7–40 位小写 hex",
    );
  }
  if (raw.created_at_utc !== undefined && raw.created_at_utc !== null) {
    const timestamp = raw.created_at_utc;
    const parseable =
      isNonEmptyString(timestamp) && Number.isFinite(Date.parse(timestamp));
    issues.require(
      parseable && UTC_TIMESTAMP_PATTERN.test(timestamp.trim()),
      `${path}.created_at_utc`,
      "必须是带 UTC 时区的 ISO-8601 时间戳（如 2026-09-29T12:00:00Z）",
    );
  }
  if (raw.notes !== undefined && raw.notes !== null) {
    issues.require(
      typeof raw.notes === "string",
      `${path}.notes`,
      "给出时必须是字符串",
    );
  }
  return {
    runId: raw.run_id,
    dataKind: raw.data_kind,
    modelId: raw.model_id ?? null,
    gitCommit: raw.git_commit ?? null,
    configHash: raw.config_hash ?? null,
    createdAtUtc: raw.created_at_utc ?? null,
    notes: raw.notes ?? null,
  };
}

function validateTrack(raw, position, slots, issues) {
  const path = `trajectory.tracks[${position}]`;
  if (!requireMapping(raw, path, issues)) {
    return null;
  }
  requireKeys(raw, ["track_id", "center_times", "activity"], path, issues);
  if (!isNonEmptyString(raw.track_id)) {
    issues.require(
      isNonEmptyString(raw.track_id),
      `${path}.track_id`,
      "必须是非空字符串（稳定身份，不是槽位号）",
    );
    return null;
  }
  const trackPath = `trajectory.tracks[${JSON.stringify(raw.track_id)}]`;

  const times = raw.center_times;
  const activity = raw.activity;
  if (!Array.isArray(times)) {
    issues.add(`${trackPath}.center_times`, "必须是数组");
    return null;
  }
  if (!Array.isArray(activity)) {
    issues.add(`${trackPath}.activity`, "必须是数组");
    return null;
  }
  if (times.length === 0) {
    issues.add(
      `${trackPath}.center_times`,
      "长度为 0 的曲线必须省略整个 track，不能存为空数组",
    );
  }
  for (let index = 0; index < times.length; index += 1) {
    const value = times[index];
    issues.require(
      isFiniteNumber(value) && value >= 0,
      `${trackPath}.center_times[${index}]`,
      `必须是 >= 0 的有限秒数，当前为 ${JSON.stringify(value)}`,
    );
  }
  for (let index = 1; index < times.length; index += 1) {
    if (isFiniteNumber(times[index - 1]) && isFiniteNumber(times[index])) {
      issues.require(
        times[index] > times[index - 1],
        `${trackPath}.center_times`,
        `必须严格递增（index ${index - 1} -> ${index}: ${times[index - 1]} -> ${times[index]}）`,
      );
    }
  }
  if (activity.length !== times.length) {
    issues.add(
      `${trackPath}.activity`,
      `长度 ${activity.length} 必须与 center_times 长度 ${times.length} 相同`,
    );
  } else {
    for (let index = 0; index < activity.length; index += 1) {
      issues.require(
        isProbability(activity[index]),
        `${trackPath}.activity[${index}]`,
        `必须是 [0, 1] 内的概率，当前为 ${JSON.stringify(activity[index])}`,
      );
    }
  }

  let confidence = null;
  if (raw.confidence !== undefined && raw.confidence !== null) {
    if (!Array.isArray(raw.confidence)) {
      issues.add(`${trackPath}.confidence`, "给出时必须是数组");
    } else if (raw.confidence.length !== times.length) {
      issues.add(
        `${trackPath}.confidence`,
        `长度 ${raw.confidence.length} 必须与 center_times 长度 ${times.length} 相同`,
      );
    } else {
      for (let index = 0; index < raw.confidence.length; index += 1) {
        issues.require(
          isProbability(raw.confidence[index]),
          `${trackPath}.confidence[${index}]`,
          `必须是 [0, 1] 内的概率，当前为 ${JSON.stringify(raw.confidence[index])}`,
        );
      }
      confidence = raw.confidence.slice();
    }
  }

  let slotIndices = null;
  if (raw.slot_indices !== undefined && raw.slot_indices !== null) {
    if (!Array.isArray(raw.slot_indices)) {
      issues.add(`${trackPath}.slot_indices`, "给出时必须是数组");
    } else if (raw.slot_indices.length !== times.length) {
      issues.add(
        `${trackPath}.slot_indices`,
        `长度 ${raw.slot_indices.length} 必须与 center_times 长度 ${times.length} 相同`,
      );
    } else {
      for (let index = 0; index < raw.slot_indices.length; index += 1) {
        const value = raw.slot_indices[index];
        if (value === null) {
          continue; // silence-memory point without a matched candidate
        }
        issues.require(
          isInteger(value) && value >= 0,
          `${trackPath}.slot_indices[${index}]`,
          `必须是 >= 0 的整数或 null，当前为 ${JSON.stringify(value)}`,
        );
        if (isInteger(value) && slots !== null && value >= slots) {
          issues.add(
            `${trackPath}.slot_indices[${index}]`,
            `槽位 ${value} 超出 0..${slots - 1}`,
          );
        }
      }
      slotIndices = raw.slot_indices.slice();
    }
  }

  if (raw.notes !== undefined && raw.notes !== null) {
    issues.require(
      typeof raw.notes === "string",
      `${trackPath}.notes`,
      "给出时必须是字符串",
    );
  }

  return {
    trackId: raw.track_id,
    centerTimes: times.slice(),
    activity: activity.slice(),
    confidence,
    slotIndices,
    notes: raw.notes ?? null,
  };
}

/**
 * Validate one parsed `trajectory.json` object.
 *
 * @returns the normalized trajectory document. Throws
 *   {@link TrajectoryValidationError} with every issue found.
 */
export function validateTrajectory(data) {
  const issues = new IssueCollector();
  requireUnknownFieldsTolerated();

  if (!requireMapping(data, "trajectory", issues)) {
    issues.throwIfAny("trajectory 必须是 JSON 对象");
  }
  if (data.schema_version !== SCHEMA_VERSION) {
    issues.add(
      "trajectory.schema_version",
      `未知协议版本 ${JSON.stringify(data.schema_version)}；本查看器只接受 "${SCHEMA_VERSION}"，拒绝猜测`,
    );
  }
  if (data.kind !== DOCUMENT_KIND) {
    issues.add(
      "trajectory.kind",
      `期望 "${DOCUMENT_KIND}"，当前为 ${JSON.stringify(data.kind)}`,
    );
  }
  requireKeys(data, ["audio", "provenance", "tracks"], "trajectory", issues);

  const audio = validateAudioSpan(data.audio, issues);
  const provenance = validateProvenance(data.provenance, issues);

  let slots = null;
  if (data.slots !== undefined && data.slots !== null) {
    if (issues.require(
      isInteger(data.slots) && data.slots >= 1,
      "trajectory.slots",
      "给出时必须是 >= 1 的整数",
    )) {
      slots = data.slots;
    }
  }

  if (data.sample_id !== undefined && data.sample_id !== null) {
    issues.require(
      isNonEmptyString(data.sample_id),
      "trajectory.sample_id",
      "给出时必须是非空字符串",
    );
  }
  if (data.params !== undefined && data.params !== null) {
    issues.require(
      isMapping(data.params),
      "trajectory.params",
      "给出时必须是 JSON 对象",
    );
  }

  const tracks = [];
  if (Array.isArray(data.tracks)) {
    const seen = new Set();
    for (let position = 0; position < data.tracks.length; position += 1) {
      const track = validateTrack(data.tracks[position], position, slots, issues);
      if (track === null) {
        continue;
      }
      if (seen.has(track.trackId)) {
        issues.add(
          `trajectory.tracks[${JSON.stringify(track.trackId)}].track_id`,
          "track_id 在同一文档内必须唯一",
        );
      }
      seen.add(track.trackId);
      tracks.push(track);
    }
  } else if (data.tracks !== undefined) {
    issues.add("trajectory.tracks", "必须是数组（无轨迹时为 []）");
  }

  if (audio) {
    if (audio.durationSeconds === 0 && tracks.length > 0) {
      issues.add(
        "trajectory.audio.duration_seconds",
        "为 0（空音频）时 tracks 必须为空",
      );
    }
    for (const track of tracks) {
      for (let index = 0; index < track.centerTimes.length; index += 1) {
        const value = track.centerTimes[index];
        if (!isFiniteNumber(value)) {
          continue;
        }
        if (
          value < audio.trackStartSeconds - TRAJECTORY_TIME_TOLERANCE_SECONDS ||
          value > audio.endSeconds + TRAJECTORY_TIME_TOLERANCE_SECONDS
        ) {
          issues.add(
            `trajectory.tracks[${JSON.stringify(track.trackId)}].center_times[${index}]`,
            `${value} 超出 audio 范围 [${audio.trackStartSeconds}, ${audio.endSeconds}]（容差 ${TRAJECTORY_TIME_TOLERANCE_SECONDS}）`,
          );
        }
      }
    }
  }

  issues.throwIfAny("trajectory 校验失败");

  return {
    schemaVersion: data.schema_version,
    kind: data.kind,
    sampleId: data.sample_id ?? null,
    slots,
    audio,
    provenance,
    tracks,
    params: isMapping(data.params) ? { ...data.params } : null,
  };
}

/** Parse raw text and validate it as a trajectory document. */
export function parseTrajectoryText(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch (error) {
    throw new TrajectoryValidationError(
      [{ path: "<file>", message: `不是有效的 JSON: ${error.message}` }],
      "JSON 解析失败",
    );
  }
  return validateTrajectory(data);
}

/** Human-readable multi-line summary of validation issues. */
export function formatIssues(issues) {
  return issues.map((issue) => `${issue.path}: ${issue.message}`).join("\n");
}

/** Label for the provenance `data_kind`, never let a mock masquerade as model. */
export function describeDataKind(dataKind) {
  switch (dataKind) {
    case "model":
      return "model（模型推理输出）";
    case "annotation":
      return "annotation（标注/算法标签）";
    case "mock":
      return "mock（模拟/占位数据，不是模型结果）";
    default:
      return String(dataKind);
  }
}
