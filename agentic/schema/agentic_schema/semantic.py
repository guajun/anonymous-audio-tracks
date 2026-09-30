"""跨字段语义校验（issue #32 冻结规则，参考实现）。

规则清单的机器可读版本在 ``schema/semantic_rules.json``；本模块的
``SEMANTIC_CODES`` 与之保持一致（tests/test_docs_contract.py 会校对）。

设计原则：结构层（Draft 2020-12）管“形状”，语义层管“跨字段一致性”，
两者都拒绝并给出 JSON Pointer。语义层对缺字段保持防御（结构层会先报缺字段）。
"""

from __future__ import annotations

import re
from typing import Any

from .errors import Issue, pointer_of
from .loader import MAX_FLOAT, _number_out_of_policy, _number_repr

#: 稳定错误码（冻结；#33/#34 可依赖）
SEMANTIC_CODES = (
    "E_BOOL_NUMBER",
    "E_FINITE",
    "E_ID_UNIQUE",
    "E_PATH",
    "E_RANGE",
    "E_TEMPO",
    "E_TIME_DURATION",
    "E_TIME_ORDER",
    "E_TIME_RANGE",
)

#: 时间比较容差（秒）
TIME_TOLERANCE = 1e-9

#: 路径安全：Windows 保留设备名
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_FORBIDDEN_CHARS = set('<>:"|?*')
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def unsafe_path_reason(value: str) -> str | None:
    """返回路径不安全的原因；安全则返回 ``None``（冻结规则，见 E_PATH 文档）。"""
    if value == "":
        return "路径为空"
    if len(value) > 512:
        return f"路径超长（{len(value)} > 512）"
    for ch in value:
        if ord(ch) < 32 or ord(ch) == 127:
            return f"包含控制字符 {ch!r}"
    if "\\" in value:
        return "不允许反斜杠（UNC/Windows 分隔符）"
    if value.startswith("/"):
        return "不允许绝对路径（/ 开头）"
    if _DRIVE_RE.match(value):
        return "不允许盘符绝对路径（如 C:）"
    if value.startswith("//"):
        return "不允许 UNC 路径（// 开头）"
    if _URL_RE.match(value):
        return "不允许 URL（scheme://）"
    bad = sorted(set(value) & _FORBIDDEN_CHARS)
    if bad:
        return f"包含非法字符 {bad!r}"
    for segment in value.split("/"):
        if segment == "":
            return "存在空路径段（多余或结尾的 /）"
        if segment in (".", ".."):
            return f"不允许路径逃逸段 {segment!r}"
        if segment != segment.strip() or segment.endswith("."):
            return f"路径段首尾空格或结尾点不安全：{segment!r}"
        stem = segment.split(".")[0].upper()
        if stem in _RESERVED_NAMES:
            return f"Windows 保留设备名不安全：{segment!r}"
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pointer(parts: list[Any]) -> str:
    return pointer_of(parts)


class _Semantic:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.issues: list[Issue] = []

    def add(self, code: str, parts: list[Any], message: str) -> None:
        self.issues.append(Issue("semantic", code, _pointer(parts), message))

    # ---- 兜底：有限数 / 布尔冒充数字 ---------------------------------
    def walk_finite(self, node: Any, parts: list[Any]) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            if _number_out_of_policy(node):
                self.add(
                    "E_FINITE",
                    parts,
                    f"数字必须是有限数且在 float64 可互操作范围内（|x| <= {MAX_FLOAT}），得到 {_number_repr(node)}",
                )
            return
        if isinstance(node, dict):
            for key, value in node.items():
                self.walk_finite(value, parts + [key])
            return
        if isinstance(node, list):
            for index, value in enumerate(node):
                self.walk_finite(value, parts + [index])

    def check_number(self, node: Any, key: str, parts: list[Any]) -> float | None:
        """取对象里的数值字段；布尔冒充数字时报 E_BOOL_NUMBER，缺失/非数返回 None。"""
        if not isinstance(node, dict) or key not in node:
            return None
        value = node[key]
        if isinstance(value, bool):
            self.add("E_BOOL_NUMBER", parts + [key], f"{key} 必须是数字，布尔值不得冒充数字")
            return None
        if not _is_number(value):
            return None  # 类型错误由结构层报告
        try:
            return float(value)
        except (OverflowError, ValueError):
            self.add(
                "E_FINITE",
                parts + [key],
                f"{key} 数字超出 float64 可互操作范围，得到 {_number_repr(value)}（拒绝）",
            )
            return None

    def check_confidence(self, node: Any, parts: list[Any]) -> None:
        value = self.check_number(node, "confidence", parts)
        if value is None:
            return
        if not (0.0 <= value <= 1.0):
            self.add("E_RANGE", parts + ["confidence"], f"confidence 必须在 [0,1]，得到 {value!r}")

    # ---- 路径安全 -----------------------------------------------------
    def check_path(self, node: Any, key: str, parts: list[Any]) -> None:
        if not isinstance(node, dict) or key not in node:
            return
        value = node[key]
        if not isinstance(value, str):
            return  # 类型错误由结构层报告
        reason = unsafe_path_reason(value)
        if reason:
            self.add("E_PATH", parts + [key], f"路径不安全：{reason}")

    # ---- 时间边界 -----------------------------------------------------
    def check_times(self, duration: float | None) -> None:
        if not isinstance(self.data, dict):
            return
        tempo = self.data.get("tempo")
        if isinstance(tempo, dict) and "beat_origin_seconds" in tempo:
            value = self.check_number(tempo, "beat_origin_seconds", ["tempo"])
            if value is None:
                return
            if value < -TIME_TOLERANCE:
                self.add("E_TIME_RANGE", ["tempo", "beat_origin_seconds"], f"节拍原点不能为负，得到 {value!r}")
            elif duration is not None and value > duration + TIME_TOLERANCE:
                self.add(
                    "E_TIME_RANGE",
                    ["tempo", "beat_origin_seconds"],
                    f"节拍原点必须在 [0, {duration}] 秒内，得到 {value!r}",
                )

        instruments = self.data.get("instruments")
        if not isinstance(instruments, list):
            return
        for i, instrument in enumerate(instruments):
            if not isinstance(instrument, dict):
                continue
            events = instrument.get("events")
            if not isinstance(events, list):
                continue
            base = ["instruments", i, "events"]
            previous: tuple[float, str] | None = None
            for j, event in enumerate(events):
                if not isinstance(event, dict):
                    continue
                event_parts = base + [j]
                onset = self.check_number(event, "onset_seconds", event_parts)
                event_id = event.get("id") if isinstance(event.get("id"), str) else ""
                if onset is not None:
                    if onset < -TIME_TOLERANCE:
                        self.add("E_TIME_RANGE", event_parts + ["onset_seconds"], f"onset_seconds 不能为负，得到 {onset!r}")
                    elif duration is not None and onset > duration + TIME_TOLERANCE:
                        self.add(
                            "E_TIME_RANGE",
                            event_parts + ["onset_seconds"],
                            f"onset_seconds 必须在 [0, {duration}] 秒内（时间边界=音频末尾），得到 {onset!r}",
                        )
                    key = (onset, event_id)
                    if previous is not None and key < previous:
                        self.add(
                            "E_TIME_ORDER",
                            event_parts,
                            f"事件必须按 (onset_seconds 升序, id 升序) 排序：{key!r} 排在 {previous!r} 之后",
                        )
                    previous = key
                duration_value = self.check_number(event, "duration_seconds", event_parts)
                if duration_value is not None:
                    if duration_value <= 0:
                        self.add(
                            "E_TIME_DURATION",
                            event_parts + ["duration_seconds"],
                            f"duration_seconds 必须 > 0，得到 {duration_value!r}",
                        )
                    elif (
                        duration is not None
                        and onset is not None
                        and onset + duration_value > duration + TIME_TOLERANCE
                    ):
                        self.add(
                            "E_TIME_DURATION",
                            event_parts + ["duration_seconds"],
                            f"onset_seconds + duration_seconds 必须 <= 音频时长 {duration}（允许相等），"
                            f"得到 {onset} + {duration_value}",
                        )
                self.check_confidence(event, event_parts)
                if isinstance(event.get("pitch"), dict):
                    pitch = event["pitch"]
                    pitch_parts = event_parts + ["pitch"]
                    midi = self.check_number(pitch, "midi", pitch_parts)
                    if midi is not None:
                        if not float(midi).is_integer():
                            self.add("E_RANGE", pitch_parts + ["midi"], f"pitch.midi 必须是整数 0..127，得到 {midi!r}")
                        elif not (0 <= midi <= 127):
                            self.add("E_RANGE", pitch_parts + ["midi"], f"pitch.midi 必须在 0..127，得到 {midi!r}")
                    self.check_confidence(pitch, pitch_parts)

    # ---- ID 唯一 ------------------------------------------------------
    def check_ids(self) -> None:
        if not isinstance(self.data, dict):
            return
        seen_instruments: set[str] = set()
        seen_events: set[str] = set()
        instruments = self.data.get("instruments")
        if not isinstance(instruments, list):
            return
        for i, instrument in enumerate(instruments):
            if not isinstance(instrument, dict):
                continue
            instrument_id = instrument.get("id")
            if isinstance(instrument_id, str):
                if instrument_id in seen_instruments:
                    self.add(
                        "E_ID_UNIQUE",
                        ["instruments", i, "id"],
                        f"instrument.id 必须全文档唯一，重复：{instrument_id!r}",
                    )
                seen_instruments.add(instrument_id)
            events = instrument.get("events")
            if not isinstance(events, list):
                continue
            for j, event in enumerate(events):
                if not isinstance(event, dict):
                    continue
                event_id = event.get("id")
                if isinstance(event_id, str):
                    if event_id in seen_events:
                        self.add(
                            "E_ID_UNIQUE",
                            ["instruments", i, "events", j, "id"],
                            f"event.id 必须全文档唯一（跨乐器），重复：{event_id!r}",
                        )
                    seen_events.add(event_id)

    # ---- tempo 未知/已知互斥 -------------------------------------------
    def check_tempo(self) -> None:
        if not isinstance(self.data, dict):
            return
        tempo = self.data.get("tempo")
        if not isinstance(tempo, dict):
            return
        self.check_number(tempo, "bpm", ["tempo"])  # 布尔/超大数兜底
        bpm = tempo.get("bpm")
        source = tempo.get("source")
        has_origin = "beat_origin_seconds" in tempo
        if bpm is None:
            if source != "unknown":
                self.add("E_TEMPO", ["tempo", "source"], "bpm 为 null 时 source 必须是 \"unknown\"")
            confidence = self.check_number(tempo, "confidence", ["tempo"])
            if confidence is not None and confidence != 0.0:
                self.add("E_TEMPO", ["tempo", "confidence"], "bpm 为 null 时 confidence 必须是 0")
            if has_origin:
                self.add(
                    "E_TEMPO",
                    ["tempo", "beat_origin_seconds"],
                    "bpm 为 null 时不允许 beat_origin_seconds（没有节拍网格就没有原点）",
                )
        elif _is_number(bpm):
            if source == "unknown":
                self.add("E_TEMPO", ["tempo", "source"], "bpm 已知时 source 不得为 \"unknown\"（未知/已知互斥）")


def validate_semantic(data: Any) -> list[Issue]:
    """跨字段语义检查；错误按 (pointer, code) 稳定排序。"""
    ctx = _Semantic(data)
    ctx.walk_finite(data, [])
    ctx.check_ids()
    ctx.check_tempo()

    duration = None
    if isinstance(data, dict):
        audio = data.get("audio")
        if isinstance(audio, dict):
            duration = ctx.check_number(audio, "duration_seconds", ["audio"])
            ctx.check_path(audio, "filename", ["audio"])
        tempo = data.get("tempo")
        if isinstance(tempo, dict):
            ctx.check_confidence(tempo, ["tempo"])
        instruments = data.get("instruments")
        if isinstance(instruments, list):
            for i, instrument in enumerate(instruments):
                if not isinstance(instrument, dict):
                    continue
                parts = ["instruments", i]
                ctx.check_confidence(instrument, parts)
                if isinstance(instrument.get("stem"), dict):
                    ctx.check_path(instrument["stem"], "filename", parts + ["stem"])
    ctx.check_times(duration)
    return sorted(ctx.issues, key=lambda issue: (issue.pointer, issue.code, issue.message))
