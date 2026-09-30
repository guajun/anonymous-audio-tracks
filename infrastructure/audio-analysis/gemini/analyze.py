"""Run one Gemini audio-analysis request and validate the JSON protocol.

The script deliberately keeps credentials in the environment and writes every
request to a new run directory. ``--dry-run`` checks local inputs without
creating a client or making a network request.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.toml"
EXAMPLE_CONFIG = ROOT / "config.example.toml"


def _resolve_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ((base or ROOT) / path).resolve()


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the optional local TOML config, falling back to safe defaults."""

    config_path = path or DEFAULT_CONFIG
    if not config_path.is_file():
        config_path = EXAMPLE_CONFIG
    if not config_path.is_file():
        return {}
    with config_path.open("rb") as stream:
        return tomllib.load(stream)


def validate(data: dict[str, Any], duration_s: float = 30) -> None:
    """Validate the stable subset of ``anonymous-audio-tracks/v1``."""

    assert data["schema"] == "anonymous-audio-tracks/v1"
    assert data["duration_s"] == duration_s
    assert isinstance(data["audio_access"], bool)
    assert isinstance(data.get("target_identified"), bool)
    sources = data["sources"]
    assert isinstance(sources, list) and len(sources) <= 1
    if not data["audio_access"] or data.get("target_identified") is False:
        assert not sources
    for source in sources:
        assert source["id"] == "S1"
        previous = -1.0
        for event in source["onsets"]:
            timestamp = event["time_s"]
            assert 0 <= timestamp < duration_s and timestamp >= previous
            assert 0 <= event["confidence"] <= 1
            assert event["timing_uncertainty_ms"] >= 0
            previous = timestamp
        for event in source["active_intervals"] + source["anchors"]:
            assert 0 <= event["start_s"] < event["end_s"] <= duration_s


def _clean_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:])
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return cleaned


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--auxiliary", type=Path, help="Optional aligned 0-30s separated reference")
    parser.add_argument("--model", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(_resolve_path(args.config) if args.config else None)
    gemini_config = config.get("gemini", {})
    paths_config = config.get("paths", {})
    duration_s = float(args.duration or gemini_config.get("duration_s", 30))

    load_dotenv(ROOT / ".env")
    model = args.model or os.getenv("GEMINI_MODEL") or gemini_config.get(
        "model", "gemini-3.8-flash"
    )
    audio = _resolve_path(
        args.audio
        or paths_config.get(
            "audio", "../../../data/audio-analysis/breeze/Breeze-first30s.wav"
        )
    )
    reference = _resolve_path(
        args.reference
        or paths_config.get(
            "reference", "../../../data/audio-analysis/breeze/Breeze-anchor-10.5-12.0.wav"
        )
    )
    prompt = _resolve_path(
        args.prompt or paths_config.get("prompt", "prompts/prompt-breeze-v2.txt")
    )
    run_root = _resolve_path(gemini_config.get("runs_dir", "runs"))

    paths = [audio, reference, prompt]
    if args.auxiliary:
        paths.append(_resolve_path(args.auxiliary))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        _parser().error("Missing input file(s): " + ", ".join(missing))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": model,
                    "duration_s": duration_s,
                    "audio": str(audio),
                    "reference": str(reference),
                    "auxiliary": str(_resolve_path(args.auxiliary)) if args.auxiliary else None,
                    "prompt": str(prompt),
                    "runs_dir": str(run_root),
                    "api_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        print("请设置 GEMINI_API_KEY 环境变量，或在本目录未跟踪的 .env 中填写。")
        return 2

    run = run_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True, exist_ok=False)
    prompt_text = prompt.read_text(encoding="utf-8-sig")
    (run / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    (run / "request.json").write_text(
        json.dumps(
            {
                "model": model,
                "duration_s": duration_s,
                "audio": str(audio),
                "reference": str(reference),
                "auxiliary": str(_resolve_path(args.auxiliary)) if args.auxiliary else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    uploaded: list[str] = []
    with genai.Client(api_key=key) as client:
        try:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            inputs = [
                ("reference (original 10.5-12.0s)", reference),
                ("full_clip (original 0-30s)", audio),
            ]
            if args.auxiliary:
                auxiliary = _resolve_path(args.auxiliary)
                inputs.append(
                    (
                        "auxiliary: machine-separated target aligned to full_clip; verify timestamps in full_clip",
                        auxiliary,
                    )
                )
            for label, path in inputs:
                print("Uploading", path.name, flush=True)
                uploaded_file = client.files.upload(file=str(path))
                uploaded.append(uploaded_file.name)
                deadline = time.monotonic() + 180
                while getattr(uploaded_file.state, "name", str(uploaded_file.state)) == "PROCESSING":
                    if time.monotonic() > deadline:
                        raise TimeoutError("File processing timeout")
                    time.sleep(2)
                    uploaded_file = client.files.get(name=uploaded_file.name)
                if getattr(uploaded_file.state, "name", str(uploaded_file.state)) == "FAILED":
                    raise RuntimeError("File processing failed")
                content.extend(
                    [
                        {"type": "text", "text": label},
                        {
                            "type": "audio",
                            "uri": uploaded_file.uri,
                            "mime_type": uploaded_file.mime_type
                            or mimetypes.guess_type(path.name)[0]
                            or "audio/wav",
                        },
                    ]
                )

            print("Calling", model, flush=True)
            response = client.interactions.create(model=model, input=content, timeout=600)
            response_text = response.output_text or ""
            (run / "response.txt").write_text(response_text, encoding="utf-8")
            try:
                data = json.loads(_clean_json(response_text))
                validate(data, duration_s)
            except (ValueError, KeyError, AssertionError, TypeError):
                print(f"返回未通过 JSON/时间字段检查；原文已保存：{run / 'response.txt'}")
                return 3
            (run / "tracks.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"已保存：{run / 'tracks.json'}")
            return 0
        except Exception as error:  # Do not expose SDK headers or credentials.
            print(
                f"调用失败：{type(error).__name__}；HTTP code={getattr(error, 'code', None)}。"
                "检查网络、模型权限及 key。"
            )
            return 1
        finally:
            for name in uploaded:
                try:
                    client.files.delete(name=name)
                except Exception:
                    print("一个临时上传文件未能自动删除，请在 Google Files 中检查。")


if __name__ == "__main__":
    raise SystemExit(main())
