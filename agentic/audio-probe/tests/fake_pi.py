#!/usr/bin/env python3
"""fake_pi.py — offline stand-in for the `pi` binary (issue #30 review item 1).

Used by tests/test_real_run_scripts.py to regression-test the REAL probe
runners (`probe/run_real_*.sh` + `probe/run_bounded.py`) WITHOUT any API call:

    FAKE_PI_MODE=ok          valid event stream, exit 0
    FAKE_PI_MODE=fail        stderr message, exit 3
    FAKE_PI_MODE=hang        sleep far beyond any timeout (runner must kill)
    FAKE_PI_MODE=error_event Pi exits 0 but the stream reports a provider error

`auth ...` subcommands always answer `{"status":"ready",...}` quickly so the
scripts' auth pre-check does not interfere with the mode under test.
"""
from __future__ import annotations

import json
import os
import sys
import time


def event(record: dict) -> None:
    print(json.dumps(record))


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "auth":
        print('{"status":"ready","provider":"google","authType":"api_key"}')
        return 0

    mode = os.environ.get("FAKE_PI_MODE", "ok")

    if mode == "hang":
        time.sleep(120)
        return 0

    if mode == "fail":
        print("fake pi: simulated command failure", file=sys.stderr)
        return 3

    event({"type": "session", "version": 3, "id": "fake-session", "timestamp": 0})

    if mode == "error_event":
        event(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "provider": "google",
                    "model": "gemini-3.8-flash",
                    "stopReason": "error",
                    "errorMessage": "fake provider unavailable",
                    "content": [{"type": "text", "text": "(failed)"}],
                    "timestamp": 1,
                },
            }
        )
        return 0

    # mode == "ok": a minimal two-request run, mirroring the real runs' shape.
    event(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "provider": "google",
                "model": "gemini-3.8-flash",
                "stopReason": "toolUse",
                "content": [{"type": "text", "text": "fake tool turn"}],
                "timestamp": 1,
            },
        }
    )
    event(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "provider": "google",
                "model": "gemini-3.8-flash",
                "stopReason": "stop",
                "content": [{"type": "text", "text": "fake final answer"}],
                "timestamp": 2,
            },
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
