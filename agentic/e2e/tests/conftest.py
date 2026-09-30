"""tests/conftest.py — agentic/e2e 离线测试共用设置（issue #33）。

全部测试：**离线**（无网络、无 API、无 GPU）；真实运行证据在 reports/ 与本地 run 产物里，
两者严格分开。mock/fixture 产物一律标 `kind="mock"`，绝不冒充真实音乐结果。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

E2E_DIR = Path(__file__).resolve().parents[1]
HARNESS_DIR = E2E_DIR / "harness"
FIXTURES_DIR = E2E_DIR / "fixtures"
for _p in (str(HARNESS_DIR), str(FIXTURES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import make_mock_fixture  # noqa: E402


@pytest.fixture(scope="session")
def fixture_audio(tmp_path_factory):
    """自生成 mock 复音 fixture（wav + 真值），供离线 DSP/抽查测试使用。"""
    out = tmp_path_factory.mktemp("e2e-fixture")
    gt = make_mock_fixture.write_fixture(out)
    return {"dir": out, "wav": out / gt["audio"]["filename"], "gt": gt}
