"""测试引导：把 agentic/schema 加入 sys.path，保证与 cwd 无关。"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # agentic/schema
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
