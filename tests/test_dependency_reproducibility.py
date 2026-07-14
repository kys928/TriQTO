from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_default_dependencies_are_pinned_and_cpu_safe() -> None:
    result = subprocess.run([sys.executable, "scripts/verify_dependency_pins.py"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
