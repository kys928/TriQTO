from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_smoke_test_pipeline_script_runs_ideal_path_without_aer() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root / "src") if not existing else os.pathsep.join((str(root / "src"), existing))
    source = """
import runpy
import sys
assert "qiskit_aer" not in sys.modules
runpy.run_path("scripts/smoke_test_pipeline.py", run_name="__main__")
assert "qiskit_aer" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TriQTO Phase 4 smoke test complete." in result.stdout
    assert "Statevector mode: ideal_statevector" in result.stdout
    assert "Shot mode: ideal_shot" in result.stdout
