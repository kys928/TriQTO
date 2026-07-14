from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_default_dependencies_are_pinned_and_cpu_safe() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_dependency_pins.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_gpu_profile_does_not_mix_cpu_and_gpu_packages() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "-r requirements-cpu.txt" not in text
    assert "qiskit-aer-gpu==" in text
    assert "\nqiskit-aer==" not in text
    assert "+cpu" not in text


def test_cpu_and_gpu_constraints_are_independent() -> None:
    root = Path(__file__).resolve().parents[1]
    gpu = (root / "constraints" / "gpu.txt").read_text(encoding="utf-8")
    assert "-r cpu.txt" not in gpu
    assert "qiskit-aer-gpu==" in gpu
    assert "\nqiskit-aer==" not in gpu
