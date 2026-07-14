#!/usr/bin/env python3
"""Reject unpinned production dependencies and CUDA/GPU packages in the default profile."""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [ROOT / "requirements-cpu.txt", ROOT / "constraints" / "cpu.txt"]
PINNED = re.compile(r"^[A-Za-z0-9_.-]+(\[[^\]]+\])?==[^\s#]+$")
FORBIDDEN_CPU = {"qiskit-aer-gpu"}


def logical_lines(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def main() -> int:
    errors: list[str] = []
    for path in DEFAULT_FILES:
        for line in logical_lines(path):
            if line.startswith("-r ") or line.startswith("--extra-index-url "):
                continue
            name = re.split(r"==|<=|>=|~=|!=|<|>", line, maxsplit=1)[0].split("[")[0].lower()
            if name in FORBIDDEN_CPU :
                errors.append(f"{path.relative_to(ROOT)} contains CPU-forbidden package {line!r}")
            if not PINNED.match(line):
                errors.append(f"{path.relative_to(ROOT)} contains unpinned dependency {line!r}")
    gpu_text = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    if "qiskit-aer-gpu==" not in gpu_text:
        errors.append("requirements-gpu.txt must pin qiskit-aer-gpu explicitly")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
