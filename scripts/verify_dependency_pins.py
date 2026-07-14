#!/usr/bin/env python3
"""Validate TriQTO's direct CPU/GPU dependency profiles.

The repository intentionally calls these files *direct-dependency pins*, not a
complete transitive lock. This verifier prevents CPU/GPU package collisions and
can also confirm that the installed CPU environment matches the declared pins.
"""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PINNED = re.compile(r"^([A-Za-z0-9_.-]+)(\[[^\]]+\])?==([^\s#]+)$")


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def logical_lines(path: Path) -> list[str]:
    return [
        line
        for raw in path.read_text(encoding="utf-8").splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    ]


def parse_direct_pins(path: Path) -> tuple[dict[str, str], list[str]]:
    pins: dict[str, str] = {}
    errors: list[str] = []
    for line in logical_lines(path):
        if line.startswith(("-r ", "--extra-index-url ")):
            continue
        match = PINNED.fullmatch(line)
        if match is None:
            errors.append(
                f"{path.relative_to(ROOT)} contains unpinned dependency {line!r}"
            )
            continue
        name = canonical_name(match.group(1))
        if name in pins:
            errors.append(
                f"{path.relative_to(ROOT)} contains duplicate dependency {name!r}"
            )
        pins[name] = match.group(3)
    return pins, errors


def validate_profiles() -> tuple[dict[str, str], list[str]]:
    errors: list[str] = []
    cpu_requirements = ROOT / "requirements-cpu.txt"
    cpu_constraints = ROOT / "constraints" / "cpu.txt"
    gpu_requirements = ROOT / "requirements-gpu.txt"
    gpu_constraints = ROOT / "constraints" / "gpu.txt"

    cpu, profile_errors = parse_direct_pins(cpu_requirements)
    errors.extend(profile_errors)
    constrained_cpu, profile_errors = parse_direct_pins(cpu_constraints)
    errors.extend(profile_errors)
    gpu, profile_errors = parse_direct_pins(gpu_requirements)
    errors.extend(profile_errors)
    constrained_gpu, profile_errors = parse_direct_pins(gpu_constraints)
    errors.extend(profile_errors)

    if cpu != constrained_cpu:
        errors.append("requirements-cpu.txt and constraints/cpu.txt pins differ")
    if gpu != constrained_gpu:
        errors.append("requirements-gpu.txt and constraints/gpu.txt pins differ")

    root_lines = logical_lines(ROOT / "requirements.txt")
    if root_lines != ["-r requirements-cpu.txt"]:
        errors.append("requirements.txt must include only requirements-cpu.txt")

    gpu_lines = logical_lines(gpu_requirements)
    if any(line == "-r requirements-cpu.txt" for line in gpu_lines):
        errors.append("requirements-gpu.txt must not inherit the CPU profile")

    if "qiskit-aer-gpu" in cpu:
        errors.append("CPU profile must not contain qiskit-aer-gpu")
    if "qiskit-aer" in gpu:
        errors.append("GPU profile must not contain qiskit-aer")
    if "qiskit-aer-gpu" not in gpu:
        errors.append("GPU profile must pin qiskit-aer-gpu")

    cpu_torch = cpu.get("torch", "")
    gpu_torch = gpu.get("torch", "")
    if not cpu_torch.endswith("+cpu"):
        errors.append("CPU profile must use an explicit +cpu Torch build")
    if not gpu_torch or gpu_torch.endswith("+cpu"):
        errors.append("GPU profile must not use the CPU Torch build")

    return cpu, errors


def installed_version_errors(expected: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for name, wanted in sorted(expected.items()):
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"required CPU package {name} is not installed")
            continue
        if actual != wanted:
            errors.append(
                f"installed {name} version {actual!r} does not match {wanted!r}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-installed",
        action="store_true",
        help="also verify installed packages against the CPU direct pins",
    )
    args = parser.parse_args(argv)

    cpu, errors = validate_profiles()
    if args.check_installed:
        errors.extend(installed_version_errors(cpu))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
