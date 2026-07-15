"""Pod/environment preflight checks for Phase 15.6."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

from .config import PodRequirements


def _system_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return float(pages * page_size / (1024**3))


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_phase156_environment(
    *,
    workspace: str | Path,
    requirements: PodRequirements,
    training_device: str,
) -> dict[str, Any]:
    """Inspect the current process without changing it."""
    if not isinstance(requirements, PodRequirements):
        raise TypeError("requirements must be PodRequirements")
    target = Path(workspace).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    cpu_count = os.cpu_count() or 0
    memory_gb = _system_memory_gb()
    disk = shutil.disk_usage(target)
    free_disk_gb = disk.free / (1024**3)

    torch_version = _package_version("torch")
    cuda_available = False
    cuda_device_name: str | None = None
    cuda_vram_gb: float | None = None
    cuda_device_count = 0
    cuda_runtime: str | None = None
    torch_error: str | None = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_runtime = torch.version.cuda
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            cuda_device_name = str(props.name)
            cuda_vram_gb = float(props.total_memory / (1024**3))
    except Exception as exc:  # pragma: no cover - environment-dependent
        torch_error = f"{type(exc).__name__}: {exc}"

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "observed": observed, "required": required})

    add("python_3_11", sys.version_info[:2] == (3, 11), platform.python_version(), "3.11.x")
    add("cpu_cores", cpu_count >= requirements.minimum_cpu_cores, cpu_count, requirements.minimum_cpu_cores)
    add(
        "system_memory_gb",
        memory_gb is not None and memory_gb >= requirements.minimum_memory_gb,
        None if memory_gb is None else round(memory_gb, 2),
        requirements.minimum_memory_gb,
    )
    add(
        "free_disk_gb",
        free_disk_gb >= requirements.minimum_free_disk_gb,
        round(free_disk_gb, 2),
        requirements.minimum_free_disk_gb,
    )
    add("qiskit_installed", _package_version("qiskit") is not None, _package_version("qiskit"), "installed")
    add(
        "qiskit_aer_installed",
        _package_version("qiskit-aer") is not None,
        _package_version("qiskit-aer"),
        "installed",
    )
    add("torch_installed", torch_version is not None and torch_error is None, torch_version, "installed")

    needs_cuda = requirements.require_cuda_for_training or training_device == "cuda"
    if needs_cuda:
        add("cuda_available", cuda_available, cuda_available, True)
        add(
            "gpu_vram_gb",
            cuda_vram_gb is not None and cuda_vram_gb >= requirements.minimum_gpu_vram_gb,
            None if cuda_vram_gb is None else round(cuda_vram_gb, 2),
            requirements.minimum_gpu_vram_gb,
        )

    ready = all(item["passed"] for item in checks)
    return {
        "schema": "triqto.phase15_6.environment.v1",
        "ready": ready,
        "workspace": str(target),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": cpu_count,
        "system_memory_gb": None if memory_gb is None else round(memory_gb, 3),
        "free_disk_gb": round(free_disk_gb, 3),
        "packages": {
            name: _package_version(name)
            for name in (
                "qiskit",
                "qiskit-aer",
                "qiskit-ibm-runtime",
                "numpy",
                "scipy",
                "torch",
                "pyarrow",
                "ripser",
                "gudhi",
            )
        },
        "cuda": {
            "available": cuda_available,
            "runtime": cuda_runtime,
            "device_count": cuda_device_count,
            "device_name": cuda_device_name,
            "device_vram_gb": None if cuda_vram_gb is None else round(cuda_vram_gb, 3),
            "inspection_error": torch_error,
        },
        "checks": checks,
        "notes": [
            "Phase 7/Qiskit Aer generation is CPU-first unless a separately validated Aer GPU profile is selected.",
            "CUDA is primarily used by the Phase 14 PyTorch training stage.",
            "A passing preflight is necessary but does not predict wall-clock runtime or cloud cost.",
        ],
    }


__all__ = ["inspect_phase156_environment"]
