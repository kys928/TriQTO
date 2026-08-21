#!/usr/bin/env python3
"""Step 10C crash-safe longer-horizon dual-initialization benchmark.

Scientific change versus Step 10B: max training ceiling 20 -> 40 only.
Engineering changes: atomic best/resume checkpoints, exact optimizer/RNG resume,
progress heartbeat, wall-time and gradient-clipping telemetry.
Final evaluation uses only the frozen fresh Step-10C outer cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
import run_step10_warmstart_vs_scratch as step10b

SCHEMA = "triqto.v0_2.step10c_crashsafe_long_horizon.v1"
OUTER_SCHEMA = "triqto.v0_2.step10c_fresh_outer_cohort.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10c_crashsafe_long_horizon.json"
DEFAULT_MIXTURE_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step10c_crashsafe_long_horizon")
DEFAULT_OUTER_PARENT = Path("/workspace/triqto-data/step10c_fresh_outer_cohort")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mixture-config", type=Path, default=DEFAULT_MIXTURE_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--mixture-product-dir", type=Path)
    parser.add_argument("--fresh-outer-product-dir", type=Path)
    parser.add_argument("--step9a-bundle-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--no-resume", action="store_true", help="Refuse to reuse an existing exact-identity resume checkpoint.")
    return parser.parse_args()


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json_fsync(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with temp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


def _resolve_outer_product(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    pointer = DEFAULT_OUTER_PARENT / "current_product.json"
    if not pointer.is_file():
        raise RuntimeError("fresh Step-10C outer current_product.json missing; generate/freeze outer cohort first")
    payload = baseline.read_json(pointer)
    if payload.get("schema") != "triqto.v0_2.step10c_fresh_outer_current_product.v1":
        raise RuntimeError("unexpected Step-10C outer pointer schema")
    return Path(payload["product_dir"]).expanduser().resolve()


def _verify_outer_product(product: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str], dict[int, int]]:
    complete = baseline.read_json(product / "dataset_complete.json")
    if complete.get("schema") != OUTER_SCHEMA or complete.get("status") != "COMPLETE_FROZEN_OUTER_VALIDATION":
        raise RuntimeError("fresh Step-10C outer product incomplete or wrong schema")
    if not bool(complete.get("outer_validation_only")) or bool(complete.get("step10b_outer_reused", True)):
        raise RuntimeError("fresh Step-10C outer cohort scientific boundary failed")
    for name, expected in complete.get("manifest_hashes", {}).items():
        path = product / "manifests" / name
        if baseline.sha256_file(path) != expected:
            raise RuntimeError(f"fresh outer manifest hash mismatch: {name}")
    if baseline.sha256_file(product / "eda.json") != complete.get("eda_sha256"):
        raise RuntimeError("fresh outer EDA hash mismatch")
    eda = baseline.read_json(product / "eda.json")
    if eda.get("status") != "PASS" or int(eda["freshness"]["overlap_with_step10_original_graphs"]) != 0 or int(eda["freshness"]["overlap_with_step10_bridge_graphs"]) != 0:
        raise RuntimeError("fresh outer EDA freshness gate failed")
    original_rows = baseline.read_csv(product / "manifests" / "original_example_manifest.csv")
    bridge_rows = baseline.read_csv(product / "manifests" / "bridge_example_manifest.csv")
    if len(original_rows) != int(complete["original_example_count"]):
        raise RuntimeError("fresh original outer manifest count mismatch")
    if len(bridge_rows) != int(complete["bridge_example_count"]):
        raise RuntimeError("fresh bridge outer manifest count mismatch")
    parent_by_root: dict[int, int] = {}
    for row in bridge_rows:
        root = int(row["root_index"])
        parent = int(row["parent_group_index"])
        if root in parent_by_root and parent_by_root[root] != parent:
            raise RuntimeError("fresh bridge root maps to multiple parents")
        parent_by_root[root] = parent
    return complete, original_rows, bridge_rows, parent_by_root


def _rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if torch.cuda.is_available() and snapshot.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}



def _runtime_fingerprint(device: torch.device) -> dict[str, Any]:
    cuda_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_type": str(device),
        "torch_num_threads": int(torch.get_num_threads()),
        "cuda_version_if_any": torch.version.cuda,
        "cuda_device_name_if_any": cuda_name,
    }

def _identity_equal(