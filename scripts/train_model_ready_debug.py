#!/usr/bin/env python3
"""Run a small model-ready training experiment using environment variables."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

from triqto.model import load_model_config
from triqto.training import load_training_config
from triqto.training.model_ready import run_model_ready_debug_training


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_ready_root = Path(_required("TRIQTO_MODEL_READY_ROOT"))
    output_root = Path(_required("TRIQTO_MODEL_READY_DEBUG_OUTPUT_ROOT"))
    model_config_path = Path(
        os.environ.get(
            "TRIQTO_MODEL_CONFIG",
            str(repo_root / "configs" / "model" / "phase15_6_base.json"),
        )
    )
    training_config_path = Path(
        os.environ.get(
            "TRIQTO_TRAINING_CONFIG",
            str(
                repo_root
                / "configs"
                / "train"
                / "phase15_6_model_ready_debug.yaml"
            ),
        )
    )
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)
    device_override = os.environ.get("TRIQTO_DEBUG_DEVICE", "").strip()
    if device_override:
        training_config = replace(training_config, device=device_override)
    task = os.environ.get("TRIQTO_DEBUG_TASK", "action_ranking").strip()
    result = run_model_ready_debug_training(
        model_ready_root=model_ready_root,
        output_root=output_root,
        model_config=model_config,
        training_config=training_config,
        task=task,
        train_items=_positive_int("TRIQTO_DEBUG_TRAIN_ITEMS", 16),
        validation_items=_positive_int("TRIQTO_DEBUG_VALIDATION_ITEMS", 8),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
