#!/usr/bin/env python3
"""Run vectorized model-ready full multi-task training from environment variables."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

from triqto.model import load_model_config
from triqto.training import load_training_config
from triqto.training.model_ready import run_model_ready_full_training


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is not set")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)).strip())
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = Path(_required("TRIQTO_MODEL_READY_ROOT"))
    output = Path(_required("TRIQTO_MODEL_READY_FULL_OUTPUT_ROOT"))
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
                / "phase15_6_model_ready_multitask_smoke.yaml"
            ),
        )
    )
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)
    device = os.environ.get("TRIQTO_FULL_DEVICE", "").strip()
    if device:
        training_config = replace(training_config, device=device)
    resume = os.environ.get("TRIQTO_FULL_RESUME_CHECKPOINT", "").strip()
    result = run_model_ready_full_training(
        model_ready_root=source,
        output_root=output,
        model_config=model_config,
        training_config=training_config,
        train_limit_per_task=_nonnegative_int(
            "TRIQTO_FULL_TRAIN_LIMIT_PER_TASK", 32
        ),
        validation_limit_per_task=_nonnegative_int(
            "TRIQTO_FULL_VALIDATION_LIMIT_PER_TASK", 16
        ),
        resume_checkpoint=Path(resume) if resume else None,
        progress_every_batches=_nonnegative_int(
            "TRIQTO_FULL_PROGRESS_EVERY_BATCHES", 10
        ),
    )
    summary = dict(result.summary)
    summary.pop("epoch_metrics", None)
    print(
        json.dumps(
            {"status": result.status, "output_root": str(result.output_root), **summary},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
