#!/usr/bin/env python3
"""Validate the Phase 12 model-ready loader, forward pass, and action losses."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from triqto.model import TriQTOModel, load_model_config
from triqto.model.constants import HEAD_ORDER, STREAM_ORDER
from triqto.training import (
    CANONICAL_TOPOLOGY_INPUT_DIM,
    build_model_ready_example,
    compute_model_ready_action_losses,
    load_model_ready_artifact,
    load_model_ready_dataset,
    select_model_ready_record,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0")


def main() -> int:
    root_raw = os.environ.get("TRIQTO_MODEL_READY_ROOT")
    if not root_raw:
        raise ValueError("TRIQTO_MODEL_READY_ROOT is required")
    model_path = Path(
        os.environ.get(
            "TRIQTO_MODEL_CONFIG",
            REPO_ROOT / "configs" / "model" / "phase15_6_base.json",
        )
    ).expanduser().resolve()
    verify_files = _env_bool("TRIQTO_MODEL_READY_VERIFY_FILES", True)

    dataset = load_model_ready_dataset(
        root_raw,
        verify_artifact_files=verify_files,
    )
    config = load_model_config(model_path)
    if config.topology_input_dim != CANONICAL_TOPOLOGY_INPUT_DIM:
        raise ValueError(
            f"model topology_input_dim={config.topology_input_dim}; expected "
            f"{CANONICAL_TOPOLOGY_INPUT_DIM}"
        )

    action_record = select_model_ready_record(
        dataset, task="action_ranking", split="train"
    )
    action_artifact = load_model_ready_artifact(dataset, action_record)
    action_example = build_model_ready_example(action_artifact, config)

    torch.manual_seed(config.initialization_seed)
    model = TriQTOModel(config)
    model.train()
    action_output = model(action_example.model_batch)
    losses = compute_model_ready_action_losses(
        action_output,
        action_example.action_targets,
    )
    losses["total"].backward()
    gate_parameters = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "action_ranking_head.should_act" in name
    ]
    if not gate_parameters or not any(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gate_parameters
    ):
        raise RuntimeError("should-act gate did not receive a finite gradient")

    topology_record = select_model_ready_record(
        dataset,
        task="joint_multitask",
        split="train",
        topology_required=True,
    )
    topology_artifact = load_model_ready_artifact(dataset, topology_record)
    topology_example = build_model_ready_example(topology_artifact, config)
    model.eval()
    with torch.no_grad():
        topology_output = model(topology_example.model_batch)

    topology_stream = STREAM_ORDER.index("topology")
    action_head = HEAD_ORDER.index("action_ranking")
    born_head = HEAD_ORDER.index("born_prediction")
    diagnosis_head = HEAD_ORDER.index("diagnosis")
    if bool(
        topology_output.effective_head_stream_mask[
            0, action_head, topology_stream
        ]
    ):
        raise RuntimeError("action-ranking head can observe topology")
    if bool(
        topology_output.effective_head_stream_mask[
            0, born_head, topology_stream
        ]
    ):
        raise RuntimeError("Born-prediction head can observe topology")
    if not bool(
        topology_output.effective_head_stream_mask[
            0, diagnosis_head, topology_stream
        ]
    ):
        raise RuntimeError("joint diagnosis head did not receive topology")

    payload = {
        "status": "validated",
        "model_ready_root": str(dataset.root),
        "training_view_dataset_id": dataset.training_view_dataset_id,
        "manifest_rows": len(dataset.records),
        "model_config": str(model_path),
        "topology_input_dim": config.topology_input_dim,
        "action_example": action_example.view_item_id,
        "action_should_act_target": bool(
            action_example.action_targets.should_act[0].item()
        ),
        "action_ranking_loss_mask": bool(
            action_example.action_targets.ranking_loss_mask[0].item()
        ),
        "losses": {
            name: float(value.detach().cpu())
            for name, value in losses.items()
        },
        "topology_example": topology_example.view_item_id,
        "topology_parameter_ablation_dim": int(
            topology_example.topology_ablation_inputs[
                "x_topology_parameter_features"
            ].size
        ),
        "topology_born_ablation_dim": int(
            topology_example.topology_ablation_inputs[
                "x_topology_born_features"
            ].size
        ),
        "action_head_topology_enabled": False,
        "born_prediction_head_topology_enabled": False,
        "diagnosis_head_topology_enabled": True,
        "lambda_top": 0.0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
