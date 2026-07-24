#!/usr/bin/env python3
"""Audit weighted objective gradients on representative real model-ready batches."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from triqto.model import TriQTOModel, model_config_from_dict
from triqto.training import load_training_checkpoint, training_config_from_dict
from triqto.training.model_ready.full_trainer import (
    _load_batch,
    _move_tree,
    _resolve_device,
    _seed_everything,
)
from triqto.training.model_ready.gradient_audit import (
    audit_loss_component_gradients,
    weighted_objective_components,
)
from triqto.training.model_ready.multitask_losses import (
    compute_model_ready_multitask_losses,
)
from triqto.training.model_ready.source import load_model_ready_dataset

_TASKS = (
    "diagnosis",
    "action_ranking",
    "born_prediction",
    "joint_multitask",
    "hardware_masked",
)
_ACTION_TASKS = {"action_ranking", "joint_multitask", "hardware_masked"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _representative_records(
    *,
    dataset,
    selection: dict[str, Any],
    task: str,
    split: str,
    batch_size: int,
) -> tuple[dict[str, Any], ...]:
    key = f"{task}:{split}"
    raw_ids = selection.get("view_item_ids", {}).get(key)
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"selection manifest misses {key}")
    rows = dataset.records_by_task_split.get((task, split), ())
    by_id = {str(row["view_item_id"]): dict(row) for row in rows}
    ordered: list[dict[str, Any]] = []
    for item_id in raw_ids:
        row = by_id.get(str(item_id))
        if row is None:
            raise ValueError(f"selection references unknown item {item_id}")
        ordered.append(row)

    selected: list[dict[str, Any]] = []
    if task in _ACTION_TASKS:
        positive = next((row for row in ordered if bool(row.get("should_act"))), None)
        if positive is not None:
            selected.append(positive)
    for row in ordered:
        if len(selected) >= batch_size:
            break
        if selected and row["view_item_id"] == selected[0]["view_item_id"]:
            continue
        selected.append(row)
    if not selected:
        raise ValueError(f"no records selected for {task}/{split}")
    return tuple(selected)


def _audit_state(
    *,
    state_name: str,
    model: TriQTOModel,
    dataset,
    selection: dict[str, Any],
    training_config,
    model_config,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    state_report: dict[str, Any] = {}
    model.train(True)
    for task_index, task in enumerate(_TASKS):
        _seed_everything(
            training_config.seed + task_index * 10_007,
            training_config.deterministic_algorithms,
        )
        records = _representative_records(
            dataset=dataset,
            selection=selection,
            task=task,
            split=split,
            batch_size=batch_size,
        )
        batch = _load_batch(dataset, records, model_config, training_config)
        batch = _move_tree(batch, device)
        model.zero_grad(set_to_none=True)
        output = model(batch.model_batch)
        losses = compute_model_ready_multitask_losses(
            output, batch, training_config.loss
        )
        components = weighted_objective_components(losses, training_config.loss)
        gradients = audit_loss_component_gradients(
            model,
            components,
            clip_threshold=training_config.max_gradient_norm,
        )
        action = batch.targets.action
        state_report[task] = {
            "state": state_name,
            "split": split,
            "item_ids": list(batch.item_ids),
            "graph_count": batch.graph_count,
            "should_act_positive_count": int(
                (action.should_act[action.should_act_mask] > 0.5).sum().item()
            ),
            "ranking_active_graph_count": int(action.ranking_loss_mask.sum().item()),
            "candidate_count": int(action.candidate_reward.numel()),
            "weighted_loss_components": {
                name: float(value.detach().cpu())
                for name, value in components.items()
            },
            "gradient_components": gradients,
        }
    return state_report


def _compact_table(report: dict[str, Any]) -> None:
    print("\nPer-component gradient norms (weighted objective terms)")
    print("state task component loss grad_norm clip_ratio largest_module")
    rows: list[tuple[float, str]] = []
    for state, tasks in report["states"].items():
        for task, payload in tasks.items():
            for component, values in payload["gradient_components"].items():
                line = (
                    f"{state} {task} {component} "
                    f"{values['loss']:.8g} {values['gradient_norm']:.8g} "
                    f"{values['gradient_norm_to_clip_ratio']:.8g} "
                    f"{values['largest_module']}"
                )
                rows.append((float(values["gradient_norm"]), line))
    for _norm, line in sorted(rows, key=lambda item: item[0], reverse=True):
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-ready-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--state", choices=("initial", "final", "both"), default="both"
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    run_root = Path(args.run_root).expanduser().resolve()
    summary = _read_json(run_root / "reports" / "summary.json")
    if summary.get("complete") is not True:
        raise ValueError("gradient audit requires a complete training run")
    selection = _read_json(run_root / "manifests" / "selection.json")
    model_config = model_config_from_dict(
        _read_json(run_root / "manifests" / "model_config.json")
    )
    training_config = training_config_from_dict(
        _read_json(run_root / "manifests" / "training_config.json")
    )
    if training_config.loss.uncertainty_weighting:
        raise ValueError(
            "exact component audit currently requires uncertainty_weighting=false"
        )
    dataset = load_model_ready_dataset(args.model_ready_root)
    if dataset.manifest_sha256 != summary.get("source_manifest_sha256"):
        raise ValueError("audit dataset manifest differs from training source")
    device = _resolve_device(args.device)

    states: dict[str, Any] = {}
    requested = ("initial", "final") if args.state == "both" else (args.state,)
    for state_index, state_name in enumerate(requested):
        _seed_everything(
            training_config.seed + state_index * 1_000_003,
            training_config.deterministic_algorithms,
        )
        model = TriQTOModel(model_config).to(device)
        checkpoint_id = None
        if state_name == "final":
            final = summary["final_checkpoint"]
            checkpoint_path = run_root / str(final["artifact_ref"])
            metadata = load_training_checkpoint(
                checkpoint_path,
                model=model,
                expected_training_run_id=str(summary["run_id"]),
            )
            checkpoint_id = str(metadata["checkpoint_id"])
        states[state_name] = _audit_state(
            state_name=state_name,
            model=model,
            dataset=dataset,
            selection=selection,
            training_config=training_config,
            model_config=model_config,
            split=args.split,
            batch_size=args.batch_size,
            device=device,
        )
        states[state_name]["_metadata"] = {
            "model_architecture_id": model.architecture_id,
            "checkpoint_id": checkpoint_id,
        }

    report = {
        "schema": "triqto.training.model_ready_gradient_audit.v1",
        "run_id": summary["run_id"],
        "source_manifest_sha256": dataset.manifest_sha256,
        "split": args.split,
        "batch_size": args.batch_size,
        "device": str(device),
        "clip_threshold": training_config.max_gradient_norm,
        "states": states,
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Wrote gradient audit: {output}")
    _compact_table(report)
    print("\nJSON report")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
