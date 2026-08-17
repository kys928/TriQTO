#!/usr/bin/env python3
"""Run the non-scientific real-artifact smoke gate for Step 7."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import benchmark_step6_cheap_baselines as baseline
from triqto.step7.graph_adapter import batch_from_step5_examples
from triqto.step7.model import Step7DiagnosticModel


SMOKE_SCHEMA = "triqto.v0_2.step7_structured_diagnostic_smoke.v1"
DEFAULT_SMOKE_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_smoke.json"
DEFAULT_EXPERIMENT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step7_structured_diagnostic_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-config", type=Path, default=DEFAULT_SMOKE_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    return parser.parse_args()


def balanced_sample_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.to(torch.long), minlength=n_classes).to(torch.float32)
    if bool((counts <= 0).any()):
        raise RuntimeError(f"smoke batch is missing a required class: {counts.tolist()}")
    raw = counts.reciprocal().index_select(0, labels.to(torch.long))
    return raw / raw.mean()


def finite_gradient_norm(model: torch.nn.Module) -> float:
    squares = torch.zeros((), dtype=torch.float64)
    seen = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("non-finite Step 7 gradient")
        squares += parameter.grad.detach().to(dtype=torch.float64).square().sum().cpu()
        seen += parameter.grad.numel()
    if seen == 0:
        raise RuntimeError("Step 7 smoke backward produced no gradients")
    return float(torch.sqrt(squares))


def load_example(product: Path, row: dict[str, str]) -> dict[str, np.ndarray]:
    artifact = product / row["artifact_path"]
    if baseline.sha256_file(artifact) != row["artifact_sha256"]:
        raise RuntimeError(f"artifact hash mismatch: {row['example_id']}")
    with np.load(artifact, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def main() -> None:
    args = parse_args()
    smoke_path = args.smoke_config.expanduser().resolve()
    experiment_path = args.experiment_config.expanduser().resolve()
    smoke = baseline.read_json(smoke_path)
    experiment = baseline.read_json(experiment_path)
    if smoke.get("schema") != SMOKE_SCHEMA or smoke.get("status") != "FROZEN_BEFORE_STEP7_NEURAL_OUTCOME":
        raise RuntimeError("unexpected Step 7 smoke config schema/status")
    if experiment.get("schema") != "triqto.v0_2.step7_structured_diagnostic_model.v1" or experiment.get("status") != "FROZEN_BEFORE_STEP7_NEURAL_OUTCOME":
        raise RuntimeError("unexpected Step 7 experiment config schema/status")
    product = args.product_dir.expanduser().resolve() if args.product_dir else Path(experiment["source_dataset"]["default_product_dir"]).expanduser().resolve()
    source_complete, rows = baseline.verify_source_product(product, experiment)
    if source_complete["product_id"] != smoke["source_product_id"]:
        raise RuntimeError("Step 7 smoke source product mismatch")

    by_root: dict[int, list[dict[str, str]]] = defaultdict(list)
    occurrence_by_root: dict[int, int] = {}
    split_by_root: dict[int, str] = {}
    for row in rows:
        root = int(row["root_index"])
        by_root[root].append(row)
        occurrence_by_root[root] = int(row["family_occurrence_index"])
        split_by_root[root] = row["split"]
    eligible = [
        root
        for root in sorted(by_root)
        if split_by_root[root] == "train" and occurrence_by_root[root] % 5 in {1, 2, 3}
    ]
    selected_roots = eligible[: int(smoke["fit_roots"])]
    if len(selected_roots) != int(smoke["fit_roots"]):
        raise RuntimeError("insufficient Step 7 smoke fit roots")
    selected_rows: list[dict[str, str]] = []
    for root in selected_roots:
        local = sorted(by_root[root], key=lambda row: row["example_id"])
        if len(local) != 13:
            raise RuntimeError(f"Step 7 smoke root {root} does not contain exactly 13 examples")
        selected_rows.extend(local)
    if any(row["split"] != "train" or int(row["family_occurrence_index"]) % 5 not in {1, 2, 3} for row in selected_rows):
        raise RuntimeError("Step 7 smoke accidentally selected non-fit examples")

    examples = [load_example(product, row) for row in selected_rows]
    device = torch.device("cpu")
    model_batch, targets = batch_from_step5_examples(examples, device=device)
    model_settings = experiment["model"]
    results: list[dict[str, Any]] = []
    for variant in smoke["variants"]:
        torch.manual_seed(int(smoke["seed"]))
        model = Step7DiagnosticModel(
            variant=str(variant),
            hidden_dim=int(model_settings["hidden_dim"]),
            graph_message_passing_layers=int(model_settings["graph_message_passing_layers"]),
            residual_mlp_layers=int(model_settings["residual_mlp_layers"]),
            dropout=float(model_settings["dropout"]),
            layer_norm_eps=float(model_settings["layer_norm_eps"]),
            initialization_seed=int(smoke["seed"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(smoke["learning_rate"]),
            weight_decay=float(smoke["weight_decay"]),
        )
        model.train()
        output = model(model_batch)
        effect_weights = balanced_sample_weights(targets.effect_present.to(torch.long), 2)
        effect_loss = (
            F.binary_cross_entropy_with_logits(
                output.effect_logit, targets.effect_present, reduction="none"
            )
            * effect_weights
        ).mean()
        mechanism_mask = targets.mechanism_loss_mask
        if not bool(mechanism_mask.any()):
            raise RuntimeError("Step 7 smoke batch contains no mechanism-supervised rows")
        supervised_mechanism = targets.mechanism[mechanism_mask]
        mechanism_weights = balanced_sample_weights(supervised_mechanism, 3)
        mechanism_loss = (
            F.cross_entropy(
                output.mechanism_logits[mechanism_mask],
                supervised_mechanism,
                reduction="none",
            )
            * mechanism_weights
        ).mean()
        total_loss = (
            float(experiment["training"]["effect_loss_weight"]) * effect_loss
            + float(experiment["training"]["mechanism_loss_weight"]) * mechanism_loss
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite Step 7 smoke loss for {variant}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = finite_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(smoke["gradient_clip_norm"]))
        optimizer.step()
        with torch.no_grad():
            post = model(model_batch)
        if not torch.isfinite(post.effect_logit).all() or not torch.isfinite(post.mechanism_logits).all():
            raise RuntimeError(f"non-finite post-step outputs for {variant}")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        results.append(
            {
                "variant": variant,
                "parameter_count": int(parameter_count),
                "trainable_parameter_count": int(trainable_count),
                "effect_loss": float(effect_loss.detach()),
                "mechanism_loss": float(mechanism_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "gradient_norm": gradient_norm,
                "effect_probability_min": float(post.effect_probability.min()),
                "effect_probability_max": float(post.effect_probability.max()),
                "uncertainty_min": float(post.effect_uncertainty.min()),
                "uncertainty_max": float(post.effect_uncertainty.max()),
            }
        )

    identity = {
        "schema": SMOKE_SCHEMA,
        "smoke_config_sha256": baseline.sha256_file(smoke_path),
        "experiment_config_sha256": baseline.sha256_file(experiment_path),
        "runner_sha256": baseline.sha256_file(Path(__file__).resolve()),
        "source_product_id": source_complete["product_id"],
        "selected_roots": selected_roots,
    }
    smoke_id = "smoke_" + hashlib.sha256(
        baseline.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / smoke_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing Step 7 smoke result {output}")
    staging = output_parent / f".{smoke_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    payload = {
        "schema": SMOKE_SCHEMA,
        "status": "COMPLETE",
        "decision": smoke["decision_on_success"],
        "smoke_id": smoke_id,
        "identity": identity,
        "fit_roots_accessed": len(selected_roots),
        "fit_examples_accessed": len(selected_rows),
        "selection_roots_accessed": 0,
        "outer_validation_roots_accessed": 0,
        "historical_v0_1_test_accessed": False,
        "spent_confirmatory_cohort_accessed": False,
        "scientific_metric_claim": False,
        "results": results,
    }
    baseline.atomic_json(staging / "smoke_complete.json", payload)
    os.replace(staging, output)
    print("\nTRIQTO STEP 7 STRUCTURED DIAGNOSTIC SMOKE COMPLETE\n")
    print(f"Decision: {smoke['decision_on_success']}")
    print(f"Fit roots/examples accessed: {len(selected_roots)}/{len(selected_rows)}")
    print("Selection roots accessed: 0")
    print("Outer validation roots accessed: 0")
    for row in results:
        print(
            f"{row['variant']}: params={row['parameter_count']} "
            f"loss={row['total_loss']:.4f} grad_norm={row['gradient_norm']:.4f}"
        )
    print("Scientific metric claim: NO")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
