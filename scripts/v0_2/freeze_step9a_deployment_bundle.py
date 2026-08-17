#!/usr/bin/env python3
"""Freeze the confirmed Step-8 late_concat ensemble into a reusable deployment bundle."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
import run_step8_one_shot_confirmatory_evaluation as step8

SCHEMA = "triqto.v0_2.step9a_deployment_freeze.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step9a_deployment_freeze.json"
DEFAULT_STEP8_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step8_untouched_confirmatory.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step9a_deployment_bundle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--step8-config", type=Path, default=DEFAULT_STEP8_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--development-product-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def _require_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"{name} failed frozen reproduction: actual={actual:.17g} "
            f"expected={expected:.17g} tolerance={tolerance:.3g}"
        )


def validate_frozen_selection(*, selected: Mapping[str, Any], frozen: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    seed = int(selected["seed"])
    expected_epoch = int(frozen["selected_epochs"][str(seed)])
    if int(selected["selected_epoch"]) != expected_epoch:
        raise RuntimeError(f"seed {seed} selected epoch changed: {selected['selected_epoch']} != {expected_epoch}")
    expected_params = int(frozen["expected_trainable_parameter_count"])
    if int(selected["trainable_parameter_count"]) != expected_params:
        raise RuntimeError(f"seed {seed} parameter count changed: {selected['trainable_parameter_count']} != {expected_params}")
    tolerance = float(frozen["selection_metric_reproduction_tolerance"])
    summary = selected["selection_summary"]
    for metric in ("mechanism_balanced_accuracy", "effect_balanced_accuracy"):
        _require_close(f"seed {seed} {metric}", float(summary[metric]), float(reference[str(seed)][metric]), tolerance)


def checkpoint_payload(*, model: torch.nn.Module, seed: int, selected_epoch: int, config_sha256: str, step7_config_sha256: str) -> dict[str, Any]:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "schema": SCHEMA,
        "architecture": "late_concat",
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "config_sha256": config_sha256,
        "step7_config_sha256": step7_config_sha256,
        "state_dict": state,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = baseline.read_json(config_path)
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected Step-9A deployment-freeze schema")
    if config.get("status") != "FROZEN_AFTER_STEP8_CONFIRMATION_BEFORE_DEPLOYMENT_BUNDLE":
        raise RuntimeError("Step-9A deployment-freeze contract is not frozen")

    frozen = config["frozen_model"]
    if frozen["variant"] != "late_concat":
        raise RuntimeError("Step-9A may only serialize the confirmed late_concat architecture")
    boundaries = config["scientific_boundaries"]
    if any(bool(boundaries[key]) for key in ("new_training_data", "new_hyperparameter_selection", "new_threshold_selection", "architecture_change", "spent_confirmatory_reuse", "hardware_execution")):
        raise RuntimeError("Step-9A scientific boundaries were relaxed")

    step8_path = args.step8_config.expanduser().resolve()
    step7_path = args.step7_config.expanduser().resolve()
    step8_config = baseline.read_json(step8_path)
    experiment = baseline.read_json(step7_path)
    source = config["source_confirmation"]
    if baseline.sha256_file(step8_path) != str(source["step8_config_sha256"]):
        raise RuntimeError("Step-8 protocol config changed since confirmation")
    if baseline.sha256_file(step7_path) != str(source["step7_config_sha256"]):
        raise RuntimeError("Step-7 model config changed since confirmation")
    if step8_config.get("schema") != step8.SCHEMA:
        raise RuntimeError("unexpected Step-8 source schema")
    if experiment.get("schema") != step7.EXPERIMENT_SCHEMA:
        raise RuntimeError("unexpected Step-7 source schema")

    development = args.development_product_dir.expanduser().resolve() if args.development_product_dir else Path(config["development_source"]["default_product_dir"]).expanduser().resolve()
    dev_complete, dev_rows = baseline.verify_source_product(development, experiment)
    if dev_complete["product_id"] != config["development_source"]["product_id"]:
        raise RuntimeError("development product identity changed")
    fit_roots, selection_roots, _outer_roots, dev_by_root = step7.split_root_indices(dev_rows, experiment)
    if len(fit_roots) != int(config["development_source"]["fit_roots"]):
        raise RuntimeError("development fit-root count changed")
    if len(selection_roots) != int(config["development_source"]["selection_roots"]):
        raise RuntimeError("development selection-root count changed")

    device = step8.resolve_device(args.device)
    root_batch_size = int(step8_config["training"]["root_batch_size"])
    print("Materializing frozen Step-8 development fit/selection data...", flush=True)
    fit_blocks = step7.materialize_blocks(product=development, rows=dev_rows, by_root=dev_by_root, roots=fit_roots, root_batch_size=root_batch_size, label="step9a-development-fit", progress_every=args.progress_every)
    selection_blocks = step7.materialize_blocks(product=development, rows=dev_rows, by_root=dev_by_root, roots=selection_roots, root_batch_size=root_batch_size, label="step9a-development-selection", progress_every=args.progress_every)
    effect_class_weight, mechanism_class_weight = step7.class_weights(fit_blocks)

    seeds = [int(value) for value in frozen["seeds"]]
    expected_seeds = [int(value) for value in step8_config["training"]["seeds"]]
    if seeds != expected_seeds:
        raise RuntimeError("Step-9A seed list differs from frozen Step-8 training")

    selections: dict[str, Any] = {}
    models: dict[int, torch.nn.Module] = {}
    selection_predictions: dict[int, step7.PredictionSet] = {}
    for seed in seeds:
        print(f"\nReproducing frozen late_concat seed={seed}", flush=True)
        selected, _history, model, selection_prediction = step8.train_final_seed(seed=seed, experiment=experiment, fit_blocks=fit_blocks, selection_blocks=selection_blocks, effect_class_weight=effect_class_weight, mechanism_class_weight=mechanism_class_weight, device=device)
        validate_frozen_selection(selected=selected, frozen=frozen, reference=config["frozen_selection_reference"])
        selections[f"seed{seed}"] = selected
        models[seed] = model
        selection_predictions[seed] = selection_prediction

    reference_prediction = selection_predictions[seeds[0]]
    for seed in seeds[1:]:
        step7.assert_prediction_alignment(reference_prediction, selection_predictions[seed], f"Step-9A selection seed {seed}")
    mean_effect_logits = np.mean(np.stack([selection_predictions[seed].effect_logits for seed in seeds]), axis=0)
    reproduced_threshold, _, _ = step7._metrics_binary(reference_prediction.effect_truth, mean_effect_logits)
    _require_close("ensemble effect threshold", float(reproduced_threshold), float(frozen["effect_threshold"]), float(frozen["effect_threshold_reproduction_tolerance"]))
    selections["ensemble"] = {
        "aggregation": frozen["ensemble_aggregation"],
        "seeds": seeds,
        "reproduced_effect_threshold": float(reproduced_threshold),
        "deployment_effect_threshold": float(frozen["effect_threshold"]),
        "threshold_source": "frozen_step8_model_selection"
    }

    identity = {
        "schema": SCHEMA,
        "config_sha256": baseline.sha256_file(config_path),
        "step8_config_sha256": baseline.sha256_file(step8_path),
        "step7_config_sha256": baseline.sha256_file(step7_path),
        "development_product_id": dev_complete["product_id"],
        "source_evaluation_id": source["evaluation_id"],
        "architecture": "late_concat",
        "seeds": seeds,
        "selected_epochs": frozen["selected_epochs"],
        "effect_threshold": float(frozen["effect_threshold"])
    }
    bundle_id = "deploy_" + hashlib.sha256(baseline.canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / bundle_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing deployment bundle: {output}")
    staging = output_parent / f".{bundle_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    try:
        checkpoint_names: list[str] = []
        checkpoint_hashes: dict[str, str] = {}
        config_sha = baseline.sha256_file(config_path)
        step7_sha = baseline.sha256_file(step7_path)
        for seed in seeds:
            name = f"seed{seed}.pt"
            path = staging / name
            selected_epoch = int(frozen["selected_epochs"][str(seed)])
            torch.save(checkpoint_payload(model=models[seed], seed=seed, selected_epoch=selected_epoch, config_sha256=config_sha, step7_config_sha256=step7_sha), path)
            checkpoint_names.append(name)
            checkpoint_hashes[name] = baseline.sha256_file(path)

            reloaded = step7.instantiate_model("late_concat", seed, experiment, torch.device("cpu"))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            reloaded.load_state_dict(payload["state_dict"])
            reloaded.eval()
            original = {key: value.detach().cpu() for key, value in models[seed].state_dict().items()}
            for key, value in reloaded.state_dict().items():
                if not torch.equal(value.detach().cpu(), original[key]):
                    raise RuntimeError(f"checkpoint reload mismatch for seed {seed}: {key}")

        baseline.atomic_json(staging / "model_selection.json", selections)
        baseline.atomic_json(staging / "inference_contract.json", config["inference_contract"])
        files = checkpoint_names + ["model_selection.json", "inference_contract.json"]
        file_hashes = {name: baseline.sha256_file(staging / name) for name in files}
        if checkpoint_hashes != {name: file_hashes[name] for name in checkpoint_names}:
            raise RuntimeError("checkpoint hashes changed during deployment freeze")
        complete = {
            "schema": SCHEMA,
            "status": "DEPLOYMENT_BUNDLE_FROZEN",
            "bundle_id": bundle_id,
            "identity": identity,
            "source_confirmation": source,
            "development_product_id": dev_complete["product_id"],
            "architecture": "late_concat",
            "seeds": seeds,
            "selected_epochs": frozen["selected_epochs"],
            "effect_threshold": float(frozen["effect_threshold"]),
            "mechanism_classes": list(frozen["mechanism_classes"]),
            "trainable_parameter_count_each": int(frozen["expected_trainable_parameter_count"]),
            "file_hashes": file_hashes,
            "selection_reproduction_passed": True,
            "checkpoint_reload_exact_passed": True,
            "new_tuning_performed": False,
            "spent_confirmatory_cohort_accessed": False,
            "hardware_executed": False
        }
        baseline.atomic_json(staging / "bundle_complete.json", complete)
        os.replace(staging, output)
    except Exception:
        import shutil
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 9A DEPLOYMENT BUNDLE FROZEN\n")
    print(f"Bundle: {bundle_id}")
    print("Architecture: late_concat")
    print(f"Seeds: {seeds}")
    print(f"Selected epochs: {frozen['selected_epochs']}")
    print(f"Effect threshold: {float(frozen['effect_threshold']):.17g}")
    print("Selection reproduction: PASS")
    print("Checkpoint reload exactness: PASS")
    print("New tuning performed: NO")
    print("Spent confirmatory cohort accessed: NO")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
