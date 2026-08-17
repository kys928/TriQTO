#!/usr/bin/env python3
"""Freeze one post-confirmation fixed-epoch late_concat deployment ensemble.

Step 8 confirmed the architecture/training procedure, but its exact CUDA checkpoint
weights were not serialized and a later replay demonstrated that the seeded CUDA
training trajectory is not exactly reproducible. Step 9A therefore does not claim
to reconstruct those weights. It trains one deployment refit on development data
only, using the already-archived Step-8 seed/epoch schedule, then freezes the
resulting checkpoint hashes as the authoritative deployment weight identity.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
import run_step8_one_shot_confirmatory_evaluation as step8

SCHEMA = "triqto.v0_2.step9a_deployment_freeze.v2"
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


def _finite_metric_summary(summary: Mapping[str, Any]) -> None:
    for name in ("mechanism_balanced_accuracy", "effect_balanced_accuracy"):
        value = float(summary[name])
        if not np.isfinite(value):
            raise RuntimeError(f"non-finite deployment-refit selection metric: {name}={value}")


def validate_refit_record(*, record: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    seed = int(record["seed"])
    expected_epoch = int(frozen["fixed_training_epochs"][str(seed)])
    if int(record["fixed_epoch"]) != expected_epoch:
        raise RuntimeError(
            f"seed {seed} fixed epoch changed: {record['fixed_epoch']} != {expected_epoch}"
        )
    expected_params = int(frozen["expected_trainable_parameter_count"])
    if int(record["trainable_parameter_count"]) != expected_params:
        raise RuntimeError(
            f"seed {seed} parameter count changed: "
            f"{record['trainable_parameter_count']} != {expected_params}"
        )
    if bool(record.get("selected_by_current_refit", True)):
        raise RuntimeError("Step-9A current refit is not allowed to select an epoch")
    _finite_metric_summary(record["selection_summary"])


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    seed: int,
    fixed_epoch: int,
    config_sha256: str,
    step7_config_sha256: str,
) -> dict[str, Any]:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "schema": SCHEMA,
        "architecture": "late_concat",
        "seed": int(seed),
        "fixed_epoch": int(fixed_epoch),
        "epoch_source": "archived_step8_selected_epoch",
        "weight_provenance": "post_confirmation_fixed_epoch_refit_development_only",
        "exact_step8_checkpoint_weight": False,
        "config_sha256": config_sha256,
        "step7_config_sha256": step7_config_sha256,
        "state_dict": state,
    }


def train_fixed_epoch_seed(
    *,
    seed: int,
    fixed_epoch: int,
    experiment: Mapping[str, Any],
    fit_blocks: Sequence[step7.CachedBlock],
    selection_blocks: Sequence[step7.CachedBlock],
    effect_class_weight: np.ndarray,
    mechanism_class_weight: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.nn.Module, step7.PredictionSet]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = step7.instantiate_model("late_concat", seed, experiment, device)
    training = experiment["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, int(fixed_epoch) + 1):
        train_metrics = step7.train_one_epoch(
            model,
            optimizer,
            fit_blocks,
            seed=seed,
            epoch=epoch,
            device=device,
            effect_class_weight=effect_class_weight,
            mechanism_class_weight=mechanism_class_weight,
            effect_loss_weight=float(training["effect_loss_weight"]),
            mechanism_loss_weight=float(training["mechanism_loss_weight"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        history.append({"seed": seed, "epoch": epoch, **train_metrics})
        print(
            f"late_concat deployment-refit seed={seed} epoch={epoch}/{fixed_epoch} "
            f"train_loss={train_metrics['total_loss']:.4f}",
            flush=True,
        )

    # Selection is observed exactly once after the already-fixed epoch. It cannot
    # alter the epoch, weights, architecture, hyperparameters, or deployed threshold.
    selection_prediction = step7.predict_blocks(model, selection_blocks, device)
    summary = step7.selection_summary(selection_prediction)
    record = {
        "seed": int(seed),
        "fixed_epoch": int(fixed_epoch),
        "epoch_source": "archived_step8_selected_epoch",
        "selected_by_current_refit": False,
        "selection_summary": summary,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    return record, history, model, selection_prediction


def deployment_bundle_id(identity: Mapping[str, Any], checkpoint_hashes: Mapping[str, str]) -> str:
    payload = {"identity": identity, "checkpoint_hashes": dict(sorted(checkpoint_hashes.items()))}
    return "deploy_" + hashlib.sha256(
        baseline.canonical_json(payload).encode("utf-8")
    ).hexdigest()[:24]


def refuse_existing_successful_bundle(output_parent: Path) -> None:
    existing = sorted(
        path for path in output_parent.glob("deploy_*")
        if path.is_dir() and (path / "bundle_complete.json").is_file()
    )
    if existing:
        raise RuntimeError(
            "an authoritative Step-9A deployment bundle already exists; "
            f"refusing a second candidate: {existing[0]}"
        )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = baseline.read_json(config_path)
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected Step-9A deployment-freeze schema")
    if config.get("status") != "FROZEN_POST_CONFIRMATION_FIXED_EPOCH_REFIT":
        raise RuntimeError("Step-9A fixed-epoch refit contract is not frozen")

    frozen = config["frozen_model"]
    if frozen["variant"] != "late_concat":
        raise RuntimeError("Step-9A may only serialize the confirmed late_concat architecture")
    if bool(frozen["current_refit_may_select_epoch"]):
        raise RuntimeError("Step-9A refit may not select an epoch")
    if bool(frozen["current_refit_may_select_threshold"]):
        raise RuntimeError("Step-9A refit may not select a deployment threshold")
    if bool(frozen["exact_step8_checkpoint_weights_claimed"]):
        raise RuntimeError("Step-9A may not claim irrecoverable Step-8 checkpoint weights")

    boundaries = config["scientific_boundaries"]
    forbidden_true = (
        "new_training_data",
        "new_hyperparameter_selection",
        "new_epoch_selection",
        "new_threshold_selection",
        "architecture_change",
        "spent_confirmatory_reuse",
        "hardware_execution",
        "exact_step8_weights_reconstructed",
    )
    if any(bool(boundaries[key]) for key in forbidden_true):
        raise RuntimeError("Step-9A scientific boundaries were relaxed")
    if not bool(boundaries["post_confirmation_fixed_epoch_refit"]):
        raise RuntimeError("Step-9A must disclose that this is a post-confirmation refit")

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

    development = (
        args.development_product_dir.expanduser().resolve()
        if args.development_product_dir
        else Path(config["development_source"]["default_product_dir"]).expanduser().resolve()
    )
    dev_complete, dev_rows = baseline.verify_source_product(development, experiment)
    if dev_complete["product_id"] != config["development_source"]["product_id"]:
        raise RuntimeError("development product identity changed")
    fit_roots, selection_roots, _outer_roots, dev_by_root = step7.split_root_indices(
        dev_rows, experiment
    )
    if len(fit_roots) != int(config["development_source"]["fit_roots"]):
        raise RuntimeError("development fit-root count changed")
    if len(selection_roots) != int(config["development_source"]["selection_roots"]):
        raise RuntimeError("development selection-root count changed")

    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    refuse_existing_successful_bundle(output_parent)

    device = step8.resolve_device(args.device)
    root_batch_size = int(step8_config["training"]["root_batch_size"])
    print("Materializing frozen development fit/selection data for Step-9A refit...", flush=True)
    fit_blocks = step7.materialize_blocks(
        product=development,
        rows=dev_rows,
        by_root=dev_by_root,
        roots=fit_roots,
        root_batch_size=root_batch_size,
        label="step9a-development-fit",
        progress_every=args.progress_every,
    )
    selection_blocks = step7.materialize_blocks(
        product=development,
        rows=dev_rows,
        by_root=dev_by_root,
        roots=selection_roots,
        root_batch_size=root_batch_size,
        label="step9a-development-selection",
        progress_every=args.progress_every,
    )
    effect_class_weight, mechanism_class_weight = step7.class_weights(fit_blocks)

    seeds = [int(value) for value in frozen["seeds"]]
    expected_seeds = [int(value) for value in step8_config["training"]["seeds"]]
    if seeds != expected_seeds:
        raise RuntimeError("Step-9A seed list differs from frozen Step-8 training")

    records: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    models: dict[int, torch.nn.Module] = {}
    selection_predictions: dict[int, step7.PredictionSet] = {}
    archived = config["archived_step8_selection_reference_descriptive_only"]

    for seed in seeds:
        fixed_epoch = int(frozen["fixed_training_epochs"][str(seed)])
        print(
            f"\nRefitting frozen late_concat seed={seed} for fixed epoch count={fixed_epoch}",
            flush=True,
        )
        record, history, model, selection_prediction = train_fixed_epoch_seed(
            seed=seed,
            fixed_epoch=fixed_epoch,
            experiment=experiment,
            fit_blocks=fit_blocks,
            selection_blocks=selection_blocks,
            effect_class_weight=effect_class_weight,
            mechanism_class_weight=mechanism_class_weight,
            device=device,
        )
        validate_refit_record(record=record, frozen=frozen)
        record["archived_step8_selection_reference"] = archived[str(seed)]
        record["selection_drift_from_archived"] = {
            metric: float(record["selection_summary"][metric]) - float(archived[str(seed)][metric])
            for metric in ("mechanism_balanced_accuracy", "effect_balanced_accuracy")
        }
        records[f"seed{seed}"] = record
        histories.extend(history)
        models[seed] = model
        selection_predictions[seed] = selection_prediction

    reference_prediction = selection_predictions[seeds[0]]
    for seed in seeds[1:]:
        step7.assert_prediction_alignment(
            reference_prediction,
            selection_predictions[seed],
            f"Step-9A deployment-refit selection seed {seed}",
        )
    mean_effect_logits = np.mean(
        np.stack([selection_predictions[seed].effect_logits for seed in seeds]), axis=0
    )
    descriptive_refit_threshold, _, _ = step7._metrics_binary(
        reference_prediction.effect_truth, mean_effect_logits
    )
    deployment_threshold = float(frozen["deployment_effect_threshold"])
    records["ensemble"] = {
        "aggregation": frozen["ensemble_aggregation"],
        "seeds": seeds,
        "deployment_effect_threshold": deployment_threshold,
        "deployment_threshold_source": frozen["deployment_effect_threshold_source"],
        "refit_selection_effect_threshold_descriptive_only": float(descriptive_refit_threshold),
        "refit_minus_deployment_threshold": float(descriptive_refit_threshold) - deployment_threshold,
        "current_refit_selected_threshold": False,
    }

    base_identity = {
        "schema": SCHEMA,
        "config_sha256": baseline.sha256_file(config_path),
        "step8_config_sha256": baseline.sha256_file(step8_path),
        "step7_config_sha256": baseline.sha256_file(step7_path),
        "development_product_id": dev_complete["product_id"],
        "source_evaluation_id": source["evaluation_id"],
        "architecture": "late_concat",
        "seeds": seeds,
        "fixed_training_epochs": frozen["fixed_training_epochs"],
        "deployment_effect_threshold": deployment_threshold,
        "weight_provenance": "post_confirmation_fixed_epoch_refit_development_only",
        "exact_step8_checkpoint_weights": False,
    }

    staging = output_parent / f".step9a.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        checkpoint_names: list[str] = []
        checkpoint_hashes: dict[str, str] = {}
        config_sha = baseline.sha256_file(config_path)
        step7_sha = baseline.sha256_file(step7_path)

        for seed in seeds:
            name = f"seed{seed}.pt"
            path = staging / name
            fixed_epoch = int(frozen["fixed_training_epochs"][str(seed)])
            torch.save(
                checkpoint_payload(
                    model=models[seed],
                    seed=seed,
                    fixed_epoch=fixed_epoch,
                    config_sha256=config_sha,
                    step7_config_sha256=step7_sha,
                ),
                path,
            )
            checkpoint_names.append(name)
            checkpoint_hashes[name] = baseline.sha256_file(path)

            reloaded = step7.instantiate_model(
                "late_concat", seed, experiment, torch.device("cpu")
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if bool(payload.get("exact_step8_checkpoint_weight", True)):
                raise RuntimeError("deployment checkpoint incorrectly claims exact Step-8 weight identity")
            reloaded.load_state_dict(payload["state_dict"])
            reloaded.eval()
            original = {
                key: value.detach().cpu() for key, value in models[seed].state_dict().items()
            }
            for key, value in reloaded.state_dict().items():
                if not torch.equal(value.detach().cpu(), original[key]):
                    raise RuntimeError(f"checkpoint reload mismatch for seed {seed}: {key}")

        bundle_id = deployment_bundle_id(base_identity, checkpoint_hashes)
        output = output_parent / bundle_id
        if output.exists():
            raise RuntimeError(f"refusing to overwrite deployment bundle: {output}")

        baseline.write_csv(staging / "training_history.csv", histories)
        baseline.atomic_json(staging / "model_selection.json", records)
        baseline.atomic_json(staging / "inference_contract.json", config["inference_contract"])
        files = checkpoint_names + [
            "training_history.csv",
            "model_selection.json",
            "inference_contract.json",
        ]
        file_hashes = {name: baseline.sha256_file(staging / name) for name in files}
        if checkpoint_hashes != {name: file_hashes[name] for name in checkpoint_names}:
            raise RuntimeError("checkpoint hashes changed during deployment freeze")

        identity = {**base_identity, "checkpoint_hashes": checkpoint_hashes}
        complete = {
            "schema": SCHEMA,
            "status": "DEPLOYMENT_REFIT_BUNDLE_FROZEN",
            "bundle_id": bundle_id,
            "identity": identity,
            "source_confirmation": source,
            "replay_failure_record": config["replay_failure_record"],
            "development_product_id": dev_complete["product_id"],
            "architecture": "late_concat",
            "seeds": seeds,
            "fixed_training_epochs": frozen["fixed_training_epochs"],
            "deployment_effect_threshold": deployment_threshold,
            "mechanism_classes": list(frozen["mechanism_classes"]),
            "trainable_parameter_count_each": int(frozen["expected_trainable_parameter_count"]),
            "checkpoint_hashes_define_weight_identity": True,
            "exact_step8_checkpoint_weights": False,
            "weight_provenance": "post_confirmation_fixed_epoch_refit_development_only",
            "current_refit_selected_epoch": False,
            "current_refit_selected_threshold": False,
            "checkpoint_reload_exact_passed": True,
            "new_tuning_performed": False,
            "spent_confirmatory_cohort_accessed": False,
            "hardware_executed": False,
            "file_hashes": file_hashes,
        }
        baseline.atomic_json(staging / "bundle_complete.json", complete)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 9A DEPLOYMENT REFIT BUNDLE FROZEN\n")
    print(f"Bundle: {bundle_id}")
    print("Architecture: late_concat")
    print(f"Seeds: {seeds}")
    print(f"Fixed training epochs: {frozen['fixed_training_epochs']}")
    print(f"Deployment effect threshold: {deployment_threshold:.17g}")
    print(f"Refit selection threshold (descriptive only): {float(descriptive_refit_threshold):.17g}")
    print("Exact Step-8 checkpoint weights claimed: NO")
    print("Current refit selected an epoch: NO")
    print("Current refit selected a threshold: NO")
    print("Checkpoint hashes define deployment weight identity: YES")
    print("Checkpoint reload exactness: PASS")
    print("Spent confirmatory cohort accessed: NO")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
