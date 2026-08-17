#!/usr/bin/env python3
"""One-shot Step 8 evaluation of the frozen late_concat architecture."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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
from triqto.step7.model import Step7DiagnosticModel

SCHEMA = "triqto.v0_2.step8_untouched_confirmatory.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step8_untouched_confirmatory.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step8_confirmatory_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--development-product-dir", type=Path)
    parser.add_argument("--confirmatory-cohort-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_sealed_cohort(cohort: Path, config: Mapping[str, Any], config_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    marker = cohort / "sealed_complete.json"
    if not marker.is_file():
        raise RuntimeError("missing Step-8 sealed completion marker")
    sealed = baseline.read_json(marker)
    if sealed.get("schema") != SCHEMA or sealed.get("status") != "SEALED_UNEVALUATED":
        raise RuntimeError("confirmatory cohort is not sealed and unevaluated")
    if sealed.get("identity", {}).get("config_sha256") != baseline.sha256_file(config_path):
        raise RuntimeError("sealed cohort does not match the frozen Step-8 protocol")
    expected = config["confirmatory_cohort"]
    if int(sealed.get("clean_circuit_root_count", -1)) != int(expected["clean_circuit_roots"]):
        raise RuntimeError("confirmatory clean-root count mismatch")
    if int(sealed.get("example_count", -1)) != int(expected["expected_examples"]):
        raise RuntimeError("confirmatory example count mismatch")
    if int(sealed.get("development_graph_overlap_count", -1)) != 0 or int(sealed.get("development_group_overlap_count", -1)) != 0:
        raise RuntimeError("confirmatory cohort overlaps development identities")
    if bool(sealed.get("target_summaries_exposed")) or bool(sealed.get("model_evaluated")):
        raise RuntimeError("confirmatory seal already exposes outcome information")
    manifests = cohort / "manifests"
    for name, expected_hash in sealed.get("manifest_hashes", {}).items():
        if baseline.sha256_file(manifests / name) != expected_hash:
            raise RuntimeError(f"confirmatory manifest hash mismatch: {name}")
    rows = read_csv(manifests / "example_manifest.csv")
    forbidden = set(config["blinding_and_sealing"]["human_visible_example_manifest_excludes"])
    if rows and forbidden.intersection(rows[0].keys()):
        raise RuntimeError("public confirmatory manifest exposes forbidden target fields")
    return sealed, rows


def build_confirmatory_groups(rows: Sequence[Mapping[str, str]], expected_roots: int, expected_per_root: int) -> tuple[list[int], dict[int, list[int]]]:
    by_root: dict[int, list[int]] = {}
    for source_index, row in enumerate(rows):
        root = int(row["root_index"])
        by_root.setdefault(root, []).append(source_index)
        if row.get("split") != "confirmatory":
            raise RuntimeError("Step-8 public manifest contains a non-confirmatory split")
    roots = sorted(by_root)
    if roots != list(range(expected_roots)):
        raise RuntimeError("confirmatory root indices are not the frozen contiguous local namespace")
    if any(len(by_root[root]) != expected_per_root for root in roots):
        raise RuntimeError("confirmatory root derivative count mismatch")
    return roots, by_root


def train_final_seed(*, seed: int, experiment: Mapping[str, Any], fit_blocks: Sequence[step7.CachedBlock], selection_blocks: Sequence[step7.CachedBlock], effect_class_weight: np.ndarray, mechanism_class_weight: np.ndarray, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], Step7DiagnosticModel, step7.PredictionSet]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = step7.instantiate_model("late_concat", seed, experiment, device)
    training = experiment["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    max_epochs = int(training["max_epochs"]); patience = int(training["early_stopping_patience"]); min_delta = float(training["early_stopping_min_delta"])
    best_summary: dict[str, Any] | None = None; best_state: dict[str, torch.Tensor] | None = None; best_epoch: int | None = None; stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = step7.train_one_epoch(model, optimizer, fit_blocks, seed=seed, epoch=epoch, device=device, effect_class_weight=effect_class_weight, mechanism_class_weight=mechanism_class_weight, effect_loss_weight=float(training["effect_loss_weight"]), mechanism_loss_weight=float(training["mechanism_loss_weight"]), gradient_clip_norm=float(training["gradient_clip_norm"]))
        selection_prediction = step7.predict_blocks(model, selection_blocks, device)
        summary = step7.selection_summary(selection_prediction)
        history.append({"seed": seed, "epoch": epoch, **train_metrics, **{f"selection_{key}": value for key, value in summary.items()}, "selected_checkpoint": False})
        print(f"late_concat seed={seed} epoch={epoch} train_loss={train_metrics['total_loss']:.4f} sel_mech_BA={summary['mechanism_balanced_accuracy']:.4f} sel_effect_BA={summary['effect_balanced_accuracy']:.4f}", flush=True)
        if step7.checkpoint_is_better(summary, best_summary, min_delta):
            best_summary = dict(summary); best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_summary is None or best_epoch is None:
        raise RuntimeError(f"failed to select frozen late_concat checkpoint for seed {seed}")
    for row in history:
        row["selected_checkpoint"] = int(row["epoch"]) == int(best_epoch)
    model.load_state_dict(best_state); model.to(device)
    selection_prediction = step7.predict_blocks(model, selection_blocks, device)
    selected = {"seed": seed, "selected_epoch": best_epoch, "selection_summary": step7.selection_summary(selection_prediction), "epochs_ran": len(history), "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return selected, history, model, selection_prediction


def write_access_marker(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"confirmatory access marker already exists: {path}; cohort is spent and may not be re-evaluated")
    baseline.atomic_json(path, payload)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve(); config = baseline.read_json(config_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_CONFIRMATORY_COHORT_GENERATION":
        raise RuntimeError("unexpected Step-8 frozen protocol")
    if config["final_architecture"]["variant"] != "late_concat" or not bool(config["final_architecture"]["architecture_search_closed"]):
        raise RuntimeError("Step-8 architecture is not the frozen Step-7.1 champion")
    experiment_path = args.step7_config.expanduser().resolve(); experiment = baseline.read_json(experiment_path)
    if experiment.get("schema") != step7.EXPERIMENT_SCHEMA:
        raise RuntimeError("unexpected Step-7 training contract")

    development = args.development_product_dir.expanduser().resolve() if args.development_product_dir else Path(config["development_source"]["default_product_dir"]).expanduser().resolve()
    dev_complete, dev_rows = baseline.verify_source_product(development, experiment)
    fit_roots, selection_roots, outer_roots, dev_by_root = step7.split_root_indices(dev_rows, experiment)
    if len(fit_roots) != int(config["training"]["fit_roots"]) or len(selection_roots) != int(config["training"]["selection_roots"]):
        raise RuntimeError("development fit/selection partition changed")
    if bool(config["training"]["step7_outer_development_roots_used_for_training_or_selection"]):
        raise RuntimeError("frozen Step-8 protocol incorrectly allows old outer roots into selection")

    cohort = args.confirmatory_cohort_dir.expanduser().resolve()
    sealed, confirm_rows = verify_sealed_cohort(cohort, config, config_path)
    confirm_roots, confirm_by_root = build_confirmatory_groups(confirm_rows, int(config["confirmatory_cohort"]["clean_circuit_roots"]), int(config["confirmatory_cohort"]["examples_per_root"]))

    output_parent = args.output_parent.expanduser().resolve(); output_parent.mkdir(parents=True, exist_ok=True)
    access_marker = output_parent / f"CONFIRMATORY_ACCESS_STARTED__{sealed['cohort_id']}.json"
    spent_marker = output_parent / f"CONFIRMATORY_SPENT__{sealed['cohort_id']}.json"
    if access_marker.exists() or spent_marker.exists():
        raise RuntimeError("this Step-8 cohort has already entered confirmatory evaluation and is scientifically spent")

    device = resolve_device(args.device)
    print("Materializing frozen development fit/selection data before confirmatory access...", flush=True)
    root_batch_size = int(config["training"]["root_batch_size"])
    fit_blocks = step7.materialize_blocks(product=development, rows=dev_rows, by_root=dev_by_root, roots=fit_roots, root_batch_size=root_batch_size, label="step8-development-fit", progress_every=args.progress_every)
    selection_blocks = step7.materialize_blocks(product=development, rows=dev_rows, by_root=dev_by_root, roots=selection_roots, root_batch_size=root_batch_size, label="step8-development-selection", progress_every=args.progress_every)
    effect_class_weight, mechanism_class_weight = step7.class_weights(fit_blocks)
    print(f"Step 8 training device: {device}", flush=True)

    seeds = [int(v) for v in config["training"]["seeds"]]
    selections: dict[str, Any] = {}; histories: list[dict[str, Any]] = []; models: dict[int, Step7DiagnosticModel] = {}; selection_predictions: dict[int, step7.PredictionSet] = {}
    for seed in seeds:
        print(f"\nTraining frozen late_concat seed={seed} before unsealing Step 8", flush=True)
        selected, history, model, selection_prediction = train_final_seed(seed=seed, experiment=experiment, fit_blocks=fit_blocks, selection_blocks=selection_blocks, effect_class_weight=effect_class_weight, mechanism_class_weight=mechanism_class_weight, device=device)
        selections[f"seed{seed}"] = selected; histories.extend(history); models[seed] = model; selection_predictions[seed] = selection_prediction
    reference_selection = selection_predictions[seeds[0]]
    for seed in seeds[1:]:
        step7.assert_prediction_alignment(reference_selection, selection_predictions[seed], f"Step 8 selection seed {seed}")
    mean_selection_effect_logits = np.mean(np.stack([selection_predictions[seed].effect_logits for seed in seeds]), axis=0)
    ensemble_threshold, _, _ = step7._metrics_binary(reference_selection.effect_truth, mean_selection_effect_logits)
    selections["ensemble"] = {"effect_threshold": ensemble_threshold, "aggregation": "mean_logits_across_frozen_seeds", "seeds": seeds}

    # This marker is the irreversible scientific boundary. Everything above used
    # development data only. Everything below is confirmatory access.
    access_identity = {
        "schema": SCHEMA,
        "status": "CONFIRMATORY_ACCESS_STARTED",
        "cohort_id": sealed["cohort_id"],
        "config_sha256": baseline.sha256_file(config_path),
        "runner_sha256": baseline.sha256_file(Path(__file__).resolve()),
        "development_product_id": dev_complete["product_id"],
        "architecture": "late_concat",
        "seeds": seeds,
        "effect_threshold_selected_before_confirmatory_access": ensemble_threshold,
    }
    write_access_marker(access_marker, access_identity)
    print("\nCONFIRMATORY ACCESS STARTED — this cohort is now scientifically spent after this attempt.\n", flush=True)

    confirm_blocks = step7.materialize_blocks(product=cohort, rows=confirm_rows, by_root=confirm_by_root, roots=confirm_roots, root_batch_size=root_batch_size, label="step8-confirmatory", progress_every=args.progress_every)
    confirm_predictions: dict[int, step7.PredictionSet] = {}
    for seed in seeds:
        confirm_predictions[seed] = step7.predict_blocks(models[seed], confirm_blocks, device)
    reference = confirm_predictions[seeds[0]]
    for seed in seeds[1:]:
        step7.assert_prediction_alignment(reference, confirm_predictions[seed], f"Step 8 confirmatory seed {seed}")
    mean_effect_logits = np.mean(np.stack([confirm_predictions[seed].effect_logits for seed in seeds]), axis=0)
    mean_mechanism_logits = np.mean(np.stack([confirm_predictions[seed].mechanism_logits for seed in seeds]), axis=0)

    eval_cfg = config["evaluation"]
    metric_rows, bootstraps, predictions = step7.bootstrap_rows(name="late_concat_confirmatory", truth=reference, effect_logits=mean_effect_logits, mechanism_logits=mean_mechanism_logits, threshold=float(ensemble_threshold), replicates=int(eval_cfg["root_group_bootstrap_replicates"]), seed=int(eval_cfg["bootstrap_seed"]), confidence=float(eval_cfg["confidence_level"]))
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_threshold = float(selections[f"seed{seed}"]["selection_summary"]["effect_threshold"])
        seed_rows.extend(step7.simple_metric_rows(prefix=f"late_concat__seed{seed}", truth=confirm_predictions[seed], effect_logits=confirm_predictions[seed].effect_logits, mechanism_logits=confirm_predictions[seed].mechanism_logits, threshold=seed_threshold))

    metric_lookup = {row["task"]: row for row in metric_rows}
    mechanism_row = metric_lookup["mechanism_diagnosis"]; effect_row = metric_lookup["effect_detection"]; integrated_row = metric_lookup["integrated_diagnosis"]
    primary_gate = eval_cfg["primary_gate"]
    mechanism_recalls = {name: float(mechanism_row[f"recall__{name}"]) for name in primary_gate["classes"]}
    primary_supported = float(mechanism_row["balanced_accuracy_ci_low"]) > float(primary_gate["balanced_accuracy_ci_low_strictly_greater_than"]) and min(mechanism_recalls.values()) >= float(primary_gate["minimum_each_mechanism_recall"])
    secondary = eval_cfg["secondary_gates"]
    effect_supported = float(effect_row["balanced_accuracy_ci_low"]) > float(secondary["effect_balanced_accuracy_ci_low_strictly_greater_than"])
    integrated_supported = float(integrated_row["balanced_accuracy_ci_low"]) > float(secondary["integrated_balanced_accuracy_ci_low_strictly_greater_than"])

    stratified_rows = step7.stratified_metric_rows(name="late_concat_confirmatory", outer=reference, rows=confirm_rows, effect_pred=predictions["effect_pred"], mechanism_pred_all=predictions["mechanism_pred_all"], strata=eval_cfg["report_strata"], minimum=30)
    decision = {
        "schema": SCHEMA,
        "decision": config["decision"]["completion_decision"],
        "evidence_status": "CONFIRMATORY_SPENT",
        "primary_mechanism_supported": primary_supported,
        "secondary_effect_supported": effect_supported,
        "secondary_integrated_supported": integrated_supported,
        "selected_architecture": "late_concat",
        "mechanism_balanced_accuracy": float(mechanism_row["balanced_accuracy"]),
        "mechanism_balanced_accuracy_ci": [float(mechanism_row["balanced_accuracy_ci_low"]), float(mechanism_row["balanced_accuracy_ci_high"])],
        "mechanism_recalls": mechanism_recalls,
        "effect_balanced_accuracy": float(effect_row["balanced_accuracy"]),
        "effect_balanced_accuracy_ci": [float(effect_row["balanced_accuracy_ci_low"]), float(effect_row["balanced_accuracy_ci_high"])],
        "integrated_balanced_accuracy": float(integrated_row["balanced_accuracy"]),
        "integrated_balanced_accuracy_ci": [float(integrated_row["balanced_accuracy_ci_low"]), float(integrated_row["balanced_accuracy_ci_high"])],
        "development_reference_mechanism_ba": float(eval_cfg["development_reference_mechanism_ba"]),
        "confirm_minus_development_reference_mechanism_ba": float(mechanism_row["balanced_accuracy"]) - float(eval_cfg["development_reference_mechanism_ba"]),
        "confirmatory_labels_selected_nothing": True,
        "architecture_changed_after_confirmatory_access": False,
        "historical_v0_1_test_accessed": False,
        "spent_phase15_6_confirmatory_accessed": False,
        "hardware_executed": False,
    }

    identity = {
        "schema": SCHEMA,
        "config_sha256": baseline.sha256_file(config_path),
        "step7_config_sha256": baseline.sha256_file(experiment_path),
        "runner_sha256": baseline.sha256_file(Path(__file__).resolve()),
        "sealed_complete_sha256": baseline.sha256_file(cohort / "sealed_complete.json"),
        "cohort_id": sealed["cohort_id"],
        "development_product_id": dev_complete["product_id"],
        "architecture": "late_concat",
        "seeds": seeds,
    }
    evaluation_id = "confirm_eval_" + hashlib.sha256(baseline.canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    output = output_parent / evaluation_id
    if output.exists():
        raise RuntimeError("refusing to overwrite Step-8 confirmatory evaluation")
    staging = output_parent / f".{evaluation_id}.staging-{uuid.uuid4().hex}"; staging.mkdir()
    baseline.write_csv(staging / "confirmatory_metrics.csv", metric_rows)
    baseline.write_csv(staging / "seed_metrics.csv", seed_rows)
    baseline.write_csv(staging / "stratified_metrics.csv", stratified_rows)
    baseline.write_csv(staging / "training_history.csv", histories)
    baseline.atomic_json(staging / "model_selection.json", selections)
    baseline.atomic_json(staging / "decision.json", decision)
    payload = {
        "confirmatory_source_index": reference.source_indices.astype(np.int64),
        "confirmatory_root_index": reference.root_indices.astype(np.int64),
        "effect_truth": reference.effect_truth.astype(np.int8),
        "mechanism_truth_all": reference.mechanism_truth_all.astype(np.int8),
        "mechanism_loss_mask": reference.mechanism_mask.astype(bool),
        "effect__late_concat__logit": mean_effect_logits.astype(np.float32),
        "effect__late_concat__pred": predictions["effect_pred"].astype(np.int8),
        "mechanism__late_concat__logits": mean_mechanism_logits.astype(np.float32),
        "mechanism__late_concat__pred_all": predictions["mechanism_pred_all"].astype(np.int8),
    }
    for seed in seeds:
        payload[f"effect__late_concat__seed{seed}__logit"] = confirm_predictions[seed].effect_logits.astype(np.float32)
        payload[f"mechanism__late_concat__seed{seed}__logits"] = confirm_predictions[seed].mechanism_logits.astype(np.float32)
    np.savez_compressed(staging / "confirmatory_predictions.npz", **payload)
    files = ["confirmatory_metrics.csv", "seed_metrics.csv", "stratified_metrics.csv", "training_history.csv", "model_selection.json", "decision.json", "confirmatory_predictions.npz"]
    complete = {
        "schema": SCHEMA,
        "status": "COMPLETE_SPENT_CONFIRMATORY",
        "evaluation_id": evaluation_id,
        "identity": identity,
        "cohort_id": sealed["cohort_id"],
        "clean_circuit_root_count": len(confirm_roots),
        "example_count": len(confirm_rows),
        "primary_mechanism_supported": primary_supported,
        "secondary_effect_supported": effect_supported,
        "secondary_integrated_supported": integrated_supported,
        "file_hashes": {name: baseline.sha256_file(staging / name) for name in files},
        "confirmatory_access_marker_sha256": baseline.sha256_file(access_marker),
    }
    baseline.atomic_json(staging / "evaluation_complete.json", complete)
    os.replace(staging, output)
    baseline.atomic_json(spent_marker, {"schema": SCHEMA, "status": "CONFIRMATORY_SPENT", "cohort_id": sealed["cohort_id"], "evaluation_id": evaluation_id, "evaluation_complete_sha256": baseline.sha256_file(output / "evaluation_complete.json"), "may_be_reused_as_confirmatory": False})

    print("\nTRIQTO STEP 8 ONE-SHOT CONFIRMATORY EVALUATION COMPLETE\n")
    print(f"Decision: {config['decision']['completion_decision']}")
    print(f"Primary mechanism supported: {'YES' if primary_supported else 'NO'}")
    print(f"Mechanism BA: {float(mechanism_row['balanced_accuracy']):.4f} 95% CI [{float(mechanism_row['balanced_accuracy_ci_low']):.4f}, {float(mechanism_row['balanced_accuracy_ci_high']):.4f}]")
    print("Mechanism recalls: " + ", ".join(f"{name}={value:.4f}" for name, value in mechanism_recalls.items()))
    print(f"Secondary effect supported: {'YES' if effect_supported else 'NO'} | BA={float(effect_row['balanced_accuracy']):.4f}")
    print(f"Secondary integrated supported: {'YES' if integrated_supported else 'NO'} | BA={float(integrated_row['balanced_accuracy']):.4f}")
    print("Architecture: late_concat (frozen before cohort generation)")
    print("Confirmatory cohort is now SPENT and may not be reused as confirmatory evidence.")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
