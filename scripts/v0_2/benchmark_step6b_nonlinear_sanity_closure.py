#!/usr/bin/env python3
"""Run the Step 6B nonlinear sanity closure on the frozen Step 5 v3 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import benchmark_step6_cheap_baselines as base

SCHEMA = "triqto.v0_2.step6b_nonlinear_sanity_closure.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step6b_nonlinear_sanity_closure.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step6b_nonlinear_sanity")
QDA_VARIANTS = {
    "graph_stats_diag_qda": "graph_stats_only",
    "diag_full_diag_qda": "diag_full",
    "diag_full_context_graph_diag_qda": "diag_full_context_graph",
    "exact_diag_full_diag_qda_oracle": "exact_diag_full_oracle",
}
PRIVILEGED = {"exact_diag_rms_threshold_oracle", "exact_diag_full_diag_qda_oracle"}
MAGNITUDE_VARIANTS = ("diag_rms_threshold", "diag_snr_proxy_threshold", "exact_diag_rms_threshold_oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--step6a-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def verify_step6a(path: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected = config["source_step6a"]
    complete = base.read_json(path / "benchmark_complete.json")
    if complete.get("benchmark_id") != expected["benchmark_id"]:
        raise RuntimeError("Step 6A benchmark ID mismatch")
    if complete.get("source_product_id") != config["source_dataset"]["product_id"]:
        raise RuntimeError("Step 6A source product mismatch")
    for name, expected_hash in complete.get("file_hashes", {}).items():
        if base.sha256_file(path / name) != expected_hash:
            raise RuntimeError(f"Step 6A result hash mismatch: {name}")
    with np.load(path / "validation_predictions.npz", allow_pickle=False) as payload:
        predictions = {key: payload[key] for key in payload.files}
    return complete, predictions


def diagnostic_rms(diag_full: np.ndarray) -> np.ndarray:
    """RMS over active finite/exact diagnostic values, excluding structural masks."""
    matrix = np.asarray(diag_full, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 147:
        raise ValueError(f"expected [N,147] canonical diagnostic matrix, got {matrix.shape}")
    local_values = matrix[:, :24]
    local_mask = matrix[:, 24:32]
    pair_values = matrix[:, 32:116]
    pair_mask = matrix[:, 116:144]
    parity = matrix[:, 144:147]
    active = 3.0 * np.sum(local_mask, axis=1) + 3.0 * np.sum(pair_mask, axis=1) + 3.0
    if np.any(active <= 0):
        raise RuntimeError("invalid active diagnostic coordinate count")
    energy = np.sum(local_values**2, axis=1) + np.sum(pair_values**2, axis=1) + np.sum(parity**2, axis=1)
    return np.sqrt(energy / active)


def fit_diag_qda(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    shrinkage: float,
    variance_floor: float,
) -> dict[str, np.ndarray]:
    matrix = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int64)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (matrix - mean) / scale
    class_means = np.zeros((n_classes, matrix.shape[1]), dtype=np.float64)
    class_vars = np.zeros_like(class_means)
    for class_index in range(n_classes):
        subset = z[labels == class_index]
        if len(subset) == 0:
            raise RuntimeError(f"QDA fit missing class {class_index}")
        class_means[class_index] = np.mean(subset, axis=0)
        raw_var = np.var(subset, axis=0)
        shrunk = (1.0 - float(shrinkage)) * raw_var + float(shrinkage)
        class_vars[class_index] = np.maximum(shrunk, float(variance_floor))
    return {"mean": mean, "scale": scale, "class_means": class_means, "class_vars": class_vars}


def diag_qda_scores(X: np.ndarray, fitted: Mapping[str, np.ndarray]) -> np.ndarray:
    z = (np.asarray(X, dtype=np.float64) - fitted["mean"]) / fitted["scale"]
    class_means = fitted["class_means"]
    class_vars = fitted["class_vars"]
    scores = np.empty((len(z), len(class_means)), dtype=np.float64)
    for class_index in range(len(class_means)):
        delta = z - class_means[class_index]
        scores[:, class_index] = -0.5 * np.sum(np.log(class_vars[class_index]) + (delta**2) / class_vars[class_index], axis=1)
    return scores


def tune_diag_qda(
    X: np.ndarray,
    y: np.ndarray,
    population_mask: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    occurrence: np.ndarray,
    n_classes: int,
    shrinkages: Sequence[float],
    variance_floor: float,
) -> dict[str, Any]:
    eligible_train = population_mask & train_mask
    train_indices = np.flatnonzero(eligible_train)
    position = {int(index): pos for pos, index in enumerate(train_indices)}
    oof = {float(s): np.full((len(train_indices), n_classes), np.nan, dtype=np.float64) for s in shrinkages}
    for heldout_root_mask in base.training_oof_folds(occurrence, train_mask):
        heldout = eligible_train & heldout_root_mask
        fit_mask = eligible_train & ~heldout_root_mask
        heldout_indices = np.flatnonzero(heldout)
        for shrinkage in shrinkages:
            fitted = fit_diag_qda(X[fit_mask], y[fit_mask], n_classes, float(shrinkage), variance_floor)
            scores = diag_qda_scores(X[heldout], fitted)
            for row_index, source_index in enumerate(heldout_indices):
                oof[float(shrinkage)][position[int(source_index)]] = scores[row_index]
    y_oof = y[train_indices]
    summaries: dict[str, Any] = {}
    best_key: tuple[float, ...] | None = None
    best_shrinkage: float | None = None
    best_threshold: float | None = None
    for shrinkage in shrinkages:
        scores = oof[float(shrinkage)]
        if np.any(~np.isfinite(scores)):
            raise RuntimeError(f"incomplete QDA OOF predictions at shrinkage {shrinkage}")
        if n_classes == 2:
            difference = scores[:, 1] - scores[:, 0]
            threshold, metrics = base.select_binary_threshold(y_oof, difference)
            summaries[str(shrinkage)] = {
                "shrinkage": float(shrinkage),
                "threshold": float(threshold),
                **metrics,
                "roc_auc": base.binary_auc(y_oof, difference),
            }
            key = (float(metrics["balanced_accuracy"]), float(metrics["macro_f1"]), float(shrinkage))
        else:
            pred = np.argmax(scores, axis=1).astype(np.int8)
            metrics = base.metrics_from_cm(base.confusion_matrix(y_oof, pred, n_classes))
            summaries[str(shrinkage)] = {
                "shrinkage": float(shrinkage),
                **metrics,
                "macro_ovr_roc_auc": base.macro_ovr_auc(y_oof, scores, n_classes),
            }
            threshold = None
            key = (float(metrics["balanced_accuracy"]), float(metrics["macro_f1"]), float(shrinkage))
        if best_key is None or key > best_key:
            best_key = key
            best_shrinkage = float(shrinkage)
            best_threshold = None if threshold is None else float(threshold)
    assert best_shrinkage is not None
    fitted = fit_diag_qda(X[eligible_train], y[eligible_train], n_classes, best_shrinkage, variance_floor)
    validation_scores = diag_qda_scores(X[validation_mask], fitted)
    if n_classes == 2:
        difference = validation_scores[:, 1] - validation_scores[:, 0]
        validation_pred = (difference >= float(best_threshold)).astype(np.int8)
        output_scores = difference
    else:
        validation_pred = np.argmax(validation_scores, axis=1).astype(np.int8)
        output_scores = validation_scores
    return {
        "best_shrinkage": best_shrinkage,
        "best_threshold": best_threshold,
        "oof_summary": summaries,
        "validation_pred_all": validation_pred,
        "validation_scores_all": output_scores,
        "n_train": int(np.sum(eligible_train)),
    }


def fixed_threshold_fit(score: np.ndarray, truth: np.ndarray, train_mask: np.ndarray, validation_mask: np.ndarray) -> dict[str, Any]:
    threshold, train_metrics = base.select_binary_threshold(truth[train_mask], score[train_mask])
    validation_score = score[validation_mask]
    validation_pred = (validation_score >= threshold).astype(np.int8)
    return {"threshold": threshold, "training_metrics": train_metrics, "validation_scores_all": validation_score, "validation_pred_all": validation_pred, "n_train": int(np.sum(train_mask))}


def add_prior_bootstraps(
    *,
    prior_predictions: Mapping[str, np.ndarray],
    names: Sequence[str],
    effect_truth: np.ndarray,
    mechanism_truth: np.ndarray,
    mechanism_relative: np.ndarray,
    mechanism_all_truth: np.ndarray,
    validation_groups: np.ndarray,
    mechanism_groups: np.ndarray,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
    boot_by_task: dict[str, dict[str, Mapping[str, np.ndarray]]],
) -> None:
    for name in names:
        epred = np.asarray(prior_predictions[f"effect__{name}__pred"], dtype=np.int8)
        mpred_all = np.asarray(prior_predictions[f"mechanism__{name}__pred_all"], dtype=np.int8)
        _, eboot = base.evaluation_row(task="effect_detection", baseline=name, privileged=name == "exact_diag_full_oracle", y_true=effect_truth, y_pred=epred, scores=None, groups=validation_groups, class_names=("no_effect", "effect"), n_train=0, selected_lambda=None, selected_threshold=None, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence)
        boot_by_task["effect_detection"][name] = eboot
        _, mboot = base.evaluation_row(task="mechanism_diagnosis", baseline=name, privileged=name == "exact_diag_full_oracle", y_true=mechanism_truth, y_pred=mpred_all[mechanism_relative], scores=None, groups=mechanism_groups, class_names=base.MECHANISM_NAMES, n_train=0, selected_lambda=None, selected_threshold=None, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence)
        boot_by_task["mechanism_diagnosis"][name] = mboot
        integrated_truth = np.where(effect_truth == 0, 0, mechanism_all_truth.astype(np.int64) + 1).astype(np.int8)
        integrated_pred = np.where(epred == 0, 0, mpred_all.astype(np.int64) + 1).astype(np.int8)
        _, iboot = base.evaluation_row(task="integrated_diagnosis", baseline=name, privileged=name == "exact_diag_full_oracle", y_true=integrated_truth, y_pred=integrated_pred, scores=None, groups=validation_groups, class_names=base.INTEGRATED_NAMES, n_train=0, selected_lambda=None, selected_threshold=None, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence)
        boot_by_task["integrated_diagnosis"][name] = iboot


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = base.read_json(config_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_AFTER_STEP6A_BEFORE_STEP6B_OUTCOME":
        raise RuntimeError("unexpected Step 6B config schema/status")
    product = args.product_dir.expanduser().resolve() if args.product_dir else Path(config["source_dataset"]["default_product_dir"]).expanduser().resolve()
    step6a_dir = args.step6a_dir.expanduser().resolve() if args.step6a_dir else Path(config["source_step6a"]["default_benchmark_dir"]).expanduser().resolve()
    source_config = {
        "source_dataset": {
            "schema": "triqto.v0_2.step5_matched_diagnostic_training_dataset.v3",
            "product_id": config["source_dataset"]["product_id"],
            "clean_circuit_root_count": 5000,
            "example_count": 65000,
            "train_clean_root_count": 4000,
            "validation_clean_root_count": 1000,
        }
    }
    source_complete, manifest_rows = base.verify_source_product(product, source_config)
    _, prior_predictions = verify_step6a(step6a_dir, config)
    table = base.load_feature_table(product, manifest_rows, args.progress_every)
    features = table["features"]
    train_mask = table["train_mask"]
    validation_mask = table["validation_mask"]
    effect = table["effect"]
    mechanism = table["mechanism"]
    mechanism_mask = table["mechanism_mask"]
    occurrence = table["family_occurrence_index"]
    groups = table["groups"]
    validation_indices = np.flatnonzero(validation_mask)
    validation_groups = groups[validation_mask]
    effect_truth = effect[validation_mask]
    mechanism_relative = mechanism_mask[validation_mask]
    mechanism_truth = mechanism[validation_mask][mechanism_relative]
    mechanism_groups = validation_groups[mechanism_relative]
    mechanism_all_truth = mechanism[validation_mask]
    shots = np.asarray([int(row["shots"]) for row in table["manifest_context"]], dtype=np.float64)

    bootstrap_replicates = int(config["evaluation"]["root_group_bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    confidence = float(config["evaluation"]["confidence_level"])
    metric_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    boot_by_task: dict[str, dict[str, Mapping[str, np.ndarray]]] = {"effect_detection": {}, "mechanism_diagnosis": {}, "integrated_diagnosis": {}}

    finite_rms = diagnostic_rms(features["diag_full"])
    exact_rms = diagnostic_rms(features["exact_diag_full_oracle"])
    score_map = {
        "diag_rms_threshold": finite_rms,
        "diag_snr_proxy_threshold": finite_rms * np.sqrt(shots),
        "exact_diag_rms_threshold_oracle": exact_rms,
    }
    for name, score in score_map.items():
        fitted = fixed_threshold_fit(score, effect, train_mask, validation_mask)
        selection[name] = {"threshold": fitted["threshold"], "training_metrics": fitted["training_metrics"], "privileged_analysis_only": name in PRIVILEGED}
        predictions[name] = {"effect_pred": fitted["validation_pred_all"], "effect_score": fitted["validation_scores_all"]}
        row, boot = base.evaluation_row(task="effect_detection", baseline=name, privileged=name in PRIVILEGED, y_true=effect_truth, y_pred=fitted["validation_pred_all"], scores=fitted["validation_scores_all"], groups=validation_groups, class_names=("no_effect", "effect"), n_train=fitted["n_train"], selected_lambda=None, selected_threshold=fitted["threshold"], bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence)
        metric_rows.append(row)
        boot_by_task["effect_detection"][name] = boot

    shrinkages = [float(v) for v in config["diagonal_qda"]["shrinkage_candidates"]]
    variance_floor = float(config["diagonal_qda"]["variance_floor"])
    for name, source_feature in QDA_VARIANTS.items():
        matrix = features[source_feature]
        print(f"Fitting Step 6B baseline: {name}", flush=True)
        effect_fit = tune_diag_qda(matrix, effect, np.ones(len(effect), dtype=bool), train_mask, validation_mask, occurrence, 2, shrinkages, variance_floor)
        mechanism_fit = tune_diag_qda(matrix, mechanism, mechanism_mask, train_mask, validation_mask, occurrence, 3, shrinkages, variance_floor)
        predictions[name] = {
            "effect_pred": effect_fit["validation_pred_all"],
            "effect_score": effect_fit["validation_scores_all"],
            "mechanism_pred_all": mechanism_fit["validation_pred_all"],
            "mechanism_scores_all": mechanism_fit["validation_scores_all"],
        }
        selection[name] = {
            "privileged_analysis_only": name in PRIVILEGED,
            "effect_detection": {"selected_shrinkage": effect_fit["best_shrinkage"], "selected_threshold": effect_fit["best_threshold"], "oof_by_shrinkage": effect_fit["oof_summary"]},
            "mechanism_diagnosis": {"selected_shrinkage": mechanism_fit["best_shrinkage"], "oof_by_shrinkage": mechanism_fit["oof_summary"]},
        }
        erow, eboot = base.evaluation_row(task="effect_detection", baseline=name, privileged=name in PRIVILEGED, y_true=effect_truth, y_pred=effect_fit["validation_pred_all"], scores=effect_fit["validation_scores_all"], groups=validation_groups, class_names=("no_effect", "effect"), n_train=effect_fit["n_train"], selected_lambda=None, selected_threshold=effect_fit["best_threshold"], bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence)
        metric_rows.append(erow); boot_by_task["effect_detection"][name] = eboot
        mech_pred = mechanism_fit["validation_pred_all"][mechanism_relative]
        mech_scores = mechanism_fit["validation_scores_all"][mechanism_relative]
        mrow, mboot = base.evaluation_row(task="mechanism_diagnosis", baseline=name, privileged=name in PRIVILEGED, y_true=mechanism_truth, y_pred=mech_pred, scores=mech_scores, groups=mechanism_groups, class_names=base.MECHANISM_NAMES, n_train=mechanism_fit["n_train"], selected_lambda=None, selected_threshold=None, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence)
        metric_rows.append(mrow); boot_by_task["mechanism_diagnosis"][name] = mboot
        integrated_truth = np.where(effect_truth == 0, 0, mechanism_all_truth.astype(np.int64) + 1).astype(np.int8)
        integrated_pred = np.where(effect_fit["validation_pred_all"] == 0, 0, mechanism_fit["validation_pred_all"].astype(np.int64) + 1).astype(np.int8)
        irow, iboot = base.evaluation_row(task="integrated_diagnosis", baseline=name, privileged=name in PRIVILEGED, y_true=integrated_truth, y_pred=integrated_pred, scores=None, groups=validation_groups, class_names=base.INTEGRATED_NAMES, n_train=int(np.sum(train_mask)), selected_lambda=None, selected_threshold=effect_fit["best_threshold"], bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence)
        metric_rows.append(irow); boot_by_task["integrated_diagnosis"][name] = iboot

    prior_names = list(config["source_step6a"]["linear_reference_baselines"])
    add_prior_bootstraps(prior_predictions=prior_predictions, names=prior_names, effect_truth=effect_truth, mechanism_truth=mechanism_truth, mechanism_relative=mechanism_relative, mechanism_all_truth=mechanism_all_truth, validation_groups=validation_groups, mechanism_groups=mechanism_groups, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence, boot_by_task=boot_by_task)
    comparisons = [tuple(pair) for pair in config["evaluation"]["paired_validation_comparisons"]]
    for task, boot in boot_by_task.items():
        valid = [pair for pair in comparisons if pair[0] in boot and pair[1] in boot]
        paired_rows.extend(base.paired_difference_rows(task, valid, boot, confidence))

    for name in config["evaluation"]["stratify_variants"]:
        if name in MAGNITUDE_VARIANTS:
            stratified_rows.extend(base.stratified_rows(task="effect_detection", baseline=name, y_true=effect_truth, y_pred=predictions[name]["effect_pred"], source_indices=validation_indices, context=table["manifest_context"], class_names=("no_effect", "effect"), strata=config["evaluation"]["strata"], minimum=int(config["evaluation"]["minimum_stratum_examples"])))
        else:
            stratified_rows.extend(base.stratified_rows(task="effect_detection", baseline=name, y_true=effect_truth, y_pred=predictions[name]["effect_pred"], source_indices=validation_indices, context=table["manifest_context"], class_names=("no_effect", "effect"), strata=config["evaluation"]["strata"], minimum=int(config["evaluation"]["minimum_stratum_examples"])))
            mech_source = validation_indices[mechanism_relative]
            stratified_rows.extend(base.stratified_rows(task="mechanism_diagnosis", baseline=name, y_true=mechanism_truth, y_pred=predictions[name]["mechanism_pred_all"][mechanism_relative], source_indices=mech_source, context=table["manifest_context"], class_names=base.MECHANISM_NAMES, strata=config["evaluation"]["strata"], minimum=int(config["evaluation"]["minimum_stratum_examples"])))

    metric_lookup = {(row["task"], row["baseline"]): row for row in metric_rows}
    paired_lookup = {(row["task"], row["left"], row["right"], row["metric"]): row for row in paired_rows}
    flags = {
        "snr_proxy_effect_beats_linear_diag_full_ci": float(paired_lookup[("effect_detection", "diag_snr_proxy_threshold", "diag_full", "balanced_accuracy")]["ci_low"]) > 0.0,
        "finite_qda_mechanism_beats_linear_diag_full_ci": float(paired_lookup[("mechanism_diagnosis", "diag_full_diag_qda", "diag_full", "balanced_accuracy")]["ci_low"]) > 0.0,
        "context_graph_qda_mechanism_beats_linear_context_graph_ci": float(paired_lookup[("mechanism_diagnosis", "diag_full_context_graph_diag_qda", "diag_full_context_graph", "balanced_accuracy")]["ci_low"]) > 0.0,
        "exact_qda_gap_over_finite_context_graph_qda_ci": float(paired_lookup[("mechanism_diagnosis", "exact_diag_full_diag_qda_oracle", "diag_full_context_graph_diag_qda", "balanced_accuracy")]["ci_low"]) > 0.0,
    }

    identity = {
        "schema": SCHEMA,
        "config_sha256": base.sha256_file(config_path),
        "runner_sha256": base.sha256_file(Path(__file__).resolve()),
        "source_product_id": source_complete["product_id"],
        "source_step6a_benchmark_id": config["source_step6a"]["benchmark_id"],
    }
    benchmark_id = "closure_" + hashlib.sha256(base.canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve(); output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / benchmark_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing closure {output}")
    staging = output_parent / f".{benchmark_id}.staging-{uuid.uuid4().hex}"; staging.mkdir()
    base.write_csv(staging / "baseline_metrics.csv", metric_rows)
    base.write_csv(staging / "paired_differences.csv", paired_rows)
    base.write_csv(staging / "stratified_metrics.csv", stratified_rows)
    base.atomic_json(staging / "model_selection.json", selection)
    base.atomic_json(staging / "decision.json", {"schema": SCHEMA, "decision": "NONLINEAR_SANITY_CLOSURE_COMPLETE", "evidence_flags": flags, "adaptive_development_followup": True, "validation_used_for_model_or_threshold_selection": False, "historical_v0_1_test_accessed": False, "spent_confirmatory_cohort_accessed": False, "hardware_executed": False, "triqto_architecture_changed": False})
    payload: dict[str, np.ndarray] = {"validation_root_index": table["root_index"][validation_mask], "effect_truth": effect_truth, "mechanism_truth_all": mechanism_all_truth, "mechanism_loss_mask": mechanism_relative}
    for name, values in predictions.items():
        payload[f"effect__{name}__pred"] = values["effect_pred"]
        payload[f"effect__{name}__score"] = values["effect_score"]
        if "mechanism_pred_all" in values:
            payload[f"mechanism__{name}__pred_all"] = values["mechanism_pred_all"]
            payload[f"mechanism__{name}__scores_all"] = values["mechanism_scores_all"]
    np.savez_compressed(staging / "validation_predictions.npz", **payload)
    files = ["baseline_metrics.csv", "paired_differences.csv", "stratified_metrics.csv", "model_selection.json", "decision.json", "validation_predictions.npz"]
    completion = {"schema": SCHEMA, "status": "COMPLETE", "closure_id": benchmark_id, "identity": identity, "source_product_id": source_complete["product_id"], "source_step6a_benchmark_id": config["source_step6a"]["benchmark_id"], "adaptive_development_followup": True, "magnitude_variants": list(MAGNITUDE_VARIANTS), "qda_variants": list(QDA_VARIANTS), "historical_v0_1_test_accessed": False, "spent_confirmatory_cohort_accessed": False, "hardware_executed": False, "triqto_architecture_changed": False, "file_hashes": {name: base.sha256_file(staging / name) for name in files}}
    base.atomic_json(staging / "closure_complete.json", completion)
    os.replace(staging, output)
    print("\nTRIQTO STEP 6B NONLINEAR SANITY CLOSURE COMPLETE\n")
    print("Decision: NONLINEAR_SANITY_CLOSURE_COMPLETE")
    print(f"diag RMS effect BA: {metric_lookup[('effect_detection', 'diag_rms_threshold')]['balanced_accuracy']:.4f}")
    print(f"diag SNR proxy effect BA: {metric_lookup[('effect_detection', 'diag_snr_proxy_threshold')]['balanced_accuracy']:.4f}")
    print(f"diag_full diagonal-QDA mechanism BA: {metric_lookup[('mechanism_diagnosis', 'diag_full_diag_qda')]['balanced_accuracy']:.4f}")
    print(f"diag_full+context+graph diagonal-QDA mechanism BA: {metric_lookup[('mechanism_diagnosis', 'diag_full_context_graph_diag_qda')]['balanced_accuracy']:.4f}")
    print(f"exact diagnostic diagonal-QDA mechanism BA: {metric_lookup[('mechanism_diagnosis', 'exact_diag_full_diag_qda_oracle')]['balanced_accuracy']:.4f}")
    print("Adaptive development follow-up: YES")
    print("Historical v0.1 test accessed: NO")
    print("Spent confirmatory cohort accessed: NO")
    print("TriQTO architecture changed: NO")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
