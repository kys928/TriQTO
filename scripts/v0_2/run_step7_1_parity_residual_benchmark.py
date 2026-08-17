#!/usr/bin/env python3
"""Run the one-shot Step 7.1 parity-residual final development revision."""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
from triqto.step7_1.model import Step71DiagnosticModel


SCHEMA = "triqto.v0_2.step7_1_parity_residual_benchmark.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_1_parity_residual_benchmark.json"
DEFAULT_EXPERIMENT = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step7_1_parity_residual")
EXPECTED_PARAMETERS = 453_830


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--step7-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def instantiate_step71(
    variant: str,
    seed: int,
    experiment: Mapping[str, Any],
    device: torch.device,
) -> Step71DiagnosticModel:
    settings = experiment["model"]
    model = Step71DiagnosticModel(
        variant=variant,
        hidden_dim=int(settings["hidden_dim"]),
        graph_message_passing_layers=int(settings["graph_message_passing_layers"]),
        residual_mlp_layers=int(settings["residual_mlp_layers"]),
        dropout=float(settings["dropout"]),
        layer_norm_eps=float(settings["layer_norm_eps"]),
        initialization_seed=int(seed),
    ).to(device)
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if actual != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Step 7.1 parameter fairness failed for {variant}: {actual} != {EXPECTED_PARAMETERS}")
    return model


def train_variant_seed(**kwargs: Any):
    original = step7.instantiate_model
    step7.instantiate_model = instantiate_step71
    try:
        selected, history, selection_prediction, outer_prediction = step7.train_variant_seed(**kwargs)
    finally:
        step7.instantiate_model = original
    selected["trainable_parameter_count"] = EXPECTED_PARAMETERS
    return selected, history, selection_prediction, outer_prediction


def verify_frozen_contract(config: Mapping[str, Any], experiment_path: Path, experiment: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_STEP7_1_OUTCOME":
        raise RuntimeError("unexpected Step 7.1 frozen config schema/status")
    actual_experiment_hash = baseline.sha256_file(experiment_path)
    if actual_experiment_hash != config["source_step7"]["experiment_config_sha256"]:
        raise RuntimeError("Step 7 experiment config changed before Step 7.1")
    training = config["training"]
    inherited = experiment["training"]
    exact = {
        "learning_rate": float(inherited["learning_rate"]),
        "weight_decay": float(inherited["weight_decay"]),
        "root_batch_size": int(inherited["root_batch_size"]),
        "max_epochs": int(inherited["max_epochs"]),
        "early_stopping_patience": int(inherited["early_stopping_patience"]),
        "early_stopping_min_delta": float(inherited["early_stopping_min_delta"]),
        "gradient_clip_norm": float(inherited["gradient_clip_norm"]),
        "effect_loss_weight": float(inherited["effect_loss_weight"]),
        "mechanism_loss_weight": float(inherited["mechanism_loss_weight"]),
    }
    for key, value in exact.items():
        if float(training[key]) != float(value):
            raise RuntimeError(f"Step 7.1 training contract drift for {key}")
    if int(config["architecture"]["expected_trainable_parameters_each"]) != EXPECTED_PARAMETERS:
        raise RuntimeError("Step 7.1 expected parameter count changed")


def verify_step7_reference(step7_dir: Path, config: Mapping[str, Any], source_product_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray], float]:
    marker = baseline.read_json(step7_dir / "benchmark_complete.json")
    expected = config["source_step7"]
    if marker.get("benchmark_id") != expected["benchmark_id"]:
        raise RuntimeError("Step 7.1 source Step-7 benchmark ID mismatch")
    if marker.get("source_product_id") != source_product_id:
        raise RuntimeError("Step 7.1 source Step-7 product mismatch")
    for name, expected_hash in marker.get("file_hashes", {}).items():
        if baseline.sha256_file(step7_dir / name) != expected_hash:
            raise RuntimeError(f"Step 7 source result hash mismatch: {name}")
    with np.load(step7_dir / "validation_predictions.npz", allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    required = (
        "validation_source_index",
        "validation_root_index",
        "effect_truth",
        "mechanism_truth_all",
        "mechanism_loss_mask",
        "effect__late_concat__logit",
        "mechanism__late_concat__logits",
    )
    if any(key not in payload for key in required):
        raise RuntimeError("Step 7 reference predictions are missing the frozen late-concat payload")
    selection = baseline.read_json(step7_dir / "model_selection.json")
    threshold = float(selection["late_concat__ensemble"]["selection_effect_threshold"])
    return marker, payload, threshold


def prediction_from_frozen(payload: Mapping[str, np.ndarray]) -> step7.PredictionSet:
    return step7.PredictionSet(
        source_indices=np.asarray(payload["validation_source_index"], dtype=np.int64),
        root_indices=np.asarray(payload["validation_root_index"], dtype=np.int64),
        effect_truth=np.asarray(payload["effect_truth"], dtype=np.int8),
        mechanism_truth_all=np.asarray(payload["mechanism_truth_all"], dtype=np.int8),
        mechanism_mask=np.asarray(payload["mechanism_loss_mask"], dtype=bool),
        effect_logits=np.asarray(payload["effect__late_concat__logit"], dtype=np.float32),
        mechanism_logits=np.asarray(payload["mechanism__late_concat__logits"], dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    experiment_path = args.experiment_config.expanduser().resolve()
    config = baseline.read_json(config_path)
    experiment = baseline.read_json(experiment_path)
    verify_frozen_contract(config, experiment_path, experiment)

    product = (args.product_dir or Path(config["source_dataset"]["default_product_dir"])).expanduser().resolve()
    step7_dir = (args.step7_dir or Path(config["source_step7"]["default_benchmark_dir"])).expanduser().resolve()
    output_parent = args.output_parent.expanduser().resolve()
    device = step7.resolve_device(args.device)

    source_complete, rows = baseline.verify_source_product(product, experiment)
    if source_complete["product_id"] != config["source_dataset"]["product_id"]:
        raise RuntimeError("Step 7.1 source product mismatch")
    step7_marker, frozen_payload, frozen_threshold = verify_step7_reference(step7_dir, config, source_complete["product_id"])

    fit_roots, selection_roots, outer_roots, by_root = step7.split_root_indices(rows, experiment)
    split = config["development_split"]
    if (len(fit_roots), len(selection_roots), len(outer_roots)) != (
        int(split["fit_roots"]), int(split["internal_selection_roots"]), int(split["outer_development_validation_roots"])
    ):
        raise RuntimeError("Step 7.1 inherited root partitions changed")

    batch_size = int(config["training"]["root_batch_size"])
    print("Materializing and SHA-verifying frozen Step 5 artifacts exactly once for Step 7.1...", flush=True)
    fit_blocks = step7.materialize_blocks(product=product, rows=rows, by_root=by_root, roots=fit_roots, root_batch_size=batch_size, label="fit", progress_every=args.progress_every)
    selection_blocks = step7.materialize_blocks(product=product, rows=rows, by_root=by_root, roots=selection_roots, root_batch_size=batch_size, label="selection", progress_every=args.progress_every)
    outer_blocks = step7.materialize_blocks(product=product, rows=rows, by_root=by_root, roots=outer_roots, root_batch_size=batch_size, label="outer-validation", progress_every=args.progress_every)
    effect_class_weight, mechanism_class_weight = step7.class_weights(fit_blocks)
    print(f"Step 7.1 training device: {device}", flush=True)

    primary = [str(value) for value in config["variants"]["primary"]]
    primary_seeds = [int(value) for value in config["variants"]["primary_seeds"]]
    ablations = [str(value) for value in config["variants"]["diagnostic_ablations"]]
    ablation_seeds = [int(value) for value in config["variants"]["ablation_seeds"]]
    if len(primary) * len(primary_seeds) + len(ablations) * len(ablation_seeds) != int(config["variants"]["expected_model_runs"]):
        raise RuntimeError("Step 7.1 run count changed")

    selections: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    selection_predictions: dict[tuple[str, int], step7.PredictionSet] = {}
    outer_predictions: dict[tuple[str, int], step7.PredictionSet] = {}
    reference_outer: step7.PredictionSet | None = None
    for variant in primary + ablations:
        seeds = primary_seeds if variant in primary else ablation_seeds
        for seed in seeds:
            print(f"\nTraining Step 7.1 variant={variant} seed={seed}", flush=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            selected, history, selection_prediction, outer_prediction = train_variant_seed(
                variant=variant,
                seed=seed,
                experiment=experiment,
                fit_blocks=fit_blocks,
                selection_blocks=selection_blocks,
                outer_blocks=outer_blocks,
                effect_class_weight=effect_class_weight,
                mechanism_class_weight=mechanism_class_weight,
                device=device,
            )
            selections[f"{variant}__seed{seed}"] = selected
            histories.extend(history)
            selection_predictions[(variant, seed)] = selection_prediction
            outer_predictions[(variant, seed)] = outer_prediction
            if reference_outer is None:
                reference_outer = outer_prediction
            else:
                step7.assert_prediction_alignment(reference_outer, outer_prediction, f"Step 7.1 {variant} seed {seed}")
    assert reference_outer is not None
    frozen_outer = prediction_from_frozen(frozen_payload)
    step7.assert_prediction_alignment(reference_outer, frozen_outer, "frozen Step-7 late_concat")

    replicates = int(config["evaluation"]["clean_root_bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    confidence = float(config["evaluation"]["confidence_level"])
    aggregate_metrics: list[dict[str, Any]] = []
    seed_metrics: list[dict[str, Any]] = []
    ablation_metrics: list[dict[str, Any]] = []
    stratified_metrics: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    primary_boot: dict[str, dict[str, Mapping[str, np.ndarray]]] = {"effect_detection": {}, "mechanism_diagnosis": {}, "integrated_diagnosis": {}}
    aggregate_payload: dict[str, dict[str, np.ndarray | float]] = {}

    for variant in primary:
        selection_set = selection_predictions[(variant, primary_seeds[0])]
        outer_set = outer_predictions[(variant, primary_seeds[0])]
        for seed in primary_seeds[1:]:
            step7.assert_prediction_alignment(selection_set, selection_predictions[(variant, seed)], f"selection {variant} seed {seed}")
            step7.assert_prediction_alignment(outer_set, outer_predictions[(variant, seed)], f"outer {variant} seed {seed}")
        selection_effect_logits = np.mean(np.stack([selection_predictions[(variant, seed)].effect_logits for seed in primary_seeds]), axis=0)
        outer_effect_logits = np.mean(np.stack([outer_predictions[(variant, seed)].effect_logits for seed in primary_seeds]), axis=0)
        outer_mechanism_logits = np.mean(np.stack([outer_predictions[(variant, seed)].mechanism_logits for seed in primary_seeds]), axis=0)
        ensemble_threshold, _, _ = step7._metrics_binary(selection_set.effect_truth, selection_effect_logits)
        selections[f"{variant}__ensemble"] = {"selection_effect_threshold": ensemble_threshold, "aggregation": "mean_logits_across_frozen_primary_seeds", "seeds": primary_seeds}
        metric_rows, boots, predictions = step7.bootstrap_rows(
            name=variant,
            truth=outer_set,
            effect_logits=outer_effect_logits,
            mechanism_logits=outer_mechanism_logits,
            threshold=ensemble_threshold,
            replicates=replicates,
            seed=bootstrap_seed,
            confidence=confidence,
        )
        aggregate_metrics.extend(metric_rows)
        for task, boot in boots.items(): primary_boot[task][variant] = boot
        aggregate_payload[variant] = {
            "effect_logits": outer_effect_logits.astype(np.float32),
            "mechanism_logits": outer_mechanism_logits.astype(np.float32),
            "effect_pred": predictions["effect_pred"].astype(np.int8),
            "mechanism_pred_all": predictions["mechanism_pred_all"].astype(np.int8),
            "threshold": float(ensemble_threshold),
        }
        stratified_metrics.extend(step7.stratified_metric_rows(
            name=variant,
            outer=outer_set,
            rows=rows,
            effect_pred=predictions["effect_pred"],
            mechanism_pred_all=predictions["mechanism_pred_all"],
            strata=experiment["evaluation"]["strata"],
            minimum=int(experiment["evaluation"]["minimum_stratum_examples"]),
        ))
        for seed in primary_seeds:
            threshold = float(selections[f"{variant}__seed{seed}"]["selection_effect_threshold"])
            seed_metrics.extend(step7.simple_metric_rows(
                prefix=f"{variant}__seed{seed}",
                truth=outer_predictions[(variant, seed)],
                effect_logits=outer_predictions[(variant, seed)].effect_logits,
                mechanism_logits=outer_predictions[(variant, seed)].mechanism_logits,
                threshold=threshold,
            ))

    pair = [tuple(config["evaluation"]["primary_paired_comparison"])]
    for task, boot in primary_boot.items():
        paired_rows.extend(baseline.paired_difference_rows(task, pair, boot, confidence))

    frozen_rows, frozen_boots, _ = step7.bootstrap_rows(
        name="frozen_step7_late_concat",
        truth=frozen_outer,
        effect_logits=frozen_outer.effect_logits,
        mechanism_logits=frozen_outer.mechanism_logits,
        threshold=frozen_threshold,
        replicates=replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    aggregate_metrics.extend(frozen_rows)
    for task in ("effect_detection", "mechanism_diagnosis", "integrated_diagnosis"):
        comparison_boot = dict(primary_boot[task])
        comparison_boot["frozen_step7_late_concat"] = frozen_boots[task]
        paired_rows.extend(baseline.paired_difference_rows(
            task,
            [("late_concat", "frozen_step7_late_concat"), ("late_concat_parity_residual", "frozen_step7_late_concat")],
            comparison_boot,
            confidence,
        ))

    seed = ablation_seeds[0]
    champion_seed = outer_predictions[("late_concat", seed)]
    champion_threshold = float(selections[f"late_concat__seed{seed}"]["selection_effect_threshold"])
    champion_rows, champion_boots, _ = step7.bootstrap_rows(
        name=f"late_concat__seed{seed}", truth=champion_seed,
        effect_logits=champion_seed.effect_logits, mechanism_logits=champion_seed.mechanism_logits,
        threshold=champion_threshold, replicates=replicates, seed=bootstrap_seed, confidence=confidence,
    )
    ablation_metrics.extend(champion_rows)
    for variant in ablations:
        outer_set = outer_predictions[(variant, seed)]
        threshold = float(selections[f"{variant}__seed{seed}"]["selection_effect_threshold"])
        rows_out, boots, _ = step7.bootstrap_rows(
            name=f"{variant}__seed{seed}", truth=outer_set,
            effect_logits=outer_set.effect_logits, mechanism_logits=outer_set.mechanism_logits,
            threshold=threshold, replicates=replicates, seed=bootstrap_seed, confidence=confidence,
        )
        ablation_metrics.extend(rows_out)
        for task in champion_boots:
            comparison = {f"late_concat__seed{seed}": champion_boots[task], f"{variant}__seed{seed}": boots[task]}
            paired_rows.extend(baseline.paired_difference_rows(task, [(f"late_concat__seed{seed}", f"{variant}__seed{seed}")], comparison, confidence))

    lookup = {(row["task"], row["left"], row["right"], row["metric"]): row for row in paired_rows}
    architecture = lookup[("mechanism_diagnosis", "late_concat_parity_residual", "late_concat", "balanced_accuracy")]
    effect = lookup[("effect_detection", "late_concat_parity_residual", "late_concat", "balanced_accuracy")]
    frozen_repro = lookup[("mechanism_diagnosis", "late_concat", "frozen_step7_late_concat", "balanced_accuracy")]
    reproducibility_tolerance = float(config["evaluation"]["champion_rerun_reproducibility_tolerance_ba"])
    candidate_effect_margin = float(config["evaluation"]["candidate_effect_noninferiority_margin_ba"])
    reproducible = abs(float(frozen_repro["difference"])) <= reproducibility_tolerance
    architecture_positive = float(architecture["ci_low"]) > 0.0
    effect_noninferior = float(effect["ci_low"]) >= -candidate_effect_margin
    candidate_wins = reproducible and architecture_positive and effect_noninferior
    selected_final = "late_concat_parity_residual" if candidate_wins else "late_concat"
    flags = {
        "champion_rerun_reproducible": reproducible,
        "candidate_mechanism_architecture_signal": architecture_positive,
        "candidate_effect_noninferior": effect_noninferior,
        "candidate_wins_final_revision": candidate_wins,
        "architecture_search_stops_after_step7_1": True,
        "new_confirmatory_cohort_required": True,
    }

    identity = {
        "schema": SCHEMA,
        "config_sha256": baseline.sha256_file(config_path),
        "experiment_config_sha256": baseline.sha256_file(experiment_path),
        "runner_sha256": baseline.sha256_file(Path(__file__).resolve()),
        "source_product_id": source_complete["product_id"],
        "source_step7_benchmark_id": step7_marker["benchmark_id"],
    }
    benchmark_id = "benchmark_" + hashlib.sha256(baseline.canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / benchmark_id
    if output.exists(): raise RuntimeError(f"refusing to overwrite Step 7.1 benchmark {output}")
    staging = output_parent / f".{benchmark_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    baseline.write_csv(staging / "aggregate_metrics.csv", aggregate_metrics)
    baseline.write_csv(staging / "seed_metrics.csv", seed_metrics)
    baseline.write_csv(staging / "paired_differences.csv", paired_rows)
    baseline.write_csv(staging / "ablation_metrics.csv", ablation_metrics)
    baseline.write_csv(staging / "stratified_metrics.csv", stratified_metrics)
    baseline.write_csv(staging / "training_history.csv", histories)
    baseline.atomic_json(staging / "model_selection.json", selections)
    baseline.atomic_json(staging / "decision.json", {
        "schema": SCHEMA,
        "decision": config["decision"]["completion_decision"],
        "evidence_status": config["evidence_status"],
        "interpretation_flags": flags,
        "selected_final_development_architecture": selected_final,
        "stop_rule": config["decision"]["stop_rule"],
        "outer_validation_used_for_selection": False,
        "historical_v0_1_test_accessed": False,
        "spent_confirmatory_cohort_accessed": False,
        "new_confirmatory_cohort_accessed": False,
        "hardware_executed": False,
        "confirmation_unlocked_automatically": False,
    })
    payload: dict[str, np.ndarray] = {
        "validation_source_index": reference_outer.source_indices.astype(np.int64),
        "validation_root_index": reference_outer.root_indices.astype(np.int64),
        "effect_truth": reference_outer.effect_truth.astype(np.int8),
        "mechanism_truth_all": reference_outer.mechanism_truth_all.astype(np.int8),
        "mechanism_loss_mask": reference_outer.mechanism_mask.astype(bool),
    }
    for variant, values in aggregate_payload.items():
        payload[f"effect__{variant}__logit"] = np.asarray(values["effect_logits"])
        payload[f"effect__{variant}__pred"] = np.asarray(values["effect_pred"])
        payload[f"mechanism__{variant}__logits"] = np.asarray(values["mechanism_logits"])
        payload[f"mechanism__{variant}__pred_all"] = np.asarray(values["mechanism_pred_all"])
    for (variant, seed_value), prediction in outer_predictions.items():
        payload[f"effect__{variant}__seed{seed_value}__logit"] = prediction.effect_logits.astype(np.float32)
        payload[f"mechanism__{variant}__seed{seed_value}__logits"] = prediction.mechanism_logits.astype(np.float32)
    np.savez_compressed(staging / "validation_predictions.npz", **payload)
    files = ["aggregate_metrics.csv", "seed_metrics.csv", "paired_differences.csv", "ablation_metrics.csv", "stratified_metrics.csv", "training_history.csv", "model_selection.json", "decision.json", "validation_predictions.npz"]
    completion = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "benchmark_id": benchmark_id,
        "identity": identity,
        "source_product_id": source_complete["product_id"],
        "source_step7_benchmark_id": step7_marker["benchmark_id"],
        "fit_roots": len(fit_roots),
        "selection_roots": len(selection_roots),
        "outer_development_validation_roots": len(outer_roots),
        "model_runs": int(config["variants"]["expected_model_runs"]),
        "primary_variants": primary,
        "primary_seeds": primary_seeds,
        "ablation_variants": ablations,
        "ablation_seeds": ablation_seeds,
        "selected_final_development_architecture": selected_final,
        "interpretation_flags": flags,
        "evidence_status": config["evidence_status"],
        "historical_v0_1_test_accessed": False,
        "spent_confirmatory_cohort_accessed": False,
        "new_confirmatory_cohort_accessed": False,
        "hardware_executed": False,
        "confirmation_unlocked_automatically": False,
        "file_hashes": {name: baseline.sha256_file(staging / name) for name in files},
    }
    baseline.atomic_json(staging / "benchmark_complete.json", completion)
    os.replace(staging, output)

    metric_lookup = {(row["task"], row["baseline"]): row for row in aggregate_metrics}
    print("\nTRIQTO STEP 7.1 FINAL PARITY-RESIDUAL REVISION COMPLETE\n")
    print(f"Decision: {config['decision']['completion_decision']}")
    for variant in primary:
        print(
            f"{variant}: effect_BA={float(metric_lookup[('effect_detection', variant)]['balanced_accuracy']):.4f} "
            f"mechanism_BA={float(metric_lookup[('mechanism_diagnosis', variant)]['balanced_accuracy']):.4f} "
            f"integrated_BA={float(metric_lookup[('integrated_diagnosis', variant)]['balanced_accuracy']):.4f}"
        )
    print(f"Champion rerun reproducible: {'YES' if reproducible else 'NO'}")
    print(f"Parity-residual mechanism architecture signal: {'YES' if architecture_positive else 'NO'}")
    print(f"Parity-residual effect noninferior: {'YES' if effect_noninferior else 'NO'}")
    print(f"Selected final development architecture: {selected_final}")
    print("Architecture search stops after Step 7.1: YES")
    print("Evidence status: ADAPTIVE DEVELOPMENT, NOT CONFIRMATORY")
    print("New confirmatory cohort accessed: NO")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
