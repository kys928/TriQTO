#!/usr/bin/env python3
"""Strict FIT-only entrypoint for Step-14 equivalence-aware development.

Unlike the generic Step-14 dataset verifier, this entrypoint never exposes a
selection row's mechanism field to the scientific code path. Combined manifest
files are integrity-hashed, then rows are gated on ``step14_partition`` before
any label or privileged root field is accessed. Selection artifacts are never
opened. True affected-qubit/injection-boundary fields are never accessed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import analyze_step14_latent_frame_inference as latent
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import fit_step14_equivalence_aware_latent_frame as base
import run_step14_cross_motif_training as step14
import run_step14_oracle_raw_evidence_ceiling as oracle_compat
import step14_equivalence_aware_common as equiv


def _strict_fit_manifest_views(product: Path, complete: dict[str, Any]) -> tuple[list[dict[str, str]], dict[int, dict[str, str]]]:
    """Return only FIT scientific fields, refusing all non-development partitions.

    The partition field is the sole field inspected before a row is accepted.
    For selection rows, no mechanism, artifact path, family id, affected qubit,
    injection boundary, or operation signature field is accessed.
    """
    manifests = product / "manifests"
    for name, wanted in complete["manifest_hashes"].items():
        if baseline.sha256_file(manifests / name) != str(wanted):
            raise RuntimeError(f"Step-14 manifest hash mismatch: {name}")

    fit_rows: list[dict[str, str]] = []
    partition_counts: dict[str, int] = {}
    with (manifests / "example_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "step14_partition", "mechanism", "root_index", "artifact_path",
            "artifact_sha256", "example_id", "family_id",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError("Step-14 example manifest schema drift")
        for row in reader:
            partition = str(row.get("step14_partition", ""))
            partition_counts[partition] = partition_counts.get(partition, 0) + 1
            if partition == "selection":
                # Deliberately do not access any scientific field on this row.
                continue
            if partition != "fit":
                raise RuntimeError(f"outer/reserve row present in development manifest: {partition!r}")
            fit_rows.append({
                "step14_partition": "fit",
                "mechanism": str(row["mechanism"]),
                "root_index": str(row["root_index"]),
                "artifact_path": str(row["artifact_path"]),
                "artifact_sha256": str(row["artifact_sha256"]),
                "example_id": str(row["example_id"]),
                "family_id": str(row["family_id"]),
            })

    safe_roots: dict[int, dict[str, str]] = {}
    root_partition_counts: dict[str, int] = {}
    with (manifests / "root_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step14_partition", "root_index", "operation_signature"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError("Step-14 root manifest schema drift")
        for row in reader:
            partition = str(row.get("step14_partition", ""))
            root_partition_counts[partition] = root_partition_counts.get(partition, 0) + 1
            if partition == "selection":
                # Do not touch affected_qubit/injection_boundary or signature.
                continue
            if partition != "fit":
                raise RuntimeError(f"outer/reserve root present in development manifest: {partition!r}")
            root_index = int(row["root_index"])
            if root_index in safe_roots:
                raise RuntimeError(f"duplicate FIT root index {root_index}")
            safe_roots[root_index] = {
                "step14_partition": "fit",
                "operation_signature": str(row["operation_signature"]),
            }

    if partition_counts != {"fit": 31200, "selection": 7800}:
        raise RuntimeError(f"development example partition count drift: {partition_counts}")
    if root_partition_counts != {"fit": 2400, "selection": 600}:
        raise RuntimeError(f"development root partition count drift: {root_partition_counts}")
    if len(fit_rows) != 31200 or len(safe_roots) != 2400:
        raise RuntimeError("strict FIT projection count drift")
    return fit_rows, safe_roots


def main() -> None:
    args = base.parse_args()
    oracle_compat.apply_frozen_support_bound()
    dev = base.load_development_protocol()
    frozen = dev["source_freeze"]

    if args.training_run_id != str(frozen["training_run_id"]):
        raise RuntimeError("training run differs from frozen equivalence-aware protocol")
    if args.selection_freeze_sha256 != str(frozen["selection_freeze_sha256"]):
        raise RuntimeError("selection-freeze identity differs from frozen protocol")

    rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)
    step14_protocol = step14.read_json(base.STEP14_CONFIG)
    step14.assert_contract(step14_protocol)
    latent_cfg = latent.load_frozen_config()
    if baseline.sha256_file(base.LATENT_CONFIG) != str(frozen["latent_protocol_sha256"]):
        raise RuntimeError("frozen latent protocol hash drift")

    cross_product = step14.resolve_cross_product(None)
    complete = base.read_json(cross_product / "dataset_complete.json")
    if complete.get("schema") != "triqto.v0_2.step14_cross_motif_dataset.v1":
        raise RuntimeError("unexpected Step-14 development dataset schema")
    if complete.get("status") != "COMPLETE_FROZEN_DEVELOPMENT":
        raise RuntimeError("Step-14 development product is incomplete")
    if bool(complete.get("model_evaluated_before_freeze", True)):
        raise RuntimeError("development product violated pre-freeze model boundary")
    if bool(complete.get("qpu_executed", True)) or bool(complete.get("future_hardware_reserve_materialized", True)):
        raise RuntimeError("development product violated QPU/reserve boundary")
    if str(complete["product_id"]) != str(frozen["development_product_id"]):
        raise RuntimeError("development product id drift")
    if baseline.sha256_file(cross_product / "dataset_complete.json") != str(frozen["development_dataset_sha256"]):
        raise RuntimeError("development dataset hash drift")

    fit_rows, safe_roots = _strict_fit_manifest_views(cross_product, complete)
    table = base.extract_fit_table(cross_product, fit_rows, safe_roots, latent_cfg, args.progress_every)
    x = table["candidate_features"]
    mask = table["candidate_mask"]
    profiles = table["profile_scores"]
    distance = table["pair_distance"]
    gate = table["pair_gate"]
    y = table["truth"]
    families = table["family"]
    if len(y) != 28800:
        raise RuntimeError(f"expected 28800 distorted FIT examples, got {len(y)}")

    device = base.resolve_device(args.device)
    hp = dev["fit_only_hyperparameter_selection"]
    fold_count = int(hp["family_level_folds"])
    folds = np.asarray([equiv.family_fold(str(value), fold_count) for value in families], dtype=np.int64)
    if set(folds.tolist()) != set(range(fold_count)):
        raise RuntimeError("FIT-family cross-validation folds are incomplete")

    crossfit_candidate_logits = np.zeros((len(y), x.shape[1], 3), dtype=np.float64)
    crossfit_uniform_logits = np.zeros((len(y), 3), dtype=np.float64)
    cv_training: dict[str, Any] = {}
    for fold in range(fold_count):
        train_idx = np.flatnonzero(folds != fold)
        eval_idx = np.flatnonzero(folds == fold)
        if set(families[train_idx].tolist()) & set(families[eval_idx].tolist()):
            raise RuntimeError("family leakage across internal FIT CV")
        seed_candidates: list[np.ndarray] = []
        seed_uniform: list[np.ndarray] = []
        fold_records: dict[str, Any] = {}
        for seed in [int(v) for v in hp["network_seeds_per_fold"]]:
            trained = equiv.train_uniform_candidate_network(
                x[train_idx], mask[train_idx], y[train_idx], x[eval_idx], mask[eval_idx],
                seed=seed, latent_cfg=latent_cfg, device=device,
            )
            seed_candidates.append(trained["eval_candidate_logits"])
            seed_uniform.append(trained["eval_uniform_pooled_logits"])
            fold_records[str(seed)] = base.public_training_record(trained)
            print(f"equivalence CV fold={fold} seed={seed} train_BA={trained['train_balanced_accuracy']:.4f}", flush=True)
        crossfit_candidate_logits[eval_idx] = np.mean(np.stack(seed_candidates), axis=0)
        crossfit_uniform_logits[eval_idx] = np.mean(np.stack(seed_uniform), axis=0)
        cv_training[str(fold)] = {
            "train_family_count": int(len(set(families[train_idx].tolist()))),
            "eval_family_count": int(len(set(families[eval_idx].tolist()))),
            "seeds": fold_records,
        }

    grid_records: list[dict[str, Any]] = []
    for tau in [float(v) for v in hp["tau_grid"]]:
        for temperature in [float(v) for v in hp["frame_temperature_grid"]]:
            logits = equiv.equivalence_aware_pool_batch(
                crossfit_candidate_logits, profiles, distance, gate, mask,
                tau=tau, frame_temperature=temperature,
            )
            record = {"tau": tau, "frame_temperature": temperature, **base.metric(y, logits)}
            grid_records.append(record)
            print(
                f"equivalence fit-CV tau={tau:g} T={temperature:g} "
                f"BA={record['mechanism_balanced_accuracy']:.6f} "
                f"min_recall={record['minimum_mechanism_recall']:.6f}", flush=True,
            )

    selected = min(grid_records, key=base.hp_rank)
    selected_tau = float(selected["tau"])
    selected_temperature = float(selected["frame_temperature"])
    uniform_crossfit = base.metric(y, crossfit_uniform_logits)

    protocol_sha = baseline.sha256_file(base.DEV_CONFIG)
    method_key = hashlib.sha256(
        (protocol_sha + str(frozen["development_dataset_sha256"]) + str(selected_tau) + str(selected_temperature)).encode("utf-8")
    ).hexdigest()[:24]
    method_id = f"equiv_fit_{method_key}"
    output_dir = args.output_parent / method_id
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite existing method bundle: {output_dir}")
        output_dir.rmdir()
    output_dir.mkdir(parents=True, exist_ok=False)

    final_training: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    fit_candidate_logits: list[np.ndarray] = []
    for seed in [int(v) for v in dev["final_fit_bundle"]["seeds"]]:
        trained = equiv.train_uniform_candidate_network(
            x, mask, y, x, mask, seed=seed, latent_cfg=latent_cfg, device=device,
        )
        fit_candidate_logits.append(trained["eval_candidate_logits"])
        checkpoint_name = f"candidate_network_seed{seed}.pt"
        checkpoint_path = output_dir / checkpoint_name
        torch.save({
            "schema": "triqto.v0_2.step14_equivalence_candidate_network.v1",
            "method_id": method_id,
            "seed": seed,
            "state_dict": trained["state_dict"],
            "normalization_mean": trained["mean"],
            "normalization_std": trained["std"],
            "selected_tau": selected_tau,
            "selected_frame_temperature": selected_temperature,
            "development_protocol_sha256": protocol_sha,
            "latent_protocol_sha256": baseline.sha256_file(base.LATENT_CONFIG),
            "training_run_id": args.training_run_id,
            "selection_freeze_sha256": args.selection_freeze_sha256,
            "fit_only": True,
            "selection_label_field_accessed": False,
            "location_supervision": False,
            "main_model_weights_updated": False,
        }, checkpoint_path)
        checkpoint_hashes[checkpoint_name] = baseline.sha256_file(checkpoint_path)
        final_training[str(seed)] = base.public_training_record(trained)
        print(f"equivalence final fit seed={seed} train_BA={trained['train_balanced_accuracy']:.4f}", flush=True)

    final_candidate_ensemble = np.mean(np.stack(fit_candidate_logits), axis=0)
    final_fit_logits = equiv.equivalence_aware_pool_batch(
        final_candidate_ensemble, profiles, distance, gate, mask,
        tau=selected_tau, frame_temperature=selected_temperature,
    )

    bundle = {
        "schema": "triqto.v0_2.step14_equivalence_aware_method_bundle.v1",
        "status": "FIT_ONLY_METHOD_BUNDLE_READY_FOR_FREEZE",
        "method_id": method_id,
        "development_protocol_sha256": protocol_sha,
        "source_training_run_id": args.training_run_id,
        "source_selection_freeze_sha256": args.selection_freeze_sha256,
        "selected_hyperparameters": {"tau": selected_tau, "frame_temperature": selected_temperature},
        "candidate_network_seeds": [int(v) for v in dev["final_fit_bundle"]["seeds"]],
        "checkpoint_sha256": checkpoint_hashes,
        "fit_only": True,
        "selection_artifacts_loaded": False,
        "selection_label_field_accessed": False,
        "selection_metrics_used": False,
        "true_location_used": False,
        "exact_statevector_geometry_used_in_deployable_similarity": False,
        "main_model_retrained": False,
        "main_model_weights_updated": False,
        "fresh_holdout_generated": False,
    }
    bundle_path = output_dir / "method_bundle.json"
    base.atomic_json(bundle_path, bundle)
    bundle_sha = baseline.sha256_file(bundle_path)

    result = {
        "schema": base.SCHEMA,
        "status": "COMPLETE_FIT_ONLY_EQUIVALENCE_AWARE_DEVELOPMENT",
        "method_id": method_id,
        "identity": {
            "development_protocol_sha256": protocol_sha,
            "latent_protocol_sha256": baseline.sha256_file(base.LATENT_CONFIG),
            "training_run_id": args.training_run_id,
            "selection_freeze_sha256": args.selection_freeze_sha256,
            "development_product_id": str(frozen["development_product_id"]),
            "development_dataset_sha256": str(frozen["development_dataset_sha256"]),
            "source_ambiguity_result_sha256": str(frozen["source_ambiguity_result_sha256"]),
        },
        "data_boundary": {
            "fit_target_examples_loaded": int(table["loaded_target_examples"]),
            "fit_roots_loaded": int(table["loaded_roots"]),
            "fit_family_count": int(len(set(families.tolist()))),
            "selection_artifacts_loaded": False,
            "selection_label_field_accessed": False,
            "selection_metrics_used": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
        },
        "location_privilege_boundary": {
            "true_location_field_accessed": False,
            "location_supervision": False,
            "exact_statevector_geometry_used_in_deployable_similarity": False,
        },
        "candidate_count_per_example": table["candidate_count_summary"],
        "fit_family_cross_validation": {
            "fold_count": fold_count,
            "uniform_latent_reference": uniform_crossfit,
            "grid": grid_records,
            "selected": selected,
            "training": cv_training,
        },
        "final_fit": {
            "equivalence_aware_metric": base.metric(y, final_fit_logits),
            "training": final_training,
            "checkpoint_sha256": checkpoint_hashes,
        },
        "method_bundle_sha256": bundle_sha,
        "fresh_holdout_generated": False,
        "main_model_retrained": False,
        "main_model_weights_updated": False,
    }
    result_path = output_dir / "fit_development_result.json"
    base.atomic_json(result_path, result)
    result_sha = baseline.sha256_file(result_path)

    complete_payload = {
        "schema": "triqto.v0_2.step14_equivalence_aware_fit_development_complete.v1",
        "status": "COMPLETE_FIT_ONLY_EQUIVALENCE_AWARE_DEVELOPMENT",
        "method_id": method_id,
        "method_dir": str(output_dir),
        "method_bundle_sha256": bundle_sha,
        "fit_development_result_sha256": result_sha,
        "checkpoint_sha256": checkpoint_hashes,
        "selected_hyperparameters": {"tau": selected_tau, "frame_temperature": selected_temperature},
        "selection_artifacts_loaded": False,
        "selection_label_field_accessed": False,
        "fresh_holdout_generated": False,
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    }
    base.atomic_json(output_dir / "fit_complete.json", complete_payload)
    complete_sha = baseline.sha256_file(output_dir / "fit_complete.json")
    base.atomic_json(args.output_parent / "current_equivalence_aware_fit.json", {
        "schema": "triqto.v0_2.step14_equivalence_aware_fit_pointer.v1",
        "method_id": method_id,
        "method_dir": str(output_dir),
        "fit_development_result_sha256": result_sha,
        "fit_complete_sha256": complete_sha,
        "method_bundle_sha256": bundle_sha,
    })
    print(json.dumps({
        "status": "COMPLETE_FIT_ONLY_EQUIVALENCE_AWARE_DEVELOPMENT",
        "method_id": method_id,
        "selected_tau": selected_tau,
        "selected_frame_temperature": selected_temperature,
        "fit_development_result_sha256": result_sha,
        "method_bundle_sha256": bundle_sha,
        "fit_complete_sha256": complete_sha,
        "checkpoint_sha256": checkpoint_hashes,
        "selection_artifacts_loaded": False,
        "selection_label_field_accessed": False,
        "fresh_holdout_generated": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
