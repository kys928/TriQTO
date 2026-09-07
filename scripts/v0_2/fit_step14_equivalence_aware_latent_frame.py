#!/usr/bin/env python3
"""Fit-only Step-14 equivalence-aware latent-frame development.

This script is deliberately incapable of evaluating the spent Step-14 selection
partition. It uses only fit-family mechanism labels to cross-fit the pooling
hyperparameters, then trains the frozen candidate-network architecture on all
fit families and persists a bundle that must be frozen before any fresh holdout
is generated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import analyze_step14_latent_frame_inference as latent
import analyze_step14_local_frame_canonicalization as frame
import analyze_step14_oracle_raw_evidence_ceiling as oracle
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import run_step14_cross_motif_training as step14
import run_step14_oracle_raw_evidence_ceiling as oracle_compat
import step14_equivalence_aware_common as equiv

ROOT = Path(__file__).resolve().parents[2]
STEP14_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
LATENT_CONFIG = ROOT / "configs/v0_2/step14_latent_frame_inference.json"
DEV_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_latent_frame_development.json"
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_development")
SCHEMA = "triqto.v0_2.step14_equivalence_aware_fit_development_result.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--selection-freeze-sha256", required=True)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_development_protocol() -> dict[str, Any]:
    value = read_json(DEV_CONFIG)
    if value.get("schema") != "triqto.v0_2.step14_equivalence_aware_latent_frame_development.v1":
        raise RuntimeError("unexpected equivalence-aware development schema")
    if value.get("status") != "FROZEN_BEFORE_FIT_ONLY_DEVELOPMENT":
        raise RuntimeError("equivalence-aware development protocol is not frozen")
    if value["development_data_boundary"].get("allowed_partition") != "fit":
        raise RuntimeError("development protocol is not fit-only")
    if not bool(value["fresh_holdout_precommitment"]["holdout_may_not_be_generated_before_final_method_freeze"]):
        raise RuntimeError("fresh holdout precommitment drift")
    return value


def pad_profiles(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    output = np.full((len(values), max_candidates, 3), -np.inf, dtype=np.float64)
    for i, value in enumerate(values):
        count = value.shape[0]
        output[i, :count] = value
    return output


def pad_pair_float(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    output = np.ones((len(values), max_candidates, max_candidates), dtype=np.float32)
    for i, value in enumerate(values):
        count = value.shape[0]
        output[i, :count, :count] = value
    return output


def pad_pair_bool(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    output = np.zeros((len(values), max_candidates, max_candidates), dtype=np.bool_)
    for i, value in enumerate(values):
        count = value.shape[0]
        output[i, :count, :count] = value
    return output


def extract_fit_table(
    product: Path,
    fit_rows: Sequence[Mapping[str, str]],
    safe_roots: Mapping[int, Mapping[str, str]],
    latent_cfg: Mapping[str, Any],
    progress_every: int,
) -> dict[str, Any]:
    selected = [row for row in fit_rows if str(row["mechanism"]) in frame.TARGET]
    if any(str(row.get("step14_partition")) != "fit" for row in selected):
        raise RuntimeError("non-fit row reached fit-only extractor")

    cache: dict[int, dict[str, Any]] = {}
    candidate_features: list[np.ndarray] = []
    profile_scores: list[np.ndarray] = []
    pair_distances: list[np.ndarray] = []
    pair_gates: list[np.ndarray] = []
    truth: list[int] = []
    families: list[str] = []
    candidate_counts: list[int] = []

    for position, row in enumerate(selected, start=1):
        if str(row["step14_partition"]) != "fit":
            raise RuntimeError("selection/outer/reserve artifact load refused")
        root_index = int(row["root_index"])
        root = safe_roots.get(root_index)
        if root is None or str(root["step14_partition"]) != "fit":
            raise RuntimeError(f"fit root missing or unsafe: {root_index}")

        artifact = product / str(row["artifact_path"])
        if baseline.sha256_file(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"artifact hash mismatch for {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as source:
            loaded = {key: source[key] for key in source.files}
        forbidden = [
            key for key in loaded
            if key.startswith("x__")
            and any(
                token in key.lower()
                for token in ("mechanism_target", "effect_target", "affected_qubit", "injection_boundary")
            )
        ]
        if forbidden:
            raise RuntimeError(f"privileged location/target leaked into x__ inputs: {forbidden}")

        delta, weights, pairs = frame.measured_delta_and_weights(loaded)
        if root_index not in cache:
            clean = frame.circuit_from_serialized(loaded)
            signature = oracle.parse_operation_signature(str(root["operation_signature"]))
            if latent.operation_qubits(clean) != signature:
                raise RuntimeError(f"reconstructed circuit mismatch at fit root {root_index}")
            candidates = latent.plausible_candidates(clean)
            finite_jacs = latent.finite_shot_candidate_jacobians(
                clean, pairs, candidates, root_index, latent_cfg
            )
            distance, gate = equiv.finite_frame_pair_geometry(finite_jacs)
            cache[root_index] = {
                "candidates": candidates,
                "finite_jacs": finite_jacs,
                "distance": distance,
                "gate": gate,
            }

        root_cache = cache[root_index]
        finite_jacs = root_cache["finite_jacs"]
        per_candidate = np.stack(
            [frame.canonicalize_evidence(delta, jac, weights)[0] for jac in finite_jacs]
        ).astype(np.float32)
        profiles = latent.profile_log_likelihoods(delta, weights, finite_jacs).astype(np.float64)

        candidate_features.append(per_candidate)
        profile_scores.append(profiles)
        pair_distances.append(root_cache["distance"])
        pair_gates.append(root_cache["gate"])
        truth.append(frame.TARGET[str(row["mechanism"])])
        families.append(str(row["family_id"]))
        candidate_counts.append(len(root_cache["candidates"]))

        if progress_every and position % progress_every == 0:
            print(
                f"equivalence fit extraction {position}/{len(selected)} roots_cached={len(cache)}",
                flush=True,
            )

    x, mask = latent.pad_candidate_features(candidate_features)
    max_candidates = int(x.shape[1])
    y = np.asarray(truth, dtype=np.int64)
    fam = np.asarray(families, dtype=object)
    counts = Counter(int(v) for v in y.tolist())
    if set(counts) != {0, 1, 2} or len(set(counts.values())) != 1:
        raise RuntimeError(f"fit mechanism classes are not balanced: {counts}")
    if len(set(fam.tolist())) != 600:
        raise RuntimeError(f"expected exactly 600 fit families, got {len(set(fam.tolist()))}")

    return {
        "candidate_features": x,
        "candidate_mask": mask,
        "profile_scores": pad_profiles(profile_scores, max_candidates),
        "pair_distance": pad_pair_float(pair_distances, max_candidates),
        "pair_gate": pad_pair_bool(pair_gates, max_candidates),
        "truth": y,
        "family": fam,
        "candidate_count_summary": frame.quantiles(candidate_counts),
        "loaded_target_examples": int(len(y)),
        "loaded_roots": int(len(cache)),
    }


def metric(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    ba, recall = latent.balanced_accuracy(y, logits)
    return {
        "mechanism_balanced_accuracy": float(ba),
        "mechanism_recall": [float(v) for v in recall],
        "minimum_mechanism_recall": float(min(recall)),
        "example_count": int(len(y)),
    }


def hp_rank(record: Mapping[str, Any]) -> tuple[float, ...]:
    tau = float(record["tau"])
    temperature = float(record["frame_temperature"])
    return (
        -float(record["mechanism_balanced_accuracy"]),
        -float(record["minimum_mechanism_recall"]),
        abs(float(np.log(tau / 0.05))),
        abs(float(np.log(temperature / 1.0))),
        tau,
        temperature,
    )


def public_training_record(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(value["seed"]),
        "final_train_loss": float(value["final_train_loss"]),
        "train_balanced_accuracy": float(value["train_balanced_accuracy"]),
        "train_recall": [float(v) for v in value["train_recall"]],
    }


def main() -> None:
    args = parse_args()
    oracle_compat.apply_frozen_support_bound()
    dev = load_development_protocol()
    frozen = dev["source_freeze"]

    if args.training_run_id != str(frozen["training_run_id"]):
        raise RuntimeError("training run differs from frozen equivalence-aware protocol")
    if args.selection_freeze_sha256 != str(frozen["selection_freeze_sha256"]):
        raise RuntimeError("selection-freeze identity differs from frozen protocol")

    rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)
    step14_protocol = step14.read_json(STEP14_CONFIG)
    step14.assert_contract(step14_protocol)
    latent_cfg = latent.load_frozen_config()
    if baseline.sha256_file(LATENT_CONFIG) != str(frozen["latent_protocol_sha256"]):
        raise RuntimeError("frozen latent protocol hash drift")

    cross_product = step14.resolve_cross_product(None)
    cross_rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(
        cross_product, step14_protocol
    )
    complete = read_json(cross_product / "dataset_complete.json")
    if str(complete["product_id"]) != str(frozen["development_product_id"]):
        raise RuntimeError("development product id drift")
    if baseline.sha256_file(cross_product / "dataset_complete.json") != str(
        frozen["development_dataset_sha256"]
    ):
        raise RuntimeError("development dataset hash drift")

    # Build a deliberately sanitized root view: privileged location fields are
    # never copied into the data structure visible to the fit-only extractor.
    root_rows = baseline.read_csv(cross_product / "manifests" / "root_manifest.csv")
    safe_roots: dict[int, dict[str, str]] = {}
    for row in root_rows:
        if str(row.get("step14_partition")) != "fit":
            continue
        safe_roots[int(row["root_index"])] = {
            "step14_partition": "fit",
            "operation_signature": str(row["operation_signature"]),
        }

    fit_rows = [row for row in cross_rows if str(row.get("step14_partition")) == "fit"]
    if not fit_rows or any(str(row.get("step14_partition")) != "fit" for row in fit_rows):
        raise RuntimeError("fit-row filtering failed closed")

    table = extract_fit_table(
        cross_product, fit_rows, safe_roots, latent_cfg, args.progress_every
    )
    x = table["candidate_features"]
    mask = table["candidate_mask"]
    profiles = table["profile_scores"]
    distance = table["pair_distance"]
    gate = table["pair_gate"]
    y = table["truth"]
    families = table["family"]
    if len(y) != 28800:
        raise RuntimeError(f"expected 28800 distorted fit examples, got {len(y)}")

    device = resolve_device(args.device)
    hp = dev["fit_only_hyperparameter_selection"]
    fold_count = int(hp["family_level_folds"])
    folds = np.asarray([equiv.family_fold(str(value), fold_count) for value in families], dtype=np.int64)
    if set(folds.tolist()) != set(range(fold_count)):
        raise RuntimeError("fit-family cross-validation folds are incomplete")

    crossfit_candidate_logits = np.zeros((len(y), x.shape[1], 3), dtype=np.float64)
    crossfit_uniform_logits = np.zeros((len(y), 3), dtype=np.float64)
    cv_training: dict[str, Any] = {}

    for fold in range(fold_count):
        train_idx = np.flatnonzero(folds != fold)
        eval_idx = np.flatnonzero(folds == fold)
        if set(families[train_idx].tolist()) & set(families[eval_idx].tolist()):
            raise RuntimeError("family leakage across internal fit CV")
        seed_candidates: list[np.ndarray] = []
        seed_uniform: list[np.ndarray] = []
        fold_records: dict[str, Any] = {}
        for seed in [int(v) for v in hp["network_seeds_per_fold"]]:
            trained = equiv.train_uniform_candidate_network(
                x[train_idx],
                mask[train_idx],
                y[train_idx],
                x[eval_idx],
                mask[eval_idx],
                seed=seed,
                latent_cfg=latent_cfg,
                device=device,
            )
            seed_candidates.append(trained["eval_candidate_logits"])
            seed_uniform.append(trained["eval_uniform_pooled_logits"])
            fold_records[str(seed)] = public_training_record(trained)
            print(
                f"equivalence CV fold={fold} seed={seed} "
                f"train_BA={trained['train_balanced_accuracy']:.4f}",
                flush=True,
            )
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
                crossfit_candidate_logits,
                profiles,
                distance,
                gate,
                mask,
                tau=tau,
                frame_temperature=temperature,
            )
            record = {
                "tau": tau,
                "frame_temperature": temperature,
                **metric(y, logits),
            }
            grid_records.append(record)
            print(
                f"equivalence fit-CV tau={tau:g} T={temperature:g} "
                f"BA={record['mechanism_balanced_accuracy']:.6f} "
                f"min_recall={record['minimum_mechanism_recall']:.6f}",
                flush=True,
            )

    selected = min(grid_records, key=hp_rank)
    selected_tau = float(selected["tau"])
    selected_temperature = float(selected["frame_temperature"])
    uniform_crossfit = metric(y, crossfit_uniform_logits)

    protocol_sha = baseline.sha256_file(DEV_CONFIG)
    method_key = hashlib.sha256(
        (
            protocol_sha
            + str(frozen["development_dataset_sha256"])
            + str(selected_tau)
            + str(selected_temperature)
        ).encode("utf-8")
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
            x, mask, y, x, mask,
            seed=seed,
            latent_cfg=latent_cfg,
            device=device,
        )
        fit_candidate_logits.append(trained["eval_candidate_logits"])
        checkpoint_name = f"candidate_network_seed{seed}.pt"
        checkpoint_path = output_dir / checkpoint_name
        torch.save(
            {
                "schema": "triqto.v0_2.step14_equivalence_candidate_network.v1",
                "method_id": method_id,
                "seed": seed,
                "state_dict": trained["state_dict"],
                "normalization_mean": trained["mean"],
                "normalization_std": trained["std"],
                "selected_tau": selected_tau,
                "selected_frame_temperature": selected_temperature,
                "development_protocol_sha256": protocol_sha,
                "latent_protocol_sha256": baseline.sha256_file(LATENT_CONFIG),
                "training_run_id": args.training_run_id,
                "selection_freeze_sha256": args.selection_freeze_sha256,
                "fit_only": True,
                "location_supervision": False,
                "main_model_weights_updated": False,
            },
            checkpoint_path,
        )
        checkpoint_hashes[checkpoint_name] = baseline.sha256_file(checkpoint_path)
        final_training[str(seed)] = public_training_record(trained)
        print(
            f"equivalence final fit seed={seed} "
            f"train_BA={trained['train_balanced_accuracy']:.4f}",
            flush=True,
        )

    final_candidate_ensemble = np.mean(np.stack(fit_candidate_logits), axis=0)
    final_fit_logits = equiv.equivalence_aware_pool_batch(
        final_candidate_ensemble,
        profiles,
        distance,
        gate,
        mask,
        tau=selected_tau,
        frame_temperature=selected_temperature,
    )

    bundle = {
        "schema": "triqto.v0_2.step14_equivalence_aware_method_bundle.v1",
        "status": "FIT_ONLY_METHOD_BUNDLE_READY_FOR_FREEZE",
        "method_id": method_id,
        "development_protocol_sha256": protocol_sha,
        "source_training_run_id": args.training_run_id,
        "source_selection_freeze_sha256": args.selection_freeze_sha256,
        "selected_hyperparameters": {
            "tau": selected_tau,
            "frame_temperature": selected_temperature,
        },
        "candidate_network_seeds": [int(v) for v in dev["final_fit_bundle"]["seeds"]],
        "checkpoint_sha256": checkpoint_hashes,
        "fit_only": True,
        "selection_artifacts_loaded": False,
        "selection_labels_read": False,
        "true_location_used": False,
        "exact_statevector_geometry_used_in_deployable_similarity": False,
        "main_model_retrained": False,
        "main_model_weights_updated": False,
        "fresh_holdout_generated": False,
    }
    bundle_path = output_dir / "method_bundle.json"
    atomic_json(bundle_path, bundle)
    bundle_sha = baseline.sha256_file(bundle_path)

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FIT_ONLY_EQUIVALENCE_AWARE_DEVELOPMENT",
        "method_id": method_id,
        "identity": {
            "development_protocol_sha256": protocol_sha,
            "latent_protocol_sha256": baseline.sha256_file(LATENT_CONFIG),
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
            "selection_labels_read": False,
            "selection_metrics_used": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
        },
        "location_privilege_boundary": {
            "true_location_read": False,
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
            "equivalence_aware_metric": metric(y, final_fit_logits),
            "training": final_training,
            "checkpoint_sha256": checkpoint_hashes,
        },
        "method_bundle_sha256": bundle_sha,
        "fresh_holdout_generated": False,
        "main_model_retrained": False,
        "main_model_weights_updated": False,
    }
    result_path = output_dir / "fit_development_result.json"
    atomic_json(result_path, result)
    result_sha = baseline.sha256_file(result_path)

    complete_payload = {
        "schema": "triqto.v0_2.step14_equivalence_aware_fit_development_complete.v1",
        "status": "COMPLETE_FIT_ONLY_EQUIVALENCE_AWARE_DEVELOPMENT",
        "method_id": method_id,
        "method_dir": str(output_dir),
        "method_bundle_sha256": bundle_sha,
        "fit_development_result_sha256": result_sha,
        "checkpoint_sha256": checkpoint_hashes,
        "selected_hyperparameters": {
            "tau": selected_tau,
            "frame_temperature": selected_temperature,
        },
        "selection_artifacts_loaded": False,
        "fresh_holdout_generated": False,
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    }
    atomic_json(output_dir / "fit_complete.json", complete_payload)
    complete_sha = baseline.sha256_file(output_dir / "fit_complete.json")
    atomic_json(
        args.output_parent / "current_equivalence_aware_fit.json",
        {
            "schema": "triqto.v0_2.step14_equivalence_aware_fit_pointer.v1",
            "method_id": method_id,
            "method_dir": str(output_dir),
            "fit_development_result_sha256": result_sha,
            "fit_complete_sha256": complete_sha,
            "method_bundle_sha256": bundle_sha,
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_FIT_ONLY_EQUIVALENCE_AWARE_DEVELOPMENT",
                "method_id": method_id,
                "selected_tau": selected_tau,
                "selected_frame_temperature": selected_temperature,
                "fit_development_result_sha256": result_sha,
                "method_bundle_sha256": bundle_sha,
                "fit_complete_sha256": complete_sha,
                "selection_artifacts_loaded": False,
                "fresh_holdout_generated": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
