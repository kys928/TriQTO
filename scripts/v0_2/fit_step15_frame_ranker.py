#!/usr/bin/env python3
"""FIT-only Step-15 frame-ranker development, final fit, and immutable freeze.

This executable is intentionally incapable of reading the spent Step-14 fresh
holdout. Hyperparameter selection is family-level CV over Step-14 FIT families
only. True location / exact response geometry are used only to construct FIT
ranking labels and never enter deployable ranker features.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import analyze_step14_candidate_frame_ambiguity as ambiguity
import analyze_step14_latent_frame_inference as latent
import analyze_step14_local_frame_canonicalization as frame
import benchmark_step6_cheap_baselines as baseline
import run_step14_cross_motif_training as step14
import run_step14_oracle_raw_evidence_ceiling as oracle_compat
import step14_equivalence_aware_common as equiv
import step15_frame_ranker_common as ranker

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step15_frame_ranking_development.json"
STEP14_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
STEP14_METHOD_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_development")
OUTPUT_PARENT = Path("/workspace/triqto-data/step15_frame_ranking")
SCHEMA = "triqto.v0_2.step15_frame_ranking_fit_result.v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as h:
        json.dump(value, h, indent=2, sort_keys=True, allow_nan=False)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.replace(tmp, path)


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_protocol() -> dict[str, Any]:
    cfg = read_json(CONFIG)
    if cfg.get("schema") != "triqto.v0_2.step15_frame_ranking_development.v1":
        raise RuntimeError("unexpected Step-15 protocol schema")
    if cfg.get("status") != "FROZEN_BEFORE_FIT_ONLY_RANKER_DEVELOPMENT":
        raise RuntimeError("Step-15 protocol is not frozen")
    boundary = cfg["development_boundary"]
    if boundary.get("allowed_partition") != "fit":
        raise RuntimeError("Step-15 development is not FIT-only")
    for key in (
        "selection_partition_accessed",
        "spent_fresh_holdout_accessed",
        "simulator_outer_accessed",
        "future_hardware_reserve_accessed",
        "qpu_executed",
    ):
        if bool(boundary.get(key)):
            raise RuntimeError(f"Step-15 boundary drift: {key}=true")
    return cfg


def response_equivalent(candidate_exact: np.ndarray, true_exact: np.ndarray, spec: Mapping[str, Any]) -> bool:
    similarity = ambiguity.axis_similarity(candidate_exact, true_exact)
    local_cfg = {
        "frame_equivalence": {
            "minimum_same_axis_abs_cosine": float(spec["minimum_same_axis_abs_cosine"]),
            "minimum_identity_assignment_margin": float(spec["minimum_identity_assignment_margin"]),
        }
    }
    return bool(ambiguity.is_equivalent(similarity, local_cfg))


def pad_float(values: Sequence[np.ndarray], max_candidates: int, feature_dim: int) -> np.ndarray:
    out = np.zeros((len(values), max_candidates, feature_dim), dtype=np.float32)
    for i, value in enumerate(values):
        out[i, : len(value)] = value
    return out


def pad_profiles(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    out = np.full((len(values), max_candidates, 3), -np.inf, dtype=np.float64)
    for i, value in enumerate(values):
        out[i, : len(value)] = value
    return out


def pad_pair_float(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    out = np.ones((len(values), max_candidates, max_candidates), dtype=np.float32)
    for i, value in enumerate(values):
        out[i, : len(value), : len(value)] = value
    return out


def pad_pair_bool(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    out = np.zeros((len(values), max_candidates, max_candidates), dtype=np.bool_)
    for i, value in enumerate(values):
        out[i, : len(value), : len(value)] = value
    return out


def pad_bool(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    out = np.zeros((len(values), max_candidates), dtype=np.bool_)
    for i, value in enumerate(values):
        out[i, : len(value)] = value
    return out


def extract_fit_rank_table(
    product: Path,
    cfg: Mapping[str, Any],
    latent_cfg: Mapping[str, Any],
    progress_every: int,
) -> dict[str, Any]:
    step14_cfg = step14.read_json(STEP14_CONFIG)
    step14.assert_contract(step14_cfg)
    rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(product, step14_cfg)
    root_rows = baseline.read_csv(product / "manifests" / "root_manifest.csv")
    roots = {
        int(r["root_index"]): r
        for r in root_rows
        if str(r.get("step14_partition")) == "fit"
    }
    selected = [
        r for r in rows
        if str(r.get("step14_partition")) == "fit" and str(r.get("mechanism")) in frame.TARGET
    ]
    if not selected or any(str(r.get("step14_partition")) != "fit" for r in selected):
        raise RuntimeError("non-FIT row reached Step-15 rank extractor")

    source = cfg["source_identity"]
    tau = float(source["step14_tau"])
    temperature = float(source["step14_frame_temperature"])
    eq_spec = cfg["fit_positive_definition"]

    root_cache: dict[int, dict[str, Any]] = {}
    canonical_all: list[np.ndarray] = []
    profiles_all: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    rank_features: list[np.ndarray] = []
    positive_masks: list[np.ndarray] = []
    families: list[str] = []
    mechanism_truth: list[int] = []
    candidate_counts: list[int] = []
    positive_sizes: list[int] = []

    for pos, row in enumerate(selected, start=1):
        root_index = int(row["root_index"])
        root = roots.get(root_index)
        if root is None:
            raise RuntimeError(f"FIT root missing for Step-15: {root_index}")
        artifact = product / str(row["artifact_path"])
        if baseline.sha256_file(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"FIT artifact hash mismatch: {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as src_file:
            loaded = {str(k): np.asarray(src_file[k]) for k in src_file.files}
        forbidden = [
            key for key in loaded
            if key.startswith("x__")
            and any(token in key.lower() for token in ("affected_qubit", "injection_boundary", "target", "mechanism_target"))
        ]
        if forbidden:
            raise RuntimeError(f"privileged field leaked into Step-15 inference input: {forbidden}")

        delta, weights, pairs = frame.measured_delta_and_weights(loaded)
        if root_index not in root_cache:
            clean = frame.circuit_from_serialized(loaded)
            candidates = latent.plausible_candidates(clean)
            finite_jacs = latent.finite_shot_candidate_jacobians(
                clean, pairs, candidates, root_index, latent_cfg
            )
            exact_jacs = latent.exact_candidate_jacobians(clean, pairs, candidates)
            distance, gate = equiv.finite_frame_pair_geometry(finite_jacs)
            true_location = (
                int(root["affected_qubit"]),
                int(root["injection_boundary_rank"]),
            )
            if true_location in candidates:
                true_exact = exact_jacs[candidates.index(true_location)]
            else:
                true_exact = frame.frame_response_jacobian(
                    clean, true_location[1], true_location[0], pairs
                )
            positive = np.asarray(
                [response_equivalent(jac, true_exact, eq_spec) for jac in exact_jacs],
                dtype=np.bool_,
            )
            if not np.any(positive):
                raise RuntimeError(f"FIT root has no response-equivalent candidate: {root_index}")
            root_cache[root_index] = {
                "clean": clean,
                "candidates": candidates,
                "finite_jacs": finite_jacs,
                "distance": distance,
                "gate": gate,
                "positive": positive,
            }

        rc = root_cache[root_index]
        canonical = np.stack(
            [frame.canonicalize_evidence(delta, jac, weights)[0] for jac in rc["finite_jacs"]]
        ).astype(np.float32)
        profiles = latent.profile_log_likelihoods(delta, weights, rc["finite_jacs"]).astype(np.float64)
        features = ranker.build_rank_features_one(
            canonical,
            profiles,
            rc["distance"],
            rc["gate"],
            rc["candidates"],
            qubit_count=int(rc["clean"].num_qubits),
            gate_count=int(len(rc["clean"].data)),
            tau=tau,
            frame_temperature=temperature,
        )
        canonical_all.append(canonical)
        profiles_all.append(profiles)
        distances.append(rc["distance"])
        gates.append(rc["gate"])
        rank_features.append(features)
        positive_masks.append(rc["positive"])
        families.append(str(row["family_id"]))
        mechanism_truth.append(int(frame.TARGET[str(row["mechanism"])]))
        candidate_counts.append(len(rc["candidates"]))
        positive_sizes.append(int(np.sum(rc["positive"])))

        if progress_every and pos % progress_every == 0:
            print(f"Step-15 FIT extraction {pos}/{len(selected)} roots_cached={len(root_cache)}", flush=True)

    canonical_pad, mask = latent.pad_candidate_features(canonical_all)
    max_candidates = int(canonical_pad.shape[1])
    features_pad = pad_float(rank_features, max_candidates, len(ranker.FEATURE_NAMES))
    positive_pad = pad_bool(positive_masks, max_candidates)
    profiles_pad = pad_profiles(profiles_all, max_candidates)
    distance_pad = pad_pair_float(distances, max_candidates)
    gate_pad = pad_pair_bool(gates, max_candidates)
    family = np.asarray(families, dtype=object)
    y = np.asarray(mechanism_truth, dtype=np.int64)

    counts = Counter(int(v) for v in y.tolist())
    if len(y) != 28800 or counts != Counter({0: 9600, 1: 9600, 2: 9600}):
        raise RuntimeError(f"Step-15 FIT mechanism balance/count drift: {counts}")
    if len(set(family.tolist())) != 600:
        raise RuntimeError("Step-15 expected exactly 600 FIT families")
    if np.any(np.sum(positive_pad & mask, axis=1) < 1):
        raise RuntimeError("Step-15 FIT bag without valid response-equivalent positive")

    return {
        "canonical": canonical_pad,
        "rank_features": features_pad,
        "mask": mask,
        "positive": positive_pad,
        "profiles": profiles_pad,
        "distance": distance_pad,
        "gate": gate_pad,
        "family": family,
        "mechanism_truth": y,
        "candidate_count": np.asarray(candidate_counts, dtype=np.int64),
        "positive_size": np.asarray(positive_sizes, dtype=np.int64),
        "root_count": int(len(root_cache)),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(x)),
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def hp_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    m = record["metrics"]
    return (
        -float(m["top1"]),
        -float(m["mrr"]),
        float(m["set_nll"]),
        float(m["set_brier"]),
        float(m["top_candidate_ece"]),
        float(record["pairwise_weight"]),
        float(record["hard_negative_count"]),
        float(record["weight_decay"]),
    )


def verify_step14_method_and_infer(
    table: Mapping[str, Any],
    cfg: Mapping[str, Any],
    latent_cfg: Mapping[str, Any],
    device: torch.device,
) -> np.ndarray:
    src = cfg["source_identity"]
    pointer = read_json(STEP14_METHOD_PARENT / "current_equivalence_aware_fit.json")
    if str(pointer["method_id"]) != str(src["step14_method_id"]):
        raise RuntimeError("Step-14 method pointer drift")
    method_dir = Path(str(pointer["method_dir"])).resolve()
    if baseline.sha256_file(method_dir / "method_bundle.json") != str(src["step14_method_bundle_sha256"]):
        raise RuntimeError("Step-14 method bundle hash drift")

    seed_logits: list[np.ndarray] = []
    for seed in [int(v) for v in src["candidate_network_seeds"]]:
        name = f"candidate_network_seed{seed}.pt"
        path = method_dir / name
        expected = str(src["candidate_network_checkpoint_sha256"][name])
        if baseline.sha256_file(path) != expected:
            raise RuntimeError(f"Step-14 checkpoint hash drift: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _uniform, candidate = equiv.infer_candidate_network(
            table["canonical"],
            table["mask"],
            state_dict=payload["state_dict"],
            mean=np.asarray(payload["normalization_mean"], dtype=np.float32),
            std=np.asarray(payload["normalization_std"], dtype=np.float32),
            latent_cfg=latent_cfg,
            device=device,
        )
        seed_logits.append(candidate)
        print(f"Step-15 frozen Step-14 candidate network seed={seed} replay complete", flush=True)
    return np.mean(np.stack(seed_logits), axis=0)


def mechanism_metric(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    ba, recall = latent.balanced_accuracy(y, logits)
    return {
        "balanced_accuracy": float(ba),
        "recall": [float(v) for v in recall],
        "minimum_recall": float(min(recall)),
    }


def main() -> None:
    args = parse_args()
    oracle_compat.apply_frozen_support_bound()
    cfg = load_protocol()
    src = cfg["source_identity"]
    latent_cfg = latent.load_frozen_config()

    cross_product = step14.resolve_cross_product(None)
    complete = read_json(cross_product / "dataset_complete.json")
    if str(complete["product_id"]) != str(src["development_product_id"]):
        raise RuntimeError("Step-15 development product drift")
    if baseline.sha256_file(cross_product / "dataset_complete.json") != str(src["development_dataset_sha256"]):
        raise RuntimeError("Step-15 development dataset hash drift")

    table = extract_fit_rank_table(cross_product, cfg, latent_cfg, args.progress_every)
    x = table["rank_features"]
    mask = table["mask"]
    positive = table["positive"]
    families = table["family"]
    baseline_scores = np.asarray(x[:, :, 0], dtype=np.float64)
    baseline_scores[~mask] = -np.inf
    baseline_rank_metrics = ranker.ranking_metrics(baseline_scores, mask, positive)

    cv = cfg["fit_only_family_cross_validation"]
    fold_count = int(cv["fold_count"])
    folds = np.asarray([equiv.family_fold(str(v), fold_count) for v in families], dtype=np.int64)
    if set(folds.tolist()) != set(range(fold_count)):
        raise RuntimeError("Step-15 family CV folds incomplete")

    device = resolve_device(args.device)
    grid: list[dict[str, Any]] = []
    for pairwise_weight, weight_decay, hard_negative_count in itertools.product(
        [float(v) for v in cv["pairwise_weight_grid"]],
        [float(v) for v in cv["weight_decay_grid"]],
        [int(v) for v in cv["hard_negative_count_grid"]],
    ):
        crossfit = np.full(mask.shape, -np.inf, dtype=np.float64)
        fold_records: dict[str, Any] = {}
        for fold in range(fold_count):
            train_idx = np.flatnonzero(folds != fold)
            eval_idx = np.flatnonzero(folds == fold)
            if set(families[train_idx].tolist()) & set(families[eval_idx].tolist()):
                raise RuntimeError("family leakage in Step-15 CV")
            fitted = ranker.train_linear_rank_adapter(
                x[train_idx],
                mask[train_idx],
                positive[train_idx],
                learning_rate=float(cv["learning_rate"]),
                weight_decay=weight_decay,
                pairwise_weight=pairwise_weight,
                pairwise_margin=float(cv["pairwise_margin"]),
                hard_negative_count=hard_negative_count,
                epochs=int(cv["epochs"]),
                batch_size=int(cv["batch_size"]),
                seed=int(cv["seed"]) + fold,
                device=device,
            )
            crossfit[eval_idx] = ranker.adapter_scores(x[eval_idx], mask[eval_idx], fitted)
            fold_records[str(fold)] = {
                "train_family_count": int(len(set(families[train_idx].tolist()))),
                "eval_family_count": int(len(set(families[eval_idx].tolist()))),
                "final_loss": {k: float(v) for k, v in fitted["final_loss"].items()},
            }
        metrics = ranker.ranking_metrics(crossfit, mask, positive)
        record = {
            "pairwise_weight": pairwise_weight,
            "weight_decay": weight_decay,
            "hard_negative_count": hard_negative_count,
            "metrics": metrics,
            "folds": fold_records,
        }
        grid.append(record)
        print(
            "Step-15 CV "
            f"pair={pairwise_weight:g} wd={weight_decay:g} hard={hard_negative_count} "
            f"top1={metrics['top1']:.6f} mrr={metrics['mrr']:.6f} "
            f"nll={metrics['set_nll']:.6f}",
            flush=True,
        )

    selected = min(grid, key=hp_key)
    final_cfg = cfg["final_fit"]
    final_adapter = ranker.train_linear_rank_adapter(
        x,
        mask,
        positive,
        learning_rate=float(cv["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
        pairwise_weight=float(selected["pairwise_weight"]),
        pairwise_margin=float(cv["pairwise_margin"]),
        hard_negative_count=int(selected["hard_negative_count"]),
        epochs=int(cv["epochs"]),
        batch_size=int(cv["batch_size"]),
        seed=int(final_cfg["seed"]),
        device=device,
    )
    final_scores = ranker.adapter_scores(x, mask, final_adapter)
    final_rank_metrics = ranker.ranking_metrics(final_scores, mask, positive)

    candidate_logits = verify_step14_method_and_infer(table, cfg, latent_cfg, device)
    tau = float(src["step14_tau"])
    temperature = float(src["step14_frame_temperature"])
    baseline_mechanism = equiv.equivalence_aware_pool_batch(
        candidate_logits,
        table["profiles"],
        table["distance"],
        table["gate"],
        mask,
        tau=tau,
        frame_temperature=temperature,
    )
    zero_adapter = {
        "weights": np.zeros(len(ranker.FEATURE_NAMES), dtype=np.float32),
        "mean": np.zeros(len(ranker.FEATURE_NAMES), dtype=np.float32),
        "std": np.ones(len(ranker.FEATURE_NAMES), dtype=np.float32),
        "seed": 0,
        "hard_negative_count": 1,
        "final_loss": {},
    }
    zero_mechanism = ranker.equivalence_aware_pool_batch_with_adapter(
        candidate_logits,
        table["profiles"],
        table["distance"],
        table["gate"],
        mask,
        x,
        zero_adapter,
        tau=tau,
        frame_temperature=temperature,
    )
    zero_exact = bool(np.array_equal(baseline_mechanism, zero_mechanism))
    if not zero_exact:
        diff = float(np.max(np.abs(baseline_mechanism - zero_mechanism)))
        raise RuntimeError(f"Step-15 zero-adapter failed exact Step-14 recovery; max_abs={diff}")

    step15_fit_mechanism = ranker.equivalence_aware_pool_batch_with_adapter(
        candidate_logits,
        table["profiles"],
        table["distance"],
        table["gate"],
        mask,
        x,
        final_adapter,
        tau=tau,
        frame_temperature=temperature,
    )

    protocol_sha = baseline.sha256_file(CONFIG)
    selected_hyperparameters = {
        "learning_rate": float(cv["learning_rate"]),
        "epochs": int(cv["epochs"]),
        "batch_size": int(cv["batch_size"]),
        "pairwise_margin": float(cv["pairwise_margin"]),
        "pairwise_weight": float(selected["pairwise_weight"]),
        "weight_decay": float(selected["weight_decay"]),
        "hard_negative_count": int(selected["hard_negative_count"]),
        "final_seed": int(final_cfg["seed"]),
    }
    method_key = hashlib.sha256(
        (protocol_sha + str(src["development_dataset_sha256"]) + json.dumps(selected_hyperparameters, sort_keys=True)).encode("utf-8")
    ).hexdigest()[:24]
    method_id = f"step15_rank_{method_key}"
    output_dir = args.output_parent / method_id
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite immutable Step-15 method: {output_dir}")
        output_dir.rmdir()
    output_dir.mkdir(parents=True, exist_ok=False)

    adapter_payload = {
        "schema": "triqto.v0_2.step15_frame_rank_adapter.v1",
        "status": "FROZEN_FIT_ONLY_FRAME_RANK_ADAPTER",
        "method_id": method_id,
        "source_step14_method_id": str(src["step14_method_id"]),
        "source_development_dataset_sha256": str(src["development_dataset_sha256"]),
        "step14_tau": tau,
        "step14_frame_temperature": temperature,
        "selected_hyperparameters": selected_hyperparameters,
        "adapter": ranker.adapter_to_json(final_adapter),
        "fit_only": True,
        "spent_holdout_accessed": False,
        "true_location_is_inference_feature": False,
        "exact_geometry_is_inference_feature": False,
        "candidate_generator_changed": False,
        "mechanism_network_changed": False,
    }
    adapter_path = output_dir / "frame_rank_adapter.json"
    atomic_json(adapter_path, adapter_payload)
    adapter_sha = baseline.sha256_file(adapter_path)

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FIT_ONLY_STEP15_FRAME_RANKING",
        "method_id": method_id,
        "identity": {
            "protocol_sha256": protocol_sha,
            "development_product_id": str(src["development_product_id"]),
            "development_dataset_sha256": str(src["development_dataset_sha256"]),
            "step14_method_id": str(src["step14_method_id"]),
            "step14_method_bundle_sha256": str(src["step14_method_bundle_sha256"]),
        },
        "data_boundary": {
            "fit_examples": int(len(families)),
            "fit_families": int(len(set(families.tolist()))),
            "fit_roots": int(table["root_count"]),
            "selection_partition_accessed": False,
            "spent_holdout_accessed": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
        },
        "candidate_count": quantiles(table["candidate_count"]),
        "response_equivalent_positive_set_size": quantiles(table["positive_size"]),
        "baseline_fit_ranking": baseline_rank_metrics,
        "family_cross_validation": {
            "fold_count": fold_count,
            "grid": grid,
            "selected": selected,
        },
        "final_fit_ranking": final_rank_metrics,
        "fit_only_downstream_mechanism": {
            "step14_baseline": mechanism_metric(table["mechanism_truth"], baseline_mechanism),
            "step15_adapter": mechanism_metric(table["mechanism_truth"], step15_fit_mechanism),
        },
        "zero_adapter_regression": {
            "exact_array_equal": zero_exact,
            "max_abs_diff": 0.0,
            "comparison_examples": int(len(families)),
            "contract": "exact recovery of frozen Step-14 equivalence_aware_pool_batch",
        },
        "adapter_sha256": adapter_sha,
    }
    result_path = output_dir / "fit_development_result.json"
    atomic_json(result_path, result)
    result_sha = baseline.sha256_file(result_path)

    freeze_payload = {
        "schema": "triqto.v0_2.step15_frame_rank_adapter_freeze.v1",
        "status": "IMMUTABLY_FROZEN_BEFORE_SPENT_HOLDOUT_EVALUATION",
        "method_id": method_id,
        "protocol_sha256": protocol_sha,
        "adapter_sha256": adapter_sha,
        "fit_development_result_sha256": result_sha,
        "source_step14_method_id": str(src["step14_method_id"]),
        "source_step14_method_bundle_sha256": str(src["step14_method_bundle_sha256"]),
        "source_development_dataset_sha256": str(src["development_dataset_sha256"]),
        "selected_hyperparameters": selected_hyperparameters,
        "zero_adapter_exact_recovery": True,
        "spent_holdout_accessed": False,
        "holdout_may_not_select_or_modify_adapter": True,
    }
    freeze_payload["freeze_payload_sha256"] = canonical_sha(freeze_payload)
    freeze_path = output_dir / "adapter_freeze.json"
    atomic_json(freeze_path, freeze_payload)
    freeze_file_sha = baseline.sha256_file(freeze_path)

    complete = {
        "schema": "triqto.v0_2.step15_frame_ranking_complete.v1",
        "status": "HASH_VERIFIED_READY_FOR_SPENT_HOLDOUT_AUDIT",
        "method_id": method_id,
        "method_dir": str(output_dir),
        "adapter_sha256": adapter_sha,
        "fit_development_result_sha256": result_sha,
        "adapter_freeze_sha256": freeze_file_sha,
        "freeze_payload_sha256": freeze_payload["freeze_payload_sha256"],
        "zero_adapter_exact_recovery": True,
        "spent_holdout_accessed": False,
    }
    complete_path = output_dir / "step15_complete.json"
    atomic_json(complete_path, complete)
    complete_sha = baseline.sha256_file(complete_path)

    pointer = {
        "schema": "triqto.v0_2.step15_frame_ranking_pointer.v1",
        "method_id": method_id,
        "method_dir": str(output_dir),
        "adapter_sha256": adapter_sha,
        "fit_development_result_sha256": result_sha,
        "adapter_freeze_sha256": freeze_file_sha,
        "step15_complete_sha256": complete_sha,
        "spent_holdout_accessed": False,
    }
    atomic_json(args.output_parent / "current_step15_frame_ranker.json", pointer)

    print(json.dumps({
        "status": complete["status"],
        "method_id": method_id,
        "adapter_sha256": adapter_sha,
        "fit_development_result_sha256": result_sha,
        "adapter_freeze_sha256": freeze_file_sha,
        "step15_complete_sha256": complete_sha,
        "cv_selected_metrics": selected["metrics"],
        "final_fit_ranking": final_rank_metrics,
        "zero_adapter_exact_recovery": True,
        "spent_holdout_accessed": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
