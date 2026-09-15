#!/usr/bin/env python3
"""Post-hoc residual-bottleneck autopsy on the spent Step-14 fresh holdout.

This diagnostic is explicitly descriptive.  It cannot change the completed
confirmatory verdict and it performs no training except the already-precommitted
FIT-only finite/exact SmallProbe references.  Frozen candidate-network weights
are never updated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import analyze_step14_candidate_frame_ambiguity as ambiguity
import analyze_step14_latent_frame_inference as latent
import analyze_step14_local_frame_canonicalization as frame
import analyze_step14_oracle_raw_evidence_ceiling as oracle
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import predict_step14_equivalence_aware_fresh_holdout_blind as blind_predict
import run_step14_cross_motif_training as step14
import run_step14_oracle_raw_evidence_ceiling as oracle_compat
import score_step14_equivalence_aware_fresh_holdout_confirmatory as confirm_scorer
import step14_equivalence_aware_common as equiv

ROOT = Path(__file__).resolve().parents[2]
POSTHOC_CONFIG = ROOT / "configs/v0_2/step14_fresh_holdout_residual_bottleneck_posthoc.json"
FINAL_FREEZE = ROOT / "configs/v0_2/step14_equivalence_aware_latent_frame_final_method_freeze.json"
CONFIRM_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_fresh_holdout_confirmatory.json"
HOLDOUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout")
METHOD_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_development")
CONFIRM_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout_confirmation")
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_fresh_holdout_residual_bottleneck")
SCHEMA = "triqto.v0_2.step14_fresh_holdout_residual_bottleneck_result.v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--final-method-freeze-payload-sha256", required=True)
    p.add_argument("--fresh-holdout-product-id", required=True)
    p.add_argument("--fresh-holdout-dataset-complete-sha256", required=True)
    p.add_argument("--oracle-free-prediction-complete-sha256", required=True)
    p.add_argument("--confirmatory-result-sha256", required=True)
    p.add_argument("--confirmatory-complete-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as h:
        json.dump(value, h, indent=2, sort_keys=True, allow_nan=False)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.replace(temp, path)


def sha(path: Path) -> str:
    return baseline.sha256_file(path)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(m), x - m, -np.inf)
    return np.squeeze(m + np.log(np.maximum(np.sum(np.exp(safe), axis=axis, keepdims=True), 1e-300)), axis=axis)


def softmax(logw: np.ndarray) -> np.ndarray:
    x = np.asarray(logw, dtype=np.float64)
    x = x - float(logsumexp(x[None, :], axis=1)[0])
    return np.exp(x)


def posterior_and_kernel(
    profiles: np.ndarray,
    distance: np.ndarray,
    gate: np.ndarray,
    tau: float,
    temperature: float,
    *,
    density_corrected: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = np.where(gate, np.exp(-np.asarray(distance, dtype=np.float64) / tau), 0.0)
    np.fill_diagonal(kernel, 1.0)
    density = np.maximum(kernel.sum(axis=1), 1e-12)
    evidence = logsumexp(np.asarray(profiles, dtype=np.float64), axis=1) - math.log(3.0)
    logw = evidence / temperature
    if density_corrected:
        logw = logw - np.log(density)
    return softmax(logw), kernel, density


def pool_one_no_density(
    candidate_logits: np.ndarray,
    profiles: np.ndarray,
    distance: np.ndarray,
    gate: np.ndarray,
    tau: float,
    temperature: float,
) -> np.ndarray:
    posterior, kernel, _density = posterior_and_kernel(
        profiles, distance, gate, tau, temperature, density_corrected=False
    )
    log_kernel = np.where(kernel > 0.0, np.log(np.maximum(kernel, 1e-300)), -np.inf)
    neighborhood_norm = logsumexp(log_kernel, axis=1)
    smoothed = np.empty_like(candidate_logits, dtype=np.float64)
    for mechanism in range(3):
        smoothed[:, mechanism] = (
            logsumexp(log_kernel + np.asarray(candidate_logits)[:, mechanism][None, :], axis=1)
            - neighborhood_norm
        )
    return logsumexp(np.log(np.maximum(posterior, 1e-300))[:, None] + smoothed, axis=0)


def metric(y: np.ndarray, logits: np.ndarray, kind: str) -> dict[str, Any]:
    out = oracle.metric_record(y, logits)
    out["diagnostic_kind"] = kind
    return out


def quantiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "p10": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def response_equivalent(candidate_exact: np.ndarray, true_exact: np.ndarray, spec: Mapping[str, Any]) -> bool:
    similarity = ambiguity.axis_similarity(candidate_exact, true_exact)
    local_cfg = {"frame_equivalence": {
        "minimum_same_axis_abs_cosine": float(spec["minimum_same_axis_abs_cosine"]),
        "minimum_identity_assignment_margin": float(spec["minimum_identity_assignment_margin"]),
    }}
    return ambiguity.is_equivalent(similarity, local_cfg)


def extract_fresh_table(
    product: Path,
    prediction_rows: Sequence[Mapping[str, str]],
    latent_cfg: Mapping[str, Any],
    response_spec: Mapping[str, Any],
    progress_every: int,
) -> dict[str, Any]:
    blind_examples = {str(r["example_id"]): r for r in baseline.read_csv(product / "manifests" / "example_manifest.csv")}
    sealed_examples = {str(r["example_id"]): r for r in baseline.read_csv(product / "sealed_truth" / "example_manifest.csv")}
    sealed_roots = {int(r["root_index"]): r for r in baseline.read_csv(product / "sealed_truth" / "root_manifest.csv")}

    root_cache: dict[int, dict[str, Any]] = {}
    candidate_features: list[np.ndarray] = []
    profiles_all: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    true_finite_features: list[np.ndarray] = []
    true_exact_features: list[np.ndarray] = []
    true_qubit_masks: list[np.ndarray] = []
    equivalence_masks: list[np.ndarray] = []
    true_indices: list[int] = []
    exact_present: list[bool] = []
    truth: list[int] = []
    mechanism_names: list[str] = []
    families: list[str] = []
    ids: list[str] = []
    root_indices: list[int] = []
    qubit_count: list[int] = []
    gate_count: list[int] = []
    true_qubits: list[int] = []
    true_boundaries: list[int] = []
    normalized_boundary: list[float] = []
    candidate_sets: list[list[tuple[int, int]]] = []
    min_angles: list[float] = []
    snr_proxy: list[float] = []
    equivalence_sizes: list[int] = []

    for pos, pred in enumerate(prediction_rows, start=1):
        example_id = str(pred["example_id"])
        blind = blind_examples.get(example_id)
        sealed = sealed_examples.get(example_id)
        if blind is None or sealed is None:
            raise RuntimeError(f"fresh blind/truth join missing: {example_id}")
        root_index = int(pred["root_index"])
        if int(blind["root_index"]) != root_index or int(sealed["root_index"]) != root_index:
            raise RuntimeError("fresh root join drift")
        root = sealed_roots.get(root_index)
        if root is None:
            raise RuntimeError(f"sealed root missing: {root_index}")
        mechanism = str(sealed["mechanism"])
        if mechanism not in frame.TARGET:
            raise RuntimeError(f"unexpected fresh mechanism: {mechanism}")
        loaded = confirm_scorer.load_x_only(product / str(blind["artifact_path"]), str(blind["artifact_sha256"]), example_id)
        delta, weights, pairs = frame.measured_delta_and_weights(loaded)

        if root_index not in root_cache:
            clean = frame.circuit_from_serialized(loaded)
            candidates = latent.plausible_candidates(clean)
            finite_jacs = latent.finite_shot_candidate_jacobians(clean, pairs, candidates, root_index, latent_cfg)
            exact_jacs = latent.exact_candidate_jacobians(clean, pairs, candidates)
            distance, gate = equiv.finite_frame_pair_geometry(finite_jacs)
            true_location = (int(root["affected_qubit"]), int(root["injection_boundary_rank"]))
            if true_location in candidates:
                true_index = int(candidates.index(true_location))
                true_finite_jac = finite_jacs[true_index]
                true_exact_jac = exact_jacs[true_index]
            else:
                true_index = -1
                true_finite_jac = latent.finite_shot_candidate_jacobians(clean, pairs, [true_location], root_index, latent_cfg)[0]
                true_exact_jac = frame.frame_response_jacobian(clean, true_location[1], true_location[0], pairs)
            eq_mask = np.asarray(
                [response_equivalent(jac, true_exact_jac, response_spec) for jac in exact_jacs],
                dtype=np.bool_,
            )
            tq_mask = np.asarray([q == true_location[0] for q, _b in candidates], dtype=np.bool_)
            geometry = frame.frame_geometry(true_exact_jac)
            root_cache[root_index] = {
                "clean": clean,
                "candidates": candidates,
                "finite_jacs": finite_jacs,
                "exact_jacs": exact_jacs,
                "distance": distance,
                "gate": gate,
                "true_location": true_location,
                "true_index": true_index,
                "true_finite_jac": true_finite_jac,
                "true_exact_jac": true_exact_jac,
                "eq_mask": eq_mask,
                "true_qubit_mask": tq_mask,
                "min_angle": float(geometry["minimum_pairwise_axis_angle_deg"]),
            }
        rc = root_cache[root_index]
        per_candidate = np.stack([frame.canonicalize_evidence(delta, jac, weights)[0] for jac in rc["finite_jacs"]]).astype(np.float32)
        profiles = latent.profile_log_likelihoods(delta, weights, rc["finite_jacs"]).astype(np.float64)
        true_finite = frame.canonicalize_evidence(delta, rc["true_finite_jac"], weights)[0].astype(np.float32)
        true_exact = frame.canonicalize_evidence(delta, rc["true_exact_jac"], weights)[0].astype(np.float32)

        candidate_features.append(per_candidate)
        profiles_all.append(profiles)
        distances.append(rc["distance"])
        gates.append(rc["gate"])
        true_finite_features.append(true_finite)
        true_exact_features.append(true_exact)
        true_qubit_masks.append(rc["true_qubit_mask"])
        equivalence_masks.append(rc["eq_mask"])
        true_indices.append(int(rc["true_index"]))
        exact_present.append(int(rc["true_index"]) >= 0)
        truth.append(int(frame.TARGET[mechanism]))
        mechanism_names.append(mechanism)
        families.append(str(sealed["family_id"]))
        ids.append(example_id)
        root_indices.append(root_index)
        n_q = int(rc["clean"].num_qubits)
        n_g = int(len(rc["clean"].data))
        tq, tb = rc["true_location"]
        qubit_count.append(n_q)
        gate_count.append(n_g)
        true_qubits.append(int(tq))
        true_boundaries.append(int(tb))
        normalized_boundary.append(float(tb / max(n_g, 1)))
        candidate_sets.append(list(rc["candidates"]))
        min_angles.append(float(rc["min_angle"]))
        snr_proxy.append(float(np.sqrt(np.mean(np.square(delta * weights)))))
        equivalence_sizes.append(int(np.sum(rc["eq_mask"])))
        if progress_every and pos % progress_every == 0:
            print(f"fresh posthoc extraction {pos}/{len(prediction_rows)} roots_cached={len(root_cache)}", flush=True)

    x, mask = latent.pad_candidate_features(candidate_features)
    max_candidates = int(x.shape[1])
    profile_pad = blind_predict.pad_profiles(profiles_all, max_candidates)
    distance_pad = blind_predict.pad_pair_float(distances, max_candidates)
    gate_pad = blind_predict.pad_pair_bool(gates, max_candidates)

    def pad_bool(values: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(values), max_candidates), dtype=np.bool_)
        for i, value in enumerate(values):
            out[i, : len(value)] = value
        return out

    y = np.asarray(truth, dtype=np.int64)
    counts = Counter(int(v) for v in y.tolist())
    if len(y) != 7200 or counts != Counter({0: 2400, 1: 2400, 2: 2400}):
        raise RuntimeError(f"fresh posthoc balance drift: {counts}")
    return {
        "x": x,
        "mask": mask,
        "profiles": profile_pad,
        "distance": distance_pad,
        "gate": gate_pad,
        "true_finite_features": np.stack(true_finite_features).astype(np.float32),
        "true_exact_features": np.stack(true_exact_features).astype(np.float32),
        "true_qubit_mask": pad_bool(true_qubit_masks),
        "equivalence_mask": pad_bool(equivalence_masks),
        "true_index": np.asarray(true_indices, dtype=np.int64),
        "exact_present": np.asarray(exact_present, dtype=np.bool_),
        "truth": y,
        "mechanism_name": np.asarray(mechanism_names, dtype=object),
        "family": np.asarray(families, dtype=object),
        "example_id": ids,
        "root_index": np.asarray(root_indices, dtype=np.int64),
        "qubit_count": np.asarray(qubit_count, dtype=np.int64),
        "gate_count": np.asarray(gate_count, dtype=np.int64),
        "true_qubit": np.asarray(true_qubits, dtype=np.int64),
        "true_boundary": np.asarray(true_boundaries, dtype=np.int64),
        "normalized_boundary": np.asarray(normalized_boundary, dtype=np.float64),
        "candidate_sets": candidate_sets,
        "minimum_frame_angle": np.asarray(min_angles, dtype=np.float64),
        "snr_proxy": np.asarray(snr_proxy, dtype=np.float64),
        "equivalence_size": np.asarray(equivalence_sizes, dtype=np.int64),
        "root_count": len(root_cache),
    }


def build_fit_true_frame_table(product: Path, latent_cfg: Mapping[str, Any], progress_every: int) -> dict[str, np.ndarray]:
    cfg = step14.read_json(ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json")
    step14.assert_contract(cfg)
    rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(product, cfg)
    roots = {
        int(r["root_index"]): r
        for r in baseline.read_csv(product / "manifests" / "root_manifest.csv")
        if str(r.get("step14_partition")) == "fit"
    }
    selected = [r for r in rows if str(r.get("step14_partition")) == "fit" and str(r.get("mechanism")) in frame.TARGET]
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    finite: list[np.ndarray] = []
    exact: list[np.ndarray] = []
    truth: list[int] = []
    for pos, row in enumerate(selected, start=1):
        root_index = int(row["root_index"])
        root = roots.get(root_index)
        if root is None:
            raise RuntimeError("FIT root missing in posthoc reference extractor")
        artifact = product / str(row["artifact_path"])
        if sha(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"FIT artifact hash mismatch: {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as src:
            loaded = {str(k): np.asarray(src[k]) for k in src.files if str(k).startswith("x__")}
        delta, weights, pairs = frame.measured_delta_and_weights(loaded)
        if root_index not in cache:
            clean = frame.circuit_from_serialized(loaded)
            loc = (int(root["affected_qubit"]), int(root["injection_boundary_rank"]))
            fj = latent.finite_shot_candidate_jacobians(clean, pairs, [loc], root_index, latent_cfg)[0]
            ej = frame.frame_response_jacobian(clean, loc[1], loc[0], pairs)
            cache[root_index] = (fj, ej)
        fj, ej = cache[root_index]
        finite.append(frame.canonicalize_evidence(delta, fj, weights)[0].astype(np.float32))
        exact.append(frame.canonicalize_evidence(delta, ej, weights)[0].astype(np.float32))
        truth.append(int(frame.TARGET[str(row["mechanism"])]))
        if progress_every and pos % progress_every == 0:
            print(f"FIT true-frame extraction {pos}/{len(selected)} roots_cached={len(cache)}", flush=True)
    y = np.asarray(truth, dtype=np.int64)
    if len(y) != 28800 or Counter(y.tolist()) != Counter({0: 9600, 1: 9600, 2: 9600}):
        raise RuntimeError("FIT true-frame posthoc balance/count drift")
    return {"finite": np.stack(finite).astype(np.float32), "exact": np.stack(exact).astype(np.float32), "truth": y}


def pool_subset_batch(
    candidate_logits: np.ndarray,
    table: Mapping[str, Any],
    subset_mask: np.ndarray,
    tau: float,
    temperature: float,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(subset_mask)
    out = np.empty((n, 3), dtype=np.float64)
    covered = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        count = int(table["mask"][i].sum())
        indices = np.flatnonzero(np.asarray(subset_mask[i, :count], dtype=np.bool_))
        if len(indices) == 0:
            if fallback is None:
                out[i] = np.nan
            else:
                out[i] = fallback[i]
            continue
        covered[i] = True
        out[i] = equiv.equivalence_aware_pool_one(
            candidate_logits[i, indices],
            table["profiles"][i, indices],
            table["distance"][i][np.ix_(indices, indices)],
            table["gate"][i][np.ix_(indices, indices)],
            tau=tau,
            frame_temperature=temperature,
        )
    return out, covered


def grouped_record(y: np.ndarray, pred: np.ndarray, oracle_pred: np.ndarray, indices: np.ndarray, entropy: np.ndarray, snr: np.ndarray) -> dict[str, Any]:
    idx = np.asarray(indices, dtype=np.int64)
    yy = y[idx]
    pp = pred[idx]
    op = oracle_pred[idx]
    recalls: dict[str, float] = {}
    for cls in sorted(set(int(v) for v in yy.tolist())):
        m = yy == cls
        recalls[str(cls)] = float(np.mean(pp[m] == yy[m]))
    return {
        "count": int(len(idx)),
        "accuracy": float(np.mean(pp == yy)),
        "error_rate": float(np.mean(pp != yy)),
        "balanced_accuracy_over_present_classes": float(np.mean(list(recalls.values()))) if recalls else 0.0,
        "recall_by_present_class": recalls,
        "oracle_correct_current_wrong_fraction": float(np.mean((op == yy) & (pp != yy))),
        "mean_candidate_posterior_entropy": float(np.mean(entropy[idx])),
        "mean_finite_shot_snr_proxy": float(np.mean(snr[idx])),
    }


def categorical_strata(name_values: Sequence[Any], y: np.ndarray, pred: np.ndarray, oracle_pred: np.ndarray, entropy: np.ndarray, snr: np.ndarray) -> dict[str, Any]:
    values = np.asarray(name_values, dtype=object)
    out: dict[str, Any] = {}
    for value in sorted(set(values.tolist()), key=lambda x: str(x)):
        idx = np.flatnonzero(values == value)
        out[str(value)] = grouped_record(y, pred, oracle_pred, idx, entropy, snr)
    return out


def fixed_bins(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    labels = np.empty(len(values), dtype=object)
    for i, value in enumerate(np.asarray(values, dtype=np.float64)):
        assigned = False
        for j in range(len(edges) - 1):
            if float(edges[j]) <= value < float(edges[j + 1]):
                labels[i] = f"[{edges[j]:.6g},{edges[j+1]:.6g})"
                assigned = True
                break
        if not assigned:
            labels[i] = "out_of_range"
    return labels


def quantile_labels(values: np.ndarray, q: int) -> tuple[np.ndarray, list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    edges = np.quantile(arr, np.linspace(0.0, 1.0, q + 1)).astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        k = int(np.searchsorted(edges[1:-1], v, side="right"))
        labels[i] = f"Q{k+1}"
    return labels, [float(v) for v in edges]


def write_per_example(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("empty per-example posthoc table")
    fields = list(rows[0].keys())
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        h.flush()
        os.fsync(h.fileno())
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    oracle_compat.apply_frozen_support_bound()
    cfg = read_json(POSTHOC_CONFIG)
    freeze = read_json(FINAL_FREEZE)
    confirm_cfg = read_json(CONFIRM_CONFIG)
    if cfg.get("status") != "FROZEN_POST_CONFIRMATION_BEFORE_POSTHOC_EXECUTION":
        raise RuntimeError("posthoc protocol is not frozen")
    src = cfg["source_identity"]
    required = {
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "final_method_freeze_payload_sha256": args.final_method_freeze_payload_sha256,
        "fresh_holdout_product_id": args.fresh_holdout_product_id,
        "fresh_holdout_dataset_complete_sha256": args.fresh_holdout_dataset_complete_sha256,
        "oracle_free_predictions_complete_sha256": args.oracle_free_prediction_complete_sha256,
        "confirmatory_result_sha256": args.confirmatory_result_sha256,
        "confirmatory_complete_sha256": args.confirmatory_complete_sha256,
    }
    for key, actual in required.items():
        if str(src[key]) != str(actual):
            raise RuntimeError(f"posthoc source identity drift: {key}")
    if str(freeze["freeze_payload_sha256"]) != args.final_method_freeze_payload_sha256:
        raise RuntimeError("final method freeze payload drift")
    rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)

    confirm_pointer = read_json(CONFIRM_PARENT / "current_confirmatory_result.json")
    confirm_dir = Path(str(confirm_pointer["result_dir"])).resolve()
    if sha(confirm_dir / "confirmatory_result.json") != args.confirmatory_result_sha256:
        raise RuntimeError("confirmatory result bytes drift")
    if sha(confirm_dir / "confirmatory_complete.json") != args.confirmatory_complete_sha256:
        raise RuntimeError("confirmatory completion bytes drift")
    confirm_result = read_json(confirm_dir / "confirmatory_result.json")
    if confirm_result.get("status") != "COMPLETE_ONE_SHOT_FRESH_HOLDOUT_CONFIRMATORY_EVALUATION":
        raise RuntimeError("confirmatory source is not complete")

    pred_pointer = read_json(CONFIRM_PARENT / "current_oracle_free_predictions.json")
    pred_rows, frozen_logits, pred_complete = confirm_scorer.prediction_rows_and_logits(
        pred_pointer, args.oracle_free_prediction_complete_sha256
    )
    if str(pred_complete["oracle_free_predictions_sha256"]) != str(src["oracle_free_predictions_sha256"]):
        raise RuntimeError("frozen prediction CSV identity drift")

    hold_pointer = read_json(HOLDOUT_PARENT / "current_fresh_holdout.json")
    product = Path(str(hold_pointer["product_dir"])).resolve()
    if str(hold_pointer["product_id"]) != args.fresh_holdout_product_id:
        raise RuntimeError("fresh holdout product drift")
    if sha(product / "dataset_complete.json") != args.fresh_holdout_dataset_complete_sha256:
        raise RuntimeError("fresh holdout completion bytes drift")

    latent_cfg = latent.load_frozen_config()
    response_spec = cfg["response_equivalence"]
    table = extract_fresh_table(product, pred_rows, latent_cfg, response_spec, args.progress_every)
    y = table["truth"]
    families = table["family"]
    device = resolve_device(args.device)

    method = confirm_cfg["method_identity"]
    tau = float(method["tau"])
    temperature = float(method["frame_temperature"])
    method_pointer = read_json(METHOD_PARENT / "current_equivalence_aware_fit.json")
    method_dir = Path(str(method_pointer["method_dir"])).resolve()

    seed_candidate_logits: list[np.ndarray] = []
    seed_full_logits: list[np.ndarray] = []
    seed_true_finite_logits: list[np.ndarray] = []
    seed_true_exact_logits: list[np.ndarray] = []
    single_mask = np.ones((len(y), 1), dtype=np.bool_)
    true_finite_bag = table["true_finite_features"][:, None, :]
    true_exact_bag = table["true_exact_features"][:, None, :]
    for seed in [int(v) for v in method["candidate_network_seeds"]]:
        name = f"candidate_network_seed{seed}.pt"
        path = method_dir / name
        if sha(path) != str(method["checkpoint_sha256"][name]):
            raise RuntimeError(f"posthoc checkpoint hash drift: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _uniform, candidates = equiv.infer_candidate_network(
            table["x"], table["mask"], state_dict=payload["state_dict"],
            mean=np.asarray(payload["normalization_mean"], dtype=np.float32),
            std=np.asarray(payload["normalization_std"], dtype=np.float32),
            latent_cfg=latent_cfg, device=device,
        )
        full = equiv.equivalence_aware_pool_batch(
            candidates, table["profiles"], table["distance"], table["gate"], table["mask"],
            tau=tau, frame_temperature=temperature,
        )
        _u1, tf = equiv.infer_candidate_network(
            true_finite_bag, single_mask, state_dict=payload["state_dict"],
            mean=np.asarray(payload["normalization_mean"], dtype=np.float32),
            std=np.asarray(payload["normalization_std"], dtype=np.float32),
            latent_cfg=latent_cfg, device=device,
        )
        _u2, te = equiv.infer_candidate_network(
            true_exact_bag, single_mask, state_dict=payload["state_dict"],
            mean=np.asarray(payload["normalization_mean"], dtype=np.float32),
            std=np.asarray(payload["normalization_std"], dtype=np.float32),
            latent_cfg=latent_cfg, device=device,
        )
        seed_candidate_logits.append(candidates)
        seed_full_logits.append(full)
        seed_true_finite_logits.append(tf[:, 0, :])
        seed_true_exact_logits.append(te[:, 0, :])
        print(f"posthoc frozen candidate network seed={seed} complete", flush=True)

    candidate_logits = np.mean(np.stack(seed_candidate_logits), axis=0)
    reproduced_full = np.mean(np.stack(seed_full_logits), axis=0)
    if not np.array_equal(np.argmax(reproduced_full, axis=1), np.argmax(frozen_logits, axis=1)):
        raise RuntimeError("posthoc reproduction changed a frozen Phase-A argmax")
    max_logit_abs_diff = float(np.max(np.abs(reproduced_full - frozen_logits)))
    if max_logit_abs_diff > 1e-4:
        raise RuntimeError(f"posthoc frozen-logit reproduction drift: {max_logit_abs_diff}")

    true_qubit_logits, true_qubit_covered = pool_subset_batch(
        candidate_logits, table, table["true_qubit_mask"], tau, temperature, fallback=frozen_logits
    )
    equiv_logits, equiv_covered = pool_subset_batch(
        candidate_logits, table, table["equivalence_mask"], tau, temperature, fallback=frozen_logits
    )
    same_head_true_finite = np.mean(np.stack(seed_true_finite_logits), axis=0)
    same_head_true_exact = np.mean(np.stack(seed_true_exact_logits), axis=0)

    no_density_logits = np.empty_like(frozen_logits)
    posterior_entropy = np.zeros(len(y), dtype=np.float64)
    posterior_entropy_norm = np.zeros(len(y), dtype=np.float64)
    true_rank = np.full(len(y), -1, dtype=np.int64)
    equiv_best_rank = np.full(len(y), -1, dtype=np.int64)
    equiv_mass = np.zeros(len(y), dtype=np.float64)
    equiv_mass_no_density = np.zeros(len(y), dtype=np.float64)
    true_mass = np.zeros(len(y), dtype=np.float64)
    for i in range(len(y)):
        count = int(table["mask"][i].sum())
        profiles = table["profiles"][i, :count]
        dist = table["distance"][i, :count, :count]
        gate = table["gate"][i, :count, :count]
        post, _kernel, _density = posterior_and_kernel(profiles, dist, gate, tau, temperature, density_corrected=True)
        post0, _kernel0, _density0 = posterior_and_kernel(profiles, dist, gate, tau, temperature, density_corrected=False)
        order = np.argsort(-post, kind="stable")
        ranks = np.empty(count, dtype=np.int64)
        ranks[order] = np.arange(1, count + 1)
        ti = int(table["true_index"][i])
        if ti >= 0:
            true_rank[i] = int(ranks[ti])
            true_mass[i] = float(post[ti])
        eqm = np.asarray(table["equivalence_mask"][i, :count], dtype=np.bool_)
        if np.any(eqm):
            equiv_best_rank[i] = int(np.min(ranks[eqm]))
            equiv_mass[i] = float(np.sum(post[eqm]))
            equiv_mass_no_density[i] = float(np.sum(post0[eqm]))
        posterior_entropy[i] = float(-np.sum(post * np.log(np.maximum(post, 1e-300))))
        posterior_entropy_norm[i] = float(posterior_entropy[i] / max(math.log(count), 1e-12)) if count > 1 else 0.0
        no_density_logits[i] = pool_one_no_density(candidate_logits[i, :count], profiles, dist, gate, tau, temperature)

    fit_product = step14.resolve_cross_product(None)
    if str(read_json(fit_product / "dataset_complete.json")["product_id"]) != str(freeze["source_identity"]["development_product_id"]):
        raise RuntimeError("development product drift during posthoc")
    if sha(fit_product / "dataset_complete.json") != str(freeze["source_identity"]["development_dataset_sha256"]):
        raise RuntimeError("development dataset bytes drift during posthoc")
    fit_table = build_fit_true_frame_table(fit_product, latent_cfg, args.progress_every)
    seeds = [int(v) for v in confirm_cfg["phase_b_confirmatory_scoring"]["privileged_exact_local_frame_reference"]["probe_seeds"]]
    finite_probe_logits, finite_probe_seeds = confirm_scorer.ensemble_probe(
        fit_table["finite"], fit_table["truth"], table["true_finite_features"], y, seeds, device
    )
    exact_probe_logits, exact_probe_seeds = confirm_scorer.ensemble_probe(
        fit_table["exact"], fit_table["truth"], table["true_exact_features"], y, seeds, device
    )

    current_ba = float(oracle.metric_record(y, frozen_logits)["mechanism_balanced_accuracy"])
    if abs(current_ba - float(src["confirmatory_oracle_free_ba"])) > 1e-12:
        raise RuntimeError("current frozen BA does not reproduce confirmatory source")
    exact_ba = float(oracle.metric_record(y, exact_probe_logits)["mechanism_balanced_accuracy"])
    if abs(exact_ba - float(src["confirmatory_exact_oracle_ba"])) > 1e-6:
        raise RuntimeError(f"exact oracle probe reproduction drift: {exact_ba}")

    ladder_logits = {
        "frozen_oracle_free_equivalence_aware": frozen_logits,
        "same_head_true_qubit_restricted": true_qubit_logits,
        "same_head_true_boundary_finite_frame": same_head_true_finite,
        "same_head_true_response_equivalence_class": equiv_logits,
        "fit_only_finite_shot_true_frame_small_probe": finite_probe_logits,
        "fit_only_exact_true_frame_small_probe": exact_probe_logits,
    }
    ladder_metrics = {name: metric(y, value, "posthoc_spent_holdout") for name, value in ladder_logits.items()}
    ladder_metrics["same_head_true_boundary_exact_frame"] = metric(y, same_head_true_exact, "supplemental_same_frozen_head_exact_frame_intervention")

    ladder_names = list(cfg["oracle_ladder"])
    paired: dict[str, Any] = {}
    bs_seed = int(cfg["statistics"]["paired_family_bootstrap_seed"])
    for j in range(1, len(ladder_names)):
        a, b = ladder_names[j], ladder_names[j - 1]
        paired[f"{a}_minus_{b}"] = oracle.bootstrap_delta(y, ladder_logits[a], ladder_logits[b], families, seed=bs_seed + j)
    for j, name in enumerate(ladder_names[1:], start=100):
        paired[f"{name}_minus_frozen_oracle_free"] = oracle.bootstrap_delta(y, ladder_logits[name], frozen_logits, families, seed=bs_seed + j)

    seed_predictions = np.stack([np.argmax(v, axis=1) for v in seed_full_logits], axis=1)
    seed_distinct = np.asarray([len(set(int(v) for v in row.tolist())) for row in seed_predictions], dtype=np.int64)
    current_pred = np.argmax(frozen_logits, axis=1)
    oracle_pred = np.argmax(exact_probe_logits, axis=1)
    no_density_pred = np.argmax(no_density_logits, axis=1)
    current_correct = current_pred == y
    no_density_correct = no_density_pred == y
    density_help = (~no_density_correct) & current_correct
    density_hurt = no_density_correct & (~current_correct)

    exact_coverage = float(np.mean(table["exact_present"]))
    equiv_present = table["equivalence_size"] > 0
    eq_coverage = float(np.mean(equiv_present))
    coverage_ranking = {
        "exact_true_frame_candidate_coverage": exact_coverage,
        "response_equivalent_candidate_coverage": eq_coverage,
        "true_qubit_subset_coverage": float(np.mean(true_qubit_covered)),
        "response_equivalence_subset_coverage": float(np.mean(equiv_covered)),
        "candidate_count": quantiles([int(v) for v in table["mask"].sum(axis=1).tolist()]),
        "exact_true_candidate_anchor_posterior_rank_when_present": quantiles(true_rank[true_rank > 0].astype(float).tolist()),
        "best_equivalent_candidate_anchor_posterior_rank_when_present": quantiles(equiv_best_rank[equiv_best_rank > 0].astype(float).tolist()),
        "exact_true_candidate_anchor_posterior_mass_when_present": quantiles(true_mass[table["exact_present"]].tolist()),
        "true_response_equivalence_class_anchor_posterior_mass": quantiles(equiv_mass.tolist()),
        "candidate_anchor_posterior_entropy_nats": quantiles(posterior_entropy.tolist()),
        "candidate_anchor_posterior_entropy_normalized": quantiles(posterior_entropy_norm.tolist()),
        "density_correction": {
            "frozen_density_corrected_ba": float(oracle.metric_record(y, frozen_logits)["mechanism_balanced_accuracy"]),
            "no_density_counterfactual_ba": float(oracle.metric_record(y, no_density_logits)["mechanism_balanced_accuracy"]),
            "ba_delta_density_minus_no_density": float(
                oracle.metric_record(y, frozen_logits)["mechanism_balanced_accuracy"]
                - oracle.metric_record(y, no_density_logits)["mechanism_balanced_accuracy"]
            ),
            "examples_helped": int(np.sum(density_help)),
            "examples_hurt": int(np.sum(density_hurt)),
            "examples_prediction_changed": int(np.sum(current_pred != no_density_pred)),
            "equivalence_mass_delta_density_minus_no_density": quantiles((equiv_mass - equiv_mass_no_density).tolist()),
        },
    }

    boundary_labels = fixed_bins(table["normalized_boundary"], cfg["error_stratification"]["boundary_depth"]["bins"])
    angle_labels = fixed_bins(table["minimum_frame_angle"], cfg["error_stratification"]["true_frame_minimum_pairwise_axis_angle_deg_bins"])
    eqsize_edges = cfg["error_stratification"]["response_equivalence_class_size_bins"]
    eqsize_labels = fixed_bins(table["equivalence_size"].astype(float), eqsize_edges)
    snr_labels, snr_edges = quantile_labels(table["snr_proxy"], int(cfg["error_stratification"]["finite_shot_snr_quantile_bins"]))
    ent_labels, ent_edges = quantile_labels(posterior_entropy, int(cfg["error_stratification"]["candidate_posterior_entropy_quantile_bins"]))
    error_stratification = {
        "mechanism_class": categorical_strata(table["mechanism_name"], y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "qubit_count": categorical_strata(table["qubit_count"], y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "circuit_family": categorical_strata(table["family"], y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "boundary_depth": categorical_strata(boundary_labels, y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "true_frame_minimum_pairwise_axis_angle": categorical_strata(angle_labels, y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "response_equivalence_class_size": categorical_strata(eqsize_labels, y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "finite_shot_snr_proxy_quantile": categorical_strata(snr_labels, y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "candidate_posterior_entropy_quantile": categorical_strata(ent_labels, y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "seed_disagreement_distinct_predictions": categorical_strata(seed_distinct, y, current_pred, oracle_pred, posterior_entropy, table["snr_proxy"]),
        "quantile_edges": {"finite_shot_snr_proxy": snr_edges, "candidate_posterior_entropy": ent_edges},
    }

    per_rows: list[dict[str, Any]] = []
    pred_maps = {name: np.argmax(logits, axis=1) for name, logits in ladder_logits.items()}
    for i in range(len(y)):
        per_rows.append({
            "example_id": table["example_id"][i],
            "family_id": str(table["family"][i]),
            "root_index": int(table["root_index"][i]),
            "mechanism": str(table["mechanism_name"][i]),
            "truth_index": int(y[i]),
            "current_pred": int(current_pred[i]),
            "current_correct": int(current_correct[i]),
            "exact_oracle_pred": int(oracle_pred[i]),
            "oracle_correct_current_wrong": int((oracle_pred[i] == y[i]) and (current_pred[i] != y[i])),
            "qubit_count": int(table["qubit_count"][i]),
            "gate_count": int(table["gate_count"][i]),
            "true_qubit": int(table["true_qubit"][i]),
            "true_boundary_rank": int(table["true_boundary"][i]),
            "normalized_boundary_depth": format(float(table["normalized_boundary"][i]), ".17g"),
            "candidate_count": int(table["mask"][i].sum()),
            "exact_true_candidate_present": int(table["exact_present"][i]),
            "true_candidate_posterior_rank": int(true_rank[i]),
            "equivalent_candidate_present": int(equiv_present[i]),
            "equivalence_class_size": int(table["equivalence_size"][i]),
            "best_equivalent_candidate_posterior_rank": int(equiv_best_rank[i]),
            "equivalence_class_posterior_mass": format(float(equiv_mass[i]), ".17g"),
            "posterior_entropy_nats": format(float(posterior_entropy[i]), ".17g"),
            "posterior_entropy_normalized": format(float(posterior_entropy_norm[i]), ".17g"),
            "finite_shot_snr_proxy": format(float(table["snr_proxy"][i]), ".17g"),
            "true_frame_min_angle_deg": format(float(table["minimum_frame_angle"][i]), ".17g"),
            "seed_distinct_predictions": int(seed_distinct[i]),
            "no_density_pred": int(no_density_pred[i]),
            "density_helped": int(density_help[i]),
            "density_hurt": int(density_hurt[i]),
            "true_qubit_restricted_pred": int(pred_maps["same_head_true_qubit_restricted"][i]),
            "same_head_true_boundary_pred": int(pred_maps["same_head_true_boundary_finite_frame"][i]),
            "response_equivalence_oracle_pred": int(pred_maps["same_head_true_response_equivalence_class"][i]),
            "finite_true_frame_probe_pred": int(pred_maps["fit_only_finite_shot_true_frame_small_probe"][i]),
            "exact_true_frame_probe_pred": int(pred_maps["fit_only_exact_true_frame_small_probe"][i]),
        })

    protocol_sha = sha(POSTHOC_CONFIG)
    key = hashlib.sha256((protocol_sha + args.confirmatory_result_sha256).encode()).hexdigest()[:24]
    diagnostic_id = f"residual_bottleneck_{key}"
    out_dir = args.output_parent.expanduser().resolve() / diagnostic_id
    if out_dir.exists() and any(out_dir.iterdir()):
        existing = read_json(out_dir / "diagnostic_complete.json")
        if existing.get("status") == "COMPLETE_SPENT_HOLDOUT_POSTHOC_RESIDUAL_BOTTLENECK_DIAGNOSTIC":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
        raise RuntimeError(f"refusing to overwrite incomplete diagnostic: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle_ladder = {
        "schema": "triqto.v0_2.step14_fresh_holdout_oracle_ladder.v1",
        "metrics": ladder_metrics,
        "paired_family_bootstrap": paired,
        "same_head_intervention_note": "same_head_true_boundary_finite_frame and same_head_true_boundary_exact_frame use the frozen candidate-network weights/normalization with the true frame supplied after confirmation; no retraining occurs",
        "coverage": {
            "true_qubit": float(np.mean(true_qubit_covered)),
            "response_equivalence": float(np.mean(equiv_covered)),
        },
        "finite_vs_exact_true_frame_probe_ba_gap": float(
            ladder_metrics["fit_only_exact_true_frame_small_probe"]["mechanism_balanced_accuracy"]
            - ladder_metrics["fit_only_finite_shot_true_frame_small_probe"]["mechanism_balanced_accuracy"]
        ),
    }
    candidate_path = out_dir / "candidate_coverage_ranking.json"
    oracle_path = out_dir / "oracle_ladder.json"
    strata_path = out_dir / "error_stratification.json"
    per_path = out_dir / "per_example_posthoc.csv"
    atomic_json(candidate_path, coverage_ranking)
    atomic_json(oracle_path, oracle_ladder)
    atomic_json(strata_path, error_stratification)
    write_per_example(per_path, per_rows)

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_SPENT_HOLDOUT_POSTHOC_RESIDUAL_BOTTLENECK_DIAGNOSTIC",
        "diagnostic_id": diagnostic_id,
        "protocol_sha256": protocol_sha,
        "source_confirmatory_result_sha256": args.confirmatory_result_sha256,
        "source_confirmatory_complete_sha256": args.confirmatory_complete_sha256,
        "source_prediction_complete_sha256": args.oracle_free_prediction_complete_sha256,
        "frozen_logit_reproduction_max_abs_diff": max_logit_abs_diff,
        "oracle_ladder": ladder_metrics,
        "candidate_coverage_ranking_summary": coverage_ranking,
        "finite_vs_exact_true_frame_probe_ba_gap": oracle_ladder["finite_vs_exact_true_frame_probe_ba_gap"],
        "scientific_boundaries": {
            "spent_holdout_posthoc_only": True,
            "confirmatory_verdict_changed": False,
            "retraining_performed": False,
            "candidate_network_weights_updated": False,
            "hyperparameters_reselected": False,
            "thresholds_changed": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
            "may_guide_v2_development": True,
            "may_be_reused_as_new_confirmation": False,
        },
        "reference_probe_seeds": seeds,
        "finite_probe_seed_records": finite_probe_seeds,
        "exact_probe_seed_records": exact_probe_seeds,
    }
    result_path = out_dir / "diagnostic_result.json"
    atomic_json(result_path, result)
    completion = {
        "schema": "triqto.v0_2.step14_fresh_holdout_residual_bottleneck_complete.v1",
        "status": "COMPLETE_SPENT_HOLDOUT_POSTHOC_RESIDUAL_BOTTLENECK_DIAGNOSTIC",
        "diagnostic_id": diagnostic_id,
        "diagnostic_dir": str(out_dir),
        "diagnostic_result_sha256": sha(result_path),
        "oracle_ladder_sha256": sha(oracle_path),
        "candidate_coverage_ranking_sha256": sha(candidate_path),
        "error_stratification_sha256": sha(strata_path),
        "per_example_posthoc_sha256": sha(per_path),
        "confirmatory_verdict_preserved": str(confirm_result["verdict"]),
        "holdout_is_spent": True,
    }
    complete_path = out_dir / "diagnostic_complete.json"
    atomic_json(complete_path, completion)
    complete_sha = sha(complete_path)
    atomic_json(args.output_parent.expanduser().resolve() / "current_residual_bottleneck_posthoc.json", {
        "schema": "triqto.v0_2.step14_fresh_holdout_residual_bottleneck_pointer.v1",
        "status": completion["status"],
        "diagnostic_id": diagnostic_id,
        "diagnostic_dir": str(out_dir),
        "diagnostic_result_sha256": completion["diagnostic_result_sha256"],
        "diagnostic_complete_sha256": complete_sha,
    })
    print(json.dumps({**completion, "diagnostic_complete_sha256": complete_sha, "headline": {
        "oracle_free_ba": ladder_metrics["frozen_oracle_free_equivalence_aware"]["mechanism_balanced_accuracy"],
        "true_qubit_ba": ladder_metrics["same_head_true_qubit_restricted"]["mechanism_balanced_accuracy"],
        "same_head_true_boundary_finite_ba": ladder_metrics["same_head_true_boundary_finite_frame"]["mechanism_balanced_accuracy"],
        "response_equivalence_ba": ladder_metrics["same_head_true_response_equivalence_class"]["mechanism_balanced_accuracy"],
        "finite_true_frame_probe_ba": ladder_metrics["fit_only_finite_shot_true_frame_small_probe"]["mechanism_balanced_accuracy"],
        "exact_true_frame_probe_ba": ladder_metrics["fit_only_exact_true_frame_small_probe"]["mechanism_balanced_accuracy"],
        "same_head_true_exact_frame_ba": ladder_metrics["same_head_true_boundary_exact_frame"]["mechanism_balanced_accuracy"],
        "exact_candidate_coverage": exact_coverage,
        "equivalent_candidate_coverage": eq_coverage,
        "density_correction_ba_delta": coverage_ranking["density_correction"]["ba_delta_density_minus_no_density"],
    }}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
