#!/usr/bin/env python3
"""Run Step 6 cheap, interpretable baselines on the frozen Step 5 v3 cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "triqto.v0_2.step6_cheap_baseline_benchmark.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step6_cheap_baseline_benchmark.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step6_cheap_baselines")
MECHANISM_NAMES = ("rz_drift", "rx_overrotation", "ry_overrotation")
INTEGRATED_NAMES = ("no_effect",) + MECHANISM_NAMES
DEPLOYABLE_VARIANTS = (
    "context_only", "graph_stats_only", "diag_local", "diag_local_pairwise",
    "diag_full", "diag_full_context", "diag_full_graph", "diag_full_context_graph",
)
PRIVILEGED_VARIANTS = ("exact_diag_full_oracle", "family_oracle")
SANITY_BASELINES = ("majority_prior", "stratified_random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"cannot parse boolean {value!r}")


def verify_source_product(product: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    source = config["source_dataset"]
    complete = read_json(product / "dataset_complete.json")
    expected = {
        "schema": source["schema"],
        "product_id": source["product_id"],
        "clean_circuit_root_count": int(source["clean_circuit_root_count"]),
        "example_count": int(source["example_count"]),
        "train_clean_root_count": int(source["train_clean_root_count"]),
        "validation_clean_root_count": int(source["validation_clean_root_count"]),
    }
    for key, expected_value in expected.items():
        if complete.get(key) != expected_value:
            raise RuntimeError(f"source dataset mismatch for {key}: {complete.get(key)!r} != {expected_value!r}")
    manifests = product / "manifests"
    for name, expected_hash in complete.get("manifest_hashes", {}).items():
        if sha256_file(manifests / name) != expected_hash:
            raise RuntimeError(f"source manifest hash mismatch: {name}")
    if sha256_file(product / "stage_validation.json") != complete.get("stage_validation_sha256"):
        raise RuntimeError("source stage_validation.json hash mismatch")
    examples = read_csv(manifests / "example_manifest.csv")
    if len(examples) != expected["example_count"]:
        raise RuntimeError("example manifest count mismatch")
    return complete, examples


def fixed_pair_slots(max_n: int) -> dict[tuple[int, int], int]:
    pairs = [(i, j) for i in range(max_n) for j in range(i + 1, max_n)]
    return {pair: index for index, pair in enumerate(pairs)}


def reorder_basis(matrix: np.ndarray, basis_codes: np.ndarray) -> np.ndarray:
    codes = [int(v) for v in np.asarray(basis_codes).reshape(-1).tolist()]
    if sorted(codes) != [0, 1, 2]:
        raise RuntimeError(f"unexpected diagnostic basis codes {codes}")
    source = np.asarray(matrix)
    if source.shape[0] != 3:
        raise RuntimeError(f"diagnostic basis dimension must be 3, got {source.shape}")
    return source[[codes.index(code) for code in (0, 1, 2)]]


def canonical_diag_features(
    loaded: Mapping[str, np.ndarray], *, max_n: int, pair_slot: Mapping[tuple[int, int], int], exact: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_codes = np.asarray(loaded["x__diagnostic_basis_codes"])
    prefix = "audit__exact_" if exact else "x__"
    local_raw = reorder_basis(np.asarray(loaded[prefix + "delta_local_expectations"], dtype=np.float64), basis_codes)
    pair_raw = reorder_basis(np.asarray(loaded[prefix + "delta_pairwise_correlations"], dtype=np.float64), basis_codes)
    parity = reorder_basis(np.asarray(loaded[prefix + "delta_global_parity"], dtype=np.float64).reshape(3, 1), basis_codes).reshape(3)
    n_qubits = int(np.asarray(loaded["x__layout_logical_to_physical"]).size)
    if local_raw.shape != (3, n_qubits) or n_qubits > max_n:
        raise RuntimeError(f"unexpected local diagnostic shape {local_raw.shape} for {n_qubits}q")
    local = np.zeros((3, max_n), dtype=np.float64)
    local[:, :n_qubits] = local_raw
    local_mask = np.zeros(max_n, dtype=np.float64)
    local_mask[:n_qubits] = 1.0
    pairs = np.asarray(loaded["x__pair_indices"], dtype=np.int64).reshape(-1, 2)
    if pair_raw.shape != (3, len(pairs)):
        raise RuntimeError(f"unexpected pair diagnostic shape {pair_raw.shape}")
    pair_values = np.zeros((3, len(pair_slot)), dtype=np.float64)
    pair_mask = np.zeros(len(pair_slot), dtype=np.float64)
    for source_index, raw_pair in enumerate(pairs):
        pair = (int(raw_pair[0]), int(raw_pair[1]))
        if pair not in pair_slot:
            raise RuntimeError(f"pair {pair} is outside fixed slots")
        target = pair_slot[pair]
        pair_values[:, target] = pair_raw[:, source_index]
        pair_mask[target] = 1.0
    return (
        np.concatenate([local.reshape(-1), local_mask]),
        np.concatenate([pair_values.reshape(-1), pair_mask]),
        parity,
    )


def graph_stats(loaded: Mapping[str, np.ndarray]) -> np.ndarray:
    n_qubits = int(np.asarray(loaded["x__layout_logical_to_physical"]).size)
    names = np.asarray(loaded["x__graph_gate_names"]).astype(str)
    qptr = np.asarray(loaded["x__graph_gate_qubit_ptr"], dtype=np.int64)
    qidx = np.asarray(loaded["x__graph_gate_qubit_indices"], dtype=np.int64)
    pptr = np.asarray(loaded["x__graph_gate_parameter_ptr"], dtype=np.int64)
    gate_count = int(len(names))
    if len(qptr) != gate_count + 1 or len(pptr) != gate_count + 1:
        raise RuntimeError("graph pointer length mismatch")
    arities = np.diff(qptr).astype(np.int64)
    parameter_counts = np.diff(pptr).astype(np.int64)
    if np.any(arities <= 0):
        raise RuntimeError("graph contains zero-arity gate")
    last_layer = np.zeros(max(1, n_qubits), dtype=np.int64)
    interaction_pairs: set[tuple[int, int]] = set()
    degree_sets = [set() for _ in range(n_qubits)]
    for gate_index in range(gate_count):
        qs = [int(v) for v in qidx[qptr[gate_index] : qptr[gate_index + 1]].tolist()]
        layer = 1 + max((int(last_layer[q]) for q in qs), default=0)
        for q in qs:
            last_layer[q] = layer
        for i, left in enumerate(qs):
            for right in qs[i + 1 :]:
                a, b = sorted((left, right))
                interaction_pairs.add((a, b))
                degree_sets[a].add(b)
                degree_sets[b].add(a)
    depth = int(np.max(last_layer)) if len(last_layer) else 0
    one_q = int(np.sum(arities == 1))
    multi_q = int(np.sum(arities >= 2))
    parameterized = int(np.sum(parameter_counts > 0))
    possible_pairs = max(1, n_qubits * (n_qubits - 1) // 2)
    degree_denominator = max(1, n_qubits - 1)
    degrees = np.asarray([len(values) for values in degree_sets], dtype=np.float64)
    return np.asarray([
        float(n_qubits), math.log1p(gate_count), math.log1p(depth),
        one_q / max(1, gate_count), multi_q / max(1, gate_count),
        parameterized / max(1, gate_count), float(np.mean(arities)) if gate_count else 0.0,
        len(interaction_pairs) / possible_pairs,
        float(np.mean(degrees)) / degree_denominator if n_qubits else 0.0,
        float(np.max(degrees)) / degree_denominator if n_qubits else 0.0,
    ], dtype=np.float64)


def load_feature_table(product: Path, rows: Sequence[Mapping[str, str]], progress_every: int) -> dict[str, Any]:
    max_n = 8
    pair_slot = fixed_pair_slots(max_n)
    context_values: list[np.ndarray] = []
    graph_values: list[np.ndarray] = []
    local_values: list[np.ndarray] = []
    pair_values: list[np.ndarray] = []
    parity_values: list[np.ndarray] = []
    exact_local_values: list[np.ndarray] = []
    exact_pair_values: list[np.ndarray] = []
    exact_parity_values: list[np.ndarray] = []
    family_values: list[str] = []
    split_values: list[str] = []
    group_values: list[str] = []
    root_values: list[int] = []
    occurrence_values: list[int] = []
    mechanism_values: list[int] = []
    effect_values: list[int] = []
    mechanism_mask_values: list[bool] = []
    manifest_context: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        artifact = product / str(row["artifact_path"])
        if sha256_file(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"artifact hash mismatch for {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as loaded_npz:
            loaded = {key: loaded_npz[key] for key in loaded_npz.files}
        forbidden = [key for key in loaded if key.startswith("x__") and any(token in key.lower() for token in ("mechanism_target", "effect_target", "phenomenology_target"))]
        if forbidden:
            raise RuntimeError(f"privileged-looking deployable fields: {forbidden}")
        layout = np.asarray(loaded["x__layout_logical_to_physical"], dtype=np.int64)
        n_qubits = int(layout.size)
        shots_array = np.asarray(loaded["x__observed_shots"], dtype=np.int64).reshape(-1)
        reference_shots = np.asarray(loaded["x__reference_shots"], dtype=np.int64).reshape(-1)
        if len(shots_array) != 3 or np.any(shots_array != shots_array[0]) or np.any(reference_shots != shots_array):
            raise RuntimeError("invalid paired shot-count contract")
        shots = int(shots_array[0])
        local, pair, parity = canonical_diag_features(loaded, max_n=max_n, pair_slot=pair_slot, exact=False)
        exact_local, exact_pair, exact_parity = canonical_diag_features(loaded, max_n=max_n, pair_slot=pair_slot, exact=True)
        effect = int(bool(np.asarray(loaded["y__effect_present_target"]).reshape(-1)[0]))
        mechanism = int(np.asarray(loaded["y__mechanism_target"]).reshape(-1)[0])
        mechanism_mask = bool(np.asarray(loaded["y__mechanism_loss_mask"]).reshape(-1)[0])
        if effect != int(as_bool(row["effect_present"])) or mechanism_mask != as_bool(row["mechanism_loss_mask"]):
            raise RuntimeError("target/manifest mismatch")
        expected_mechanism = {"clean_control": -1, "rz_drift": 0, "rx_overrotation": 1, "ry_overrotation": 2}[str(row["mechanism"])]
        if mechanism != expected_mechanism:
            raise RuntimeError("mechanism target/manifest mismatch")
        context_values.append(np.asarray([float(n_qubits), math.log2(shots)], dtype=np.float64))
        graph_values.append(graph_stats(loaded))
        local_values.append(local); pair_values.append(pair); parity_values.append(parity)
        exact_local_values.append(exact_local); exact_pair_values.append(exact_pair); exact_parity_values.append(exact_parity)
        family_values.append(str(row["family"])); split_values.append(str(row["split"])); group_values.append(str(row["clean_circuit_group_id"]))
        root_values.append(int(row["root_index"])); occurrence_values.append(int(row["family_occurrence_index"]))
        mechanism_values.append(mechanism); effect_values.append(effect); mechanism_mask_values.append(mechanism_mask)
        manifest_context.append({"family": str(row["family"]), "n_qubits": int(row["n_qubits"]), "shots": int(row["shots"]), "strength": float(row["strength"]), "insertion_depth_bin": str(row["insertion_depth_bin"])})
        if progress_every and index % progress_every == 0:
            print(f"Loaded and verified {index}/{len(rows)} Step 5 artifacts", flush=True)
    train_mask = np.asarray([value == "train" for value in split_values], dtype=bool)
    validation_mask = np.asarray([value == "validation" for value in split_values], dtype=bool)
    if np.any(train_mask & validation_mask) or not np.all(train_mask | validation_mask):
        raise RuntimeError("unexpected split membership")
    families = sorted({family_values[index] for index in np.flatnonzero(train_mask)})
    family_to_index = {name: idx for idx, name in enumerate(families)}
    if any(value not in family_to_index for value in family_values):
        raise RuntimeError("validation contains unseen family")
    family_onehot = np.zeros((len(rows), len(families)), dtype=np.float64)
    for index, family in enumerate(family_values):
        family_onehot[index, family_to_index[family]] = 1.0
    context = np.stack(context_values); graph = np.stack(graph_values)
    local = np.stack(local_values); pair = np.stack(pair_values); parity = np.stack(parity_values)
    exact_local = np.stack(exact_local_values); exact_pair = np.stack(exact_pair_values); exact_parity = np.stack(exact_parity_values)
    diag_local_pairwise = np.concatenate([local, pair], axis=1)
    diag_full = np.concatenate([local, pair, parity], axis=1)
    exact_diag_full = np.concatenate([exact_local, exact_pair, exact_parity], axis=1)
    features = {
        "context_only": context,
        "graph_stats_only": graph,
        "diag_local": local,
        "diag_local_pairwise": diag_local_pairwise,
        "diag_full": diag_full,
        "diag_full_context": np.concatenate([diag_full, context], axis=1),
        "diag_full_graph": np.concatenate([diag_full, graph], axis=1),
        "diag_full_context_graph": np.concatenate([diag_full, context, graph], axis=1),
        "exact_diag_full_oracle": exact_diag_full,
        "family_oracle": family_onehot,
    }
    for name, matrix in features.items():
        if not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"non-finite values in {name}")
    return {
        "features": features,
        "families": families,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "groups": np.asarray(group_values),
        "root_index": np.asarray(root_values, dtype=np.int64),
        "family_occurrence_index": np.asarray(occurrence_values, dtype=np.int64),
        "effect": np.asarray(effect_values, dtype=np.int8),
        "mechanism": np.asarray(mechanism_values, dtype=np.int8),
        "mechanism_mask": np.asarray(mechanism_mask_values, dtype=bool),
        "manifest_context": manifest_context,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return cm


def metrics_from_cm(cm: np.ndarray) -> dict[str, Any]:
    cmf = np.asarray(cm, dtype=np.float64)
    support = cmf.sum(axis=1); predicted = cmf.sum(axis=0); tp = np.diag(cmf)
    recalls = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2.0 * precision * recalls, precision + recalls, out=np.zeros_like(tp), where=(precision + recalls) > 0)
    return {"balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1)), "recalls": recalls.tolist(), "support": support.astype(np.int64).tolist()}


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8); s = np.asarray(scores, dtype=np.float64)
    positive = y == 1; negative = y == 0; n_pos = int(np.sum(positive)); n_neg = int(np.sum(negative))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort"); sorted_scores = s[order]; ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_ovr_auc(y_true: np.ndarray, scores: np.ndarray, n_classes: int) -> float:
    values = [binary_auc((y_true == i).astype(np.int8), scores[:, i]) for i in range(n_classes)]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def balanced_sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise RuntimeError(f"training fold missing class: {counts.tolist()}")
    raw = np.asarray([1.0 / counts[int(label)] for label in y], dtype=np.float64)
    return raw / float(np.mean(raw))


def build_ridge_system(X: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0); scale = np.std(X, axis=0); scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (X - mean) / scale
    design = np.concatenate([np.ones((len(X), 1)), standardized], axis=1)
    weights = balanced_sample_weights(y, n_classes); sqrt_w = np.sqrt(weights)[:, None]
    weighted_design = design * sqrt_w
    target = y.astype(np.float64).reshape(-1, 1) if n_classes == 2 else np.eye(n_classes, dtype=np.float64)[y.astype(np.int64)]
    return mean, scale, weighted_design.T @ weighted_design, weighted_design.T @ (target * sqrt_w)


def solve_ridge(gram: np.ndarray, rhs: np.ndarray, ridge_lambda: float) -> np.ndarray:
    system = gram.copy(); regularizer = np.eye(system.shape[0], dtype=np.float64) * float(ridge_lambda); regularizer[0, 0] = 0.0; system += regularizer
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def ridge_scores(X: np.ndarray, mean: np.ndarray, scale: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((len(X), 1)), (X - mean) / scale], axis=1)
    return (design @ beta).reshape(len(X), -1)


def select_binary_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, Any]]:
    y = np.asarray(y, dtype=np.int8); s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, kind="mergesort"); sorted_y = y[order]; sorted_s = s[order]; n = len(y)
    prefix_pos = np.concatenate([[0], np.cumsum(sorted_y == 1)]); prefix_neg = np.concatenate([[0], np.cumsum(sorted_y == 0)])
    total_pos = int(prefix_pos[-1]); total_neg = int(prefix_neg[-1])
    valid = np.ones(n + 1, dtype=bool)
    if n > 1:
        valid[1:n] = sorted_s[:-1] < sorted_s[1:]
    cuts = np.flatnonzero(valid); fn = prefix_pos[cuts].astype(float); tn = prefix_neg[cuts].astype(float); tp = total_pos - fn; fp = total_neg - tn
    recall_pos = tp / total_pos; recall_neg = tn / total_neg; balanced = 0.5 * (recall_pos + recall_neg)
    precision_pos = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0); precision_neg = np.divide(tn, tn + fn, out=np.zeros_like(tn), where=(tn + fn) > 0)
    f1_pos = np.divide(2 * precision_pos * recall_pos, precision_pos + recall_pos, out=np.zeros_like(tp), where=(precision_pos + recall_pos) > 0)
    f1_neg = np.divide(2 * precision_neg * recall_neg, precision_neg + recall_neg, out=np.zeros_like(tn), where=(precision_neg + recall_neg) > 0)
    macro_f1 = 0.5 * (f1_pos + f1_neg); minimum_recall = np.minimum(recall_pos, recall_neg)
    thresholds = np.empty(len(cuts), dtype=float)
    for idx, cut in enumerate(cuts):
        if cut == 0: thresholds[idx] = float(sorted_s[0] - max(1e-12, abs(float(sorted_s[0])) * 1e-12))
        elif cut == n: thresholds[idx] = float(sorted_s[-1] + max(1e-12, abs(float(sorted_s[-1])) * 1e-12))
        else: thresholds[idx] = float(0.5 * (sorted_s[cut - 1] + sorted_s[cut]))
    best = max(range(len(cuts)), key=lambda i: (float(minimum_recall[i]), float(balanced[i]), float(macro_f1[i]), -abs(float(thresholds[i]) - 0.5)))
    threshold = float(thresholds[best]); pred = (s >= threshold).astype(np.int8); metrics = metrics_from_cm(confusion_matrix(y, pred, 2)); metrics["minimum_class_recall"] = float(minimum_recall[best])
    return threshold, metrics


def training_oof_folds(occurrence: np.ndarray, train_mask: np.ndarray) -> list[np.ndarray]:
    residues = sorted(set((occurrence[train_mask] % 5).tolist()))
    if residues != [1, 2, 3, 4]:
        raise RuntimeError(f"unexpected Step 5 training fold residues {residues}")
    return [train_mask & ((occurrence % 5) == residue) for residue in residues]


def tune_and_fit_ridge(X: np.ndarray, y: np.ndarray, population_mask: np.ndarray, train_mask: np.ndarray, validation_mask: np.ndarray, occurrence: np.ndarray, n_classes: int, lambdas: Sequence[float]) -> dict[str, Any]:
    eligible_train = population_mask & train_mask
    train_indices = np.flatnonzero(eligible_train); position = {int(index): pos for pos, index in enumerate(train_indices)}
    oof_by_lambda = {float(lam): np.full((len(train_indices), 1 if n_classes == 2 else n_classes), np.nan) for lam in lambdas}
    for heldout_root_mask in training_oof_folds(occurrence, train_mask):
        heldout = eligible_train & heldout_root_mask; fit = eligible_train & ~heldout_root_mask; heldout_indices = np.flatnonzero(heldout)
        mean, scale, gram, rhs = build_ridge_system(X[fit], y[fit], n_classes)
        for lam in lambdas:
            scores = ridge_scores(X[heldout], mean, scale, solve_ridge(gram, rhs, float(lam)))
            for row_index, source_index in enumerate(heldout_indices):
                oof_by_lambda[float(lam)][position[int(source_index)]] = scores[row_index]
    y_oof = y[train_indices]; best_lambda = None; best_threshold = None; best_key = None; oof_summary: dict[str, Any] = {}
    for lam in lambdas:
        scores = oof_by_lambda[float(lam)]
        if np.any(~np.isfinite(scores)):
            raise RuntimeError(f"incomplete OOF predictions for lambda {lam}")
        if n_classes == 2:
            threshold, metrics = select_binary_threshold(y_oof, scores[:, 0]); key = (float(metrics["balanced_accuracy"]), float(metrics["macro_f1"]), -float(lam))
            oof_summary[str(lam)] = {"lambda": float(lam), "threshold": threshold, **metrics, "roc_auc": binary_auc(y_oof, scores[:, 0])}
        else:
            pred = np.argmax(scores, axis=1).astype(np.int8); metrics = metrics_from_cm(confusion_matrix(y_oof, pred, n_classes)); threshold = None; key = (float(metrics["balanced_accuracy"]), float(metrics["macro_f1"]), -float(lam))
            oof_summary[str(lam)] = {"lambda": float(lam), **metrics, "macro_ovr_roc_auc": macro_ovr_auc(y_oof, scores, n_classes)}
        if best_key is None or key > best_key:
            best_key = key; best_lambda = float(lam); best_threshold = None if threshold is None else float(threshold)
    assert best_lambda is not None
    mean, scale, gram, rhs = build_ridge_system(X[eligible_train], y[eligible_train], n_classes); beta = solve_ridge(gram, rhs, best_lambda)
    validation_scores = ridge_scores(X[validation_mask], mean, scale, beta)
    validation_pred_all = (validation_scores[:, 0] >= float(best_threshold)).astype(np.int8) if n_classes == 2 else np.argmax(validation_scores, axis=1).astype(np.int8)
    return {"best_lambda": best_lambda, "best_threshold": best_threshold, "oof_summary": oof_summary, "validation_scores_all": validation_scores, "validation_pred_all": validation_pred_all, "n_train": int(np.sum(eligible_train))}


def root_confusions(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted(set(groups.tolist()))); index = {value: i for i, value in enumerate(unique.tolist())}; cms = np.zeros((len(unique), n_classes, n_classes), dtype=np.int64)
    for truth, pred, group in zip(y_true, y_pred, groups):
        cms[index[group], int(truth), int(pred)] += 1
    return unique, cms


def bootstrap_counts(n_roots: int, replicates: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).multinomial(n_roots, np.full(n_roots, 1.0 / n_roots), size=replicates)


def metric_arrays_from_bootstrap(root_cms: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    boot = counts @ root_cms.reshape(root_cms.shape[0], -1); n_classes = root_cms.shape[1]; ba = np.empty(len(boot)); f1 = np.empty(len(boot))
    for index, row in enumerate(boot):
        metrics = metrics_from_cm(row.reshape(n_classes, n_classes)); ba[index] = metrics["balanced_accuracy"]; f1[index] = metrics["macro_f1"]
    return ba, f1


def ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def evaluation_row(*, task: str, baseline: str, privileged: bool, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None, groups: np.ndarray, class_names: Sequence[str], n_train: int, selected_lambda: float | None, selected_threshold: float | None, bootstrap_replicates: int, bootstrap_seed: int, confidence: float) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_classes = len(class_names); cm = confusion_matrix(y_true, y_pred, n_classes); metrics = metrics_from_cm(cm); unique_groups, root_cms = root_confusions(y_true, y_pred, groups, n_classes); counts = bootstrap_counts(len(unique_groups), bootstrap_replicates, bootstrap_seed); ba_samples, f1_samples = metric_arrays_from_bootstrap(root_cms, counts); ba_ci = ci(ba_samples, confidence); f1_ci = ci(f1_samples, confidence)
    row: dict[str, Any] = {"task": task, "baseline": baseline, "privileged_analysis_only": privileged, "n_train": n_train, "n_validation": len(y_true), "validation_clean_root_count": len(unique_groups), "selected_lambda": "" if selected_lambda is None else selected_lambda, "selected_threshold": "" if selected_threshold is None else selected_threshold, "balanced_accuracy": metrics["balanced_accuracy"], "balanced_accuracy_ci_low": ba_ci[0], "balanced_accuracy_ci_high": ba_ci[1], "macro_f1": metrics["macro_f1"], "macro_f1_ci_low": f1_ci[0], "macro_f1_ci_high": f1_ci[1]}
    for index, name in enumerate(class_names):
        row[f"recall__{name}"] = metrics["recalls"][index]; row[f"support__{name}"] = metrics["support"][index]
    if scores is not None:
        row["roc_auc" if n_classes == 2 else "macro_ovr_roc_auc"] = binary_auc(y_true, scores.reshape(-1)) if n_classes == 2 else macro_ovr_auc(y_true, scores, n_classes)
    return row, {"root_ids": unique_groups, "root_cms": root_cms, "bootstrap_counts": counts, "bootstrap_ba": ba_samples, "bootstrap_f1": f1_samples}


def random_predictions(y_train: np.ndarray, n_validation: int, n_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_train.astype(np.int64), minlength=n_classes).astype(float); probabilities = counts / counts.sum(); rng = np.random.default_rng(seed); return rng.choice(np.arange(n_classes), size=n_validation, p=probabilities).astype(np.int8), np.tile(probabilities[None, :], (n_validation, 1))


def majority_predictions(y_train: np.ndarray, n_validation: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_train.astype(np.int64), minlength=n_classes).astype(float); probabilities = counts / counts.sum(); return np.full(n_validation, int(np.argmax(counts)), dtype=np.int8), np.tile(probabilities[None, :], (n_validation, 1))


def paired_difference_rows(task: str, comparisons: Sequence[Sequence[str]], result_bootstrap: Mapping[str, Mapping[str, np.ndarray]], confidence: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        if left not in result_bootstrap or right not in result_bootstrap:
            continue
        left_data = result_bootstrap[left]; right_data = result_bootstrap[right]
        if not np.array_equal(left_data["root_ids"], right_data["root_ids"]) or not np.array_equal(left_data["bootstrap_counts"], right_data["bootstrap_counts"]):
            raise RuntimeError(f"paired bootstrap mismatch for {task}: {left} vs {right}")
        for metric_key, label in (("bootstrap_ba", "balanced_accuracy"), ("bootstrap_f1", "macro_f1")):
            difference = left_data[metric_key] - right_data[metric_key]; low, high = ci(difference, confidence)
            rows.append({"task": task, "left": left, "right": right, "metric": label, "mean_bootstrap_difference": float(np.mean(difference)), "ci_low": low, "ci_high": high})
    return rows


def stratified_rows(*, task: str, baseline: str, y_true: np.ndarray, y_pred: np.ndarray, source_indices: np.ndarray, context: Sequence[Mapping[str, Any]], class_names: Sequence[str], strata: Sequence[str], minimum: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []; n_classes = len(class_names)
    for stratum in strata:
        values = [context[int(index)][stratum] for index in source_indices]
        for value in sorted(set(values), key=lambda item: str(item)):
            mask = np.asarray([item == value for item in values], dtype=bool)
            if int(np.sum(mask)) < minimum:
                continue
            metrics = metrics_from_cm(confusion_matrix(y_true[mask], y_pred[mask], n_classes)); row: dict[str, Any] = {"task": task, "baseline": baseline, "stratum": stratum, "value": value, "n": int(np.sum(mask)), "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"]}
            for class_index, class_name in enumerate(class_names):
                row[f"recall__{class_name}"] = metrics["recalls"][class_index]; row[f"support__{class_name}"] = metrics["support"][class_index]
            output.append(row)
    return output


def main() -> None:
    args = parse_args(); config_path = args.config.expanduser().resolve(); config = read_json(config_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_BASELINE_OUTCOME":
        raise RuntimeError("unexpected Step 6 config schema/status")
    product = args.product_dir.expanduser().resolve() if args.product_dir else Path(config["source_dataset"]["default_product_dir"]).expanduser().resolve()
    source_complete, manifest_rows = verify_source_product(product, config); table = load_feature_table(product, manifest_rows, args.progress_every)
    train_mask = table["train_mask"]; validation_mask = table["validation_mask"]; effect = table["effect"]; mechanism = table["mechanism"]; mechanism_mask = table["mechanism_mask"]; occurrence = table["family_occurrence_index"]; groups = table["groups"]; features = table["features"]
    validation_indices = np.flatnonzero(validation_mask); validation_groups_all = groups[validation_mask]; effect_validation_truth = effect[validation_mask]; mechanism_validation_relative = mechanism_mask[validation_mask]; mechanism_validation_truth = mechanism[validation_mask][mechanism_validation_relative]; mechanism_validation_groups = validation_groups_all[mechanism_validation_relative]
    lambdas = [float(value) for value in config["linear_model"]["ridge_lambdas"]]; bootstrap_replicates = int(config["evaluation"]["root_group_bootstrap_replicates"]); bootstrap_seed = int(config["evaluation"]["bootstrap_seed"]); confidence = float(config["evaluation"]["confidence_level"]); comparisons = config["evaluation"]["paired_difference_comparisons"]
    metric_rows: list[dict[str, Any]] = []; paired_rows: list[dict[str, Any]] = []; stratified_output: list[dict[str, Any]] = []; selection_records: dict[str, Any] = {}; feature_dimensions = {name: int(matrix.shape[1]) for name, matrix in features.items()}
    effect_predictions: dict[str, np.ndarray] = {}; effect_scores: dict[str, np.ndarray] = {}; mechanism_predictions_all: dict[str, np.ndarray] = {}; mechanism_scores_all: dict[str, np.ndarray] = {}; boot_by_task: dict[str, dict[str, Mapping[str, np.ndarray]]] = {"effect_detection": {}, "mechanism_diagnosis": {}, "integrated_diagnosis": {}}
    train_effect = effect[train_mask]; train_mechanism = mechanism[train_mask & mechanism_mask]
    for offset, baseline in enumerate(SANITY_BASELINES):
        if baseline == "majority_prior":
            epred, escore = majority_predictions(train_effect, len(validation_indices), 2); mpred, mscore = majority_predictions(train_mechanism, len(validation_indices), 3)
        else:
            epred, escore = random_predictions(train_effect, len(validation_indices), 2, bootstrap_seed + 100 + offset); mpred, mscore = random_predictions(train_mechanism, len(validation_indices), 3, bootstrap_seed + 200 + offset)
        effect_predictions[baseline] = epred; effect_scores[baseline] = escore[:, 1]; mechanism_predictions_all[baseline] = mpred; mechanism_scores_all[baseline] = mscore
    all_variants = list(DEPLOYABLE_VARIANTS) + list(PRIVILEGED_VARIANTS)
    for variant in all_variants:
        matrix = features[variant]; print(f"Fitting Step 6 baseline: {variant}", flush=True)
        effect_fit = tune_and_fit_ridge(matrix, effect, np.ones(len(effect), dtype=bool), train_mask, validation_mask, occurrence, 2, lambdas)
        mechanism_fit = tune_and_fit_ridge(matrix, mechanism, mechanism_mask, train_mask, validation_mask, occurrence, 3, lambdas)
        effect_predictions[variant] = effect_fit["validation_pred_all"]; effect_scores[variant] = effect_fit["validation_scores_all"][:, 0]; mechanism_predictions_all[variant] = mechanism_fit["validation_pred_all"]; mechanism_scores_all[variant] = mechanism_fit["validation_scores_all"]
        selection_records[variant] = {"privileged_analysis_only": variant in PRIVILEGED_VARIANTS, "effect_detection": {"selected_lambda": effect_fit["best_lambda"], "selected_threshold": effect_fit["best_threshold"], "oof_by_lambda": effect_fit["oof_summary"]}, "mechanism_diagnosis": {"selected_lambda": mechanism_fit["best_lambda"], "oof_by_lambda": mechanism_fit["oof_summary"]}}
    for baseline in list(SANITY_BASELINES) + all_variants:
        privileged = baseline in PRIVILEGED_VARIANTS; effect_lambda = None if baseline in SANITY_BASELINES else float(selection_records[baseline]["effect_detection"]["selected_lambda"]); effect_threshold = None if baseline in SANITY_BASELINES else float(selection_records[baseline]["effect_detection"]["selected_threshold"]); mechanism_lambda = None if baseline in SANITY_BASELINES else float(selection_records[baseline]["mechanism_diagnosis"]["selected_lambda"])
        effect_row, effect_boot = evaluation_row(task="effect_detection", baseline=baseline, privileged=privileged, y_true=effect_validation_truth, y_pred=effect_predictions[baseline], scores=effect_scores[baseline], groups=validation_groups_all, class_names=("no_effect", "effect"), n_train=int(np.sum(train_mask)), selected_lambda=effect_lambda, selected_threshold=effect_threshold, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence); metric_rows.append(effect_row); boot_by_task["effect_detection"][baseline] = effect_boot
        mech_pred = mechanism_predictions_all[baseline][mechanism_validation_relative]; mech_scores = mechanism_scores_all[baseline][mechanism_validation_relative]
        mechanism_row, mechanism_boot = evaluation_row(task="mechanism_diagnosis", baseline=baseline, privileged=privileged, y_true=mechanism_validation_truth, y_pred=mech_pred, scores=mech_scores, groups=mechanism_validation_groups, class_names=MECHANISM_NAMES, n_train=int(np.sum(train_mask & mechanism_mask)), selected_lambda=mechanism_lambda, selected_threshold=None, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence); metric_rows.append(mechanism_row); boot_by_task["mechanism_diagnosis"][baseline] = mechanism_boot
        integrated_truth = np.where(effect_validation_truth == 0, 0, mechanism[validation_mask].astype(np.int64) + 1).astype(np.int8); integrated_pred = np.where(effect_predictions[baseline] == 0, 0, mechanism_predictions_all[baseline].astype(np.int64) + 1).astype(np.int8)
        integrated_row, integrated_boot = evaluation_row(task="integrated_diagnosis", baseline=baseline, privileged=privileged, y_true=integrated_truth, y_pred=integrated_pred, scores=None, groups=validation_groups_all, class_names=INTEGRATED_NAMES, n_train=int(np.sum(train_mask)), selected_lambda=None, selected_threshold=effect_threshold, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed, confidence=confidence); metric_rows.append(integrated_row); boot_by_task["integrated_diagnosis"][baseline] = integrated_boot
    for task, task_boot in boot_by_task.items():
        paired_rows.extend(paired_difference_rows(task, comparisons, task_boot, confidence))
    strata = list(config["evaluation"]["strata"]); minimum = int(config["evaluation"]["minimum_stratum_examples"])
    for baseline in config["evaluation"]["stratify_primary_deployable_variants"]:
        stratified_output.extend(stratified_rows(task="effect_detection", baseline=baseline, y_true=effect_validation_truth, y_pred=effect_predictions[baseline], source_indices=validation_indices, context=table["manifest_context"], class_names=("no_effect", "effect"), strata=strata, minimum=minimum))
        mechanism_source_indices = validation_indices[mechanism_validation_relative]
        stratified_output.extend(stratified_rows(task="mechanism_diagnosis", baseline=baseline, y_true=mechanism_validation_truth, y_pred=mechanism_predictions_all[baseline][mechanism_validation_relative], source_indices=mechanism_source_indices, context=table["manifest_context"], class_names=MECHANISM_NAMES, strata=strata, minimum=minimum))
    metric_lookup = {(row["task"], row["baseline"]): row for row in metric_rows}; paired_lookup = {(row["task"], row["left"], row["right"], row["metric"]): row for row in paired_rows}
    evidence_flags = {
        "finite_shot_diag_full_effect_above_chance_ci": float(metric_lookup[("effect_detection", "diag_full")]["balanced_accuracy_ci_low"]) > 0.5,
        "finite_shot_diag_full_mechanism_above_chance_ci": float(metric_lookup[("mechanism_diagnosis", "diag_full")]["balanced_accuracy_ci_low"]) > 1.0 / 3.0,
        "pairwise_effect_gain_ci_positive": float(paired_lookup[("effect_detection", "diag_local_pairwise", "diag_local", "balanced_accuracy")]["ci_low"]) > 0.0,
        "pairwise_mechanism_gain_ci_positive": float(paired_lookup[("mechanism_diagnosis", "diag_local_pairwise", "diag_local", "balanced_accuracy")]["ci_low"]) > 0.0,
        "parity_effect_gain_ci_positive": float(paired_lookup[("effect_detection", "diag_full", "diag_local_pairwise", "balanced_accuracy")]["ci_low"]) > 0.0,
        "parity_mechanism_gain_ci_positive": float(paired_lookup[("mechanism_diagnosis", "diag_full", "diag_local_pairwise", "balanced_accuracy")]["ci_low"]) > 0.0,
        "simple_graph_gain_mechanism_ci_positive": float(paired_lookup[("mechanism_diagnosis", "diag_full_graph", "diag_full", "balanced_accuracy")]["ci_low"]) > 0.0,
        "exact_oracle_gap_mechanism_ci_positive": float(paired_lookup[("mechanism_diagnosis", "exact_diag_full_oracle", "diag_full", "balanced_accuracy")]["ci_low"]) > 0.0,
    }
    identity = {"schema": SCHEMA, "config_sha256": sha256_file(config_path), "runner_sha256": sha256_file(Path(__file__).resolve()), "source_product_id": source_complete["product_id"]}; benchmark_id = "benchmark_" + hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve(); output_parent.mkdir(parents=True, exist_ok=True); output = output_parent / benchmark_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing benchmark {output}")
    staging = output_parent / f".{benchmark_id}.staging-{uuid.uuid4().hex}"; staging.mkdir()
    write_csv(staging / "baseline_metrics.csv", metric_rows); write_csv(staging / "paired_differences.csv", paired_rows); write_csv(staging / "stratified_metrics.csv", stratified_output); atomic_json(staging / "model_selection.json", selection_records); atomic_json(staging / "feature_dimensions.json", feature_dimensions)
    atomic_json(staging / "decision.json", {"schema": SCHEMA, "decision": "BASELINE_BENCHMARK_COMPLETE", "evidence_flags": evidence_flags, "historical_v0_1_test_accessed": False, "spent_confirmatory_cohort_accessed": False, "hardware_executed": False, "triqto_architecture_changed": False, "validation_used_for_model_or_threshold_selection": False})
    prediction_payload: dict[str, np.ndarray] = {"validation_root_index": table["root_index"][validation_mask], "effect_truth": effect_validation_truth, "mechanism_truth_all": mechanism[validation_mask], "mechanism_loss_mask": mechanism_validation_relative}
    for baseline in list(SANITY_BASELINES) + all_variants:
        prediction_payload[f"effect__{baseline}__pred"] = effect_predictions[baseline]; prediction_payload[f"effect__{baseline}__score"] = effect_scores[baseline]; prediction_payload[f"mechanism__{baseline}__pred_all"] = mechanism_predictions_all[baseline]; prediction_payload[f"mechanism__{baseline}__scores_all"] = mechanism_scores_all[baseline]
    np.savez_compressed(staging / "validation_predictions.npz", **prediction_payload)
    output_files = ["baseline_metrics.csv", "paired_differences.csv", "stratified_metrics.csv", "model_selection.json", "feature_dimensions.json", "decision.json", "validation_predictions.npz"]
    completion = {"schema": SCHEMA, "status": "COMPLETE", "benchmark_id": benchmark_id, "identity": identity, "source_product_id": source_complete["product_id"], "source_clean_roots": source_complete["clean_circuit_root_count"], "source_examples": source_complete["example_count"], "deployable_variants": list(DEPLOYABLE_VARIANTS), "privileged_analysis_variants": list(PRIVILEGED_VARIANTS), "sanity_baselines": list(SANITY_BASELINES), "tasks": ["effect_detection", "mechanism_diagnosis", "integrated_diagnosis"], "historical_v0_1_test_accessed": False, "spent_confirmatory_cohort_accessed": False, "hardware_executed": False, "classifier_type": "class_balanced_ridge_least_squares", "triqto_architecture_changed": False, "file_hashes": {name: sha256_file(staging / name) for name in output_files}}
    atomic_json(staging / "benchmark_complete.json", completion); os.replace(staging, output)
    print("\nTRIQTO STEP 6 CHEAP BASELINE BENCHMARK COMPLETE\n"); print("Decision: BASELINE_BENCHMARK_COMPLETE"); print(f"Source product: {source_complete['product_id']}"); print(f"Validation examples: {len(validation_indices)}"); print(f"diag_full effect BA: {metric_lookup[('effect_detection', 'diag_full')]['balanced_accuracy']:.4f}"); print(f"diag_full mechanism BA: {metric_lookup[('mechanism_diagnosis', 'diag_full')]['balanced_accuracy']:.4f}"); print(f"diag_full_context_graph mechanism BA: {metric_lookup[('mechanism_diagnosis', 'diag_full_context_graph')]['balanced_accuracy']:.4f}"); print(f"exact oracle mechanism BA: {metric_lookup[('mechanism_diagnosis', 'exact_diag_full_oracle')]['balanced_accuracy']:.4f}"); print("Historical v0.1 test accessed: NO"); print("Spent confirmatory cohort accessed: NO"); print("TriQTO architecture changed: NO"); print(f"Results: {output}")


if __name__ == "__main__":
    main()
