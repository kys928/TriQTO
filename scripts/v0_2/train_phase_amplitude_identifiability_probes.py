#!/usr/bin/env python3
"""Frozen TriQTO v0.2 phase/amplitude identifiability probe evaluation.

Evidence regimes:
  A_full        current distorted graph + backend + stored Z Born
  A_born_only   stored Z Born only
  A_clean_graph audited clean graph + backend + stored Z Born (primary A)
  B             A_clean_graph + X/Y probabilities and expectations
  C             B + diagnostic-only privileged Hilbert features

Models:
  linear        L2 logistic regression
  nonlinear     one-hidden-layer MLP

The script uses train-only feature-schema fitting, grouped stratified CV,
identical preprocessing/seeds/tuning policy, group-safe OOF Platt calibration,
retuned label-shuffle controls, group bootstrap CIs, paired bootstrap deltas,
and stratified validation reports. It never reads the historical v0.1 test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

try:
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise RuntimeError("scikit-learn is required in the pod environment") from exc


DEFAULT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_pilot_v2/data/"
    "v0_2_phase_amplitude_identifiability_pilot"
)
SCHEMA = "triqto.v0_2.phase_amplitude_probe_evaluation.v1"
REGIMES = ("A_full", "A_born_only", "A_clean_graph", "B", "C")
MODELS = ("linear", "nonlinear")
MODEL_SEEDS = (17, 29, 43)
CV_SEED = 20260803
BOOT_SEED = 424242
LINEAR_GRID = ({"C": .01}, {"C": .1}, {"C": 1.0}, {"C": 10.0})
NONLINEAR_GRID = tuple(
    {"hidden_units": h, "alpha": a}
    for h in (16, 32)
    for a in (1e-3, 1e-2, 1e-1)
)
STRONG = {
    "balanced_accuracy": .85,
    "macro_f1": .85,
    "phase_recall": .85,
    "amplitude_recall": .85,
    "auroc": .90,
}


@dataclass(frozen=True)
class Example:
    entity_id: str
    artifact: Path
    group: str
    split: str
    y: int
    coarse_label: str
    raw_label: str
    family: str
    strength: str
    affected_qubit: str
    n_qubits: int
    phase_sensitive: bool


@dataclass(frozen=True)
class FeatureSchema:
    gate_vocab: tuple[str, ...]
    max_state_dim: int
    max_qubits: int
    gate_width: int
    node_width: int
    edge_width: int
    backend_width: int
    hilbert_names: tuple[str, ...]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def canon(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def text_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def scalar(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError("expected scalar")
    item = value.reshape(-1)[0]
    return item.item() if isinstance(item, np.generic) else item


def product_dir(config: argparse.Namespace) -> Path:
    if config.product_dir:
        result = config.product_dir.expanduser().resolve()
    else:
        pointer = config.product_parent.expanduser().resolve() / "current_product.json"
        result = Path(str(read_json(pointer)["product_dir"])).resolve()
    if not (result / "generation_complete.json").is_file():
        raise RuntimeError(f"pilot generation is incomplete: {result}")
    return result


def load_examples(root: Path) -> list[Example]:
    rows = pq.read_table(root / "manifests" / "item_manifest.parquet").to_pylist()
    output: list[Example] = []
    for row in rows:
        split = str(row["split"])
        if split not in {"train", "validation"}:
            raise RuntimeError(f"forbidden split: {split}")
        coarse = str(row["coarse_label"])
        output.append(
            Example(
                entity_id=str(row["entity_id"]),
                artifact=(root / str(row["artifact_ref"])).resolve(),
                group=str(row["split_group_id"]),
                split=split,
                y=0 if coarse == "phase_like" else 1,
                coarse_label=coarse,
                raw_label=str(row["raw_label"]),
                family=str(row["family"]),
                strength=str(row["strength_key"]),
                affected_qubit=str(row["affected_qubit_signature"]),
                n_qubits=int(row["n_qubits"]),
                phase_sensitive=bool(row["phase_sensitive_family"]),
            )
        )
    if len(output) != 280:
        raise RuntimeError(f"expected 280 examples, found {len(output)}")
    return output


def arrays(example: Example) -> dict[str, np.ndarray]:
    with np.load(example.artifact, allow_pickle=False) as archive:
        value = {name: archive[name] for name in archive.files}
    if str(scalar(value["entity_id"])) != example.entity_id:
        raise ValueError(f"entity mismatch: {example.artifact}")
    return value


def fit_schema(train: Sequence[Example]) -> FeatureSchema:
    gates: set[str] = set()
    dimensions = {"gate": set(), "node": set(), "edge": set(), "backend": set()}
    max_state = max_qubits = 0
    hilbert: set[str] = set()
    for item in train:
        value = arrays(item)
        gates.update(np.asarray(value["a__x_graph_gate_names"]).astype(str).tolist())
        dimensions["gate"].add(value["a__x_graph_gate_features"].shape[1])
        dimensions["node"].add(value["a__x_graph_node_features"].shape[1])
        dimensions["edge"].add(value["a__x_graph_edge_features"].shape[1])
        dimensions["backend"].add(np.asarray(value["a__x_backend_features"]).size)
        max_state = max(max_state, np.asarray(value["c__clean_statevector_real"]).size)
        max_qubits = max(max_qubits, item.n_qubits)
        hilbert.update(
            np.asarray(value["c__hilbert_summary_names"]).astype(str).tolist()
        )
    if any(len(values) != 1 for values in dimensions.values()):
        raise ValueError(f"inconsistent feature widths: {dimensions}")
    return FeatureSchema(
        gate_vocab=tuple(sorted(gates)) + ("<UNK>",),
        max_state_dim=max_state,
        max_qubits=max_qubits,
        gate_width=next(iter(dimensions["gate"])),
        node_width=next(iter(dimensions["node"])),
        edge_width=next(iter(dimensions["edge"])),
        backend_width=next(iter(dimensions["backend"])),
        hilbert_names=tuple(sorted(hilbert)),
    )


def pad(value: np.ndarray, length: int) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.size > length:
        raise ValueError(f"vector {flat.size} exceeds train schema {length}")
    output = np.zeros(length, dtype=np.float64)
    output[:flat.size] = flat
    return output


def dense_probability(
    bitstrings: np.ndarray, probabilities: np.ndarray, dimension: int
) -> np.ndarray:
    output = np.zeros(dimension, dtype=np.float64)
    seen: set[int] = set()
    for bits, probability in zip(
        np.asarray(bitstrings).astype(str).reshape(-1),
        np.asarray(probabilities, dtype=np.float64).reshape(-1),
        strict=True,
    ):
        index = int(bits, 2)
        if index >= dimension or index in seen:
            raise ValueError(f"invalid or duplicate Born outcome: {bits}")
        seen.add(index)
        output[index] = float(probability)
    return output


def aggregate_matrix(value: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ValueError(f"bad matrix shape {matrix.shape}; expected width {width}")
    if matrix.shape[0] == 0:
        return np.zeros(width * 4, dtype=np.float64)
    return np.concatenate(
        (
            np.mean(matrix, axis=0),
            np.std(matrix, axis=0),
            np.min(matrix, axis=0),
            np.max(matrix, axis=0),
        )
    )


def aggregate_1d(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return np.zeros(5, dtype=np.float64)
    return np.asarray(
        [flat.size, np.mean(flat), np.std(flat), np.min(flat), np.max(flat)],
        dtype=np.float64,
    )


def retained_parameters(
    pointer: np.ndarray, values: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    ptr = np.asarray(pointer, dtype=np.int64)
    flat = np.asarray(values, dtype=np.float64)
    chunks = [
        flat[ptr[index] : ptr[index + 1]]
        for index, retained in enumerate(keep)
        if retained
    ]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)


def graph_vector(
    value: Mapping[str, np.ndarray], schema: FeatureSchema, *, clean: bool
) -> np.ndarray:
    names = np.asarray(value["a__x_graph_gate_names"]).astype(str).reshape(-1)
    keep = np.ones(names.size, dtype=bool)
    if clean:
        removed = np.asarray(
            value["audit__removed_distortion_gate_indices"], dtype=np.int64
        ).reshape(-1)
        if np.any((removed < 0) | (removed >= names.size)):
            raise ValueError("bad audited distortion indices")
        keep[removed] = False

    retained_names = names[keep]
    lookup = {name: index for index, name in enumerate(schema.gate_vocab)}
    counts = np.zeros(len(schema.gate_vocab), dtype=np.float64)
    for name in retained_names:
        counts[lookup.get(name, lookup["<UNK>"])] += 1.0
    proportions = counts / max(1.0, float(retained_names.size))

    gate_matrix = np.asarray(
        value["a__x_graph_gate_features"], dtype=np.float64
    )[keep]
    node_matrix = value["a__x_graph_node_features"]
    edge_matrix = value["a__x_graph_edge_features"]

    qubit_ptr = np.asarray(value["a__x_graph_gate_qubit_ptr"], dtype=np.int64)
    arity = np.diff(qubit_ptr)[keep]
    parameter_ptr = value["a__x_graph_gate_parameter_ptr"]
    parameter_sin = retained_parameters(
        parameter_ptr, value["a__x_graph_gate_parameter_sin"], keep
    )
    parameter_cos = retained_parameters(
        parameter_ptr, value["a__x_graph_gate_parameter_cos"], keep
    )

    structural = np.asarray(
        [
            np.asarray(value["a__x_graph_node_index"]).size,
            retained_names.size,
            np.asarray(value["a__x_graph_edge_index"]).shape[1],
            float(np.mean(arity)) if arity.size else 0.0,
            np.sum(arity == 2),
            np.sum(retained_names == "measure"),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        (
            counts,
            proportions,
            aggregate_matrix(gate_matrix, schema.gate_width),
            aggregate_matrix(node_matrix, schema.node_width),
            aggregate_matrix(edge_matrix, schema.edge_width),
            aggregate_1d(parameter_sin),
            aggregate_1d(parameter_cos),
            structural,
        )
    )


def backend_vector(value: Mapping[str, np.ndarray], schema: FeatureSchema) -> np.ndarray:
    features = np.asarray(
        value["a__x_backend_features"], dtype=np.float64
    ).reshape(-1)
    if features.size != schema.backend_width:
        raise ValueError("backend width changed")
    return np.concatenate(
        (
            features,
            np.asarray(
                value["a__x_backend_feature_available_mask"], dtype=np.float64
            ).reshape(-1),
            np.asarray(
                value["a__x_backend_available_mask"], dtype=np.float64
            ).reshape(-1),
        )
    )


def z_vector(value: Mapping[str, np.ndarray], schema: FeatureSchema) -> np.ndarray:
    bits = value["a__x_born_input_outcome_bitstrings"]
    return np.concatenate(
        (
            dense_probability(
                bits, value["a__x_born_input_probabilities"], schema.max_state_dim
            ),
            dense_probability(
                bits,
                value["a__x_born_input_sqrt_probabilities"],
                schema.max_state_dim,
            ),
        )
    )


def b_vector(value: Mapping[str, np.ndarray], schema: FeatureSchema) -> np.ndarray:
    return np.concatenate(
        (
            pad(value["b__distorted_x_probabilities"], schema.max_state_dim),
            pad(value["b__distorted_y_probabilities"], schema.max_state_dim),
            pad(value["b__distorted_x_expectations"], schema.max_qubits),
            pad(value["b__distorted_y_expectations"], schema.max_qubits),
        )
    )


def c_vector(value: Mapping[str, np.ndarray], schema: FeatureSchema) -> np.ndarray:
    state = [
        pad(value[name], schema.max_state_dim)
        for name in (
            "c__clean_statevector_real",
            "c__clean_statevector_imag",
            "c__distorted_statevector_real",
            "c__distorted_statevector_imag",
            "c__aligned_state_delta_real",
            "c__aligned_state_delta_imag",
        )
    ]
    names = np.asarray(value["c__hilbert_summary_names"]).astype(str).tolist()
    values = np.asarray(
        value["c__hilbert_summary_values"], dtype=np.float64
    ).tolist()
    summary_map = dict(zip(names, values, strict=True))
    summary = np.asarray(
        [summary_map.get(name, 0.0) for name in schema.hilbert_names],
        dtype=np.float64,
    )
    expectations = [
        pad(value[name], schema.max_qubits)
        for name in (
            "c__clean_x_expectations",
            "c__clean_y_expectations",
            "c__clean_z_expectations",
            "c__distorted_z_expectations",
        )
    ]
    return np.concatenate((*state, summary, *expectations))


def vector(item: Example, schema: FeatureSchema, regime: str) -> np.ndarray:
    value = arrays(item)
    born = z_vector(value, schema)
    if regime == "A_born_only":
        return born
    graph = graph_vector(value, schema, clean=(regime != "A_full"))
    base = np.concatenate((graph, backend_vector(value, schema), born))
    if regime in {"A_full", "A_clean_graph"}:
        return base
    with_xy = np.concatenate((base, b_vector(value, schema)))
    if regime == "B":
        return with_xy
    if regime == "C":
        return np.concatenate((with_xy, c_vector(value, schema)))
    raise ValueError(regime)


def matrix(
    examples: Sequence[Example], schema: FeatureSchema, regime: str
) -> np.ndarray:
    rows = [vector(item, schema, regime) for item in examples]
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError(f"variable widths for {regime}: {widths}")
    output = np.vstack(rows).astype(np.float64)
    if not np.isfinite(output).all():
        raise ValueError(f"non-finite values in {regime}")
    return output


def estimator(model: str, parameters: Mapping[str, Any], seed: int) -> Pipeline:
    if model == "linear":
        classifier = LogisticRegression(
            C=float(parameters["C"]),
            solver="lbfgs",
            penalty="l2",
            max_iter=5000,
            random_state=seed,
        )
    elif model == "nonlinear":
        classifier = MLPClassifier(
            hidden_layer_sizes=(int(parameters["hidden_units"]),),
            activation="tanh",
            solver="lbfgs",
            alpha=float(parameters["alpha"]),
            max_iter=1500,
            random_state=seed,
        )
    else:
        raise ValueError(model)
    return Pipeline(
        [
            ("variance", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.995, svd_solver="full")),
            ("model", classifier),
        ]
    )


def scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=np.float64)
    probability = np.clip(
        np.asarray(model.predict_proba(features), dtype=np.float64)[:, 1],
        1e-8,
        1 - 1e-8,
    )
    return np.log(probability / (1 - probability))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def cv_splits(y: np.ndarray, groups: np.ndarray, folds: int):
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=CV_SEED
    )
    output = list(splitter.split(np.zeros_like(y), y, groups))
    for train_index, valid_index in output:
        if set(groups[train_index]) & set(groups[valid_index]):
            raise RuntimeError("group leakage in cross-validation")
        if len(np.unique(y[valid_index])) != 2:
            raise RuntimeError("cross-validation fold lacks one class")
    return output


def metrics(y: np.ndarray, probability: np.ndarray, ece_bins: int = 10):
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    phase_recall = tn / (tn + fp)
    amplitude_recall = tp / (tp + fn)

    edges = np.linspace(0.0, 1.0, ece_bins + 1)
    ece = 0.0
    for index in range(ece_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (probability >= lower) & (probability <= upper)
            if index == ece_bins - 1
            else (probability >= lower) & (probability < upper)
        )
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(probability[mask])) - float(np.mean(y[mask]))
            )

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "phase_recall": float(phase_recall),
        "amplitude_recall": float(amplitude_recall),
        "phase_to_amplitude_confusion": float(1 - phase_recall),
        "amplitude_to_phase_confusion": float(1 - amplitude_recall),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": float(ece),
    }


def tune(
    model_name: str,
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits,
    progress_prefix: str = "",
):
    grid = LINEAR_GRID if model_name == "linear" else NONLINEAR_GRID
    rows = []
    for candidate_index, parameters in enumerate(grid, start=1):
        fold_rows = []
        for train_index, valid_index in splits:
            probabilities = []
            for seed in MODEL_SEEDS:
                model = estimator(model_name, parameters, seed)
                model.fit(features[train_index], y[train_index])
                probabilities.append(sigmoid(scores(model, features[valid_index])))
            fold_rows.append(
                metrics(y[valid_index], np.mean(probabilities, axis=0))
            )
        row = {
            "parameters": dict(parameters),
            "mean_balanced_accuracy": float(
                np.mean([item["balanced_accuracy"] for item in fold_rows])
            ),
            "mean_macro_f1": float(
                np.mean([item["macro_f1"] for item in fold_rows])
            ),
            "mean_worst_recall": float(
                np.mean(
                    [
                        min(item["phase_recall"], item["amplitude_recall"])
                        for item in fold_rows
                    ]
                )
            ),
        }
        rows.append(row)
        if progress_prefix:
            print(
                f"{progress_prefix} candidate {candidate_index}/{len(grid)} "
                f"BA={row['mean_balanced_accuracy']:.4f}",
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            -row["mean_balanced_accuracy"],
            -row["mean_macro_f1"],
            -row["mean_worst_recall"],
            canon(row["parameters"]),
        )
    )
    return dict(rows[0]["parameters"]), rows


def calibrated_ensemble(
    model_name: str,
    parameters: Mapping[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    splits,
):
    oof_by_seed, valid_by_seed = [], []
    for seed in MODEL_SEEDS:
        oof = np.full(train_y.size, np.nan)
        for train_index, valid_index in splits:
            model = estimator(model_name, parameters, seed)
            model.fit(train_x[train_index], train_y[train_index])
            oof[valid_index] = scores(model, train_x[valid_index])
        if not np.isfinite(oof).all():
            raise RuntimeError("incomplete OOF scores")
        oof_by_seed.append(oof)

        model = estimator(model_name, parameters, seed)
        model.fit(train_x, train_y)
        valid_by_seed.append(scores(model, valid_x))

    mean_oof = np.mean(oof_by_seed, axis=0)
    mean_valid = np.mean(valid_by_seed, axis=0)
    calibrator = LogisticRegression(
        C=1e6, solver="lbfgs", max_iter=5000, random_state=CV_SEED
    )
    calibrator.fit(mean_oof.reshape(-1, 1), train_y)
    return (
        calibrator.predict_proba(mean_oof.reshape(-1, 1))[:, 1],
        calibrator.predict_proba(mean_valid.reshape(-1, 1))[:, 1],
        {
            "method": "group-safe OOF Platt scaling",
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        },
    )


def bootstrap_ci(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    ece_bins: int,
    seed: int,
):
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    attempts = 0
    while len(samples["balanced_accuracy"]) < repeats:
        attempts += 1
        if attempts > repeats * 20:
            raise RuntimeError("could not obtain valid bootstrap replicates")
        chosen = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([indices[group] for group in chosen])
        if len(np.unique(y[index])) != 2:
            continue
        row = metrics(y[index], probability[index], ece_bins)
        for name, value in row.items():
            samples[name].append(value)
    return {
        name: {
            "low": float(np.quantile(values, .025)),
            "high": float(np.quantile(values, .975)),
        }
        for name, values in samples.items()
    }


def shuffled(y: np.ndarray, seed: int) -> np.ndarray:
    output = np.asarray(y).copy()
    np.random.default_rng(seed).shuffle(output)
    return np.roll(output, 1) if np.array_equal(output, y) else output


def label_shuffle_controls(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    repeats: int,
    folds: int,
    prefix: str,
):
    rows = []
    for repeat in range(repeats):
        permuted = shuffled(train_y, 91000 + repeat)
        splits = cv_splits(permuted, train_groups, folds)
        parameters, _ = tune(
            model_name, train_x, permuted, train_groups, splits
        )
        probabilities = []
        for seed in MODEL_SEEDS:
            model = estimator(model_name, parameters, seed)
            model.fit(train_x, permuted)
            probabilities.append(sigmoid(scores(model, valid_x)))
        row = {
            "repeat": repeat,
            "parameters": parameters,
            **metrics(valid_y, np.mean(probabilities, axis=0)),
        }
        rows.append(row)
        if prefix:
            print(
                f"{prefix} shuffle {repeat + 1}/{repeats} "
                f"BA={row['balanced_accuracy']:.4f}",
                flush=True,
            )
    return rows


def stratified(
    examples: Sequence[Example], y: np.ndarray, probability: np.ndarray
):
    getters = {
        "family": lambda item: item.family,
        "strength": lambda item: item.strength,
        "n_qubits": lambda item: item.n_qubits,
        "phase_sensitive_family": lambda item: item.phase_sensitive,
        "affected_qubit": lambda item: item.affected_qubit,
        "raw_label": lambda item: item.raw_label,
    }
    rows = []
    for field, getter in getters.items():
        for value in sorted({getter(item) for item in examples}, key=str):
            index = np.asarray(
                [
                    position
                    for position, item in enumerate(examples)
                    if getter(item) == value
                ],
                dtype=np.int64,
            )
            prediction = (probability[index] >= .5).astype(np.int64)
            row = {
                "stratum": field,
                "value": str(value),
                "support": int(index.size),
                "phase_support": int(np.sum(y[index] == 0)),
                "amplitude_support": int(np.sum(y[index] == 1)),
                "accuracy": float(np.mean(prediction == y[index])),
                "mean_amplitude_probability": float(np.mean(probability[index])),
            }
            if np.any(y[index] == 0):
                row["phase_recall"] = float(
                    np.mean(prediction[y[index] == 0] == 0)
                )
            if np.any(y[index] == 1):
                row["amplitude_recall"] = float(
                    np.mean(prediction[y[index] == 1] == 1)
                )
            if len(np.unique(y[index])) == 2:
                row.update(metrics(y[index], probability[index]))
            rows.append(row)
    return rows


def paired_delta(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
):
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    names = (
        "balanced_accuracy",
        "macro_f1",
        "phase_recall",
        "amplitude_recall",
        "auroc",
    )
    samples: dict[str, list[float]] = defaultdict(list)
    while len(samples["balanced_accuracy"]) < repeats:
        chosen = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([indices[group] for group in chosen])
        if len(np.unique(y[index])) != 2:
            continue
        a, b = metrics(y[index], left[index]), metrics(y[index], right[index])
        for name in names:
            samples[name].append(b[name] - a[name])
    point_left, point_right = metrics(y, left), metrics(y, right)
    return {
        name: {
            "point": float(point_right[name] - point_left[name]),
            "low": float(np.quantile(values, .025)),
            "high": float(np.quantile(values, .975)),
        }
        for name, values in samples.items()
    }


def model_status(row: Mapping[str, float], ci: Mapping[str, Any]) -> str:
    if (
        all(row[name] >= threshold for name, threshold in STRONG.items())
        and ci["balanced_accuracy"]["low"] > .5
    ):
        return "strong"
    if (
        row["balanced_accuracy"] <= .65
        or min(row["phase_recall"], row["amplitude_recall"]) <= .60
        or row["auroc"] <= .70
    ):
        return "weak"
    return "intermediate"


def decision(statuses: Mapping[str, str]):
    key = (
        statuses["A_clean_graph"],
        statuses["B"],
        statuses["C"],
    )
    table = {
        ("weak", "strong", "strong"): (
            "MEASUREMENT_BASIS_BOTTLENECK",
            "A is weak while B and C are strong.",
        ),
        ("weak", "weak", "strong"): (
            "OBSERVABLE_SUMMARY_INSUFFICIENT",
            "A and B are weak while privileged C is strong.",
        ),
        ("strong", "strong", "strong"): (
            "REPRESENTATION_OR_MULTITASK_BOTTLENECK",
            "A, B, and C are all strong.",
        ),
        ("weak", "weak", "weak"): (
            "LABEL_OR_GENERATOR_AMBIGUITY",
            "A, B, and C are all weak.",
        ),
    }
    if key in table:
        code, interpretation = table[key]
        return {"status": code, "interpretation": interpretation, "applied": True}
    return {
        "status": "INCONCLUSIVE_OR_MIXED",
        "interpretation": (
            "At least one regime was intermediate or the model families "
            "disagreed; the four-way table is not forced."
        ),
        "applied": False,
    }


def main() -> None:
    config = args()
    root = product_dir(config)
    complete = read_json(root / "generation_complete.json")
    if complete.get("test_split_accessed") is not False:
        raise RuntimeError("pilot did not certify historical-test isolation")

    all_examples = load_examples(root)
    train = [item for item in all_examples if item.split == "train"]
    valid = [item for item in all_examples if item.split == "validation"]
    train_y = np.asarray([item.y for item in train], dtype=np.int64)
    valid_y = np.asarray([item.y for item in valid], dtype=np.int64)
    train_groups = np.asarray([item.group for item in train], dtype=object)
    valid_groups = np.asarray([item.group for item in valid], dtype=object)
    if set(train_groups) & set(valid_groups):
        raise RuntimeError("train/validation group overlap")

    schema = fit_schema(train)
    protocol = {
        "schema": SCHEMA,
        "product_id": str(complete["product_id"]),
        "product_manifest_sha256": str(complete["manifest_sha256"]),
        "script_sha256": file_hash(Path(__file__).resolve()),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "regimes": list(REGIMES),
        "models": list(MODELS),
        "model_seeds": list(MODEL_SEEDS),
        "cv_seed": CV_SEED,
        "cv_folds": config.cv_folds,
        "shuffle_repeats": config.shuffle_repeats,
        "bootstrap_repeats": config.bootstrap_repeats,
        "ece_bins": config.ece_bins,
        "linear_grid": list(LINEAR_GRID),
        "nonlinear_grid": list(NONLINEAR_GRID),
        "strong_thresholds": STRONG,
        "historical_v0_1_test_accessed": False,
    }
    protocol_id = text_hash(canon(protocol))
    output = (
        root
        / "reports"
        / "identifiability_probes"
        / f"probeeval_{protocol_id.removeprefix('sha256:')[:20]}"
    )
    output.mkdir(parents=True, exist_ok=True)
    protocol["protocol_id"] = protocol_id
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise RuntimeError("protocol directory contains different content")
    atomic_json(protocol_path, protocol)

    features = {}
    for regime in REGIMES:
        if config.progress:
            print(f"Extracting {regime}", flush=True)
        features[regime] = (
            matrix(train, schema, regime),
            matrix(valid, schema, regime),
        )

    splits = cv_splits(train_y, train_groups, config.cv_folds)
    results: dict[str, Any] = {}
    validation_probabilities: dict[tuple[str, str], np.ndarray] = {}
    metric_rows, prediction_rows, stratified_rows = [], [], []

    for regime_index, regime in enumerate(REGIMES):
        results[regime] = {}
        train_x, valid_x = features[regime]
        for model_index, model_name in enumerate(MODELS):
            prefix = f"[{regime}/{model_name}]" if config.progress else ""
            parameters, tuning_rows = tune(
                model_name,
                train_x,
                train_y,
                train_groups,
                splits,
                progress_prefix=prefix,
            )
            train_probability, valid_probability, calibration = (
                calibrated_ensemble(
                    model_name,
                    parameters,
                    train_x,
                    train_y,
                    valid_x,
                    splits,
                )
            )
            validation_probabilities[(regime, model_name)] = valid_probability
            row = metrics(valid_y, valid_probability, config.ece_bins)
            ci = bootstrap_ci(
                valid_y,
                valid_probability,
                valid_groups,
                config.bootstrap_repeats,
                config.ece_bins,
                BOOT_SEED + regime_index * 100 + model_index,
            )
            controls = label_shuffle_controls(
                model_name,
                train_x,
                train_y,
                train_groups,
                valid_x,
                valid_y,
                config.shuffle_repeats,
                config.cv_folds,
                prefix,
            )
            shuffle_ba = [item["balanced_accuracy"] for item in controls]
            status = model_status(row, ci)
            results[regime][model_name] = {
                "best_parameters": parameters,
                "tuning": tuning_rows,
                "calibration": calibration,
                "validation_metrics": row,
                "bootstrap_95_ci": ci,
                "status": status,
                "label_shuffle_controls": controls,
                "label_shuffle_summary": {
                    "mean_balanced_accuracy": float(np.mean(shuffle_ba)),
                    "max_balanced_accuracy": float(np.max(shuffle_ba)),
                    "empirical_p_value": float(
                        (1 + sum(value >= row["balanced_accuracy"] for value in shuffle_ba))
                        / (1 + len(shuffle_ba))
                    ),
                },
            }

            metric_rows.append(
                {
                    "regime": regime,
                    "model": model_name,
                    "status": status,
                    **row,
                    **{
                        f"{name}_ci_low": interval["low"]
                        for name, interval in ci.items()
                    },
                    **{
                        f"{name}_ci_high": interval["high"]
                        for name, interval in ci.items()
                    },
                }
            )
            for index, item in enumerate(valid):
                prediction_rows.append(
                    {
                        "entity_id": item.entity_id,
                        "split_group_id": item.group,
                        "coarse_label": item.coarse_label,
                        "raw_label": item.raw_label,
                        "family": item.family,
                        "strength": item.strength,
                        "affected_qubit": item.affected_qubit,
                        "n_qubits": item.n_qubits,
                        "phase_sensitive_family": item.phase_sensitive,
                        "regime": regime,
                        "model": model_name,
                        "true_binary_label": int(valid_y[index]),
                        "amplitude_probability": float(valid_probability[index]),
                        "predicted_binary_label": int(valid_probability[index] >= .5),
                    }
                )
            for stratum in stratified(valid, valid_y, valid_probability):
                stratified_rows.append(
                    {"regime": regime, "model": model_name, **stratum}
                )

    regime_statuses = {}
    for regime in REGIMES:
        values = {results[regime][model]["status"] for model in MODELS}
        if values == {"strong"}:
            regime_statuses[regime] = "strong"
        elif values == {"weak"}:
            regime_statuses[regime] = "weak"
        else:
            regime_statuses[regime] = "mixed_or_intermediate"

    paired = {}
    for model_index, model_name in enumerate(MODELS):
        paired[model_name] = {
            "B_minus_A_clean_graph": paired_delta(
                valid_y,
                validation_probabilities[("A_clean_graph", model_name)],
                validation_probabilities[("B", model_name)],
                valid_groups,
                config.bootstrap_repeats,
                BOOT_SEED + 1000 + model_index,
            ),
            "C_minus_B": paired_delta(
                valid_y,
                validation_probabilities[("B", model_name)],
                validation_probabilities[("C", model_name)],
                valid_groups,
                config.bootstrap_repeats,
                BOOT_SEED + 2000 + model_index,
            ),
        }

    final_decision = decision(regime_statuses)
    final_decision.update(
        {
            "regime_statuses": regime_statuses,
            "primary_A_regime": "A_clean_graph",
            "shortcut_controls": {
                "A_full": "Contains the visible synthetic distortion gate.",
                "A_born_only": "Tests stored Z evidence without graph inputs.",
            },
        }
    )
    report = {
        "schema": SCHEMA,
        "protocol_id": protocol_id,
        "product_id": complete["product_id"],
        "historical_v0_1_test_accessed": False,
        "train_count": len(train),
        "validation_count": len(valid),
        "feature_schema": {
            "gate_vocab": list(schema.gate_vocab),
            "max_state_dim": schema.max_state_dim,
            "max_qubits": schema.max_qubits,
            "gate_width": schema.gate_width,
            "node_width": schema.node_width,
            "edge_width": schema.edge_width,
            "backend_width": schema.backend_width,
            "hilbert_names": list(schema.hilbert_names),
        },
        "results": results,
        "paired_bootstrap_deltas": paired,
        "decision": final_decision,
    }
    atomic_json(output / "results.json", report)
    atomic_json(output / "decision.json", final_decision)
    write_csv(output / "metrics.csv", metric_rows)
    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(output / "stratified_metrics.csv", stratified_rows)
    atomic_json(
        output / "complete.json",
        {
            "schema": SCHEMA,
            "protocol_id": protocol_id,
            "product_id": complete["product_id"],
            "status": "COMPLETE",
            "results_sha256": file_hash(output / "results.json"),
            "decision_sha256": file_hash(output / "decision.json"),
            "historical_v0_1_test_accessed": False,
        },
    )

    print()
    print("=" * 78)
    print("TRIQTO V0.2 PHASE/AMPLITUDE PROBE EVALUATION COMPLETE")
    print("=" * 78)
    for regime in REGIMES:
        print(f"{regime}: {regime_statuses[regime]}")
        for model_name in MODELS:
            row = results[regime][model_name]["validation_metrics"]
            print(
                f"  {model_name}: BA={row['balanced_accuracy']:.4f}, "
                f"macro-F1={row['macro_f1']:.4f}, "
                f"phase recall={row['phase_recall']:.4f}, "
                f"amplitude recall={row['amplitude_recall']:.4f}, "
                f"AUROC={row['auroc']:.4f}, ECE={row['ece']:.4f}"
            )
    print()
    print(f"Decision: {final_decision['status']}")
    print(final_decision["interpretation"])
    print(f"Output: {output}")
    print("The historical v0.1 test split was not accessed.")


if __name__ == "__main__":
    main()
