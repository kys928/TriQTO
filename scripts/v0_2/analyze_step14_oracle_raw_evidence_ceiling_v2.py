#!/usr/bin/env python3
"""Step-14 oracle-localization / raw-evidence ceiling decomposition (v2).

Scientific boundary
-------------------
This is a post-hoc diagnostic probe study. It does NOT update a TriQTO model
checkpoint, does NOT access Step-14 simulator-outer or future-hardware-reserve
examples, and does NOT execute a QPU. The only learned objects are disposable
probe classifiers trained on the already-frozen Step-14 FIT partition.

The affected-qubit location, true injection boundary, and insertion-context
class are privileged generator metadata. They are used ONLY by explicitly
oracle-labelled analysis stages and are never written back into the dataset,
checkpoint, or deployable model input.

Nested comparison
-----------------
1. raw_diagnostics
   Exact finite-shot model-visible diagnostic tensors, canonicalized but with
   no TriQTO learned encoder.
2. raw_plus_circuit_context
   (1) + the exact deterministic graph features/structure that the frozen
   Step-7 graph adapter constructs before its learned graph encoder.
3. raw_plus_affected_qubit_oracle
   (2) + analysis-only affected-qubit identity and deterministic extraction of
   the already-visible local/incident raw diagnostics at that location.
4. raw_plus_affected_qubit_local_context_oracle
   (3) + analysis-only true insertion boundary/context and the actual nearby
   gate identities, qubit roles, and raw gate parameter sin/cos values.
5. raw_context_nonlinear_ceiling
   Intentionally high-capacity nonlinear probe on stage (2), an approximate
   nonprivileged raw-input ceiling.
6. oracle_local_nonlinear_ceiling
   Same high-capacity probe on stage (4), useful to distinguish a raw-evidence
   ceiling from a missing-local-frame problem.

All feature standardization is fit-only. Selection is evaluation-only. Paired
bootstrap deltas resample Step-14 circuit families, not individual examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import benchmark_step6_cheap_baselines as baseline
import run_step14_cross_motif_training as step14
import analyze_step14_representation_fusion_head as rep14
from triqto.step7 import graph_adapter as step7_graph

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_oracle_raw_evidence_ceiling")
SCHEMA = "triqto.v0_2.step14_oracle_raw_evidence_ceiling.v2"

MAX_QUBITS = 5
MAX_GATES = 16
MAX_DIRECTED_EDGES = MAX_GATES * 2
GATE_VOCAB = ("h", "rx", "ry", "rz", "cx", "cz")
CONTEXT_VOCAB = (
    "pre_entangling",
    "inter_entangling",
    "post_entangling_recombination",
)
MECHANISM_TO_TARGET = {"rz_drift": 0, "rx_overrotation": 1, "ry_overrotation": 2}
PRIMARY_STAGES = (
    "raw_diagnostics",
    "raw_plus_circuit_context",
    "raw_plus_affected_qubit_oracle",
    "raw_plus_affected_qubit_local_context_oracle",
)
CEILING_STAGES = (
    "raw_context_nonlinear_ceiling",
    "oracle_local_nonlinear_ceiling",
)
PROBE_SEEDS = (6101, 6102, 6103)
PROBE_SPEC = {
    "primary_probe": "one_hidden_layer_MLP_classifier",
    "primary_hidden_dim": 64,
    "primary_epochs": 25,
    "primary_dropout": 0.1,
    "primary_learning_rate": 0.003,
    "ceiling_probe": "three_hidden_layer_high_capacity_MLP_classifier",
    "ceiling_hidden_dims": [512, 256, 128],
    "ceiling_epochs": 40,
    "ceiling_dropout": 0.1,
    "ceiling_learning_rate": 0.001,
    "optimizer": "AdamW",
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "batch_size": 1024,
    "probe_seeds": list(PROBE_SEEDS),
    "standardization": "fit_mean_std_only_per_stage",
    "selection_used_for_training_or_early_stopping": False,
    "main_model_weights_updated": False,
    "bootstrap_replicates": 1000,
    "bootstrap_unit": "cross_motif_family_id",
    "bootstrap_seed": 2026090102,
    # Frozen before result observation. These gates are deliberately coarse;
    # the continuous BAs/deltas/CIs remain the primary scientific result.
    "meaningful_oracle_delta_minimum": 0.05,
    "nonprivileged_low_ceiling_ba": 0.60,
    "nonprivileged_high_ceiling_ba": 0.70,
    "nonlinear_capacity_gain_minimum": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--selection-freeze-sha256", required=True)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def stable_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for oracle/raw-evidence diagnostic but unavailable")
    return torch.device(name)


def onehot(index: int, width: int) -> np.ndarray:
    if index < 0 or index >= width:
        raise RuntimeError(f"one-hot index {index} outside width {width}")
    out = np.zeros(width, dtype=np.float64)
    out[index] = 1.0
    return out


def parse_operation_signature(raw: str) -> list[tuple[str, tuple[int, ...]]]:
    values = json.loads(raw)
    if not isinstance(values, list) or len(values) > MAX_GATES:
        raise RuntimeError("invalid or oversized Step-14 operation signature")
    parsed: list[tuple[str, tuple[int, ...]]] = []
    for item in values:
        text = str(item)
        if ":" not in text:
            raise RuntimeError(f"malformed operation signature entry {text!r}")
        name, qtext = text.split(":", 1)
        name = name.lower()
        if name not in GATE_VOCAB:
            raise RuntimeError(f"unexpected Step-14 gate {name!r}")
        qubits = tuple(int(piece[1:]) for piece in qtext.split("-") if piece)
        if not qubits or any(q < 0 or q >= MAX_QUBITS for q in qubits):
            raise RuntimeError(f"invalid qubits in operation signature {text!r}")
        parsed.append((name, qubits))
    return parsed


def gate_parameter_rows(loaded: Mapping[str, np.ndarray], gate_count: int) -> np.ndarray:
    """Per-gate raw sin/cos summaries from the model-visible serialized graph."""
    ptr = np.asarray(loaded["x__graph_gate_parameter_ptr"], dtype=np.int64).reshape(-1)
    sin = np.asarray(loaded["x__graph_gate_parameter_sin"], dtype=np.float64).reshape(-1)
    cos = np.asarray(loaded["x__graph_gate_parameter_cos"], dtype=np.float64).reshape(-1)
    if ptr.shape != (gate_count + 1,) or ptr[0] != 0 or ptr[-1] != len(sin) or sin.shape != cos.shape:
        raise RuntimeError("serialized graph parameter contract mismatch")
    rows = np.zeros((gate_count, 5), dtype=np.float64)
    for gate in range(gate_count):
        start, end = int(ptr[gate]), int(ptr[gate + 1])
        if end <= start:
            continue
        s = sin[start:end]
        c = cos[start:end]
        rows[gate] = [
            float(end - start),
            float(np.sum(s)),
            float(np.sum(c)),
            float(np.mean(s)),
            float(np.mean(c)),
        ]
    return rows


def raw_diagnostic_features(loaded: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """All finite-shot diagnostic evidence before any TriQTO learned encoder."""
    pair_slots = baseline.fixed_pair_slots(MAX_QUBITS)
    local, pair, parity = baseline.canonical_diag_features(
        loaded, max_n=MAX_QUBITS, pair_slot=pair_slots, exact=False
    )
    basis_codes = np.asarray(loaded["x__diagnostic_basis_codes"])
    order = [int(v) for v in np.asarray(basis_codes).reshape(-1).tolist()]
    if sorted(order) != [0, 1, 2]:
        raise RuntimeError("unexpected diagnostic basis code contract")
    reorder = [order.index(code) for code in (0, 1, 2)]

    observed = np.asarray(loaded["x__observed_shots"], dtype=np.float64).reshape(-1)[reorder]
    reference = np.asarray(loaded["x__reference_shots"], dtype=np.float64).reshape(-1)[reorder]
    ref_mask = np.asarray(loaded["x__reference_available_mask"], dtype=np.float64).reshape(-1)[reorder]
    ref_kind = np.asarray(loaded["x__reference_kind_code"], dtype=np.float64).reshape(-1)
    if observed.shape != (3,) or reference.shape != (3,) or np.any(observed <= 0) or np.any(reference <= 0):
        raise RuntimeError("unexpected Step-14 diagnostic shot-count contract")
    if ref_mask.shape != (3,) or ref_kind.shape != (1,):
        raise RuntimeError("unexpected Step-14 reference metadata contract")

    metadata = np.concatenate(
        (np.log2(observed), np.log2(reference), ref_mask, ref_kind)
    )
    return np.concatenate((local, pair, parity, metadata)), local


def ordinary_circuit_context(
    root: Mapping[str, str], loaded: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Exact deterministic pre-encoder graph context plus explicit structure.

    ``_graph_features_for_example`` is the same adapter used to construct the
    frozen model's GraphTensorBatch. It is deterministic and contains no learned
    weights. We flatten/pad those pre-encoder tensors so the probe gets ordinary
    model-visible circuit context without invoking the TriQTO graph encoder.
    """
    events = parse_operation_signature(str(root["operation_signature"]))
    graph = step7_graph._graph_features_for_example(loaded)
    node = np.asarray(graph["node_features"], dtype=np.float64)
    gate = np.asarray(graph["gate_features"], dtype=np.float64)
    edge = np.asarray(graph["edge_features"], dtype=np.float64)
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    qptr = np.asarray(graph["gate_qubit_ptr"], dtype=np.int64)
    qidx = np.asarray(graph["gate_qubit_indices"], dtype=np.int64)

    n_qubits = node.shape[0]
    n_gates = gate.shape[0]
    if n_qubits > MAX_QUBITS or n_gates > MAX_GATES or len(events) != n_gates:
        raise RuntimeError("Step-14 graph exceeds frozen oracle diagnostic support")
    if edge.shape[0] > MAX_DIRECTED_EDGES or edge_index.shape != (2, edge.shape[0]):
        raise RuntimeError("Step-14 directed edge support exceeds frozen diagnostic bound")

    node_pad = np.zeros((MAX_QUBITS, node.shape[1]), dtype=np.float64)
    node_mask = np.zeros(MAX_QUBITS, dtype=np.float64)
    node_pad[:n_qubits] = node
    node_mask[:n_qubits] = 1.0

    gate_pad = np.zeros((MAX_GATES, gate.shape[1]), dtype=np.float64)
    gate_mask = np.zeros(MAX_GATES, dtype=np.float64)
    gate_pad[:n_gates] = gate
    gate_mask[:n_gates] = 1.0

    edge_pad = np.zeros((MAX_DIRECTED_EDGES, edge.shape[1]), dtype=np.float64)
    edge_mask = np.zeros(MAX_DIRECTED_EDGES, dtype=np.float64)
    edge_endpoint = np.zeros((MAX_DIRECTED_EDGES, 2 * MAX_QUBITS), dtype=np.float64)
    edge_pad[: edge.shape[0]] = edge
    edge_mask[: edge.shape[0]] = 1.0
    for i in range(edge.shape[0]):
        source, destination = int(edge_index[0, i]), int(edge_index[1, i])
        edge_endpoint[i, :MAX_QUBITS] = onehot(source, MAX_QUBITS)
        edge_endpoint[i, MAX_QUBITS:] = onehot(destination, MAX_QUBITS)

    # Explicit gate-to-qubit incidence preserves the raw graph structure in a
    # fixed vector instead of forcing the probe to infer it from categorical IDs.
    incidence = np.zeros((MAX_GATES, 2 * MAX_QUBITS + 1), dtype=np.float64)
    if qptr.shape != (n_gates + 1,):
        raise RuntimeError("gate-qubit pointer contract mismatch")
    for gate_index in range(n_gates):
        qubits = qidx[int(qptr[gate_index]) : int(qptr[gate_index + 1])].tolist()
        if qubits:
            incidence[gate_index, :MAX_QUBITS] = onehot(int(qubits[0]), MAX_QUBITS)
        if len(qubits) > 1:
            incidence[gate_index, MAX_QUBITS : 2 * MAX_QUBITS] = onehot(int(qubits[1]), MAX_QUBITS)
        incidence[gate_index, -1] = float(len(qubits))

    return np.concatenate(
        (
            node_pad.reshape(-1), node_mask,
            gate_pad.reshape(-1), gate_mask,
            edge_pad.reshape(-1), edge_mask,
            edge_endpoint.reshape(-1),
            incidence.reshape(-1),
        )
    )


def localized_oracle_features(
    loaded: Mapping[str, np.ndarray], affected: int
) -> tuple[np.ndarray, np.ndarray]:
    """Analysis-only localization applied to already-model-visible diagnostics."""
    basis_codes = np.asarray(loaded["x__diagnostic_basis_codes"])
    local_raw = baseline.reorder_basis(
        np.asarray(loaded["x__delta_local_expectations"], dtype=np.float64), basis_codes
    )
    pair_raw = baseline.reorder_basis(
        np.asarray(loaded["x__delta_pairwise_correlations"], dtype=np.float64), basis_codes
    )
    pairs = np.asarray(loaded["x__pair_indices"], dtype=np.int64).reshape(-1, 2)
    n_qubits = int(np.asarray(loaded["x__layout_logical_to_physical"]).size)
    if affected < 0 or affected >= n_qubits or local_raw.shape != (3, n_qubits):
        raise RuntimeError("affected-qubit oracle is outside the circuit")
    if pair_raw.shape != (3, len(pairs)):
        raise RuntimeError("pairwise diagnostic contract mismatch")

    incident_indices = [i for i, pair in enumerate(pairs.tolist()) if affected in pair]
    incident_values = (
        pair_raw[:, incident_indices]
        if incident_indices
        else np.zeros((3, 0), dtype=np.float64)
    )
    incident_mean = (
        np.mean(incident_values, axis=1)
        if incident_values.shape[1]
        else np.zeros(3, dtype=np.float64)
    )
    incident_maxabs = (
        np.max(np.abs(incident_values), axis=1)
        if incident_values.shape[1]
        else np.zeros(3, dtype=np.float64)
    )
    affected_local = local_raw[:, affected].astype(np.float64)
    core = np.concatenate((affected_local, incident_mean, incident_maxabs))
    return np.concatenate((onehot(affected, MAX_QUBITS), core)), core


def local_insertion_context(
    root: Mapping[str, str], loaded: Mapping[str, np.ndarray], localized_core: np.ndarray
) -> np.ndarray:
    """Analysis-only actual gate neighborhood around the true insertion boundary."""
    affected = int(root["affected_qubit"])
    boundary = int(root["injection_boundary_rank"])
    context = str(root["injection_context_class"])
    events = parse_operation_signature(str(root["operation_signature"]))
    if boundary < 0 or boundary > len(events):
        raise RuntimeError("invalid oracle injection boundary")
    if context not in CONTEXT_VOCAB:
        raise RuntimeError("unexpected Step-14 injection-context class")

    parameter_rows = gate_parameter_rows(loaded, len(events))
    qptr = np.asarray(loaded["x__graph_gate_qubit_ptr"], dtype=np.int64).reshape(-1)
    qidx = np.asarray(loaded["x__graph_gate_qubit_indices"], dtype=np.int64).reshape(-1)
    if qptr.shape != (len(events) + 1,):
        raise RuntimeError("gate-qubit pointer mismatch in local oracle context")

    # Two operations before and two at/after the injection boundary. Each row
    # includes the actual raw model-visible gate parameter sin/cos summaries.
    window_rows: list[np.ndarray] = []
    window_width = len(GATE_VOCAB) + 3 * MAX_QUBITS + 4 + 5
    for offset in (-2, -1, 0, 1):
        index = boundary + offset
        if index < 0 or index >= len(events):
            window_rows.append(np.zeros(window_width, dtype=np.float64))
            continue
        name, qubits = events[index]
        raw_qubits = tuple(
            int(v) for v in qidx[int(qptr[index]) : int(qptr[index + 1])].tolist()
        )
        if raw_qubits != qubits:
            raise RuntimeError("root operation signature disagrees with serialized graph")
        role = np.zeros(4, dtype=np.float64)
        if affected in qubits:
            if len(qubits) == 1:
                role[0] = 1.0
            elif qubits[0] == affected:
                role[1] = 1.0
            elif len(qubits) > 1 and qubits[1] == affected:
                role[2] = 1.0
        role[3] = float(affected in qubits)
        q0 = onehot(qubits[0], MAX_QUBITS)
        q1 = onehot(qubits[1], MAX_QUBITS) if len(qubits) > 1 else np.zeros(MAX_QUBITS)
        affected_onehot = onehot(affected, MAX_QUBITS)
        window_rows.append(
            np.concatenate(
                (
                    onehot(GATE_VOCAB.index(name), len(GATE_VOCAB)),
                    q0, q1, affected_onehot, role, parameter_rows[index],
                )
            )
        )

    before_counts = np.zeros(len(GATE_VOCAB), dtype=np.float64)
    after_counts = np.zeros(len(GATE_VOCAB), dtype=np.float64)
    before_params = np.zeros(2, dtype=np.float64)
    after_params = np.zeros(2, dtype=np.float64)
    for index, (name, qubits) in enumerate(events):
        if affected not in qubits:
            continue
        target = before_counts if index < boundary else after_counts
        target[GATE_VOCAB.index(name)] += 1.0
        param_target = before_params if index < boundary else after_params
        param_target += parameter_rows[index][1:3]

    denominator = max(1.0, float(len(events)))
    base = np.concatenate(
        (
            np.concatenate(window_rows),
            before_counts / denominator,
            after_counts / denominator,
            before_params / denominator,
            after_params / denominator,
            np.asarray([boundary / denominator], dtype=np.float64),
            onehot(CONTEXT_VOCAB.index(context), len(CONTEXT_VOCAB)),
        )
    )
    # Fixed cross terms make local-evidence × local-frame relationships linearly
    # accessible to the same small probe. This is deterministic, not an encoder.
    interactions = np.outer(localized_core, base).reshape(-1)
    return np.concatenate((base, interactions))


def load_table(
    product: Path,
    rows: Sequence[Mapping[str, str]],
    roots: Mapping[int, Mapping[str, str]],
    progress_every: int,
) -> dict[str, Any]:
    features: dict[str, list[np.ndarray]] = {name: [] for name in PRIMARY_STAGES}
    truth: list[int] = []
    partitions: list[str] = []
    families: list[str] = []

    selected = [row for row in rows if str(row["mechanism"]) in MECHANISM_TO_TARGET]
    for position, row in enumerate(selected, start=1):
        root_index = int(row["root_index"])
        root = roots.get(root_index)
        if root is None:
            raise RuntimeError(f"missing root manifest entry {root_index}")
        partition = str(row["step14_partition"])
        if partition not in {"fit", "selection"} or str(root["step14_partition"]) != partition:
            raise RuntimeError("oracle diagnostic encountered outer/reserve or split mismatch")

        artifact = product / str(row["artifact_path"])
        if baseline.sha256_file(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"artifact hash mismatch for {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as archive:
            loaded = {key: archive[key] for key in archive.files}

        forbidden = [
            key for key in loaded
            if key.startswith("x__")
            and any(
                token in key.lower()
                for token in (
                    "mechanism_target", "effect_target", "phenomenology_target",
                    "affected_qubit", "injection_boundary", "injection_context",
                )
            )
        ]
        if forbidden:
            raise RuntimeError(
                f"privileged field leaked into deployable x__ input: {forbidden}"
            )

        raw, _ = raw_diagnostic_features(loaded)
        ordinary = ordinary_circuit_context(root, loaded)
        affected = int(root["affected_qubit"])
        localized, localized_core = localized_oracle_features(loaded, affected)
        local_context = local_insertion_context(root, loaded, localized_core)

        features["raw_diagnostics"].append(raw)
        features["raw_plus_circuit_context"].append(np.concatenate((raw, ordinary)))
        features["raw_plus_affected_qubit_oracle"].append(
            np.concatenate((raw, ordinary, localized))
        )
        features["raw_plus_affected_qubit_local_context_oracle"].append(
            np.concatenate((raw, ordinary, localized, local_context))
        )
        truth.append(MECHANISM_TO_TARGET[str(row["mechanism"])])
        partitions.append(partition)
        families.append(str(root["family_id"]))

        if progress_every and position % progress_every == 0:
            print(
                f"oracle/raw feature extraction {position}/{len(selected)}",
                flush=True,
            )

    arrays = {
        name: np.stack(values).astype(np.float32)
        for name, values in features.items()
    }
    y = np.asarray(truth, dtype=np.int64)
    partition_array = np.asarray(partitions, dtype=object)
    family_array = np.asarray(families, dtype=object)
    fit = partition_array == "fit"
    selection = partition_array == "selection"
    if not np.all(fit | selection) or np.any(fit & selection):
        raise RuntimeError("unexpected Step-14 development partition")

    for stage, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite feature in stage {stage}")
    for split_name, mask in (("fit", fit), ("selection", selection)):
        counts = Counter(int(v) for v in y[mask].tolist())
        if set(counts) != {0, 1, 2} or len(set(counts.values())) != 1:
            raise RuntimeError(
                f"{split_name} mechanism classes are not exactly balanced: {counts}"
            )
    if set(family_array[fit].tolist()) & set(family_array[selection].tolist()):
        raise RuntimeError("family leakage between fit and selection")

    return {
        "features": arrays,
        "truth": y,
        "partition": partition_array,
        "family": family_array,
    }


class SmallProbe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CeilingProbe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def balanced_accuracy(
    truth: np.ndarray, logits: np.ndarray
) -> tuple[float, list[float]]:
    guess = np.argmax(logits, axis=1).astype(np.int64)
    matrix = baseline.confusion_matrix(truth.astype(np.int64), guess, 3)
    recalls: list[float] = []
    for cls in range(3):
        denominator = float(np.sum(matrix[cls]))
        recalls.append(float(matrix[cls, cls] / denominator) if denominator else 0.0)
    return float(np.mean(recalls)), recalls


def fit_probe(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_selection: np.ndarray,
    y_selection: np.ndarray,
    *,
    seed: int,
    high_capacity: bool,
    device: torch.device,
) -> dict[str, Any]:
    mean = x_fit.mean(axis=0, keepdims=True)
    std = x_fit.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    fit_x = ((x_fit - mean) / std).astype(np.float32)
    selection_x = ((x_selection - mean) / std).astype(np.float32)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model: nn.Module = (
        CeilingProbe(fit_x.shape[1])
        if high_capacity
        else SmallProbe(fit_x.shape[1])
    )
    model.to(device)
    epochs = int(
        PROBE_SPEC["ceiling_epochs"]
        if high_capacity
        else PROBE_SPEC["primary_epochs"]
    )
    learning_rate = float(
        PROBE_SPEC["ceiling_learning_rate"]
        if high_capacity
        else PROBE_SPEC["primary_learning_rate"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(PROBE_SPEC["weight_decay"]),
    )
    batch_size = int(PROBE_SPEC["batch_size"])
    y_tensor = torch.as_tensor(y_fit, dtype=torch.long)
    final_loss = float("nan")

    for epoch in range(epochs):
        model.train()
        order = np.arange(len(y_fit), dtype=np.int64)
        np.random.default_rng(seed * 1000003 + epoch * 9176).shuffle(order)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x_batch = torch.as_tensor(
                fit_x[indices], dtype=torch.float32, device=device
            )
            y_batch = y_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(PROBE_SPEC["gradient_clip_norm"])
            )
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        final_loss = total_loss / max(1, seen)

    model.eval()

    def logits_for(value: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(value), 4096):
                batch = torch.as_tensor(
                    value[start : start + 4096], dtype=torch.float32, device=device
                )
                chunks.append(model(batch).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    fit_logits = logits_for(fit_x)
    selection_logits = logits_for(selection_x)
    fit_ba, fit_recall = balanced_accuracy(y_fit, fit_logits)
    selection_ba, selection_recall = balanced_accuracy(y_selection, selection_logits)
    return {
        "seed": seed,
        "final_fit_loss": final_loss,
        "fit_balanced_accuracy": fit_ba,
        "fit_recall": fit_recall,
        "selection_balanced_accuracy": selection_ba,
        "selection_recall": selection_recall,
        "selection_logits": selection_logits,
    }


def metric_record(truth: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    ba, recall = balanced_accuracy(truth, logits)
    return {
        "example_count": int(len(truth)),
        "mechanism_balanced_accuracy": ba,
        "mechanism_recall": recall,
        "minimum_mechanism_recall": float(min(recall)),
    }


def bootstrap_delta(
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(str(v) for v in groups.tolist())), dtype=object)
    by_group = {
        str(group): np.flatnonzero(groups == group)
        for group in unique.tolist()
    }
    observed = (
        balanced_accuracy(truth, candidate)[0]
        - balanced_accuracy(truth, reference)[0]
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(int(PROBE_SPEC["bootstrap_replicates"]), dtype=np.float64)
    for replicate in range(len(draws)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate(
            [by_group[str(group)] for group in sampled.tolist()]
        )
        draws[replicate] = (
            balanced_accuracy(truth[indices], candidate[indices])[0]
            - balanced_accuracy(truth[indices], reference[indices])[0]
        )
    lower, upper = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "mean_delta": float(observed),
        "bootstrap_ci": [float(lower), float(upper)],
        "bootstrap_replicates": int(len(draws)),
        "bootstrap_unit_count": int(len(unique)),
    }


def make_verdicts(
    metrics: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    meaningful = float(PROBE_SPEC["meaningful_oracle_delta_minimum"])
    localization = deltas["affected_qubit_localization_gain"]
    local_context = deltas["local_insertion_context_gain"]
    localization_positive = (
        float(localization["mean_delta"]) >= meaningful
        and float(localization["bootstrap_ci"][0]) > 0.0
    )
    local_context_positive = (
        float(local_context["mean_delta"]) >= meaningful
        and float(local_context["bootstrap_ci"][0]) > 0.0
    )

    raw_ceiling = float(
        metrics["raw_context_nonlinear_ceiling"]["mechanism_balanced_accuracy"]
    )
    nonlinear_gain = deltas["nonlinear_raw_capacity_gain"]
    if raw_ceiling < float(PROBE_SPEC["nonprivileged_low_ceiling_ba"]):
        ceiling = "CONSISTENT_WITH_LOW_NONPRIVILEGED_RAW_INPUT_CEILING"
    elif raw_ceiling >= float(PROBE_SPEC["nonprivileged_high_ceiling_ba"]) or (
        float(nonlinear_gain["mean_delta"])
        >= float(PROBE_SPEC["nonlinear_capacity_gain_minimum"])
        and float(nonlinear_gain["bootstrap_ci"][0]) > 0.0
    ):
        ceiling = "EVIDENCE_AGAINST_LOW_RAW_INPUT_CEILING__SUBSTANTIAL_SIGNAL_RECOVERABLE"
    else:
        ceiling = "INDETERMINATE_INTERMEDIATE_NONPRIVILEGED_RAW_INPUT_CEILING"

    return {
        "localization_or_local_context_hypothesis": (
            "SUPPORTED_AT_FROZEN_GATE"
            if localization_positive or local_context_positive
            else "NOT_SUPPORTED_AT_FROZEN_GATE"
        ),
        "affected_qubit_localization_component": (
            "SUPPORTED_AT_FROZEN_GATE"
            if localization_positive
            else "NOT_SUPPORTED_AT_FROZEN_GATE"
        ),
        "true_local_insertion_context_component": (
            "SUPPORTED_AT_FROZEN_GATE"
            if local_context_positive
            else "NOT_SUPPORTED_AT_FROZEN_GATE"
        ),
        "nonprivileged_raw_evidence_ceiling_hypothesis": ceiling,
        "ceiling_is_approximate_not_information_theoretic": True,
        "privileged_information_remains_analysis_only": True,
    }


def main() -> None:
    args = parse_args()
    config = step14.read_json(CONFIG)
    step14.assert_contract(config)
    _run_dir, _freeze = rep14.verify_training_freeze(
        args.training_run_id, args.selection_freeze_sha256
    )
    product = step14.resolve_cross_product(None)
    cross_rows, _rows_by_root, _fit_roots, _selection_roots = step14.verify_cross_product(
        product, config
    )
    complete = read_json(product / "dataset_complete.json")
    manifests = product / "manifests"
    root_rows = baseline.read_csv(manifests / "root_manifest.csv")
    roots = {int(row["root_index"]): row for row in root_rows}
    if len(roots) != len(root_rows):
        raise RuntimeError("duplicate root index in Step-14 root manifest")
    if any(str(row["step14_partition"]) not in {"fit", "selection"} for row in root_rows):
        raise RuntimeError("oracle diagnostic refuses outer/reserve root metadata")

    identity = {
        "schema": SCHEMA,
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "protocol_config_sha256": baseline.sha256_file(CONFIG),
        "cross_dataset_product_id": str(complete["product_id"]),
        "cross_dataset_complete_sha256": baseline.sha256_file(
            product / "dataset_complete.json"
        ),
        "root_manifest_sha256": baseline.sha256_file(manifests / "root_manifest.csv"),
        "example_manifest_sha256": baseline.sha256_file(
            manifests / "example_manifest.csv"
        ),
        "probe_spec": PROBE_SPEC,
        "primary_stages": list(PRIMARY_STAGES),
        "ceiling_stages": list(CEILING_STAGES),
        "ordinary_context_source": (
            "exact_step7_deterministic_graph_adapter_pre_encoder_features_and_structure"
        ),
        "privileged_fields": [
            "affected_qubit",
            "injection_boundary_rank",
            "injection_context_class",
        ],
        "privileged_fields_analysis_only": True,
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_accessed": False,
    }
    diagnostic_id = "oracle_raw_" + stable_hash(identity).split(":", 1)[1][:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_dir = output_parent / diagnostic_id
    complete_path = output_dir / "diagnostic_complete.json"
    if complete_path.is_file():
        previous = read_json(complete_path)
        if previous.get("identity") != identity:
            raise RuntimeError("existing oracle/raw diagnostic identity mismatch")
        print(json.dumps(previous, indent=2, sort_keys=True))
        return

    print(
        "STEP 14 ORACLE LOCALIZATION / RAW-EVIDENCE CEILING V2 — "
        "MAIN MODEL FROZEN — DEVELOPMENT FIT/SELECTION ONLY — NO QPU",
        flush=True,
    )
    print(
        "PRIVILEGED affected-qubit/boundary/context metadata is ANALYSIS ONLY",
        flush=True,
    )
    table = load_table(product, cross_rows, roots, args.progress_every)
    fit_mask = table["partition"] == "fit"
    selection_mask = table["partition"] == "selection"
    y_fit = table["truth"][fit_mask]
    y_selection = table["truth"][selection_mask]
    selection_families = table["family"][selection_mask]
    device = resolve_device(args.device)

    stage_source = {
        "raw_diagnostics": "raw_diagnostics",
        "raw_plus_circuit_context": "raw_plus_circuit_context",
        "raw_plus_affected_qubit_oracle": "raw_plus_affected_qubit_oracle",
        "raw_plus_affected_qubit_local_context_oracle": (
            "raw_plus_affected_qubit_local_context_oracle"
        ),
        "raw_context_nonlinear_ceiling": "raw_plus_circuit_context",
        "oracle_local_nonlinear_ceiling": (
            "raw_plus_affected_qubit_local_context_oracle"
        ),
    }
    per_seed: dict[str, dict[str, Any]] = {}
    averaged_logits: dict[str, np.ndarray] = {}
    ensemble_metrics: dict[str, Any] = {}

    for stage, source in stage_source.items():
        high_capacity = stage in CEILING_STAGES
        features = table["features"][source]
        logits_parts: list[np.ndarray] = []
        for seed in PROBE_SEEDS:
            result = fit_probe(
                features[fit_mask],
                y_fit,
                features[selection_mask],
                y_selection,
                seed=seed,
                high_capacity=high_capacity,
                device=device,
            )
            selection_logits = result.pop("selection_logits")
            logits_parts.append(selection_logits)
            per_seed.setdefault(str(seed), {})[stage] = result
            print(
                f"{stage} seed={seed} "
                f"fit_BA={result['fit_balanced_accuracy']:.4f} "
                f"selection_BA={result['selection_balanced_accuracy']:.4f}",
                flush=True,
            )
        mean_logits = np.mean(np.stack(logits_parts, axis=0), axis=0).astype(np.float32)
        averaged_logits[stage] = mean_logits
        ensemble_metrics[stage] = metric_record(y_selection, mean_logits)
        ensemble_metrics[stage]["feature_width"] = int(features.shape[1])
        ensemble_metrics[stage]["probe_kind"] = (
            "high_capacity_ceiling" if high_capacity else "small_fixed_probe"
        )

    comparisons = {
        "ordinary_circuit_context_gain": (
            "raw_plus_circuit_context", "raw_diagnostics"
        ),
        "affected_qubit_localization_gain": (
            "raw_plus_affected_qubit_oracle", "raw_plus_circuit_context"
        ),
        "local_insertion_context_gain": (
            "raw_plus_affected_qubit_local_context_oracle",
            "raw_plus_affected_qubit_oracle",
        ),
        "full_oracle_gain_over_nonprivileged": (
            "raw_plus_affected_qubit_local_context_oracle",
            "raw_plus_circuit_context",
        ),
        "nonlinear_raw_capacity_gain": (
            "raw_context_nonlinear_ceiling", "raw_plus_circuit_context"
        ),
        "nonlinear_oracle_capacity_gain": (
            "oracle_local_nonlinear_ceiling",
            "raw_plus_affected_qubit_local_context_oracle",
        ),
        "oracle_ceiling_over_nonprivileged_ceiling": (
            "oracle_local_nonlinear_ceiling", "raw_context_nonlinear_ceiling"
        ),
    }
    deltas: dict[str, Any] = {}
    for offset, (name, (candidate, reference)) in enumerate(comparisons.items()):
        record = bootstrap_delta(
            y_selection,
            averaged_logits[candidate],
            averaged_logits[reference],
            selection_families,
            seed=int(PROBE_SPEC["bootstrap_seed"]) + offset,
        )
        record.update({"candidate": candidate, "reference": reference})
        deltas[name] = record

    verdicts = make_verdicts(ensemble_metrics, deltas)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_ORACLE_RAW_EVIDENCE_CEILING_V2",
        "diagnostic_id": diagnostic_id,
        "identity": identity,
        "scientific_boundaries": {
            "main_model_retrained": False,
            "main_model_weights_updated": False,
            "only_disposable_probe_classifiers_trained": True,
            "probe_training_partition": "fit_only",
            "selection_used_for_probe_training_or_early_stopping": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
            "affected_qubit_and_insertion_metadata_analysis_only": True,
            "oracle_features_written_back_to_dataset_or_checkpoint": False,
        },
        "feature_contract": {
            "raw_diagnostics": (
                "finite-shot signed local/pair/parity model-visible diagnostics, "
                "observed/reference shots, reference mask/kind; no learned encoder"
            ),
            "raw_plus_circuit_context": (
                "raw diagnostics plus exact deterministic Step-7 pre-encoder graph "
                "node/gate/edge features and graph structure; no affected qubit or "
                "insertion boundary/context"
            ),
            "raw_plus_affected_qubit_oracle": (
                "previous stage plus analysis-only affected-qubit identity and "
                "localized already-visible local/incident diagnostics"
            ),
            "raw_plus_affected_qubit_local_context_oracle": (
                "previous stage plus analysis-only true insertion boundary/context "
                "and actual local gate/qubit/parameter neighborhood"
            ),
            "raw_context_nonlinear_ceiling": (
                "intentionally high-capacity nonlinear probe on nonprivileged raw "
                "diagnostics + ordinary circuit context"
            ),
            "oracle_local_nonlinear_ceiling": (
                "same high-capacity probe on richest analysis-only oracle stage"
            ),
        },
        "counts": {
            "fit_injected_examples": int(np.sum(fit_mask)),
            "selection_injected_examples": int(np.sum(selection_mask)),
            "fit_families": int(len(set(table["family"][fit_mask].tolist()))),
            "selection_families": int(
                len(set(table["family"][selection_mask].tolist()))
            ),
        },
        "per_seed": per_seed,
        "selection_ensemble_metrics": ensemble_metrics,
        "paired_family_bootstrap_deltas": deltas,
        "hypothesis_verdicts": verdicts,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "diagnostic_result.json"
    atomic_json(result_path, result)
    result_sha = baseline.sha256_file(result_path)
    completion = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_ORACLE_RAW_EVIDENCE_CEILING_V2",
        "diagnostic_id": diagnostic_id,
        "identity": identity,
        "diagnostic_result_sha256": result_sha,
        "hypothesis_verdicts": verdicts,
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
        "privileged_information_analysis_only": True,
    }
    atomic_json(complete_path, completion)
    atomic_json(
        output_parent / "current_oracle_raw_evidence_diagnostic.json",
        {
            "schema": "triqto.v0_2.step14_current_oracle_raw_evidence_diagnostic.v2",
            "diagnostic_id": diagnostic_id,
            "diagnostic_dir": str(output_dir),
            "diagnostic_complete_sha256": baseline.sha256_file(complete_path),
            "diagnostic_result_sha256": result_sha,
        },
    )
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
