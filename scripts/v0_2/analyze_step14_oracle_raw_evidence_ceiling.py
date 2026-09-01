#!/usr/bin/env python3
"""Frozen Step-14 oracle-localization / raw-evidence ceiling decomposition.

This is analysis only. It never updates a TriQTO checkpoint, never materializes or
reads Step-14 simulator-outer/future-hardware-reserve data, and never executes a
QPU. Privileged affected-qubit and injection-boundary metadata are consumed only
inside the explicitly oracle-labelled feature stages written by this diagnostic.

The primary nested stages are deterministic feature constructions followed by the
same small probe classifier:
  1) raw model-visible finite-shot diagnostics, with no TriQTO learned encoder;
  2) raw diagnostics + ordinary model-visible circuit context;
  3) + privileged affected-qubit localization (analysis only);
  4) + privileged actual local gate neighborhood/insertion context (analysis only).
Two intentionally high-capacity nonlinear probes provide approximate ceilings for
(2) the non-privileged raw input and (4) the richest oracle input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import benchmark_step6_cheap_baselines as baseline
import run_step14_cross_motif_training as step14
import analyze_step14_representation_fusion_head as rep14

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_oracle_raw_evidence_ceiling")
SCHEMA = "triqto.v0_2.step14_oracle_raw_evidence_ceiling.v1"
MAX_QUBITS = 5
MAX_GATES = 16
GATE_VOCAB = ("h", "rx", "ry", "rz", "cx", "cz")
CONTEXT_VOCAB = ("pre_entangling", "inter_entangling", "post_entangling_recombination")
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
    "primary_probe": "one_hidden_layer_MLP_classifier_on_deterministic_features",
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
    "meaningful_oracle_delta_minimum": 0.05,
    "nonprivileged_low_ceiling_ba": 0.60,
    "nonprivileged_high_ceiling_ba": 0.70,
    "nonlinear_capacity_gain_minimum": 0.10,
    "bootstrap_replicates": 1000,
    "bootstrap_unit": "cross_motif_family_id",
    "bootstrap_seed": 2026090102,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for oracle/raw-evidence diagnostic but is unavailable")
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
    out: list[tuple[str, tuple[int, ...]]] = []
    for item in values:
        text = str(item)
        if ":" not in text:
            raise RuntimeError(f"malformed operation signature entry {text!r}")
        name, qtext = text.split(":", 1)
        if name not in GATE_VOCAB:
            raise RuntimeError(f"unexpected Step-14 gate {name!r}")
        qubits = tuple(int(piece[1:]) for piece in qtext.split("-") if piece)
        if not qubits or any(q < 0 or q >= MAX_QUBITS for q in qubits):
            raise RuntimeError(f"invalid qubits in operation signature {text!r}")
        out.append((name, qubits))
    return out


def ordinary_circuit_context(root: Mapping[str, str], loaded: Mapping[str, np.ndarray]) -> np.ndarray:
    """Deterministic circuit context available without hidden injection metadata."""
    events = parse_operation_signature(str(root["operation_signature"]))
    rows: list[np.ndarray] = []
    for pos in range(MAX_GATES):
        if pos >= len(events):
            rows.append(np.zeros(len(GATE_VOCAB) + 2 * MAX_QUBITS + 2, dtype=np.float64))
            continue
        name, qubits = events[pos]
        gate = onehot(GATE_VOCAB.index(name), len(GATE_VOCAB))
        q0 = onehot(qubits[0], MAX_QUBITS)
        q1 = onehot(qubits[1], MAX_QUBITS) if len(qubits) > 1 else np.zeros(MAX_QUBITS, dtype=np.float64)
        rows.append(np.concatenate((gate, q0, q1, np.asarray([float(len(qubits)), 1.0]))))
    return np.concatenate((baseline.graph_stats(loaded), np.concatenate(rows)))


def raw_diagnostic_features(loaded: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_slots = baseline.fixed_pair_slots(MAX_QUBITS)
    local, pair, parity = baseline.canonical_diag_features(
        loaded, max_n=MAX_QUBITS, pair_slot=pair_slots, exact=False
    )
    observed = np.asarray(loaded["x__observed_shots"], dtype=np.float64).reshape(-1)
    reference = np.asarray(loaded["x__reference_shots"], dtype=np.float64).reshape(-1)
    if observed.shape != (3,) or reference.shape != (3,) or np.any(observed <= 0) or np.any(reference <= 0):
        raise RuntimeError("unexpected Step-14 diagnostic shot-count contract")
    shot_meta = np.concatenate((np.log2(observed), np.log2(reference)))
    return np.concatenate((local, pair, parity, shot_meta)), local, pair


def localized_oracle_features(
    loaded: Mapping[str, np.ndarray], affected: int
) -> tuple[np.ndarray, np.ndarray]:
    """Use affected-qubit location only to extract already-model-visible diagnostics."""
    basis_codes = np.asarray(loaded["x__diagnostic_basis_codes"])
    local_raw = baseline.reorder_basis(
        np.asarray(loaded["x__delta_local_expectations"], dtype=np.float64), basis_codes
    )
    n_qubits = int(np.asarray(loaded["x__layout_logical_to_physical"]).size)
    if affected < 0 or affected >= n_qubits or local_raw.shape != (3, n_qubits):
        raise RuntimeError("affected-qubit oracle is outside the circuit")
    pair_raw = baseline.reorder_basis(
        np.asarray(loaded["x__delta_pairwise_correlations"], dtype=np.float64), basis_codes
    )
    pairs = np.asarray(loaded["x__pair_indices"], dtype=np.int64).reshape(-1, 2)
    incident = [i for i, pair in enumerate(pairs.tolist()) if affected in pair]
    incident_mean = np.mean(pair_raw[:, incident], axis=1) if incident else np.zeros(3, dtype=np.float64)
    affected_local = local_raw[:, affected].astype(np.float64)
    core = np.concatenate((affected_local, incident_mean))
    return np.concatenate((onehot(affected, MAX_QUBITS), core)), core


def local_insertion_context(root: Mapping[str, str], localized_core: np.ndarray) -> np.ndarray:
    """Privileged local neighborhood around the true insertion boundary, analysis only."""
    affected = int(root["affected_qubit"])
    boundary = int(root["injection_boundary_rank"])
    events = parse_operation_signature(str(root["operation_signature"]))
    if boundary < 0 or boundary > len(events):
        raise RuntimeError("invalid oracle injection boundary")
    context = str(root["injection_context_class"])
    if context not in CONTEXT_VOCAB:
        raise RuntimeError("unexpected Step-14 injection-context class")

    window: list[np.ndarray] = []
    for offset in (-2, -1, 0, 1):
        idx = boundary + offset
        if idx < 0 or idx >= len(events):
            window.append(np.zeros(len(GATE_VOCAB) + 4, dtype=np.float64))
            continue
        name, qubits = events[idx]
        role = np.zeros(3, dtype=np.float64)
        if affected in qubits:
            if len(qubits) == 1:
                role[0] = 1.0
            elif qubits[0] == affected:
                role[1] = 1.0
            elif len(qubits) > 1 and qubits[1] == affected:
                role[2] = 1.0
        window.append(
            np.concatenate(
                (
                    onehot(GATE_VOCAB.index(name), len(GATE_VOCAB)),
                    np.asarray([float(affected in qubits)]),
                    role,
                )
            )
        )

    before = np.zeros(len(GATE_VOCAB), dtype=np.float64)
    after = np.zeros(len(GATE_VOCAB), dtype=np.float64)
    for idx, (name, qubits) in enumerate(events):
        if affected not in qubits:
            continue
        target = before if idx < boundary else after
        target[GATE_VOCAB.index(name)] += 1.0
    denom = max(1.0, float(len(events)))
    base = np.concatenate(
        (
            np.concatenate(window),
            before / denom,
            after / denom,
            np.asarray([boundary / denom], dtype=np.float64),
            onehot(CONTEXT_VOCAB.index(context), len(CONTEXT_VOCAB)),
        )
    )
    # Fixed, non-learned cross terms expose the local frame to the same small probe.
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
    source_rows: list[int] = []

    wanted = {"rz_drift": 0, "rx_overrotation": 1, "ry_overrotation": 2}
    selected = [(idx, row) for idx, row in enumerate(rows) if str(row["mechanism"]) in wanted]
    for pos, (source_index, row) in enumerate(selected, start=1):
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
        with np.load(artifact, allow_pickle=False) as z:
            loaded = {key: z[key] for key in z.files}
        forbidden = [key for key in loaded if key.startswith("x__") and any(tok in key.lower() for tok in ("mechanism_target", "effect_target", "affected_qubit", "injection_boundary"))]
        if forbidden:
            raise RuntimeError(f"privileged field leaked into deployable x__ input: {forbidden}")

        raw, _raw_local, _raw_pair = raw_diagnostic_features(loaded)
        ordinary = ordinary_circuit_context(root, loaded)
        affected = int(root["affected_qubit"])
        localized, localized_core = localized_oracle_features(loaded, affected)
        local_context = local_insertion_context(root, localized_core)

        features["raw_diagnostics"].append(raw)
        features["raw_plus_circuit_context"].append(np.concatenate((raw, ordinary)))
        features["raw_plus_affected_qubit_oracle"].append(np.concatenate((raw, ordinary, localized)))
        features["raw_plus_affected_qubit_local_context_oracle"].append(
            np.concatenate((raw, ordinary, localized, local_context))
        )
        truth.append(wanted[str(row["mechanism"])])
        partitions.append(partition)
        families.append(str(row["family_id"]))
        source_rows.append(source_index)
        if progress_every and pos % progress_every == 0:
            print(f"oracle/raw feature extraction {pos}/{len(selected)}", flush=True)

    arrays = {name: np.stack(values).astype(np.float32) for name, values in features.items()}
    y = np.asarray(truth, dtype=np.int64)
    part = np.asarray(partitions, dtype=object)
    fam = np.asarray(families, dtype=object)
    src = np.asarray(source_rows, dtype=np.int64)
    if not np.all(np.isfinite(np.concatenate([value.reshape(len(y), -1) for value in arrays.values()], axis=1))):
        raise RuntimeError("non-finite oracle/raw diagnostic feature")
    fit = part == "fit"
    selection = part == "selection"
    if np.any(fit & selection) or not np.all(fit | selection):
        raise RuntimeError("unexpected Step-14 development partition")
    for name, mask in (("fit", fit), ("selection", selection)):
        counts = Counter(int(v) for v in y[mask].tolist())
        if set(counts) != {0, 1, 2} or len(set(counts.values())) != 1:
            raise RuntimeError(f"{name} mechanism classes are not exactly balanced: {counts}")
    return {"features": arrays, "truth": y, "partition": part, "family": fam, "source_rows": src}


class SmallProbe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3))

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


def balanced_accuracy(truth: np.ndarray, logits: np.ndarray) -> tuple[float, list[float]]:
    guess = np.argmax(logits, axis=1).astype(np.int64)
    cm = baseline.confusion_matrix(truth.astype(np.int64), guess, 3)
    recalls = []
    for cls in range(3):
        denom = float(np.sum(cm[cls]))
        recalls.append(float(cm[cls, cls] / denom) if denom else 0.0)
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
    sel_x = ((x_selection - mean) / std).astype(np.float32)

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    model: nn.Module = CeilingProbe(fit_x.shape[1]) if high_capacity else SmallProbe(fit_x.shape[1])
    model.to(device)
    epochs = int(PROBE_SPEC["ceiling_epochs"] if high_capacity else PROBE_SPEC["primary_epochs"])
    lr = float(PROBE_SPEC["ceiling_learning_rate"] if high_capacity else PROBE_SPEC["primary_learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(PROBE_SPEC["weight_decay"]))
    batch_size = int(PROBE_SPEC["batch_size"])
    y_tensor = torch.as_tensor(y_fit, dtype=torch.long)
    final_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        order = np.arange(len(y_fit), dtype=np.int64)
        np.random.default_rng(seed * 1000003 + epoch * 9176).shuffle(order)
        total = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(fit_x[idx], dtype=torch.float32, device=device)
            yb = y_tensor[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(PROBE_SPEC["gradient_clip_norm"]))
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx); seen += len(idx)
        final_loss = total / max(1, seen)

    model.eval()
    def logits_for(x: np.ndarray) -> np.ndarray:
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                xb = torch.as_tensor(x[start:start + 4096], dtype=torch.float32, device=device)
                chunks.append(model(xb).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)
    fit_logits = logits_for(fit_x)
    selection_logits = logits_for(sel_x)
    fit_ba, fit_recall = balanced_accuracy(y_fit, fit_logits)
    sel_ba, sel_recall = balanced_accuracy(y_selection, selection_logits)
    return {
        "seed": seed,
        "final_fit_loss": final_loss,
        "fit_balanced_accuracy": fit_ba,
        "fit_recall": fit_recall,
        "selection_balanced_accuracy": sel_ba,
        "selection_recall": sel_recall,
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
    by_group = {str(group): np.flatnonzero(groups == group) for group in unique.tolist()}
    observed = balanced_accuracy(truth, candidate)[0] - balanced_accuracy(truth, reference)[0]
    rng = np.random.default_rng(seed)
    draws = np.empty(int(PROBE_SPEC["bootstrap_replicates"]), dtype=np.float64)
    for rep in range(len(draws)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_group[str(group)] for group in sampled.tolist()])
        draws[rep] = balanced_accuracy(truth[idx], candidate[idx])[0] - balanced_accuracy(truth[idx], reference[idx])[0]
    lo, hi = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "mean_delta": float(observed),
        "bootstrap_ci": [float(lo), float(hi)],
        "bootstrap_replicates": int(len(draws)),
        "bootstrap_unit_count": int(len(unique)),
    }


def verdicts(metrics: Mapping[str, Mapping[str, Any]], deltas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    threshold = float(PROBE_SPEC["meaningful_oracle_delta_minimum"])
    loc = deltas["affected_qubit_localization_gain"]
    ctx = deltas["local_insertion_context_gain"]
    localization_supported = (
        (float(loc["mean_delta"]) >= threshold and float(loc["bootstrap_ci"][0]) > 0.0)
        or (float(ctx["mean_delta"]) >= threshold and float(ctx["bootstrap_ci"][0]) > 0.0)
    )
    raw_ceiling = float(metrics["raw_context_nonlinear_ceiling"]["mechanism_balanced_accuracy"])
    nonlinear_gain = deltas["nonlinear_raw_capacity_gain"]
    if raw_ceiling < float(PROBE_SPEC["nonprivileged_low_ceiling_ba"]):
        ceiling_verdict = "SUPPORTED_LOW_NONPRIVILEGED_RAW_INPUT_CEILING"
    elif raw_ceiling >= float(PROBE_SPEC["nonprivileged_high_ceiling_ba"]) or (
        float(nonlinear_gain["mean_delta"]) >= float(PROBE_SPEC["nonlinear_capacity_gain_minimum"])
        and float(nonlinear_gain["bootstrap_ci"][0]) > 0.0
    ):
        ceiling_verdict = "REFUTED__HIGH_CAPACITY_RAW_INPUT_RECOVERS_SUBSTANTIAL_SIGNAL"
    else:
        ceiling_verdict = "INDETERMINATE_INTERMEDIATE_RAW_INPUT_CEILING"
    return {
        "localization_or_local_context_hypothesis": "SUPPORTED" if localization_supported else "NOT_SUPPORTED_AT_FROZEN_GATE",
        "raw_model_visible_evidence_ceiling_hypothesis": ceiling_verdict,
        "privileged_information_remains_analysis_only": True,
    }


def main() -> None:
    args = parse_args()
    cfg = step14.read_json(CONFIG); step14.assert_contract(cfg)
    _run_dir, freeze = rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)
    cross_product = step14.resolve_cross_product(None)
    cross_rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(cross_product, cfg)
    complete = read_json(cross_product / "dataset_complete.json")
    manifests = cross_product / "manifests"
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
        "cross_dataset_complete_sha256": baseline.sha256_file(cross_product / "dataset_complete.json"),
        "root_manifest_sha256": baseline.sha256_file(manifests / "root_manifest.csv"),
        "example_manifest_sha256": baseline.sha256_file(manifests / "example_manifest.csv"),
        "probe_spec": PROBE_SPEC,
        "primary_stages": list(PRIMARY_STAGES),
        "ceiling_stages": list(CEILING_STAGES),
        "privileged_fields": ["affected_qubit", "injection_boundary_rank", "injection_context_class"],
        "privileged_fields_analysis_only": True,
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_accessed": False,
    }
    diagnostic_id = "oracle_raw_" + stable_hash(identity).split(":", 1)[1][:24]
    out_dir = args.output_parent.expanduser().resolve() / diagnostic_id
    complete_path = out_dir / "diagnostic_complete.json"
    if complete_path.is_file():
        previous = read_json(complete_path)
        if previous.get("identity") != identity:
            raise RuntimeError("existing oracle/raw diagnostic identity mismatch")
        print(json.dumps(previous, indent=2, sort_keys=True)); return

    print("STEP 14 ORACLE LOCALIZATION / RAW-EVIDENCE CEILING — MAIN MODEL FROZEN — NO OUTER / NO QPU", flush=True)
    print("PRIVILEGED affected-qubit/boundary metadata is ANALYSIS ONLY", flush=True)
    table = load_table(cross_product, cross_rows, roots, args.progress_every)
    fit_mask = table["partition"] == "fit"; selection_mask = table["partition"] == "selection"
    y_fit = table["truth"][fit_mask]; y_sel = table["truth"][selection_mask]
    groups_sel = table["family"][selection_mask]
    device = resolve_device(args.device)

    per_seed: dict[str, dict[str, Any]] = {}
    averaged_logits: dict[str, np.ndarray] = {}
    ensemble_metrics: dict[str, Any] = {}
    stage_sources = {
        **{name: name for name in PRIMARY_STAGES},
        "raw_context_nonlinear_ceiling": "raw_plus_circuit_context",
        "oracle_local_nonlinear_ceiling": "raw_plus_affected_qubit_local_context_oracle",
    }
    for stage, source in stage_sources.items():
        high_capacity = stage in CEILING_STAGES
        x = table["features"][source]
        parts = []
        for seed in PROBE_SEEDS:
            result = fit_probe(
                x[fit_mask], y_fit, x[selection_mask], y_sel,
                seed=seed, high_capacity=high_capacity, device=device,
            )
            logits = result.pop("selection_logits")
            parts.append(logits)
            per_seed.setdefault(str(seed), {})[stage] = result
            print(
                f"{stage} seed={seed} fit_BA={result['fit_balanced_accuracy']:.4f} "
                f"selection_BA={result['selection_balanced_accuracy']:.4f}", flush=True,
            )
        mean_logits = np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)
        averaged_logits[stage] = mean_logits
        ensemble_metrics[stage] = metric_record(y_sel, mean_logits)
        ensemble_metrics[stage]["feature_width"] = int(x.shape[1])
        ensemble_metrics[stage]["probe_kind"] = "high_capacity_ceiling" if high_capacity else "small_fixed_probe"

    comparisons = {
        "ordinary_circuit_context_gain": ("raw_plus_circuit_context", "raw_diagnostics"),
        "affected_qubit_localization_gain": ("raw_plus_affected_qubit_oracle", "raw_plus_circuit_context"),
        "local_insertion_context_gain": ("raw_plus_affected_qubit_local_context_oracle", "raw_plus_affected_qubit_oracle"),
        "nonlinear_raw_capacity_gain": ("raw_context_nonlinear_ceiling", "raw_plus_circuit_context"),
        "nonlinear_oracle_capacity_gain": ("oracle_local_nonlinear_ceiling", "raw_plus_affected_qubit_local_context_oracle"),
        "oracle_ceiling_over_nonprivileged_ceiling": ("oracle_local_nonlinear_ceiling", "raw_context_nonlinear_ceiling"),
    }
    deltas = {}
    for offset, (name, (candidate, reference)) in enumerate(comparisons.items()):
        record = bootstrap_delta(
            y_sel, averaged_logits[candidate], averaged_logits[reference], groups_sel,
            seed=int(PROBE_SPEC["bootstrap_seed"]) + offset,
        )
        record.update({"candidate": candidate, "reference": reference})
        deltas[name] = record

    verdict = verdicts(ensemble_metrics, deltas)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_ORACLE_RAW_EVIDENCE_CEILING",
        "diagnostic_id": diagnostic_id,
        "identity": identity,
        "scientific_boundaries": {
            "main_model_retrained": False,
            "main_model_weights_updated": False,
            "selection_used_for_probe_training_or_early_stopping": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
            "affected_qubit_and_insertion_metadata_analysis_only": True,
            "oracle_features_written_back_to_dataset_or_checkpoint": False,
        },
        "feature_contract": {
            "raw_diagnostics": "finite-shot signed local/pair/parity diagnostics plus observed/reference shot metadata; no TriQTO learned encoder",
            "raw_plus_circuit_context": "raw diagnostics plus deterministic ordinary reference-circuit gate sequence and graph statistics; excludes affected qubit and injection boundary/context",
            "raw_plus_affected_qubit_oracle": "previous stage plus analysis-only affected-qubit one-hot and deterministic extraction of local/incident raw diagnostics",
            "raw_plus_affected_qubit_local_context_oracle": "previous stage plus analysis-only true insertion boundary, context class, and local before/after gate neighborhood with fixed interaction expansion",
            "raw_context_nonlinear_ceiling": "intentionally high-capacity nonlinear probe on nonprivileged raw diagnostics + ordinary circuit context",
            "oracle_local_nonlinear_ceiling": "intentionally high-capacity nonlinear probe on the richest analysis-only oracle stage",
        },
        "counts": {
            "fit_injected_examples": int(np.sum(fit_mask)),
            "selection_injected_examples": int(np.sum(selection_mask)),
            "fit_families": int(len(set(table["family"][fit_mask].tolist()))),
            "selection_families": int(len(set(table["family"][selection_mask].tolist()))),
        },
        "per_seed": per_seed,
        "selection_ensemble_metrics": ensemble_metrics,
        "paired_family_bootstrap_deltas": deltas,
        "hypothesis_verdicts": verdict,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(out_dir / "diagnostic_result.json", result)
    result_sha = baseline.sha256_file(out_dir / "diagnostic_result.json")
    completion = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_ORACLE_RAW_EVIDENCE_CEILING",
        "diagnostic_id": diagnostic_id,
        "identity": identity,
        "diagnostic_result_sha256": result_sha,
        "hypothesis_verdicts": verdict,
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
        "privileged_information_analysis_only": True,
    }
    atomic_json(complete_path, completion)
    atomic_json(args.output_parent.expanduser().resolve() / "current_oracle_raw_evidence_diagnostic.json", {
        "schema": "triqto.v0_2.step14_current_oracle_raw_evidence_diagnostic.v1",
        "diagnostic_id": diagnostic_id,
        "diagnostic_dir": str(out_dir),
        "diagnostic_complete_sha256": baseline.sha256_file(complete_path),
        "diagnostic_result_sha256": result_sha,
    })
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
