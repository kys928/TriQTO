#!/usr/bin/env python3
"""Step-14 oracle-free latent-location local-frame inference diagnostic.

This is a frozen post-hoc diagnostic. It never updates the main TriQTO model and
uses only Step-14 FIT + selection development families. The true affected qubit
and true injection boundary are forbidden from candidate construction, scoring,
probe fitting, hyperparameter selection, and mechanism prediction. They are read
only after predictions have been formed, solely for localization audit metrics.

Primary candidate frames are hardware-facing in data contract: for every
protocol-plausible (qubit, boundary) candidate, the simulator emulates finite-shot
+/- rotation calibration circuits in X/Y/Z measurement settings. Exact
statevectors are transient simulator machinery only; exact Jacobians are computed
separately as an analysis-only upper bound and cannot determine the primary
verdict.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

import analyze_step14_local_frame_canonicalization as frame
import analyze_step14_oracle_raw_evidence_ceiling as oracle
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import generate_step5_matched_diagnostic_training_dataset_v3 as step5v3
import run_step14_cross_motif_training as step14

BASE = step5v3.BASE
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
LATENT_CONFIG = ROOT / "configs/v0_2/step14_latent_frame_inference.json"
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_latent_frame_inference")
LOCAL_FRAME_PARENT = Path("/workspace/triqto-data/step14_local_frame_canonicalization")
SCHEMA = "triqto.v0_2.step14_latent_frame_inference_result.v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_frozen_config() -> dict[str, Any]:
    cfg = json.loads(LATENT_CONFIG.read_text(encoding="utf-8"))
    if cfg.get("schema") != "triqto.v0_2.step14_latent_frame_inference.v1":
        raise RuntimeError("unexpected latent-frame protocol schema")
    if cfg.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise RuntimeError("latent-frame protocol is not frozen before execution")
    return cfg


def verify_local_frame_source(cfg: Mapping[str, Any]) -> dict[str, Any]:
    frozen = cfg["source_freeze"]
    pointer_path = LOCAL_FRAME_PARENT / "current_local_frame_canonicalization.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if str(pointer.get("diagnostic_id")) != str(frozen["local_frame_diagnostic_id"]):
        raise RuntimeError("local-frame source diagnostic id drift")
    if str(pointer.get("diagnostic_result_sha256")) != str(frozen["local_frame_result_sha256"]):
        raise RuntimeError("local-frame source result pointer hash drift")
    result_path = LOCAL_FRAME_PARENT / str(frozen["local_frame_diagnostic_id"]) / "diagnostic_result.json"
    if baseline.sha256_file(result_path) != str(frozen["local_frame_result_sha256"]):
        raise RuntimeError("local-frame source result bytes drift")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE_FROZEN_LOCAL_FRAME_CANONICALIZATION":
        raise RuntimeError("local-frame source result is not complete")
    return result


def operation_qubits(clean) -> list[tuple[str, tuple[int, ...]]]:
    output: list[tuple[str, tuple[int, ...]]] = []
    for item in clean.data:
        output.append(
            (
                str(item.operation.name).lower(),
                tuple(int(clean.find_bit(q).index) for q in item.qubits),
            )
        )
    return output


def plausible_candidates(clean) -> list[tuple[int, int]]:
    """Candidate set from known circuit structure only; no hidden location input."""
    operations = operation_qubits(clean)
    entanglers = [i for i, (name, _qs) in enumerate(operations) if name in {"cx", "cz"}]
    if not entanglers:
        raise RuntimeError("Step-14 circuit unexpectedly has no CX/CZ event")
    boundaries: set[int] = set()
    for index in entanglers:
        for boundary in (index, index + 1):
            if 1 <= boundary < len(operations):
                boundaries.add(int(boundary))
    candidates: list[tuple[int, int]] = []
    for boundary in sorted(boundaries):
        before = operations[:boundary]
        after = operations[boundary:]
        for qubit in range(clean.num_qubits):
            has_before = any(qubit in qs for _name, qs in before)
            has_after = any(qubit in qs for _name, qs in after)
            if has_before and has_after:
                candidates.append((int(qubit), int(boundary)))
    if not candidates:
        raise RuntimeError("no protocol-plausible latent location candidates")
    return candidates


def flatten_stats(stats_by_basis: Sequence[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    values: list[float] = []
    for local, pairwise, parity in stats_by_basis:
        values.extend(float(v) for v in np.asarray(local, dtype=np.float64).tolist())
        values.extend(float(v) for v in np.asarray(pairwise, dtype=np.float64).tolist())
        values.append(float(parity))
    return np.asarray(values, dtype=np.float64)


def sampled_evidence_vector(
    circuit,
    pairs: np.ndarray,
    *,
    shots: int,
    seed_parts: Sequence[Any],
) -> np.ndarray:
    """Return only sampled Born statistics; statevector is transient backend emulation."""
    n_qubits = int(circuit.num_qubits)
    state = BASE.normalized_state(circuit)
    eig = BASE.eigenvalue_table(n_qubits)
    stats: list[tuple[np.ndarray, np.ndarray, float]] = []
    for basis_name in frame.AXES:
        probabilities = BASE.basis_probabilities(state, n_qubits, basis_name)
        seed = BASE.stable_seed(*seed_parts, basis_name)
        stats.append(BASE.empirical_stats(probabilities, shots, eig, pairs, seed))
    # Deliberately return no state/probability object.
    return flatten_stats(stats)


def finite_shot_candidate_jacobians(
    clean,
    pairs: np.ndarray,
    candidates: Sequence[tuple[int, int]],
    root_index: int,
    cfg: Mapping[str, Any],
) -> list[np.ndarray]:
    spec = cfg["hardware_facing_frame_calibration"]
    epsilon = float(spec["probe_strength_radians"])
    shots = int(spec["shots_per_basis_per_sign"])
    base_seed = int(spec["calibration_seed"])
    if epsilon <= 0 or shots <= 0:
        raise RuntimeError("invalid frozen finite-shot calibration specification")
    result: list[np.ndarray] = []
    for qubit, boundary in candidates:
        jac = np.zeros((3 * (clean.num_qubits + len(pairs) + 1), 3), dtype=np.float64)
        for axis_index, mechanism in enumerate(frame.MECHANISMS):
            plus = BASE.inject_hidden_rotation(clean, boundary, qubit, mechanism, +epsilon)
            minus = BASE.inject_hidden_rotation(clean, boundary, qubit, mechanism, -epsilon)
            plus_vec = sampled_evidence_vector(
                plus,
                pairs,
                shots=shots,
                seed_parts=("step14-latent-frame", base_seed, root_index, qubit, boundary, mechanism, "plus"),
            )
            minus_vec = sampled_evidence_vector(
                minus,
                pairs,
                shots=shots,
                seed_parts=("step14-latent-frame", base_seed, root_index, qubit, boundary, mechanism, "minus"),
            )
            jac[:, axis_index] = (plus_vec - minus_vec) / (2.0 * epsilon)
        if not np.all(np.isfinite(jac)):
            raise RuntimeError("non-finite finite-shot candidate frame")
        result.append(jac)
    return result


def exact_candidate_jacobians(clean, pairs: np.ndarray, candidates: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    return [frame.frame_response_jacobian(clean, b, q, pairs) for q, b in candidates]


def profile_log_likelihoods(delta: np.ndarray, weights: np.ndarray, jacobians: Sequence[np.ndarray]) -> np.ndarray:
    """Profile over one signed amplitude for each candidate/mechanism axis."""
    dw = np.asarray(delta * weights, dtype=np.float64)
    scores = np.empty((len(jacobians), 3), dtype=np.float64)
    for candidate_index, jac in enumerate(jacobians):
        if jac.shape != (len(delta), 3):
            raise RuntimeError("candidate frame width mismatch")
        jw = np.asarray(jac * weights[:, None], dtype=np.float64)
        for mechanism_index in range(3):
            response = jw[:, mechanism_index]
            denominator = float(np.dot(response, response)) + 1.0e-12
            amplitude = float(np.dot(response, dw) / denominator)
            residual = dw - amplitude * response
            scores[candidate_index, mechanism_index] = -0.5 * float(np.dot(residual, residual))
    return scores


def logmeanexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    return np.squeeze(maximum, axis=axis) + np.log(np.mean(shifted, axis=axis))


def mechanism_scores(candidate_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return logmeanexp(candidate_scores, axis=0), np.max(candidate_scores, axis=0)


def pad_candidate_features(values: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        raise RuntimeError("empty candidate feature table")
    width = int(values[0].shape[1])
    max_candidates = max(int(v.shape[0]) for v in values)
    padded = np.zeros((len(values), max_candidates, width), dtype=np.float32)
    mask = np.zeros((len(values), max_candidates), dtype=np.bool_)
    for index, value in enumerate(values):
        if value.ndim != 2 or value.shape[1] != width:
            raise RuntimeError("candidate feature shape drift")
        count = int(value.shape[0])
        padded[index, :count] = value.astype(np.float32)
        mask[index, :count] = True
    return padded, mask


class LatentFrameMIL(nn.Module):
    def __init__(self, width: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_logits = self.net(x)
        masked = candidate_logits.masked_fill(~mask.unsqueeze(-1), -1.0e9)
        pooled = torch.logsumexp(masked, dim=1) - torch.log(mask.sum(dim=1, keepdim=True).to(x.dtype))
        return pooled, candidate_logits


def balanced_accuracy(y: np.ndarray, logits: np.ndarray) -> tuple[float, list[float]]:
    pred = np.argmax(logits, axis=1)
    recalls = [float(np.mean(pred[y == c] == c)) for c in range(3)]
    return float(np.mean(recalls)), recalls


def fit_latent_mil(
    x_fit: np.ndarray,
    mask_fit: np.ndarray,
    y_fit: np.ndarray,
    x_sel: np.ndarray,
    mask_sel: np.ndarray,
    y_sel: np.ndarray,
    *,
    seed: int,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    spec = cfg["latent_mil_probe"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    valid_fit = x_fit[mask_fit]
    mean = valid_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = valid_fit.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1.0e-6)
    x_fit_n = (x_fit - mean[None, None, :]) / std[None, None, :]
    x_sel_n = (x_sel - mean[None, None, :]) / std[None, None, :]
    x_fit_n[~mask_fit] = 0.0
    x_sel_n[~mask_sel] = 0.0

    model = LatentFrameMIL(x_fit.shape[2], int(spec["shared_hidden_dim"]), float(spec["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"])
    )
    batch_size = int(spec["batch_size"])
    epochs = int(spec["epochs"])
    generator = np.random.default_rng(seed)
    model.train()
    final_loss = 0.0
    for _epoch in range(epochs):
        permutation = generator.permutation(len(y_fit))
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(x_fit_n[indices]).to(device)
            mb = torch.from_numpy(mask_fit[indices]).to(device)
            yb = torch.from_numpy(y_fit[indices]).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            pooled, _candidate = model(xb, mb)
            loss = nn.functional.cross_entropy(pooled, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        final_loss = loss_sum / max(seen, 1)

    model.eval()
    def infer(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pooled_parts: list[np.ndarray] = []
        candidate_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[start:start + batch_size]).to(device)
                mb = torch.from_numpy(mask[start:start + batch_size]).to(device)
                pooled, candidate = model(xb, mb)
                pooled_parts.append(pooled.cpu().numpy())
                candidate_parts.append(candidate.cpu().numpy())
        return np.concatenate(pooled_parts), np.concatenate(candidate_parts)

    fit_logits, _fit_candidates = infer(x_fit_n, mask_fit)
    selection_logits, selection_candidate_logits = infer(x_sel_n, mask_sel)
    fit_ba, fit_recall = balanced_accuracy(y_fit, fit_logits)
    selection_ba, selection_recall = balanced_accuracy(y_sel, selection_logits)
    return {
        "seed": int(seed),
        "final_fit_loss": float(final_loss),
        "fit_balanced_accuracy": fit_ba,
        "fit_recall": fit_recall,
        "selection_balanced_accuracy": selection_ba,
        "selection_recall": selection_recall,
        "selection_logits": selection_logits.astype(np.float64),
        "selection_candidate_logits": selection_candidate_logits.astype(np.float64),
    }


def build_table(
    product: Path,
    rows: Sequence[Mapping[str, str]],
    roots: Mapping[int, Mapping[str, str]],
    cfg: Mapping[str, Any],
    progress_every: int,
) -> dict[str, Any]:
    selected = [(index, row) for index, row in enumerate(rows) if str(row["mechanism"]) in frame.TARGET]
    cache: dict[int, dict[str, Any]] = {}
    raw_features: list[np.ndarray] = []
    candidate_features: list[np.ndarray] = []
    finite_marginal: list[np.ndarray] = []
    finite_max: list[np.ndarray] = []
    exact_marginal: list[np.ndarray] = []
    exact_max: list[np.ndarray] = []
    truth: list[int] = []
    partitions: list[str] = []
    families: list[str] = []
    true_locations: list[tuple[int, int]] = []
    predicted_finite_profile_locations: list[tuple[int, int]] = []
    candidate_sets: list[list[tuple[int, int]]] = []
    candidate_counts: list[int] = []

    for position, (_source_index, row) in enumerate(selected, start=1):
        root_index = int(row["root_index"])
        root = roots.get(root_index)
        if root is None:
            raise RuntimeError(f"missing Step-14 root {root_index}")
        partition = str(row["step14_partition"])
        if partition not in {"fit", "selection"} or str(root["step14_partition"]) != partition:
            raise RuntimeError("latent-frame diagnostic encountered outer/reserve or split mismatch")
        artifact = product / str(row["artifact_path"])
        if baseline.sha256_file(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"artifact hash mismatch for {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as source:
            loaded = {key: source[key] for key in source.files}
        forbidden = [
            key for key in loaded
            if key.startswith("x__") and any(token in key.lower() for token in ("mechanism_target", "effect_target", "affected_qubit", "injection_boundary"))
        ]
        if forbidden:
            raise RuntimeError(f"privileged location/target leaked into x__ inputs: {forbidden}")

        raw, _local, _pair = oracle.raw_diagnostic_features(loaded)
        delta, weights, pairs = frame.measured_delta_and_weights(loaded)

        if root_index not in cache:
            clean = frame.circuit_from_serialized(loaded)
            signature = oracle.parse_operation_signature(str(root["operation_signature"]))
            if operation_qubits(clean) != signature:
                raise RuntimeError(f"reconstructed Step-14 circuit mismatch at root {root_index}")
            candidates = plausible_candidates(clean)
            finite_jacs = finite_shot_candidate_jacobians(clean, pairs, candidates, root_index, cfg)
            exact_jacs = exact_candidate_jacobians(clean, pairs, candidates)
            cache[root_index] = {
                "candidates": candidates,
                "finite_jacs": finite_jacs,
                "exact_jacs": exact_jacs,
            }
        root_cache = cache[root_index]
        candidates = root_cache["candidates"]
        finite_jacs = root_cache["finite_jacs"]
        exact_jacs = root_cache["exact_jacs"]

        finite_candidate_scores = profile_log_likelihoods(delta, weights, finite_jacs)
        finite_marg, finite_best = mechanism_scores(finite_candidate_scores)
        exact_candidate_scores = profile_log_likelihoods(delta, weights, exact_jacs)
        exact_marg, exact_best = mechanism_scores(exact_candidate_scores)
        per_candidate_features = np.stack(
            [frame.canonicalize_evidence(delta, jac, weights)[0] for jac in finite_jacs]
        ).astype(np.float32)

        # Prediction is fully formed before the privileged true location is read.
        best_flat = int(np.argmax(finite_candidate_scores))
        best_candidate_index = best_flat // 3
        predicted_location = tuple(candidates[best_candidate_index])

        true_location = (int(root["affected_qubit"]), int(root["injection_boundary_rank"]))
        if true_location not in candidates:
            raise RuntimeError(
                f"protocol-plausible candidate set failed to contain true location at root {root_index}: {true_location}"
            )

        raw_features.append(raw)
        candidate_features.append(per_candidate_features)
        finite_marginal.append(finite_marg)
        finite_max.append(finite_best)
        exact_marginal.append(exact_marg)
        exact_max.append(exact_best)
        truth.append(frame.TARGET[str(row["mechanism"])])
        partitions.append(partition)
        families.append(str(row["family_id"]))
        true_locations.append(true_location)
        predicted_finite_profile_locations.append(predicted_location)
        candidate_sets.append(list(candidates))
        candidate_counts.append(len(candidates))
        if progress_every and position % progress_every == 0:
            print(f"latent-frame extraction {position}/{len(selected)} roots_cached={len(cache)}", flush=True)

    raw_array = np.stack(raw_features).astype(np.float32)
    candidate_array, candidate_mask = pad_candidate_features(candidate_features)
    y = np.asarray(truth, dtype=np.int64)
    part = np.asarray(partitions, dtype=object)
    fam = np.asarray(families, dtype=object)
    for name, mask in (("fit", part == "fit"), ("selection", part == "selection")):
        counts = Counter(int(v) for v in y[mask].tolist())
        if set(counts) != {0, 1, 2} or len(set(counts.values())) != 1:
            raise RuntimeError(f"{name} mechanism classes are not balanced: {counts}")
    return {
        "raw": raw_array,
        "candidate_features": candidate_array,
        "candidate_mask": candidate_mask,
        "truth": y,
        "partition": part,
        "family": fam,
        "finite_marginal_logits": np.stack(finite_marginal).astype(np.float64),
        "finite_max_logits": np.stack(finite_max).astype(np.float64),
        "exact_marginal_logits": np.stack(exact_marginal).astype(np.float64),
        "exact_max_logits": np.stack(exact_max).astype(np.float64),
        "true_locations": true_locations,
        "finite_profile_locations": predicted_finite_profile_locations,
        "candidate_sets": candidate_sets,
        "candidate_count_summary": frame.quantiles(candidate_counts),
    }


def localization_audit(
    true_locations: Sequence[tuple[int, int]],
    predicted_locations: Sequence[tuple[int, int]],
) -> dict[str, float]:
    true = list(true_locations)
    pred = list(predicted_locations)
    if len(true) != len(pred):
        raise RuntimeError("localization audit length mismatch")
    return {
        "joint_exact_accuracy": float(np.mean([a == b for a, b in zip(true, pred)])),
        "qubit_accuracy": float(np.mean([a[0] == b[0] for a, b in zip(true, pred)])),
        "boundary_accuracy": float(np.mean([a[1] == b[1] for a, b in zip(true, pred)])),
    }


def mil_localization(
    pooled_logits: np.ndarray,
    candidate_logits: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_sets: Sequence[Sequence[tuple[int, int]]],
) -> list[tuple[int, int]]:
    mechanisms = np.argmax(pooled_logits, axis=1)
    output: list[tuple[int, int]] = []
    for i, mechanism in enumerate(mechanisms.tolist()):
        valid_count = int(np.sum(candidate_mask[i]))
        scores = candidate_logits[i, :valid_count, int(mechanism)]
        output.append(tuple(candidate_sets[i][int(np.argmax(scores))]))
    return output


def metric_record(y: np.ndarray, logits: np.ndarray, kind: str) -> dict[str, Any]:
    record = oracle.metric_record(y, logits)
    record["inference_kind"] = kind
    return record


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config()
    frame.oracle.MAX_GATES = frame.MAX_GATES
    protocol = step14.read_json(CONFIG)
    step14.assert_contract(protocol)
    rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)
    frozen = cfg["source_freeze"]
    if args.training_run_id != str(frozen["training_run_id"]):
        raise RuntimeError("latent-frame run id differs from frozen protocol")
    if args.selection_freeze_sha256 != str(frozen["selection_freeze_sha256"]):
        raise RuntimeError("latent-frame selection-freeze hash differs from frozen protocol")
    local_source = verify_local_frame_source(cfg)

    cross_product = step14.resolve_cross_product(None)
    cross_rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(cross_product, protocol)
    complete = json.loads((cross_product / "dataset_complete.json").read_text(encoding="utf-8"))
    if str(complete["product_id"]) != str(frozen["development_product_id"]):
        raise RuntimeError("Step-14 development product drift")
    if baseline.sha256_file(cross_product / "dataset_complete.json") != str(frozen["development_dataset_sha256"]):
        raise RuntimeError("Step-14 development dataset hash drift")
    manifests = cross_product / "manifests"
    root_rows = baseline.read_csv(manifests / "root_manifest.csv")
    roots = {int(row["root_index"]): row for row in root_rows}
    if len(roots) != len(root_rows):
        raise RuntimeError("duplicate root index")

    table = build_table(cross_product, cross_rows, roots, cfg, args.progress_every)
    fit_mask = table["partition"] == "fit"
    selection_mask = table["partition"] == "selection"
    y_fit = table["truth"][fit_mask]
    y_sel = table["truth"][selection_mask]
    groups_sel = table["family"][selection_mask]
    device = oracle.resolve_device(args.device)

    # Reproduce the frozen raw small-probe baseline to obtain selection logits for paired bootstrap.
    raw_logits: list[np.ndarray] = []
    raw_per_seed: dict[str, Any] = {}
    for seed in frame.PROBE_SEEDS:
        record = oracle.fit_probe(
            table["raw"][fit_mask], y_fit, table["raw"][selection_mask], y_sel,
            seed=seed, high_capacity=False, device=device,
        )
        raw_logits.append(np.asarray(record["selection_logits"], dtype=np.float64))
        raw_per_seed[str(seed)] = frame.public_probe_record(record)
    raw_ensemble = np.mean(np.stack(raw_logits), axis=0)

    x = table["candidate_features"]
    mask = table["candidate_mask"]
    mil_logits: list[np.ndarray] = []
    mil_candidate_logits: list[np.ndarray] = []
    mil_per_seed: dict[str, Any] = {}
    for seed in [int(v) for v in cfg["latent_mil_probe"]["probe_seeds"]]:
        record = fit_latent_mil(
            x[fit_mask], mask[fit_mask], y_fit,
            x[selection_mask], mask[selection_mask], y_sel,
            seed=seed, cfg=cfg, device=device,
        )
        mil_logits.append(record.pop("selection_logits"))
        mil_candidate_logits.append(record.pop("selection_candidate_logits"))
        mil_per_seed[str(seed)] = record
        print(
            f"latent MIL seed={seed} fit_BA={record['fit_balanced_accuracy']:.4f} "
            f"selection_BA={record['selection_balanced_accuracy']:.4f}", flush=True,
        )
    mil_ensemble = np.mean(np.stack(mil_logits), axis=0)
    mil_candidate_ensemble = np.mean(np.stack(mil_candidate_logits), axis=0)

    selection_logits = {
        "raw_small_probe": raw_ensemble,
        "finite_shot_profile_marginal": table["finite_marginal_logits"][selection_mask],
        "finite_shot_profile_max": table["finite_max_logits"][selection_mask],
        "finite_shot_latent_mil_ensemble": mil_ensemble,
        "exact_statevector_profile_marginal_upper_bound": table["exact_marginal_logits"][selection_mask],
        "exact_statevector_profile_max_upper_bound": table["exact_max_logits"][selection_mask],
    }
    metrics = {
        name: metric_record(
            y_sel, logits,
            "primary_hardware_facing_latent_mil" if name == "finite_shot_latent_mil_ensemble"
            else "hardware_facing_deterministic" if name.startswith("finite_shot")
            else "analysis_only_exact_upper_bound" if name.startswith("exact_statevector")
            else "frozen_raw_baseline",
        )
        for name, logits in selection_logits.items()
    }

    deltas: dict[str, Any] = {}
    comparisons = {
        "finite_mil_over_raw": ("finite_shot_latent_mil_ensemble", "raw_small_probe"),
        "finite_marginal_over_raw": ("finite_shot_profile_marginal", "raw_small_probe"),
        "finite_max_over_raw": ("finite_shot_profile_max", "raw_small_probe"),
        "exact_marginal_over_raw": ("exact_statevector_profile_marginal_upper_bound", "raw_small_probe"),
        "finite_mil_vs_exact_marginal": ("finite_shot_latent_mil_ensemble", "exact_statevector_profile_marginal_upper_bound"),
    }
    bootstrap_seed = int(cfg["statistics"]["bootstrap_seed"])
    for offset, (name, (candidate, reference)) in enumerate(comparisons.items()):
        record = oracle.bootstrap_delta(
            y_sel, selection_logits[candidate], selection_logits[reference], groups_sel,
            seed=bootstrap_seed + offset,
        )
        record.update({"candidate": candidate, "reference": reference})
        deltas[name] = record

    primary = metrics["finite_shot_latent_mil_ensemble"]
    primary_ba = float(primary["mechanism_balanced_accuracy"])
    primary_min_recall = float(primary["minimum_mechanism_recall"])
    raw_ba = float(metrics["raw_small_probe"]["mechanism_balanced_accuracy"])
    oracle_ba = float(frozen["local_frame_oracle_small_probe_ba"])
    denominator = max(oracle_ba - raw_ba, 1.0e-12)
    gain_recovery = float((primary_ba - raw_ba) / denominator)
    primary_delta = deltas["finite_mil_over_raw"]
    gate = cfg["primary_support_gate"]
    gate_checks = {
        "minimum_selection_mechanism_ba": primary_ba >= float(gate["minimum_selection_mechanism_ba"]),
        "minimum_selection_minimum_mechanism_recall": primary_min_recall >= float(gate["minimum_selection_minimum_mechanism_recall"]),
        "minimum_paired_ba_gain_over_raw": float(primary_delta["mean_delta"]) >= float(gate["minimum_paired_ba_gain_over_raw"]),
        "paired_gain_ci_lower_above_zero": float(primary_delta["bootstrap_ci"][0]) > 0.0,
        "minimum_fraction_oracle_gain_recovered": gain_recovery >= float(gate["minimum_fraction_of_oracle_canonicalization_gain_recovered"]),
    }
    full_support = all(gate_checks.values())
    partial_support = (
        not full_support
        and float(primary_delta["bootstrap_ci"][0]) > 0.0
        and primary_ba >= 0.60
    )
    policy = cfg["interpretation_policy"]
    verdict = (
        str(policy["full_support"]) if full_support
        else str(policy["partial_support"]) if partial_support
        else str(policy["failure"])
    )

    selection_true_locations = [table["true_locations"][i] for i in np.flatnonzero(selection_mask)]
    selection_profile_locations = [table["finite_profile_locations"][i] for i in np.flatnonzero(selection_mask)]
    selection_candidate_sets = [table["candidate_sets"][i] for i in np.flatnonzero(selection_mask)]
    mil_locations = mil_localization(
        mil_ensemble,
        mil_candidate_ensemble,
        mask[selection_mask],
        selection_candidate_sets,
    )
    location_audit = {
        "finite_profile_joint_ml": localization_audit(selection_true_locations, selection_profile_locations),
        "finite_latent_mil": localization_audit(selection_true_locations, mil_locations),
        "note": "True locations are read only after mechanism/location predictions are formed and do not enter any primary score or training path. Exact location accuracy is audit-only because different candidate locations can induce equivalent response frames.",
    }

    identity = {
        "schema": SCHEMA,
        "latent_protocol_sha256": baseline.sha256_file(LATENT_CONFIG),
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "cross_dataset_product_id": str(complete["product_id"]),
        "cross_dataset_complete_sha256": baseline.sha256_file(cross_product / "dataset_complete.json"),
        "root_manifest_sha256": baseline.sha256_file(manifests / "root_manifest.csv"),
        "example_manifest_sha256": baseline.sha256_file(manifests / "example_manifest.csv"),
        "source_local_frame_diagnostic_id": str(frozen["local_frame_diagnostic_id"]),
        "source_local_frame_result_sha256": str(frozen["local_frame_result_sha256"]),
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_accessed": False,
    }
    diagnostic_id = "latent_frame_" + oracle.stable_hash(identity).split(":", 1)[1][:24]
    out_parent = args.output_parent.expanduser().resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_parent / diagnostic_id
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite latent-frame diagnostic {out_dir}")
    out_dir.mkdir()

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_ORACLE_FREE_LATENT_FRAME_INFERENCE",
        "diagnostic_id": diagnostic_id,
        "identity": identity,
        "counts": {
            "fit_injected_examples": int(np.sum(fit_mask)),
            "selection_injected_examples": int(np.sum(selection_mask)),
            "fit_families": int(len(set(table["family"][fit_mask].tolist()))),
            "selection_families": int(len(set(table["family"][selection_mask].tolist()))),
            "candidate_count_per_example": table["candidate_count_summary"],
        },
        "selection_metrics": metrics,
        "paired_family_bootstrap_deltas": deltas,
        "latent_mil_per_seed": mil_per_seed,
        "raw_baseline_per_seed": raw_per_seed,
        "localization_audit": location_audit,
        "primary_support_gate": {
            "checks": gate_checks,
            "passed": full_support,
            "oracle_canonicalization_gain_recovered_fraction": gain_recovery,
            "raw_ba": raw_ba,
            "primary_finite_shot_latent_mil_ba": primary_ba,
            "oracle_local_frame_ba": oracle_ba,
        },
        "hypothesis_verdicts": {
            "oracle_free_latent_frame_inference": verdict,
            "hardware_facing_data_contract_in_simulation": bool(full_support),
            "physical_hardware_validation": False,
            "exact_statevector_candidate_search_is_analysis_only": True,
        },
        "scientific_boundaries": {
            "main_model_retrained": False,
            "main_model_weights_updated": False,
            "selection_used_for_training_or_early_stopping": False,
            "true_location_used_for_candidate_generation_scoring_or_training": False,
            "true_location_used_only_for_post_prediction_audit": True,
            "primary_candidate_frames_use_only_finite_shot_probe_evidence": True,
            "simulator_statevector_used_only_as_backend_emulator_for_primary_probe_counts": True,
            "exact_statevector_jacobians_used_only_for_analysis_upper_bound": True,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
        },
        "source_local_frame_verdicts": local_source.get("hypothesis_verdicts", {}),
    }
    atomic_json(out_dir / "diagnostic_result.json", result)
    result_sha = baseline.sha256_file(out_dir / "diagnostic_result.json")
    complete_payload = {
        "schema": SCHEMA,
        "status": result["status"],
        "diagnostic_id": diagnostic_id,
        "diagnostic_result_sha256": result_sha,
        "hypothesis_verdicts": result["hypothesis_verdicts"],
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    }
    atomic_json(out_dir / "diagnostic_complete.json", complete_payload)
    complete_sha = baseline.sha256_file(out_dir / "diagnostic_complete.json")
    pointer = {
        "schema": "triqto.v0_2.step14_current_latent_frame_inference.v1",
        "diagnostic_id": diagnostic_id,
        "diagnostic_dir": str(out_dir),
        "diagnostic_result_sha256": result_sha,
        "diagnostic_complete_sha256": complete_sha,
    }
    atomic_json(out_parent / "current_latent_frame_inference.json", pointer)
    print(json.dumps({**complete_payload, "diagnostic_complete_sha256": complete_sha}, indent=2), flush=True)


if __name__ == "__main__":
    main()
