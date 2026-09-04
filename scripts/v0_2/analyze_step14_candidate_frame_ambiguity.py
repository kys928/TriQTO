#!/usr/bin/env python3
"""Frozen Step-14 post-hoc candidate-frame ambiguity decomposition.

This diagnostic does not update the TriQTO model and does not alter the frozen
Step-14 latent-frame verdict. It reproduces the completed oracle-free latent MIL
and privileged local-frame reference on FIT + selection only, then uses true
(qubit, boundary) metadata strictly after predictions are formed for paired
counterfactual/audit decomposition.

Primary question: how much of the remaining privileged-local-frame advantage is
associated with qubit ambiguity, boundary ambiguity, mechanism-preserving
observationally equivalent boundaries, or mechanism-confusable frame geometry?
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np

import analyze_step14_latent_frame_inference as latent
import analyze_step14_local_frame_canonicalization as frame
import analyze_step14_oracle_raw_evidence_ceiling as oracle
import run_step14_oracle_raw_evidence_ceiling as oracle_compat
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import generate_step5_matched_diagnostic_training_dataset_v3 as step5v3
import run_step14_cross_motif_training as step14

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
AMBIGUITY_CONFIG = ROOT / "configs/v0_2/step14_candidate_frame_ambiguity_decomposition.json"
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_candidate_frame_ambiguity")
LATENT_PARENT = Path("/workspace/triqto-data/step14_latent_frame_inference")
LOCAL_FRAME_PARENT = Path("/workspace/triqto-data/step14_local_frame_canonicalization")
SCHEMA = "triqto.v0_2.step14_candidate_frame_ambiguity_decomposition_result.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--selection-freeze-sha256", required=True)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_config() -> dict[str, Any]:
    value = json.loads(AMBIGUITY_CONFIG.read_text(encoding="utf-8"))
    if value.get("schema") != "triqto.v0_2.step14_candidate_frame_ambiguity_decomposition.v1":
        raise RuntimeError("unexpected candidate-frame ambiguity protocol schema")
    if value.get("status") != "FROZEN_BEFORE_EXECUTION":
        raise RuntimeError("candidate-frame ambiguity protocol is not frozen before execution")
    return value


def verify_source(path: Path, expected_hash: str, expected_status: str) -> dict[str, Any]:
    if baseline.sha256_file(path) != expected_hash:
        raise RuntimeError(f"source diagnostic hash drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != expected_status:
        raise RuntimeError(f"source diagnostic status drift: {path}")
    return value


def normalize_columns(jacobian: np.ndarray) -> np.ndarray:
    array = np.asarray(jacobian, dtype=np.float64)
    norms = np.linalg.norm(array, axis=0)
    return array / np.maximum(norms, 1.0e-12)[None, :]


def axis_similarity(candidate: np.ndarray, true_frame: np.ndarray) -> np.ndarray:
    return np.abs(normalize_columns(candidate).T @ normalize_columns(true_frame))


def best_nonidentity_assignment_mean(similarity: np.ndarray) -> float:
    best = -1.0
    for perm in itertools.permutations(range(3)):
        if perm == (0, 1, 2):
            continue
        score = float(np.mean([similarity[i, perm[i]] for i in range(3)]))
        best = max(best, score)
    return best


def is_equivalent(similarity: np.ndarray, cfg: Mapping[str, Any]) -> bool:
    spec = cfg["frame_equivalence"]
    diagonal = np.diag(similarity)
    identity_mean = float(np.mean(diagonal))
    margin = identity_mean - best_nonidentity_assignment_mean(similarity)
    return bool(
        float(np.min(diagonal)) >= float(spec["minimum_same_axis_abs_cosine"])
        and margin >= float(spec["minimum_identity_assignment_margin"])
    )


def quantiles(values: Sequence[float]) -> dict[str, float]:
    return frame.quantiles(values)


def stable_logmeanexp_masked(candidate_logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(candidate_logits, dtype=np.float64)
    valid = np.asarray(mask, dtype=np.bool_)
    if logits.ndim != 3 or valid.shape != logits.shape[:2]:
        raise RuntimeError("restricted MIL pooling shape mismatch")
    counts = valid.sum(axis=1)
    if np.any(counts < 1):
        raise RuntimeError("restricted MIL pool unexpectedly empty")
    masked = np.where(valid[:, :, None], logits, -np.inf)
    maximum = np.max(masked, axis=1)
    shifted = np.where(valid[:, :, None], np.exp(masked - maximum[:, None, :]), 0.0)
    return maximum + np.log(np.sum(shifted, axis=1)) - np.log(counts[:, None])


def metric(y: np.ndarray, logits: np.ndarray, kind: str) -> dict[str, Any]:
    out = oracle.metric_record(y, logits)
    out["inference_kind"] = kind
    return out


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
    finite_candidate_features: list[np.ndarray] = []
    true_finite_features: list[np.ndarray] = []
    true_exact_features: list[np.ndarray] = []
    truth: list[int] = []
    partitions: list[str] = []
    families: list[str] = []
    candidates_all: list[list[tuple[int, int]]] = []
    true_locations: list[tuple[int, int]] = []
    true_indices: list[int] = []
    true_qubit_masks: list[np.ndarray] = []
    equivalence_masks: list[np.ndarray] = []
    similarity_tables: list[np.ndarray] = []
    candidate_counts: list[int] = []
    true_frame_min_angles: list[float] = []

    for position, (_source_index, row) in enumerate(selected, start=1):
        root_index = int(row["root_index"])
        root = roots.get(root_index)
        if root is None:
            raise RuntimeError(f"missing Step-14 root {root_index}")
        partition = str(row["step14_partition"])
        if partition not in {"fit", "selection"} or str(root["step14_partition"]) != partition:
            raise RuntimeError("ambiguity diagnostic encountered outer/reserve or split mismatch")
        artifact = product / str(row["artifact_path"])
        if baseline.sha256_file(artifact) != str(row["artifact_sha256"]):
            raise RuntimeError(f"artifact hash mismatch for {row['example_id']}")
        with np.load(artifact, allow_pickle=False) as source:
            loaded = {key: source[key] for key in source.files}
        forbidden = [
            key for key in loaded
            if key.startswith("x__") and any(
                token in key.lower()
                for token in ("mechanism_target", "effect_target", "affected_qubit", "injection_boundary")
            )
        ]
        if forbidden:
            raise RuntimeError(f"privileged location/target leaked into x__ inputs: {forbidden}")

        raw, _local, _pair = oracle.raw_diagnostic_features(loaded)
        delta, weights, pairs = frame.measured_delta_and_weights(loaded)

        if root_index not in cache:
            clean = frame.circuit_from_serialized(loaded)
            signature = oracle.parse_operation_signature(str(root["operation_signature"]))
            if latent.operation_qubits(clean) != signature:
                raise RuntimeError(f"reconstructed Step-14 circuit mismatch at root {root_index}")
            candidates = latent.plausible_candidates(clean)
            finite_jacs = latent.finite_shot_candidate_jacobians(clean, pairs, candidates, root_index, cfg["latent_protocol"])
            exact_jacs = latent.exact_candidate_jacobians(clean, pairs, candidates)
            cache[root_index] = {
                "candidates": candidates,
                "finite_jacs": finite_jacs,
                "exact_jacs": exact_jacs,
            }
        root_cache = cache[root_index]
        candidates = list(root_cache["candidates"])
        finite_jacs = root_cache["finite_jacs"]
        exact_jacs = root_cache["exact_jacs"]

        # All deployable candidate features are formed before privileged location is read.
        per_candidate = np.stack(
            [frame.canonicalize_evidence(delta, jac, weights)[0] for jac in finite_jacs]
        ).astype(np.float32)

        true_location = (int(root["affected_qubit"]), int(root["injection_boundary_rank"]))
        if true_location not in candidates:
            raise RuntimeError(f"true location absent from plausible candidates at root {root_index}")
        true_index = int(candidates.index(true_location))
        true_finite = frame.canonicalize_evidence(delta, finite_jacs[true_index], weights)[0].astype(np.float32)
        true_exact = frame.canonicalize_evidence(delta, exact_jacs[true_index], weights)[0].astype(np.float32)

        similarities = np.stack([axis_similarity(jac, exact_jacs[true_index]) for jac in exact_jacs])
        true_qubit = np.asarray([q == true_location[0] for q, _b in candidates], dtype=np.bool_)
        equivalence = np.asarray([is_equivalent(sim, cfg) for sim in similarities], dtype=np.bool_) & true_qubit
        equivalence[true_index] = True
        if not np.any(true_qubit) or not np.any(equivalence):
            raise RuntimeError("privileged audit restriction mask unexpectedly empty")

        geometry = frame.frame_geometry(exact_jacs[true_index])
        raw_features.append(raw)
        finite_candidate_features.append(per_candidate)
        true_finite_features.append(true_finite)
        true_exact_features.append(true_exact)
        truth.append(frame.TARGET[str(row["mechanism"])])
        partitions.append(partition)
        families.append(str(row["family_id"]))
        candidates_all.append(candidates)
        true_locations.append(true_location)
        true_indices.append(true_index)
        true_qubit_masks.append(true_qubit)
        equivalence_masks.append(equivalence)
        similarity_tables.append(similarities.astype(np.float32))
        candidate_counts.append(len(candidates))
        true_frame_min_angles.append(float(geometry["minimum_pairwise_axis_angle_deg"]))

        if progress_every and position % progress_every == 0:
            print(f"ambiguity extraction {position}/{len(selected)} roots_cached={len(cache)}", flush=True)

    candidate_array, candidate_mask = latent.pad_candidate_features(finite_candidate_features)
    max_candidates = candidate_array.shape[1]

    def pad_bool(values: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((len(values), max_candidates), dtype=np.bool_)
        for i, value in enumerate(values):
            out[i, : len(value)] = value
        return out

    similarity_array = np.zeros((len(similarity_tables), max_candidates, 3, 3), dtype=np.float32)
    for i, value in enumerate(similarity_tables):
        similarity_array[i, : value.shape[0]] = value

    y = np.asarray(truth, dtype=np.int64)
    part = np.asarray(partitions, dtype=object)
    fam = np.asarray(families, dtype=object)
    for name, split_mask in (("fit", part == "fit"), ("selection", part == "selection")):
        counts = Counter(int(v) for v in y[split_mask].tolist())
        if set(counts) != {0, 1, 2} or len(set(counts.values())) != 1:
            raise RuntimeError(f"{name} mechanism classes are not balanced: {counts}")

    return {
        "raw": np.stack(raw_features).astype(np.float32),
        "candidate_features": candidate_array,
        "candidate_mask": candidate_mask,
        "true_finite_features": np.stack(true_finite_features).astype(np.float32),
        "true_exact_features": np.stack(true_exact_features).astype(np.float32),
        "truth": y,
        "partition": part,
        "family": fam,
        "candidate_sets": candidates_all,
        "true_locations": true_locations,
        "true_indices": np.asarray(true_indices, dtype=np.int64),
        "true_qubit_mask": pad_bool(true_qubit_masks),
        "equivalence_mask": pad_bool(equivalence_masks),
        "axis_similarity": similarity_array,
        "candidate_count_summary": quantiles(candidate_counts),
        "true_frame_minimum_pairwise_axis_angle_deg": np.asarray(true_frame_min_angles, dtype=np.float64),
    }


def ensemble_fixed_probe(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_sel: np.ndarray,
    y_sel: np.ndarray,
    device,
) -> tuple[np.ndarray, dict[str, Any]]:
    logits: list[np.ndarray] = []
    per_seed: dict[str, Any] = {}
    for seed in frame.PROBE_SEEDS:
        record = oracle.fit_probe(
            x_fit, y_fit, x_sel, y_sel,
            seed=int(seed), high_capacity=False, device=device,
        )
        logits.append(np.asarray(record["selection_logits"], dtype=np.float64))
        per_seed[str(seed)] = frame.public_probe_record(record)
    return np.mean(np.stack(logits), axis=0), per_seed


def predicted_locations(
    pooled_logits: np.ndarray,
    candidate_logits: np.ndarray,
    candidate_mask: np.ndarray,
    candidate_sets: Sequence[Sequence[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    mechanisms = np.argmax(pooled_logits, axis=1)
    indices = np.zeros(len(mechanisms), dtype=np.int64)
    for i, mechanism in enumerate(mechanisms.tolist()):
        valid_count = int(candidate_mask[i].sum())
        indices[i] = int(np.argmax(candidate_logits[i, :valid_count, int(mechanism)]))
    locations = np.asarray([candidate_sets[i][int(indices[i])] for i in range(len(indices))], dtype=np.int64)
    return locations, indices


def main() -> None:
    args = parse_args()
    # Frozen Step-14 reference circuits legally contain up to 17 operations.
    # Reuse the already-audited compatibility fix rather than widening the parser ad hoc.
    oracle_compat.apply_frozen_support_bound()
    cfg = load_config()
    frozen = cfg["source_freeze"]
    if args.training_run_id != str(frozen["training_run_id"]):
        raise RuntimeError("ambiguity run id differs from frozen protocol")
    if args.selection_freeze_sha256 != str(frozen["selection_freeze_sha256"]):
        raise RuntimeError("ambiguity selection-freeze hash differs from frozen protocol")

    rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)
    protocol = step14.read_json(CONFIG)
    step14.assert_contract(protocol)
    latent_cfg = latent.load_frozen_config()
    if baseline.sha256_file(latent.LATENT_CONFIG) != str(frozen["latent_protocol_sha256"]):
        raise RuntimeError("latent protocol hash drift")
    cfg = {**cfg, "latent_protocol": latent_cfg}

    latent_result_path = LATENT_PARENT / str(frozen["latent_diagnostic_id"]) / "diagnostic_result.json"
    local_result_path = LOCAL_FRAME_PARENT / str(frozen["local_frame_diagnostic_id"]) / "diagnostic_result.json"
    latent_source = verify_source(
        latent_result_path,
        str(frozen["latent_result_sha256"]),
        "COMPLETE_FROZEN_ORACLE_FREE_LATENT_FRAME_INFERENCE",
    )
    local_source = verify_source(
        local_result_path,
        str(frozen["local_frame_result_sha256"]),
        "COMPLETE_FROZEN_LOCAL_FRAME_CANONICALIZATION",
    )

    cross_product = step14.resolve_cross_product(None)
    cross_rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(cross_product, protocol)
    complete = json.loads((cross_product / "dataset_complete.json").read_text(encoding="utf-8"))
    if str(complete["product_id"]) != str(frozen["development_product_id"]):
        raise RuntimeError("Step-14 development product drift")
    if baseline.sha256_file(cross_product / "dataset_complete.json") != str(frozen["development_dataset_sha256"]):
        raise RuntimeError("Step-14 development dataset hash drift")
    manifests = cross_product / "manifests"
    roots_rows = baseline.read_csv(manifests / "root_manifest.csv")
    roots = {int(row["root_index"]): row for row in roots_rows}
    if len(roots) != len(roots_rows):
        raise RuntimeError("duplicate root index")

    table = build_table(cross_product, cross_rows, roots, cfg, args.progress_every)
    fit_mask = table["partition"] == "fit"
    sel_mask = table["partition"] == "selection"
    y_fit = table["truth"][fit_mask]
    y_sel = table["truth"][sel_mask]
    groups_sel = table["family"][sel_mask]
    device = oracle.resolve_device(args.device)

    # Reproduce frozen latent MIL. No true location enters this training or full prediction path.
    full_seed_logits: list[np.ndarray] = []
    candidate_seed_logits: list[np.ndarray] = []
    true_qubit_seed_logits: list[np.ndarray] = []
    equiv_seed_logits: list[np.ndarray] = []
    true_boundary_seed_logits: list[np.ndarray] = []
    mil_per_seed: dict[str, Any] = {}
    for seed in [int(v) for v in latent_cfg["latent_mil_probe"]["probe_seeds"]]:
        record = latent.fit_latent_mil(
            table["candidate_features"][fit_mask], table["candidate_mask"][fit_mask], y_fit,
            table["candidate_features"][sel_mask], table["candidate_mask"][sel_mask], y_sel,
            seed=seed, cfg=latent_cfg, device=device,
        )
        full = np.asarray(record.pop("selection_logits"), dtype=np.float64)
        candidates = np.asarray(record.pop("selection_candidate_logits"), dtype=np.float64)
        full_seed_logits.append(full)
        candidate_seed_logits.append(candidates)
        true_qubit_seed_logits.append(stable_logmeanexp_masked(candidates, table["true_qubit_mask"][sel_mask]))
        equiv_seed_logits.append(stable_logmeanexp_masked(candidates, table["equivalence_mask"][sel_mask]))
        true_indices = table["true_indices"][sel_mask]
        true_boundary_seed_logits.append(candidates[np.arange(len(candidates)), true_indices])
        mil_per_seed[str(seed)] = record
        print(
            f"ambiguity latent seed={seed} fit_BA={record['fit_balanced_accuracy']:.4f} "
            f"selection_BA={record['selection_balanced_accuracy']:.4f}", flush=True,
        )

    full_logits = np.mean(np.stack(full_seed_logits), axis=0)
    candidate_logits = np.mean(np.stack(candidate_seed_logits), axis=0)
    true_qubit_logits = np.mean(np.stack(true_qubit_seed_logits), axis=0)
    equiv_logits = np.mean(np.stack(equiv_seed_logits), axis=0)
    true_boundary_logits = np.mean(np.stack(true_boundary_seed_logits), axis=0)

    finite_true_logits, finite_true_per_seed = ensemble_fixed_probe(
        table["true_finite_features"][fit_mask], y_fit,
        table["true_finite_features"][sel_mask], y_sel,
        device,
    )
    exact_true_logits, exact_true_per_seed = ensemble_fixed_probe(
        table["true_exact_features"][fit_mask], y_fit,
        table["true_exact_features"][sel_mask], y_sel,
        device,
    )

    metrics = {
        "oracle_free_full_latent_mil": metric(y_sel, full_logits, "deployable_source_reproduction"),
        "true_qubit_restricted_latent_mil": metric(y_sel, true_qubit_logits, "privileged_post_prediction_counterfactual"),
        "true_frame_equivalence_class_latent_mil": metric(y_sel, equiv_logits, "privileged_post_prediction_counterfactual"),
        "true_boundary_single_candidate_latent_mil": metric(y_sel, true_boundary_logits, "privileged_post_prediction_counterfactual"),
        "true_boundary_finite_shot_small_probe": metric(y_sel, finite_true_logits, "privileged_hardware_facing_reference"),
        "true_boundary_exact_local_frame_small_probe": metric(y_sel, exact_true_logits, "analysis_only_privileged_reference_reproduction"),
    }

    source_latent_ba = float(frozen["latent_primary_ba"])
    reproduced_latent_ba = float(metrics["oracle_free_full_latent_mil"]["mechanism_balanced_accuracy"])
    if abs(reproduced_latent_ba - source_latent_ba) > float(cfg["reproduction_tolerance_ba"]):
        raise RuntimeError(f"latent MIL reproduction drift: {reproduced_latent_ba} vs {source_latent_ba}")
    source_oracle_ba = float(frozen["local_frame_oracle_small_probe_ba"])
    reproduced_oracle_ba = float(metrics["true_boundary_exact_local_frame_small_probe"]["mechanism_balanced_accuracy"])
    if abs(reproduced_oracle_ba - source_oracle_ba) > float(cfg["reproduction_tolerance_ba"]):
        raise RuntimeError(f"local-frame oracle reproduction drift: {reproduced_oracle_ba} vs {source_oracle_ba}")

    comparisons = {
        "true_qubit_over_full": (true_qubit_logits, full_logits),
        "equivalence_class_over_true_qubit": (equiv_logits, true_qubit_logits),
        "true_boundary_over_true_qubit": (true_boundary_logits, true_qubit_logits),
        "finite_true_frame_probe_over_full": (finite_true_logits, full_logits),
        "exact_oracle_over_full": (exact_true_logits, full_logits),
        "exact_oracle_over_finite_true_frame_probe": (exact_true_logits, finite_true_logits),
    }
    paired_deltas: dict[str, Any] = {}
    bootstrap_seed = int(cfg["statistics"]["bootstrap_seed"])
    for offset, (name, (candidate, reference)) in enumerate(comparisons.items()):
        record = oracle.bootstrap_delta(y_sel, candidate, reference, groups_sel, seed=bootstrap_seed + offset)
        paired_deltas[name] = record

    full_pred = np.argmax(full_logits, axis=1)
    oracle_pred = np.argmax(exact_true_logits, axis=1)
    full_correct = full_pred == y_sel
    oracle_correct = oracle_pred == y_sel
    gross_loss = oracle_correct & ~full_correct
    reverse_offset = full_correct & ~oracle_correct

    selection_candidates = [table["candidate_sets"][i] for i in np.flatnonzero(sel_mask)]
    selection_mask = table["candidate_mask"][sel_mask]
    pred_locations, pred_indices = predicted_locations(full_logits, candidate_logits, selection_mask, selection_candidates)
    true_locations = np.asarray([table["true_locations"][i] for i in np.flatnonzero(sel_mask)], dtype=np.int64)
    true_indices = table["true_indices"][sel_mask]
    eq_mask = table["equivalence_mask"][sel_mask]
    similarity = table["axis_similarity"][sel_mask]
    min_angles = table["true_frame_minimum_pairwise_axis_angle_deg"][sel_mask]

    confusion_threshold = float(cfg["mechanism_confusability"]["minimum_wrong_axis_abs_cosine"])
    gross_categories = Counter()
    gross_confusable = Counter()
    wrong_axis_similarity: list[float] = []
    for i in np.flatnonzero(gross_loss):
        q_pred, b_pred = (int(v) for v in pred_locations[i])
        q_true, b_true = (int(v) for v in true_locations[i])
        pred_mech = int(full_pred[i])
        true_mech = int(y_sel[i])
        sim = float(similarity[i, int(pred_indices[i]), pred_mech, true_mech])
        wrong_axis_similarity.append(sim)
        confusable = sim >= confusion_threshold
        if q_pred != q_true:
            category = "wrong_qubit"
        elif b_pred != b_true and bool(eq_mask[i, int(pred_indices[i])]):
            category = "observationally_equivalent_wrong_boundary"
        elif b_pred != b_true:
            category = "non_equivalent_wrong_boundary"
        else:
            category = "exact_location_but_wrong_mechanism"
        gross_categories[category] += 1
        if confusable:
            gross_confusable[category] += 1

    # Location accuracy and equivalence-adjusted accuracy are audit-only.
    exact_location = np.all(pred_locations == true_locations, axis=1)
    qubit_correct = pred_locations[:, 0] == true_locations[:, 0]
    boundary_correct = pred_locations[:, 1] == true_locations[:, 1]
    predicted_equiv = eq_mask[np.arange(len(pred_indices)), pred_indices]

    sensitivity: dict[str, Any] = {}
    for threshold in [float(v) for v in cfg["mechanism_confusability"]["sensitivity_abs_cosines"]]:
        count = 0
        for i in np.flatnonzero(gross_loss):
            pred_mech = int(full_pred[i])
            true_mech = int(y_sel[i])
            sim = float(similarity[i, int(pred_indices[i]), pred_mech, true_mech])
            count += int(sim >= threshold)
        sensitivity[str(threshold)] = {
            "gross_oracle_correct_latent_wrong_count": int(count),
            "fraction_of_gross_loss": float(count / max(int(gross_loss.sum()), 1)),
        }

    gap = reproduced_oracle_ba - reproduced_latent_ba
    selection_n = int(len(y_sel))
    # Classes are exactly balanced, therefore BA == ordinary accuracy here.
    gap_net_examples = int(round(gap * selection_n))
    gross_count = int(gross_loss.sum())
    reverse_count = int(reverse_offset.sum())
    if gross_count - reverse_count != gap_net_examples:
        raise RuntimeError("paired disagreement accounting does not reproduce balanced-accuracy gap")

    decomposition = {
        "source_oracle_minus_latent_ba_gap": float(gap),
        "net_selection_examples_in_gap": gap_net_examples,
        "gross_oracle_correct_latent_wrong": gross_count,
        "reverse_latent_correct_oracle_wrong": reverse_count,
        "gross_loss_location_categories": {key: int(value) for key, value in sorted(gross_categories.items())},
        "gross_loss_mechanism_confusable_by_location_category": {
            key: int(value) for key, value in sorted(gross_confusable.items())
        },
        "gross_loss_wrong_axis_abs_cosine": quantiles(wrong_axis_similarity),
        "counterfactual_ba_increments": {
            "remove_qubit_ambiguity_true_qubit_minus_full": float(
                metrics["true_qubit_restricted_latent_mil"]["mechanism_balanced_accuracy"] - reproduced_latent_ba
            ),
            "restrict_to_equivalent_frame_class_minus_true_qubit": float(
                metrics["true_frame_equivalence_class_latent_mil"]["mechanism_balanced_accuracy"]
                - metrics["true_qubit_restricted_latent_mil"]["mechanism_balanced_accuracy"]
            ),
            "remove_boundary_ambiguity_true_boundary_minus_true_qubit": float(
                metrics["true_boundary_single_candidate_latent_mil"]["mechanism_balanced_accuracy"]
                - metrics["true_qubit_restricted_latent_mil"]["mechanism_balanced_accuracy"]
            ),
            "privileged_finite_true_frame_probe_minus_full": float(
                metrics["true_boundary_finite_shot_small_probe"]["mechanism_balanced_accuracy"] - reproduced_latent_ba
            ),
            "exact_frame_over_finite_calibration": float(
                reproduced_oracle_ba - metrics["true_boundary_finite_shot_small_probe"]["mechanism_balanced_accuracy"]
            ),
        },
    }

    audit = {
        "qubit_accuracy": float(np.mean(qubit_correct)),
        "boundary_accuracy": float(np.mean(boundary_correct)),
        "joint_exact_accuracy": float(np.mean(exact_location)),
        "equivalence_adjusted_frame_accuracy": float(np.mean(predicted_equiv)),
        "wrong_boundary_but_equivalent_fraction_of_selection": float(np.mean((~boundary_correct) & predicted_equiv)),
        "wrong_boundary_but_equivalent_fraction_among_wrong_boundaries": float(
            np.sum((~boundary_correct) & predicted_equiv) / max(int(np.sum(~boundary_correct)), 1)
        ),
        "true_frame_minimum_pairwise_axis_angle_deg": quantiles(min_angles.tolist()),
        "fraction_true_frame_minimum_axis_angle_below_15deg": float(np.mean(min_angles < 15.0)),
        "fraction_true_frame_minimum_axis_angle_below_25deg": float(np.mean(min_angles < 25.0)),
        "mechanism_confusability_sensitivity": sensitivity,
    }

    identity = {
        "schema": SCHEMA,
        "ambiguity_protocol_sha256": baseline.sha256_file(AMBIGUITY_CONFIG),
        "latent_protocol_sha256": baseline.sha256_file(latent.LATENT_CONFIG),
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "development_product_id": str(complete["product_id"]),
        "development_dataset_sha256": baseline.sha256_file(cross_product / "dataset_complete.json"),
        "source_latent_diagnostic_id": str(frozen["latent_diagnostic_id"]),
        "source_latent_result_sha256": str(frozen["latent_result_sha256"]),
        "source_local_frame_diagnostic_id": str(frozen["local_frame_diagnostic_id"]),
        "source_local_frame_result_sha256": str(frozen["local_frame_result_sha256"]),
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_accessed": False,
    }
    diagnostic_id = "frame_ambiguity_" + oracle.stable_hash(identity).split(":", 1)[1][:24]
    out_parent = args.output_parent.expanduser().resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    out_dir = out_parent / diagnostic_id
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite candidate-frame ambiguity diagnostic {out_dir}")
    out_dir.mkdir()

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_CANDIDATE_FRAME_AMBIGUITY_DECOMPOSITION",
        "diagnostic_id": diagnostic_id,
        "identity": identity,
        "counts": {
            "fit_examples": int(fit_mask.sum()),
            "selection_examples": selection_n,
            "candidate_count_per_example": table["candidate_count_summary"],
        },
        "selection_metrics": metrics,
        "paired_family_bootstrap_deltas": paired_deltas,
        "ambiguity_decomposition": decomposition,
        "location_and_geometry_audit": audit,
        "latent_mil_reproduction_per_seed": mil_per_seed,
        "finite_true_frame_probe_per_seed": finite_true_per_seed,
        "exact_true_frame_probe_per_seed": exact_true_per_seed,
        "source_verdicts": {
            "latent": latent_source.get("hypothesis_verdicts", {}),
            "local_frame": local_source.get("hypothesis_verdicts", {}),
        },
        "interpretation_boundaries": {
            "descriptive_post_hoc_decomposition_only": True,
            "does_not_change_frozen_latent_verdict": True,
            "counterfactual_true_qubit_boundary_masks_applied_only_after_fit_and_prediction": True,
            "exact_statevector_geometry_used_only_for_equivalence_confusability_audit": True,
            "equivalence_and_confusability_categories_are_not_forced_to_sum_to_ba_gap": True,
            "main_model_retrained": False,
            "main_model_weights_updated": False,
            "selection_used_for_training_or_early_stopping": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
        },
    }
    atomic_json(out_dir / "diagnostic_result.json", result)
    result_sha = baseline.sha256_file(out_dir / "diagnostic_result.json")
    complete_payload = {
        "schema": SCHEMA,
        "status": result["status"],
        "diagnostic_id": diagnostic_id,
        "diagnostic_result_sha256": result_sha,
        "main_model_weights_updated": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    }
    atomic_json(out_dir / "diagnostic_complete.json", complete_payload)
    complete_sha = baseline.sha256_file(out_dir / "diagnostic_complete.json")
    pointer = {
        "schema": "triqto.v0_2.step14_current_candidate_frame_ambiguity.v1",
        "diagnostic_id": diagnostic_id,
        "diagnostic_dir": str(out_dir),
        "diagnostic_result_sha256": result_sha,
        "diagnostic_complete_sha256": complete_sha,
    }
    atomic_json(out_parent / "current_candidate_frame_ambiguity.json", pointer)
    print(json.dumps({**complete_payload, "diagnostic_complete_sha256": complete_sha}, indent=2), flush=True)


if __name__ == "__main__":
    main()
