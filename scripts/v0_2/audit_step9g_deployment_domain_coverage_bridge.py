#!/usr/bin/env python3
"""Step 9G zero-QPU deployment-domain simulator coverage bridge audit.

This post-hoc audit follows frozen Step 9F. It tests whether circuit-only local
response-frame prediction failed because the Step-9D deployment motif was far
outside the Step-5 simulator support.

No QPU submission and no TriQTO retraining occur here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit

from analyze_step9d_posthoc_context_identifiability import (
    DEFAULT_CONFIG,
    DEFAULT_POINTER,
    DEFAULT_QPU_RESULTS,
    MECHANISMS,
    _ideal_vector,
    _matched_contexts,
    _pilot_phase_exact_frame,
    _resolve_product,
)
from audit_step9e_deployable_local_frame import (
    TARGET_QUBIT,
    TARGET_STRENGTH,
    _circuit_features,
    _context_rows,
    _feature_matrix,
    _fit_rbf_kernel_ridge,
    _frame_alignment,
    _phase_qpu_rows,
    _qpu_reference_circuit_and_context,
    _split_frame,
    _target_matrix,
    _tier3_probe_frame,
)
from audit_step9f_frame_failure_decomposition import (
    _evaluate_frames,
    _nearest_stats,
    _qpu_decode_with_frame,
    _standardized_features,
)


ANALYSIS_NAME = "step9g_deployment_domain_simulator_coverage_bridge_v1"
DEFAULT_OUTPUT = Path("/workspace/triqto-data/step9g_posthoc/deployment_domain_coverage_bridge_v1.json")
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260819
DEFAULT_RIDGE_ALPHA = 1e-2
DEFAULT_BRIDGE_ROOTS = 240
DEFAULT_QUERY_SHOTS = 4096
PRIMARY_DECODER = "euclidean"
MOTIFS = (
    "pilot_core_variant",
    "spectator_pre",
    "spectator_mid",
    "spectator_tail",
)


def _stable_seed(*parts: Any) -> int:
    import hashlib

    payload = json.dumps([str(part) for part in parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _wrap_delta(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _angles(root_index: int) -> tuple[float, float]:
    rng = np.random.default_rng(_stable_seed("step9g", "bridge_angles", root_index))
    phi = float(rng.uniform(-math.pi, math.pi))
    beta = float(rng.uniform(-1.2, 1.2))
    # Explicitly exclude the exact frozen Step-9D pilot angle.
    if abs(_wrap_delta(phi, 0.7)) < 0.01:
        phi = float(phi + 0.05)
        if phi > math.pi:
            phi -= 2.0 * math.pi
    return phi, beta


def _build_bridge_circuit(root_index: int) -> tuple[QuantumCircuit, int, str, float, float]:
    motif = MOTIFS[root_index % len(MOTIFS)]
    phi, beta = _angles(root_index)
    qc = QuantumCircuit(2, name=f"step9g_{motif}_{root_index}")

    if motif == "pilot_core_variant":
        qc.h(0)
        qc.rz(phi, 0)
        boundary = 2
        qc.h(0)
        qc.cx(0, 1)
    elif motif == "spectator_pre":
        qc.ry(beta, 1)
        qc.h(0)
        qc.rz(phi, 0)
        boundary = 3
        qc.h(0)
        qc.cx(0, 1)
    elif motif == "spectator_mid":
        qc.h(0)
        qc.rz(phi, 0)
        boundary = 2
        qc.rz(beta, 1)
        qc.h(0)
        qc.cx(0, 1)
    elif motif == "spectator_tail":
        qc.h(0)
        qc.rz(phi, 0)
        boundary = 2
        qc.h(0)
        qc.ry(beta, 1)
        qc.cx(0, 1)
    else:
        raise RuntimeError(f"unknown bridge motif {motif!r}")

    return qc, boundary, motif, phi, beta


def _exact_frame(circuit: QuantumCircuit, boundary: int) -> dict[str, np.ndarray]:
    from audit_step9e_deployable_local_frame import _inject_rotation

    frame: dict[str, np.ndarray] = {}
    for mechanism in MECHANISMS:
        observed = _inject_rotation(
            circuit,
            boundary,
            TARGET_QUBIT,
            mechanism,
            TARGET_STRENGTH,
        )
        frame[mechanism] = _ideal_vector(circuit, observed)
    return frame


def _bridge_contexts(root_count: int, query_shots: int) -> list[dict[str, Any]]:
    if root_count <= 0 or root_count % 20 != 0:
        raise ValueError("--bridge-roots must be positive and divisible by 20")
    contexts: list[dict[str, Any]] = []
    motif_counts = {motif: {"train": 0, "validation": 0} for motif in MOTIFS}

    qpu_circuit, qpu_boundary, qpu_qubit = _qpu_reference_circuit_and_context(DEFAULT_CONFIG)
    qpu_feature = _circuit_features(qpu_circuit, qpu_boundary, qpu_qubit)

    for root_index in range(root_count):
        circuit, boundary, motif, phi, beta = _build_bridge_circuit(root_index)
        if abs(_wrap_delta(phi, 0.7)) < 0.009999:
            raise RuntimeError("bridge accidentally contains excluded pilot angle neighborhood")
        feature = _circuit_features(circuit, boundary, TARGET_QUBIT)
        if feature.shape == qpu_feature.shape and float(np.linalg.norm(feature - qpu_feature)) <= 1e-12:
            raise RuntimeError("exact frozen Step-9D pilot circuit leaked into bridge")

        split = "validation" if root_index % 5 == 0 else "train"
        motif_counts[motif][split] += 1
        exact = _exact_frame(circuit, boundary)
        finite = _tier3_probe_frame(
            circuit,
            boundary,
            TARGET_QUBIT,
            probe_strength=TARGET_STRENGTH,
            target_strength=TARGET_STRENGTH,
            shots=query_shots,
            seed_parts=["step9g", "hidden_query", root_index, motif],
        )
        contexts.append(
            {
                "root_index": int(root_index),
                "boundary": int(boundary),
                "split": split,
                "motif": motif,
                "phi": float(phi),
                "beta": float(beta),
                "circuit": circuit,
                "exact": exact,
                "finite": finite,
                "shots": {mechanism: int(query_shots) for mechanism in MECHANISMS},
            }
        )

    expected_per_motif = root_count // len(MOTIFS)
    expected_validation_per_motif = root_count // 20
    for motif in MOTIFS:
        if sum(motif_counts[motif].values()) != expected_per_motif:
            raise RuntimeError(f"bridge motif imbalance for {motif}: {motif_counts[motif]}")
        if motif_counts[motif]["validation"] != expected_validation_per_motif:
            raise RuntimeError(f"bridge validation imbalance for {motif}: {motif_counts[motif]}")
    return contexts


def _support_eval(
    model: Any,
    x_validation: np.ndarray,
    qpu_raw: np.ndarray,
) -> dict[str, Any]:
    validation_z = _standardized_features(model, x_validation)
    qpu_z = _standardized_features(model, qpu_raw)
    val_distance, val_similarity = _nearest_stats(model.x_train, validation_z, model.gamma)
    qpu_distance, qpu_similarity = _nearest_stats(model.x_train, qpu_z, model.gamma)
    qd = float(qpu_distance[0])
    qs = float(qpu_similarity[0])
    vmax = float(np.max(val_distance))
    smin = float(np.min(val_similarity))
    return {
        "validation_nearest_distance": {
            "min": float(np.min(val_distance)),
            "median": float(np.median(val_distance)),
            "p90": float(np.percentile(val_distance, 90.0)),
            "max": vmax,
        },
        "validation_max_kernel_similarity": {
            "min": smin,
            "median": float(np.median(val_similarity)),
            "p10": float(np.percentile(val_similarity, 10.0)),
            "max": float(np.max(val_similarity)),
        },
        "qpu_nearest_distance": qd,
        "qpu_max_kernel_similarity": qs,
        "qpu_distance_percentile_vs_bridge_validation": float(100.0 * np.mean(val_distance <= qd)),
        "qpu_similarity_percentile_vs_bridge_validation": float(100.0 * np.mean(val_similarity <= qs)),
        "inside_bridge_support_envelope": bool(qd <= vmax and qs >= smin),
    }


def _train_and_evaluate(
    name: str,
    train_contexts: Sequence[Mapping[str, Any]],
    bridge_validation: Sequence[Mapping[str, Any]],
    step5_validation: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    bootstraps: int,
    seed: int,
    config_path: Path,
    qpu_results_path: Path,
) -> dict[str, Any]:
    x_train = _feature_matrix(train_contexts, "tier1_circuit_only")
    y_train = _target_matrix(train_contexts)
    x_bridge_validation = _feature_matrix(bridge_validation, "tier1_circuit_only")
    x_step5_validation = _feature_matrix(step5_validation, "tier1_circuit_only")
    model = _fit_rbf_kernel_ridge(x_train, y_train, alpha)

    bridge_frames = [_split_frame(row) for row in model.predict(x_bridge_validation)]
    step5_frames = [_split_frame(row) for row in model.predict(x_step5_validation)]
    bridge_eval = _evaluate_frames(
        bridge_validation,
        bridge_frames,
        bootstraps=bootstraps,
        seed=seed,
    )
    step5_eval = _evaluate_frames(
        step5_validation,
        step5_frames,
        bootstraps=bootstraps,
        seed=seed + 100,
    )

    _, qpu_rows = _phase_qpu_rows(qpu_results_path)
    qpu_circuit, qpu_boundary, qpu_qubit = _qpu_reference_circuit_and_context(config_path)
    qpu_raw = _circuit_features(qpu_circuit, qpu_boundary, qpu_qubit)[None, :]
    qpu_predicted = _split_frame(model.predict(qpu_raw)[0])
    qpu_exact = _pilot_phase_exact_frame(config_path)
    qpu = {
        "predicted_frame_alignment_to_exact_simulator_frame": _frame_alignment(qpu_predicted, qpu_exact),
        "decoders": _qpu_decode_with_frame(qpu_rows, qpu_predicted),
    }
    support = _support_eval(model, x_bridge_validation, qpu_raw)

    primary_gate = bool(bridge_eval["reference_gates"][PRIMARY_DECODER]["passed"])
    qpu_all_three = bool(qpu["decoders"][PRIMARY_DECODER]["all_three_correct"])
    support_ok = bool(support["inside_bridge_support_envelope"])
    full_rescue = bool(primary_gate and qpu_all_three and support_ok)

    return {
        "name": name,
        "train_context_count": len(train_contexts),
        "feature_dimension": int(x_train.shape[1]),
        "ridge_alpha": float(alpha),
        "rbf_gamma": float(model.gamma),
        "bridge_validation": bridge_eval,
        "original_step5_validation": step5_eval,
        "frozen_step9d_qpu": qpu,
        "coverage": support,
        "coverage_rescue_components": {
            "bridge_validation_primary_gate_pass": primary_gate,
            "frozen_qpu_primary_decoder_all_three_correct": qpu_all_three,
            "qpu_inside_bridge_support_envelope": support_ok,
            "full_targeted_coverage_rescue": full_rescue,
        },
    }


def _decision(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    combined = bool(results["step5_plus_bridge"]["coverage_rescue_components"]["full_targeted_coverage_rescue"])
    bridge_only = bool(results["bridge_only"]["coverage_rescue_components"]["full_targeted_coverage_rescue"])
    if combined:
        return {
            "status": "DEPLOYMENT_DOMAIN_COVERAGE_RESCUES_CIRCUIT_ONLY_FRAME_IN_TARGETED_AUDIT",
            "leading_diagnosis": "missing deployment-domain simulator coverage",
            "preferred_support": "step5_plus_bridge",
            "next_action": (
                "Freeze this result. Then redesign the simulator training distribution to include a leakage-safe deployment-domain bridge and compare warm-start checkpoint reuse versus scratch only after the input contract is fixed. Do not assume an architecture replacement is required."
            ),
        }
    if bridge_only:
        return {
            "status": "BRIDGE_DOMAIN_IS_LEARNABLE_BUT_MIXED_SUPPORT_DOES_NOT_FULLY_RESCUE",
            "leading_diagnosis": "domain-conditioned or mixture representation may be required",
            "preferred_support": "bridge_only",
            "next_action": (
                "Do not retrain TriQTO yet. Audit a domain-conditioned/mixture-aware frame representation before changing the main model."
            ),
        }
    return {
        "status": "DEPLOYMENT_DOMAIN_COVERAGE_BRIDGE_DOES_NOT_RESCUE_CIRCUIT_ONLY_FRAME",
        "leading_diagnosis": "tested circuit-only representation remains insufficient",
        "preferred_support": None,
        "next_action": (
            "Do not retrain TriQTO yet. The Step-9F SNR-qualified known-axis probe is now the evidence-backed hardware-valid fallback; freeze a bounded QPU probe protocol before any submission."
        ),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qpu-case-results", type=Path, default=DEFAULT_QPU_RESULTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bridge-roots", type=int, default=DEFAULT_BRIDGE_ROOTS)
    parser.add_argument("--query-shots", type=int, default=DEFAULT_QUERY_SHOTS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    args = parser.parse_args()

    if args.query_shots <= 0 or args.bootstraps <= 0 or args.ridge_alpha <= 0:
        raise ValueError("query shots, bootstraps, and ridge alpha must be positive")

    product = _resolve_product(args.product_pointer, args.product_dir)
    step5_all = _context_rows(product, _matched_contexts(product))
    step5_train = [row for row in step5_all if row["split"] == "train"]
    step5_validation = [row for row in step5_all if row["split"] == "validation"]

    bridge_all = _bridge_contexts(args.bridge_roots, args.query_shots)
    bridge_train = [row for row in bridge_all if row["split"] == "train"]
    bridge_validation = [row for row in bridge_all if row["split"] == "validation"]

    training_sets = {
        "step5_only": step5_train,
        "bridge_only": bridge_train,
        "step5_plus_bridge": [*step5_train, *bridge_train],
    }
    results: dict[str, Any] = {}
    for index, (name, train_contexts) in enumerate(training_sets.items()):
        results[name] = _train_and_evaluate(
            name,
            train_contexts,
            bridge_validation,
            step5_validation,
            alpha=args.ridge_alpha,
            bootstraps=args.bootstraps,
            seed=args.bootstrap_seed + 1000 * index,
            config_path=args.config,
            qpu_results_path=args.qpu_case_results,
        )

    decision = _decision(results)
    motif_counts = {
        motif: {
            "train": int(sum(row["motif"] == motif and row["split"] == "train" for row in bridge_all)),
            "validation": int(sum(row["motif"] == motif and row["split"] == "validation" for row in bridge_all)),
        }
        for motif in MOTIFS
    }
    output = {
        "analysis": ANALYSIS_NAME,
        "scientific_boundary": {
            "posthoc_only": True,
            "qpu_submission": False,
            "triqto_checkpoint_retraining": False,
            "triqto_weight_change": False,
            "triqto_threshold_change": False,
            "exact_simulator_frames_are_audit_targets_only": True,
        },
        "step5_product_id": _resolve_product(args.product_pointer, args.product_dir).name,
        "frozen_design": {
            "bridge_roots": int(args.bridge_roots),
            "bridge_motifs": list(MOTIFS),
            "bridge_split_rule": "root_index_mod_5 == 0 -> validation",
            "exact_pilot_angle_0p7_excluded": True,
            "query_strength": TARGET_STRENGTH,
            "query_shots_per_basis": int(args.query_shots),
            "primary_decoder": PRIMARY_DECODER,
            "ridge_alpha": float(args.ridge_alpha),
            "bootstraps": int(args.bootstraps),
        },
        "context_counts": {
            "step5_train": len(step5_train),
            "step5_validation": len(step5_validation),
            "bridge_train": len(bridge_train),
            "bridge_validation": len(bridge_validation),
            "bridge_by_motif": motif_counts,
        },
        "training_support_results": results,
        "decision_gate": decision,
    }

    print("STEP 9G DEPLOYMENT-DOMAIN COVERAGE BRIDGE — NO QPU / NO TRIQTO RETRAINING")
    print("Step-5 product:", product.name)
    print(
        "contexts: step5 train/val=",
        len(step5_train),
        "/",
        len(step5_validation),
        "bridge train/val=",
        len(bridge_train),
        "/",
        len(bridge_validation),
    )
    print("bridge motifs:", motif_counts)

    for name in ("step5_only", "bridge_only", "step5_plus_bridge"):
        row = results[name]
        bridge = row["bridge_validation"]
        gate = bridge["reference_gates"][PRIMARY_DECODER]
        decoder = bridge["decoders"][PRIMARY_DECODER]
        ci = decoder["balanced_accuracy_cluster_bootstrap_95_ci"]
        qpu = row["frozen_step9d_qpu"]["decoders"][PRIMARY_DECODER]
        coverage = row["coverage"]
        qalign = row["frozen_step9d_qpu"]["predicted_frame_alignment_to_exact_simulator_frame"]
        print(f"\n{name.upper()}")
        print(
            "  bridge euclidean BA=",
            _fmt(decoder["balanced_accuracy"]),
            "CI=[",
            _fmt(ci[0]),
            ",",
            _fmt(ci[1]),
            "] min-recall=",
            _fmt(gate["minimum_mechanism_recall"]),
            "gate=",
            "PASS" if gate["passed"] else "FAIL",
            sep="",
        )
        print(
            "  QPU euclidean=",
            f"{qpu['correct_count']}/3",
            " predicted-frame median cosine=",
            _fmt(qalign["median_cosine"]),
        )
        print(
            "  QPU coverage distance=",
            _fmt(coverage["qpu_nearest_distance"]),
            " val-max=",
            _fmt(coverage["validation_nearest_distance"]["max"]),
            " similarity=",
            _fmt(coverage["qpu_max_kernel_similarity"]),
            " val-min=",
            _fmt(coverage["validation_max_kernel_similarity"]["min"]),
            " inside=",
            coverage["inside_bridge_support_envelope"],
        )
        print(
            "  full targeted coverage rescue=",
            row["coverage_rescue_components"]["full_targeted_coverage_rescue"],
        )

    print("\nDECISION GATE:", decision["status"])
    print("  leading diagnosis:", decision["leading_diagnosis"])
    print("  next:", decision["next_action"])

    path = args.output_json.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("\nwrote:", path)


if __name__ == "__main__":
    main()
