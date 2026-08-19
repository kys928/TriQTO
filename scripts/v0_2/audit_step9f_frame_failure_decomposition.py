#!/usr/bin/env python3
"""Step 9F zero-QPU frame-failure decomposition.

This post-hoc audit follows the frozen negative Step-9E result. It does not
retrain TriQTO and does not submit any QPU jobs.

It separates three questions:

A. Probe linearity: do exact low-strength RZ/RX/RY response directions match the
   exact 0.15 target frame before shot noise is introduced?
B. Probe SNR: how does a hardware-compatible finite-shot known-axis probe frame
   behave over a frozen strength x shot ladder?
C. Circuit estimator coverage: is Tier-1 failure mostly an RBF estimator issue,
   or do nearest train contexts in the same deployable feature metric also fail?

Exact simulator response frames are used only as audit targets/reference truth.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analyze_step9d_posthoc_context_identifiability import (
    DEFAULT_CONFIG,
    DEFAULT_POINTER,
    DEFAULT_QPU_RESULTS,
    MECHANISMS,
    _ideal_vector,
    _matched_contexts,
    _pilot_phase_exact_frame,
    _qpu_vector,
    _resolve_product,
    _score_frame,
)
from audit_step9e_deployable_local_frame import (
    TARGET_QUBIT,
    TARGET_STRENGTH,
    _aggregate_alignment,
    _circuit_features,
    _context_rows,
    _decoder_records,
    _feature_matrix,
    _fit_rbf_kernel_ridge,
    _frame_alignment,
    _gate,
    _inject_rotation,
    _phase_qpu_rows,
    _predicted_frames,
    _qpu_reference_circuit_and_context,
    _qpu_tier_features,
    _split_frame,
    _summarize_decoder,
    _target_matrix,
    _tier3_probe_frame,
)


ANALYSIS_NAME = "step9f_zero_qpu_frame_failure_decomposition_v1"
DEFAULT_OUTPUT = Path("/workspace/triqto-data/step9f_posthoc/frame_failure_decomposition_v1.json")
DEFAULT_BOOTSTRAPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260819
DEFAULT_REPEATS = 24
DEFAULT_RIDGE_ALPHA = 1e-2
PROBE_STRENGTHS = (0.025, 0.05, 0.10, 0.15)
PROBE_SHOTS = (512, 1024, 2048, 4096, 8192)
REFERENCE_BA = 0.80
REFERENCE_CI_LOWER = 0.75
REFERENCE_MIN_RECALL = 0.70


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _exact_probe_frame(context: Mapping[str, Any], strength: float) -> dict[str, np.ndarray]:
    clean = context["circuit"]
    boundary = int(context["boundary"])
    scale = float(TARGET_STRENGTH / strength)
    frame: dict[str, np.ndarray] = {}
    for mechanism in MECHANISMS:
        observed = _inject_rotation(clean, boundary, TARGET_QUBIT, mechanism, strength)
        frame[mechanism] = scale * _ideal_vector(clean, observed)
    return frame


def _evaluate_exact_linearity(
    validation: Sequence[Mapping[str, Any]],
    *,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strength_index, strength in enumerate(PROBE_STRENGTHS):
        frames = [_exact_probe_frame(context, strength) for context in validation]
        alignments = [
            _frame_alignment(frame, context["exact"])
            for frame, context in zip(frames, validation, strict=True)
        ]
        decoders: dict[str, Any] = {}
        gates: dict[str, Any] = {}
        for metric_index, metric in enumerate(("cosine", "euclidean")):
            records = _decoder_records(validation, frames, metric)
            summary = _summarize_decoder(
                records,
                bootstraps,
                seed + 100 * strength_index + metric_index,
            )
            decoders[metric] = summary
            gates[metric] = _gate(summary)
        result[f"{strength:.3f}"] = {
            "probe_strength": float(strength),
            "scale_to_target": float(TARGET_STRENGTH / strength),
            "frame_alignment_to_exact_0p15": _aggregate_alignment(alignments),
            "decoders_using_frozen_step5_queries": decoders,
            "reference_gates": gates,
        }
    return result


def _repeated_probe_records(
    validation: Sequence[Mapping[str, Any]],
    *,
    strength: float,
    shots: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alignments: list[dict[str, Any]] = []
    cosine_records: list[dict[str, Any]] = []
    euclidean_records: list[dict[str, Any]] = []
    for context_index, context in enumerate(validation):
        for repeat in range(repeats):
            frame = _tier3_probe_frame(
                context["circuit"],
                int(context["boundary"]),
                TARGET_QUBIT,
                probe_strength=strength,
                target_strength=TARGET_STRENGTH,
                shots=shots,
                seed_parts=[
                    "step9f",
                    "snr",
                    context["root_index"],
                    context["boundary"],
                    strength,
                    shots,
                    repeat,
                ],
            )
            alignment = _frame_alignment(frame, context["exact"])
            alignment["context_index"] = context_index
            alignment["repeat"] = repeat
            alignments.append(alignment)
            for metric, sink in (("cosine", cosine_records), ("euclidean", euclidean_records)):
                for mechanism in MECHANISMS:
                    scores, prediction, margin = _score_frame(
                        np.asarray(context["finite"][mechanism], dtype=np.float64),
                        frame,
                        metric,
                    )
                    sink.append(
                        {
                            "context_index": context_index,
                            "repeat": repeat,
                            "root_index": int(context["root_index"]),
                            "boundary": int(context["boundary"]),
                            "true": mechanism,
                            "prediction": prediction,
                            "correct": bool(prediction == mechanism),
                            "scores": scores,
                            "margin": margin,
                        }
                    )
    return alignments, cosine_records, euclidean_records


def _evaluate_probe_snr(
    validation: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    counter = 0
    for strength in PROBE_STRENGTHS:
        for shots in PROBE_SHOTS:
            alignments, cosine_records, euclidean_records = _repeated_probe_records(
                validation,
                strength=strength,
                shots=shots,
                repeats=repeats,
            )
            decoders: dict[str, Any] = {}
            gates: dict[str, Any] = {}
            for metric_index, (metric, records) in enumerate(
                (("cosine", cosine_records), ("euclidean", euclidean_records))
            ):
                summary = _summarize_decoder(
                    records,
                    bootstraps,
                    seed + 1000 * counter + metric_index,
                )
                decoders[metric] = summary
                gates[metric] = _gate(summary)
            key = f"strength_{strength:.3f}__shots_{shots}"
            cells[key] = {
                "probe_strength": float(strength),
                "shots_per_basis_per_axis": int(shots),
                "repeats_per_context": int(repeats),
                "frame_alignment_to_exact_0p15": _aggregate_alignment(alignments),
                "decoders": decoders,
                "reference_gates": gates,
            }
            counter += 1
    return cells


def _standardized_features(model: Any, raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    return (values - model.mean) / model.scale


def _nearest_stats(train_z: np.ndarray, query_z: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    d2 = np.sum((query_z[:, None, :] - train_z[None, :, :]) ** 2, axis=2)
    nearest = np.sqrt(np.min(d2, axis=1))
    max_similarity = np.max(np.exp(-float(gamma) * d2), axis=1)
    return nearest, max_similarity


def _nn_predict(
    train_z: np.ndarray,
    train_targets: np.ndarray,
    query_z: np.ndarray,
    *,
    k: int,
    gamma: float,
) -> np.ndarray:
    d2 = np.sum((query_z[:, None, :] - train_z[None, :, :]) ** 2, axis=2)
    order = np.argsort(d2, axis=1)[:, :k]
    rows: list[np.ndarray] = []
    for i in range(query_z.shape[0]):
        idx = order[i]
        if k == 1:
            rows.append(np.asarray(train_targets[idx[0]], dtype=np.float64))
            continue
        weights = np.exp(-float(gamma) * d2[i, idx])
        total = float(np.sum(weights))
        if total <= 1e-14:
            weights = np.ones_like(weights) / float(len(weights))
        else:
            weights = weights / total
        rows.append(np.sum(train_targets[idx] * weights[:, None], axis=0))
    return np.vstack(rows)


def _evaluate_frames(
    contexts: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, np.ndarray]],
    *,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    alignments = [
        _frame_alignment(frame, context["exact"])
        for frame, context in zip(frames, contexts, strict=True)
    ]
    decoders: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for metric_index, metric in enumerate(("cosine", "euclidean")):
        records = _decoder_records(contexts, frames, metric)
        summary = _summarize_decoder(records, bootstraps, seed + metric_index)
        decoders[metric] = summary
        gates[metric] = _gate(summary)
    return {
        "frame_alignment_to_exact": _aggregate_alignment(alignments),
        "decoders": decoders,
        "reference_gates": gates,
        "per_context_median_frame_cosine": [
            alignment["median_cosine"] for alignment in alignments
        ],
    }


def _qpu_decode_with_frame(
    qpu_rows: Mapping[str, Mapping[str, Any]],
    frame: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("cosine", "euclidean"):
        rows: list[dict[str, Any]] = []
        for mechanism in MECHANISMS:
            query = _qpu_vector(qpu_rows[mechanism])
            scores, prediction, margin = _score_frame(query, frame, metric)
            rows.append(
                {
                    "expected": mechanism,
                    "prediction": prediction,
                    "correct": bool(prediction == mechanism),
                    "margin": margin,
                    "scores": scores,
                }
            )
        result[metric] = {
            "correct_count": int(sum(bool(row["correct"]) for row in rows)),
            "case_count": len(rows),
            "all_three_correct": bool(all(bool(row["correct"]) for row in rows)),
            "rows": rows,
        }
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _evaluate_estimator_coverage(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    bootstraps: int,
    seed: int,
    config_path: Path,
    qpu_results_path: Path,
) -> dict[str, Any]:
    x_train = _feature_matrix(train, "tier1_circuit_only")
    y_train = _target_matrix(train)
    x_validation = _feature_matrix(validation, "tier1_circuit_only")
    model = _fit_rbf_kernel_ridge(x_train, y_train, alpha)

    train_z = np.asarray(model.x_train, dtype=np.float64)
    validation_z = _standardized_features(model, x_validation)
    val_nearest, val_similarity = _nearest_stats(train_z, validation_z, model.gamma)

    rbf_frames = _predicted_frames(model, x_validation)
    nn1_frames = [
        _split_frame(row)
        for row in _nn_predict(
            train_z, y_train, validation_z, k=1, gamma=model.gamma
        )
    ]
    nn5_frames = [
        _split_frame(row)
        for row in _nn_predict(
            train_z, y_train, validation_z, k=5, gamma=model.gamma
        )
    ]

    estimator_results = {
        "rbf_kernel_ridge": _evaluate_frames(
            validation, rbf_frames, bootstraps=bootstraps, seed=seed
        ),
        "one_nearest_neighbor": _evaluate_frames(
            validation, nn1_frames, bootstraps=bootstraps, seed=seed + 100
        ),
        "five_neighbor_kernel_average": _evaluate_frames(
            validation, nn5_frames, bootstraps=bootstraps, seed=seed + 200
        ),
    }

    clean_qpu, qpu_rows = _phase_qpu_rows(qpu_results_path)
    qpu_circuit, qpu_boundary, qpu_qubit = _qpu_reference_circuit_and_context(config_path)
    qpu_raw = _qpu_tier_features(
        "tier1_circuit_only",
        qpu_circuit,
        qpu_boundary,
        qpu_qubit,
        clean_qpu,
    )[None, :]
    qpu_z = _standardized_features(model, qpu_raw)
    qpu_nearest, qpu_similarity = _nearest_stats(train_z, qpu_z, model.gamma)

    qpu_frames = {
        "rbf_kernel_ridge": _split_frame(model.predict(qpu_raw)[0]),
        "one_nearest_neighbor": _split_frame(
            _nn_predict(train_z, y_train, qpu_z, k=1, gamma=model.gamma)[0]
        ),
        "five_neighbor_kernel_average": _split_frame(
            _nn_predict(train_z, y_train, qpu_z, k=5, gamma=model.gamma)[0]
        ),
    }
    exact_qpu = _pilot_phase_exact_frame(config_path)
    qpu_results = {
        name: {
            "frame_alignment_to_exact_simulator_frame": _frame_alignment(frame, exact_qpu),
            "decoders": _qpu_decode_with_frame(qpu_rows, frame),
        }
        for name, frame in qpu_frames.items()
    }

    qpu_distance = float(qpu_nearest[0])
    qpu_similarity_value = float(qpu_similarity[0])
    distance_percentile = float(100.0 * np.mean(val_nearest <= qpu_distance))
    similarity_percentile = float(100.0 * np.mean(val_similarity <= qpu_similarity_value))

    coverage = {
        "standardized_nearest_train_distance": {
            "validation_min": float(np.min(val_nearest)),
            "validation_median": float(np.median(val_nearest)),
            "validation_p90": float(np.percentile(val_nearest, 90.0)),
            "validation_max": float(np.max(val_nearest)),
            "qpu_context": qpu_distance,
            "qpu_percentile_vs_validation": distance_percentile,
        },
        "max_rbf_similarity_to_train": {
            "validation_min": float(np.min(val_similarity)),
            "validation_median": float(np.median(val_similarity)),
            "validation_p10": float(np.percentile(val_similarity, 10.0)),
            "validation_max": float(np.max(val_similarity)),
            "qpu_context": qpu_similarity_value,
            "qpu_percentile_vs_validation": similarity_percentile,
        },
    }

    correlations: dict[str, Any] = {}
    for name, result in estimator_results.items():
        cosines = result.pop("per_context_median_frame_cosine")
        correlations[name] = {
            "pearson_nearest_distance_vs_median_frame_cosine": _pearson(
                val_nearest.tolist(),
                [float(x) if x is not None else float("nan") for x in cosines],
            )
        }

    return {
        "feature_dimension": int(x_train.shape[1]),
        "train_context_count": len(train),
        "validation_context_count": len(validation),
        "rbf_gamma": float(model.gamma),
        "ridge_alpha": float(alpha),
        "coverage": coverage,
        "validation_estimators": estimator_results,
        "coverage_correlations": correlations,
        "frozen_qpu_context": {
            "candidate_boundary": int(qpu_boundary),
            "candidate_qubit": int(qpu_qubit),
            "estimators": qpu_results,
        },
    }


def _best_snr_cells(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("cosine", "euclidean"):
        ranked = sorted(
            cells.items(),
            key=lambda item: float(item[1]["decoders"][metric]["balanced_accuracy"] or -1.0),
            reverse=True,
        )
        key, cell = ranked[0]
        passing = [
            (name, row)
            for name, row in cells.items()
            if bool(row["reference_gates"][metric]["passed"])
        ]
        cheapest = None
        if passing:
            cheapest_name, cheapest_row = min(
                passing,
                key=lambda item: (
                    int(item[1]["shots_per_basis_per_axis"]),
                    float(item[1]["probe_strength"]),
                ),
            )
            cheapest = {
                "cell": cheapest_name,
                "probe_strength": cheapest_row["probe_strength"],
                "shots": cheapest_row["shots_per_basis_per_axis"],
                "balanced_accuracy": cheapest_row["decoders"][metric]["balanced_accuracy"],
            }
        output[metric] = {
            "best_cell": key,
            "best_balanced_accuracy": cell["decoders"][metric]["balanced_accuracy"],
            "best_ci": cell["decoders"][metric]["balanced_accuracy_cluster_bootstrap_95_ci"],
            "cheapest_cell_passing_reference_gate": cheapest,
        }
    return output


def _diagnostic_interpretation(
    linearity: Mapping[str, Mapping[str, Any]],
    snr: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    exact_005 = linearity["0.050"]["frame_alignment_to_exact_0p15"]["median_cosine_all"]
    exact_010 = linearity["0.100"]["frame_alignment_to_exact_0p15"]["median_cosine_all"]
    snr_best = _best_snr_cells(snr)
    qpu_percentile = coverage["coverage"]["standardized_nearest_train_distance"][
        "qpu_percentile_vs_validation"
    ]

    if exact_005 is not None and float(exact_005) >= 0.90:
        probe_shape = "LOW_STRENGTH_DIRECTION_IS_APPROXIMATELY_LINEAR"
    elif exact_010 is not None and float(exact_010) >= 0.90:
        probe_shape = "VERY_SMALL_PROBE_IS_NONLINEAR_BUT_0P10_IS_LOCALLY_ALIGNED"
    else:
        probe_shape = "LOW_STRENGTH_RESPONSE_DIRECTION_ROTATES_MATERIALLY"

    if any(
        snr_best[metric]["cheapest_cell_passing_reference_gate"] is not None
        for metric in ("cosine", "euclidean")
    ):
        probe_snr = "SIMULATED_PROBE_SNR_RECOVERS_AT_SOME_FROZEN_BUDGET"
    else:
        probe_snr = "NO_STRENGTH_SHOT_CELL_REACHES_REFERENCE_GATE"

    if float(qpu_percentile) >= 95.0:
        domain = "STEP9D_QPU_CONTEXT_IS_EXTREME_IN_TIER1_FEATURE_DISTANCE"
    else:
        domain = "STEP9D_QPU_CONTEXT_NOT_AN_EXTREME_DISTANCE_OUTLIER"

    validation = coverage["validation_estimators"]
    rbf_ba = float(validation["rbf_kernel_ridge"]["decoders"]["cosine"]["balanced_accuracy"])
    nn_best = max(
        float(validation[name]["decoders"]["cosine"]["balanced_accuracy"])
        for name in ("one_nearest_neighbor", "five_neighbor_kernel_average")
    )
    estimator = (
        "NEIGHBOR_TRANSFER_MATERIALLY_OUTPERFORMS_RBF"
        if nn_best >= rbf_ba + 0.10
        else "NEIGHBOR_TRANSFER_DOES_NOT_MATERIALLY_RESCUE_TIER1"
    )

    return {
        "probe_linearity": probe_shape,
        "probe_snr": probe_snr,
        "tier1_domain_coverage": domain,
        "tier1_estimator_specificity": estimator,
        "note": (
            "These are post-hoc diagnostic labels, not confirmatory claims and not authorization "
            "to retrain TriQTO or submit a new QPU job."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qpu-case-results", type=Path, default=DEFAULT_QPU_RESULTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.bootstraps <= 0:
        raise ValueError("--bootstraps must be positive")
    if args.ridge_alpha <= 0:
        raise ValueError("--ridge-alpha must be positive")

    product = _resolve_product(args.product_pointer, args.product_dir)
    complete = json.loads((product / "dataset_complete.json").read_text(encoding="utf-8"))
    contexts = _context_rows(product, _matched_contexts(product))
    train = [row for row in contexts if row["split"] == "train"]
    validation = [row for row in contexts if row["split"] == "validation"]
    if not train or not validation:
        raise RuntimeError("Step 9F requires frozen Step-5 train and validation contexts")

    print("STEP 9F ZERO-QPU FRAME-FAILURE DECOMPOSITION — NO QPU / NO TRIQTO RETRAINING")
    print("Step-5 product:", complete.get("product_id"))
    print("contexts:", len(contexts), "train:", len(train), "validation:", len(validation))

    print("\nA. EXACT PROBE LINEARITY")
    linearity = _evaluate_exact_linearity(
        validation,
        bootstraps=args.bootstraps,
        seed=args.bootstrap_seed,
    )
    for key, row in linearity.items():
        align = row["frame_alignment_to_exact_0p15"]
        cosine = row["decoders_using_frozen_step5_queries"]["cosine"]
        euclidean = row["decoders_using_frozen_step5_queries"]["euclidean"]
        print(
            f"  strength={key}: frame median cos={_fmt(align['median_cosine_all'])} "
            f"p10={_fmt(align['p10_cosine_all'])}; "
            f"cosine BA={_fmt(cosine['balanced_accuracy'])}; "
            f"euclidean BA={_fmt(euclidean['balanced_accuracy'])}"
        )

    print("\nB. FINITE-SHOT PROBE SNR LADDER")
    snr = _evaluate_probe_snr(
        validation,
        repeats=args.repeats,
        bootstraps=args.bootstraps,
        seed=args.bootstrap_seed + 10000,
    )
    for strength in PROBE_STRENGTHS:
        print(f"  strength={strength:.3f}")
        for shots in PROBE_SHOTS:
            row = snr[f"strength_{strength:.3f}__shots_{shots}"]
            align = row["frame_alignment_to_exact_0p15"]
            c = row["decoders"]["cosine"]
            e = row["decoders"]["euclidean"]
            print(
                f"    shots={shots:5d}: frame cos={_fmt(align['median_cosine_all'])} "
                f"cosBA={_fmt(c['balanced_accuracy'])} eucBA={_fmt(e['balanced_accuracy'])}"
            )

    print("\nC. TIER1 ESTIMATOR / COVERAGE DECOMPOSITION")
    coverage = _evaluate_estimator_coverage(
        train,
        validation,
        alpha=args.ridge_alpha,
        bootstraps=args.bootstraps,
        seed=args.bootstrap_seed + 30000,
        config_path=args.config,
        qpu_results_path=args.qpu_case_results,
    )
    dist = coverage["coverage"]["standardized_nearest_train_distance"]
    sim = coverage["coverage"]["max_rbf_similarity_to_train"]
    print(
        "  validation nearest distance median=",
        _fmt(dist["validation_median"]),
        "p90=",
        _fmt(dist["validation_p90"]),
        "QPU=",
        _fmt(dist["qpu_context"]),
        "percentile=",
        _fmt(dist["qpu_percentile_vs_validation"], 1),
    )
    print(
        "  validation max-kernel-sim median=",
        _fmt(sim["validation_median"]),
        "QPU=",
        _fmt(sim["qpu_context"]),
    )
    for estimator in (
        "rbf_kernel_ridge",
        "one_nearest_neighbor",
        "five_neighbor_kernel_average",
    ):
        val = coverage["validation_estimators"][estimator]
        qpu = coverage["frozen_qpu_context"]["estimators"][estimator]
        print(
            f"  {estimator}: val cosBA={_fmt(val['decoders']['cosine']['balanced_accuracy'])} "
            f"eucBA={_fmt(val['decoders']['euclidean']['balanced_accuracy'])}; "
            f"QPU cosine={qpu['decoders']['cosine']['correct_count']}/3 "
            f"euclidean={qpu['decoders']['euclidean']['correct_count']}/3"
        )

    best_snr = _best_snr_cells(snr)
    interpretation = _diagnostic_interpretation(linearity, snr, coverage)
    print("\nDIAGNOSTIC INTERPRETATION")
    print("  probe linearity:", interpretation["probe_linearity"])
    print("  probe SNR:", interpretation["probe_snr"])
    print("  Tier1 domain:", interpretation["tier1_domain_coverage"])
    print("  Tier1 estimator:", interpretation["tier1_estimator_specificity"])
    print("  best SNR cosine:", best_snr["cosine"])
    print("  best SNR euclidean:", best_snr["euclidean"])

    result = {
        "analysis": ANALYSIS_NAME,
        "scientific_boundary": {
            "posthoc_only": True,
            "qpu_submission": False,
            "triqto_checkpoint_retraining": False,
            "triqto_weight_change": False,
            "triqto_threshold_change": False,
            "exact_simulator_frames_are_audit_only": True,
        },
        "step5_product_id": str(complete.get("product_id")),
        "context_counts": {
            "matched": len(contexts),
            "train": len(train),
            "validation": len(validation),
        },
        "frozen_design": {
            "probe_strengths": list(PROBE_STRENGTHS),
            "probe_shots_per_basis_per_axis": list(PROBE_SHOTS),
            "monte_carlo_repeats_per_context_cell": int(args.repeats),
            "bootstraps": int(args.bootstraps),
            "ridge_alpha": float(args.ridge_alpha),
            "reference_markers": {
                "balanced_accuracy": REFERENCE_BA,
                "cluster_bootstrap_95_lower": REFERENCE_CI_LOWER,
                "minimum_mechanism_recall": REFERENCE_MIN_RECALL,
            },
        },
        "probe_linearity_exact": linearity,
        "probe_snr_ladder": snr,
        "probe_snr_best_cells": best_snr,
        "tier1_estimator_coverage": coverage,
        "diagnostic_interpretation": interpretation,
        "hard_boundary": (
            "Step 9F diagnoses the representation failure only. Freeze and review this result "
            "before choosing any new representation, retraining, or QPU acquisition."
        ),
    }

    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nwrote:", output)


if __name__ == "__main__":
    main()
