#!/usr/bin/env python3
"""Step 9D post-hoc context-conditioned mechanism identifiability audit.

This is a zero-QPU, zero-training audit. It asks whether the existing six-program
Born diagnostic becomes mechanism-identifiable when interpreted in a local
response frame conditioned on the exact clean circuit and insertion context.

Two deliberately simple local-oracle decoders are evaluated:

1. cosine: compare the finite-shot diagnostic direction with the three exact
   same-context RZ/RX/RY response directions;
2. euclidean: compare the finite-shot diagnostic vector with the three exact
   same-context response templates at the frozen intervention strength.

The Step-5 audit uses only already-frozen artifacts. The exact local frame comes
from ``audit__exact_*`` simulator-only arrays, while the query uses the deployable
finite-shot ``x__delta_*`` arrays. The same idea is then applied to the frozen
Step-9D phase QPU counts, with the exact local response frame generated from the
frozen Step-9D intended circuit.

Therefore a positive result establishes only that the six-program evidence is
informative *given a simulator-privileged local frame*. It does not make that
frame hardware-deployable and does not authorize retraining.

Scientific boundary: post-hoc only; no QPU submission, no retraining, no weight
change, no threshold change, and no confirmatory interpretation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from qiskit.quantum_info import Pauli, Statevector

from triqto.hardware.diagnostic_acquisition import paired_diagnostic_arrays
from triqto.hardware.qpu_pilot import build_pilot_cases


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = Path(
    "/workspace/triqto-data/step5_matched_diagnostic_training_v3/current_product.json"
)
DEFAULT_CONFIG = ROOT / "configs/v0_2/step9d_exploratory_qpu_pilot_v2.json"
DEFAULT_QPU_RESULTS = (
    ROOT / "docs/evidence/step9d_exploratory_qpu_pilot/case_results.jsonl"
)

MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
BASIS_ORDER = ("Z", "X", "Y")
PAIR_NAMES = (
    ("rz_drift", "rx_overrotation"),
    ("rz_drift", "ry_overrotation"),
    ("rx_overrotation", "ry_overrotation"),
)

# Predeclared exploratory decision gate. These are design gates, not hypothesis
# test significance thresholds.
GATE_BALANCED_ACCURACY = 0.80
GATE_BOOTSTRAP_LOWER = 0.75
GATE_MIN_RECALL = 0.70
DEFAULT_BOOTSTRAPS = 4000
DEFAULT_BOOTSTRAP_SEED = 20260819


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"expected boolean text, got {value!r}")


def _flatten_arrays(local: np.ndarray, pair: np.ndarray, parity: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(local, dtype=np.float64).reshape(-1),
            np.asarray(pair, dtype=np.float64).reshape(-1),
            np.asarray(parity, dtype=np.float64).reshape(-1),
        )
    )


def _artifact_vectors(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        finite = _flatten_arrays(
            data["x__delta_local_expectations"],
            data["x__delta_pairwise_correlations"],
            data["x__delta_global_parity"],
        )
        exact = _flatten_arrays(
            data["audit__exact_delta_local_expectations"],
            data["audit__exact_delta_pairwise_correlations"],
            data["audit__exact_delta_global_parity"],
        )
        observed_shots = np.asarray(
            data["x__observed_shots"], dtype=np.int64
        ).reshape(-1)
        reference_shots = np.asarray(
            data["x__reference_shots"], dtype=np.int64
        ).reshape(-1)
    if len(set(int(x) for x in observed_shots.tolist())) != 1:
        raise RuntimeError(f"basis-varying observed shots in {path}")
    if len(set(int(x) for x in reference_shots.tolist())) != 1:
        raise RuntimeError(f"basis-varying reference shots in {path}")
    if int(observed_shots[0]) != int(reference_shots[0]):
        raise RuntimeError(f"observed/reference shot mismatch in {path}")
    return finite, exact, int(observed_shots[0])


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    lnorm = float(np.linalg.norm(left))
    rnorm = float(np.linalg.norm(right))
    if lnorm <= 1e-12 or rnorm <= 1e-12:
        return None
    return float(np.dot(left, right) / (lnorm * rnorm))


def _euclidean_score(left: np.ndarray, right: np.ndarray) -> float:
    # Higher is better, so negate the distance.
    return -float(np.linalg.norm(left - right))


def _rank_scores(
    scores: Mapping[str, float | None],
) -> tuple[str | None, float | None]:
    finite = [
        (key, float(value))
        for key, value in scores.items()
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return None, None
    finite.sort(key=lambda item: item[1], reverse=True)
    if len(finite) == 1:
        return finite[0][0], None
    return finite[0][0], float(finite[0][1] - finite[1][1])


def _score_frame(
    query: np.ndarray,
    frame: Mapping[str, np.ndarray],
    metric: str,
) -> tuple[dict[str, float | None], str | None, float | None]:
    if metric == "cosine":
        scores = {
            mechanism: _cosine(query, frame[mechanism])
            for mechanism in MECHANISMS
        }
    elif metric == "euclidean":
        scores = {
            mechanism: _euclidean_score(query, frame[mechanism])
            for mechanism in MECHANISMS
        }
    else:
        raise ValueError(f"unknown metric {metric!r}")
    pred, margin = _rank_scores(scores)
    return scores, pred, margin


def _pauli_label(n_qubits: int, basis: str, qubits: tuple[int, ...]) -> str:
    chars = ["I"] * n_qubits
    for qubit in qubits:
        chars[n_qubits - 1 - int(qubit)] = str(basis)
    return "".join(chars)


def _expectation(state: Statevector, basis: str, qubits: tuple[int, ...]) -> float:
    value = state.expectation_value(
        Pauli(_pauli_label(state.num_qubits, basis, qubits))
    )
    return float(np.real_if_close(value))


def _all_pairs(n_qubits: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(n_qubits)
        for right in range(left + 1, n_qubits)
    ]


def _ideal_vector(reference_circuit, observed_circuit) -> np.ndarray:
    if reference_circuit.num_qubits != observed_circuit.num_qubits:
        raise RuntimeError("reference/observed qubit-count mismatch")
    n_qubits = reference_circuit.num_qubits
    reference = Statevector.from_instruction(reference_circuit)
    observed = Statevector.from_instruction(observed_circuit)
    pairs = _all_pairs(n_qubits)

    local = np.zeros((3, n_qubits), dtype=np.float64)
    pair = np.zeros((3, len(pairs)), dtype=np.float64)
    parity = np.zeros(3, dtype=np.float64)

    for basis_index, basis in enumerate(BASIS_ORDER):
        for qubit in range(n_qubits):
            local[basis_index, qubit] = (
                _expectation(observed, basis, (qubit,))
                - _expectation(reference, basis, (qubit,))
            )
        for pair_index, (left, right) in enumerate(pairs):
            pair[basis_index, pair_index] = (
                _expectation(observed, basis, (left, right))
                - _expectation(reference, basis, (left, right))
            )
        parity[basis_index] = (
            _expectation(observed, basis, tuple(range(n_qubits)))
            - _expectation(reference, basis, tuple(range(n_qubits)))
        )

    return _flatten_arrays(local, pair, parity)


def _resolve_product(pointer: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    payload = _read_json(pointer.expanduser().resolve())
    return Path(payload["product_dir"]).expanduser().resolve()


def _matched_contexts(product: Path) -> list[dict[str, Any]]:
    rows = _read_csv(product / "manifests/example_manifest.csv")
    selected = [
        row
        for row in rows
        if row["family"] == "phase_interference"
        and int(row["n_qubits"]) == 2
        and not _as_bool(row["clean_control"])
        and _as_bool(row["mechanism_loss_mask"])
        and int(row["affected_qubit"]) == 0
        and abs(float(row["strength"]) - 0.15) <= 1e-12
    ]

    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in selected:
        key = (row["clean_circuit_group_id"], row["insertion_boundary_rank"])
        grouped[key][row["mechanism"]] = row

    contexts: list[dict[str, Any]] = []
    for (group_id, boundary), mechanisms in grouped.items():
        if set(mechanisms) != set(MECHANISMS):
            continue
        depth_bins = {row["insertion_depth_bin"] for row in mechanisms.values()}
        roots = {row["root_index"] for row in mechanisms.values()}
        splits = {row["split"] for row in mechanisms.values()}
        if len(depth_bins) != 1 or len(roots) != 1 or len(splits) != 1:
            raise RuntimeError("matched Step-5 triplet metadata drift")

        finite: dict[str, np.ndarray] = {}
        exact: dict[str, np.ndarray] = {}
        shots: dict[str, int] = {}
        for mechanism in MECHANISMS:
            artifact = product / mechanisms[mechanism]["artifact_path"]
            finite[mechanism], exact[mechanism], shots[mechanism] = (
                _artifact_vectors(artifact)
            )

        dimensions = {int(vector.shape[0]) for vector in exact.values()}
        if dimensions != {12}:
            raise RuntimeError(
                f"expected 12D 2q diagnostic vectors, got {dimensions}"
            )

        contexts.append(
            {
                "group_id": group_id,
                "boundary": int(boundary),
                "depth_bin": next(iter(depth_bins)),
                "root_index": int(next(iter(roots))),
                "split": next(iter(splits)),
                "finite": finite,
                "exact": exact,
                "shots": shots,
            }
        )

    contexts.sort(key=lambda row: (row["root_index"], row["boundary"]))
    return contexts


def _pairwise_angles(
    frame: Mapping[str, np.ndarray],
) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for left, right in PAIR_NAMES:
        cosine = _cosine(frame[left], frame[right])
        if cosine is None:
            output[f"{left}__{right}"] = None
            continue
        clipped = float(np.clip(cosine, -1.0, 1.0))
        output[f"{left}__{right}"] = float(
            np.degrees(np.arccos(clipped))
        )
    return output


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _frame_geometry(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    minimum_angles: list[float] = []
    self_margins: dict[str, list[float]] = {
        mechanism: [] for mechanism in MECHANISMS
    }
    exact_self_correct: Counter[str] = Counter()

    for index, context in enumerate(contexts):
        angles = _pairwise_angles(context["exact"])
        finite_angles = [
            float(value) for value in angles.values() if value is not None
        ]
        minimum_angle = min(finite_angles) if finite_angles else None
        if minimum_angle is not None:
            minimum_angles.append(minimum_angle)

        exact_self: dict[str, Any] = {}
        for mechanism in MECHANISMS:
            scores, pred, margin = _score_frame(
                context["exact"][mechanism], context["exact"], "cosine"
            )
            if pred == mechanism:
                exact_self_correct[mechanism] += 1
            if margin is not None:
                self_margins[mechanism].append(margin)
            exact_self[mechanism] = {
                "prediction": pred,
                "scores": scores,
                "margin": margin,
            }

        rows.append(
            {
                "context_index": index,
                "root_index": context["root_index"],
                "boundary": context["boundary"],
                "depth_bin": context["depth_bin"],
                "split": context["split"],
                "norms": {
                    mechanism: float(
                        np.linalg.norm(context["exact"][mechanism])
                    )
                    for mechanism in MECHANISMS
                },
                "pairwise_angles_deg": angles,
                "minimum_pairwise_angle_deg": minimum_angle,
                "exact_self": exact_self,
            }
        )

    return {
        "context_count": len(contexts),
        "minimum_pairwise_angle_deg": {
            "min": min(minimum_angles) if minimum_angles else None,
            "p10": _percentile(minimum_angles, 10.0),
            "median": _percentile(minimum_angles, 50.0),
            "p90": _percentile(minimum_angles, 90.0),
            "fraction_below_15_deg": (
                float(np.mean(np.asarray(minimum_angles) < 15.0))
                if minimum_angles
                else None
            ),
            "fraction_below_30_deg": (
                float(np.mean(np.asarray(minimum_angles) < 30.0))
                if minimum_angles
                else None
            ),
        },
        "exact_self_correct": {
            mechanism: int(exact_self_correct[mechanism])
            for mechanism in MECHANISMS
        },
        "median_exact_self_margin": {
            mechanism: _percentile(values, 50.0)
            for mechanism, values in self_margins.items()
        },
        "rows": rows,
    }


def _decoder_records(
    contexts: list[dict[str, Any]], metric: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for context_index, context in enumerate(contexts):
        frame = context["exact"]
        for mechanism in MECHANISMS:
            scores, pred, margin = _score_frame(
                context["finite"][mechanism], frame, metric
            )
            records.append(
                {
                    "context_index": context_index,
                    "root_index": context["root_index"],
                    "boundary": context["boundary"],
                    "depth_bin": context["depth_bin"],
                    "split": context["split"],
                    "shots": int(context["shots"][mechanism]),
                    "true": mechanism,
                    "prediction": pred,
                    "correct": bool(pred == mechanism),
                    "scores": scores,
                    "margin": margin,
                    "finite_norm": float(
                        np.linalg.norm(context["finite"][mechanism])
                    ),
                    "exact_norm": float(
                        np.linalg.norm(context["exact"][mechanism])
                    ),
                    "finite_to_own_exact_cosine": _cosine(
                        context["finite"][mechanism],
                        context["exact"][mechanism],
                    ),
                }
            )
    return records


def _basic_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in records if row["prediction"] in MECHANISMS]
    confusion = {
        true: {pred: 0 for pred in MECHANISMS}
        for true in MECHANISMS
    }
    recalls: dict[str, float | None] = {}
    for row in usable:
        confusion[row["true"]][row["prediction"]] += 1

    for mechanism in MECHANISMS:
        subset = [row for row in usable if row["true"] == mechanism]
        recalls[mechanism] = (
            float(sum(row["correct"] for row in subset) / len(subset))
            if subset
            else None
        )

    recall_values = [
        float(value) for value in recalls.values() if value is not None
    ]
    margins = [
        float(row["margin"])
        for row in usable
        if row["margin"] is not None
        and math.isfinite(float(row["margin"]))
    ]
    own_cosines = [
        float(row["finite_to_own_exact_cosine"])
        for row in usable
        if row["finite_to_own_exact_cosine"] is not None
    ]
    return {
        "record_count": len(records),
        "usable_count": len(usable),
        "accuracy": (
            float(sum(row["correct"] for row in usable) / len(usable))
            if usable
            else None
        ),
        "balanced_accuracy": (
            float(np.mean(recall_values)) if recall_values else None
        ),
        "recall_by_mechanism": recalls,
        "confusion": confusion,
        "median_winner_margin": _percentile(margins, 50.0),
        "median_finite_to_own_exact_cosine": _percentile(
            own_cosines, 50.0
        ),
    }


def _group_metrics(
    records: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[field]) for row in records})
    return {
        value: _basic_metrics(
            [row for row in records if str(row[field]) == value]
        )
        for value in values
    }


def _cluster_bootstrap(
    records: list[dict[str, Any]],
    context_count: int,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    by_context: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_context[int(row["context_index"])].append(row)

    missing = [
        index for index in range(context_count) if index not in by_context
    ]
    if missing:
        raise RuntimeError(f"decoder records missing contexts: {missing[:5]}")

    rng = np.random.default_rng(seed)
    ba_values: list[float] = []
    recall_values: dict[str, list[float]] = {
        mechanism: [] for mechanism in MECHANISMS
    }

    for _ in range(repeats):
        sampled = rng.integers(0, context_count, size=context_count)
        sample_rows: list[dict[str, Any]] = []
        for index in sampled.tolist():
            sample_rows.extend(by_context[int(index)])
        metrics = _basic_metrics(sample_rows)
        ba = metrics["balanced_accuracy"]
        if ba is not None:
            ba_values.append(float(ba))
        for mechanism in MECHANISMS:
            recall = metrics["recall_by_mechanism"][mechanism]
            if recall is not None:
                recall_values[mechanism].append(float(recall))

    return {
        "repeats": repeats,
        "seed": seed,
        "balanced_accuracy_95": [
            _percentile(ba_values, 2.5),
            _percentile(ba_values, 97.5),
        ],
        "recall_95": {
            mechanism: [
                _percentile(values, 2.5),
                _percentile(values, 97.5),
            ]
            for mechanism, values in recall_values.items()
        },
    }


def _summarize_decoder(
    records: list[dict[str, Any]],
    context_count: int,
    *,
    bootstraps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    overall = _basic_metrics(records)
    bootstrap = _cluster_bootstrap(
        records,
        context_count,
        repeats=bootstraps,
        seed=bootstrap_seed,
    )
    return {
        "overall": overall,
        "cluster_bootstrap": bootstrap,
        "by_depth": _group_metrics(records, "depth_bin"),
        "by_split": _group_metrics(records, "split"),
        "by_shots": _group_metrics(records, "shots"),
        "records": records,
    }


def _step5_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    overall = summary["overall"]
    bootstrap = summary["cluster_bootstrap"]
    ba = overall["balanced_accuracy"]
    lower = bootstrap["balanced_accuracy_95"][0]
    recalls = overall["recall_by_mechanism"]
    recall_values = [
        float(recalls[mechanism])
        for mechanism in MECHANISMS
        if recalls[mechanism] is not None
    ]
    minimum_recall = (
        min(recall_values)
        if len(recall_values) == len(MECHANISMS)
        else None
    )
    passed = bool(
        ba is not None
        and lower is not None
        and minimum_recall is not None
        and float(ba) >= GATE_BALANCED_ACCURACY
        and float(lower) >= GATE_BOOTSTRAP_LOWER
        and float(minimum_recall) >= GATE_MIN_RECALL
    )
    return {
        "passed": passed,
        "balanced_accuracy": ba,
        "balanced_accuracy_95_lower": lower,
        "minimum_mechanism_recall": minimum_recall,
        "thresholds": {
            "balanced_accuracy": GATE_BALANCED_ACCURACY,
            "balanced_accuracy_95_lower": GATE_BOOTSTRAP_LOWER,
            "minimum_mechanism_recall": GATE_MIN_RECALL,
        },
    }


def _pilot_phase_exact_frame(config_path: Path) -> dict[str, np.ndarray]:
    config = _read_json(config_path)
    cases = {
        case.case_id: case
        for case in build_pilot_cases(config, (0, 1, 2))
    }
    frame: dict[str, np.ndarray] = {}
    for mechanism in MECHANISMS:
        case_id = f"phase_interference__{mechanism}"
        if case_id not in cases:
            raise RuntimeError(f"missing frozen pilot case {case_id}")
        case = cases[case_id]
        frame[mechanism] = _ideal_vector(
            case.reference_circuit,
            case.observed_circuit,
        )
    return frame


def _qpu_vector(row: Mapping[str, Any]) -> np.ndarray:
    counts = row["counts_by_program"]
    reference = {
        basis: counts[f"reference_{basis}"] for basis in BASIS_ORDER
    }
    observed = {
        basis: counts[f"observed_{basis}"] for basis in BASIS_ORDER
    }
    n_qubits = len(row["physical_layout"])
    arrays = paired_diagnostic_arrays(reference, observed, n_qubits)
    return _flatten_arrays(
        arrays["x__delta_local_expectations"],
        arrays["x__delta_pairwise_correlations"],
        arrays["x__delta_global_parity"],
    )


def _qpu_audit(
    qpu_results_path: Path,
    exact_frame: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    rows = _read_jsonl(qpu_results_path)
    phase = {
        str(row["expected_mechanism"]): row
        for row in rows
        if row.get("family") == "phase_interference"
        and row.get("expected_mechanism") in MECHANISMS
    }
    if set(phase) != set(MECHANISMS):
        raise RuntimeError(
            "expected all three frozen phase QPU mechanism rows, got "
            f"{sorted(phase)}"
        )

    result: dict[str, Any] = {
        "exact_frame_pairwise_angles_deg": _pairwise_angles(exact_frame),
        "decoders": {},
    }
    for metric in ("cosine", "euclidean"):
        rows_out: list[dict[str, Any]] = []
        for mechanism in MECHANISMS:
            qpu = phase[mechanism]
            query = _qpu_vector(qpu)
            scores, pred, margin = _score_frame(query, exact_frame, metric)
            rows_out.append(
                {
                    "expected": mechanism,
                    "prediction": pred,
                    "correct": bool(pred == mechanism),
                    "scores": scores,
                    "margin": margin,
                    "hardware_norm": float(np.linalg.norm(query)),
                    "hardware_to_own_exact_cosine": _cosine(
                        query, exact_frame[mechanism]
                    ),
                    "frozen_model_prediction": qpu["prediction"][
                        "mechanism_prediction"
                    ],
                }
            )
        correct = int(sum(row["correct"] for row in rows_out))
        result["decoders"][metric] = {
            "correct_count": correct,
            "case_count": len(rows_out),
            "all_three_correct": bool(correct == len(MECHANISMS)),
            "rows": rows_out,
        }
    return result


def _decision_gate(
    step5: Mapping[str, Any],
    qpu: Mapping[str, Any],
) -> dict[str, Any]:
    step5_status = {
        metric: _step5_gate(step5[metric])
        for metric in ("cosine", "euclidean")
    }
    passing_step5 = [
        metric for metric, row in step5_status.items() if row["passed"]
    ]
    passing_qpu = [
        metric
        for metric in passing_step5
        if qpu["decoders"][metric]["all_three_correct"]
    ]

    if passing_qpu:
        status = (
            "LOCAL_RESPONSE_FRAME_SUFFICIENT_IN_TARGETED_AUDIT__"
            "SIMULATOR_PRIVILEGED"
        )
        next_action = (
            "Do not retrain yet. Design and test a hardware-valid approximation "
            "to the local response frame from deployable circuit/context "
            "information."
        )
    elif passing_step5:
        status = (
            "LOCAL_RESPONSE_FRAME_WORKS_ON_STEP5_BUT_QPU_ALIGNMENT_IS_"
            "INCOMPLETE"
        )
        next_action = (
            "Do not treat this as an architecture-only failure. Localize the "
            "simulator-to-QPU response-frame mismatch before redesign or "
            "retraining."
        )
    else:
        status = (
            "LOCAL_RESPONSE_FRAME_DOES_NOT_DEMONSTRATE_DIAGNOSTIC_"
            "SUFFICIENCY"
        )
        next_action = (
            "Do not assume a bigger classifier will solve the problem. "
            "Investigate ambiguous local frames, finite-shot acquisition, and "
            "whether richer hardware-valid diagnostics or abstention are "
            "required."
        )

    return {
        "status": status,
        "passing_step5_decoders": passing_step5,
        "passing_step5_and_qpu_decoders": passing_qpu,
        "step5_decoder_gates": step5_status,
        "simulator_privileged_frame": True,
        "hardware_deployable_frame_demonstrated": False,
        "next_action": next_action,
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
    parser.add_argument(
        "--product-pointer", type=Path, default=DEFAULT_POINTER
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--qpu-case-results", type=Path, default=DEFAULT_QPU_RESULTS
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    args = parser.parse_args()

    if args.bootstraps <= 0:
        raise ValueError("--bootstraps must be positive")

    product = _resolve_product(args.product_pointer, args.product_dir)
    complete = _read_json(product / "dataset_complete.json")
    product_id = str(complete.get("product_id"))
    contexts = _matched_contexts(product)
    if not contexts:
        raise RuntimeError(
            "no fully supervised matched phase 2q/q0/strength=.15 contexts"
        )

    geometry = _frame_geometry(contexts)
    step5: dict[str, Any] = {}
    for metric_index, metric in enumerate(("cosine", "euclidean")):
        records = _decoder_records(contexts, metric)
        step5[metric] = _summarize_decoder(
            records,
            len(contexts),
            bootstraps=args.bootstraps,
            bootstrap_seed=args.bootstrap_seed + metric_index,
        )

    pilot_frame = _pilot_phase_exact_frame(args.config)
    qpu = _qpu_audit(args.qpu_case_results, pilot_frame)
    decision = _decision_gate(step5, qpu)

    result = {
        "analysis": (
            "step9d_posthoc_context_conditioned_identifiability_audit_v1"
        ),
        "scientific_boundary": {
            "posthoc_only": True,
            "qpu_submission": False,
            "retraining": False,
            "weight_change": False,
            "threshold_change": False,
            "confirmatory_interpretation": False,
            "simulator_privileged_local_frame": True,
        },
        "step5_product_id": product_id,
        "context_contract": {
            "family": "phase_interference",
            "n_qubits": 2,
            "affected_qubit": 0,
            "strength": 0.15,
            "require_mechanism_supervision": True,
            "require_complete_rz_rx_ry_triplet": True,
        },
        "matched_context_count": len(contexts),
        "frame_definition": {
            "description": (
                "Exact same-context RZ/RX/RY response vectors at strength 0.15. "
                "This is an operational local response frame, not an "
                "infinitesimal mathematical tangent."
            ),
            "step5_source": "audit__exact_* arrays",
            "step9d_qpu_source": (
                "exact Statevector response of frozen pilot cases"
            ),
            "deployable": False,
        },
        "predeclared_gate": {
            "balanced_accuracy_minimum": GATE_BALANCED_ACCURACY,
            "cluster_bootstrap_95_lower_minimum": GATE_BOOTSTRAP_LOWER,
            "minimum_mechanism_recall": GATE_MIN_RECALL,
            "qpu_alignment_rule": (
                "A Step-5-passing decoder must also classify all three frozen "
                "Step-9D phase QPU cases correctly to reach the targeted "
                "simulator-privileged local-frame sufficiency status."
            ),
        },
        "step5_exact_frame_geometry": geometry,
        "step5_finite_shot_local_oracle": step5,
        "step9d_phase_qpu_local_oracle": qpu,
        "decision_gate": decision,
    }

    print(
        "STEP 9D POST-HOC CONTEXT-CONDITIONED IDENTIFIABILITY AUDIT — "
        "NO QPU ACCESS / NO RETRAINING"
    )
    print("Step-5 product:", product_id)
    print(
        "matched contexts:",
        len(contexts),
        "finite-shot queries:",
        len(contexts) * 3,
    )
    print()

    angle_summary = geometry["minimum_pairwise_angle_deg"]
    print("EXACT LOCAL FRAME GEOMETRY")
    print(
        "  min-angle deg: min=",
        _fmt(angle_summary["min"], 2),
        "p10=",
        _fmt(angle_summary["p10"], 2),
        "median=",
        _fmt(angle_summary["median"], 2),
    )
    print(
        "  fraction min-angle <15deg=",
        _fmt(angle_summary["fraction_below_15_deg"]),
        "<30deg=",
        _fmt(angle_summary["fraction_below_30_deg"]),
    )
    print("  exact self-correct:", geometry["exact_self_correct"])
    print()

    for metric in ("cosine", "euclidean"):
        summary = step5[metric]
        overall = summary["overall"]
        bootstrap = summary["cluster_bootstrap"]
        gate = _step5_gate(summary)
        print("STEP-5 FINITE-SHOT LOCAL ORACLE —", metric.upper())
        print(
            "  BA=",
            _fmt(overall["balanced_accuracy"]),
            "95% cluster CI=",
            [
                _fmt(value)
                for value in bootstrap["balanced_accuracy_95"]
            ],
            "recalls=",
            {
                key: _fmt(value)
                for key, value in overall["recall_by_mechanism"].items()
            },
        )
        print("  confusion:", overall["confusion"])
        print(
            "  median margin=",
            _fmt(overall["median_winner_margin"]),
            "median finite->own exact cosine=",
            _fmt(overall["median_finite_to_own_exact_cosine"]),
        )
        if "4096" in summary["by_shots"]:
            shot = summary["by_shots"]["4096"]
            print(
                "  4096-shot subset: n=",
                shot["usable_count"],
                "BA=",
                _fmt(shot["balanced_accuracy"]),
                "recalls=",
                {
                    key: _fmt(value)
                    for key, value in shot["recall_by_mechanism"].items()
                },
            )
        print(
            "  predeclared Step-5 gate:",
            "PASS" if gate["passed"] else "FAIL",
        )
        print()

    print("FROZEN STEP-9D PHASE QPU AGAINST EXACT LOCAL FRAME")
    for metric in ("cosine", "euclidean"):
        audit = qpu["decoders"][metric]
        print(" ", metric.upper(), f"{audit['correct_count']}/{audit['case_count']}")
        for row in audit["rows"]:
            print(
                "   ",
                row["expected"],
                "->",
                row["prediction"],
                "margin=",
                _fmt(row["margin"]),
                "own-cos=",
                _fmt(row["hardware_to_own_exact_cosine"]),
                "frozen-model=",
                row["frozen_model_prediction"],
            )
    print()

    print("DECISION GATE:", decision["status"])
    print(
        "  passing Step-5 decoders:",
        decision["passing_step5_decoders"],
    )
    print(
        "  passing Step-5 + all-three-QPU decoders:",
        decision["passing_step5_and_qpu_decoders"],
    )
    print(
        "  simulator-privileged frame:",
        decision["simulator_privileged_frame"],
    )
    print(
        "  hardware-deployable frame demonstrated:",
        decision["hardware_deployable_frame_demonstrated"],
    )
    print("  next:", decision["next_action"])

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print()
        print("wrote:", args.output_json)


if __name__ == "__main__":
    main()
