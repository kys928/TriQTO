#!/usr/bin/env python3
"""Post-hoc matched-context replay for Step-9D phase mechanism failure.

This is a zero-QPU, zero-training analysis.  It uses the exact Step-5 v3
artifact diagnostics for matched 2-qubit phase-interference contexts where:

* affected logical qubit = 0;
* intervention strength = 0.15;
* all three RZ/RX/RY examples are mechanism-supervised;
* clean root and insertion boundary are identical within each triplet.

For each Step-9D ideal phase diagnostic vector, the script asks which mechanism
vector in each matched Step-5 context is most directionally similar.  This
measures how circuit/insertion context transports mechanism signatures before
any learned representation or classifier is involved.

Scientific boundary: post-hoc only; no QPU access, retraining, weight change,
threshold change, or confirmatory interpretation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.quantum_info import Pauli, Statevector

from triqto.hardware.qpu_pilot import build_pilot_cases


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = Path(
    "/workspace/triqto-data/step5_matched_diagnostic_training_v3/current_product.json"
)
DEFAULT_CONFIG = ROOT / "configs/v0_2/step9d_exploratory_qpu_pilot_v2.json"
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
BASIS_ORDER = ("Z", "X", "Y")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _pauli_label(n_qubits: int, basis: str, qubits: tuple[int, ...]) -> str:
    chars = ["I"] * n_qubits
    for qubit in qubits:
        chars[n_qubits - 1 - int(qubit)] = str(basis)
    return "".join(chars)


def _expectation(state: Statevector, basis: str, qubits: tuple[int, ...]) -> float:
    value = state.expectation_value(Pauli(_pauli_label(state.num_qubits, basis, qubits)))
    return float(np.real_if_close(value))


def _all_pairs(n_qubits: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(n_qubits) for right in range(left + 1, n_qubits)]


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

    return np.concatenate((local.reshape(-1), pair.reshape(-1), parity.reshape(-1)))


def _artifact_exact_vector(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        local = np.asarray(data["audit__exact_delta_local_expectations"], dtype=np.float64)
        pair = np.asarray(data["audit__exact_delta_pairwise_correlations"], dtype=np.float64)
        parity = np.asarray(data["audit__exact_delta_global_parity"], dtype=np.float64)
    return np.concatenate((local.reshape(-1), pair.reshape(-1), parity.reshape(-1)))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    lnorm = float(np.linalg.norm(left))
    rnorm = float(np.linalg.norm(right))
    if lnorm <= 1e-12 or rnorm <= 1e-12:
        return float("nan")
    return float(np.dot(left, right) / (lnorm * rnorm))


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
        vectors = {
            mechanism: _artifact_exact_vector(product / mechanisms[mechanism]["artifact_path"])
            for mechanism in MECHANISMS
        }
        contexts.append(
            {
                "group_id": group_id,
                "boundary": int(boundary),
                "depth_bin": next(iter(depth_bins)),
                "root_index": int(next(iter(roots))),
                "split": next(iter(splits)),
                "vectors": vectors,
            }
        )
    contexts.sort(key=lambda row: (row["root_index"], row["boundary"]))
    return contexts


def _pilot_phase_vectors(config_path: Path) -> dict[str, np.ndarray]:
    config = _read_json(config_path)
    cases = {case.case_id: case for case in build_pilot_cases(config, (0, 1, 2))}
    result: dict[str, np.ndarray] = {}
    for mechanism in MECHANISMS:
        case = cases[f"phase_interference__{mechanism}"]
        result[mechanism] = _ideal_vector(case.reference_circuit, case.observed_circuit)
    return result


def _summarize_query(query_mechanism: str, query: np.ndarray, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    votes: Counter[str] = Counter()
    by_depth: dict[str, Counter[str]] = defaultdict(Counter)
    cosine_by_candidate: dict[str, list[float]] = {mechanism: [] for mechanism in MECHANISMS}
    margins: list[float] = []
    rows: list[dict[str, Any]] = []

    for context in contexts:
        scores = {
            mechanism: _cosine(query, context["vectors"][mechanism])
            for mechanism in MECHANISMS
        }
        finite = {key: value for key, value in scores.items() if np.isfinite(value)}
        if len(finite) != len(MECHANISMS):
            continue
        ranked = sorted(finite.items(), key=lambda item: item[1], reverse=True)
        winner = ranked[0][0]
        margin = float(ranked[0][1] - ranked[1][1])
        votes[winner] += 1
        by_depth[context["depth_bin"]][winner] += 1
        margins.append(margin)
        for mechanism, score in scores.items():
            cosine_by_candidate[mechanism].append(float(score))
        rows.append(
            {
                "root_index": context["root_index"],
                "boundary": context["boundary"],
                "depth_bin": context["depth_bin"],
                "winner": winner,
                "scores": {key: float(value) for key, value in scores.items()},
                "margin": margin,
            }
        )

    total = sum(votes.values())
    return {
        "expected_mechanism": query_mechanism,
        "usable_context_count": total,
        "winner_counts": dict(votes),
        "winner_fractions": {
            mechanism: (float(votes[mechanism]) / total if total else float("nan"))
            for mechanism in MECHANISMS
        },
        "correct_context_fraction": (
            float(votes[query_mechanism]) / total if total else float("nan")
        ),
        "median_cosine_by_candidate": {
            mechanism: float(np.median(values)) if values else float("nan")
            for mechanism, values in cosine_by_candidate.items()
        },
        "median_winner_margin": float(np.median(margins)) if margins else float("nan"),
        "by_depth": {
            depth: {
                "count": int(sum(counter.values())),
                "winner_counts": dict(counter),
            }
            for depth, counter in sorted(by_depth.items())
        },
        "context_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    product = _resolve_product(args.product_pointer, args.product_dir)
    complete = _read_json(product / "dataset_complete.json")
    product_id = str(complete.get("product_id"))
    contexts = _matched_contexts(product)
    if not contexts:
        raise RuntimeError("no fully supervised matched 2q phase contexts at q0/strength 0.15")

    pilot = _pilot_phase_vectors(args.config)
    summaries = {
        mechanism: _summarize_query(mechanism, pilot[mechanism], contexts)
        for mechanism in MECHANISMS
    }

    depth_counts = Counter(context["depth_bin"] for context in contexts)
    split_counts = Counter(context["split"] for context in contexts)
    result = {
        "analysis": "step9d_posthoc_matched_context_replay_v1",
        "scientific_boundary": {
            "posthoc_only": True,
            "qpu_submission": False,
            "retraining": False,
            "weight_change": False,
            "threshold_change": False,
            "confirmatory_interpretation": False,
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
        "depth_counts": dict(depth_counts),
        "split_counts": dict(split_counts),
        "pilot_vector_norms": {
            mechanism: float(np.linalg.norm(vector)) for mechanism, vector in pilot.items()
        },
        "queries": summaries,
        "erratum": {
            "statement": (
                "Earlier post-hoc discussion referenced src/triqto/circuits/phase_interference.py "
                "as if it generated Step-5 training data. Step-5 actually used its dedicated "
                "build_clean_circuit() generator, whose phase_interference branch includes a CZ chain."
            ),
            "impact": (
                "The specific earlier no-entangler characterization was incorrect. The broader "
                "Step-5-vs-Step-9D circuit/context distribution-shift hypothesis remains testable "
                "and is what this matched-context analysis evaluates."
            ),
        },
    }

    print("STEP 9D POST-HOC MATCHED CONTEXT REPLAY — NO QPU ACCESS")
    print("Step-5 product:", product_id)
    print("matched phase 2q/q0/strength=.15 triplets:", len(contexts))
    print("depth counts:", dict(sorted(depth_counts.items())))
    print()

    for mechanism in MECHANISMS:
        summary = summaries[mechanism]
        print(mechanism)
        print("  winner counts:", summary["winner_counts"])
        print(
            "  winner fractions:",
            {key: round(value, 4) for key, value in summary["winner_fractions"].items()},
        )
        print("  correct-context fraction:", f"{summary['correct_context_fraction']:.4f}")
        print(
            "  median cosine by candidate:",
            {key: round(value, 4) for key, value in summary["median_cosine_by_candidate"].items()},
        )
        print("  median winner margin:", f"{summary['median_winner_margin']:.4f}")
        for depth, row in summary["by_depth"].items():
            print(f"  {depth}: n={row['count']} winners={row['winner_counts']}")
        print()

    print("ERRATUM: Step-5 phase generator includes a CZ chain; earlier no-entangler wording was incorrect.")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print("wrote:", args.output_json)


if __name__ == "__main__":
    main()
