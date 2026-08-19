#!/usr/bin/env python3
"""Post-hoc Step-9D raw diagnostic training-coverage analysis.

This script is diagnostic only. It performs no QPU access, no retraining, and
no model/threshold/weight changes.

For each distorted Step-9D pilot case it compares:
1) recorded QPU finite-shot diagnostics against the matching frozen Step-5
   finite-shot development distribution;
2) exact ideal Step-9D diagnostics against the matching Step-5 audit-exact
   development distribution.

Geometry is evaluated only within the same circuit family and qubit count.
Distances are standardized per training subset. Query nearest-neighbor distance
is also ranked against the leave-one-out nearest-neighbor distances among the
training examples to provide a descriptive OOD percentile.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit.quantum_info import Pauli, Statevector

from triqto.hardware.diagnostic_acquisition import (
    BASIS_ORDER,
    all_pair_indices,
    paired_diagnostic_arrays,
)
from triqto.hardware.qpu_pilot import build_pilot_cases


DEFAULT_PRODUCT_POINTER = Path(
    "/workspace/triqto-data/step5_matched_diagnostic_training_v3/current_product.json"
)
DEFAULT_CONFIG = Path("configs/v0_2/step9d_exploratory_qpu_pilot_v2.json")
DEFAULT_EVIDENCE = Path("docs/evidence/step9d_exploratory_qpu_pilot")
DEFAULT_OUTPUT = Path(
    "/workspace/triqto-data/step9d_posthoc/training_coverage_v1.json"
)
EXPECTED_PRODUCT_ID = "product_b2d78ad2309b71a55f9bb54f"
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
K_NEIGHBORS = 25


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"cannot parse bool: {value!r}")


def _resolve_product(pointer: Path) -> Path:
    payload = _read_json(pointer.expanduser().resolve())
    product = Path(payload["product_dir"]).expanduser().resolve()
    complete = _read_json(product / "dataset_complete.json")
    if complete.get("product_id") != EXPECTED_PRODUCT_ID:
        raise RuntimeError(
            f"wrong Step-5 product: {complete.get('product_id')} != {EXPECTED_PRODUCT_ID}"
        )
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("Step-5 product is not COMPLETE")
    return product


def _pauli_label(n_qubits: int, basis: str, qubits: Sequence[int]) -> str:
    chars = ["I"] * n_qubits
    for qubit in qubits:
        chars[n_qubits - 1 - int(qubit)] = str(basis)
    return "".join(chars)


def _expectation(
    state: Statevector, basis: str, qubits: Sequence[int]
) -> float:
    value = state.expectation_value(Pauli(_pauli_label(state.num_qubits, basis, qubits)))
    return float(np.real_if_close(value))


def _ideal_basis_arrays(reference_circuit, observed_circuit) -> dict[str, np.ndarray]:
    if reference_circuit.num_qubits != observed_circuit.num_qubits:
        raise RuntimeError("reference/observed qubit-count mismatch")
    n_qubits = reference_circuit.num_qubits
    pairs = all_pair_indices(n_qubits)
    reference = Statevector.from_instruction(reference_circuit)
    observed = Statevector.from_instruction(observed_circuit)

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
                _expectation(observed, basis, (int(left), int(right)))
                - _expectation(reference, basis, (int(left), int(right)))
            )
        all_qubits = tuple(range(n_qubits))
        parity[basis_index] = (
            _expectation(observed, basis, all_qubits)
            - _expectation(reference, basis, all_qubits)
        )

    return {
        "local": local,
        "pair": pair,
        "parity": parity,
    }


def _vector(local: np.ndarray, pair: np.ndarray, parity: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(local, dtype=np.float64).reshape(-1),
            np.asarray(pair, dtype=np.float64).reshape(-1),
            np.asarray(parity, dtype=np.float64).reshape(-1),
        )
    )


def _artifact_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        finite = _vector(
            data["x__delta_local_expectations"],
            data["x__delta_pairwise_correlations"],
            data["x__delta_global_parity"],
        )
        exact = _vector(
            data["audit__exact_delta_local_expectations"],
            data["audit__exact_delta_pairwise_correlations"],
            data["audit__exact_delta_global_parity"],
        )
    return finite, exact


def _hardware_vector(
    row: Mapping[str, Any], n_qubits: int
) -> np.ndarray:
    counts = row["counts_by_program"]
    reference_counts = {
        basis: counts[f"reference_{basis}"] for basis in BASIS_ORDER
    }
    observed_counts = {
        basis: counts[f"observed_{basis}"] for basis in BASIS_ORDER
    }
    arrays = paired_diagnostic_arrays(
        reference_counts,
        observed_counts,
        n_qubits,
    )
    return _vector(
        arrays["x__delta_local_expectations"],
        arrays["x__delta_pairwise_correlations"],
        arrays["x__delta_global_parity"],
    )


def _pairwise_loo_nn(x: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    n = int(x.shape[0])
    if n < 2:
        return np.full(n, np.nan, dtype=np.float64)
    norms = np.einsum("ij,ij->i", x, x)
    output = np.full(n, np.inf, dtype=np.float64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = x[start:stop]
        d2 = (
            norms[start:stop, None]
            + norms[None, :]
            - 2.0 * np.matmul(block, x.T)
        )
        np.maximum(d2, 0.0, out=d2)
        rows = np.arange(stop - start)
        d2[rows, start + rows] = np.inf
        output[start:stop] = np.sqrt(np.min(d2, axis=1))
    return output


def _prepare_geometry(
    x: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict[str, str]],
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    if x.ndim != 2 or len(x) != len(labels) or len(x) != len(metadata):
        raise ValueError("geometry training arrays are inconsistent")
    if len(x) < 2:
        raise ValueError("geometry subset needs at least two examples")

    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    active = scale > 1e-10
    if not bool(active.any()):
        active = np.ones_like(scale, dtype=bool)
        scale = np.ones_like(scale)
    z = (x[:, active] - mean[active]) / scale[active]
    loo = _pairwise_loo_nn(z)

    centroids: dict[str, np.ndarray] = {}
    for mechanism in MECHANISMS:
        mask = labels == mechanism
        if bool(mask.any()):
            centroids[mechanism] = z[mask].mean(axis=0)

    return {
        "mean": mean,
        "scale": scale,
        "active": active,
        "z": z,
        "labels": labels,
        "metadata": metadata,
        "loo_nn": loo,
        "centroids": centroids,
        "class_counts": dict(Counter(labels.tolist())),
    }


def _evaluate_query(
    prepared: Mapping[str, Any],
    query: np.ndarray,
    expected: str,
    *,
    k: int = K_NEIGHBORS,
) -> dict[str, Any]:
    mean = np.asarray(prepared["mean"], dtype=np.float64)
    scale = np.asarray(prepared["scale"], dtype=np.float64)
    active = np.asarray(prepared["active"], dtype=bool)
    z = np.asarray(prepared["z"], dtype=np.float64)
    labels = np.asarray(prepared["labels"], dtype=str)
    metadata = list(prepared["metadata"])
    loo = np.asarray(prepared["loo_nn"], dtype=np.float64)

    query = np.asarray(query, dtype=np.float64)
    if query.shape != mean.shape:
        raise ValueError(f"query dimension {query.shape} != training dimension {mean.shape}")

    qz = (query[active] - mean[active]) / scale[active]
    distances = np.linalg.norm(z - qz[None, :], axis=1)
    order = np.argsort(distances)
    k_eff = min(int(k), len(order))
    top = order[:k_eff]
    top_labels = labels[top]
    top_counts = Counter(top_labels.tolist())
    majority_label, majority_count = sorted(
        top_counts.items(), key=lambda item: (-item[1], item[0])
    )[0]

    expected_positions = np.where(labels[order] == expected)[0]
    expected_rank = (
        int(expected_positions[0]) + 1 if len(expected_positions) else None
    )
    nearest_expected_distance = (
        float(distances[order[expected_positions[0]]])
        if len(expected_positions)
        else None
    )
    other_positions = np.where(labels[order] != expected)[0]
    nearest_other_distance = (
        float(distances[order[other_positions[0]]])
        if len(other_positions)
        else None
    )

    centroid_distances = {
        mechanism: float(np.linalg.norm(qz - np.asarray(centroid)))
        for mechanism, centroid in prepared["centroids"].items()
    }
    nearest_centroid = min(
        centroid_distances, key=centroid_distances.get
    ) if centroid_distances else None

    nearest_distance = float(distances[order[0]])
    finite_loo = loo[np.isfinite(loo)]
    ood_percentile = (
        float(100.0 * np.mean(finite_loo <= nearest_distance))
        if finite_loo.size
        else None
    )

    nearest_examples = []
    for index in order[: min(5, len(order))]:
        row = metadata[int(index)]
        nearest_examples.append(
            {
                "example_id": row.get("example_id"),
                "mechanism": row.get("mechanism"),
                "distance_z": float(distances[int(index)]),
                "strength": row.get("strength"),
                "shots": row.get("shots"),
                "split": row.get("split"),
                "affected_qubit": row.get("affected_qubit"),
                "insertion_depth_bin": row.get("insertion_depth_bin"),
            }
        )

    return {
        "training_count": int(len(labels)),
        "class_counts": dict(prepared["class_counts"]),
        "active_dimensions": int(active.sum()),
        "nearest_label": str(labels[order[0]]),
        "nearest_distance_z": nearest_distance,
        "top_k": int(k_eff),
        "top_k_counts": dict(top_counts),
        "top_k_majority_label": str(majority_label),
        "top_k_majority_fraction": float(majority_count / k_eff),
        "expected_mechanism_nearest_rank": expected_rank,
        "nearest_expected_distance_z": nearest_expected_distance,
        "nearest_other_distance_z": nearest_other_distance,
        "nearest_centroid_label": nearest_centroid,
        "centroid_distances_z": centroid_distances,
        "query_nn_ood_percentile": ood_percentile,
        "nearest_examples": nearest_examples,
    }


def _subset_mask(
    rows: Sequence[Mapping[str, str]],
    *,
    strength: float | None = None,
    shots: int | None = None,
) -> np.ndarray:
    mask = np.ones(len(rows), dtype=bool)
    if strength is not None:
        mask &= np.asarray(
            [abs(float(row["strength"]) - float(strength)) < 1e-12 for row in rows],
            dtype=bool,
        )
    if shots is not None:
        mask &= np.asarray(
            [int(row["shots"]) == int(shots) for row in rows],
            dtype=bool,
        )
    return mask


def _prepared_subset(
    x: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    mask: np.ndarray,
) -> dict[str, Any] | None:
    if int(mask.sum()) < 2:
        return None
    selected_rows = [rows[index] for index in np.flatnonzero(mask)]
    return _prepare_geometry(x[mask], labels[mask], selected_rows)


def _format_probe(name: str, result: Mapping[str, Any] | None) -> str:
    if result is None:
        return f"  {name}: unavailable"
    return (
        f"  {name}: "
        f"NN={result['nearest_label']} "
        f"top{result['top_k']}={result['top_k_majority_label']}"
        f"({result['top_k_majority_fraction']:.2f}) "
        f"true-rank={result['expected_mechanism_nearest_rank']} "
        f"OOD={result['query_nn_ood_percentile']:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product-pointer",
        type=Path,
        default=DEFAULT_PRODUCT_POINTER,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    product = _resolve_product(args.product_pointer)
    config = _read_json(args.config)
    evidence_dir = args.evidence_dir.expanduser().resolve()
    summary = _read_json(evidence_dir / "pilot_summary.json")
    qpu_rows = _read_jsonl(evidence_dir / "case_results.jsonl")
    qpu_by_case = {str(row["case_id"]): row for row in qpu_rows}
    physical_chain = tuple(int(value) for value in summary["physical_chain"])
    pilot_cases = build_pilot_cases(config, physical_chain)
    distorted_cases = [case for case in pilot_cases if case.expected_effect]

    manifest_rows = _read_csv(product / "manifests/example_manifest.csv")
    supervised = [
        row
        for row in manifest_rows
        if not _as_bool(row["clean_control"])
        and _as_bool(row["mechanism_loss_mask"])
        and row["mechanism"] in MECHANISMS
    ]

    family_keys = sorted(
        {(case.family, case.reference_circuit.num_qubits) for case in distorted_cases}
    )
    family_data: dict[tuple[str, int], dict[str, Any]] = {}

    print("STEP 9D POST-HOC RAW TRAINING-COVERAGE ANALYSIS — NO QPU ACCESS")
    print("Step-5 product:", product.name)
    print()

    for family, n_qubits in family_keys:
        rows = [
            row
            for row in supervised
            if row["family"] == family and int(row["n_qubits"]) == n_qubits
        ]
        if not rows:
            available = sorted({row["family"] for row in supervised})
            raise RuntimeError(
                f"no supervised Step-5 rows for family={family!r}, "
                f"n_qubits={n_qubits}; available families={available}"
            )

        finite_vectors: list[np.ndarray] = []
        exact_vectors: list[np.ndarray] = []
        for row in rows:
            artifact = product / row["artifact_path"]
            if not artifact.is_file():
                raise RuntimeError(f"missing Step-5 artifact: {artifact}")
            finite, exact = _artifact_vectors(artifact)
            finite_vectors.append(finite)
            exact_vectors.append(exact)

        finite_x = np.stack(finite_vectors, axis=0)
        exact_x = np.stack(exact_vectors, axis=0)
        labels = np.asarray([row["mechanism"] for row in rows], dtype=str)
        strength_mask = _subset_mask(rows, strength=0.15)
        strength_shot_mask = _subset_mask(rows, strength=0.15, shots=4096)

        family_data[(family, n_qubits)] = {
            "rows": rows,
            "labels": labels,
            "finite_x": finite_x,
            "exact_x": exact_x,
            "finite_all": _prepare_geometry(finite_x, labels, rows),
            "finite_strength_015": _prepared_subset(
                finite_x, labels, rows, strength_mask
            ),
            "finite_strength_015_shots_4096": _prepared_subset(
                finite_x, labels, rows, strength_shot_mask
            ),
            "exact_all": _prepare_geometry(exact_x, labels, rows),
            "exact_strength_015": _prepared_subset(
                exact_x, labels, rows, strength_mask
            ),
        }

        print(
            f"{family} {n_qubits}q: supervised={len(rows)} "
            f"classes={dict(Counter(labels.tolist()))}"
        )
    print()

    results: dict[str, Any] = {
        "analysis": "step9d_raw_training_coverage_v1",
        "scientific_boundary": {
            "posthoc_only": True,
            "qpu_submission": False,
            "retraining": False,
            "weight_change": False,
            "threshold_change": False,
            "confirmatory_interpretation": False,
        },
        "step5_product_id": product.name,
        "distance_definition": (
            "Euclidean distance after per-subset feature standardization; "
            "dimensions with training std <= 1e-10 are excluded."
        ),
        "ood_definition": (
            "Percentile of query nearest-neighbor distance relative to Step-5 "
            "leave-one-out nearest-neighbor distances in the same standardized subset."
        ),
        "cases": {},
    }

    for case in distorted_cases:
        key = (case.family, case.reference_circuit.num_qubits)
        data = family_data[key]
        expected = str(case.expected_mechanism)
        if expected not in MECHANISMS:
            raise RuntimeError(f"unexpected mechanism: {expected}")

        qpu_row = qpu_by_case[case.case_id]
        hardware = _hardware_vector(
            qpu_row, case.reference_circuit.num_qubits
        )
        ideal_arrays = _ideal_basis_arrays(
            case.reference_circuit, case.observed_circuit
        )
        ideal = _vector(
            ideal_arrays["local"],
            ideal_arrays["pair"],
            ideal_arrays["parity"],
        )

        if hardware.shape != data["finite_x"].shape[1:]:
            raise RuntimeError(
                f"{case.case_id} hardware dimension mismatch: "
                f"{hardware.shape} vs {data['finite_x'].shape[1:]}"
            )
        if ideal.shape != data["exact_x"].shape[1:]:
            raise RuntimeError(
                f"{case.case_id} ideal dimension mismatch: "
                f"{ideal.shape} vs {data['exact_x'].shape[1:]}"
            )

        probes = {
            "hardware_vs_finite_all": _evaluate_query(
                data["finite_all"], hardware, expected
            ),
            "hardware_vs_finite_strength_0_15": (
                _evaluate_query(
                    data["finite_strength_015"], hardware, expected
                )
                if data["finite_strength_015"] is not None
                else None
            ),
            "hardware_vs_finite_strength_0_15_shots_4096": (
                _evaluate_query(
                    data["finite_strength_015_shots_4096"],
                    hardware,
                    expected,
                )
                if data["finite_strength_015_shots_4096"] is not None
                else None
            ),
            "ideal_vs_exact_all": _evaluate_query(
                data["exact_all"], ideal, expected
            ),
            "ideal_vs_exact_strength_0_15": (
                _evaluate_query(
                    data["exact_strength_015"], ideal, expected
                )
                if data["exact_strength_015"] is not None
                else None
            ),
        }

        results["cases"][case.case_id] = {
            "family": case.family,
            "n_qubits": int(case.reference_circuit.num_qubits),
            "expected_mechanism": expected,
            "strength": float(case.strength),
            "hardware_vector_l2": float(np.linalg.norm(hardware)),
            "ideal_vector_l2": float(np.linalg.norm(ideal)),
            "hardware_vs_ideal_cosine": float(
                np.dot(hardware, ideal)
                / (np.linalg.norm(hardware) * np.linalg.norm(ideal))
            ),
            "probes": probes,
        }

        print(case.case_id)
        print("  expected:", expected)
        print(_format_probe("hardware/finite all", probes["hardware_vs_finite_all"]))
        print(
            _format_probe(
                "hardware/finite s=.15",
                probes["hardware_vs_finite_strength_0_15"],
            )
        )
        print(
            _format_probe(
                "hardware/finite s=.15 shots=4096",
                probes["hardware_vs_finite_strength_0_15_shots_4096"],
            )
        )
        print(_format_probe("ideal/exact all", probes["ideal_vs_exact_all"]))
        print(
            _format_probe(
                "ideal/exact s=.15",
                probes["ideal_vs_exact_strength_0_15"],
            )
        )
        print()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("wrote:", args.output_json)


if __name__ == "__main__":
    main()
