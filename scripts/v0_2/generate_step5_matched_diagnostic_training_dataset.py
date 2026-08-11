#!/usr/bin/env python3
"""Generate Step 5 matched TriQTO diagnostic training cohorts.

The generator creates train/validation development data only.  Each independent
clean circuit root supplies an intended/reference circuit graph.  Hidden
simulator perturbations (RZ/RX/RY) are inserted only in the execution used to
produce observed measurement evidence; the perturbation gate is never exposed
in the deployable graph input.

Primary diagnostic inputs are empirical finite-shot paired-reference deltas for
local same-basis Pauli expectations, all same-basis two-body correlations, and
global basis parity.  Exact statevectors are transient generation machinery and
are never persisted in deployable example artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


SCHEMA = "triqto.v0_2.step5_matched_diagnostic_training_dataset.v1"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/v0_2/step5_matched_diagnostic_training_dataset.json"
)
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step5_matched_diagnostic_training")
BASIS_ORDER = ("Z", "X", "Y")
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
MECHANISM_CODES = {"rz_drift": 0, "rx_overrotation": 1, "ry_overrotation": 2}
PHENOTYPE_CODES = {
    "phase_dominant": 0,
    "mixed": 1,
    "population_dominant": 2,
    "negligible": 3,
    "clean_control": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--clean-circuit-roots", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts: Any) -> int:
    payload = canonical_json([str(part) for part in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def split_for_root(root_index: int) -> str:
    return "validation" if root_index % 5 == 0 else "train"


def family_for_root(root_index: int, config: Mapping[str, Any]) -> str:
    cycle = list(config["clean_circuit_generation"]["family_cycle"])
    return str(cycle[root_index % len(cycle)])


def choose_n_qubits(root_index: int, family: str, config: Mapping[str, Any]) -> int:
    if family == "bell_like" and bool(
        config["clean_circuit_generation"]["bell_like_forces_two_qubits"]
    ):
        return 2
    choices = [int(value) for value in config["clean_circuit_generation"]["qubit_choices"]]
    rng = np.random.default_rng(
        stable_seed(config["clean_circuit_generation"]["base_seed"], "nq", root_index)
    )
    return int(rng.choice(choices))


def rand_angle(rng: np.random.Generator, scale: float = math.pi) -> float:
    return float(rng.uniform(-scale, scale))


def build_clean_circuit(
    root_index: int, family: str, n_qubits: int, config: Mapping[str, Any]
) -> QuantumCircuit:
    base_seed = int(config["clean_circuit_generation"]["base_seed"])
    rng = np.random.default_rng(stable_seed(base_seed, "circuit", root_index, family, n_qubits))
    circuit = QuantumCircuit(n_qubits)

    if family == "bell_like":
        if n_qubits != 2:
            raise ValueError("bell_like requires two qubits")
        circuit.ry(rand_angle(rng, 0.7), 0)
        circuit.rz(rand_angle(rng, 0.7), 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(rand_angle(rng, 1.1), 0)
        circuit.ry(rand_angle(rng, 1.1), 1)

    elif family == "ghz":
        circuit.ry(rand_angle(rng, 0.35), 0)
        circuit.h(0)
        for qubit in range(1, n_qubits):
            circuit.cx(qubit - 1, qubit)
        for qubit in range(n_qubits):
            circuit.rz(rand_angle(rng, 0.55), qubit)
        circuit.ry(rand_angle(rng, 0.35), int(rng.integers(0, n_qubits)))

    elif family == "hardware_efficient_ansatz":
        for _layer in range(2):
            for qubit in range(n_qubits):
                circuit.ry(rand_angle(rng), qubit)
                circuit.rz(rand_angle(rng), qubit)
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            if n_qubits > 2:
                circuit.cx(n_qubits - 1, 0)

    elif family == "phase_interference":
        for qubit in range(n_qubits):
            circuit.h(qubit)
        for qubit in range(n_qubits):
            circuit.rz(rand_angle(rng), qubit)
        for qubit in range(n_qubits - 1):
            circuit.cz(qubit, qubit + 1)
        for qubit in range(n_qubits):
            if (qubit + root_index) % 2 == 0:
                circuit.h(qubit)
            else:
                circuit.ry(rand_angle(rng, 0.8), qubit)
        circuit.rz(rand_angle(rng, 1.2), int(rng.integers(0, n_qubits)))

    elif family == "qaoa_like":
        for qubit in range(n_qubits):
            circuit.h(qubit)
        for _layer in range(2):
            gamma = rand_angle(rng, 0.9)
            beta = rand_angle(rng, 0.9)
            for qubit in range(n_qubits - 1):
                circuit.rzz(gamma * float(rng.uniform(0.7, 1.3)), qubit, qubit + 1)
            if n_qubits > 2:
                circuit.rzz(gamma * float(rng.uniform(0.7, 1.3)), n_qubits - 1, 0)
            for qubit in range(n_qubits):
                circuit.rx(beta * float(rng.uniform(0.7, 1.3)), qubit)

    elif family == "qft_like":
        for qubit in range(n_qubits):
            circuit.rz(rand_angle(rng, 0.4), qubit)
        for target in range(n_qubits):
            circuit.h(target)
            for control in range(target + 1, n_qubits):
                distance = control - target
                angle = math.pi / float(2**distance)
                angle *= float(rng.uniform(0.85, 1.15))
                circuit.cp(angle, control, target)
        for qubit in range(n_qubits // 2):
            circuit.swap(qubit, n_qubits - qubit - 1)

    elif family == "random_shallow":
        for qubit in range(n_qubits):
            if rng.random() < 0.7:
                circuit.h(qubit)
        one_qubit = ("rx", "ry", "rz")
        event_count = max(8, 3 * n_qubits)
        for _ in range(event_count):
            if n_qubits > 1 and rng.random() < 0.35:
                left = int(rng.integers(0, n_qubits))
                right = int(rng.integers(0, n_qubits - 1))
                if right >= left:
                    right += 1
                if rng.random() < 0.5:
                    circuit.cx(left, right)
                else:
                    circuit.cz(left, right)
            else:
                qubit = int(rng.integers(0, n_qubits))
                gate = str(rng.choice(one_qubit))
                angle = rand_angle(rng)
                getattr(circuit, gate)(angle, qubit)
    else:
        raise ValueError(f"unsupported clean circuit family {family!r}")

    if len(circuit.data) < int(config["clean_circuit_generation"]["minimum_unitary_events"]):
        raise RuntimeError("generated clean circuit is too short")
    return circuit


def serialize_graph(circuit: QuantumCircuit) -> dict[str, np.ndarray]:
    gate_names: list[str] = []
    qubit_indices: list[int] = []
    qubit_ptr = [0]
    param_sin: list[float] = []
    param_cos: list[float] = []
    param_ptr = [0]

    for instruction in circuit.data:
        operation = instruction.operation
        gate_names.append(str(operation.name).lower())
        for qubit in instruction.qubits:
            qubit_indices.append(int(circuit.find_bit(qubit).index))
        qubit_ptr.append(len(qubit_indices))
        for parameter in operation.params:
            value = float(parameter)
            param_sin.append(float(math.sin(value)))
            param_cos.append(float(math.cos(value)))
        param_ptr.append(len(param_sin))

    max_name = max(1, max(len(name) for name in gate_names))
    return {
        "x__graph_gate_names": np.asarray(gate_names, dtype=f"<U{max_name}"),
        "x__graph_gate_qubit_ptr": np.asarray(qubit_ptr, dtype=np.int32),
        "x__graph_gate_qubit_indices": np.asarray(qubit_indices, dtype=np.int16),
        "x__graph_gate_parameter_ptr": np.asarray(param_ptr, dtype=np.int32),
        "x__graph_gate_parameter_sin": np.asarray(param_sin, dtype=np.float64),
        "x__graph_gate_parameter_cos": np.asarray(param_cos, dtype=np.float64),
    }


def graph_hash(graph: Mapping[str, np.ndarray]) -> str:
    payload = {
        "names": [str(value) for value in graph["x__graph_gate_names"].tolist()],
        "qptr": [int(value) for value in graph["x__graph_gate_qubit_ptr"].tolist()],
        "qidx": [int(value) for value in graph["x__graph_gate_qubit_indices"].tolist()],
        "pptr": [int(value) for value in graph["x__graph_gate_parameter_ptr"].tolist()],
        "psin": [round(float(value), 12) for value in graph["x__graph_gate_parameter_sin"].tolist()],
        "pcos": [round(float(value), 12) for value in graph["x__graph_gate_parameter_cos"].tolist()],
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def append_instruction_copy(
    output: QuantumCircuit, source: QuantumCircuit, instruction: Any
) -> None:
    qargs = [output.qubits[source.find_bit(qubit).index] for qubit in instruction.qubits]
    if instruction.clbits:
        raise ValueError("clean Step 5 circuit unexpectedly contains classical bits")
    output.append(instruction.operation, qargs, [])


def inject_hidden_rotation(
    clean: QuantumCircuit,
    boundary_rank: int,
    affected_qubit: int,
    mechanism: str,
    strength: float,
) -> QuantumCircuit:
    if not (1 <= boundary_rank <= len(clean.data)):
        raise ValueError("insertion boundary outside clean circuit")
    observed = QuantumCircuit(clean.num_qubits)
    for instruction in clean.data[:boundary_rank]:
        append_instruction_copy(observed, clean, instruction)
    if mechanism == "rz_drift":
        observed.rz(strength, affected_qubit)
    elif mechanism == "rx_overrotation":
        observed.rx(strength, affected_qubit)
    elif mechanism == "ry_overrotation":
        observed.ry(strength, affected_qubit)
    else:
        raise ValueError(f"unsupported mechanism {mechanism!r}")
    for instruction in clean.data[boundary_rank:]:
        append_instruction_copy(observed, clean, instruction)
    return observed


def normalized_state(circuit: QuantumCircuit) -> np.ndarray:
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid statevector norm")
    return state / norm


def basis_probabilities(state: np.ndarray, n_qubits: int, basis: str) -> np.ndarray:
    transform = QuantumCircuit(n_qubits)
    if basis == "X":
        for qubit in range(n_qubits):
            transform.h(qubit)
    elif basis == "Y":
        for qubit in range(n_qubits):
            transform.sdg(qubit)
            transform.h(qubit)
    elif basis != "Z":
        raise ValueError(f"unknown basis {basis!r}")
    rotated = np.asarray(Statevector(state).evolve(transform).data, dtype=np.complex128)
    probabilities = np.abs(rotated) ** 2
    probabilities = np.asarray(probabilities / np.sum(probabilities), dtype=np.float64)
    return probabilities


def eigenvalue_table(n_qubits: int) -> np.ndarray:
    indices = np.arange(1 << n_qubits, dtype=np.int64)[:, None]
    bits = (indices >> np.arange(n_qubits, dtype=np.int64)[None, :]) & 1
    return 1.0 - 2.0 * bits.astype(np.float64)


def pair_indices(n_qubits: int) -> np.ndarray:
    return np.asarray(
        [(left, right) for left in range(n_qubits) for right in range(left + 1, n_qubits)],
        dtype=np.int16,
    ).reshape(-1, 2)


def stats_from_distribution(
    probabilities: np.ndarray, eig: np.ndarray, pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    local = np.asarray(probabilities @ eig, dtype=np.float64)
    if len(pairs):
        pair_products = eig[:, pairs[:, 0]] * eig[:, pairs[:, 1]]
        pairwise = np.asarray(probabilities @ pair_products, dtype=np.float64)
    else:
        pairwise = np.empty((0,), dtype=np.float64)
    parity = float(probabilities @ np.prod(eig, axis=1))
    return local, pairwise, parity


def empirical_stats(
    probabilities: np.ndarray,
    shots: int,
    eig: np.ndarray,
    pairs: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(int(shots), probabilities)
    empirical = counts.astype(np.float64) / float(shots)
    return stats_from_distribution(empirical, eig, pairs)


def state_diagnostics(
    clean_state: np.ndarray,
    distorted_state: np.ndarray,
    *,
    epsilon: float,
    negligible_floor: float,
    dominance_ratio: float,
) -> dict[str, Any]:
    clean_prob = np.abs(clean_state) ** 2
    distorted_prob = np.abs(distorted_state) ** 2
    bc = float(np.sum(np.sqrt(clean_prob * distorted_prob)))
    quantum_overlap = float(abs(np.vdot(clean_state, distorted_state)))
    population = max(0.0, 1.0 - bc)
    phase = max(0.0, bc - quantum_overlap)
    total = max(0.0, 1.0 - quantum_overlap)
    log_ratio = float(math.log((phase + epsilon) / (population + epsilon)))
    if total < negligible_floor:
        phenotype = "negligible"
    elif phase >= dominance_ratio * population:
        phenotype = "phase_dominant"
    elif population >= dominance_ratio * phase:
        phenotype = "population_dominant"
    else:
        phenotype = "mixed"
    return {
        "population_component": population,
        "phase_component": phase,
        "dominance_log_ratio": log_ratio,
        "total_overlap_loss": total,
        "phenotype": phenotype,
        "effect_present": bool(total >= negligible_floor),
    }


def depth_boundary(unitary_count: int, fraction: float) -> int:
    return max(1, min(unitary_count, int(math.ceil(fraction * unitary_count))))


def depth_bin(boundary: int, unitary_count: int) -> str:
    if boundary == unitary_count:
        return "terminal"
    fraction = float(boundary) / float(unitary_count)
    if fraction <= 0.34:
        return "early"
    if fraction <= 0.67:
        return "middle"
    return "late"


def make_reference_bundle(
    clean_basis_probs: Mapping[str, np.ndarray],
    n_qubits: int,
    shots: int,
    root_index: int,
    context_key: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, float]], np.ndarray]:
    eig = eigenvalue_table(n_qubits)
    pairs = pair_indices(n_qubits)
    result: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for basis in BASIS_ORDER:
        result[basis] = empirical_stats(
            clean_basis_probs[basis],
            shots,
            eig,
            pairs,
            stable_seed("step5", "reference", root_index, context_key, basis, shots),
        )
    return result, pairs


def exact_basis_stats(
    state: np.ndarray, n_qubits: int, pairs: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    eig = eigenvalue_table(n_qubits)
    output: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for basis in BASIS_ORDER:
        output[basis] = stats_from_distribution(
            basis_probabilities(state, n_qubits, basis), eig, pairs
        )
    return output


def diagnostic_arrays(
    observed_state: np.ndarray,
    clean_basis_probs: Mapping[str, np.ndarray],
    reference_bundle: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    n_qubits: int,
    pairs: np.ndarray,
    shots: int,
    root_index: int,
    context_key: str,
    mechanism: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    eig = eigenvalue_table(n_qubits)
    delta_local: list[np.ndarray] = []
    delta_pairwise: list[np.ndarray] = []
    delta_parity: list[float] = []
    exact_delta_local: list[np.ndarray] = []
    exact_delta_pairwise: list[np.ndarray] = []
    exact_delta_parity: list[float] = []

    for basis in BASIS_ORDER:
        observed_prob = basis_probabilities(observed_state, n_qubits, basis)
        observed_empirical = empirical_stats(
            observed_prob,
            shots,
            eig,
            pairs,
            stable_seed(
                "step5", "observed", root_index, context_key, mechanism, basis, shots
            ),
        )
        reference_empirical = reference_bundle[basis]
        observed_exact = stats_from_distribution(observed_prob, eig, pairs)
        reference_exact = stats_from_distribution(clean_basis_probs[basis], eig, pairs)

        delta_local.append(observed_empirical[0] - reference_empirical[0])
        delta_pairwise.append(observed_empirical[1] - reference_empirical[1])
        delta_parity.append(observed_empirical[2] - reference_empirical[2])
        exact_delta_local.append(observed_exact[0] - reference_exact[0])
        exact_delta_pairwise.append(observed_exact[1] - reference_exact[1])
        exact_delta_parity.append(observed_exact[2] - reference_exact[2])

    primary = {
        "x__diagnostic_basis_codes": np.asarray([0, 1, 2], dtype=np.int8),
        "x__delta_local_expectations": np.asarray(delta_local, dtype=np.float64),
        "x__pair_indices": np.asarray(pairs, dtype=np.int16),
        "x__delta_pairwise_correlations": np.asarray(delta_pairwise, dtype=np.float64),
        "x__delta_global_parity": np.asarray(delta_parity, dtype=np.float64),
        "x__observed_shots": np.full(3, int(shots), dtype=np.int32),
        "x__reference_shots": np.full(3, int(shots), dtype=np.int32),
        "x__reference_available_mask": np.ones(3, dtype=np.bool_),
        "x__reference_kind_code": np.asarray([0], dtype=np.int8),
    }
    audit = {
        "audit__exact_delta_local_expectations": np.asarray(
            exact_delta_local, dtype=np.float64
        ),
        "audit__exact_delta_pairwise_correlations": np.asarray(
            exact_delta_pairwise, dtype=np.float64
        ),
        "audit__exact_delta_global_parity": np.asarray(
            exact_delta_parity, dtype=np.float64
        ),
    }
    return primary, audit


def save_example(
    path: Path,
    *,
    graph: Mapping[str, np.ndarray],
    n_qubits: int,
    diagnostic: Mapping[str, np.ndarray],
    audit_diagnostic: Mapping[str, np.ndarray],
    example_id: str,
    clean_group_id: str,
    clean_control: bool,
    effect_present: bool,
    mechanism_code: int,
    mechanism_loss_mask: bool,
    phenotype: str,
    continuous: Mapping[str, float],
    affected_qubit: int,
    boundary: int,
    strength: float,
) -> None:
    arrays: dict[str, np.ndarray] = {key: np.asarray(value) for key, value in graph.items()}
    arrays.update({key: np.asarray(value) for key, value in diagnostic.items()})
    arrays.update({key: np.asarray(value) for key, value in audit_diagnostic.items()})
    arrays.update(
        {
            "x__layout_logical_to_physical": np.arange(n_qubits, dtype=np.int16),
            "y__clean_control_target": np.asarray([clean_control], dtype=np.bool_),
            "y__effect_present_target": np.asarray([effect_present], dtype=np.bool_),
            "y__mechanism_target": np.asarray([mechanism_code], dtype=np.int8),
            "y__mechanism_loss_mask": np.asarray([mechanism_loss_mask], dtype=np.bool_),
            "y__phenomenology_target": np.asarray(
                [PHENOTYPE_CODES[phenotype]], dtype=np.int8
            ),
            "y__population_component": np.asarray(
                [float(continuous["population_component"])], dtype=np.float64
            ),
            "y__phase_component": np.asarray(
                [float(continuous["phase_component"])], dtype=np.float64
            ),
            "y__dominance_log_ratio": np.asarray(
                [float(continuous["dominance_log_ratio"])], dtype=np.float64
            ),
            "y__total_overlap_loss": np.asarray(
                [float(continuous["total_overlap_loss"])], dtype=np.float64
            ),
            "audit__affected_qubit": np.asarray([affected_qubit], dtype=np.int16),
            "audit__insertion_boundary_rank": np.asarray([boundary], dtype=np.int32),
            "audit__strength": np.asarray([strength], dtype=np.float64),
            "meta__example_id": np.asarray([example_id], dtype=f"<U{len(example_id)}"),
            "meta__clean_circuit_group_id": np.asarray(
                [clean_group_id], dtype=f"<U{len(clean_group_id)}"
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def validate_array_contract(arrays: Mapping[str, np.ndarray], bound: float) -> None:
    forbidden_x_tokens = (
        "mechanism",
        "phenomenology",
        "effect_present",
        "population_component",
        "phase_component",
        "overlap",
        "affected_qubit",
        "insertion",
        "strength",
        "statevector",
    )
    for key, value in arrays.items():
        if key.startswith("x__") and any(token in key for token in forbidden_x_tokens):
            raise RuntimeError(f"privileged/target-derived deployable input name: {key}")
        if key.startswith("x__delta_"):
            numeric = np.asarray(value, dtype=np.float64)
            if not np.all(np.isfinite(numeric)):
                raise RuntimeError(f"non-finite diagnostic values in {key}")
            if numeric.size and float(np.max(np.abs(numeric))) > bound:
                raise RuntimeError(f"diagnostic bound exceeded in {key}")


def summarize_rows(
    roots: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    family_rows: list[dict[str, Any]] = []
    for family in sorted({str(row["family"]) for row in roots}):
        root_subset = [row for row in roots if str(row["family"]) == family]
        ex_subset = [row for row in examples if str(row["family"]) == family]
        family_rows.append(
            {
                "family": family,
                "clean_root_count": len(root_subset),
                "example_count": len(ex_subset),
                "effectful_example_count": sum(bool(row["effect_present"]) for row in ex_subset),
                "negligible_injected_count": sum(
                    (not bool(row["clean_control"])) and (not bool(row["effect_present"]))
                    for row in ex_subset
                ),
            }
        )

    split_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        root_subset = [row for row in roots if row["split"] == split]
        ex_subset = [row for row in examples if row["split"] == split]
        split_rows.append(
            {
                "split": split,
                "clean_root_count": len(root_subset),
                "example_count": len(ex_subset),
                "mechanism_supervised_count": sum(bool(row["mechanism_loss_mask"]) for row in ex_subset),
            }
        )

    mechanism_rows: list[dict[str, Any]] = []
    for mechanism in ("clean_control",) + MECHANISMS:
        subset = [row for row in examples if row["mechanism"] == mechanism]
        mechanism_rows.append(
            {
                "mechanism": mechanism,
                "example_count": len(subset),
                "effectful_count": sum(bool(row["effect_present"]) for row in subset),
                "mechanism_loss_active_count": sum(bool(row["mechanism_loss_mask"]) for row in subset),
            }
        )
    return family_rows, split_rows, mechanism_rows


def validate_stage(
    roots: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    root_count: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["stage_validation"]
    expected_per_root = int(policy["expected_examples_per_clean_root"])
    expected_examples = root_count * expected_per_root
    if len(roots) != root_count:
        raise RuntimeError("clean root count mismatch")
    if len(examples) != expected_examples:
        raise RuntimeError(
            f"example count mismatch: {len(examples)} != {expected_examples}"
        )

    group_to_splits: dict[str, set[str]] = defaultdict(set)
    group_to_graphs: dict[str, set[str]] = defaultdict(set)
    group_to_counts: Counter[str] = Counter()
    for row in examples:
        group = str(row["clean_circuit_group_id"])
        group_to_splits[group].add(str(row["split"]))
        group_to_graphs[group].add(str(row["graph_sha256"]))
        group_to_counts[group] += 1
        if bool(row["clean_control"]) and bool(row["mechanism_loss_mask"]):
            raise RuntimeError("clean control has mechanism loss enabled")
        if (not bool(row["effect_present"])) and bool(row["mechanism_loss_mask"]):
            raise RuntimeError("negligible example has mechanism loss enabled")
    if any(len(values) != 1 for values in group_to_splits.values()):
        raise RuntimeError("clean circuit group crosses train/validation split")
    if any(len(values) != 1 for values in group_to_graphs.values()):
        raise RuntimeError("root derivatives do not share exactly one graph hash")
    if any(count != expected_per_root for count in group_to_counts.values()):
        raise RuntimeError("root derivative count mismatch")

    root_hashes = [str(row["graph_sha256"]) for row in roots]
    if len(set(root_hashes)) != len(root_hashes):
        raise RuntimeError("duplicate clean circuit graph detected")

    mechanism_counts = Counter(str(row["mechanism"]) for row in examples)
    expected_mechanism = root_count * int(
        config["matched_intervention_design"]["contexts_per_clean_root"]
    )
    for mechanism in MECHANISMS:
        if mechanism_counts[mechanism] != expected_mechanism:
            raise RuntimeError(f"mechanism count mismatch for {mechanism}")
    if mechanism_counts["clean_control"] != root_count:
        raise RuntimeError("clean control count mismatch")

    train_roots = sum(row["split"] == "train" for row in roots)
    validation_roots = sum(row["split"] == "validation" for row in roots)
    if train_roots != root_count - root_count // 5:
        raise RuntimeError("unexpected train root count")
    if validation_roots != root_count // 5:
        raise RuntimeError("unexpected validation root count")

    if root_count == 500:
        family_counts = Counter(str(row["family"]) for row in roots)
        if min(family_counts.values()) < int(policy["minimum_family_root_count_at_500_stage"]):
            raise RuntimeError("500-root family coverage below frozen minimum")
        non_bell_n = Counter(
            int(row["n_qubits"])
            for row in roots
            if str(row["family"]) != "bell_like"
        )
        for n_qubits in config["clean_circuit_generation"]["qubit_choices"]:
            if non_bell_n[int(n_qubits)] < int(
                policy["minimum_each_non_bell_qubit_count_at_500_stage"]
            ):
                raise RuntimeError("500-root qubit-count coverage below frozen minimum")
        for mechanism in MECHANISMS:
            if mechanism_counts[mechanism] < int(
                policy["minimum_each_mechanism_count_at_500_stage"]
            ):
                raise RuntimeError("500-root mechanism coverage below frozen minimum")

    return {
        "status": "PASS",
        "clean_root_count": root_count,
        "example_count": len(examples),
        "train_clean_root_count": train_roots,
        "validation_clean_root_count": validation_roots,
        "clean_group_cross_split_count": 0,
        "duplicate_clean_graph_count": 0,
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "mechanism_supervised_example_count": sum(
            bool(row["mechanism_loss_mask"]) for row in examples
        ),
        "negligible_injected_example_count": sum(
            (not bool(row["clean_control"])) and (not bool(row["effect_present"]))
            for row in examples
        ),
        "clean_control_count": mechanism_counts["clean_control"],
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected Step 5 config schema")

    root_count = int(args.clean_circuit_roots)
    allowed = [int(value) for value in config["stage_progression"]["allowed_root_counts"]]
    if root_count not in allowed:
        raise ValueError(f"--clean-circuit-roots must be one of {allowed}")

    runner_path = Path(__file__).resolve()
    identity = {
        "schema": SCHEMA,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner_path),
        "clean_circuit_root_count": root_count,
    }
    product_id = "product_" + hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    product_root = output_parent / product_id
    if product_root.exists():
        raise RuntimeError(f"refusing to overwrite existing Step 5 product: {product_root}")
    staging = output_parent / f".{product_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    root_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    seen_graph_hashes: set[str] = set()
    bound = float(config["stage_validation"]["diagnostic_delta_absolute_bound"])
    phenomenology_cfg = config["privileged_supervision"]
    epsilon = float(phenomenology_cfg["epsilon"])
    negligible_floor = float(phenomenology_cfg["negligible_overlap_loss_floor"])
    dominance_ratio = float(phenomenology_cfg["phenomenology_strong_dominance_ratio"])
    depths = [float(value) for value in config["matched_intervention_design"]["depth_target_fractions"]]
    strengths = [float(value) for value in config["matched_intervention_design"]["strength_schedule_by_context"]]
    shots_cycle = [int(value) for value in config["finite_shot_acquisition"]["shots_cycle"]]

    try:
        for root_index in range(root_count):
            family = family_for_root(root_index, config)
            n_qubits = choose_n_qubits(root_index, family, config)
            clean = build_clean_circuit(root_index, family, n_qubits, config)
            graph = serialize_graph(clean)
            ghash = graph_hash(graph)
            if ghash in seen_graph_hashes:
                raise RuntimeError(f"duplicate generated clean graph at root {root_index}")
            seen_graph_hashes.add(ghash)
            clean_group_id = sha256_bytes(
                canonical_json(
                    {
                        "root_index": root_index,
                        "family": family,
                        "n_qubits": n_qubits,
                        "graph_sha256": ghash,
                    }
                ).encode("utf-8")
            )
            split = split_for_root(root_index)
            clean_state = normalized_state(clean)
            pairs = pair_indices(n_qubits)
            clean_basis_probs = {
                basis: basis_probabilities(clean_state, n_qubits, basis)
                for basis in BASIS_ORDER
            }
            layout_identity = "identity:" + ",".join(str(i) for i in range(n_qubits))

            root_rows.append(
                {
                    "root_index": root_index,
                    "clean_circuit_group_id": clean_group_id,
                    "split": split,
                    "family": family,
                    "n_qubits": n_qubits,
                    "unitary_event_count": len(clean.data),
                    "graph_sha256": ghash,
                    "physical_layout_identity": layout_identity,
                }
            )

            # One clean/no-distortion control.  Observed and reference are independent
            # finite-shot samples of the same clean state, so B_delta contains realistic
            # shot noise around zero rather than an artificial exact-zero fingerprint.
            clean_shots = shots_cycle[root_index % len(shots_cycle)]
            clean_context_key = "clean_control"
            reference_bundle, pairs = make_reference_bundle(
                clean_basis_probs,
                n_qubits,
                clean_shots,
                root_index,
                clean_context_key,
            )
            diagnostic, audit_diag = diagnostic_arrays(
                clean_state,
                clean_basis_probs,
                reference_bundle,
                n_qubits,
                pairs,
                clean_shots,
                root_index,
                clean_context_key,
                "clean_control",
            )
            example_id = sha256_bytes(
                canonical_json([clean_group_id, "clean_control"]).encode("utf-8")
            )
            zero_targets = {
                "population_component": 0.0,
                "phase_component": 0.0,
                "dominance_log_ratio": 0.0,
                "total_overlap_loss": 0.0,
            }
            artifact_rel = Path("artifacts") / split / f"{example_id.split(':',1)[1]}.npz"
            artifact_path = staging / artifact_rel
            save_example(
                artifact_path,
                graph=graph,
                n_qubits=n_qubits,
                diagnostic=diagnostic,
                audit_diagnostic=audit_diag,
                example_id=example_id,
                clean_group_id=clean_group_id,
                clean_control=True,
                effect_present=False,
                mechanism_code=-1,
                mechanism_loss_mask=False,
                phenotype="clean_control",
                continuous=zero_targets,
                affected_qubit=-1,
                boundary=-1,
                strength=0.0,
            )
            with np.load(artifact_path, allow_pickle=False) as loaded:
                validate_array_contract(dict(loaded), bound)
            example_rows.append(
                {
                    "example_id": example_id,
                    "root_index": root_index,
                    "clean_circuit_group_id": clean_group_id,
                    "split": split,
                    "family": family,
                    "n_qubits": n_qubits,
                    "clean_control": True,
                    "mechanism": "clean_control",
                    "effect_present": False,
                    "mechanism_loss_mask": False,
                    "phenomenology": "clean_control",
                    "affected_qubit": -1,
                    "insertion_boundary_rank": -1,
                    "insertion_depth_bin": "clean_control",
                    "strength": 0.0,
                    "shots": clean_shots,
                    "reference_kind": config["finite_shot_acquisition"]["reference_kind"],
                    "backend_identity": config["finite_shot_acquisition"]["simulation_backend_identity"],
                    "physical_layout_identity": layout_identity,
                    "reference_window_id": f"root{root_index}:clean",
                    "graph_sha256": ghash,
                    "artifact_path": artifact_rel.as_posix(),
                    "artifact_sha256": sha256_file(artifact_path),
                }
            )

            for context_index, (fraction, strength) in enumerate(zip(depths, strengths)):
                boundary = depth_boundary(len(clean.data), fraction)
                affected_qubit = int((root_index + context_index) % n_qubits)
                shots = shots_cycle[(root_index + context_index) % len(shots_cycle)]
                context_key = f"ctx{context_index}:q{affected_qubit}:b{boundary}:s{strength:.12g}"
                reference_bundle, pairs = make_reference_bundle(
                    clean_basis_probs,
                    n_qubits,
                    shots,
                    root_index,
                    context_key,
                )

                for mechanism in MECHANISMS:
                    observed = inject_hidden_rotation(
                        clean,
                        boundary,
                        affected_qubit,
                        mechanism,
                        strength,
                    )
                    observed_state = normalized_state(observed)
                    truth = state_diagnostics(
                        clean_state,
                        observed_state,
                        epsilon=epsilon,
                        negligible_floor=negligible_floor,
                        dominance_ratio=dominance_ratio,
                    )
                    diagnostic, audit_diag = diagnostic_arrays(
                        observed_state,
                        clean_basis_probs,
                        reference_bundle,
                        n_qubits,
                        pairs,
                        shots,
                        root_index,
                        context_key,
                        mechanism,
                    )
                    mechanism_loss_mask = bool(truth["effect_present"])
                    example_id = sha256_bytes(
                        canonical_json(
                            [
                                clean_group_id,
                                context_index,
                                affected_qubit,
                                boundary,
                                strength,
                                mechanism,
                            ]
                        ).encode("utf-8")
                    )
                    artifact_rel = (
                        Path("artifacts")
                        / split
                        / f"{example_id.split(':',1)[1]}.npz"
                    )
                    artifact_path = staging / artifact_rel
                    save_example(
                        artifact_path,
                        graph=graph,
                        n_qubits=n_qubits,
                        diagnostic=diagnostic,
                        audit_diagnostic=audit_diag,
                        example_id=example_id,
                        clean_group_id=clean_group_id,
                        clean_control=False,
                        effect_present=bool(truth["effect_present"]),
                        mechanism_code=MECHANISM_CODES[mechanism],
                        mechanism_loss_mask=mechanism_loss_mask,
                        phenotype=str(truth["phenotype"]),
                        continuous=truth,
                        affected_qubit=affected_qubit,
                        boundary=boundary,
                        strength=strength,
                    )
                    with np.load(artifact_path, allow_pickle=False) as loaded:
                        validate_array_contract(dict(loaded), bound)
                    example_rows.append(
                        {
                            "example_id": example_id,
                            "root_index": root_index,
                            "clean_circuit_group_id": clean_group_id,
                            "split": split,
                            "family": family,
                            "n_qubits": n_qubits,
                            "clean_control": False,
                            "mechanism": mechanism,
                            "effect_present": bool(truth["effect_present"]),
                            "mechanism_loss_mask": mechanism_loss_mask,
                            "phenomenology": str(truth["phenotype"]),
                            "affected_qubit": affected_qubit,
                            "insertion_boundary_rank": boundary,
                            "insertion_depth_bin": depth_bin(boundary, len(clean.data)),
                            "strength": strength,
                            "shots": shots,
                            "reference_kind": config["finite_shot_acquisition"]["reference_kind"],
                            "backend_identity": config["finite_shot_acquisition"]["simulation_backend_identity"],
                            "physical_layout_identity": layout_identity,
                            "reference_window_id": f"root{root_index}:ctx{context_index}",
                            "graph_sha256": ghash,
                            "artifact_path": artifact_rel.as_posix(),
                            "artifact_sha256": sha256_file(artifact_path),
                        }
                    )

            if args.progress_every > 0 and (root_index + 1) % args.progress_every == 0:
                print(
                    f"Generated {root_index + 1}/{root_count} clean roots "
                    f"({len(example_rows)} examples)",
                    flush=True,
                )

        manifests = staging / "manifests"
        write_csv(manifests / "clean_circuit_manifest.csv", root_rows)
        write_csv(manifests / "example_manifest.csv", example_rows)
        family_rows, split_rows, mechanism_rows = summarize_rows(root_rows, example_rows)
        write_csv(manifests / "family_summary.csv", family_rows)
        write_csv(manifests / "split_summary.csv", split_rows)
        write_csv(manifests / "mechanism_summary.csv", mechanism_rows)

        validation = validate_stage(root_rows, example_rows, root_count, config)
        atomic_json(staging / "stage_validation.json", validation)

        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "product_id": product_id,
            "identity": identity,
            "clean_circuit_root_count": root_count,
            "example_count": len(example_rows),
            "train_clean_root_count": validation["train_clean_root_count"],
            "validation_clean_root_count": validation["validation_clean_root_count"],
            "train_example_count": next(
                int(row["example_count"]) for row in split_rows if row["split"] == "train"
            ),
            "validation_example_count": next(
                int(row["example_count"])
                for row in split_rows
                if row["split"] == "validation"
            ),
            "mechanism_counts": validation["mechanism_counts"],
            "mechanism_supervised_example_count": validation[
                "mechanism_supervised_example_count"
            ],
            "negligible_injected_example_count": validation[
                "negligible_injected_example_count"
            ],
            "clean_control_count": validation["clean_control_count"],
            "selected_diagnostic_variant": config["deployable_diagnostic_input"][
                "selected_step4_1_variant"
            ],
            "primary_input_is_empirical_finite_shot": True,
            "statevectors_persisted_in_example_artifacts": False,
            "historical_v0_1_test_accessed": False,
            "spent_confirmatory_cohort_accessed": False,
            "classifier_trained": False,
            "model_architecture_changed": False,
            "manifest_hashes": {
                "clean_circuit_manifest.csv": sha256_file(
                    manifests / "clean_circuit_manifest.csv"
                ),
                "example_manifest.csv": sha256_file(manifests / "example_manifest.csv"),
                "family_summary.csv": sha256_file(manifests / "family_summary.csv"),
                "split_summary.csv": sha256_file(manifests / "split_summary.csv"),
                "mechanism_summary.csv": sha256_file(manifests / "mechanism_summary.csv"),
                "stage_validation.json": sha256_file(staging / "stage_validation.json"),
            },
            "scientific_boundary": (
                "Step 5 development dataset generation only. The hidden intervention is absent "
                "from deployable graph inputs; finite-shot diagnostics are simulator-sampled; "
                "no model-quality or hardware-robustness claim is made."
            ),
        }
        atomic_json(staging / "dataset_complete.json", completion)
        os.replace(staging, product_root)
        pointer = {
            "schema": "triqto.v0_2.step5_current_product.v1",
            "product_dir": str(product_root),
            "product_id": product_id,
            "clean_circuit_root_count": root_count,
            "dataset_complete_sha256": sha256_file(product_root / "dataset_complete.json"),
        }
        atomic_json(output_parent / "current_product.json", pointer)

        print("\nTRIQTO STEP 5 MATCHED DIAGNOSTIC TRAINING DATASET COMPLETE\n")
        print(f"Stage clean roots: {root_count}")
        print(f"Examples: {len(example_rows)}")
        print(
            f"Train/validation clean roots: {validation['train_clean_root_count']}/"
            f"{validation['validation_clean_root_count']}"
        )
        print(f"Mechanism counts: {validation['mechanism_counts']}")
        print(
            "Mechanism-supervised examples: "
            f"{validation['mechanism_supervised_example_count']}"
        )
        print(
            "Injected-but-negligible examples: "
            f"{validation['negligible_injected_example_count']}"
        )
        print(f"Clean controls: {validation['clean_control_count']}")
        print(
            "Selected diagnostic core: "
            f"{config['deployable_diagnostic_input']['selected_step4_1_variant']}"
        )
        print("Primary diagnostics: empirical finite-shot")
        print("Hidden intervention present in graph input: NO")
        print("Statevectors persisted in example artifacts: NO")
        print("Historical v0.1 test accessed: NO")
        print("Spent confirmatory cohort accessed: NO")
        print("Classifier trained: NO")
        print(f"Product: {product_root}")

    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
