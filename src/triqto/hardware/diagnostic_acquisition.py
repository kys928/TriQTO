"""Step-9B hardware-facing paired diagnostic acquisition.

This module preserves the frozen Step-7/8 inference semantics:

* graph evidence comes only from the intended/reference circuit;
* diagnostic basis order is Z, X, Y;
* Qiskit count bitstrings are reversed into logical q0, q1, ... order;
* signed diagnostics are observed minus paired reference;
* all same-basis logical-qubit pairs and global parity are retained.

The module can consume any SamplerV2-compatible object. Physical-hardware
submission remains governed by the explicit confirmation boundary in
``triqto.hardware.ibm_runtime``; Step 9B itself performs no QPU execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from triqto.model.contracts import GraphTensorBatch
from triqto.step7.contracts import DiagnosticTensorBatch, Step7ModelBatch
from triqto.step7.graph_adapter import _graph_features_for_example

BASIS_ORDER = ("Z", "X", "Y")
BASIS_CODES = {"Z": 0, "X": 1, "Y": 2}
PROGRAM_ORDER = (
    "reference_Z",
    "reference_X",
    "reference_Y",
    "observed_Z",
    "observed_X",
    "observed_Y",
)
MEASUREMENT_REGISTER = "meas"
REFERENCE_KIND_CODE = 0


@dataclass(frozen=True, slots=True)
class EmpiricalPauliStats:
    local: np.ndarray
    pairwise: np.ndarray
    parity: float
    shots: int


@dataclass(slots=True)
class HardwareDiagnosticAcquisition:
    """One paired acquisition already converted to the frozen model contract."""

    model_batch: Step7ModelBatch
    counts_by_program: dict[str, dict[str, int]]
    isa_circuits: tuple[QuantumCircuit, ...]
    initial_layout: tuple[int, ...]
    shots: int
    job_id: str | None


def _validate_unmeasured_circuit(circuit: QuantumCircuit, name: str) -> None:
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(f"{name} must be a QuantumCircuit")
    if circuit.num_qubits <= 0:
        raise ValueError(f"{name} must contain at least one qubit")
    if circuit.num_clbits:
        raise ValueError(f"{name} must not contain classical bits before Step-9B measurement")
    forbidden = {"measure", "reset"}
    present = {instruction.operation.name for instruction in circuit.data}
    if forbidden & present:
        raise ValueError(f"{name} contains pre-existing measurement/reset operations")
    if circuit.parameters:
        raise ValueError(f"{name} must have all parameters bound before hardware acquisition")


def all_pair_indices(n_qubits: int) -> np.ndarray:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    return np.asarray(
        [(left, right) for left in range(n_qubits) for right in range(left + 1, n_qubits)],
        dtype=np.int64,
    ).reshape(-1, 2)


def _logical_eigenvalues(bitstring: str, n_qubits: int) -> np.ndarray:
    """Map Qiskit count key ``c[n-1]...c[0]`` to logical q0,q1,... eigenvalues."""

    key = str(bitstring)
    if len(key) != n_qubits or any(char not in "01" for char in key):
        raise ValueError(f"invalid {n_qubits}-bit Sampler count key: {bitstring!r}")
    logical_bits = np.fromiter((int(char) for char in reversed(key)), dtype=np.int8)
    return 1.0 - 2.0 * logical_bits.astype(np.float64)


def empirical_stats_from_counts(
    counts: Mapping[str, int], n_qubits: int, pairs: np.ndarray | None = None
) -> EmpiricalPauliStats:
    """Compute local Z eigenvalues, all pair products, and global parity from counts."""

    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    pair_array = all_pair_indices(n_qubits) if pairs is None else np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pair_array.size:
        if int(pair_array.min()) < 0 or int(pair_array.max()) >= n_qubits:
            raise ValueError("pair index outside logical qubit range")
        if bool(np.any(pair_array[:, 0] >= pair_array[:, 1])):
            raise ValueError("pairs must use canonical ascending logical-qubit endpoints")

    total = 0
    local = np.zeros(n_qubits, dtype=np.float64)
    pairwise = np.zeros(len(pair_array), dtype=np.float64)
    parity = 0.0
    for bitstring, raw_count in counts.items():
        count = int(raw_count)
        if count < 0:
            raise ValueError("Sampler counts must be nonnegative")
        if count == 0:
            continue
        eig = _logical_eigenvalues(str(bitstring), n_qubits)
        total += count
        local += count * eig
        if len(pair_array):
            pairwise += count * eig[pair_array[:, 0]] * eig[pair_array[:, 1]]
        parity += count * float(np.prod(eig))
    if total <= 0:
        raise ValueError("Sampler counts must contain at least one realized shot")
    return EmpiricalPauliStats(
        local=local / float(total),
        pairwise=pairwise / float(total),
        parity=parity / float(total),
        shots=int(total),
    )


def paired_diagnostic_arrays(
    reference_counts: Mapping[str, Mapping[str, int]],
    observed_counts: Mapping[str, Mapping[str, int]],
    n_qubits: int,
) -> dict[str, np.ndarray]:
    """Convert paired Z/X/Y counts to the exact deployable Step-7 diagnostic arrays."""

    missing_reference = [basis for basis in BASIS_ORDER if basis not in reference_counts]
    missing_observed = [basis for basis in BASIS_ORDER if basis not in observed_counts]
    if missing_reference or missing_observed:
        raise ValueError(
            f"paired diagnostics require Z/X/Y counts; missing reference={missing_reference}, observed={missing_observed}"
        )
    pairs = all_pair_indices(n_qubits)
    delta_local: list[np.ndarray] = []
    delta_pair: list[np.ndarray] = []
    delta_parity: list[float] = []
    observed_shots: list[int] = []
    reference_shots: list[int] = []
    for basis in BASIS_ORDER:
        ref = empirical_stats_from_counts(reference_counts[basis], n_qubits, pairs)
        obs = empirical_stats_from_counts(observed_counts[basis], n_qubits, pairs)
        delta_local.append(obs.local - ref.local)
        delta_pair.append(obs.pairwise - ref.pairwise)
        delta_parity.append(obs.parity - ref.parity)
        observed_shots.append(obs.shots)
        reference_shots.append(ref.shots)

    return {
        "x__diagnostic_basis_codes": np.asarray([0, 1, 2], dtype=np.int8),
        "x__delta_local_expectations": np.asarray(delta_local, dtype=np.float64),
        "x__pair_indices": pairs.astype(np.int16, copy=False),
        "x__delta_pairwise_correlations": np.asarray(delta_pair, dtype=np.float64),
        "x__delta_global_parity": np.asarray(delta_parity, dtype=np.float64),
        "x__observed_shots": np.asarray(observed_shots, dtype=np.int32),
        "x__reference_shots": np.asarray(reference_shots, dtype=np.int32),
        "x__reference_available_mask": np.ones(3, dtype=np.bool_),
        "x__reference_kind_code": np.asarray([REFERENCE_KIND_CODE], dtype=np.int8),
    }


def serialize_intended_graph(
    circuit: QuantumCircuit, logical_to_physical: Sequence[int]
) -> dict[str, np.ndarray]:
    """Serialize the intended circuit in the same event representation used by Step 5."""

    _validate_unmeasured_circuit(circuit, "reference circuit")
    layout = tuple(int(value) for value in logical_to_physical)
    if len(layout) != circuit.num_qubits or len(set(layout)) != len(layout) or min(layout) < 0:
        raise ValueError("logical_to_physical must contain one unique nonnegative physical qubit per logical qubit")

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
            if not math.isfinite(value):
                raise ValueError("circuit parameter must be finite")
            param_sin.append(float(math.sin(value)))
            param_cos.append(float(math.cos(value)))
        param_ptr.append(len(param_sin))
    max_name = max(1, max((len(name) for name in gate_names), default=1))
    return {
        "x__graph_gate_names": np.asarray(gate_names, dtype=f"<U{max_name}"),
        "x__graph_gate_qubit_ptr": np.asarray(qubit_ptr, dtype=np.int32),
        "x__graph_gate_qubit_indices": np.asarray(qubit_indices, dtype=np.int16),
        "x__graph_gate_parameter_ptr": np.asarray(param_ptr, dtype=np.int32),
        "x__graph_gate_parameter_sin": np.asarray(param_sin, dtype=np.float64),
        "x__graph_gate_parameter_cos": np.asarray(param_cos, dtype=np.float64),
        "x__layout_logical_to_physical": np.asarray(layout, dtype=np.int16),
    }


def _single_graph_batch(
    intended_circuit: QuantumCircuit,
    logical_to_physical: Sequence[int],
    *,
    device: torch.device | str,
) -> GraphTensorBatch:
    target_device = torch.device(device)
    graph = _graph_features_for_example(
        serialize_intended_graph(intended_circuit, logical_to_physical)
    )
    n_qubits = int(graph["node_features"].shape[0])
    n_gates = int(graph["gate_features"].shape[0])
    return GraphTensorBatch(
        node_features=torch.as_tensor(graph["node_features"], dtype=torch.float32, device=target_device),
        edge_index=torch.as_tensor(graph["edge_index"], dtype=torch.long, device=target_device),
        edge_features=torch.as_tensor(graph["edge_features"], dtype=torch.float32, device=target_device),
        edge_event_index=torch.as_tensor(graph["edge_event_index"], dtype=torch.long, device=target_device),
        gate_features=torch.as_tensor(graph["gate_features"], dtype=torch.float32, device=target_device),
        gate_qubit_ptr=torch.as_tensor(graph["gate_qubit_ptr"], dtype=torch.long, device=target_device),
        gate_qubit_indices=torch.as_tensor(graph["gate_qubit_indices"], dtype=torch.long, device=target_device),
        node_batch=torch.zeros(n_qubits, dtype=torch.long, device=target_device),
        gate_batch=torch.zeros(n_gates, dtype=torch.long, device=target_device),
        graph_count=1,
    )


def build_step7_model_batch_from_counts(
    intended_circuit: QuantumCircuit,
    logical_to_physical: Sequence[int],
    reference_counts: Mapping[str, Mapping[str, int]],
    observed_counts: Mapping[str, Mapping[str, int]],
    *,
    device: torch.device | str = "cpu",
) -> Step7ModelBatch:
    """Build exactly the graph + ``DiagnosticTensorBatch`` consumed by late_concat."""

    graph = _single_graph_batch(intended_circuit, logical_to_physical, device=device)
    arrays = paired_diagnostic_arrays(
        reference_counts, observed_counts, intended_circuit.num_qubits
    )
    target_device = graph.node_features.device
    pair_index = np.asarray(arrays["x__pair_indices"], dtype=np.int64).reshape(-1, 2)
    diagnostic = DiagnosticTensorBatch(
        local_values=torch.as_tensor(
            np.asarray(arrays["x__delta_local_expectations"], dtype=np.float32).T,
            dtype=torch.float32,
            device=target_device,
        ),
        pair_values=torch.as_tensor(
            np.asarray(arrays["x__delta_pairwise_correlations"], dtype=np.float32).T,
            dtype=torch.float32,
            device=target_device,
        ),
        pair_index=torch.as_tensor(pair_index.T, dtype=torch.long, device=target_device),
        pair_batch=torch.zeros(len(pair_index), dtype=torch.long, device=target_device),
        global_parity=torch.as_tensor(
            np.asarray(arrays["x__delta_global_parity"], dtype=np.float32).reshape(1, 3),
            dtype=torch.float32,
            device=target_device,
        ),
        basis_codes=torch.as_tensor([[0, 1, 2]], dtype=torch.long, device=target_device),
        observed_shots=torch.as_tensor(
            np.asarray(arrays["x__observed_shots"], dtype=np.int64).reshape(1, 3),
            dtype=torch.long,
            device=target_device,
        ),
        reference_shots=torch.as_tensor(
            np.asarray(arrays["x__reference_shots"], dtype=np.int64).reshape(1, 3),
            dtype=torch.long,
            device=target_device,
        ),
        reference_available_mask=torch.ones((1, 3), dtype=torch.bool, device=target_device),
        reference_kind_code=torch.full((1, 1), REFERENCE_KIND_CODE, dtype=torch.long, device=target_device),
        available_mask=torch.ones(1, dtype=torch.bool, device=target_device),
    )
    batch = Step7ModelBatch(graph=graph, diagnostic=diagnostic)
    diagnostic.validate(graph)
    return batch


def build_measurement_circuit(circuit: QuantumCircuit, basis: str) -> QuantumCircuit:
    """Append one all-qubit Pauli-basis measurement program with register ``meas``."""

    _validate_unmeasured_circuit(circuit, "acquisition circuit")
    basis_name = str(basis).upper()
    if basis_name not in BASIS_ORDER:
        raise ValueError(f"unknown diagnostic basis {basis!r}")
    output = circuit.copy(name=f"{circuit.name or 'circuit'}__{basis_name}")
    if basis_name == "X":
        for qubit in range(output.num_qubits):
            output.h(qubit)
    elif basis_name == "Y":
        for qubit in range(output.num_qubits):
            output.sdg(qubit)
            output.h(qubit)
    register = ClassicalRegister(output.num_qubits, MEASUREMENT_REGISTER)
    output.add_register(register)
    output.measure(output.qubits, register[:])
    return output


def build_paired_measurement_circuits(
    reference_circuit: QuantumCircuit, observed_circuit: QuantumCircuit
) -> tuple[QuantumCircuit, ...]:
    _validate_unmeasured_circuit(reference_circuit, "reference circuit")
    _validate_unmeasured_circuit(observed_circuit, "observed circuit")
    if reference_circuit.num_qubits != observed_circuit.num_qubits:
        raise ValueError("reference and observed circuits must have the same logical qubit count")
    return tuple(
        [build_measurement_circuit(reference_circuit, basis) for basis in BASIS_ORDER]
        + [build_measurement_circuit(observed_circuit, basis) for basis in BASIS_ORDER]
    )


def compile_paired_measurement_circuits(
    reference_circuit: QuantumCircuit,
    observed_circuit: QuantumCircuit,
    backend: Any,
    *,
    initial_layout: Sequence[int],
    optimization_level: int = 1,
    seed_transpiler: int = 17091,
) -> tuple[QuantumCircuit, ...]:
    """Compile all six programs to one backend ISA using the same explicit initial layout."""

    circuits = build_paired_measurement_circuits(reference_circuit, observed_circuit)
    layout = tuple(int(value) for value in initial_layout)
    if len(layout) != reference_circuit.num_qubits or len(set(layout)) != len(layout) or min(layout) < 0:
        raise ValueError("initial_layout must map every logical qubit to one unique physical qubit")
    backend_qubits = getattr(backend, "num_qubits", None)
    if backend_qubits is not None and max(layout) >= int(backend_qubits):
        raise ValueError("initial_layout references a physical qubit outside the backend")
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=int(optimization_level),
        initial_layout=list(layout),
        seed_transpiler=int(seed_transpiler),
    )
    compiled = pass_manager.run(list(circuits))
    return tuple(compiled)


def _extract_counts(pub_result: Any, register_name: str = MEASUREMENT_REGISTER) -> dict[str, int]:
    data = getattr(pub_result, "data", None)
    if data is None:
        raise TypeError("Sampler PUB result has no data field")
    register = getattr(data, register_name, None)
    if register is None:
        raise TypeError(f"Sampler PUB result has no {register_name!r} classical register")
    getter = getattr(register, "get_counts", None)
    if not callable(getter):
        raise TypeError(f"Sampler register {register_name!r} does not expose get_counts()")
    counts = {str(key): int(value) for key, value in dict(getter()).items()}
    if not counts:
        raise ValueError("Sampler PUB returned empty counts")
    return counts


def make_ibm_runtime_sampler(backend: Any, *, options: Mapping[str, Any] | None = None) -> Any:
    """Create the pinned IBM Runtime SamplerV2 object without submitting a job."""

    from qiskit_ibm_runtime import SamplerV2

    return SamplerV2(mode=backend, options=dict(options) if options is not None else None)


def acquire_paired_diagnostics(
    reference_circuit: QuantumCircuit,
    observed_circuit: QuantumCircuit,
    backend: Any,
    sampler: Any,
    *,
    initial_layout: Sequence[int],
    shots: int,
    optimization_level: int = 1,
    seed_transpiler: int = 17091,
    device: torch.device | str = "cpu",
) -> HardwareDiagnosticAcquisition:
    """Run six already-defined paired programs through a SamplerV2-compatible object."""

    if isinstance(shots, bool) or int(shots) <= 0:
        raise ValueError("shots must be a positive integer")
    isa_circuits = compile_paired_measurement_circuits(
        reference_circuit,
        observed_circuit,
        backend,
        initial_layout=initial_layout,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    job = sampler.run(list(isa_circuits), shots=int(shots))
    result = job.result()
    if len(result) != len(PROGRAM_ORDER):
        raise RuntimeError(
            f"expected {len(PROGRAM_ORDER)} Sampler PUB results, received {len(result)}"
        )
    counts_by_program = {
        label: _extract_counts(pub_result)
        for label, pub_result in zip(PROGRAM_ORDER, result, strict=True)
    }
    reference_counts = {
        basis: counts_by_program[f"reference_{basis}"] for basis in BASIS_ORDER
    }
    observed_counts = {
        basis: counts_by_program[f"observed_{basis}"] for basis in BASIS_ORDER
    }
    model_batch = build_step7_model_batch_from_counts(
        reference_circuit,
        initial_layout,
        reference_counts,
        observed_counts,
        device=device,
    )
    job_id_method = getattr(job, "job_id", None)
    job_id = str(job_id_method()) if callable(job_id_method) else None
    return HardwareDiagnosticAcquisition(
        model_batch=model_batch,
        counts_by_program=counts_by_program,
        isa_circuits=isa_circuits,
        initial_layout=tuple(int(value) for value in initial_layout),
        shots=int(shots),
        job_id=job_id,
    )


__all__ = [
    "BASIS_ORDER",
    "BASIS_CODES",
    "PROGRAM_ORDER",
    "EmpiricalPauliStats",
    "HardwareDiagnosticAcquisition",
    "acquire_paired_diagnostics",
    "all_pair_indices",
    "build_measurement_circuit",
    "build_paired_measurement_circuits",
    "build_step7_model_batch_from_counts",
    "compile_paired_measurement_circuits",
    "empirical_stats_from_counts",
    "make_ibm_runtime_sampler",
    "paired_diagnostic_arrays",
    "serialize_intended_graph",
]
