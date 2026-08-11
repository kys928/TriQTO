#!/usr/bin/env python3
"""Step 3 v3: exact replay with terminal measurement-marker semantics.

Step 3 v2 correctly refused before numerical state comparison because the graph
sequence contains ``measure`` events and its replay helper treated every graph
event as a unitary circuit instruction. This wrapper preserves the v2 scientific
audit and frozen decision thresholds while correcting only execution semantics:

* ``measure`` is a nonunitary readout marker and is excluded from statevector
  propagation;
* measurement markers must form a terminal readout tail: once measurement starts,
  no later unitary gate is allowed;
* insertion-depth stratification is computed over unitary events rather than
  measurement markers.

The v2 all-source replay-validation gate remains mandatory. If replay does not
reproduce all archived clean and distorted statevectors up to global phase, the
audit still refuses before matched counterfactual generation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


HERE = Path(__file__).resolve().parent
V2_PATH = HERE / "audit_matched_bdelta_replay_identifiability.py"
V2_MODULE_NAME = "triqto_v0_2_matched_bdelta_replay_v2"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/v0_2/matched_bdelta_replay_identifiability_audit_v3.json"
)
READOUT_GATE_NAMES = frozenset({"measure"})
NON_EVOLUTION_GATE_NAMES = frozenset({"measure", "barrier"})


def load_v2():
    spec = importlib.util.spec_from_file_location(V2_MODULE_NAME, V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Step 3 v2 runner from {V2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[V2_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


V2 = load_v2()
ORIGINAL_REPLAY_SOURCE = V2.replay_source


def validate_measurement_tail(records: Sequence[Mapping[str, Any]]) -> int:
    """Validate and count terminal one-qubit measurement markers."""
    measurement_started = False
    measurement_count = 0
    for record in records:
        name = str(record["name"]).lower()
        if name in READOUT_GATE_NAMES:
            measurement_started = True
            measurement_count += 1
            if len(record["qubits"]) != 1:
                raise ValueError("measure event must reference exactly one qubit")
            if len(record["params"]) != 0:
                raise ValueError("measure event must not contain unitary parameters")
            continue
        if measurement_started and name not in NON_EVOLUTION_GATE_NAMES:
            raise ValueError(
                f"unitary gate {name!r} appears after measurement tail began"
            )
    return measurement_count


def simulate_records(
    records: Sequence[Mapping[str, Any]],
    n_qubits: int,
    *,
    removed_index: int | None = None,
    replacement_axis: str | None = None,
    replacement_angle: float | None = None,
) -> np.ndarray:
    """Replay unitary events while excluding terminal measurement markers."""
    validate_measurement_tail(records)
    circuit = QuantumCircuit(n_qubits)
    for record in records:
        index = int(record["index"])
        name = str(record["name"]).lower()
        if name in READOUT_GATE_NAMES:
            continue
        if removed_index is not None and index == removed_index:
            if replacement_axis is None:
                continue
            qubits = list(record["qubits"])
            if len(qubits) != 1 or replacement_angle is None:
                raise ValueError("matched replacement requires one qubit and one angle")
            getattr(circuit, replacement_axis)(float(replacement_angle), int(qubits[0]))
            continue
        V2.append_gate(circuit, record)
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    state /= np.linalg.norm(state)
    return state


def unitary_insertion_bin(
    records: Sequence[Mapping[str, Any]], removed_index: int
) -> tuple[float, str, int, bool]:
    """Return depth over physical evolution events, excluding readout/barriers."""
    unitary_indices = [
        int(record["index"])
        for record in records
        if str(record["name"]).lower() not in NON_EVOLUTION_GATE_NAMES
    ]
    if removed_index not in unitary_indices:
        raise ValueError("removed distortion is not a unitary replay event")
    rank = unitary_indices.index(removed_index)
    fraction, label = V2.insertion_bin(rank, len(unitary_indices))
    later_unitary = any(index > removed_index for index in unitary_indices)
    return fraction, label, len(unitary_indices), not later_unitary


def replay_source(
    root: Path,
    row: Mapping[str, Any],
    *,
    overlap_loss_max: float,
    amplitude_error_max: float,
) -> dict[str, Any]:
    """Use v2 replay validation, then correct readout-aware depth metadata."""
    info = ORIGINAL_REPLAY_SOURCE(
        root,
        row,
        overlap_loss_max=overlap_loss_max,
        amplitude_error_max=amplitude_error_max,
    )
    records = info["records"]
    measurement_count = validate_measurement_tail(records)
    fraction, depth_bin, unitary_count, terminal_unitary = unitary_insertion_bin(
        records,
        int(info["removed_index"]),
    )
    info["graph_event_count"] = int(info["gate_count"])
    info["unitary_event_count"] = unitary_count
    info["terminal_measurement_count"] = measurement_count
    info["terminal_with_respect_to_unitary_evolution"] = terminal_unitary
    info["insertion_depth_fraction"] = fraction
    info["insertion_depth_fraction_bin"] = depth_bin
    return info


def main() -> None:
    # Patch only the two execution helpers used by the already-frozen v2 main.
    V2.simulate_records = simulate_records
    V2.replay_source = replay_source

    original_argv = list(sys.argv)
    try:
        if not any(
            arg == "--config" or arg.startswith("--config=")
            for arg in sys.argv[1:]
        ):
            sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
        V2.main()
    finally:
        sys.argv[:] = original_argv


if __name__ == "__main__":
    main()
