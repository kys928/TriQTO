"""Pure deterministic canonicalization for TriQTO preprocessing."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping

import networkx as nx
import numpy as np

from .config import CanonicalizationConfig, NumericalTolerances


def _clean_float(value: float, decimals: int) -> float:
    rounded = round(float(value), decimals)
    return 0.0 if rounded == 0.0 else rounded


def canonical_complex(value: complex, decimals: int) -> tuple[float, float]:
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise ValueError("canonical complex values must be finite")
    return (_clean_float(number.real, decimals), _clean_float(number.imag, decimals))


def canonicalize_angle(value: float, config: CanonicalizationConfig) -> tuple[float, int]:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("angle must be finite")
    low, high = config.angle_interval
    period = config.angle_period
    canonical = ((number - low) % period) + low
    if canonical >= high or math.isclose(canonical, high, rel_tol=0.0, abs_tol=1e-15):
        canonical = low
    canonical = 0.0 if canonical == 0.0 else canonical
    wraps = int(round((number - canonical) / period))
    return canonical, wraps


def canonicalize_parameter_bindings(
    bindings: Mapping[str, Any], config: CanonicalizationConfig
) -> tuple[dict[str, float], dict[str, int]]:
    canonical: dict[str, float] = {}
    wraps: dict[str, int] = {}
    for name in sorted(bindings, key=str):
        value = bindings[name]
        if isinstance(value, bool):
            raise TypeError(f"parameter {name!r} must be numeric, not bool")
        angle, count = canonicalize_angle(float(value), config)
        canonical[str(name)] = angle
        wraps[str(name)] = count
    return canonical, wraps


def canonical_gate_name(name: Any, config: CanonicalizationConfig) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("gate name must be nonblank")
    alias = config.gate_alias_map.get(raw, config.gate_alias_map.get(raw.lower(), raw.lower()))
    return str(alias).strip().lower()


def canonical_basis_label(value: Any, config: CanonicalizationConfig) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("measurement basis must be nonblank")
    alias = config.basis_alias_map.get(raw, config.basis_alias_map.get(raw.lower()))
    if alias is not None:
        return str(alias)
    compact = raw.replace(" ", "").upper()
    if compact and all(character in "IXYZ" for character in compact):
        return compact
    if compact.startswith("POVM:") or compact.startswith("OBS:"):
        return compact
    raise ValueError(f"unsupported or ambiguous measurement basis {value!r}")


def canonicalize_probability_map(
    probabilities: Mapping[str, float], *, width: int, decimals: int
) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_key, raw_value in probabilities.items():
        key = str(raw_key).replace(" ", "")
        if len(key) > width or any(character not in "01" for character in key):
            raise ValueError(f"invalid bitstring {raw_key!r} for width {width}")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("probability must be finite")
        padded = key.zfill(width)
        result[padded] = result.get(padded, 0.0) + value
    return {
        key: _clean_float(result[key], decimals)
        for key in sorted(result)
    }


def canonicalize_counts(counts: Mapping[str, int], *, width: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_key, raw_value in counts.items():
        key = str(raw_key).replace(" ", "")
        if len(key) > width or any(character not in "01" for character in key):
            raise ValueError(f"invalid bitstring {raw_key!r} for width {width}")
        if isinstance(raw_value, bool) or int(raw_value) < 0:
            raise ValueError("counts must be nonnegative integers")
        padded = key.zfill(width)
        result[padded] = result.get(padded, 0) + int(raw_value)
    return {key: result[key] for key in sorted(result)}


def canonicalize_statevector_global_phase(
    statevector: np.ndarray,
    *,
    epsilon: float,
    norm_tolerance: float,
) -> tuple[np.ndarray, complex, int]:
    vector = np.asarray(statevector, dtype=np.complex128).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector.real).all() or not np.isfinite(vector.imag).all():
        raise ValueError("statevector must be nonempty and finite")
    norm = float(np.vdot(vector, vector).real)
    if norm <= norm_tolerance:
        raise ValueError("all-zero statevector is invalid")
    if abs(norm - 1.0) > norm_tolerance:
        raise ValueError(f"statevector norm {norm} exceeds canonicalization tolerance")
    vector /= math.sqrt(norm)
    pivots = np.flatnonzero(np.abs(vector) > epsilon)
    if pivots.size == 0:
        raise ValueError("statevector has no nonzero canonical pivot")
    pivot = int(pivots[0])
    phase_factor = np.exp(-1j * np.angle(vector[pivot]))
    canonical = vector * phase_factor
    canonical[pivot] = complex(abs(canonical[pivot]), 0.0)
    tiny_real = np.abs(canonical.real) <= epsilon
    tiny_imag = np.abs(canonical.imag) <= epsilon
    canonical.real[tiny_real] = 0.0
    canonical.imag[tiny_imag] = 0.0
    return canonical, complex(phase_factor), pivot


def _bit_index(circuit: Any, bit: Any) -> int:
    found = circuit.find_bit(bit)
    return int(getattr(found, "index", found[0] if isinstance(found, tuple) else found))


def _canonical_parameter(value: Any, decimals: int) -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("circuit parameters must be finite")
        return _clean_float(number, decimals)
    if isinstance(value, complex):
        return list(canonical_complex(value, decimals))
    return {"symbol": str(value)}


def canonicalize_circuit(
    circuit: Any,
    config: CanonicalizationConfig,
    tolerances: NumericalTolerances,
) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for index, entry in enumerate(circuit.data):
        if hasattr(entry, "operation"):
            operation = entry.operation
            qargs = tuple(entry.qubits)
            cargs = tuple(entry.clbits)
        else:
            operation, qargs, cargs = entry
            qargs = tuple(qargs)
            cargs = tuple(cargs)
        name = canonical_gate_name(operation.name, config)
        if name == "barrier" and config.barrier_semantically_irrelevant:
            continue
        params = [
            _canonical_parameter(value, tolerances.hash_rounding_decimals)
            for value in getattr(operation, "params", ())
        ]
        payload: dict[str, Any] = {
            "index": index,
            "name": name,
            "original_name": str(operation.name),
            "qubits": [_bit_index(circuit, bit) for bit in qargs],
            "clbits": [_bit_index(circuit, bit) for bit in cargs],
            "parameters": params,
        }
        condition = getattr(operation, "condition", None)
        if condition is not None:
            payload["condition"] = str(condition)
        operations.append(payload)
    metadata = getattr(circuit, "metadata", None)
    stable_metadata = (
        {str(key): metadata[key] for key in sorted(metadata, key=str)}
        if isinstance(metadata, Mapping)
        else {}
    )
    return {
        "name": str(getattr(circuit, "name", "")),
        "n_qubits": int(circuit.num_qubits),
        "n_clbits": int(circuit.num_clbits),
        "global_phase": _canonical_parameter(
            getattr(circuit, "global_phase", 0.0), tolerances.hash_rounding_decimals
        ),
        "operations": operations,
        "metadata": stable_metadata,
        "bit_order": config.bit_order,
    }


def circuit_parameter_signature(canonical_circuit: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for operation in canonical_circuit.get("operations", []):
        for parameter_index, value in enumerate(operation.get("parameters", [])):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                key = f"op{operation['index']}:{operation['name']}:p{parameter_index}"
                result[key] = float(value)
    return result


def circuit_graph_payload(canonical_circuit: Mapping[str, Any]) -> dict[str, Any]:
    n_qubits = int(canonical_circuit.get("n_qubits", 0))
    graph = nx.Graph()
    graph.add_nodes_from(range(n_qubits))
    one_qubit = 0
    two_qubit = 0
    measurements = 0
    edge_events: dict[tuple[int, int], Counter[str]] = {}
    node_events: dict[int, Counter[str]] = {node: Counter() for node in range(n_qubits)}
    for operation in canonical_circuit.get("operations", []):
        name = str(operation.get("name", "unknown"))
        qubits = [int(value) for value in operation.get("qubits", [])]
        if name == "measure":
            measurements += 1
        if len(qubits) == 1:
            one_qubit += 1
            node_events.setdefault(qubits[0], Counter())[name] += 1
        elif len(qubits) >= 2:
            two_qubit += 1
            for left_index in range(len(qubits)):
                for right_index in range(left_index + 1, len(qubits)):
                    edge = tuple(sorted((qubits[left_index], qubits[right_index])))
                    edge_events.setdefault(edge, Counter())[name] += 1
                    graph.add_edge(*edge)
    for node in graph.nodes:
        graph.nodes[node]["label"] = ";".join(
            f"{name}:{count}" for name, count in sorted(node_events.get(node, {}).items())
        )
    for left, right in graph.edges:
        graph.edges[left, right]["label"] = ";".join(
            f"{name}:{count}" for name, count in sorted(edge_events.get((left, right), {}).items())
        )
    structural = nx.weisfeiler_lehman_graph_hash(graph, iterations=3)
    feature = nx.weisfeiler_lehman_graph_hash(
        graph, edge_attr="label", node_attr="label", iterations=3
    )
    labeled_edges = [
        {
            "source": left,
            "target": right,
            "events": dict(sorted(edge_events[(left, right)].items())),
        }
        for left, right in sorted(edge_events)
    ]
    return {
        "node_count": n_qubits,
        "labeled_edges": labeled_edges,
        "degree_sequence": sorted((degree for _, degree in graph.degree()), reverse=True),
        "connected_components": nx.number_connected_components(graph) if n_qubits else 0,
        "one_qubit_event_count": one_qubit,
        "two_qubit_event_count": two_qubit,
        "measurement_event_count": measurements,
        "wl_structural_hash": structural,
        "wl_feature_hash": feature,
    }
