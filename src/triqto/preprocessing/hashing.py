"""Versioned cryptographic identities for distinct scientific equivalence notions."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .canonical import canonical_complex
from .config import PreprocessingConfig
from .constants import (
    CANONICALIZATION_VERSION,
    HASH_SERIALIZATION_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
)
from .records import HashBundle


def normalize_hash_payload(value: Any, *, decimals: int) -> Any:
    if is_dataclass(value):
        return normalize_hash_payload(asdict(value), decimals=decimals)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [list(canonical_complex(item, decimals)) for item in value.reshape(-1).tolist()]
        return normalize_hash_payload(value.tolist(), decimals=decimals)
    if isinstance(value, Mapping):
        return {
            str(key): normalize_hash_payload(value[key], decimals=decimals)
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_hash_payload(item, decimals=decimals) for item in value]
    if isinstance(value, set):
        normalized = [normalize_hash_payload(item, decimals=decimals) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, complex):
        return list(canonical_complex(value, decimals))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("nonfinite values cannot enter a cryptographic identity")
        rounded = round(numeric, decimals)
        return 0.0 if rounded == 0.0 else rounded
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "tolist"):
        return normalize_hash_payload(value.tolist(), decimals=decimals)
    raise TypeError(f"unsupported hash payload type {type(value)!r}")


def normalize_raw_payload(value: Any) -> Any:
    """Preserve source-row scalar identity without canonicalization rounding."""
    if is_dataclass(value):
        return normalize_raw_payload(asdict(value))
    if isinstance(value, Enum):
        return normalize_raw_payload(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return normalize_raw_payload(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): normalize_raw_payload(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_raw_payload(item) for item in value]
    if isinstance(value, set):
        items = [normalize_raw_payload(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, complex):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError("nonfinite complex value cannot enter raw identity")
        return {"__complex_hex__": [float(value.real).hex(), float(value.imag).hex()]}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("nonfinite value cannot enter raw identity")
        return {"__float_hex__": numeric.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "tolist"):
        return normalize_raw_payload(value.tolist())
    raise TypeError(f"unsupported raw hash payload type {type(value)!r}")


def sha256_raw_payload(payload: Any) -> str:
    envelope = {
        "hash_type": "raw_record_source_rows",
        "hash_version": "v1",
        "digest_algorithm": "sha256",
        "serialization_version": "triqto.raw_hash_hex_float.v1",
        "payload": normalize_raw_payload(payload),
    }
    serialized = json.dumps(
        envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def canonical_hash_json(payload: Any, *, decimals: int) -> str:
    normalized = normalize_hash_payload(payload, decimals=decimals)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_payload(hash_type: str, payload: Any, *, config: PreprocessingConfig) -> str:
    envelope = {
        "hash_type": hash_type,
        "hash_version": "v1",
        "digest_algorithm": "sha256",
        "serialization_version": HASH_SERIALIZATION_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "payload": payload,
    }
    serialized = canonical_hash_json(
        envelope,
        decimals=config.numerical_tolerances.hash_rounding_decimals,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_statevector(
    statevector: np.ndarray | None,
    *,
    qubit_ordering: str,
    config: PreprocessingConfig,
) -> str | None:
    if statevector is None:
        return None
    return sha256_payload(
        "state_equivalence",
        {
            "kind": "pure_state_global_phase_canonical",
            "qubit_ordering": qubit_ordering,
            "amplitudes": statevector,
        },
        config=config,
    )


def build_hash_bundle(
    *,
    raw_record_payload: Mapping[str, Any],
    canonical_record_payload: Mapping[str, Any],
    canonical_circuit_payload: Mapping[str, Any],
    canonical_parameters: Mapping[str, float],
    statevector: np.ndarray | None,
    born_distribution: Mapping[str, float],
    measurement_basis: str,
    counts: Mapping[str, int] | None,
    shot_count: int | None,
    hardware_context: Mapping[str, Any],
    target_payload: Any,
    counterfactual_payload: Any,
    graph_payload: Mapping[str, Any],
    config: PreprocessingConfig,
) -> HashBundle:
    raw_hash = sha256_raw_payload(raw_record_payload)
    canonical_record_hash = sha256_payload("canonical_record", canonical_record_payload, config=config)
    circuit_hash = sha256_payload("canonical_circuit", canonical_circuit_payload, config=config)
    circuit_parameter_hash = sha256_payload(
        "circuit_plus_parameters",
        {"circuit": canonical_circuit_payload, "parameters": canonical_parameters},
        config=config,
    )
    born_hash = sha256_payload(
        "born_distribution",
        {
            "measurement_basis": measurement_basis,
            "probabilities": born_distribution,
            "kind": "exact_or_estimated_shape",
        },
        config=config,
    )
    measurement_hash = None
    if counts is not None:
        measurement_hash = sha256_payload(
            "measurement_instance",
            {
                "measurement_basis": measurement_basis,
                "counts": counts,
                "shot_count": shot_count,
            },
            config=config,
        )
    state_hash = hash_statevector(
        statevector,
        qubit_ordering=config.canonicalization.bit_order,
        config=config,
    )
    hardware_hash = sha256_payload("hardware_context", hardware_context, config=config)
    target_hash = (
        None if target_payload is None else sha256_payload("target_identity", target_payload, config=config)
    )
    counterfactual_hash = sha256_payload("counterfactual_set", counterfactual_payload, config=config)
    labeled_graph_hash = sha256_payload(
        "labeled_graph", graph_payload.get("labeled_edges", []), config=config
    )
    structural_graph_hash = sha256_payload(
        "structural_graph",
        {
            "wl": graph_payload.get("wl_structural_hash"),
            "node_count": graph_payload.get("node_count"),
            "degree_sequence": graph_payload.get("degree_sequence", []),
        },
        config=config,
    )
    feature_graph_hash = sha256_payload(
        "feature_graph",
        {
            "wl": graph_payload.get("wl_feature_hash"),
            "event_counts": {
                key: graph_payload.get(key)
                for key in (
                    "one_qubit_event_count",
                    "two_qubit_event_count",
                    "measurement_event_count",
                )
            },
        },
        config=config,
    )
    return HashBundle(
        raw_record_hash=raw_hash,
        canonical_record_hash=canonical_record_hash,
        canonical_circuit_hash=circuit_hash,
        circuit_parameter_hash=circuit_parameter_hash,
        state_equivalence_hash=state_hash,
        born_distribution_hash=born_hash,
        measurement_instance_hash=measurement_hash,
        hardware_context_hash=hardware_hash,
        target_hash=target_hash,
        counterfactual_set_hash=counterfactual_hash,
        labeled_graph_hash=labeled_graph_hash,
        structural_graph_hash=structural_graph_hash,
        feature_graph_hash=feature_graph_hash,
    )
