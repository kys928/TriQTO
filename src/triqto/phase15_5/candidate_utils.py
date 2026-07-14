"""Candidate feature and noisy-evidence helpers for Phase 15.5."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from triqto.core.ids import make_deterministic_id
from triqto.data_generation import derive_child_seed
from triqto.metrics import hellinger_distance, jensen_shannon_divergence, total_variation_distance
from triqto.simulation import NoiseSpec, measurement_setting_for, simulate_noisy_aer_shots
from .policy import FAMILY_NAMES

def _noise_strength(profile: Any) -> float:
    values: list[float] = []
    for channel in profile.channels:
        for key in ("probability", "error"):
            value = channel.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return float(sum(values) / len(values)) if values else 0.0

def _distribution_summary(probabilities: Mapping[str, float], n_qubits: int) -> list[float]:
    values = np.asarray(list(probabilities.values()), dtype=np.float64)
    entropy = -float(np.sum(values[values > 0.0] * np.log2(values[values > 0.0]))) if values.size else 0.0
    max_entropy = max(1.0, float(n_qubits))
    support_fraction = len(probabilities) / float(2**n_qubits)
    parity = 0.0
    z_expectations = np.zeros(n_qubits, dtype=np.float64)
    for bitstring, probability in probabilities.items():
        bits = [int(value) for value in bitstring.zfill(n_qubits)]
        parity += probability * (1.0 if sum(bits) % 2 == 0 else -1.0)
        for index, bit in enumerate(reversed(bits)):
            z_expectations[index] += probability * (1.0 if bit == 0 else -1.0)
    return [
        entropy / max_entropy,
        float(values.max()) if values.size else 0.0,
        support_fraction,
        parity,
        float(np.mean(np.abs(z_expectations))) if n_qubits else 0.0,
    ]

def _backend_summary(sample: Any) -> tuple[float, float]:
    summary = sample.metadata.get("backend_feature_values", {}).get("coupling_summary", {})
    degree_mean = float(summary.get("degree_mean", 0.0))
    degree_max = float(summary.get("degree_max", 0.0))
    scale = max(1.0, float(sample.n_qubits - 1))
    return degree_mean / scale, degree_max / scale

def _cost_utility(objective: Mapping[str, Any], *, accepted: bool) -> float:
    if not accepted:
        return 0.0
    depth = float(objective.get("depth_delta", 0.0))
    size = float(objective.get("size_delta", 0.0))
    two = float(objective.get("two_qubit_gate_delta", 0.0))
    swaps = float(objective.get("swap_count", 0.0))
    gain = max(0.0, -depth) + 0.5 * max(0.0, -size) + 2.0 * max(0.0, -two)
    regression = max(0.0, depth) + 0.5 * max(0.0, size) + 2.0 * max(0.0, two) + 0.5 * max(0.0, swaps)
    return float(min(1.0, max(0.0, 0.5 + 0.5 * (gain - regression) / (1.0 + gain + regression))))

def _candidate_features(
    *,
    kind: str,
    basis: str | None = None,
    optimization_level: int | None = None,
    objective: Mapping[str, Any] | None = None,
    acquires_evidence: bool = False,
    semantic_validation_available: bool = False,
    backend_evidence_available: bool = False,
) -> list[float]:
    objective = dict(objective or {})
    basis = basis or ""
    return [
        float(kind == "no_op"),
        float(kind == "probe"),
        float(kind == "layout"),
        float(kind == "routing"),
        float(kind == "depth"),
        float(basis == "Z"),
        float(basis == "X"),
        float(basis == "Y"),
        float(optimization_level or 0) / 3.0,
        max(-1.0, min(1.0, float(objective.get("depth_delta", 0.0)) / 16.0)),
        max(-1.0, min(1.0, float(objective.get("size_delta", 0.0)) / 32.0)),
        max(-1.0, min(1.0, float(objective.get("two_qubit_gate_delta", 0.0)) / 8.0)),
        max(0.0, min(1.0, float(objective.get("swap_count", 0.0)) / 8.0)),
        float(acquires_evidence),
        float(semantic_validation_available),
        float(backend_evidence_available),
    ]

def _no_op_row(sample: Any, profile_id: str, split: str, split_group_id: str, family_id: int, context: np.ndarray) -> dict[str, Any]:
    family = FAMILY_NAMES[family_id]
    utility = 0.0 if family_id == 0 else 0.5
    payload = {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": family, "kind": "no_op"}
    return {
        "candidate_id": make_deterministic_id("phase155_candidate", payload),
        "group_id": make_deterministic_id("phase155_group", {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": family}),
        "split_group_id": split_group_id,
        "split": split,
        "family_id": family_id,
        "context": context,
        "candidate": np.asarray(_candidate_features(kind="no_op"), dtype=np.float64),
        "utility": utility,
        "available": True,
        "metadata": {**payload, "available": True, "utility_target_role": "family_specific_simulation_supervision"},
    }

def _evidence_for_basis(clean: Any, distorted: Any, *, noise: NoiseSpec, shots: int, seed: int, basis: str) -> dict[str, Any]:
    setting = measurement_setting_for(clean.num_qubits, basis)
    clean_seed = derive_child_seed(seed, "phase155_clean_noisy", {"basis": basis})
    distorted_seed = derive_child_seed(seed, "phase155_distorted_noisy", {"basis": basis})
    clean_result = simulate_noisy_aer_shots(clean, noise=noise, shots=shots, seed=clean_seed, measurement_basis=setting)
    distorted_result = simulate_noisy_aer_shots(distorted, noise=noise, shots=shots, seed=distorted_seed, measurement_basis=setting)
    return {
        "basis": basis,
        "measurement_setting": setting.to_metadata(),
        "clean_seed": clean_seed,
        "distorted_seed": distorted_seed,
        "clean_counts": dict(sorted(clean_result.counts.items())),
        "distorted_counts": dict(sorted(distorted_result.counts.items())),
        "clean_probabilities": dict(sorted(clean_result.probabilities.items())),
        "distorted_probabilities": dict(sorted(distorted_result.probabilities.items())),
        "total_variation": total_variation_distance(clean_result.probabilities, distorted_result.probabilities),
        "jensen_shannon_divergence": jensen_shannon_divergence(clean_result.probabilities, distorted_result.probabilities),
        "hellinger": hellinger_distance(clean_result.probabilities, distorted_result.probabilities),
    }
