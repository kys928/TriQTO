"""Physics-aware effect sizes, severity tags, and observed-effect audits."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .config import EffectConfig


def _aligned_probability_vectors(
    left: Mapping[str, float], right: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(left) | set(right))
    p = np.asarray([max(0.0, float(left.get(key, 0.0))) for key in keys], dtype=float)
    q = np.asarray([max(0.0, float(right.get(key, 0.0))) for key in keys], dtype=float)
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        raise ValueError("probability distributions must have positive mass")
    return p / p.sum(), q / q.sum()


def hellinger_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    p, q = _aligned_probability_vectors(left, right)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def total_variation_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    p, q = _aligned_probability_vectors(left, right)
    return float(0.5 * np.sum(np.abs(p - q)))


def jensen_shannon_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    p, q = _aligned_probability_vectors(left, right)
    midpoint = 0.5 * (p + q)

    def divergence(values: np.ndarray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return float(math.sqrt(max(0.0, 0.5 * divergence(p) + 0.5 * divergence(q))))


def fisher_rao_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    p, q = _aligned_probability_vectors(left, right)
    affinity = float(np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0))
    return float(2.0 * math.acos(affinity))


def _pure_state_effects(
    clean_statevector: np.ndarray | None,
    distorted_statevector: np.ndarray | None,
) -> dict[str, float | None]:
    if clean_statevector is None or distorted_statevector is None:
        return {
            "fidelity": None,
            "infidelity": None,
            "fubini_study": None,
            "pure_trace_distance": None,
        }
    left = np.asarray(clean_statevector, dtype=np.complex128).reshape(-1)
    right = np.asarray(distorted_statevector, dtype=np.complex128).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("statevectors must have matching dimensions")
    overlap = complex(np.vdot(left, right))
    amplitude = float(np.clip(abs(overlap), 0.0, 1.0))
    fidelity = float(np.clip(amplitude * amplitude, 0.0, 1.0))
    return {
        "fidelity": fidelity,
        "infidelity": 1.0 - fidelity,
        "fubini_study": float(math.acos(amplitude)),
        "pure_trace_distance": float(math.sqrt(max(0.0, 1.0 - fidelity))),
    }


def _periodic_parameter_distance(
    clean_parameters: Mapping[str, float], distorted_parameters: Mapping[str, float]
) -> float | None:
    keys = sorted(set(clean_parameters) & set(distorted_parameters))
    if not keys:
        return None
    normalized: list[float] = []
    for key in keys:
        delta = (float(distorted_parameters[key]) - float(clean_parameters[key]) + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        normalized.append(abs(delta) / math.pi)
    return float(np.sqrt(np.mean(np.square(normalized))))


def _graph_distance(
    clean_graph: Mapping[str, Any], distorted_graph: Mapping[str, Any]
) -> float | None:
    if not clean_graph or not distorted_graph:
        return None
    structure_changed = float(
        clean_graph.get("wl_structural_hash") != distorted_graph.get("wl_structural_hash")
    )
    feature_changed = float(
        clean_graph.get("wl_feature_hash") != distorted_graph.get("wl_feature_hash")
    )
    numeric_keys = (
        "one_qubit_event_count",
        "two_qubit_event_count",
        "measurement_event_count",
        "node_count",
    )
    changes: list[float] = []
    for key in numeric_keys:
        left = float(clean_graph.get(key, 0.0) or 0.0)
        right = float(distorted_graph.get(key, 0.0) or 0.0)
        changes.append(abs(right - left) / max(1.0, abs(left), abs(right)))
    return float(np.clip(0.35 * structure_changed + 0.35 * feature_changed + 0.30 * np.mean(changes), 0.0, 1.0))


def compute_effects(
    *,
    clean_probabilities: Mapping[str, float],
    distorted_probabilities: Mapping[str, float],
    clean_statevector: np.ndarray | None,
    distorted_statevector: np.ndarray | None,
    clean_parameters: Mapping[str, float],
    distorted_parameters: Mapping[str, float],
    clean_graph: Mapping[str, Any],
    distorted_graph: Mapping[str, Any],
    config: EffectConfig,
) -> tuple[dict[str, float | None], float | None, dict[str, float]]:
    components: dict[str, float | None] = {}
    components.update(_pure_state_effects(clean_statevector, distorted_statevector))
    components.update(
        {
            "hellinger": hellinger_distance(clean_probabilities, distorted_probabilities),
            "jensen_shannon_distance": jensen_shannon_distance(
                clean_probabilities, distorted_probabilities
            ),
            "total_variation": total_variation_distance(
                clean_probabilities, distorted_probabilities
            ),
            "fisher_rao": fisher_rao_distance(clean_probabilities, distorted_probabilities),
        }
    )
    components["parameter_distance"] = _periodic_parameter_distance(
        clean_parameters, distorted_parameters
    )
    components["graph_distance"] = _graph_distance(clean_graph, distorted_graph)

    group_scores: dict[str, float | None] = {
        "hilbert": (
            None
            if components["infidelity"] is None
            else float(
                np.mean(
                    [
                        float(components["infidelity"]),
                        float(components["pure_trace_distance"]),
                        min(1.0, float(components["fubini_study"]) / (math.pi / 2.0)),
                    ]
                )
            )
        ),
        "born": float(
            np.mean(
                [
                    float(components["hellinger"]),
                    float(components["jensen_shannon_distance"]),
                    float(components["total_variation"]),
                    min(1.0, float(components["fisher_rao"]) / math.pi),
                ]
            )
        ),
        "parameter": components["parameter_distance"],
        "graph": components["graph_distance"],
        "metadata": None,
    }
    for name, value in group_scores.items():
        components[f"{name}_aggregate"] = value

    active = {
        name: (float(config.combined_distance_weights.get(name, 0.0)), float(score))
        for name, score in group_scores.items()
        if score is not None and float(config.combined_distance_weights.get(name, 0.0)) > 0.0
    }
    weight_sum = sum(weight for weight, _ in active.values())
    if weight_sum <= 0.0:
        return components, None, {}
    contributions = {
        name: (weight / weight_sum) * score
        for name, (weight, score) in active.items()
    }
    combined = float(sum(contributions.values()))
    return components, combined, contributions


def severity_from_score(score: float | None, config: EffectConfig) -> str:
    if score is None or not math.isfinite(float(score)):
        return "catastrophic"
    thresholds = config.severity_thresholds
    value = float(score)
    if value <= float(thresholds["negligible_max"]):
        return "negligible"
    if value <= float(thresholds["weak_max"]):
        return "weak"
    if value <= float(thresholds["moderate_max"]):
        return "moderate"
    if value <= float(thresholds["strong_max"]):
        return "strong"
    return "catastrophic"


def audit_observed_effect(
    *,
    intervention_label: str,
    measurement_basis: str,
    effect_components: Mapping[str, float | None],
    combined_score: float | None,
    graph_changed: bool,
    readout_only: bool,
    entanglement_evidence: float | None,
    config: EffectConfig,
) -> tuple[str, float, bool, list[str]]:
    flags: list[str] = []
    if combined_score is None:
        return "ambiguous", 0.0, True, ["insufficient_effect_evidence"]
    severity = severity_from_score(combined_score, config)
    if severity == "negligible":
        return "negligible", 1.0, False, ["eligible_no_action", "eligible_abstention"]

    born = float(effect_components.get("born_aggregate") or 0.0)
    hilbert_raw = effect_components.get("hilbert_aggregate")
    hilbert = None if hilbert_raw is None else float(hilbert_raw)
    graph = float(effect_components.get("graph_distance") or 0.0)
    basis = str(measurement_basis).upper()

    phase_hidden = (
        hilbert is not None
        and hilbert >= config.phase_sensitive_threshold
        and born < config.born_effect_threshold
    )
    if phase_hidden and basis == "Z":
        flags.extend(["phase_nonidentifiable_in_z_basis", "requires_multibasis_or_hilbert_evidence"])
        return "unobservable_in_available_basis", 0.65, True, flags

    candidates: list[tuple[str, float]] = []
    if readout_only and born >= config.born_effect_threshold:
        candidates.append(("readout_dominated_change", born))
    if graph_changed and graph >= config.layout_depth_threshold:
        candidates.append(("layout_depth_dominated_change", graph))
    if phase_hidden or (
        hilbert is not None
        and basis in {"X", "Y"}
        and hilbert >= config.phase_sensitive_threshold
    ):
        candidates.append(("phase_sensitive_change", hilbert or 0.0))
    if born >= config.born_effect_threshold:
        candidates.append(("amplitude_probability_change", born))
    if entanglement_evidence is not None and entanglement_evidence >= config.phase_sensitive_threshold:
        candidates.append(("entanglement_correlation_change", float(entanglement_evidence)))

    if not candidates:
        flags.append("effect_detected_but_not_identifiable")
        return "ambiguous", 0.4, True, flags
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if len(candidates) >= 2 and candidates[1][1] >= 0.75 * max(candidates[0][1], 1e-15):
        flags.append("multiple_effect_channels")
        return "mixed", min(1.0, candidates[0][1] + candidates[1][1]), True, flags
    label, strength = candidates[0]
    if intervention_label and label.split("_")[0] not in intervention_label.lower():
        flags.append("intervention_observed_effect_mismatch")
    confidence = float(np.clip(0.5 + 0.5 * strength, 0.0, 1.0))
    return label, confidence, False, flags
