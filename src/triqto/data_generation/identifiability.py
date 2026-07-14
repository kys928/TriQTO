"""Machine-enforced diagnosis identifiability contracts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from qiskit import QuantumCircuit

from triqto.circuits.serialization import circuit_to_qasm3_text
from triqto.core.ids import make_deterministic_id
from triqto.simulation import MeasurementSetting

IDENTIFIABILITY_STATUSES = (
    "identifiable",
    "conditionally_identifiable",
    "unidentifiable",
)
UNIDENTIFIABLE_REASONS = (
    "marker_only_no_observable_change",
    "computational_basis_phase_blindness",
    "insufficient_measurement_settings",
    "backend_feature_unavailable",
)
CONDITIONAL_IDENTIFIABILITY_REASONS = (
    "requires_selected_measurement_settings",
)
UNIDENTIFIABLE_POLICIES = ("mask", "error", "allow")


@dataclass(frozen=True, slots=True)
class IdentifiabilityAssessment:
    """Diagnosis supervision status under the declared observable evidence."""

    status: str
    reason: str | None
    visible_measurement_setting_ids: tuple[str, ...]
    blind_measurement_setting_ids: tuple[str, ...]
    maximum_total_variation: float

    @property
    def diagnosis_supervised_by_default(self) -> bool:
        return self.status != "unidentifiable"


def _total_variation(
    clean: Mapping[str, Any],
    distorted: Mapping[str, Any],
) -> float:
    support = set(clean) | set(distorted)
    value = 0.5 * math.fsum(
        abs(float(clean.get(outcome, 0.0)) - float(distorted.get(outcome, 0.0)))
        for outcome in support
    )
    if not math.isfinite(value):
        raise ValueError("measurement evidence produced non-finite total variation")
    return value


def assess_identifiability(
    *,
    distortion_type: str,
    marker_only: bool,
    measurement_settings: Mapping[str, MeasurementSetting],
    clean_probabilities: Mapping[str, Mapping[str, Any]],
    distorted_probabilities: Mapping[str, Mapping[str, Any]],
    atol: float,
) -> IdentifiabilityAssessment:
    """Assess whether the allowed ``(circuit, M, Born)`` evidence can carry a label."""
    if not isinstance(distortion_type, str) or not distortion_type.strip():
        raise ValueError("distortion_type must be nonblank")
    if not isinstance(marker_only, bool):
        raise TypeError("marker_only must be bool")
    if isinstance(atol, bool) or not isinstance(atol, (int, float)):
        raise TypeError("atol must be numeric and not bool")
    tolerance = float(atol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    setting_ids = tuple(sorted(measurement_settings))
    if not setting_ids:
        raise ValueError("at least one measurement setting is required")
    if set(clean_probabilities) != set(setting_ids):
        raise ValueError("clean probability settings do not match measurement settings")
    if set(distorted_probabilities) != set(setting_ids):
        raise ValueError("distorted probability settings do not match measurement settings")
    shifts = {
        setting_id: _total_variation(
            clean_probabilities[setting_id],
            distorted_probabilities[setting_id],
        )
        for setting_id in setting_ids
    }
    visible = tuple(setting_id for setting_id in setting_ids if shifts[setting_id] > tolerance)
    blind = tuple(setting_id for setting_id in setting_ids if shifts[setting_id] <= tolerance)
    maximum = max(shifts.values(), default=0.0)
    if marker_only:
        reason = (
            "backend_feature_unavailable"
            if "layout" in distortion_type
            else "marker_only_no_observable_change"
        )
        return IdentifiabilityAssessment(
            "unidentifiable",
            reason,
            visible,
            blind,
            maximum,
        )
    if not visible:
        all_computational = all(
            set(measurement_settings[setting_id].bases) == {"Z"}
            for setting_id in setting_ids
        )
        reason = (
            "computational_basis_phase_blindness"
            if "phase" in distortion_type and all_computational
            else "insufficient_measurement_settings"
        )
        return IdentifiabilityAssessment(
            "unidentifiable",
            reason,
            visible,
            blind,
            maximum,
        )
    if blind:
        return IdentifiabilityAssessment(
            "conditionally_identifiable",
            "requires_selected_measurement_settings",
            visible,
            blind,
            maximum,
        )
    return IdentifiabilityAssessment(
        "identifiable",
        None,
        visible,
        blind,
        maximum,
    )


def observable_evidence_fingerprint(
    circuit: QuantumCircuit,
    *,
    measurement_settings: Mapping[str, MeasurementSetting],
    probabilities: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash only deployable evidence, excluding labels, IDs, paths, and provenance."""
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be QuantumCircuit")
    if set(measurement_settings) != set(probabilities):
        raise ValueError("probability settings do not match measurement settings")
    payload = {
        "circuit_qasm3": circuit_to_qasm3_text(circuit),
        "measurement_evidence": [
            {
                "setting": list(measurement_settings[setting_id].bases),
                "probabilities": {
                    str(outcome): float(value)
                    for outcome, value in sorted(probabilities[setting_id].items())
                },
            }
            for setting_id in sorted(measurement_settings)
        ],
    }
    return make_deterministic_id("observable_diagnosis_evidence", payload)


def reject_conflicting_identifiable_labels(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject conflicting supervised labels attached to identical allowed evidence."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{index}] must be a mapping")
        fingerprint = row.get("observable_evidence_fingerprint")
        label = row.get("distortion_type")
        status = row.get("identifiability_status")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"rows[{index}] has invalid observable_evidence_fingerprint")
        if not isinstance(label, str) or not label:
            raise ValueError(f"rows[{index}] has invalid distortion_type")
        if status not in IDENTIFIABILITY_STATUSES:
            raise ValueError(f"rows[{index}] has invalid identifiability_status")
        grouped.setdefault(fingerprint, []).append((label, status))
    for fingerprint, values in grouped.items():
        labels = {label for label, _ in values}
        if len(labels) <= 1:
            continue
        supervised = [(label, status) for label, status in values if status != "unidentifiable"]
        if supervised:
            raise ValueError(
                "Conflicting supervised diagnosis labels share identical allowed evidence: "
                f"fingerprint={fingerprint}, labels={sorted(labels)}"
            )


def validate_unidentifiable_policy(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("unidentifiable_policy must be a string")
    normalized = value.strip().lower()
    if normalized not in UNIDENTIFIABLE_POLICIES:
        raise ValueError(
            f"unidentifiable_policy must be one of {UNIDENTIFIABLE_POLICIES}"
        )
    return normalized


__all__ = [
    "CONDITIONAL_IDENTIFIABILITY_REASONS",
    "IDENTIFIABILITY_STATUSES",
    "UNIDENTIFIABLE_POLICIES",
    "UNIDENTIFIABLE_REASONS",
    "IdentifiabilityAssessment",
    "assess_identifiability",
    "observable_evidence_fingerprint",
    "reject_conflicting_identifiable_labels",
    "validate_unidentifiable_policy",
]
