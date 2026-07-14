"""Identifiability contracts for generated supervised targets."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

IDENTIFIABLE = "identifiable"
CONDITIONALLY_IDENTIFIABLE = "conditionally_identifiable"
UNIDENTIFIABLE = "unidentifiable"


@dataclass(frozen=True, slots=True)
class IdentifiabilityAssessment:
    status: str
    reason: str
    diagnosis_supervision_mask: bool
    action_supervision_mask: bool
    born_target_mask: bool = True

    def to_metadata(self) -> dict[str, object]:
        return {
            "identifiability_status": self.status,
            "identifiability_reason": self.reason,
            "diagnosis_supervision_mask": self.diagnosis_supervision_mask,
            "action_supervision_mask": self.action_supervision_mask,
            "born_target_mask": self.born_target_mask,
        }


def assess_identifiability(*, distortion_type: str, distortion_metadata: Mapping[str, object], measurement_bases: tuple[str, ...], born_zero_shift: bool) -> IdentifiabilityAssessment:
    bases = {basis.upper() for basis in measurement_bases}
    marker_only = bool(distortion_metadata.get("marker_only", False))
    if marker_only:
        return IdentifiabilityAssessment(UNIDENTIFIABLE, "marker_only_no_observable_change", False, False)
    if distortion_type in {"readout_bitflip_marker", "layout_permutation_marker"}:
        return IdentifiabilityAssessment(UNIDENTIFIABLE, "marker_only_no_observable_change", False, False)
    if distortion_type == "phase_rz_drift" and born_zero_shift and bases == {"Z"}:
        return IdentifiabilityAssessment(UNIDENTIFIABLE, "computational_basis_phase_blindness", False, False)
    if born_zero_shift and bases <= {"Z"}:
        return IdentifiabilityAssessment(UNIDENTIFIABLE, "insufficient_measurement_settings", False, False)
    if distortion_type == "phase_rz_drift" and {"X", "Y"} & bases:
        return IdentifiabilityAssessment(CONDITIONALLY_IDENTIFIABLE, "requires_non_z_measurement_setting", True, True)
    return IdentifiabilityAssessment(IDENTIFIABLE, "observable_born_shift", True, True)


def coverage_summary(metadata_rows: list[Mapping[str, object]]) -> dict[str, object]:
    statuses = Counter(str(row.get("identifiability_status", "unknown")) for row in metadata_rows)
    reasons = Counter(str(row.get("identifiability_reason", "unknown")) for row in metadata_rows)
    supervised = sum(bool(row.get("diagnosis_supervision_mask", False)) for row in metadata_rows)
    total = len(metadata_rows)
    return {
        "total_targets": total,
        "diagnosis_supervised_targets": supervised,
        "identifiable_coverage": supervised / total if total else 0.0,
        "status_counts": dict(sorted(statuses.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }
