"""Schema, probability, and count validation primitives."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from .config import PreprocessingConfig
from .records import ValidationFinding


class RecordQuarantined(ValueError):
    """Internal marker for records that cannot enter training-facing outputs."""


class ValidationCollector:
    def __init__(self) -> None:
        self.findings: list[ValidationFinding] = []

    def add(self, rule_id: str, severity: str, field_path: str, observed: Any,
            expected: str, disposition: str, *, repair_applied: bool = False) -> None:
        summary = repr(observed)
        if len(summary) > 500:
            summary = summary[:497] + "..."
        self.findings.append(ValidationFinding(
            rule_id=rule_id, severity=severity, field_path=field_path,
            observed_summary=summary, expected_constraint=expected,
            disposition=disposition, repair_applied=repair_applied,
        ))

    @property
    def quarantined(self) -> bool:
        return any(item.disposition == "quarantine" for item in self.findings)

    @property
    def repaired(self) -> bool:
        return any(item.repair_applied for item in self.findings)

    @property
    def warnings(self) -> bool:
        return any(item.severity == "warning" for item in self.findings)

    def disposition(self) -> str:
        if self.quarantined:
            return "quarantine"
        if self.repaired:
            return "repaired_with_audit"
        if self.warnings:
            return "pass_with_warning"
        return "pass"


def _finite_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))


def validate_manifest_record(record: Any, *, record_name: str, collector: ValidationCollector) -> None:
    try:
        record.validate()
    except Exception as exc:
        collector.add("schema.record_validate", "error", record_name, str(exc),
                      "existing TriQTO manifest schema validation must pass", "quarantine")
    metadata = getattr(record, "metadata", {})
    if metadata is not None and not isinstance(metadata, Mapping):
        collector.add("schema.metadata_mapping", "error", f"{record_name}.metadata",
                      type(metadata).__name__, "metadata must be a mapping", "quarantine")


def validate_probability_distribution(probabilities: Mapping[str, Any], *, width: int,
                                      config: PreprocessingConfig, collector: ValidationCollector,
                                      field_path: str) -> dict[str, float]:
    tol = config.numerical_tolerances
    if not isinstance(probabilities, Mapping) or not probabilities:
        collector.add("physics.probability_mapping", "error", field_path,
                      type(probabilities).__name__, "nonempty probability mapping", "quarantine")
        return {}
    result: dict[str, float] = {}
    for raw_key, raw_value in probabilities.items():
        key = str(raw_key).replace(" ", "")
        if len(key) > width or any(ch not in "01" for ch in key):
            collector.add("schema.probability_bitstring_width", "error", f"{field_path}.{raw_key}",
                          raw_key, f"binary outcome width <= {width}", "quarantine")
            continue
        if not _finite_numeric(raw_value):
            collector.add("numerical.probability_finite", "error", f"{field_path}.{raw_key}",
                          raw_value, "finite probability", "quarantine")
            continue
        value = float(raw_value)
        if value < -tol.probability_negative_repair:
            collector.add("physics.probability_nonnegative", "error", f"{field_path}.{raw_key}",
                          value, "probability >= 0", "quarantine")
            continue
        if value < 0.0:
            if config.validation.repair_small_numerical_drift:
                collector.add("physics.probability_tiny_negative", "warning", f"{field_path}.{raw_key}",
                              value, "tiny negative within repair tolerance", "repaired_with_audit",
                              repair_applied=True)
                value = 0.0
            else:
                collector.add("physics.probability_tiny_negative", "error", f"{field_path}.{raw_key}",
                              value, "probability >= 0", "quarantine")
        padded = key.zfill(width)
        result[padded] = result.get(padded, 0.0) + value
    total = float(sum(result.values()))
    deviation = abs(total - 1.0)
    if deviation > tol.probability_sum_repair:
        collector.add("physics.probability_sum", "error", field_path, total,
                      "probabilities sum to 1 within repair tolerance", "quarantine")
    elif deviation > tol.probability_sum_warning:
        if config.validation.repair_small_numerical_drift and total > 0.0:
            result = {key: value / total for key, value in result.items()}
            collector.add("physics.probability_sum_repair", "warning", field_path, total,
                          "small floating-point deviation from one", "repaired_with_audit",
                          repair_applied=True)
        else:
            collector.add("physics.probability_sum_warning", "warning", field_path, total,
                          "probabilities should sum to one", "pass_with_warning")
    return result


def validate_counts(counts: Mapping[str, Any], *, width: int, declared_shots: int | None,
                    collector: ValidationCollector, field_path: str) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        collector.add("schema.counts_mapping", "error", field_path, type(counts).__name__,
                      "outcome-to-count mapping", "quarantine")
        return {}
    result: dict[str, int] = {}
    for raw_key, raw_count in counts.items():
        key = str(raw_key).replace(" ", "")
        if len(key) > width or any(ch not in "01" for ch in key):
            collector.add("schema.count_bitstring_width", "error", f"{field_path}.{raw_key}", raw_key,
                          f"binary outcome width <= {width}", "quarantine")
            continue
        if not isinstance(raw_count, (int, np.integer)) or isinstance(raw_count, (bool, np.bool_)) or int(raw_count) < 0:
            collector.add("schema.count_nonnegative_integer", "error", f"{field_path}.{raw_key}",
                          raw_count, "nonnegative integer count", "quarantine")
            continue
        padded = key.zfill(width)
        result[padded] = result.get(padded, 0) + int(raw_count)
    total = sum(result.values())
    if declared_shots is not None:
        if not isinstance(declared_shots, int) or isinstance(declared_shots, bool) or declared_shots < 0:
            collector.add("schema.shot_count", "error", f"{field_path}.declared_shots",
                          declared_shots, "nonnegative integer shot count", "quarantine")
        elif total != declared_shots:
            collector.add("physics.counts_match_shots", "error", field_path,
                          {"sum_counts": total, "declared_shots": declared_shots},
                          "sum(counts) == declared shots", "quarantine")
    return result
