"""Deterministic leakage-safe IID/OOD split utilities for Phase 15-style audits."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from triqto.core.ids import make_deterministic_id

SUPPORTED_HOLDOUT_AXES = {"family", "n_qubits", "distortion_type", "backend_id"}


@dataclass(frozen=True, slots=True)
class BackendHoldoutConfig:
    schema_version: str
    evaluation: str
    axis: str
    heldout_backend_ids: tuple[str, ...]
    claim_label: str = "exact_fake_backend_axis_holdout"
    seed: int = 2026
    backend_assignment_level: str = "clean_circuit"
    evidence_tier: str = "fake_backend_fixture"
    physical_hardware: bool = False

    def __post_init__(self) -> None:
        if self.axis != "backend_id":
            raise ValueError("backend holdout config axis must be backend_id")
        values = tuple(sorted(str(value) for value in self.heldout_backend_ids))
        if not values or any(not value for value in values):
            raise ValueError("heldout_backend_ids must contain at least one backend id")
        if self.backend_assignment_level != "clean_circuit":
            raise ValueError("backend holdout requires clean_circuit backend assignment")
        if self.evidence_tier != "fake_backend_fixture":
            raise ValueError("backend holdout evidence_tier must be fake_backend_fixture")
        if self.physical_hardware is not False:
            raise ValueError("fake-backend holdout config must not claim physical hardware")
        if "physical" in self.claim_label.lower() or "hardware generalization" in self.claim_label.lower():
            raise ValueError("backend holdout claim label must not imply physical hardware generalization")
        object.__setattr__(self, "heldout_backend_ids", values)

    @property
    def split_definition(self) -> SplitDefinition:
        return SplitDefinition(axis="backend_id", heldout_values=self.heldout_backend_ids, split_name=self.claim_label, seed=self.seed)


def load_backend_holdout_config(path: str | Path) -> BackendHoldoutConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("backend holdout config must be a mapping")
    allowed = {
        "schema_version",
        "evaluation",
        "axis",
        "heldout_backend_ids",
        "claim_label",
        "seed",
        "backend_assignment_level",
        "evidence_tier",
        "physical_hardware",
    }
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"unknown backend holdout config fields: {sorted(extra)}")
    return BackendHoldoutConfig(**dict(payload))


def _backend_evidence_available(record: Mapping[str, object]) -> bool:
    if record.get("backend_available") is True:
        return True
    mask = record.get("backend_feature_available_mask")
    if isinstance(mask, Sequence) and not isinstance(mask, (str, bytes)):
        return any(bool(value) for value in mask)
    return False


def audit_backend_holdout_for_phase15(records: Sequence[Mapping[str, object]], config: BackendHoldoutConfig) -> dict[str, object]:
    """Independently re-audit exact fake-backend-axis holdout metadata for Phase 15.

    This function intentionally recomputes splits from sample records rather than trusting
    preexisting training/evaluation artifacts. It fails closed when backend evidence is
    absent, backend identities overlap train/validation, or the claim would be mislabeled
    as physical-hardware generalization.
    """
    if not records:
        raise ValueError("backend holdout audit records must be non-empty")
    for record in records:
        if not _backend_evidence_available(record):
            raise ValueError("backend holdout requires explicit backend evidence availability")
    assignment = assign_axis_holdout(records, config.split_definition)
    clean_audit = audit_backend_clean_assignment(records)
    disjoint = audit_axis_disjointness(records, assignment, config.split_definition)
    train_validation_ids = {
        _value(record, "backend_id")
        for record in records
        if assignment[str(record.get("sample_id") or "")] in {"train", "validation"}
    }
    heldout = set(config.heldout_backend_ids)
    if train_validation_ids & heldout:
        raise ValueError("train/validation backend ids overlap held-out backend ids")
    return {
        "phase15_backend_holdout_audit_schema": "triqto.phase15.backend_holdout_audit.v1",
        "evaluation": config.evaluation,
        "claim_label": config.claim_label,
        "heldout_axis": "backend_id",
        "heldout_backend_ids": list(config.heldout_backend_ids),
        "evidence_tier": config.evidence_tier,
        "physical_hardware": False,
        "not_physical_hardware_generalization": True,
        "assignment": assignment,
        **clean_audit,
        **{f"disjointness_{key}": value for key, value in disjoint.items()},
    }


@dataclass(frozen=True, slots=True)
class SplitDefinition:
    axis: str
    heldout_values: tuple[str, ...]
    split_name: str
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.axis not in SUPPORTED_HOLDOUT_AXES:
            raise ValueError(f"unsupported holdout axis: {self.axis}")
        values = tuple(sorted(str(value) for value in self.heldout_values))
        if not values:
            raise ValueError("heldout_values must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError("heldout_values must be unique")
        if not self.split_name or not isinstance(self.split_name, str):
            raise ValueError("split_name must be nonblank")
        object.__setattr__(self, "heldout_values", values)

    @property
    def split_id(self) -> str:
        return make_deterministic_id(
            "splitdef",
            {"axis": self.axis, "heldout_values": list(self.heldout_values), "split_name": self.split_name, "seed": self.seed},
        )


def _value(record: Mapping[str, object], axis: str) -> str:
    if axis == "n_qubits":
        key = "n_qubits"
    else:
        key = axis
    if key not in record:
        raise ValueError(f"record missing split axis {key!r}")
    value = record[key]
    if value is None or value == "":
        raise ValueError(f"record has empty split axis {key!r}")
    return str(value)


def _lineage(record: Mapping[str, object]) -> str:
    for key in ("lineage_id", "clean_circuit_id", "family_clean_hash", "sample_id"):
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError("record must provide lineage_id, clean_circuit_id, family_clean_hash, or sample_id")


def assign_axis_holdout(records: Sequence[Mapping[str, object]], definition: SplitDefinition) -> dict[str, str]:
    """Assign train/validation/test labels for one audited OOD holdout axis.

    Held-out axis values are assigned exclusively to ``test``. Remaining lineage groups
    are deterministically split into train/validation by stable hash. This function
    fails closed when a holdout is impossible or when one lineage appears on both sides.
    """
    if not records:
        raise ValueError("records must be non-empty")
    heldout = set(definition.heldout_values)
    seen_values = {_value(record, definition.axis) for record in records}
    missing = heldout - seen_values
    if missing:
        raise ValueError(f"heldout values absent from records: {sorted(missing)}")
    if seen_values <= heldout:
        raise ValueError("holdout would leave no train/validation records")

    lineage_to_holdout: dict[str, bool] = {}
    sample_to_split: dict[str, str] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            raise ValueError("every record must include sample_id")
        in_holdout = _value(record, definition.axis) in heldout
        lineage = _lineage(record)
        previous = lineage_to_holdout.get(lineage)
        if previous is None:
            lineage_to_holdout[lineage] = in_holdout
        elif previous != in_holdout:
            raise ValueError(f"lineage {lineage} crosses heldout boundary")
        if in_holdout:
            sample_to_split[sample_id] = "test"
        else:
            bucket = make_deterministic_id("splitbucket", {"split_id": definition.split_id, "lineage": lineage})[-2:]
            sample_to_split[sample_id] = "validation" if int(bucket, 16) < 32 else "train"

    counts = Counter(sample_to_split.values())
    if counts["test"] == 0 or counts["train"] == 0:
        raise ValueError("split must contain non-empty train and test partitions")
    return dict(sorted(sample_to_split.items()))


def audit_axis_disjointness(records: Sequence[Mapping[str, object]], assignment: Mapping[str, str], definition: SplitDefinition) -> dict[str, object]:
    by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    lineage_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        split = assignment.get(sample_id)
        if split not in by_split:
            raise ValueError(f"missing or invalid split for sample {sample_id}")
        by_split[split].add(_value(record, definition.axis))
        lineage_by_split[split].add(_lineage(record))
    heldout = set(definition.heldout_values)
    if not by_split["test"] <= heldout:
        raise ValueError("test split contains non-heldout axis values")
    if (by_split["train"] | by_split["validation"]) & heldout:
        raise ValueError("train/validation contain heldout axis values")
    if lineage_by_split["test"] & (lineage_by_split["train"] | lineage_by_split["validation"]):
        raise ValueError("lineage leakage across test and train/validation")
    return {
        "split_id": definition.split_id,
        "claim_label": f"ood_{definition.axis}",
        "heldout_axis": definition.axis,
        "heldout_values": list(definition.heldout_values),
        "split_counts": dict(sorted(Counter(assignment.values()).items())),
        "audited_disjointness": True,
    }


def audit_backend_clean_assignment(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Verify backend identities are assigned at clean-circuit/lineage level.

    This prevents backend evidence from becoming a per-distortion, per-sample-order, or
    per-label side channel. The audit is intentionally conservative: every sample in a
    clean-circuit lineage must expose the same backend_id and a machine-readable
    backend_assignment_level of ``clean_circuit`` when present.
    """
    if not records:
        raise ValueError("records must be non-empty")
    lineage_to_backend: dict[str, str] = {}
    for record in records:
        lineage = _lineage(record)
        backend_id = record.get("backend_id")
        if not isinstance(backend_id, str) or not backend_id:
            raise ValueError("every backend-holdout record must include backend_id")
        level = record.get("backend_assignment_level", "clean_circuit")
        if level != "clean_circuit":
            raise ValueError("backend_assignment_level must be clean_circuit")
        previous = lineage_to_backend.get(lineage)
        if previous is None:
            lineage_to_backend[lineage] = backend_id
        elif previous != backend_id:
            raise ValueError(f"clean-circuit lineage {lineage} has multiple backend_ids")
    counts = Counter(lineage_to_backend.values())
    if len(counts) < 1:
        raise ValueError("no backend assignments were audited")
    return {
        "backend_assignment_level": "clean_circuit",
        "audited_backend_assignment": True,
        "clean_lineage_count": len(lineage_to_backend),
        "backend_lineage_counts": dict(sorted(counts.items())),
    }


def assign_iid_split(records: Sequence[Mapping[str, object]], *, seed: int = 2026) -> dict[str, str]:
    """Deterministic lineage-grouped IID split labeled explicitly as iid_test."""
    if not records:
        raise ValueError("records must be non-empty")
    sample_to_split: dict[str, str] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            raise ValueError("every record must include sample_id")
        bucket = int(make_deterministic_id("iidsplit", {"seed": seed, "lineage": _lineage(record)})[-2:], 16)
        sample_to_split[sample_id] = "iid_test" if bucket < 26 else ("validation" if bucket < 52 else "train")
    return dict(sorted(sample_to_split.items()))


__all__ = [
    "SUPPORTED_HOLDOUT_AXES",
    "BackendHoldoutConfig",
    "SplitDefinition",
    "assign_axis_holdout",
    "assign_iid_split",
    "audit_axis_disjointness",
    "audit_backend_clean_assignment",
    "audit_backend_holdout_for_phase15",
    "load_backend_holdout_config",
]
