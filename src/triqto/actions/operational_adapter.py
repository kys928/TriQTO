"""Leakage-safe operational-action adapter for Phase 12/14 candidate tensors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from torch import Tensor
import yaml

from triqto.model import ACTION_EDIT_TYPES, ActionCandidateTensorBatch
from .operational import OperationalActionResult

OPERATIONAL_VIEW_ADAPTER_SCHEMA = "triqto.operational_view_adapter.v1"
OPERATIONAL_ACTION_FEATURE_NAMES = (
    "depth_delta",
    "size_delta",
    "two_qubit_gate_delta",
    "acquires_evidence",
    "is_no_op",
)
OPERATIONAL_ACTION_FAMILIES = (
    "logical_correction",
    "diagnostic_evidence_acquisition",
    "compilation",
    "semantics_preserving_optimization",
)
_FAMILY_ID = {name: index for index, name in enumerate(OPERATIONAL_ACTION_FAMILIES)}


@dataclass(frozen=True, slots=True)
class OperationalViewAdapterConfig:
    schema_version: str = OPERATIONAL_VIEW_ADAPTER_SCHEMA
    feature_names: tuple[str, ...] = OPERATIONAL_ACTION_FEATURE_NAMES
    family_order: tuple[str, ...] = OPERATIONAL_ACTION_FAMILIES
    require_availability_mask: bool = True
    require_zero_operational_targets: bool = True
    require_no_privilege: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != OPERATIONAL_VIEW_ADAPTER_SCHEMA:
            raise ValueError("unsupported operational view adapter schema")
        if tuple(self.feature_names) != OPERATIONAL_ACTION_FEATURE_NAMES:
            raise ValueError("operational feature names/order are versioned and fixed")
        if tuple(self.family_order) != OPERATIONAL_ACTION_FAMILIES:
            raise ValueError("operational family order is versioned and fixed")
        for name in ("require_availability_mask", "require_zero_operational_targets", "require_no_privilege"):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must remain true")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "family_order", tuple(self.family_order))


def operational_view_adapter_config_to_dict(config: OperationalViewAdapterConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["feature_names"] = list(config.feature_names)
    payload["family_order"] = list(config.family_order)
    return payload


def load_operational_view_adapter_config(path: str | Path) -> OperationalViewAdapterConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("operational view adapter config must contain a mapping")
    allowed = set(OperationalViewAdapterConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    if set(payload) - allowed:
        raise ValueError(f"unknown operational view adapter fields: {sorted(set(payload) - allowed)}")
    data = dict(payload)
    for name in ("feature_names", "family_order"):
        if name in data:
            data[name] = tuple(data[name])
    return OperationalViewAdapterConfig(**data)


@dataclass(slots=True)
class OperationalActionTensorBatch:
    model_candidates: ActionCandidateTensorBatch
    candidate_ids: tuple[str, ...]
    candidate_family_ids: Tensor
    candidate_target_mask: Tensor
    privileged_information_mask: Tensor

    def validate(self, graph_count: int) -> None:
        count = self.model_candidates.candidate_features.shape[0]
        if len(self.candidate_ids) != count or len(set(self.candidate_ids)) != count:
            raise ValueError("candidate IDs must be unique and match rows")
        if self.candidate_family_ids.dtype != torch.long or self.candidate_family_ids.shape != (count,):
            raise ValueError("family IDs must be int64 with candidate shape")
        for value in (self.candidate_target_mask, self.privileged_information_mask):
            if value.dtype != torch.bool or value.shape != (count,):
                raise ValueError("operational masks must be bool with candidate shape")
        if bool(self.candidate_target_mask.any()) or bool(self.privileged_information_mask.any()):
            raise ValueError("operational candidates cannot use logical targets or privilege")
        batch = self.model_candidates.candidate_batch
        if count and (int(batch.min()) < 0 or int(batch.max()) >= graph_count):
            raise ValueError("candidate graph index is out of range")
        unavailable = ~self.model_candidates.candidate_available_mask
        if bool((self.model_candidates.candidate_features[unavailable] != 0).any()):
            raise ValueError("unavailable candidate features must be zero")
        edit_owner = self.model_candidates.edit_candidate_index
        if edit_owner.numel() and bool(unavailable.index_select(0, edit_owner).any()):
            raise ValueError("unavailable candidates cannot own edit rows")


def _family(result: OperationalActionResult) -> str:
    if result.action_type == "basis_probe":
        return "diagnostic_evidence_acquisition"
    if result.action_type in {"layout_selection", "routing_transpilation"}:
        return "compilation"
    if result.action_type == "depth_reduction":
        return "semantics_preserving_optimization"
    raise ValueError(f"unsupported operational action {result.action_type!r}")


def _delta(result: OperationalActionResult, name: str) -> float:
    before = float(result.before_metadata.get(name, 0.0))
    return float(result.after_metadata.get(name, before)) - before


def _text(values: Sequence[str]) -> np.ndarray:
    width = max([1, *[len(value) for value in values]])
    return np.asarray(values, dtype=f"<U{width}")


def operational_actions_to_phase12_arrays(
    results: Sequence[OperationalActionResult],
    config: OperationalViewAdapterConfig | None = None,
) -> dict[str, np.ndarray]:
    adapter_config = config or OperationalViewAdapterConfig()
    if not isinstance(adapter_config, OperationalViewAdapterConfig):
        raise TypeError("config must be OperationalViewAdapterConfig or None")
    ordered = sorted(results, key=lambda value: value.action_id)
    if not ordered:
        raise ValueError("operational adapter requires at least one action")
    ids, features, available, families = [], [], [], []
    edit_ptr, edit_types, edit_qubit_ptr, edit_qubits = [0], [], [0], []
    edit_map = {"basis_probe": "diagnostic_basis", "layout_selection": "layout", "routing_transpilation": "routing", "depth_reduction": None}
    for result in ordered:
        ids.append(result.action_id)
        row_available = bool(result.availability_mask)
        available.append(row_available)
        family = _family(result)
        families.append(_FAMILY_ID[family])
        row = [
            _delta(result, "depth"),
            _delta(result, "size"),
            _delta(result, "two_qubit_gate_count"),
            float(result.acquires_evidence),
            float(result.status == "no_op"),
        ]
        features.append(row if row_available else [0.0] * len(adapter_config.feature_names))
        edit_type = edit_map[result.action_type]
        if row_available and edit_type is not None:
            if edit_type not in ACTION_EDIT_TYPES:
                raise ValueError(f"missing model action vocabulary {edit_type!r}")
            edit_types.append(edit_type)
            edit_qubits.append(0)
            edit_qubit_ptr.append(len(edit_qubits))
        edit_ptr.append(len(edit_types))
    family_names = [_family(value) for value in ordered]
    return {
        "action_candidate_ids": _text(ids),
        "action_candidate_feature_names": _text(adapter_config.feature_names),
        "action_candidate_features": np.asarray(features, dtype=np.float64),
        "action_candidate_available_mask": np.asarray(available, dtype=np.bool_),
        "action_candidate_family_ids": np.asarray(families, dtype=np.int64),
        "action_candidate_family_names": _text(family_names),
        "action_candidate_target_mask": np.zeros(len(ordered), dtype=np.bool_),
        "action_privileged_oracle_mask": np.zeros(len(ordered), dtype=np.bool_),
        "action_edit_ptr": np.asarray(edit_ptr, dtype=np.int64),
        "action_edit_types": _text(edit_types),
        "action_edit_magnitudes": np.zeros(len(edit_types), dtype=np.float64),
        "action_edit_qubit_ptr": np.asarray(edit_qubit_ptr, dtype=np.int64),
        "action_edit_qubits": np.asarray(edit_qubits, dtype=np.int64),
    }


def build_operational_action_tensor_batch(
    results: Sequence[OperationalActionResult],
    *,
    graph_index: int = 0,
    config: OperationalViewAdapterConfig | None = None,
) -> OperationalActionTensorBatch:
    arrays = operational_actions_to_phase12_arrays(results, config)
    count = len(arrays["action_candidate_ids"])
    names = [str(value) for value in arrays["action_edit_types"].tolist()]
    edit_positions = {name: index for index, name in enumerate(ACTION_EDIT_TYPES)}
    owner: list[int] = []
    ptr = arrays["action_edit_ptr"]
    for candidate in range(count):
        owner.extend([candidate] * (int(ptr[candidate + 1]) - int(ptr[candidate])))
    result = OperationalActionTensorBatch(
        model_candidates=ActionCandidateTensorBatch(
            candidate_features=torch.as_tensor(arrays["action_candidate_features"], dtype=torch.float32),
            candidate_batch=torch.full((count,), graph_index, dtype=torch.long),
            candidate_available_mask=torch.as_tensor(arrays["action_candidate_available_mask"], dtype=torch.bool),
            edit_type_ids=torch.tensor([edit_positions[name] for name in names], dtype=torch.long),
            edit_magnitudes=torch.zeros(len(names), dtype=torch.float32),
            edit_qubit_positions=torch.zeros(len(names), dtype=torch.float32),
            edit_candidate_index=torch.tensor(owner, dtype=torch.long),
        ),
        candidate_ids=tuple(str(value) for value in arrays["action_candidate_ids"].tolist()),
        candidate_family_ids=torch.as_tensor(arrays["action_candidate_family_ids"], dtype=torch.long),
        candidate_target_mask=torch.zeros(count, dtype=torch.bool),
        privileged_information_mask=torch.zeros(count, dtype=torch.bool),
    )
    result.validate(graph_index + 1)
    return result


def collate_operational_action_tensor_batches(batches: Sequence[OperationalActionTensorBatch]) -> OperationalActionTensorBatch:
    if not batches:
        raise ValueError("cannot collate empty operational batches")
    offsets, total = [], 0
    for batch in batches:
        offsets.append(total)
        total += len(batch.candidate_ids)
    model = ActionCandidateTensorBatch(
        candidate_features=torch.cat([value.model_candidates.candidate_features for value in batches]),
        candidate_batch=torch.cat([torch.full((len(value.candidate_ids),), index, dtype=torch.long) for index, value in enumerate(batches)]),
        candidate_available_mask=torch.cat([value.model_candidates.candidate_available_mask for value in batches]),
        edit_type_ids=torch.cat([value.model_candidates.edit_type_ids for value in batches]),
        edit_magnitudes=torch.cat([value.model_candidates.edit_magnitudes for value in batches]),
        edit_qubit_positions=torch.cat([value.model_candidates.edit_qubit_positions for value in batches]),
        edit_candidate_index=torch.cat([value.model_candidates.edit_candidate_index + offsets[index] for index, value in enumerate(batches)]),
    )
    result = OperationalActionTensorBatch(
        model_candidates=model,
        candidate_ids=tuple(item for value in batches for item in value.candidate_ids),
        candidate_family_ids=torch.cat([value.candidate_family_ids for value in batches]),
        candidate_target_mask=torch.cat([value.candidate_target_mask for value in batches]),
        privileged_information_mask=torch.cat([value.privileged_information_mask for value in batches]),
    )
    result.validate(len(batches))
    return result


__all__ = [
    "OPERATIONAL_ACTION_FAMILIES",
    "OPERATIONAL_ACTION_FEATURE_NAMES",
    "OPERATIONAL_VIEW_ADAPTER_SCHEMA",
    "OperationalActionTensorBatch",
    "OperationalViewAdapterConfig",
    "build_operational_action_tensor_batch",
    "collate_operational_action_tensor_batches",
    "load_operational_view_adapter_config",
    "operational_actions_to_phase12_arrays",
    "operational_view_adapter_config_to_dict",
]
