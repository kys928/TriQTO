"""Integrity validation for Phase 9 actions, circuits, and rollout evidence."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from triqto.graph import validate_probability_arrays

from .config import ActionEngineConfig
from .constants import GENERATION_SOURCES, PRIMARY_REWARD_METRICS, SUPPORTED_EDIT_TYPES
from .identities import (
    action_content_hash,
    action_risk_from_edits,
    action_rollout_id_from_config_id,
    action_scientific_config_id,
    candidate_action_id,
    candidate_circuit_id,
    circuit_semantic_hash,
    rollout_content_hash,
)
from .models import ActionCandidate, ActionEdit, ActionRollout, AppliedAction


def _require_nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _require_real_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def validate_action_edit(
    edit: ActionEdit,
    config: ActionEngineConfig,
    *,
    n_qubits: int | None = None,
) -> None:
    """Validate one primitive bounded circuit edit."""
    if not isinstance(edit, ActionEdit):
        raise TypeError("edit must be ActionEdit")
    if edit.edit_type not in SUPPORTED_EDIT_TYPES:
        raise ValueError(f"Unsupported action edit type {edit.edit_type!r}")
    expected_arity = 2 if edit.edit_type == "append_rzz" else 1
    if not isinstance(edit.qubits, tuple):
        raise TypeError("ActionEdit.qubits must be a tuple")
    if len(edit.qubits) != expected_arity:
        raise ValueError(
            f"{edit.edit_type} requires {expected_arity} qubit operand(s)"
        )
    if len(set(edit.qubits)) != len(edit.qubits):
        raise ValueError("ActionEdit qubits must be unique")
    for qubit in edit.qubits:
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError("ActionEdit qubit indices must be integers and not bool")
        if qubit < 0:
            raise ValueError("ActionEdit qubit indices must be nonnegative")
        if n_qubits is not None and qubit >= n_qubits:
            raise ValueError(
                f"ActionEdit qubit {qubit} is out of range for {n_qubits} qubits"
            )
    magnitude = _require_finite(edit.magnitude, "ActionEdit.magnitude")
    if abs(magnitude) > config.max_abs_angle + config.improvement_atol:
        raise ValueError("ActionEdit magnitude exceeds max_abs_angle")
    if magnitude == 0.0:
        raise ValueError("Zero-magnitude edits are not allowed; use the no-op action")


def validate_action_candidate(
    candidate: ActionCandidate,
    config: ActionEngineConfig,
    *,
    n_qubits: int | None = None,
    require_hash: bool = True,
) -> None:
    """Validate a complete candidate action and its deterministic identity."""
    if not isinstance(candidate, ActionCandidate):
        raise TypeError("candidate must be ActionCandidate")
    for name in (
        "action_id",
        "sample_id",
        "graph_pair_id",
        "source_circuit_id",
        "source_run_id",
        "distortion_id",
    ):
        _require_nonblank(getattr(candidate, name), f"ActionCandidate.{name}")
    if not isinstance(candidate.edits, tuple):
        raise TypeError("ActionCandidate.edits must be a tuple")
    if len(candidate.edits) > config.max_edits_per_action:
        raise ValueError("ActionCandidate exceeds max_edits_per_action")
    for edit in candidate.edits:
        validate_action_edit(edit, config, n_qubits=n_qubits)

    if not isinstance(candidate.generation_sources, tuple):
        raise TypeError("ActionCandidate.generation_sources must be a tuple")
    if not candidate.generation_sources:
        raise ValueError("ActionCandidate.generation_sources must not be empty")
    if candidate.generation_sources != tuple(sorted(candidate.generation_sources)):
        raise ValueError("ActionCandidate.generation_sources must be sorted")
    if len(set(candidate.generation_sources)) != len(candidate.generation_sources):
        raise ValueError("ActionCandidate.generation_sources must be unique")
    unknown = set(candidate.generation_sources) - set(GENERATION_SOURCES)
    if unknown:
        raise ValueError(f"Unknown action generation sources: {sorted(unknown)}")
    if not candidate.edits and "no_op" not in candidate.generation_sources:
        raise ValueError("An empty action must include the no_op generation source")
    if candidate.edits and candidate.generation_sources == ("no_op",):
        raise ValueError("A nonempty action cannot be generated only as no_op")

    risk = _require_finite(candidate.risk_score, "ActionCandidate.risk_score")
    if risk < 0.0 or risk > 1.0:
        raise ValueError("ActionCandidate.risk_score must be in [0, 1]")
    expected_risk = action_risk_from_edits(candidate.edits, config.max_abs_angle)
    if not math.isclose(risk, expected_risk, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("ActionCandidate.risk_score does not match its edit payload")
    if not isinstance(candidate.metadata, Mapping):
        raise TypeError("ActionCandidate.metadata must be a mapping")

    expected_id = candidate_action_id(
        sample_id=candidate.sample_id,
        graph_pair_id=candidate.graph_pair_id,
        source_circuit_id=candidate.source_circuit_id,
        source_run_id=candidate.source_run_id,
        edits=candidate.edits,
    )
    if candidate.action_id != expected_id:
        raise ValueError("ActionCandidate.action_id does not match its scientific payload")
    expected_hash = action_content_hash(candidate)
    if require_hash and candidate.content_hash != expected_hash:
        raise ValueError("ActionCandidate.content_hash mismatch")
    if not require_hash and candidate.content_hash not in {"", expected_hash}:
        raise ValueError("ActionCandidate.content_hash is malformed")


def validate_applied_action(
    applied: AppliedAction,
    candidate: ActionCandidate,
) -> None:
    """Validate circuit identity and structural metadata after action application."""
    if not isinstance(applied, AppliedAction):
        raise TypeError("applied must be AppliedAction")
    if applied.action_id != candidate.action_id:
        raise ValueError("AppliedAction.action_id mismatch")
    expected_circuit_id = candidate_circuit_id(
        candidate.source_circuit_id,
        candidate.action_id,
    )
    if applied.candidate_circuit_id != expected_circuit_id:
        raise ValueError("AppliedAction.candidate_circuit_id mismatch")
    for name in (
        "source_depth",
        "candidate_depth",
        "source_gate_count",
        "candidate_gate_count",
    ):
        value = getattr(applied, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"AppliedAction.{name} must be a nonnegative integer")
    if not isinstance(applied.decomposition_metadata, Mapping):
        raise TypeError("AppliedAction.decomposition_metadata must be a mapping")
    expected_hash = circuit_semantic_hash(applied.circuit)
    if applied.circuit_hash != expected_hash:
        raise ValueError("AppliedAction.circuit_hash mismatch")


def _require_float64_vector(array: Any, name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != np.float64 or array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional float64 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_action_rollout(
    rollout: ActionRollout,
    *,
    require_hash: bool = True,
) -> None:
    """Validate exact simulator evidence and deterministic ranking fields."""
    if not isinstance(rollout, ActionRollout):
        raise TypeError("rollout must be ActionRollout")
    for name in (
        "rollout_id",
        "action_id",
        "sample_id",
        "graph_pair_id",
        "candidate_circuit_id",
        "clean_target_run_id",
        "scientific_config_id",
    ):
        _require_nonblank(getattr(rollout, name), f"ActionRollout.{name}")
    expected_rollout_id = action_rollout_id_from_config_id(
        rollout.action_id,
        rollout.clean_target_run_id,
        rollout.scientific_config_id,
    )
    if rollout.rollout_id != expected_rollout_id:
        raise ValueError("ActionRollout.rollout_id identity mismatch")
    if isinstance(rollout.rank, bool) or not isinstance(rollout.rank, int):
        raise TypeError("ActionRollout.rank must be an integer and not bool")
    if rollout.rank <= 0:
        raise ValueError("ActionRollout.rank must be positive")
    _require_finite(rollout.reward, "ActionRollout.reward")
    risk = _require_finite(rollout.risk_score, "ActionRollout.risk_score")
    if risk < 0.0 or risk > 1.0:
        raise ValueError("ActionRollout.risk_score must be in [0, 1]")

    if not isinstance(rollout.metric_names, np.ndarray):
        raise TypeError("ActionRollout.metric_names must be a NumPy array")
    if rollout.metric_names.ndim != 1 or rollout.metric_names.dtype.kind != "U":
        raise TypeError("ActionRollout.metric_names must be one-dimensional Unicode")
    if tuple(rollout.metric_names.tolist()) != PRIMARY_REWARD_METRICS:
        raise ValueError("ActionRollout.metric_names must use the fixed Phase 9 order")
    if len(set(rollout.metric_names.tolist())) != rollout.metric_names.size:
        raise ValueError("ActionRollout.metric_names must be unique")

    baseline = _require_float64_vector(
        rollout.baseline_metric_values,
        "ActionRollout.baseline_metric_values",
    )
    candidate = _require_float64_vector(
        rollout.candidate_metric_values,
        "ActionRollout.candidate_metric_values",
    )
    improvement = _require_float64_vector(
        rollout.improvement_values,
        "ActionRollout.improvement_values",
    )
    if not (
        rollout.metric_names.size
        == baseline.size
        == candidate.size
        == improvement.size
    ):
        raise ValueError("ActionRollout metric arrays must have matching lengths")
    if not np.allclose(improvement, baseline - candidate, rtol=0.0, atol=1e-15):
        raise ValueError("ActionRollout improvement values must equal baseline-candidate")

    if not isinstance(rollout.outcome_bitstrings, np.ndarray):
        raise TypeError("ActionRollout.outcome_bitstrings must be a NumPy array")
    if rollout.outcome_bitstrings.size == 0:
        raise ValueError("ActionRollout Born outcome table must not be empty")
    width = len(str(rollout.outcome_bitstrings[0]))
    validate_probability_arrays(
        rollout.outcome_bitstrings,
        rollout.exact_probabilities,
        width,
    )

    for name in (
        "dominates_baseline",
        "primary_metric_nonworsening",
        "selected",
    ):
        _require_real_bool(getattr(rollout, name), f"ActionRollout.{name}")
    for name in ("depth_delta", "gate_delta"):
        value = getattr(rollout, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"ActionRollout.{name} must be an integer and not bool")
    if not isinstance(rollout.metadata, Mapping):
        raise TypeError("ActionRollout.metadata must be a mapping")
    required_metadata = {
        "candidate_circuit_hash",
        "decomposition_metadata",
        "metric_weights",
        "improvement_atol",
        "weighted_improvement",
        "depth_penalty",
        "gate_penalty",
        "edit_penalty",
        "risk_penalty",
        "exact_born_recovery",
        "validation_mode",
        "candidate_generation_is_not_a_learned_policy",
    }
    missing_metadata = required_metadata - set(rollout.metadata)
    if missing_metadata:
        raise ValueError(
            "ActionRollout.metadata is missing required fields: "
            f"{sorted(missing_metadata)}"
        )
    circuit_hash = rollout.metadata["candidate_circuit_hash"]
    if (
        not isinstance(circuit_hash, str)
        or not circuit_hash.startswith("sha256:")
        or len(circuit_hash) != 71
    ):
        raise ValueError("ActionRollout candidate_circuit_hash is malformed")
    if not isinstance(rollout.metadata["decomposition_metadata"], Mapping):
        raise TypeError("ActionRollout decomposition_metadata must be a mapping")
    weights = rollout.metadata["metric_weights"]
    if not isinstance(weights, list) or len(weights) != len(PRIMARY_REWARD_METRICS):
        raise TypeError("ActionRollout metric_weights must be a three-value list")
    weight_array = np.asarray(
        [_require_finite(value, "ActionRollout metric weight") for value in weights],
        dtype=np.float64,
    )
    if np.any(weight_array < 0.0) or float(weight_array.sum()) <= 0.0:
        raise ValueError("ActionRollout metric_weights must be nonnegative and nonzero")
    atol = _require_finite(
        rollout.metadata["improvement_atol"],
        "ActionRollout improvement_atol",
    )
    if atol < 0.0:
        raise ValueError("ActionRollout improvement_atol must be nonnegative")
    weighted_improvement = _require_finite(
        rollout.metadata["weighted_improvement"],
        "ActionRollout weighted_improvement",
    )
    expected_weighted = float(np.dot(weight_array, improvement))
    if not math.isclose(
        weighted_improvement, expected_weighted, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("ActionRollout weighted_improvement mismatch")
    penalties = []
    for name in ("depth_penalty", "gate_penalty", "edit_penalty", "risk_penalty"):
        penalty = _require_finite(
            rollout.metadata[name],
            f"ActionRollout {name}",
        )
        if penalty < 0.0:
            raise ValueError(f"ActionRollout {name} must be nonnegative")
        penalties.append(penalty)
    expected_reward = weighted_improvement - sum(penalties)
    if not math.isclose(rollout.reward, expected_reward, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("ActionRollout reward does not match its breakdown")
    expected_nonworsening = bool(np.all(candidate <= baseline + atol))
    expected_dominates = bool(
        expected_nonworsening and np.any(candidate < baseline - atol)
    )
    expected_exact_recovery = bool(np.all(candidate <= atol))
    if rollout.primary_metric_nonworsening != expected_nonworsening:
        raise ValueError("ActionRollout primary_metric_nonworsening mismatch")
    if rollout.dominates_baseline != expected_dominates:
        raise ValueError("ActionRollout dominates_baseline mismatch")
    if not isinstance(rollout.metadata["exact_born_recovery"], bool):
        raise TypeError("ActionRollout exact_born_recovery must be bool")
    if rollout.metadata["exact_born_recovery"] != expected_exact_recovery:
        raise ValueError("ActionRollout exact_born_recovery mismatch")
    if rollout.metadata["validation_mode"] != "ideal_statevector":
        raise ValueError("ActionRollout validation_mode must be ideal_statevector")
    if rollout.metadata["candidate_generation_is_not_a_learned_policy"] is not True:
        raise ValueError(
            "ActionRollout must explicitly mark candidate generation as non-learned"
        )

    expected_hash = rollout_content_hash(rollout)
    if require_hash and rollout.content_hash != expected_hash:
        raise ValueError("ActionRollout.content_hash mismatch")
    if not require_hash and rollout.content_hash not in {"", expected_hash}:
        raise ValueError("ActionRollout.content_hash is malformed")


def validate_action_dataset_joins(
    candidate_records: list[Any],
    rollout_records: list[Any],
    *,
    candidates_by_id: Mapping[str, ActionCandidate] | None = None,
    rollouts_by_id: Mapping[str, ActionRollout] | None = None,
    source_samples: list[Any] | None = None,
    graph_pair_records: list[Any] | None = None,
    config: ActionEngineConfig | None = None,
) -> None:
    """Validate uniqueness and semantic joins across Phase 9 manifests."""
    validation_config = config or ActionEngineConfig()
    if not isinstance(validation_config, ActionEngineConfig):
        raise TypeError("config must be ActionEngineConfig or None")
    candidate_index: dict[str, Any] = {}
    candidate_circuit_ids: set[str] = set()
    action_refs: set[str] = set()
    circuit_refs: set[str] = set()
    for record in candidate_records:
        record.validate()
        if record.action_id in candidate_index:
            raise ValueError(f"Duplicate action candidate ID {record.action_id}")
        if record.candidate_circuit_id in candidate_circuit_ids:
            raise ValueError(
                f"Duplicate candidate circuit ID {record.candidate_circuit_id}"
            )
        if record.action_ref in action_refs:
            raise ValueError(f"Duplicate action artifact reference {record.action_ref}")
        if record.circuit_ref in circuit_refs:
            raise ValueError(f"Duplicate candidate circuit reference {record.circuit_ref}")
        candidate_index[record.action_id] = record
        candidate_circuit_ids.add(record.candidate_circuit_id)
        action_refs.add(record.action_ref)
        circuit_refs.add(record.circuit_ref)

    rollout_index: dict[str, Any] = {}
    rollout_by_action: dict[str, Any] = {}
    rollout_refs: set[str] = set()
    per_sample: dict[str, list[Any]] = {}
    for record in rollout_records:
        record.validate()
        if record.rollout_id in rollout_index:
            raise ValueError(f"Duplicate action rollout ID {record.rollout_id}")
        if record.action_id in rollout_by_action:
            raise ValueError(
                f"Multiple rollout records reference action {record.action_id}"
            )
        candidate = candidate_index.get(record.action_id)
        if candidate is None:
            raise ValueError(
                f"ActionRolloutRecord {record.rollout_id} references missing action "
                f"{record.action_id}"
            )
        for name in ("sample_id", "graph_pair_id", "candidate_circuit_id"):
            if getattr(record, name) != getattr(candidate, name):
                raise ValueError(
                    f"ActionRolloutRecord {record.rollout_id} {name} does not "
                    "match its ActionCandidateRecordV1"
                )
        if record.rollout_ref in rollout_refs:
            raise ValueError(f"Duplicate rollout artifact reference {record.rollout_ref}")
        rollout_index[record.rollout_id] = record
        rollout_by_action[record.action_id] = record
        rollout_refs.add(record.rollout_ref)
        per_sample.setdefault(record.sample_id, []).append(record)

    missing_rollouts = set(candidate_index) - set(rollout_by_action)
    if missing_rollouts:
        raise ValueError(
            f"Action candidates are missing rollout records: {sorted(missing_rollouts)}"
        )

    for sample_id, records in per_sample.items():
        ranks = sorted(record.rank for record in records)
        if ranks != list(range(1, len(records) + 1)):
            raise ValueError(
                f"Sample {sample_id} rollout ranks must be contiguous from one"
            )
        selected = [record for record in records if record.selected]
        if len(selected) != 1 or selected[0].rank != 1:
            raise ValueError(
                f"Sample {sample_id} must have exactly one selected rank-one action"
            )

    if candidates_by_id is not None:
        if set(candidates_by_id) != set(candidate_index):
            raise ValueError(
                "Candidate manifest IDs do not match in-memory ActionCandidate IDs"
            )
        for action_id, candidate in candidates_by_id.items():
            record = candidate_index[action_id]
            validate_action_candidate(candidate, validation_config, require_hash=True)
            for name in (
                "sample_id",
                "graph_pair_id",
                "source_circuit_id",
                "source_run_id",
                "distortion_id",
            ):
                if getattr(record, name) != getattr(candidate, name):
                    raise ValueError(
                        f"ActionCandidateRecordV1 {action_id} {name} mismatch"
                    )
            expected_candidate_circuit_id = candidate_circuit_id(
                candidate.source_circuit_id, candidate.action_id
            )
            if record.candidate_circuit_id != expected_candidate_circuit_id:
                raise ValueError(
                    f"ActionCandidateRecordV1 {action_id} candidate_circuit_id mismatch"
                )
            if record.content_hash != candidate.content_hash:
                raise ValueError(
                    f"ActionCandidateRecordV1 {action_id} content_hash mismatch"
                )
            if record.risk_score != candidate.risk_score:
                raise ValueError(
                    f"ActionCandidateRecordV1 {action_id} risk_score mismatch"
                )
            if record.validity_mask is not True:
                raise ValueError(
                    f"ActionCandidateRecordV1 {action_id} validity_mask must be true"
                )
            if record.edit_count != len(candidate.edits):
                raise ValueError(
                    f"ActionCandidateRecordV1 {action_id} edit_count mismatch"
                )
            if record.generation_sources != list(candidate.generation_sources):
                raise ValueError(
                    f"ActionCandidateRecordV1 {action_id} generation_sources mismatch"
                )

    if rollouts_by_id is not None:
        if set(rollouts_by_id) != set(rollout_index):
            raise ValueError(
                "Rollout manifest IDs do not match in-memory ActionRollout IDs"
            )
        for rollout_id, rollout in rollouts_by_id.items():
            record = rollout_index[rollout_id]
            validate_action_rollout(rollout, require_hash=True)
            if record.content_hash != rollout.content_hash:
                raise ValueError(
                    f"ActionRolloutRecord {rollout_id} content_hash mismatch"
                )
            for name in (
                "action_id",
                "sample_id",
                "graph_pair_id",
                "candidate_circuit_id",
                "clean_target_run_id",
                "scientific_config_id",
                "rank",
                "reward",
                "risk_score",
                "dominates_baseline",
                "primary_metric_nonworsening",
                "selected",
            ):
                if getattr(record, name) != getattr(rollout, name):
                    raise ValueError(
                        f"ActionRolloutRecord {rollout_id} {name} mismatch"
                    )

    if graph_pair_records is not None:
        pair_by_sample: dict[str, Any] = {}
        for pair in graph_pair_records:
            if pair.sample_id in pair_by_sample:
                raise ValueError(
                    f"Duplicate GraphPairRecord for sample {pair.sample_id}"
                )
            pair_by_sample[pair.sample_id] = pair
        for record in candidate_records:
            pair = pair_by_sample.get(record.sample_id)
            if pair is None:
                raise ValueError(
                    f"Action candidate {record.action_id} references a sample with no "
                    "GraphPairRecord"
                )
            if record.graph_pair_id != pair.graph_pair_id:
                raise ValueError(
                    f"Action candidate {record.action_id} graph_pair_id mismatch"
                )
            if record.distortion_id != pair.distortion_id:
                raise ValueError(
                    f"Action candidate {record.action_id} distortion_id mismatch"
                )

    if source_samples is not None:
        sample_index: dict[str, Any] = {}
        for sample in source_samples:
            if sample.sample_id in sample_index:
                raise ValueError(f"Duplicate Phase 7 sample {sample.sample_id}")
            sample_index[sample.sample_id] = sample
        sample_ids = set(sample_index)
        candidate_sample_ids = {record.sample_id for record in candidate_records}
        rollout_sample_ids = {record.sample_id for record in rollout_records}
        if candidate_sample_ids != sample_ids or rollout_sample_ids != sample_ids:
            raise ValueError(
                "Phase 9 manifests do not cover the Phase 7 sample set exactly"
            )
        counts: dict[str, int] = {}
        for record in candidate_records:
            sample = sample_index[record.sample_id]
            expected = {
                "source_circuit_id": sample.distorted_circuit_id,
                "source_run_id": sample.distorted_run_id,
                "distortion_id": sample.distortion_id,
            }
            for name, value in expected.items():
                if getattr(record, name) != value:
                    raise ValueError(
                        f"Action candidate {record.action_id} {name} mismatch"
                    )
            counts[record.sample_id] = counts.get(record.sample_id, 0) + 1
        for sample_id, count in counts.items():
            if count > validation_config.max_candidates_per_sample:
                raise ValueError(
                    f"Sample {sample_id} exceeds max_candidates_per_sample"
                )
        expected_scientific_config_id = action_scientific_config_id(
            validation_config
        )
        for record in rollout_records:
            sample = sample_index[record.sample_id]
            if record.clean_target_run_id != sample.clean_run_id:
                raise ValueError(
                    f"Action rollout {record.rollout_id} clean_target_run_id mismatch"
                )
            if record.scientific_config_id != expected_scientific_config_id:
                raise ValueError(
                    f"Action rollout {record.rollout_id} scientific_config_id mismatch"
                )

    if rollouts_by_id is not None:
        rollout_objects_by_sample: dict[str, list[ActionRollout]] = {}
        for rollout in rollouts_by_id.values():
            rollout_objects_by_sample.setdefault(rollout.sample_id, []).append(rollout)
        for sample_id, sample_rollouts in rollout_objects_by_sample.items():
            expected_order = sorted(
                sample_rollouts,
                key=lambda item: (
                    not item.primary_metric_nonworsening,
                    -item.reward,
                    item.risk_score,
                    item.action_id,
                ),
            )
            for expected_rank, rollout in enumerate(expected_order, start=1):
                if rollout.rank != expected_rank:
                    raise ValueError(
                        f"Sample {sample_id} rollout ranking is inconsistent"
                    )
                if rollout.selected != (expected_rank == 1):
                    raise ValueError(
                        f"Sample {sample_id} selected flag is inconsistent"
                    )


__all__ = [
    "validate_action_candidate",
    "validate_action_dataset_joins",
    "validate_action_edit",
    "validate_action_rollout",
    "validate_applied_action",
]
