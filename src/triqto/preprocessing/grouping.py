"""Duplicate classification, split-specific leakage relations, and hard negatives."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping

from .records import DuplicateRelation, HardNegativePair, LeakageRelation, ProcessedSample


def _stable_id(namespace: str, payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"{namespace}_{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:24]}"


def _accepted(samples: Iterable[ProcessedSample]) -> list[ProcessedSample]:
    return sorted((sample for sample in samples if sample.accepted), key=lambda item: item.sample_id)


def _groups(
    samples: Iterable[ProcessedSample], key_fn: Callable[[ProcessedSample], Any]
) -> dict[str, list[ProcessedSample]]:
    grouped: dict[str, list[ProcessedSample]] = defaultdict(list)
    for sample in samples:
        key = key_fn(sample)
        if key is None or key == "":
            continue
        encoded = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
        grouped[encoded].append(sample)
    return {
        key: sorted(members, key=lambda item: item.sample_id)
        for key, members in grouped.items()
        if len(members) > 1
    }


def _context_differences(members: list[ProcessedSample]) -> dict[str, Any]:
    fields = {
        "hardware_context_hash": [item.hashes.hardware_context_hash for item in members],
        "state_equivalence_hash": [item.hashes.state_equivalence_hash for item in members],
        "born_distribution_hash": [item.hashes.born_distribution_hash for item in members],
        "canonical_circuit_hash": [item.hashes.canonical_circuit_hash for item in members],
        "circuit_parameter_hash": [item.hashes.circuit_parameter_hash for item in members],
        "intervention_label": [item.intervention_label for item in members],
        "source_type": [item.source_type for item in members],
    }
    return {
        name: sorted({str(value) for value in values if value is not None})
        for name, values in fields.items()
        if len({str(value) for value in values if value is not None}) > 1
    }


def _duplicate_record(
    relation_type: str,
    members: list[ProcessedSample],
    *,
    retention_policy: str,
) -> DuplicateRelation:
    ids = tuple(item.sample_id for item in members)
    return DuplicateRelation(
        relation_type=relation_type,
        group_id=_stable_id(relation_type, ids),
        representative_sample_id=min(ids),
        member_sample_ids=ids,
        multiplicity=len(ids),
        context_differences=_context_differences(members),
        retention_policy=retention_policy,
    )


def build_duplicate_relations(samples: Iterable[ProcessedSample]) -> list[DuplicateRelation]:
    rows = _accepted(samples)
    records: list[DuplicateRelation] = []

    specifications: list[
        tuple[str, Callable[[ProcessedSample], Any], str, Callable[[list[ProcessedSample]], bool] | None]
    ] = [
        (
            "byte_exact_duplicate",
            lambda item: item.hashes.raw_record_hash,
            "preserve_raw_multiplicity; training_view_may_collapse_exact_duplicates_with_weight",
            None,
        ),
        (
            "canonical_record_duplicate",
            lambda item: item.hashes.canonical_record_hash,
            "preserve_multiplicity_and_select_lexicographic_representative",
            None,
        ),
        (
            "same_circuit_same_parameters",
            lambda item: item.hashes.circuit_parameter_hash,
            "retain_repeated_observations_and_context",
            None,
        ),
        (
            "same_state_same_context",
            lambda item: (
                item.hashes.state_equivalence_hash,
                item.hashes.hardware_context_hash,
            )
            if item.hashes.state_equivalence_hash is not None
            else None,
            "retain_provenance_and_repeated_measurements",
            None,
        ),
        (
            "same_born_same_state",
            lambda item: (
                item.hashes.born_distribution_hash,
                item.hashes.state_equivalence_hash,
            )
            if item.hashes.state_equivalence_hash is not None
            else None,
            "retain_measurement_instances_and_provenance",
            None,
        ),
        (
            "same_hardware_repeated_observation",
            lambda item: (
                item.hashes.circuit_parameter_hash,
                item.hashes.hardware_context_hash,
            ),
            "retain_as_repeated_observations",
            lambda members: len({item.hashes.measurement_instance_hash for item in members}) > 1,
        ),
        (
            "same_base_sample_different_distortion",
            lambda item: item.clean_circuit_id,
            "link_clean_distorted_descendants; never_collapse",
            lambda members: len({item.distortion_id for item in members}) > 1,
        ),
        (
            "same_candidate_set",
            lambda item: item.hashes.counterfactual_set_hash,
            "preserve_counterfactual_set_membership",
            None,
        ),
    ]

    for relation_type, key_fn, policy, predicate in specifications:
        for members in _groups(rows, key_fn).values():
            if predicate is None or predicate(members):
                records.append(
                    _duplicate_record(relation_type, members, retention_policy=policy)
                )

    for members in _groups(rows, lambda item: item.hashes.canonical_circuit_hash).values():
        if len({item.hashes.circuit_parameter_hash for item in members}) > 1:
            records.append(
                _duplicate_record(
                    "same_circuit_different_parameters",
                    members,
                    retention_policy="retain_parameter_variants",
                )
            )
        if len({item.hashes.hardware_context_hash for item in members}) > 1:
            records.append(
                _duplicate_record(
                    "same_logical_circuit_different_hardware",
                    members,
                    retention_policy="retain_all_hardware_contexts",
                )
            )

    for members in _groups(rows, lambda item: item.hashes.state_equivalence_hash).values():
        if len({item.hashes.canonical_circuit_hash for item in members}) > 1:
            records.append(
                _duplicate_record(
                    "same_state_different_circuit",
                    members,
                    retention_policy="retain_distinct_circuit_realizations",
                )
            )

    for members in _groups(rows, lambda item: item.hashes.born_distribution_hash).values():
        state_hashes = {item.hashes.state_equivalence_hash for item in members if item.hashes.state_equivalence_hash}
        has_unknown_state = any(item.hashes.state_equivalence_hash is None for item in members)
        if len(state_hashes) > 1:
            records.append(
                _duplicate_record(
                    "same_born_different_state",
                    members,
                    retention_policy="never_collapse_measurement_degenerate_states",
                )
            )
        elif has_unknown_state and not state_hashes:
            records.append(
                _duplicate_record(
                    "possible_duplicate_requiring_review",
                    members,
                    retention_policy="retain_pending_multibasis_or_hilbert_evidence",
                )
            )

    unique: dict[tuple[str, tuple[str, ...]], DuplicateRelation] = {}
    for record in records:
        unique[(record.relation_type, record.member_sample_ids)] = record
    return sorted(unique.values(), key=lambda item: (item.relation_type, item.group_id))


def _leakage_record(
    relation_type: str,
    members: list[ProcessedSample],
    evidence: Mapping[str, Any],
) -> LeakageRelation:
    ids = tuple(item.sample_id for item in members)
    return LeakageRelation(
        relation_type=relation_type,
        relation_id=_stable_id(relation_type, ids),
        member_sample_ids=ids,
        evidence=dict(evidence),
    )


def build_leakage_relations(samples: Iterable[ProcessedSample]) -> list[LeakageRelation]:
    rows = _accepted(samples)
    records: list[LeakageRelation] = []

    specs: list[tuple[str, Callable[[ProcessedSample], Any], dict[str, Any]]] = [
        (
            "base_circuit_descendants",
            lambda item: item.clean_circuit_id,
            {
                "forbid_cross_split": True,
                "forbid_cross_split_in_instance_split": True,
                "reason": "clean parent and distorted descendants must remain together",
            },
        ),
        (
            "target_identity",
            lambda item: item.hashes.target_hash,
            {"forbid_cross_split": True, "reason": "same target identity"},
        ),
        (
            "optimization_trajectory",
            lambda item: item.provenance.get("trajectory_id"),
            {"forbid_cross_split": True, "reason": "same optimization trajectory"},
        ),
        (
            "parameter_neighbourhood",
            lambda item: item.provenance.get("parameter_neighbourhood_id"),
            {"forbid_cross_split": True, "reason": "same local parameter neighbourhood"},
        ),
        (
            "counterfactual_set",
            lambda item: item.hashes.counterfactual_set_hash,
            {"forbid_cross_split": True, "reason": "same clean/distorted/candidate comparison set"},
        ),
        (
            "structural_graph_equivalence",
            lambda item: item.hashes.structural_graph_hash,
            {
                "forbid_cross_split": False,
                "forbid_cross_split_in_symmetry_strict_split": True,
                "reason": "qubit-renaming-invariant graph class",
            },
        ),
        (
            "calibration_snapshot",
            lambda item: item.hardware_context.get("calibration_snapshot_id"),
            {
                "forbid_cross_split": False,
                "forbid_cross_split_in_temporal_split": True,
                "reason": "same hardware calibration snapshot",
            },
        ),
        (
            "calibration_window",
            lambda item: item.hardware_context.get("calibration_window_id"),
            {
                "forbid_cross_split": False,
                "forbid_cross_split_in_temporal_split": True,
                "reason": "same calibration time window",
            },
        ),
        (
            "backend_run",
            lambda item: item.hardware_context.get("backend_run_id"),
            {"forbid_cross_split": True, "reason": "same backend execution batch"},
        ),
        (
            "deterministic_regeneration_family",
            lambda item: (
                item.provenance.get("scientific_generation_id"),
                item.provenance.get("generation_seed"),
                item.clean_circuit_id,
            ),
            {"forbid_cross_split": True, "reason": "same deterministic regeneration family"},
        ),
    ]
    for relation_type, key_fn, evidence in specs:
        for members in _groups(rows, key_fn).values():
            records.append(_leakage_record(relation_type, members, evidence))

    duplicate_hash_specs = [
        ("raw_duplicate_equivalence", lambda item: item.hashes.raw_record_hash),
        ("canonical_duplicate_equivalence", lambda item: item.hashes.canonical_record_hash),
        ("circuit_parameter_equivalence", lambda item: item.hashes.circuit_parameter_hash),
    ]
    for relation_type, key_fn in duplicate_hash_specs:
        for members in _groups(rows, key_fn).values():
            records.append(
                _leakage_record(
                    relation_type,
                    members,
                    {
                        "forbid_cross_split": True,
                        "reason": "duplicate/equivalent records must not cross partitions",
                    },
                )
            )
    return sorted(records, key=lambda item: (item.relation_type, item.relation_id))


def _effect_similarity(left: ProcessedSample, right: ProcessedSample) -> float | None:
    if left.combined_effect_score is None or right.combined_effect_score is None:
        return None
    return max(0.0, 1.0 - abs(float(left.combined_effect_score) - float(right.combined_effect_score)))


def _parameter_similarity(left: ProcessedSample, right: ProcessedSample) -> float | None:
    keys = sorted(set(left.parameter_bindings_canonical) & set(right.parameter_bindings_canonical))
    if not keys:
        return None
    distances = []
    for key in keys:
        delta = (left.parameter_bindings_canonical[key] - right.parameter_bindings_canonical[key] + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        distances.append(abs(delta) / math.pi)
    return max(0.0, 1.0 - sum(distances) / len(distances))


def build_hard_negative_pairs(
    samples: Iterable[ProcessedSample], *, maximum_per_category: int
) -> list[HardNegativePair]:
    rows = _accepted(samples)
    candidates: list[HardNegativePair] = []

    def add_pair(left: ProcessedSample, right: ProcessedSample, category: str, evidence: dict[str, Any], confidence: float, tasks: tuple[str, ...]) -> None:
        anchor, negative = sorted((left.sample_id, right.sample_id))
        candidates.append(
            HardNegativePair(
                anchor_sample_id=anchor,
                negative_sample_id=negative,
                category=category,
                similarity_scores={
                    "effect_similarity": _effect_similarity(left, right),
                    "parameter_similarity": _parameter_similarity(left, right),
                    "same_born": float(left.hashes.born_distribution_hash == right.hashes.born_distribution_hash),
                    "same_graph_structure": float(left.hashes.structural_graph_hash == right.hashes.structural_graph_hash),
                },
                distinguishing_evidence=evidence,
                confidence=float(max(0.0, min(1.0, confidence))),
                eligible_tasks=tasks,
            )
        )

    for members in _groups(rows, lambda item: item.hashes.born_distribution_hash).values():
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if (
                    left.measurement_basis == "Z"
                    and left.hashes.state_equivalence_hash
                    and right.hashes.state_equivalence_hash
                    and left.hashes.state_equivalence_hash != right.hashes.state_equivalence_hash
                ):
                    add_pair(
                        left,
                        right,
                        "same_z_born_different_state",
                        {
                            "state_hashes": [left.hashes.state_equivalence_hash, right.hashes.state_equivalence_hash],
                            "basis": "Z",
                        },
                        1.0,
                        ("contrastive_state_identity", "phase_sensitive_diagnosis"),
                    )

    for members in _groups(rows, lambda item: item.intervention_label).values():
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if left.observed_effect_label != right.observed_effect_label:
                    add_pair(
                        left,
                        right,
                        "same_intervention_different_observed_effect",
                        {
                            "intervention": left.intervention_label,
                            "observed_effects": [left.observed_effect_label, right.observed_effect_label],
                        },
                        min(left.observed_effect_confidence, right.observed_effect_confidence),
                        ("observed_effect_classification", "abstention"),
                    )

    for members in _groups(rows, lambda item: item.hashes.structural_graph_hash).values():
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                if left.hashes.hardware_context_hash != right.hashes.hardware_context_hash:
                    add_pair(
                        left,
                        right,
                        "similar_graph_different_hardware_context",
                        {
                            "hardware_hashes": [left.hashes.hardware_context_hash, right.hashes.hardware_context_hash]
                        },
                        0.9,
                        ("hardware_context_diagnosis", "layout_generalization"),
                    )

    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            similarity = _effect_similarity(left, right)
            if similarity is not None and similarity >= 0.98 and left.intervention_label != right.intervention_label:
                add_pair(
                    left,
                    right,
                    "similar_effect_different_physical_cause",
                    {
                        "interventions": [left.intervention_label, right.intervention_label],
                        "effect_scores": [left.combined_effect_score, right.combined_effect_score],
                    },
                    similarity,
                    ("distortion_diagnosis", "causal_abstention"),
                )

    unique: dict[tuple[str, str, str], HardNegativePair] = {}
    for record in candidates:
        unique[(record.category, record.anchor_sample_id, record.negative_sample_id)] = record
    by_category: dict[str, list[HardNegativePair]] = defaultdict(list)
    for record in unique.values():
        by_category[record.category].append(record)
    result: list[HardNegativePair] = []
    for category in sorted(by_category):
        rows_for_category = sorted(
            by_category[category],
            key=lambda item: (-item.confidence, item.anchor_sample_id, item.negative_sample_id),
        )
        result.extend(rows_for_category[:maximum_per_category])
    return result
