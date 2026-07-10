"""Dataset-level orchestration for deterministic Phase 9 action validation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from triqto.storage.action_schema import (
    ActionCandidateRecordV1,
    ActionRolloutRecord,
)

from .candidates import generate_action_candidates
from .config import ActionEngineConfig
from .identities import (
    action_engine_id,
    action_operational_config_id,
    action_schema_id,
)
from .models import ActionEngineResult
from .rollout_runner import run_action_rollouts
from .source import load_action_engine_sources, verify_action_source_snapshots
from .validators import validate_action_dataset_joins


def _index_unique(records: list[Any], field_name: str, record_name: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for record in records:
        identifier = getattr(record, field_name, None)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{record_name}.{field_name} must be nonblank")
        if identifier in indexed:
            raise ValueError(f"Duplicate {record_name} {field_name} {identifier}")
        indexed[identifier] = record
    return indexed


def _selected_action_kind(candidate: Any) -> str:
    if not candidate.edits:
        return "no_op"
    if len(candidate.edits) > 1:
        return "composite"
    return candidate.edits[0].edit_type


def build_action_engine_result(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    config: ActionEngineConfig | None = None,
) -> ActionEngineResult:
    """Generate, apply, exactly validate, and rank actions for every Phase 7 sample."""
    action_config = config or ActionEngineConfig()
    if not isinstance(action_config, ActionEngineConfig):
        raise TypeError("config must be ActionEngineConfig or None")
    sources = load_action_engine_sources(phase7_source_root, graph_source_root)
    phase7 = sources.phase7
    graph = sources.graph

    distortions = _index_unique(
        phase7.distortions,
        "distortion_id",
        "DistortionRecord",
    )
    candidates = []
    rollouts = []
    for sample in sorted(phase7.samples, key=lambda item: item.sample_id):
        try:
            graph_pair_record = graph.pair_records_by_sample_id[sample.sample_id]
        except KeyError as exc:
            raise ValueError(
                f"Phase 7 sample {sample.sample_id} has no Phase 8 GraphPairRecord"
            ) from exc
        try:
            distortion = distortions[sample.distortion_id]
        except KeyError as exc:
            raise ValueError(
                f"Sample {sample.sample_id} references missing distortion "
                f"{sample.distortion_id}"
            ) from exc
        try:
            distorted_circuit = phase7.circuits_by_id[
                sample.distorted_circuit_id
            ]
        except KeyError as exc:
            raise ValueError(
                f"Sample {sample.sample_id} references missing distorted circuit"
            ) from exc

        sample_candidates = generate_action_candidates(
            sample=sample,
            graph_pair_record=graph_pair_record,
            distortion=distortion,
            distorted_circuit=distorted_circuit,
            config=action_config,
        )
        sample_rollouts = run_action_rollouts(
            distorted_circuit=distorted_circuit,
            clean_target_run_id=sample.clean_run_id,
            clean_probabilities=phase7.probabilities_by_run_id[
                sample.clean_run_id
            ],
            distorted_probabilities=phase7.probabilities_by_run_id[
                sample.distorted_run_id
            ],
            candidates=sample_candidates,
            config=action_config,
        )
        candidates.extend(sample_candidates)
        rollouts.extend(sample_rollouts)

    candidates.sort(key=lambda item: item.action_id)
    rollouts.sort(key=lambda item: (item.sample_id, item.rank, item.action_id))
    rollout_by_action_id = _index_unique(
        rollouts,
        "action_id",
        "ActionRollout",
    )

    candidate_records: list[ActionCandidateRecordV1] = []
    for candidate in candidates:
        rollout = rollout_by_action_id[candidate.action_id]
        circuit_hash = rollout.metadata.get("candidate_circuit_hash")
        if not isinstance(circuit_hash, str):
            raise ValueError(
                f"Rollout {rollout.rollout_id} is missing candidate_circuit_hash"
            )
        record = ActionCandidateRecordV1(
            action_id=candidate.action_id,
            sample_id=candidate.sample_id,
            graph_pair_id=candidate.graph_pair_id,
            source_circuit_id=candidate.source_circuit_id,
            source_run_id=candidate.source_run_id,
            distortion_id=candidate.distortion_id,
            candidate_circuit_id=rollout.candidate_circuit_id,
            generation_sources=list(candidate.generation_sources),
            action_ref=f"artifacts/actions/{candidate.action_id}.json",
            circuit_ref=(
                f"artifacts/circuits/{rollout.candidate_circuit_id}.qpy"
            ),
            content_hash=candidate.content_hash,
            circuit_hash=circuit_hash,
            edit_count=len(candidate.edits),
            validity_mask=True,
            risk_score=candidate.risk_score,
            metadata={
                "phase": 9,
                "candidate_generation_is_not_a_learned_policy": True,
            },
        )
        record.validate()
        candidate_records.append(record)

    rollout_records: list[ActionRolloutRecord] = []
    for rollout in rollouts:
        record = ActionRolloutRecord(
            rollout_id=rollout.rollout_id,
            action_id=rollout.action_id,
            sample_id=rollout.sample_id,
            graph_pair_id=rollout.graph_pair_id,
            candidate_circuit_id=rollout.candidate_circuit_id,
            clean_target_run_id=rollout.clean_target_run_id,
            scientific_config_id=rollout.scientific_config_id,
            rollout_ref=f"artifacts/rollouts/{rollout.rollout_id}.npz",
            content_hash=rollout.content_hash,
            rank=rollout.rank,
            reward=rollout.reward,
            risk_score=rollout.risk_score,
            dominates_baseline=rollout.dominates_baseline,
            primary_metric_nonworsening=(
                rollout.primary_metric_nonworsening
            ),
            selected=rollout.selected,
            metadata={
                "exact_born_recovery": bool(
                    rollout.metadata.get("exact_born_recovery", False)
                ),
                "validation_mode": "ideal_statevector",
                "phase": 9,
            },
        )
        record.validate()
        rollout_records.append(record)

    validate_action_dataset_joins(
        candidate_records,
        rollout_records,
        candidates_by_id={item.action_id: item for item in candidates},
        rollouts_by_id={item.rollout_id: item for item in rollouts},
        source_samples=phase7.samples,
        graph_pair_records=graph.pair_records,
        config=action_config,
    )
    verify_action_source_snapshots(sources)

    engine_id = action_engine_id(
        phase7.source_scientific_generation_id,
        graph.completion_marker["graph_conversion_id"],
        action_config,
    )
    operational_id = action_operational_config_id(action_config)
    selected_rollouts = [rollout for rollout in rollouts if rollout.selected]
    candidates_by_id = {candidate.action_id: candidate for candidate in candidates}
    selected_kind_counts = Counter(
        _selected_action_kind(candidates_by_id[rollout.action_id])
        for rollout in selected_rollouts
    )
    candidate_count_by_sample = Counter(
        candidate.sample_id for candidate in candidates
    )
    summary = {
        "source_scientific_generation_id": phase7.source_scientific_generation_id,
        "graph_conversion_id": graph.completion_marker["graph_conversion_id"],
        "action_engine_id": engine_id,
        "operational_config_id": operational_id,
        "action_schema_id": action_schema_id(),
        "source_sample_count": len(phase7.samples),
        "candidate_count": len(candidates),
        "rollout_count": len(rollouts),
        "selected_action_count": len(selected_rollouts),
        "oracle_candidate_count": sum(
            "oracle_inverse" in candidate.generation_sources
            for candidate in candidates
        ),
        "blind_candidate_count": sum(
            "blind_physics_prior" in candidate.generation_sources
            for candidate in candidates
        ),
        "no_op_candidate_count": sum(not candidate.edits for candidate in candidates),
        "nonworsening_rollout_count": sum(
            rollout.primary_metric_nonworsening for rollout in rollouts
        ),
        "improving_rollout_count": sum(
            rollout.dominates_baseline for rollout in rollouts
        ),
        "exact_born_recovery_count": sum(
            bool(rollout.metadata.get("exact_born_recovery", False))
            for rollout in rollouts
        ),
        "selected_no_op_count": selected_kind_counts.get("no_op", 0),
        "selected_action_type_counts": dict(sorted(selected_kind_counts.items())),
        "candidate_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(
                Counter(candidate_count_by_sample.values()).items()
            )
        },
        "mean_selected_reward": (
            fmean(rollout.reward for rollout in selected_rollouts)
            if selected_rollouts
            else 0.0
        ),
        "phase7_managed_file_count": len(phase7.source_snapshot.entries),
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "graph_managed_file_count": len(graph.snapshot.entries),
        "graph_snapshot_hash": graph.snapshot.aggregate_sha256,
        "source_immutability_verified": True,
        "validation_mode": "ideal_statevector",
        "learned_policy_present": False,
        "schema_versions": {
            "action": action_config.schema_version,
            "graph": graph.config.schema_version,
            "phase7": phase7.generation_config.schema_version,
        },
    }
    return ActionEngineResult(
        phase7_source_root=phase7.source_root,
        graph_source_root=graph.root,
        config=action_config,
        source_scientific_generation_id=phase7.source_scientific_generation_id,
        graph_conversion_id=graph.completion_marker["graph_conversion_id"],
        action_engine_id=engine_id,
        operational_config_id=operational_id,
        action_schema_id=action_schema_id(),
        candidates=candidates,
        rollouts=rollouts,
        candidate_records=candidate_records,
        rollout_records=rollout_records,
        phase7_snapshot=phase7.source_snapshot,
        graph_snapshot=graph.snapshot,
        summary=summary,
    )


__all__ = ["build_action_engine_result"]
