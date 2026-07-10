"""Dataset-level orchestration for deterministic Phase 10 baselines."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from triqto.storage.baseline_schema import BaselineResultRecord

from .cobyla import run_cobyla
from .config import BaselineSuiteConfig
from .constants import PRIMARY_METRIC_NAMES
from .identities import (
    baseline_operational_config_id,
    baseline_result_content_hash,
    baseline_result_id,
    baseline_schema_id,
    baseline_suite_id,
)
from .loss_only_optimizer import select_loss_only
from .models import BaselineResult, BaselineSuiteResult, EvaluationSnapshot
from .optimizer_common import (
    ExactObjectiveEvaluator,
    build_optimizer_axes,
    optimizer_axis_payload,
    weighted_objective,
)
from .random_correction import select_random_correction
from .rule_only import select_rule_only
from .source import load_baseline_sources, verify_baseline_source_snapshots
from .spsa import run_spsa
from .transpiler_only import run_transpiler_only
from .validators import validate_baseline_dataset_joins, validate_baseline_result


def _reference_rollout(rollouts: tuple[Any, ...]) -> Any:
    if not rollouts:
        raise ValueError("Each Phase 7 sample requires Phase 9 rollouts")
    reference = rollouts[0]
    for rollout in rollouts[1:]:
        if not np.array_equal(
            rollout.metric_names,
            reference.metric_names,
        ):
            raise ValueError("Phase 9 rollout metric-name mismatch within sample")
        if not np.allclose(
            rollout.baseline_metric_values,
            reference.baseline_metric_values,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("Phase 9 baseline metrics differ within one sample")
    if tuple(reference.metric_names.tolist()) != PRIMARY_METRIC_NAMES:
        raise ValueError("Phase 9 rollout metrics do not match the Phase 10 contract")
    return reference


def _base_metadata(
    config: BaselineSuiteConfig,
    *,
    method_family: str,
    clean_target_access: str,
    distortion_metadata_access: str,
    hardware_aware: bool = False,
) -> dict[str, Any]:
    return {
        "method_family": method_family,
        "clean_target_access": clean_target_access,
        "distortion_metadata_access": distortion_metadata_access,
        "learned_model_used": False,
        "hardware_aware": hardware_aware,
        "metric_weights": list(config.metric_weights),
        "improvement_atol": config.improvement_atol,
        "evaluation_mode": "ideal_statevector",
        "phase": 10,
        "universal_correction_claimed": False,
        "quantum_advantage_claimed": False,
    }


def _finalize_result(result: BaselineResult, config: BaselineSuiteConfig) -> BaselineResult:
    result.content_hash = baseline_result_content_hash(result)
    validate_baseline_result(result, config, require_hash=True)
    return result


def _result_from_rollout(
    *,
    sample: Any,
    graph_pair_id: str,
    baseline_name: str,
    suite_id: str,
    rollout: Any,
    config: BaselineSuiteConfig,
    method_metadata: dict[str, Any],
) -> BaselineResult:
    baseline_values = np.asarray(
        rollout.baseline_metric_values, dtype=np.float64
    ).copy()
    result_values = np.asarray(
        rollout.candidate_metric_values, dtype=np.float64
    ).copy()
    before = weighted_objective(baseline_values, config)
    after = weighted_objective(result_values, config)
    improvement = before - after
    metadata = {
        **_base_metadata(
            config,
            method_family="phase9_candidate_selector",
            clean_target_access=(
                "selection_and_evaluation"
                if method_metadata.get("clean_target_used_for_selection") is True
                else "evaluation_only"
            ),
            distortion_metadata_access=(
                "privileged_synthetic_oracle"
                if method_metadata.get("distortion_metadata_used_for_selection")
                is True
                else "none"
            ),
        ),
        "selection": method_metadata,
        "phase9_rollout_id": rollout.rollout_id,
        "phase9_rollout_rank": rollout.rank,
        "phase9_candidate_reward": rollout.reward,
        "phase9_candidate_risk": rollout.risk_score,
        "reused_validated_phase9_rollout": True,
    }
    result = BaselineResult(
        baseline_result_id=baseline_result_id(
            sample.sample_id, baseline_name, suite_id
        ),
        baseline_suite_id=suite_id,
        sample_id=sample.sample_id,
        graph_pair_id=graph_pair_id,
        baseline_name=baseline_name,
        source_circuit_id=sample.distorted_circuit_id,
        clean_target_run_id=sample.clean_run_id,
        selected_action_id=rollout.action_id,
        metric_names=np.asarray(PRIMARY_METRIC_NAMES, dtype="<U32"),
        baseline_metric_values=baseline_values,
        result_metric_values=result_values,
        improvement_values=(baseline_values - result_values).astype(
            np.float64, copy=False
        ),
        outcome_bitstrings=rollout.outcome_bitstrings.copy(),
        exact_probabilities=rollout.exact_probabilities.copy(),
        parameter_vector=np.asarray([], dtype=np.float64),
        objective_before=before,
        objective_after=after,
        objective_improvement=improvement,
        success=improvement > config.improvement_atol,
        evaluations=0,
        iterations=0,
        metadata=metadata,
    )
    return _finalize_result(result, config)


def _result_from_snapshot(
    *,
    sample: Any,
    graph_pair_id: str,
    baseline_name: str,
    suite_id: str,
    reference_rollout: Any,
    snapshot: EvaluationSnapshot,
    config: BaselineSuiteConfig,
    evaluations: int,
    iterations: int,
    method_family: str,
    clean_target_access: str,
    method_metadata: dict[str, Any],
) -> BaselineResult:
    baseline_values = np.asarray(
        reference_rollout.baseline_metric_values, dtype=np.float64
    ).copy()
    result_values = np.asarray(snapshot.metric_values, dtype=np.float64).copy()
    before = weighted_objective(baseline_values, config)
    after = weighted_objective(result_values, config)
    improvement = before - after
    result = BaselineResult(
        baseline_result_id=baseline_result_id(
            sample.sample_id, baseline_name, suite_id
        ),
        baseline_suite_id=suite_id,
        sample_id=sample.sample_id,
        graph_pair_id=graph_pair_id,
        baseline_name=baseline_name,
        source_circuit_id=sample.distorted_circuit_id,
        clean_target_run_id=sample.clean_run_id,
        selected_action_id=None,
        metric_names=np.asarray(PRIMARY_METRIC_NAMES, dtype="<U32"),
        baseline_metric_values=baseline_values,
        result_metric_values=result_values,
        improvement_values=(baseline_values - result_values).astype(
            np.float64, copy=False
        ),
        outcome_bitstrings=snapshot.outcome_bitstrings.copy(),
        exact_probabilities=snapshot.exact_probabilities.copy(),
        parameter_vector=snapshot.vector.astype(np.float64, copy=True),
        objective_before=before,
        objective_after=after,
        objective_improvement=improvement,
        success=improvement > config.improvement_atol,
        evaluations=evaluations,
        iterations=iterations,
        metadata={
            **_base_metadata(
                config,
                method_family=method_family,
                clean_target_access=clean_target_access,
                distortion_metadata_access="none",
            ),
            "method": method_metadata,
            "evaluation_snapshot": snapshot.metadata,
            "reused_validated_phase9_rollout": False,
        },
    )
    return _finalize_result(result, config)


def run_baseline_suite(
    phase7_source_root: str | Path,
    graph_source_root: str | Path,
    action_source_root: str | Path,
    config: BaselineSuiteConfig | None = None,
) -> BaselineSuiteResult:
    """Run every enabled baseline for every completed Phase 7 sample."""
    baseline_config = config or BaselineSuiteConfig()
    if not isinstance(baseline_config, BaselineSuiteConfig):
        raise TypeError("config must be BaselineSuiteConfig or None")
    sources = load_baseline_sources(
        phase7_source_root,
        graph_source_root,
        action_source_root,
    )
    phase7 = sources.phase7
    graph = sources.graph
    action = sources.action
    suite_id = baseline_suite_id(
        phase7.source_scientific_generation_id,
        graph.completion_marker["graph_conversion_id"],
        action.completion_marker["action_engine_id"],
        baseline_config,
    )

    results: list[BaselineResult] = []
    for sample in sorted(phase7.samples, key=lambda item: item.sample_id):
        graph_pair_record = graph.pair_records_by_sample_id.get(sample.sample_id)
        if graph_pair_record is None:
            raise ValueError(f"Sample {sample.sample_id} has no Phase 8 graph pair")
        rollouts = action.rollouts_by_sample_id.get(sample.sample_id)
        if rollouts is None:
            raise ValueError(f"Sample {sample.sample_id} has no Phase 9 rollouts")
        reference = _reference_rollout(rollouts)
        source_circuit = phase7.circuits_by_id[sample.distorted_circuit_id]
        clean_probabilities = phase7.probabilities_by_run_id[sample.clean_run_id]

        for baseline_name in baseline_config.enabled_baselines:
            if baseline_name == "random_correction":
                selected, metadata = select_random_correction(
                    sample_id=sample.sample_id,
                    rollouts=rollouts,
                    candidates_by_id=action.candidates_by_id,
                    config=baseline_config,
                )
                result = _result_from_rollout(
                    sample=sample,
                    graph_pair_id=graph_pair_record.graph_pair_id,
                    baseline_name=baseline_name,
                    suite_id=suite_id,
                    rollout=selected,
                    config=baseline_config,
                    method_metadata=metadata,
                )
            elif baseline_name == "rule_only":
                selected, metadata = select_rule_only(
                    rollouts=rollouts,
                    candidates_by_id=action.candidates_by_id,
                )
                result = _result_from_rollout(
                    sample=sample,
                    graph_pair_id=graph_pair_record.graph_pair_id,
                    baseline_name=baseline_name,
                    suite_id=suite_id,
                    rollout=selected,
                    config=baseline_config,
                    method_metadata=metadata,
                )
            elif baseline_name == "loss_only":
                selected, metadata = select_loss_only(
                    rollouts=rollouts,
                    candidates_by_id=action.candidates_by_id,
                    config=baseline_config,
                )
                result = _result_from_rollout(
                    sample=sample,
                    graph_pair_id=graph_pair_record.graph_pair_id,
                    baseline_name=baseline_name,
                    suite_id=suite_id,
                    rollout=selected,
                    config=baseline_config,
                    method_metadata=metadata,
                )
            elif baseline_name in {"spsa", "cobyla"}:
                axes = build_optimizer_axes(source_circuit)
                if len(axes) > baseline_config.max_optimizer_dimensions:
                    raise RuntimeError(
                        f"Sample {sample.sample_id} optimizer dimension {len(axes)} "
                        "exceeds max_optimizer_dimensions"
                    )
                evaluator = ExactObjectiveEvaluator(
                    source_circuit=source_circuit,
                    clean_probabilities=clean_probabilities,
                    axes=axes,
                    config=baseline_config,
                )
                if baseline_name == "spsa":
                    snapshot, iterations, method_metadata = run_spsa(
                        sample_id=sample.sample_id,
                        evaluator=evaluator,
                        config=baseline_config,
                    )
                else:
                    snapshot, iterations, method_metadata = run_cobyla(
                        evaluator=evaluator,
                        config=baseline_config,
                    )
                method_metadata = {
                    **method_metadata,
                    "optimizer_axes": optimizer_axis_payload(axes),
                    "source_depth": source_circuit.depth(),
                    "source_gate_count": len(source_circuit.data),
                }
                result = _result_from_snapshot(
                    sample=sample,
                    graph_pair_id=graph_pair_record.graph_pair_id,
                    baseline_name=baseline_name,
                    suite_id=suite_id,
                    reference_rollout=reference,
                    snapshot=snapshot,
                    config=baseline_config,
                    evaluations=evaluator.evaluations,
                    iterations=iterations,
                    method_family="continuous_clean_target_optimizer",
                    clean_target_access="objective_and_evaluation",
                    method_metadata=method_metadata,
                )
            elif baseline_name == "transpiler_only":
                snapshot, method_metadata = run_transpiler_only(
                    sample_id=sample.sample_id,
                    source_circuit=source_circuit,
                    clean_probabilities=clean_probabilities,
                    config=baseline_config,
                )
                result = _result_from_snapshot(
                    sample=sample,
                    graph_pair_id=graph_pair_record.graph_pair_id,
                    baseline_name=baseline_name,
                    suite_id=suite_id,
                    reference_rollout=reference,
                    snapshot=snapshot,
                    config=baseline_config,
                    evaluations=1,
                    iterations=0,
                    method_family="compiler_semantic_control",
                    clean_target_access="evaluation_only",
                    method_metadata=method_metadata,
                )
            else:  # pragma: no cover - config validates the fixed vocabulary
                raise ValueError(f"Unsupported baseline {baseline_name}")
            results.append(result)

    order = {name: index for index, name in enumerate(baseline_config.enabled_baselines)}
    results.sort(key=lambda item: (item.sample_id, order[item.baseline_name]))
    result_records: list[BaselineResultRecord] = []
    for result in results:
        record = BaselineResultRecord(
            baseline_result_id=result.baseline_result_id,
            baseline_suite_id=result.baseline_suite_id,
            sample_id=result.sample_id,
            graph_pair_id=result.graph_pair_id,
            baseline_name=result.baseline_name,
            source_circuit_id=result.source_circuit_id,
            clean_target_run_id=result.clean_target_run_id,
            selected_action_id=result.selected_action_id,
            artifact_ref=f"artifacts/results/{result.baseline_result_id}.npz",
            content_hash=result.content_hash,
            objective_before=result.objective_before,
            objective_after=result.objective_after,
            objective_improvement=result.objective_improvement,
            success=result.success,
            evaluations=result.evaluations,
            iterations=result.iterations,
            metadata={
                "phase": 10,
                "method_family": result.metadata["method_family"],
                "learned_model_used": False,
            },
        )
        record.validate()
        result_records.append(record)

    validate_baseline_dataset_joins(
        result_records,
        results_by_id={item.baseline_result_id: item for item in results},
        source_samples=phase7.samples,
        graph_pair_records=graph.pair_records,
        action_candidates_by_id=action.candidates_by_id,
        config=baseline_config,
    )
    verify_baseline_source_snapshots(sources)

    per_baseline: dict[str, dict[str, Any]] = {}
    for name in baseline_config.enabled_baselines:
        items = [result for result in results if result.baseline_name == name]
        per_baseline[name] = {
            "result_count": len(items),
            "success_count": sum(item.success for item in items),
            "mean_objective_before": fmean(item.objective_before for item in items),
            "mean_objective_after": fmean(item.objective_after for item in items),
            "mean_objective_improvement": fmean(
                item.objective_improvement for item in items
            ),
            "mean_evaluations": fmean(item.evaluations for item in items),
        }
    selected_action_counts = Counter(
        result.baseline_name
        for result in results
        if result.selected_action_id is not None
    )
    summary = {
        "source_scientific_generation_id": phase7.source_scientific_generation_id,
        "graph_conversion_id": graph.completion_marker["graph_conversion_id"],
        "action_engine_id": action.completion_marker["action_engine_id"],
        "baseline_suite_id": suite_id,
        "operational_config_id": baseline_operational_config_id(baseline_config),
        "baseline_schema_id": baseline_schema_id(),
        "source_sample_count": len(phase7.samples),
        "enabled_baselines": list(baseline_config.enabled_baselines),
        "baseline_count": len(baseline_config.enabled_baselines),
        "result_count": len(results),
        "expected_result_count": (
            len(phase7.samples) * len(baseline_config.enabled_baselines)
        ),
        "per_baseline": per_baseline,
        "phase9_action_selector_result_counts": dict(
            sorted(selected_action_counts.items())
        ),
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "graph_snapshot_hash": graph.snapshot.aggregate_sha256,
        "action_snapshot_hash": action.snapshot.aggregate_sha256,
        "source_immutability_verified": True,
        "evaluation_mode": "ideal_statevector",
        "learned_triqto_policy_present": False,
        "triqto_model_compared": False,
        "rule_only_uses_privileged_synthetic_metadata": True,
        "loss_only_spsa_cobyla_use_clean_target_objectives": True,
        "transpiler_only_is_backend_free_semantic_control": True,
        "hardware_aware_baseline_present": False,
        "quantum_advantage_claimed": False,
    }
    if summary["result_count"] != summary["expected_result_count"]:
        raise ValueError("Phase 10 result count does not match sample/baseline product")

    return BaselineSuiteResult(
        phase7_source_root=phase7.source_root,
        graph_source_root=graph.root,
        action_source_root=action.root,
        config=baseline_config,
        source_scientific_generation_id=phase7.source_scientific_generation_id,
        graph_conversion_id=graph.completion_marker["graph_conversion_id"],
        action_engine_id=action.completion_marker["action_engine_id"],
        baseline_suite_id=suite_id,
        operational_config_id=baseline_operational_config_id(baseline_config),
        baseline_schema_id=baseline_schema_id(),
        results=results,
        result_records=result_records,
        phase7_snapshot=phase7.source_snapshot,
        graph_snapshot=graph.snapshot,
        action_snapshot=action.snapshot,
        summary=summary,
    )


__all__ = ["run_baseline_suite"]
