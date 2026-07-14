"""Deterministic held-out evaluation, ablations, and baseline comparisons."""
from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass
import copy
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from triqto.baselines import (
    load_baseline_sources,
    verify_baseline_source_snapshots,
    weighted_objective,
)
from triqto.graph.utils import resolve_safe_file
from triqto.model import TriQTOModel
from triqto.model.constants import STREAM_ORDER
from triqto.training import (
    collate_training_examples,
    load_completed_training_view_dataset,
    load_training_checkpoint,
    load_training_examples,
    snapshot_managed_files,
)
from triqto.training_views import load_training_view_item_artifact

from .config import EvaluationConfig
from .identities import (
    evaluation_baseline_id,
    evaluation_item_content_hash,
    evaluation_item_id,
    evaluation_operational_config_id,
    evaluation_recipe_id,
    evaluation_run_id,
    evaluation_schema_id,
)
from .metrics import build_aggregates, distribution_metrics_by_graph
from .models import (
    BaselineComparison,
    EvaluationItemResult,
    EvaluationRunResult,
)
from .source import load_completed_baseline_dataset, load_completed_training_run

_PRIVILEGED_BASELINES = {
    "rule_only",
    "loss_only",
    "spsa",
    "cobyla",
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Evaluation requested CUDA but CUDA is unavailable")
    return torch.device(name)


def _move_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            setattr(value, field.name, _move_tree(getattr(value, field.name), device))
        return value
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return value


def _apply_ablation(supervised: Any, ablation: str) -> Any:
    """Remove one optional stream without relaxing any hard model policy."""
    result = copy.deepcopy(supervised)
    if ablation == "full":
        return result
    if ablation not in {"no_topology", "no_hilbert"}:
        raise ValueError(f"Unsupported ablation {ablation!r}")
    stream_name = "topology" if ablation == "no_topology" else "hilbert"
    stream_index = STREAM_ORDER.index(stream_name)
    for batch in (
        result.model_batch,
        result.auxiliary_hilbert_to_born_batch,
    ):
        if batch is None:
            continue
        batch.head_stream_mask[:, :, stream_index] = False
        if stream_name == "topology":
            batch.topology = None
            batch.topology_hilbert_dependent_mask = torch.zeros_like(
                batch.topology_hilbert_dependent_mask
            )
        else:
            batch.hilbert = None
    return result


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _raw_item(dataset: Any, view_item_id: str) -> Any:
    record = dataset.records_by_id.get(view_item_id)
    if record is None:
        raise ValueError(f"Missing Phase 12 item record {view_item_id}")
    return load_training_view_item_artifact(
        resolve_safe_file(
            dataset.root,
            record.artifact_ref,
            f"Phase 12 item {view_item_id}",
        ),
        dataset.config,
        expected_content_hash=record.content_hash,
    )


def _load_test_examples(
    dataset: Any,
    *,
    task: str,
    spec: Any,
    phase7_root: str | Path | None,
) -> list[Any]:
    """Reuse the Phase 14 adapter without broadening its optimization API.

    Phase 14 intentionally exposes only train/validation through
    ``load_training_examples``. Phase 15 supplies a shallow read-only index proxy whose
    validation slot points at the physically test-labelled records. The adapter still
    reads each item's true ``split`` field, so every returned example must remain test.
    """
    proxy = copy.copy(dataset)
    proxy.records_by_task_split = dict(dataset.records_by_task_split)
    proxy.records_by_task_split[(task, "validation")] = tuple(
        dataset.records_by_task_split.get((task, "test"), ())
    )
    examples = load_training_examples(
        proxy,
        tasks=(task,),
        split="validation",
        spec=spec,
        phase7_root=phase7_root,
    )
    if any(example.split != "test" for example in examples):
        raise ValueError("Phase 15 adapter proxy returned a non-test example")
    return examples


def _candidate_ids(item: Any) -> tuple[str, ...]:
    values = item.arrays.get("action_candidate_ids")
    if values is None:
        return ()
    return tuple(str(value) for value in values.tolist())


def _safe_float(value: Any, name: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        result = float(value.detach().cpu().item())
    else:
        result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _item_metadata(example: Any) -> tuple[str | None, int, str | None]:
    family = example.metadata.get("family")
    family_value = str(family) if isinstance(family, str) and family else None
    raw_n_qubits = example.metadata.get("n_qubits", example.n_qubits)
    if isinstance(raw_n_qubits, bool) or not isinstance(raw_n_qubits, int):
        raise TypeError("Evaluation n_qubits metadata must be an integer")
    if raw_n_qubits <= 0:
        raise ValueError("Evaluation n_qubits metadata must be positive")
    distortion = example.metadata.get("distortion_id")
    distortion_value = (
        str(distortion)
        if isinstance(distortion, str) and distortion
        else None
    )
    return family_value, raw_n_qubits, distortion_value


def _evaluate_items_for_output(
    *,
    output: Any,
    auxiliary: Any,
    supervised: Any,
    examples: list[Any],
    raw_items: list[Any],
    ablation: str,
    evaluation_run_identifier: str,
    config: EvaluationConfig,
) -> list[EvaluationItemResult]:
    graph_count = supervised.graph_count
    if graph_count != len(examples):
        raise ValueError("Collated graph count does not match evaluation examples")

    diagnosis_probabilities = torch.softmax(
        output.distortion.class_logits,
        dim=-1,
    )
    born_metrics = (
        distribution_metrics_by_graph(
            output.born_prediction.probabilities,
            supervised.targets.born_prediction.probabilities,
            supervised.targets.born_prediction.row_mask,
            supervised.targets.born_prediction.outcome_batch,
            graph_count,
            epsilon=config.distribution_epsilon,
        )
        if supervised.targets.born_prediction.probabilities.numel()
        else [{} for _ in examples]
    )
    hilbert_metrics = (
        distribution_metrics_by_graph(
            auxiliary.born_prediction.probabilities,
            supervised.targets.hilbert_to_born.probabilities,
            supervised.targets.hilbert_to_born.row_mask,
            supervised.targets.hilbert_to_born.outcome_batch,
            graph_count,
            epsilon=config.distribution_epsilon,
        )
        if (
            auxiliary is not None
            and supervised.targets.hilbert_to_born.probabilities.numel()
        )
        else [{} for _ in examples]
    )

    item_results: list[EvaluationItemResult] = []
    for graph_index, (example, raw) in enumerate(
        zip(examples, raw_items, strict=True)
    ):
        metrics: dict[str, float] = {}
        calibration: dict[str, float] = {}
        arrays: dict[str, np.ndarray] = {}
        predicted_action_id: str | None = None
        target_action_id: str | None = None
        target_action_rank: int | None = None

        diagnosis_target = supervised.targets.diagnosis
        if bool(diagnosis_target.class_mask[graph_index]):
            target_index = int(diagnosis_target.class_index[graph_index])
            probabilities = diagnosis_probabilities[graph_index]
            predicted_index = int(torch.argmax(probabilities))
            confidence = _safe_float(
                probabilities[predicted_index],
                "diagnosis confidence",
            )
            correct = float(predicted_index == target_index)
            metrics["diagnosis_accuracy"] = correct
            metrics["diagnosis_nll"] = _safe_float(
                -torch.log(probabilities[target_index].clamp_min(1e-12)),
                "diagnosis nll",
            )
            calibration = {"confidence": confidence, "correct": correct}
            arrays["diagnosis_probabilities"] = (
                probabilities.detach().cpu().numpy()
            )
            arrays["diagnosis_target_index"] = np.asarray(
                [target_index],
                dtype=np.int64,
            )
        if bool(diagnosis_target.strength_mask[graph_index]):
            metrics["diagnosis_strength_absolute_error"] = abs(
                _safe_float(
                    output.distortion.strength_mean[graph_index],
                    "predicted strength",
                )
                - _safe_float(
                    diagnosis_target.strength[graph_index],
                    "target strength",
                )
            )
        node_mask = supervised.model_batch.graph.node_batch == graph_index
        if bool(node_mask.any()) and bool(
            diagnosis_target.affected_qubit_mask[node_mask].any()
        ):
            predicted_qubits = (
                torch.sigmoid(output.distortion.affected_qubit_logits[node_mask])
                >= 0.5
            )
            target_qubits = diagnosis_target.affected_qubit[node_mask] >= 0.5
            metrics["diagnosis_affected_qubit_accuracy"] = _safe_float(
                (predicted_qubits == target_qubits).to(torch.float32).mean(),
                "affected qubit accuracy",
            )

        candidate_mask = (
            supervised.targets.action.candidate_batch == graph_index
        )
        candidate_ids = _candidate_ids(raw)
        if bool(candidate_mask.any()):
            local_scores = output.action_ranking.candidate_scores[candidate_mask]
            local_probabilities = (
                output.action_ranking.candidate_probabilities[candidate_mask]
            )
            local_selected = (
                supervised.targets.action.selected_mask[candidate_mask]
            )
            local_rank = supervised.targets.action.rank[candidate_mask]
            local_reward = supervised.targets.action.reward[candidate_mask]
            local_oracle = (
                supervised.targets.action.privileged_oracle_mask[candidate_mask]
            )
            if len(candidate_ids) != local_scores.numel():
                raise ValueError(
                    f"Action candidate ID count mismatch for {example.view_item_id}"
                )
            predicted_local = int(torch.argmax(local_scores))
            selected_indices = torch.nonzero(
                local_selected,
                as_tuple=False,
            ).reshape(-1)
            if selected_indices.numel() != 1:
                raise ValueError(
                    "Held-out action target must select exactly one candidate"
                )
            target_local = int(selected_indices[0])
            predicted_action_id = candidate_ids[predicted_local]
            target_action_id = candidate_ids[target_local]
            target_action_rank = int(local_rank[predicted_local])
            predicted_order = torch.argsort(local_scores, descending=True)
            target_position = int(
                torch.nonzero(
                    predicted_order == target_local,
                    as_tuple=False,
                ).reshape(-1)[0]
            )
            correct = float(predicted_local == target_local)
            confidence = _safe_float(
                local_probabilities[predicted_local],
                "action confidence",
            )
            metrics.update(
                {
                    "action_top1_accuracy": correct,
                    "action_target_reciprocal_rank": 1.0
                    / (target_position + 1),
                    "action_selected_target_reward": _safe_float(
                        local_reward[predicted_local],
                        "selected target reward",
                    ),
                    "action_selected_target_rank": float(target_action_rank),
                    "action_selected_oracle_fraction": float(
                        bool(local_oracle[predicted_local])
                    ),
                }
            )
            if not calibration:
                calibration = {
                    "confidence": confidence,
                    "correct": correct,
                }
            arrays["action_candidate_probabilities"] = (
                local_probabilities.detach().cpu().numpy()
            )
            arrays["action_candidate_scores"] = (
                local_scores.detach().cpu().numpy()
            )
            arrays["action_target_selected_mask"] = (
                local_selected.detach().cpu().numpy()
            )
            arrays["action_target_rank"] = local_rank.detach().cpu().numpy()
            arrays["action_target_reward"] = (
                local_reward.detach().cpu().numpy()
            )

        metrics.update(born_metrics[graph_index])
        if hilbert_metrics[graph_index]:
            metrics.update(
                {
                    f"hilbert_to_born_{name.removeprefix('born_')}": value
                    for name, value in hilbert_metrics[graph_index].items()
                }
            )
        if supervised.targets.born_prediction.probabilities.numel():
            outcome_mask = (
                supervised.targets.born_prediction.outcome_batch == graph_index
            ) & supervised.targets.born_prediction.row_mask
            if bool(outcome_mask.any()):
                arrays["born_predicted_probabilities"] = (
                    output.born_prediction.probabilities[outcome_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )
                arrays["born_target_probabilities"] = (
                    supervised.targets.born_prediction.probabilities[outcome_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )
        if (
            auxiliary is not None
            and supervised.targets.hilbert_to_born.probabilities.numel()
        ):
            outcome_mask = (
                supervised.targets.hilbert_to_born.outcome_batch == graph_index
            ) & supervised.targets.hilbert_to_born.row_mask
            if bool(outcome_mask.any()):
                arrays["hilbert_to_born_predicted_probabilities"] = (
                    auxiliary.born_prediction.probabilities[outcome_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )
                arrays["hilbert_to_born_target_probabilities"] = (
                    supervised.targets.hilbert_to_born.probabilities[outcome_mask]
                    .detach()
                    .cpu()
                    .numpy()
                )

        if not metrics:
            raise ValueError(
                f"Held-out item {example.view_item_id} produced no supervised metric"
            )
        family, n_qubits, distortion_id = _item_metadata(example)
        identifier = evaluation_item_id(
            evaluation_run_identifier,
            example.view_item_id,
            ablation,
        )
        payload = {
            "evaluation_item_id": identifier,
            "evaluation_run_id": evaluation_run_identifier,
            "view_item_id": example.view_item_id,
            "entity_id": example.entity_id,
            "task": example.task,
            "split": example.split,
            "ablation": ablation,
            "family": family,
            "n_qubits": n_qubits,
            "distortion_id": distortion_id,
            "metrics": metrics,
            "calibration": calibration,
            "predicted_action_id": predicted_action_id,
            "target_action_id": target_action_id,
            "target_action_rank": target_action_rank,
            "metadata": {
                **dict(example.metadata),
                "hardware_execution_performed": False,
                "heldout_split": "test",
            },
        }
        content_hash = evaluation_item_content_hash(payload, arrays)
        item_results.append(
            EvaluationItemResult(
                evaluation_item_id=identifier,
                evaluation_run_id=evaluation_run_identifier,
                view_item_id=example.view_item_id,
                entity_id=example.entity_id,
                task=example.task,
                split=example.split,
                ablation=ablation,
                family=family,
                n_qubits=n_qubits,
                distortion_id=distortion_id,
                metrics=metrics,
                calibration=calibration,
                predicted_action_id=predicted_action_id,
                target_action_id=target_action_id,
                target_action_rank=target_action_rank,
                arrays=arrays,
                metadata=payload["metadata"],
                content_hash=content_hash,
            )
        )
    return item_results


def _build_baseline_comparisons(
    *,
    item_results: list[EvaluationItemResult],
    evaluation_run_identifier: str,
    action_sources: Any,
    baseline_dataset: Any,
) -> list[BaselineComparison]:
    comparisons: list[BaselineComparison] = []
    full_action_items = [
        item
        for item in item_results
        if item.ablation == "full"
        and item.task
        in {"action_ranking", "joint_multitask", "hardware_masked"}
        and item.predicted_action_id is not None
    ]
    for item in full_action_items:
        sample_id = item.entity_id
        rollouts = action_sources.action.rollouts_by_sample_id.get(sample_id)
        if rollouts is None:
            raise ValueError(
                f"No Phase 9 rollouts for held-out sample {sample_id}"
            )
        rollout_by_action = {
            rollout.action_id: rollout
            for rollout in rollouts
        }
        learned_rollout = rollout_by_action.get(item.predicted_action_id)
        if learned_rollout is None:
            raise ValueError(
                f"Learned action {item.predicted_action_id} has no Phase 9 rollout"
            )
        learned_after = weighted_objective(
            learned_rollout.candidate_metric_values,
            baseline_dataset.config,
        )
        for baseline_name in baseline_dataset.config.enabled_baselines:
            baseline = baseline_dataset.results_by_sample_and_name.get(
                (sample_id, baseline_name)
            )
            if baseline is None:
                raise ValueError(
                    f"Phase 10 baseline {baseline_name} misses sample {sample_id}"
                )
            learned_success = (
                learned_after
                < baseline.objective_before
                - baseline_dataset.config.improvement_atol
            )
            comparisons.append(
                BaselineComparison(
                    evaluation_baseline_id=evaluation_baseline_id(
                        evaluation_run_identifier,
                        sample_id,
                        baseline_name,
                    ),
                    evaluation_run_id=evaluation_run_identifier,
                    sample_id=sample_id,
                    baseline_name=baseline_name,
                    learned_action_id=item.predicted_action_id,
                    baseline_action_id=baseline.selected_action_id,
                    objective_before=float(baseline.objective_before),
                    learned_objective_after=float(learned_after),
                    baseline_objective_after=float(baseline.objective_after),
                    learned_minus_baseline=float(
                        learned_after - baseline.objective_after
                    ),
                    learned_success=bool(learned_success),
                    baseline_success=bool(baseline.success),
                    baseline_privileged=(
                        baseline_name in _PRIVILEGED_BASELINES
                    ),
                    metadata={
                        "comparison_objective": (
                            "phase10_weighted_exact_born_objective"
                        ),
                        "lower_is_better": True,
                        "learned_policy_uses_clean_target_during_selection": False,
                        "baseline_access_privilege_preserved": True,
                    },
                )
            )
    return comparisons


def _verify_source_snapshots(
    *,
    dataset: Any,
    training_run: Any,
    action_sources: Any,
    baseline_dataset: Any,
) -> None:
    actual_phase12 = snapshot_managed_files(
        dataset.root,
        dataset.managed_files,
    )
    if actual_phase12 != dataset.snapshot:
        raise RuntimeError("Managed Phase 12 files changed during Phase 15")
    actual_phase14 = snapshot_managed_files(
        training_run.root,
        training_run.managed_files,
    )
    if actual_phase14.aggregate_sha256 != training_run.snapshot_hash:
        raise RuntimeError("Managed Phase 14 files changed during Phase 15")
    if action_sources is not None:
        verify_baseline_source_snapshots(action_sources)
    if baseline_dataset is not None:
        actual_phase10 = snapshot_managed_files(
            baseline_dataset.root,
            baseline_dataset.managed_files,
        )
        if actual_phase10.aggregate_sha256 != baseline_dataset.snapshot_hash:
            raise RuntimeError("Managed Phase 10 files changed during Phase 15")


def run_evaluation(
    *,
    training_view_root: str | Path,
    training_run_root: str | Path,
    output_root: str | Path,
    evaluation_config: EvaluationConfig,
    phase7_root: str | Path | None = None,
    graph_root: str | Path | None = None,
    action_root: str | Path | None = None,
    baseline_root: str | Path | None = None,
) -> EvaluationRunResult:
    """Evaluate one Phase 14 checkpoint on the untouched Phase 12 test split."""
    if not isinstance(evaluation_config, EvaluationConfig):
        raise TypeError("evaluation_config must be EvaluationConfig")
    output = Path(output_root).expanduser().resolve(strict=False)
    sources = [
        Path(training_view_root).expanduser().resolve(strict=False),
        Path(training_run_root).expanduser().resolve(strict=False),
    ]
    optional_roots = [phase7_root, graph_root, action_root, baseline_root]
    sources.extend(
        Path(value).expanduser().resolve(strict=False)
        for value in optional_roots
        if value is not None
    )
    for source in sources:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(
                f"Evaluation output root {output} overlaps source root {source}"
            )
    if output.exists():
        raise FileExistsError(f"Evaluation output root already exists: {output}")

    dataset = load_completed_training_view_dataset(training_view_root)
    training_run = load_completed_training_run(
        training_run_root,
        training_view_root=training_view_root,
        checkpoint_selection=evaluation_config.checkpoint_selection,
    )

    action_sources = None
    baseline_dataset = None
    baseline_suite_id = None
    if evaluation_config.include_baseline_comparison:
        if any(
            value is None
            for value in (phase7_root, graph_root, action_root, baseline_root)
        ):
            raise ValueError(
                "Baseline comparison requires phase7_root, graph_root, "
                "action_root, and baseline_root"
            )
        action_sources = load_baseline_sources(
            phase7_root,
            graph_root,
            action_root,
        )
        expected_source_ids = {
            "source_scientific_generation_id": (
                action_sources.phase7.source_scientific_generation_id
            ),
            "graph_conversion_id": (
                action_sources.graph.completion_marker["graph_conversion_id"]
            ),
            "action_engine_id": (
                action_sources.action.completion_marker["action_engine_id"]
            ),
        }
        baseline_dataset = load_completed_baseline_dataset(
            baseline_root,
            expected_source_ids=expected_source_ids,
        )
        baseline_suite_id = (
            baseline_dataset.completion_marker["baseline_suite_id"]
        )

    schema_identifier = evaluation_schema_id()
    recipe_identifier = evaluation_recipe_id(
        dataset.training_view_dataset_id,
        training_run.training_run_id,
        training_run.checkpoint_record.checkpoint_id,
        evaluation_config,
        baseline_suite_id=baseline_suite_id,
    )
    operational_identifier = evaluation_operational_config_id(
        evaluation_config
    )
    run_identifier = evaluation_run_id(
        recipe_identifier,
        operational_identifier,
    )

    _seed_everything(evaluation_config.seed)
    device = _resolve_device(evaluation_config.device)
    model = TriQTOModel(training_run.model_config).to(device)
    load_training_checkpoint(
        training_run.checkpoint_path,
        model=model,
        restore_rng=False,
        expected_training_run_id=training_run.training_run_id,
    )
    model.eval()

    examples: list[Any] = []
    task_counts: Counter[str] = Counter()
    for task in evaluation_config.tasks:
        task_examples = _load_test_examples(
            dataset,
            task=task,
            spec=training_run.data_spec,
            phase7_root=phase7_root,
        )
        examples.extend(task_examples)
        task_counts[task] = len(task_examples)
    examples.sort(key=lambda item: (item.task, item.view_item_id))
    if len(examples) > evaluation_config.max_items:
        raise RuntimeError(
            f"Phase 15 has {len(examples)} test items, exceeding "
            f"max_items={evaluation_config.max_items}"
        )
    if evaluation_config.require_test_items and not examples:
        raise ValueError("Phase 15 found no held-out test items")
    if len({item.view_item_id for item in examples}) != len(examples):
        raise ValueError("Duplicate held-out view_item_id in evaluation universe")
    if any(item.split != "test" for item in examples):
        raise ValueError("Phase 15 evaluation loader returned a non-test item")

    raw_by_id = {
        example.view_item_id: _raw_item(dataset, example.view_item_id)
        for example in examples
    }
    item_results: list[EvaluationItemResult] = []
    with torch.no_grad():
        for chunk in _chunks(examples, evaluation_config.batch_size):
            raw_items = [raw_by_id[item.view_item_id] for item in chunk]
            base_supervised = collate_training_examples(chunk)
            for ablation in evaluation_config.ablations:
                supervised = _apply_ablation(base_supervised, ablation)
                supervised = _move_tree(supervised, device)
                output_value = model(supervised.model_batch)
                auxiliary = (
                    model(supervised.auxiliary_hilbert_to_born_batch)
                    if supervised.auxiliary_hilbert_to_born_batch is not None
                    else None
                )
                item_results.extend(
                    _evaluate_items_for_output(
                        output=output_value,
                        auxiliary=auxiliary,
                        supervised=supervised,
                        examples=chunk,
                        raw_items=raw_items,
                        ablation=ablation,
                        evaluation_run_identifier=run_identifier,
                        config=evaluation_config,
                    )
                )

    expected_result_count = len(examples) * len(evaluation_config.ablations)
    if len(item_results) != expected_result_count:
        raise ValueError("Phase 15 item result count mismatch")
    aggregates = build_aggregates(
        item_results,
        evaluation_run_id=run_identifier,
        calibration_bins=evaluation_config.calibration_bins,
    )
    baseline_comparisons = (
        _build_baseline_comparisons(
            item_results=item_results,
            evaluation_run_identifier=run_identifier,
            action_sources=action_sources,
            baseline_dataset=baseline_dataset,
        )
        if action_sources is not None and baseline_dataset is not None
        else []
    )

    _verify_source_snapshots(
        dataset=dataset,
        training_run=training_run,
        action_sources=action_sources,
        baseline_dataset=baseline_dataset,
    )

    summary = {
        "evaluation_schema_id": schema_identifier,
        "evaluation_recipe_id": recipe_identifier,
        "operational_config_id": operational_identifier,
        "evaluation_run_id": run_identifier,
        "training_view_dataset_id": dataset.training_view_dataset_id,
        "training_run_id": training_run.training_run_id,
        "checkpoint_id": training_run.checkpoint_record.checkpoint_id,
        "checkpoint_kind": training_run.checkpoint_record.kind,
        "phase12_snapshot_hash": dataset.snapshot.aggregate_sha256,
        "phase14_snapshot_hash": training_run.snapshot_hash,
        "phase10_snapshot_hash": (
            baseline_dataset.snapshot_hash
            if baseline_dataset is not None
            else None
        ),
        "heldout_split": "test",
        "heldout_evaluation_performed": True,
        "test_split_item_count": len(examples),
        "evaluation_item_result_count": len(item_results),
        "aggregate_count": len(aggregates),
        "baseline_comparison_count": len(baseline_comparisons),
        "task_item_counts": dict(sorted(task_counts.items())),
        "ablations": list(evaluation_config.ablations),
        "topology_loss_weight": 0.0,
        "hardware_execution_performed": False,
        "real_hardware_data_present": False,
        "training_performed": False,
        "gradients_enabled": False,
        "validation_split_used": False,
        "universal_correction_claim": False,
        "quantum_advantage_claim": False,
    }
    return EvaluationRunResult(
        training_view_root=dataset.root,
        training_run_root=training_run.root,
        phase7_root=(
            Path(phase7_root).expanduser().resolve(strict=False)
            if phase7_root is not None
            else None
        ),
        graph_root=(
            Path(graph_root).expanduser().resolve(strict=False)
            if graph_root is not None
            else None
        ),
        action_root=(
            Path(action_root).expanduser().resolve(strict=False)
            if action_root is not None
            else None
        ),
        baseline_root=(
            Path(baseline_root).expanduser().resolve(strict=False)
            if baseline_root is not None
            else None
        ),
        config=evaluation_config,
        evaluation_schema_id=schema_identifier,
        evaluation_recipe_id=recipe_identifier,
        operational_config_id=operational_identifier,
        evaluation_run_id=run_identifier,
        training_view_dataset_id=dataset.training_view_dataset_id,
        training_run_id=training_run.training_run_id,
        checkpoint_id=training_run.checkpoint_record.checkpoint_id,
        item_results=item_results,
        aggregates=aggregates,
        baseline_comparisons=baseline_comparisons,
        summary=summary,
    )


__all__ = ["run_evaluation"]
