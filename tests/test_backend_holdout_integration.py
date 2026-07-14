from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triqto.actions import ActionEngineConfig, build_action_engine_result, write_action_dataset
from triqto.data_generation import DatasetGenerationConfig, generate_dataset, load_generation_config, write_dataset
from triqto.evaluation.generalization_tests import (
    BackendHoldoutConfig,
    audit_backend_holdout_for_phase15,
    load_backend_holdout_config,
)
from triqto.graph import GraphConversionConfig, convert_completed_dataset_to_graphs, write_graph_dataset
from triqto.model import TriQTOModel, TriQTOModelConfig
from triqto.topology import TopologyAuditConfig, build_topology_audit_result, write_topology_dataset
from triqto.training import (
    CurriculumStageConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    build_training_data_spec,
    load_completed_training_view_dataset,
    load_training_examples,
    run_training,
)
from triqto.training_views import TrainingViewConfig, build_training_view_result, load_training_view_config, write_training_view_dataset


def _backend_config() -> DatasetGenerationConfig:
    return load_generation_config("configs/data/backend_holdout_generation.json")


def _build_sources(tmp_path: Path):
    phase7_root = tmp_path / "phase7"
    graph_root = tmp_path / "phase8"
    action_root = tmp_path / "phase9"
    topology_root = tmp_path / "phase11"
    result = generate_dataset(_backend_config())
    write_dataset(result, phase7_root)
    graph = convert_completed_dataset_to_graphs(phase7_root, GraphConversionConfig(include_supplemental_counts=False))
    write_graph_dataset(graph, graph_root)
    action = build_action_engine_result(
        phase7_root,
        graph_root,
        ActionEngineConfig(candidate_magnitudes=(0.1, 0.2), max_candidates_per_sample=64, max_edits_per_action=16),
    )
    write_action_dataset(action, action_root)
    topology = build_topology_audit_result(
        phase7_root,
        graph_root,
        action_root,
        TopologyAuditConfig(min_points=3, betti_grid_size=8, top_k_lifetimes=2, max_points_per_group=128, max_groups=64, max_statevector_amplitudes=64, include_hilbert=False),
    )
    write_topology_dataset(topology, topology_root)
    return result, graph, phase7_root, graph_root, action_root, topology_root


def _holdout_records_from_samples(result):
    return [
        {
            "sample_id": sample.sample_id,
            "clean_circuit_id": sample.clean_circuit_id,
            "family": sample.family,
            "n_qubits": sample.n_qubits,
            "distortion_type": sample.metadata["distortion_name"],
            "backend_id": sample.metadata["backend_id"],
            "backend_assignment_level": sample.metadata["backend_assignment_level"],
            "backend_available": True,
        }
        for sample in result.samples
    ]


def test_backend_assignment_is_deterministic_clean_level_and_not_distortion_order() -> None:
    first = generate_dataset(_backend_config())
    second = generate_dataset(_backend_config())
    assert [sample.metadata["backend_id"] for sample in first.samples] == [sample.metadata["backend_id"] for sample in second.samples]
    by_clean: dict[str, set[str]] = {}
    for sample in first.samples:
        by_clean.setdefault(sample.clean_circuit_id, set()).add(sample.metadata["backend_id"])
        assert sample.metadata["backend_assignment_level"] == "clean_circuit"
        assert sample.metadata["backend_assignment_key"] == sample.clean_circuit_id
    assert all(len(values) == 1 for values in by_clean.values())
    assert len({next(iter(values)) for values in by_clean.values()}) >= 2

    reversed_config = DatasetGenerationConfig(
        **{**json.loads(Path("configs/data/backend_holdout_generation.json").read_text()), "distortion_specs": list(reversed(json.loads(Path("configs/data/backend_holdout_generation.json").read_text())["distortion_specs"]))}
    )
    reversed_result = generate_dataset(reversed_config)
    original = {sample.clean_circuit_id: sample.metadata["backend_id"] for sample in first.samples}
    reordered = {sample.clean_circuit_id: sample.metadata["backend_id"] for sample in reversed_result.samples}
    assert original == reordered


def test_graph_view_model_training_and_phase15_backend_holdout_path(tmp_path: Path) -> None:
    result, graph_result, phase7_root, graph_root, action_root, topology_root = _build_sources(tmp_path)
    phase12_root = tmp_path / "phase12"
    phase14_root = tmp_path / "phase14"

    assert all(record.metadata["backend_assignment_level"] == "clean_circuit" for record in graph_result.graph_pair_records)
    assert {record.metadata["backend_id"] for record in graph_result.graph_pair_records} == {sample.metadata["backend_id"] for sample in result.samples}
    view_config = load_training_view_config("configs/training_views/backend_holdout.yaml")
    view_result = build_training_view_result(phase7_root, graph_root, action_root, topology_root, view_config)
    diagnosis = [item for item in view_result.items if item.task == "diagnosis"]
    hardware = [item for item in view_result.items if item.task == "hardware_masked"]
    assert diagnosis and hardware
    assert all(item.metadata["backend_available"] is True for item in diagnosis)
    assert all(item.arrays["backend_available_mask"].tolist() == [True] for item in diagnosis)
    assert all("gate_error_summary" in set(item.arrays["backend_missing_feature_names"].tolist()) for item in diagnosis)
    assert all(item.metadata["backend_available"] is False for item in hardware)
    assert all(item.arrays["backend_available_mask"].tolist() == [True] for item in hardware)
    assert all(item.arrays["hardware_head_input_mask"][:, -1].tolist() == [False, False, False, False] for item in hardware)

    write_training_view_dataset(view_result, phase12_root)
    dataset = load_completed_training_view_dataset(phase12_root)
    model_config = TriQTOModelConfig(hidden_dim=32, graph_message_passing_layers=1, residual_mlp_layers=1, backend_input_dim=16, topology_input_dim=8, hilbert_deformation_dim=8, topology_prediction_dim=8, dropout=0.0)
    training_config = TrainingConfig(
        run_name="backend_holdout_smoke",
        seed=2026,
        stages=(CurriculumStageConfig(name="diagnosis", epochs=1, tasks=("diagnosis",)),),
        batch_size=2,
        optimizer=OptimizerConfig(name="adamw", learning_rate=1e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(name="constant", warmup_steps=0, minimum_learning_rate_ratio=1.0),
        loss=LossConfig(geometry_weight=0.0, uncertainty_weighting=False),
        deterministic_algorithms=True,
        device="cpu",
        checkpoint_every_epochs=1,
        early_stopping_patience=0,
        topology_loss_weight=0.0,
    )
    spec = build_training_data_spec(dataset, model_config, training_config)
    train_examples = load_training_examples(dataset, tasks=("diagnosis",), split="train", spec=spec, phase7_root=phase7_root)
    assert train_examples and train_examples[0].model_batch.backend is not None
    model_output = TriQTOModel(model_config).eval()(train_examples[0].model_batch)
    assert model_output.stream_embeddings.shape[1] >= 6
    train_result = run_training(training_view_root=phase12_root, output_root=phase14_root, training_config=training_config, model_config=model_config, phase7_root=phase7_root)
    assert train_result.global_step > 0

    phase15_config = load_backend_holdout_config("configs/eval/phase15_backend_holdout.yaml")
    audit = audit_backend_holdout_for_phase15(_holdout_records_from_samples(result), phase15_config)
    assert audit["claim_label"] == "exact_fake_backend_axis_holdout"
    assert audit["physical_hardware"] is False
    heldout = set(audit["heldout_backend_ids"])
    for record in _holdout_records_from_samples(result):
        split = audit["assignment"][record["sample_id"]]
        if split in {"train", "validation"}:
            assert record["backend_id"] not in heldout
        if record["backend_id"] in heldout:
            assert split == "test"


def test_backend_normalization_uses_unique_training_entities(tmp_path: Path) -> None:
    _, _, phase7_root, graph_root, action_root, topology_root = _build_sources(tmp_path)
    phase12_root = tmp_path / "phase12"
    view_result = build_training_view_result(
        phase7_root,
        graph_root,
        action_root,
        topology_root,
        TrainingViewConfig(tasks=("diagnosis", "action_ranking", "born_prediction", "joint_multitask"), train_fraction=1.0, validation_fraction=0.0, test_fraction=0.0, include_hilbert=False, include_topology=False),
    )
    write_training_view_dataset(view_result, phase12_root)
    dataset = load_completed_training_view_dataset(phase12_root)
    spec = build_training_data_spec(dataset, TriQTOModelConfig(), TrainingConfig(normalize_backend_features=True))
    unique_rows: dict[tuple[str, str], np.ndarray] = {}
    for item in view_result.items:
        if "backend_features" in item.arrays and bool(item.arrays["backend_available_mask"][0]):
            backend_id = str(item.arrays["backend_id"][0])
            unique_rows.setdefault((item.split_group_id, backend_id), item.arrays["backend_features"].reshape(-1))
    expected = np.stack(list(unique_rows.values()), axis=0).mean(axis=0)
    assert np.allclose(np.asarray(spec.backend_feature_mean), expected)


def test_phase15_backend_holdout_rejects_absent_evidence_and_leakage() -> None:
    result = generate_dataset(_backend_config())
    config = load_backend_holdout_config("configs/eval/phase15_backend_holdout.yaml")
    records = _holdout_records_from_samples(result)
    audit_backend_holdout_for_phase15(records, config)
    missing = [dict(row) for row in records]
    missing[0]["backend_available"] = False
    with pytest.raises(ValueError, match="backend evidence"):
        audit_backend_holdout_for_phase15(missing, config)
    leaked = [dict(row) for row in records]
    heldout = config.heldout_backend_ids[0]
    nonheldout = next(row for row in leaked if row["backend_id"] != heldout)
    nonheldout["backend_id"] = heldout
    with pytest.raises(ValueError, match="holdout would leave no train|multiple backend_ids|overlap|crosses heldout"):
        audit_backend_holdout_for_phase15(leaked, config)


def test_backend_holdout_configs_execute() -> None:
    result = generate_dataset(_backend_config())
    assert result.summary["backend_assignment_level"] == "clean_circuit"
    assert set(result.summary["backend_counts"]) == {
        "backend_71ef1457ca6a40489b80c880fcfa67bc",
        "backend_e7791b66840d793a4c06c22c9ddc30d5",
    }
    view_config = load_training_view_config("configs/training_views/backend_holdout.yaml")
    assert "diagnosis" in view_config.tasks and "hardware_masked" in view_config.tasks
    phase15_config = load_backend_holdout_config("configs/eval/phase15_backend_holdout.yaml")
    assert phase15_config.heldout_backend_ids == ("backend_e7791b66840d793a4c06c22c9ddc30d5",)
    assert BackendHoldoutConfig(
        schema_version=phase15_config.schema_version,
        evaluation=phase15_config.evaluation,
        axis=phase15_config.axis,
        heldout_backend_ids=phase15_config.heldout_backend_ids,
        claim_label=phase15_config.claim_label,
        seed=phase15_config.seed,
        backend_assignment_level=phase15_config.backend_assignment_level,
        evidence_tier=phase15_config.evidence_tier,
        physical_hardware=phase15_config.physical_hardware,
    ) == phase15_config
