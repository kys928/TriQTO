from __future__ import annotations

import json
from pathlib import Path

from triqto.actions import (
    ActionEngineConfig,
    build_action_engine_result,
    write_action_dataset,
)
from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    generate_dataset,
    write_dataset,
)
from triqto.graph import (
    GraphConversionConfig,
    convert_completed_dataset_to_graphs,
    snapshot_managed_files,
    write_graph_dataset,
)
from triqto.model import TriQTOModelConfig
from triqto.topology import (
    TopologyAuditConfig,
    build_topology_audit_result,
    write_topology_dataset,
)
from triqto.training import (
    CurriculumStageConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    run_training,
)
from triqto.training_views import (
    TrainingViewConfig,
    build_training_view_result,
    write_training_view_dataset,
)
from triqto.training_views.splits import assign_split


def _managed_snapshot(root: Path, marker_name: str):
    marker = json.loads((root / marker_name).read_text())
    return snapshot_managed_files(root, tuple(marker["managed_files"]))


def test_phase7_to_phase14_born_training_pipeline_is_complete_and_immutable(
    tmp_path: Path,
) -> None:
    phase7_root = tmp_path / "phase7"
    graph_root = tmp_path / "phase8"
    action_root = tmp_path / "phase9"
    topology_root = tmp_path / "phase11"
    phase12_root = tmp_path / "phase12"
    phase14_root = tmp_path / "phase14"

    generation_config = DatasetGenerationConfig(
        dataset_name="phase14-integration-source",
        base_seed=1414,
        circuit_specs=[
            CircuitGenerationSpec(
                family="bell",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=8,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            )
        ],
        store_statevectors=False,
        max_samples=8,
    )
    generation_result = generate_dataset(generation_config)
    write_dataset(generation_result, phase7_root)

    graph_result = convert_completed_dataset_to_graphs(
        phase7_root,
        GraphConversionConfig(include_supplemental_counts=False),
    )
    write_graph_dataset(graph_result, graph_root)

    action_result = build_action_engine_result(
        phase7_root,
        graph_root,
        ActionEngineConfig(
            candidate_magnitudes=(0.2,),
            max_candidates_per_sample=64,
            max_edits_per_action=16,
        ),
    )
    write_action_dataset(action_result, action_root)

    topology_result = build_topology_audit_result(
        phase7_root,
        graph_root,
        action_root,
        TopologyAuditConfig(
            min_points=3,
            betti_grid_size=8,
            top_k_lifetimes=2,
            max_points_per_group=128,
            max_groups=64,
            max_statevector_amplitudes=64,
        ),
    )
    write_topology_dataset(topology_result, topology_root)

    clean_circuit_ids = sorted(
        {sample.clean_circuit_id for sample in generation_result.samples}
    )
    assert len(clean_circuit_ids) >= 2
    view_config = None
    for split_seed in range(10_000):
        candidate = TrainingViewConfig(
            tasks=("born_prediction",),
            split_seed=split_seed,
            train_fraction=0.5,
            validation_fraction=0.5,
            test_fraction=0.0,
            max_items=1000,
            max_candidates_per_item=128,
            max_source_refs_per_item=1024,
        )
        assigned = {assign_split(circuit_id, candidate) for circuit_id in clean_circuit_ids}
        if assigned == {"train", "validation"}:
            view_config = candidate
            break
    assert view_config is not None

    view_result = build_training_view_result(
        phase7_root,
        graph_root,
        action_root,
        topology_root,
        view_config,
    )
    write_training_view_dataset(view_result, phase12_root)

    phase7_before = _managed_snapshot(phase7_root, "dataset_complete.json")
    phase12_before = _managed_snapshot(
        phase12_root,
        "training_view_complete.json",
    )

    training_config = TrainingConfig(
        run_name="phase14_integration",
        seed=1414,
        stages=(
            CurriculumStageConfig(
                name="born_only",
                epochs=1,
                tasks=("born_prediction",),
            ),
        ),
        batch_size=2,
        optimizer=OptimizerConfig(
            name="adamw",
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
        scheduler=SchedulerConfig(
            name="constant",
            warmup_steps=0,
            minimum_learning_rate_ratio=1.0,
        ),
        loss=LossConfig(
            geometry_weight=0.0,
            uncertainty_weighting=False,
        ),
        deterministic_algorithms=True,
        device="cpu",
        checkpoint_every_epochs=1,
        early_stopping_patience=0,
        topology_loss_weight=0.0,
    )
    model_config = TriQTOModelConfig(
        hidden_dim=32,
        graph_message_passing_layers=1,
        residual_mlp_layers=1,
        backend_input_dim=4,
        topology_input_dim=64,
        hilbert_deformation_dim=8,
        topology_prediction_dim=8,
        dropout=0.0,
        initialization_seed=1414,
    )

    result = run_training(
        training_view_root=phase12_root,
        output_root=phase14_root,
        training_config=training_config,
        model_config=model_config,
        phase7_root=phase7_root,
    )

    marker = json.loads((phase14_root / "training_complete.json").read_text())
    assert marker["complete"] is True
    assert marker["training_run_id"] == result.training_run_id
    assert marker["topology_loss_weight"] == 0.0
    assert result.global_step > 0
    assert len(result.epoch_metrics) == 1
    assert result.epoch_metrics[0].topology_loss_weight == 0.0
    assert list((phase14_root / "artifacts" / "checkpoints").glob("*.npz"))

    assert _managed_snapshot(phase7_root, "dataset_complete.json") == phase7_before
    assert _managed_snapshot(
        phase12_root,
        "training_view_complete.json",
    ) == phase12_before
