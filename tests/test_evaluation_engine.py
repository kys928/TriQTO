from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from triqto.evaluation import (
    EvaluationConfig,
    EvaluationItemResult,
    distribution_metrics_by_graph,
    evaluation_config_from_dict,
    evaluation_config_to_dict,
    expected_calibration_error,
    load_evaluation_item_artifact,
    save_evaluation_item_artifact,
)
from triqto.evaluation.evaluator import _apply_ablation
from triqto.model import (
    GraphTensorBatch,
    HilbertTensorBatch,
    TriQTOBatch,
)
from triqto.model.constants import HEAD_ORDER, STREAM_ORDER
from triqto.training.models import (
    ActionTargets,
    BornTargets,
    DiagnosisTargets,
    GeometryTargets,
    SupervisedBatch,
    TrainingTargets,
)


def test_evaluation_config_is_strict_and_roundtrips() -> None:
    config = EvaluationConfig(
        tasks=("diagnosis", "born_prediction"),
        ablations=("full", "no_topology"),
        device="cpu",
    )
    assert evaluation_config_from_dict(evaluation_config_to_dict(config)) == config
    with pytest.raises(ValueError, match="full"):
        EvaluationConfig(ablations=("no_topology",))
    with pytest.raises(ValueError, match="fixed order"):
        EvaluationConfig(tasks=("born_prediction", "diagnosis"))
    with pytest.raises(ValueError, match="checkpoint_selection"):
        EvaluationConfig(checkpoint_selection="latest")
    with pytest.raises(TypeError, match="bool"):
        EvaluationConfig(batch_size=True)


def test_distribution_metrics_average_complete_graph_supports() -> None:
    predicted = torch.tensor(
        [0.5, 0.5, 0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    rows = distribution_metrics_by_graph(
        predicted,
        target,
        torch.ones(6, dtype=torch.bool),
        torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long),
        2,
        epsilon=1e-12,
    )
    assert len(rows) == 2
    assert rows[0]["born_kl"] == pytest.approx(np.log(2.0))
    assert rows[1]["born_kl"] == pytest.approx(np.log(2.0))
    assert rows[0]["born_hellinger"] == pytest.approx(
        rows[1]["born_hellinger"]
    )


def test_calibration_metrics_are_transparent() -> None:
    values = expected_calibration_error(
        [0.9, 0.8, 0.2, 0.1],
        [1.0, 1.0, 0.0, 0.0],
        bins=2,
    )
    assert values["calibration_empirical_accuracy"] == pytest.approx(0.5)
    assert values["calibration_mean_confidence"] == pytest.approx(0.5)
    assert 0.0 <= values["calibration_ece"] <= 1.0
    with pytest.raises(ValueError, match="confidence"):
        expected_calibration_error([1.2], [1.0], bins=2)


def _supervised_with_hilbert_and_topology() -> SupervisedBatch:
    graph = GraphTensorBatch(
        node_features=torch.zeros((1, 18)),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_features=torch.zeros((0, 14)),
        edge_event_index=torch.zeros(0, dtype=torch.long),
        gate_features=torch.zeros((0, 24)),
        gate_qubit_ptr=torch.tensor([0], dtype=torch.long),
        gate_qubit_indices=torch.zeros(0, dtype=torch.long),
        node_batch=torch.tensor([0], dtype=torch.long),
        gate_batch=torch.zeros(0, dtype=torch.long),
        graph_count=1,
    )
    hilbert = HilbertTensorBatch(
        amplitudes_real_imag=torch.tensor([[1.0, 0.0]]),
        basis_bits=torch.tensor([[0.0]]),
        basis_bit_mask=torch.tensor([[True]]),
        batch_index=torch.tensor([0], dtype=torch.long),
        available_mask=torch.tensor([True]),
    )
    head_stream = torch.ones(
        (1, len(HEAD_ORDER), len(STREAM_ORDER)),
        dtype=torch.bool,
    )
    batch = TriQTOBatch(
        graph=graph,
        hilbert=hilbert,
        head_stream_mask=head_stream,
        head_active_mask=torch.ones((1, len(HEAD_ORDER)), dtype=torch.bool),
    )
    empty_long = torch.zeros(0, dtype=torch.long)
    empty_float = torch.zeros(0)
    empty_bool = torch.zeros(0, dtype=torch.bool)
    targets = TrainingTargets(
        diagnosis=DiagnosisTargets(
            class_index=torch.zeros(1, dtype=torch.long),
            class_mask=torch.zeros(1, dtype=torch.bool),
            strength=torch.zeros(1),
            strength_mask=torch.zeros(1, dtype=torch.bool),
            affected_qubit=torch.zeros(1),
            affected_qubit_mask=torch.zeros(1, dtype=torch.bool),
        ),
        action=ActionTargets(
            rank=empty_long,
            reward=empty_float,
            selected_mask=empty_bool,
            candidate_target_mask=empty_bool,
            privileged_oracle_mask=empty_bool,
            candidate_batch=empty_long,
        ),
        born_prediction=BornTargets(empty_float, empty_long, empty_bool),
        hilbert_to_born=BornTargets(empty_float, empty_long, empty_bool),
        geometry=GeometryTargets(
            target_distance=torch.zeros((1, 1)),
            pair_mask=torch.zeros((1, 1), dtype=torch.bool),
        ),
    )
    return SupervisedBatch(
        item_ids=("item",),
        entity_ids=("entity",),
        tasks=("born_prediction",),
        splits=("test",),
        split_group_ids=("group",),
        model_batch=batch,
        auxiliary_hilbert_to_born_batch=None,
        targets=targets,
        graph_task_names=("born_prediction",),
        privileged_item_mask=torch.tensor([False]),
    )


def test_inference_ablations_remove_stream_and_mask() -> None:
    supervised = _supervised_with_hilbert_and_topology()
    no_hilbert = _apply_ablation(supervised, "no_hilbert")
    assert no_hilbert.model_batch.hilbert is None
    assert not no_hilbert.model_batch.head_stream_mask[
        :, :, STREAM_ORDER.index("hilbert")
    ].any()
    assert supervised.model_batch.hilbert is not None


def test_evaluation_item_artifact_roundtrip_and_corruption(tmp_path: Path) -> None:
    item = EvaluationItemResult(
        evaluation_item_id="evalitem_test",
        evaluation_run_id="evalrun_test",
        view_item_id="view_test",
        entity_id="sample_test",
        task="born_prediction",
        split="test",
        ablation="full",
        family="bell",
        n_qubits=2,
        distortion_id=None,
        metrics={"born_hellinger": 0.1},
        calibration={},
        arrays={
            "born_predicted_probabilities": np.asarray([0.6, 0.4]),
            "born_target_probabilities": np.asarray([0.5, 0.5]),
        },
        metadata={"hardware_execution_performed": False},
    )
    path = tmp_path / "item.npz"
    content_hash = save_evaluation_item_artifact(item, path)
    loaded = load_evaluation_item_artifact(
        path,
        expected_content_hash=content_hash,
    )
    assert loaded["evaluation_item_id"] == item.evaluation_item_id
    with np.load(path, allow_pickle=False) as artifact:
        arrays = {name: artifact[name].copy() for name in artifact.files}
    arrays["born_predicted_probabilities"][0] += 0.1
    corrupted = tmp_path / "corrupted.npz"
    np.savez_compressed(corrupted, **arrays)
    with pytest.raises(ValueError, match="content hash"):
        load_evaluation_item_artifact(corrupted)
