from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
import torch

from triqto.data_generation import CircuitGenerationSpec, DatasetGenerationConfig, DistortionSpec, generate_dataset
from triqto.graph import convert_completed_dataset_to_graphs
from triqto.model.contracts import BornTensorBatch
from triqto.simulation import simulate_ideal_statevector


def test_basis_conditioned_probabilities_are_normalized_and_distinct() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    z = simulate_ideal_statevector(circuit, measurement_basis="Z")
    x = simulate_ideal_statevector(circuit, measurement_basis="X")
    assert sum(z.probabilities.values()) == pytest.approx(1.0)
    assert sum(x.probabilities.values()) == pytest.approx(1.0)
    assert z.metadata["measurement_setting"]["measurement_bases"] == ["Z"]
    assert x.metadata["measurement_setting"]["measurement_bases"] == ["X"]
    assert z.probabilities != x.probabilities


def _phase_config(*, bases=("Z",), strict=False) -> DatasetGenerationConfig:
    return DatasetGenerationConfig(
        dataset_name="identifiability",
        base_seed=77,
        circuit_specs=[
            CircuitGenerationSpec(
                family="hardware_efficient_ansatz",
                n_qubits=1,
                generator_kwargs={"layers": 1, "entanglement": "none", "measure": True},
                repetitions=1,
            )
        ],
        distortion_specs=[DistortionSpec(name="phase_rz_drift", kwargs={"strength": 0.5, "qubits": [0]})],
        store_statevectors=False,
        measurement_bases=bases,
        strict_identifiability=strict,
        max_samples=1,
    )


def test_z_only_rz_phase_blindness_masks_diagnosis_and_action() -> None:
    result = generate_dataset(_phase_config(bases=("Z",)))
    sample = result.samples[0]
    assert sample.metadata["identifiability_status"] == "unidentifiable"
    assert sample.metadata["identifiability_reason"] == "computational_basis_phase_blindness"
    assert sample.metadata["diagnosis_supervision_mask"] is False
    assert sample.metadata["action_supervision_mask"] is False
    assert sample.metadata["born_target_mask"] is True
    assert result.summary["identifiability"]["diagnosis_supervised_targets"] == 0


def test_non_z_basis_makes_phase_target_conditionally_identifiable() -> None:
    result = generate_dataset(_phase_config(bases=("X",)))
    sample = result.samples[0]
    assert sample.metadata["identifiability_status"] == "conditionally_identifiable"
    assert sample.metadata["diagnosis_supervision_mask"] is True
    assert sample.clean_result.metadata["measurement_setting"]["measurement_bases"] == ["X"]


def test_strict_identifiability_rejects_unidentifiable_target() -> None:
    with pytest.raises(ValueError, match="Unidentifiable target"):
        generate_dataset(_phase_config(bases=("Z",), strict=True))


def test_graph_pair_carries_measurement_and_identifiability(tmp_path) -> None:
    root = tmp_path / "source"
    from triqto.data_generation import write_dataset

    write_dataset(generate_dataset(_phase_config(bases=("Z",))), root)
    graph_result = convert_completed_dataset_to_graphs(root)
    pair = graph_result.pairs[0]
    assert pair.identifiability_status == "unidentifiable"
    assert pair.diagnosis_supervision_mask is False
    graph = next(g for g in graph_result.graphs if g.graph_id == pair.clean_graph_id)
    assert graph.measurement_basis_codes.tolist() == [0]
    assert graph.scientific_metadata["probability_domain"] == "p(y|M)"


def test_born_tensor_batch_validates_basis_context_against_leakage() -> None:
    batch = BornTensorBatch(
        outcome_bits=torch.tensor([[0.0], [1.0]]),
        outcome_bit_mask=torch.tensor([[True], [True]]),
        probabilities=torch.tensor([0.5, 0.5]),
        batch_index=torch.tensor([0, 0], dtype=torch.long),
        available_mask=torch.tensor([True]),
        measurement_basis_codes=torch.tensor([[0], [1]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="identical within each graph"):
        batch.validate(1)
