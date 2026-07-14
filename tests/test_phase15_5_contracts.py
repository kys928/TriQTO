from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

from triqto.phase15_5 import NoiseProfileConfig, Phase155Config
from triqto.phase15_5.policy import (
    PolicyDataset,
    load_policy_checkpoint,
    save_policy_checkpoint,
    score_dataset,
    train_operational_policy,
)
from triqto.simulation import NoiseSpec, simulate_noisy_aer_shots


def _dataset() -> PolicyDataset:
    candidate_ids = tuple(f"candidate_{index}" for index in range(12))
    group_ids = (
        "train_a", "train_a", "train_b", "train_b",
        "validation_a", "validation_a", "validation_b", "validation_b",
        "test_a", "test_a", "test_b", "test_b",
    )
    splits = ("train",) * 4 + ("validation",) * 4 + ("test",) * 4
    split_groups = tuple(f"group_{index // 2}" for index in range(12))
    family_ids = np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    context = np.asarray([[float(index // 2), 1.0] for index in range(12)], dtype=np.float64)
    candidate = np.asarray([[float(index % 2), float((index + 1) % 2)] for index in range(12)], dtype=np.float64)
    utilities = np.asarray([0.1, 0.9] * 6, dtype=np.float64)
    available = np.ones(12, dtype=np.bool_)
    return PolicyDataset(
        candidate_ids=candidate_ids,
        group_ids=group_ids,
        split_group_ids=split_groups,
        splits=splits,
        family_ids=family_ids,
        context_features=context,
        candidate_features=candidate,
        utilities=utilities,
        available_mask=available,
    )


def test_phase155_config_rejects_hardware_and_nonzero_topology() -> None:
    profile = NoiseProfileConfig(name="smoke", channels=({"type": "readout_error", "probability": 0.01},), shots=32)
    config = Phase155Config(noise_profiles=(profile,), max_samples_per_split=1, epochs=1, bootstrap_replicates=4)
    assert config.physical_hardware is False
    assert config.topology_loss_weight == 0.0
    with pytest.raises(ValueError, match="physical hardware"):
        Phase155Config(noise_profiles=(profile,), physical_hardware=True)
    with pytest.raises(ValueError, match="topology_loss_weight"):
        Phase155Config(noise_profiles=(profile,), topology_loss_weight=0.1)


def test_noisy_shots_are_basis_conditioned_and_deterministic() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    noise = NoiseSpec(({"type": "readout_error", "probability": 0.01},))
    first = simulate_noisy_aer_shots(circuit, noise=noise, shots=64, seed=7, measurement_basis="X")
    second = simulate_noisy_aer_shots(circuit, noise=noise, shots=64, seed=7, measurement_basis="X")
    assert first.counts == second.counts
    assert first.metadata["measurement_setting"]["measurement_bases"] == ["X"]
    assert first.metadata["physical_hardware"] is False
    with pytest.raises(ValueError, match="length|qubit"):
        simulate_noisy_aer_shots(circuit, noise=noise, shots=8, seed=1, measurement_basis=("X", "Y"))


def test_policy_training_uses_train_validation_and_preserves_test(tmp_path: Path) -> None:
    dataset = _dataset()
    result = train_operational_policy(
        dataset,
        hidden_dim=8,
        epochs=3,
        learning_rate=1e-2,
        weight_decay=0.0,
        utility_mse_weight=0.1,
        seed=11,
    )
    checkpoint = tmp_path / "policy.npz"
    metadata = save_policy_checkpoint(
        checkpoint,
        training_result=result,
        source_identity={"phase12": "fixture", "checkpoint": "trained_fixture"},
        config_identity={"run": "fixture"},
    )
    loaded = load_policy_checkpoint(checkpoint)
    scores = score_dataset(
        loaded["model"],
        dataset,
        context_mean=loaded["context_mean"],
        context_std=loaded["context_std"],
        candidate_mean=loaded["candidate_mean"],
        candidate_std=loaded["candidate_std"],
    )
    assert scores.shape == (12,)
    assert metadata["trained"] is True
    assert metadata["physical_hardware"] is False
    assert metadata["topology_loss_weight"] == 0.0
    assert result["best_epoch"] in {0, 1, 2}


def test_policy_checkpoint_tamper_is_rejected(tmp_path: Path) -> None:
    dataset = _dataset()
    result = train_operational_policy(dataset, hidden_dim=8, epochs=1, learning_rate=1e-2, weight_decay=0.0, utility_mse_weight=0.0, seed=3)
    checkpoint = tmp_path / "policy.npz"
    save_policy_checkpoint(checkpoint, training_result=result, source_identity={"source": "fixture"}, config_identity={"config": "fixture"})
    metadata_path = checkpoint.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["best_epoch"] += 1
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        load_policy_checkpoint(checkpoint)
