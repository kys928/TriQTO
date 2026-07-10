from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from triqto.actions import (
    ActionCandidate,
    ActionEdit,
    ActionEngineConfig,
    action_content_hash,
    action_risk_score,
    candidate_action_id,
    load_action_artifact,
    load_candidate_circuit,
    load_rollout_artifact,
    run_action_rollouts,
    save_action_artifact,
    save_candidate_circuit,
    save_rollout_artifact,
)
from triqto.simulation import simulate_ideal_statevector


def candidate(edits, source, config):
    edits = tuple(edits)
    action_id = candidate_action_id(
        sample_id="sample_1",
        graph_pair_id="graphpair_1",
        source_circuit_id="distorted_circuit",
        source_run_id="distorted_run",
        edits=edits,
    )
    item = ActionCandidate(
        action_id=action_id,
        sample_id="sample_1",
        graph_pair_id="graphpair_1",
        source_circuit_id="distorted_circuit",
        source_run_id="distorted_run",
        distortion_id="distortion_1",
        edits=edits,
        generation_sources=(source,),
        risk_score=action_risk_score(edits, config),
        metadata={},
    )
    item.content_hash = action_content_hash(item)
    return item


def test_exact_inverse_is_ranked_above_no_op():
    distorted = QuantumCircuit(1, 1)
    distorted.rx(0.2, 0)
    distorted.measure(0, 0)
    distorted_probabilities = simulate_ideal_statevector(distorted).probabilities
    config = ActionEngineConfig(candidate_magnitudes=(0.2,))
    no_op = candidate((), "no_op", config)
    inverse = candidate(
        (ActionEdit("append_rx", (0,), -0.2),),
        "oracle_inverse",
        config,
    )

    rollouts = run_action_rollouts(
        distorted_circuit=distorted,
        clean_target_run_id="clean_run",
        clean_probabilities={"0": 1.0, "1": 0.0},
        distorted_probabilities=distorted_probabilities,
        candidates=[no_op, inverse],
        config=config,
    )

    assert len(rollouts) == 2
    assert rollouts[0].action_id == inverse.action_id
    assert rollouts[0].selected is True
    assert rollouts[0].rank == 1
    assert rollouts[0].metadata["exact_born_recovery"] is True
    assert np.allclose(rollouts[0].candidate_metric_values, 0.0, atol=1e-12)
    assert rollouts[1].action_id == no_op.action_id
    assert rollouts[1].reward == pytest.approx(0.0)


def test_action_and_rollout_artifacts_roundtrip_without_pickle(tmp_path):
    distorted = QuantumCircuit(1)
    distorted.rx(0.2, 0)
    distorted_probabilities = simulate_ideal_statevector(distorted).probabilities
    config = ActionEngineConfig(candidate_magnitudes=(0.2,))
    inverse = candidate(
        (ActionEdit("append_rx", (0,), -0.2),),
        "oracle_inverse",
        config,
    )
    rollout = run_action_rollouts(
        distorted_circuit=distorted,
        clean_target_run_id="clean_run",
        clean_probabilities={"0": 1.0, "1": 0.0},
        distorted_probabilities=distorted_probabilities,
        candidates=[inverse],
        config=config,
    )[0]

    action_path = tmp_path / "action.json"
    circuit_path = tmp_path / "candidate.qpy"
    rollout_path = tmp_path / "rollout.npz"
    save_action_artifact(inverse, config, action_path)
    save_candidate_circuit(rollout.candidate_circuit, circuit_path)
    save_rollout_artifact(rollout, rollout_path)

    loaded_action = load_action_artifact(action_path, config, inverse.content_hash)
    loaded_circuit = load_candidate_circuit(
        circuit_path,
        rollout.metadata["candidate_circuit_hash"],
    )
    loaded_rollout = load_rollout_artifact(
        rollout_path,
        loaded_circuit,
        rollout.content_hash,
    )
    assert loaded_action == inverse
    assert loaded_rollout.content_hash == rollout.content_hash
    with np.load(rollout_path, allow_pickle=False) as payload:
        assert all(payload[name].dtype.kind != "O" for name in payload.files)


def test_rollout_probability_validation_rejects_bad_input():
    circuit = QuantumCircuit(1)
    config = ActionEngineConfig(candidate_magnitudes=(0.2,))
    no_op = candidate((), "no_op", config)
    with pytest.raises((TypeError, ValueError)):
        run_action_rollouts(
            distorted_circuit=circuit,
            clean_target_run_id="clean_run",
            clean_probabilities={"0": 1.0, "1": 0.0},
            distorted_probabilities={"0": True, "1": 0.0},
            candidates=[no_op],
            config=config,
        )


def test_action_artifact_detects_risk_corruption(tmp_path):
    import json

    config = ActionEngineConfig(candidate_magnitudes=(0.2,))
    item = candidate(
        (ActionEdit("append_rx", (0,), -0.2),),
        "oracle_inverse",
        config,
    )
    path = tmp_path / "action.json"
    save_action_artifact(item, config, path)
    payload = json.loads(path.read_text())
    payload["risk_score"] = 0.99
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="risk_score|content_hash"):
        load_action_artifact(path, config, item.content_hash)


def test_rollout_artifact_detects_array_corruption(tmp_path):
    distorted = QuantumCircuit(1)
    distorted.rx(0.2, 0)
    distorted_probabilities = simulate_ideal_statevector(distorted).probabilities
    config = ActionEngineConfig(candidate_magnitudes=(0.2,))
    inverse = candidate(
        (ActionEdit("append_rx", (0,), -0.2),),
        "oracle_inverse",
        config,
    )
    rollout = run_action_rollouts(
        distorted_circuit=distorted,
        clean_target_run_id="clean_run",
        clean_probabilities={"0": 1.0, "1": 0.0},
        distorted_probabilities=distorted_probabilities,
        candidates=[inverse],
        config=config,
    )[0]
    path = tmp_path / "rollout.npz"
    save_rollout_artifact(rollout, path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    arrays["candidate_metric_values"][0] += 0.1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="improvement|hash|mismatch"):
        load_rollout_artifact(path, rollout.candidate_circuit, rollout.content_hash)
