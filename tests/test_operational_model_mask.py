from __future__ import annotations

import torch
from qiskit import QuantumCircuit

from triqto.actions.operational import basis_probe_action
from triqto.actions.operational_adapter import build_operational_action_tensor_batch
from triqto.model import TriQTOModelConfig
from triqto.model.heads.action_ranking_head import ActionRankingHead


def test_unavailable_operational_action_cannot_receive_selection_probability() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    available = basis_probe_action(2, ("X", "Y"), circuit=circuit, shots=16, seed=7)
    unavailable = basis_probe_action(2, ("X",), circuit=circuit, shots=16, seed=7)
    operational = build_operational_action_tensor_batch((available, unavailable))
    config = TriQTOModelConfig(
        hidden_dim=16,
        graph_message_passing_layers=1,
        residual_mlp_layers=1,
        topology_input_dim=4,
        hilbert_deformation_dim=4,
        topology_prediction_dim=4,
        dropout=0.0,
        initialization_seed=7,
    )
    head = ActionRankingHead(config).eval()
    output = head(
        torch.zeros(1, config.hidden_dim),
        operational.model_candidates,
        torch.tensor([True]),
    )
    unavailable_mask = ~operational.model_candidates.candidate_available_mask
    assert unavailable_mask.sum().item() == 1
    assert torch.all(output.candidate_probabilities[unavailable_mask] == 0)
    assert torch.all(output.candidate_scores[unavailable_mask] == 0)
    assert torch.all(output.predicted_rewards[unavailable_mask] == 0)
    assert torch.allclose(
        output.candidate_probabilities[~unavailable_mask].sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )
