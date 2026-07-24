from __future__ import annotations

import pytest
import torch
from torch import nn

from triqto.training import LossConfig
from triqto.training.model_ready.gradient_audit import (
    audit_loss_component_gradients,
    weighted_objective_components,
)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(value)).squeeze()


def _losses(value: torch.Tensor, config: LossConfig) -> dict[str, torch.Tensor]:
    zero = value * 0.0
    diagnosis_type = value.square()
    diagnosis_strength = (value - 1.0).square()
    diagnosis_affected = zero
    action_should_act = zero
    action_rank = zero
    action_reward = zero
    born_kl = zero
    born_hellinger = zero
    geometry = zero
    topology = zero
    total = (
        config.diagnosis_type_weight * diagnosis_type
        + config.diagnosis_strength_weight * diagnosis_strength
    )
    return {
        "diagnosis_type": diagnosis_type,
        "diagnosis_strength": diagnosis_strength,
        "diagnosis_affected_qubit": diagnosis_affected,
        "action_should_act": action_should_act,
        "action_rank_distribution": action_rank,
        "action_reward": action_reward,
        "born_kl": born_kl,
        "born_hellinger": born_hellinger,
        "geometry": geometry,
        "topology": topology,
        "total": total,
    }


def test_gradient_audit_reconstructs_weighted_objective_without_mutating_grad() -> None:
    model = _ToyModel()
    config = LossConfig(
        diagnosis_type_weight=2.0,
        diagnosis_strength_weight=0.25,
        uncertainty_weighting=False,
        topology_weight=0.0,
    )
    value = model(torch.tensor([[1.0, -2.0]], dtype=torch.float32))
    components = weighted_objective_components(_losses(value, config), config)

    assert torch.allclose(
        sum(
            (component for name, component in components.items() if name != "objective_total"),
            components["objective_total"] * 0.0,
        ),
        components["objective_total"],
    )
    report = audit_loss_component_gradients(
        model,
        components,
        clip_threshold=1.0,
    )
    assert report["objective_total"]["gradient_norm"] > 0.0
    assert report["diagnosis_type"]["largest_module"] in {"encoder", "head"}
    assert report["topology"]["gradient_norm"] == 0.0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_exact_component_audit_rejects_uncertainty_weighting() -> None:
    model = _ToyModel()
    config = LossConfig(uncertainty_weighting=True, topology_weight=0.0)
    value = model(torch.ones((1, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="uncertainty_weighting=false"):
        weighted_objective_components(_losses(value, config), config)
