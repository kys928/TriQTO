"""Deterministic per-component gradient diagnostics for model-ready objectives."""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from triqto.training.config import LossConfig


_COMPONENT_WEIGHTS = (
    ("diagnosis_type", "diagnosis_type_weight"),
    ("diagnosis_strength", "diagnosis_strength_weight"),
    ("diagnosis_affected_qubit", "diagnosis_affected_qubit_weight"),
    ("action_should_act", "action_selection_weight"),
    ("action_rank_distribution", "action_rank_distribution_weight"),
    ("action_reward", "action_reward_weight"),
    ("born_kl", "born_kl_weight"),
    ("born_hellinger", "born_hellinger_weight"),
)


def weighted_objective_components(
    losses: Mapping[str, Tensor],
    config: LossConfig,
) -> dict[str, Tensor]:
    """Return the exact additive objective terms used when uncertainty weighting is off."""
    if not isinstance(config, LossConfig):
        raise TypeError("config must be LossConfig")
    if config.uncertainty_weighting:
        raise ValueError(
            "exact additive gradient audit requires uncertainty_weighting=false"
        )
    required = {
        name for name, _weight in _COMPONENT_WEIGHTS
    } | {"geometry", "topology", "total"}
    missing = required - set(losses)
    if missing:
        raise ValueError(f"loss mapping misses components {sorted(missing)}")

    components: dict[str, Tensor] = {}
    for name, weight_name in _COMPONENT_WEIGHTS:
        value = losses[name]
        if value.ndim != 0:
            raise ValueError(f"loss component {name} must be scalar")
        components[name] = value * float(getattr(config, weight_name))
    # Geometry is already multiplied by geometry_weight inside the objective.
    components["geometry"] = losses["geometry"]
    components["topology"] = losses["topology"]
    reconstructed = sum(components.values(), losses["total"] * 0.0)
    if not torch.allclose(
        reconstructed.detach(),
        losses["total"].detach(),
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError(
            "weighted component sum does not reconstruct total objective"
        )
    components["objective_total"] = losses["total"]
    return components


def _gradient_report(
    model: nn.Module,
    loss: Tensor,
    *,
    retain_graph: bool,
    clip_threshold: float,
) -> dict[str, object]:
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise ValueError("gradient audit loss must be one finite scalar")
    if not math.isfinite(clip_threshold) or clip_threshold <= 0.0:
        raise ValueError("clip_threshold must be finite and positive")

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    gradients = torch.autograd.grad(
        loss,
        [parameter for _name, parameter in named_parameters],
        retain_graph=retain_graph,
        allow_unused=True,
    )

    total_sq = 0.0
    module_sq: dict[str, float] = {}
    nonzero_parameter_count = 0
    maximum_parameter = None
    maximum_parameter_norm = 0.0
    for (name, _parameter), gradient in zip(named_parameters, gradients, strict=True):
        if gradient is None:
            continue
        norm = float(torch.linalg.vector_norm(gradient.detach()).cpu())
        if not math.isfinite(norm):
            raise FloatingPointError(f"non-finite gradient norm for {name}")
        squared = norm * norm
        total_sq += squared
        module = name.split(".", 1)[0]
        module_sq[module] = module_sq.get(module, 0.0) + squared
        if norm > 0.0:
            nonzero_parameter_count += 1
        if norm > maximum_parameter_norm:
            maximum_parameter_norm = norm
            maximum_parameter = name

    total_norm = math.sqrt(total_sq)
    module_norms = {
        name: math.sqrt(value)
        for name, value in sorted(
            module_sq.items(), key=lambda item: item[1], reverse=True
        )
    }
    largest_module = next(iter(module_norms), None)
    return {
        "loss": float(loss.detach().cpu()),
        "gradient_norm": total_norm,
        "gradient_norm_to_clip_ratio": total_norm / clip_threshold,
        "nonzero_parameter_count": nonzero_parameter_count,
        "largest_module": largest_module,
        "largest_module_gradient_norm": (
            None if largest_module is None else module_norms[largest_module]
        ),
        "largest_parameter": maximum_parameter,
        "largest_parameter_gradient_norm": maximum_parameter_norm,
        "module_gradient_norms": module_norms,
    }


def audit_loss_component_gradients(
    model: nn.Module,
    components: Mapping[str, Tensor],
    *,
    clip_threshold: float,
) -> dict[str, dict[str, object]]:
    """Measure gradients for every additive component without mutating ``.grad``."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module")
    if not components:
        raise ValueError("components must not be empty")
    ordered = list(components.items())
    report: dict[str, dict[str, object]] = {}
    for index, (name, loss) in enumerate(ordered):
        if not isinstance(name, str) or not name:
            raise ValueError("component names must be nonblank strings")
        report[name] = _gradient_report(
            model,
            loss,
            retain_graph=index + 1 < len(ordered),
            clip_threshold=clip_threshold,
        )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("gradient audit unexpectedly mutated parameter.grad")
    return report


__all__ = [
    "audit_loss_component_gradients",
    "weighted_objective_components",
]
