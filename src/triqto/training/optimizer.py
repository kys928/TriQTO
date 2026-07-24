"""Deterministic optimizer construction and gradient utilities."""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn

from .config import OptimizerConfig


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Cannot build an optimizer for a model with no trainable parameters")
    if config.name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
    if config.name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    raise ValueError(f"Unsupported optimizer {config.name!r}")


def finite_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Return an exact finite global norm with one host synchronization.

    Individual parameter norms stay on their original device and are reduced there.
    Only the final scalar is transferred to the CPU. This preserves the float64 audit
    calculation without synchronizing CUDA once per parameter.
    """
    gradients = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    device = gradients[0].device
    if any(gradient.device != device for gradient in gradients):
        raise ValueError("gradient norm requires all gradients on one device")
    squared = torch.stack(
        [gradient.double().square().sum() for gradient in gradients]
    ).sum()
    squared_value = float(squared.detach().cpu())
    if not math.isfinite(squared_value):
        raise FloatingPointError("Non-finite gradient detected")
    return math.sqrt(squared_value)


def clip_gradient_norm(model: nn.Module, maximum: float) -> float:
    """Clip once and return the pre-clip global norm with one device sync.

    ``torch.nn.utils.clip_grad_norm_`` already computes the pre-clip norm. Reusing its
    return value avoids a redundant full gradient scan before clipping.
    """
    if maximum <= 0 or not math.isfinite(maximum):
        raise ValueError("maximum gradient norm must be finite and positive")
    total = torch.nn.utils.clip_grad_norm_(
        model.parameters(), maximum, error_if_nonfinite=True
    )
    before = float(total.detach().cpu()) if isinstance(total, Tensor) else float(total)
    if not math.isfinite(before):
        raise FloatingPointError("Non-finite gradient detected")
    return before


def expected_post_clip_gradient_norm(before: float, maximum: float) -> float:
    """Return the norm implied by PyTorch global-norm clipping without rescanning."""
    if not math.isfinite(before) or before < 0.0:
        raise ValueError("before must be finite and nonnegative")
    if maximum <= 0 or not math.isfinite(maximum):
        raise ValueError("maximum gradient norm must be finite and positive")
    coefficient = min(1.0, maximum / (before + 1.0e-6))
    return before * coefficient


__all__ = [
    "build_optimizer",
    "clip_gradient_norm",
    "expected_post_clip_gradient_norm",
    "finite_gradient_norm",
]
