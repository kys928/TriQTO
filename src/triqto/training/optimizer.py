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
    """Return an exact finite global norm for audit/debug callers.

    This helper intentionally performs a full scan. The hot training path uses
    ``clip_gradient_norm`` instead, because ``torch.nn.utils.clip_grad_norm_`` already
    computes the same pre-clip norm while applying clipping.
    """
    squares = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite gradient detected")
        squares += float(gradient.double().square().sum().cpu())
        found = True
    return math.sqrt(squares) if found else 0.0


def clip_gradient_norm(model: nn.Module, maximum: float) -> float:
    """Clip once and return the pre-clip global norm with one device sync.

    The previous implementation scanned every parameter on the GPU, copied each
    partial norm to the CPU, and then called ``clip_grad_norm_``, which recomputed the
    same norm. On CUDA this caused many avoidable synchronizations per optimizer step.
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
