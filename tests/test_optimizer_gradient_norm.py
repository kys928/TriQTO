from __future__ import annotations

import math

import torch
from torch import nn

from triqto.training.optimizer import (
    clip_gradient_norm,
    expected_post_clip_gradient_norm,
    finite_gradient_norm,
)


def test_gradient_norm_and_clipping_preserve_global_norm_contract() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    parameters = list(model.parameters())
    for index, parameter in enumerate(parameters, start=1):
        parameter.grad = torch.full_like(parameter, float(index))

    expected_before = math.sqrt(
        sum(float(parameter.grad.double().square().sum()) for parameter in parameters)
    )
    assert math.isclose(
        finite_gradient_norm(parameters), expected_before, rel_tol=1.0e-12, abs_tol=1.0e-12
    )

    maximum = 2.5
    observed_before = clip_gradient_norm(model, maximum)
    observed_after = finite_gradient_norm(model.parameters())
    expected_after = expected_post_clip_gradient_norm(observed_before, maximum)

    assert math.isclose(observed_before, expected_before, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert math.isclose(observed_after, expected_after, rel_tol=1.0e-6, abs_tol=1.0e-6)
    assert observed_after <= maximum + 1.0e-6


def test_empty_gradient_norm_is_zero() -> None:
    model = nn.Linear(2, 1)
    assert finite_gradient_norm(model.parameters()) == 0.0
