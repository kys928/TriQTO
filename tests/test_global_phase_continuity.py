from __future__ import annotations

import torch

from triqto.model.tensor_ops import canonicalize_global_phase


def test_canonicalization_has_no_argmax_tie_jump() -> None:
    first = torch.tensor([[1.0, 0.0], [0.99, 0.01]], requires_grad=True)
    second = torch.tensor([[0.99, 0.01], [1.0, 0.0]], requires_grad=True)
    idx = torch.tensor([0, 0], dtype=torch.long)
    out1 = canonicalize_global_phase(first, idx, 1)
    out2 = canonicalize_global_phase(second, idx, 1)
    assert torch.linalg.norm(out1.flip(0) - out2) < 0.05
    out1.sum().backward()
    assert first.grad is not None and torch.isfinite(first.grad).all()


def test_canonicalization_handles_near_zero_state_deterministically() -> None:
    amps = torch.zeros((2, 2), dtype=torch.float32)
    out = canonicalize_global_phase(amps, torch.tensor([0, 0]), 1)
    assert torch.equal(out, amps)
