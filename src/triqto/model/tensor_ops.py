"""Pure-PyTorch ragged operations used by the variable-size Phase 13 model."""
from __future__ import annotations

import torch
from torch import Tensor


def segment_sum(values: Tensor, index: Tensor, segment_count: int) -> Tensor:
    if values.ndim == 1:
        output = values.new_zeros(segment_count)
    else:
        output = values.new_zeros((segment_count, *values.shape[1:]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


def segment_mean(values: Tensor, index: Tensor, segment_count: int) -> Tensor:
    total = segment_sum(values, index, segment_count)
    counts = segment_sum(torch.ones(index.shape[0], dtype=values.dtype, device=values.device), index, segment_count)
    shape = (segment_count,) + (1,) * (values.ndim - 1)
    return total / counts.clamp_min(1).reshape(shape)


def segment_max(values: Tensor, index: Tensor, segment_count: int) -> Tensor:
    if values.ndim == 1:
        output = values.new_full((segment_count,), -torch.inf)
        if values.numel():
            output.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    else:
        output = values.new_full((segment_count, *values.shape[1:]), -torch.inf)
        if values.numel():
            expanded = index.reshape((-1,) + (1,) * (values.ndim - 1)).expand_as(values)
            output.scatter_reduce_(0, expanded, values, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


def segment_softmax(logits: Tensor, index: Tensor, segment_count: int, mask: Tensor | None = None) -> Tensor:
    if logits.ndim != 1 or index.ndim != 1 or logits.shape != index.shape:
        raise ValueError("segment_softmax expects equal one-dimensional logits and index")
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    if mask.dtype != torch.bool or mask.shape != logits.shape:
        raise ValueError("segment_softmax mask must be bool with logits shape")
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -torch.inf))
    maxima = segment_max(masked_logits, index, segment_count)
    shifted = masked_logits - maxima.index_select(0, index)
    exponentials = torch.where(mask, torch.exp(shifted), torch.zeros_like(logits))
    denominators = segment_sum(exponentials, index, segment_count)
    result = exponentials / denominators.index_select(0, index).clamp_min(torch.finfo(logits.dtype).tiny)
    return torch.where(mask, result, torch.zeros_like(result))


def masked_zero(embedding: Tensor, available_mask: Tensor) -> Tensor:
    if embedding.ndim != 2 or available_mask.ndim != 1 or embedding.shape[0] != available_mask.shape[0]:
        raise ValueError("masked_zero expects [B,D] embedding and [B] mask")
    return embedding * available_mask.to(dtype=embedding.dtype).unsqueeze(1)


def canonicalize_global_phase(amplitudes_real_imag: Tensor, batch_index: Tensor, graph_count: int) -> Tensor:
    """Rotate each state by a deterministic reference amplitude's conjugate phase."""
    if amplitudes_real_imag.ndim != 2 or amplitudes_real_imag.shape[1] != 2:
        raise ValueError("amplitudes_real_imag must have shape [A,2]")
    rotated = amplitudes_real_imag.clone()
    for graph_index in range(graph_count):
        positions = torch.nonzero(batch_index == graph_index, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        local = amplitudes_real_imag.index_select(0, positions)
        magnitude_sq = local.square().sum(dim=1)
        reference_local = int(torch.argmax(magnitude_sq))
        reference = local[reference_local]
        norm = reference.square().sum().sqrt().clamp_min(torch.finfo(local.dtype).tiny)
        ref_real = reference[0] / norm
        ref_imag = reference[1] / norm
        real = local[:, 0]
        imag = local[:, 1]
        canonical = torch.stack(
            (real * ref_real + imag * ref_imag, imag * ref_real - real * ref_imag),
            dim=1,
        )
        rotated.index_copy_(0, positions, canonical)
    return rotated


__all__ = [
    "canonicalize_global_phase",
    "masked_zero",
    "segment_max",
    "segment_mean",
    "segment_softmax",
    "segment_sum",
]
