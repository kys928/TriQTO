"""Architecture-state inspection only; training checkpoints remain Phase 14."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .config import model_config_to_dict
from .identities import model_architecture_id, model_config_id, model_schema_id


def state_dict_signature(module: nn.Module) -> str:
    """Hash parameter/buffer names, shapes, dtypes, and exact initialized bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, Tensor):
            raise TypeError("state_dict values must be tensors")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def architecture_manifest(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("model must expose a Phase 13 config")
    architecture_id = model_architecture_id(config)
    state = model.state_dict()
    return {
        "model_schema_id": model_schema_id(),
        "model_architecture_id": architecture_id,
        "model_config_id": model_config_id(config),
        "config": model_config_to_dict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "state_dict_shapes": {
            name: list(tensor.shape) for name, tensor in sorted(state.items())
        },
        "initialized_state_signature": state_dict_signature(model),
        "trained": False,
        "optimizer_state_present": False,
        "training_checkpoint": False,
        "topology_loss_weight": 0.0,
    }


def load_architecture_state_dict_strict(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
) -> None:
    """Strictly restore architecture weights; optimizer/training state is unsupported."""
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    expected = set(model.state_dict())
    actual = set(state_dict)
    if actual != expected:
        raise ValueError(
            "Architecture state keys mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    model.load_state_dict(dict(state_dict), strict=True)


__all__ = [
    "architecture_manifest",
    "load_architecture_state_dict_strict",
    "state_dict_signature",
]
