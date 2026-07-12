"""Strict configuration for the Phase 13 TriQTO neural architecture."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from triqto.graph.constants import EDGE_FEATURE_NAMES, GATE_FEATURE_NAMES, NODE_FEATURE_NAMES

from .constants import (
    ACTION_EDIT_TYPES,
    DISTORTION_LABELS,
    MODEL_SCHEMA_VERSION,
    TOPOLOGY_LOSS_WEIGHT,
    UNCERTAINTY_TARGETS,
)


def _int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _float(value: Any, name: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and not bool")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum or (maximum is not None and numeric > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be finite and {bound}")
    return numeric


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be exactly bool")
    return value


def _strings(value: Any, name: str, expected: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    normalized = tuple(value)
    if normalized != expected:
        raise ValueError(f"{name} is versioned and must equal {list(expected)}")
    return normalized


@dataclass(frozen=True, slots=True)
class TriQTOModelConfig:
    """Architecture choices only; no optimizer, epoch, or training settings."""

    schema_version: str = MODEL_SCHEMA_VERSION
    model_name: str = "triqto_base"
    hidden_dim: int = 128
    graph_message_passing_layers: int = 4
    residual_mlp_layers: int = 2
    node_input_dim: int = len(NODE_FEATURE_NAMES)
    edge_input_dim: int = len(EDGE_FEATURE_NAMES)
    gate_input_dim: int = len(GATE_FEATURE_NAMES)
    backend_input_dim: int = 16
    topology_input_dim: int = 64
    action_candidate_feature_dim: int = 5
    action_edit_type_count: int = len(ACTION_EDIT_TYPES)
    hilbert_deformation_dim: int = 32
    topology_prediction_dim: int = 32
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initialization_seed: int = 130013
    use_hilbert: bool = True
    use_backend: bool = True
    use_topology: bool = True
    distortion_labels: tuple[str, ...] = DISTORTION_LABELS
    uncertainty_targets: tuple[str, ...] = UNCERTAINTY_TARGETS
    topology_loss_weight: float = TOPOLOGY_LOSS_WEIGHT

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or self.schema_version.strip() != MODEL_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {MODEL_SCHEMA_VERSION!r}")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be nonblank text")
        hidden = _int(self.hidden_dim, "hidden_dim", minimum=8)
        if hidden % 2:
            raise ValueError("hidden_dim must be even for phase quadrature channels")
        for name in (
            "graph_message_passing_layers",
            "residual_mlp_layers",
            "node_input_dim",
            "edge_input_dim",
            "gate_input_dim",
            "backend_input_dim",
            "topology_input_dim",
            "action_candidate_feature_dim",
            "action_edit_type_count",
            "hilbert_deformation_dim",
            "topology_prediction_dim",
        ):
            _int(getattr(self, name), name)
        _int(self.initialization_seed, "initialization_seed", minimum=0)
        dropout = _float(self.dropout, "dropout", minimum=0.0, maximum=0.8)
        eps = _float(self.layer_norm_eps, "layer_norm_eps", minimum=1e-12, maximum=1e-2)
        use_hilbert = _bool(self.use_hilbert, "use_hilbert")
        use_backend = _bool(self.use_backend, "use_backend")
        use_topology = _bool(self.use_topology, "use_topology")
        labels = _strings(self.distortion_labels, "distortion_labels", DISTORTION_LABELS)
        uncertainty = _strings(self.uncertainty_targets, "uncertainty_targets", UNCERTAINTY_TARGETS)
        topology_weight = _float(self.topology_loss_weight, "topology_loss_weight", minimum=0.0)
        if topology_weight != 0.0:
            raise ValueError("Phase 13 topology_loss_weight must remain exactly 0.0")
        object.__setattr__(self, "schema_version", MODEL_SCHEMA_VERSION)
        object.__setattr__(self, "model_name", self.model_name.strip())
        object.__setattr__(self, "hidden_dim", hidden)
        object.__setattr__(self, "dropout", dropout)
        object.__setattr__(self, "layer_norm_eps", eps)
        object.__setattr__(self, "use_hilbert", use_hilbert)
        object.__setattr__(self, "use_backend", use_backend)
        object.__setattr__(self, "use_topology", use_topology)
        object.__setattr__(self, "distortion_labels", labels)
        object.__setattr__(self, "uncertainty_targets", uncertainty)
        object.__setattr__(self, "topology_loss_weight", 0.0)
        json.dumps(model_config_to_dict(self), sort_keys=True, allow_nan=False)


def model_config_to_dict(config: TriQTOModelConfig) -> dict[str, Any]:
    if not isinstance(config, TriQTOModelConfig):
        raise TypeError("config must be TriQTOModelConfig")
    payload = asdict(config)
    payload["distortion_labels"] = list(config.distortion_labels)
    payload["uncertainty_targets"] = list(config.uncertainty_targets)
    return payload


def model_config_from_dict(payload: Mapping[str, Any]) -> TriQTOModelConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("model config payload must be a mapping")
    allowed = set(TriQTOModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"Unknown model config fields: {sorted(extra)}")
    return TriQTOModelConfig(**dict(payload))


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid non-finite model config constant: {value}")


def load_model_config(path: str | Path) -> TriQTOModelConfig:
    target = Path(path)
    text = target.read_text()
    payload = yaml.safe_load(text) if target.suffix.lower() in {".yaml", ".yml"} else json.loads(text, parse_constant=_reject_constant)
    if not isinstance(payload, Mapping):
        raise TypeError("model config document must contain a mapping")
    return model_config_from_dict(payload)


def save_model_config(config: TriQTOModelConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model_config_to_dict(config), sort_keys=True, indent=2, allow_nan=False) + "\n")
    return target


__all__ = [
    "TriQTOModelConfig",
    "load_model_config",
    "model_config_from_dict",
    "model_config_to_dict",
    "save_model_config",
]
