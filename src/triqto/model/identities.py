"""Deterministic identities for the untrained Phase 13 architecture."""
from __future__ import annotations

from typing import Any

from triqto.core.ids import make_deterministic_id

from .config import TriQTOModelConfig, model_config_to_dict
from .constants import (
    HEAD_ORDER,
    HEAD_STREAM_POLICY,
    MASK_FUSION_VERSION,
    MODEL_INPUT_CONTRACT_VERSION,
    MODEL_OUTPUT_CONTRACT_VERSION,
    MODEL_SCHEMA_VERSION,
    PHASE_COUPLED_LAYER_VERSION,
    STREAM_ORDER,
)


def model_schema_id() -> str:
    return make_deterministic_id(
        "modelschema",
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "input_contract": MODEL_INPUT_CONTRACT_VERSION,
            "output_contract": MODEL_OUTPUT_CONTRACT_VERSION,
            "phase_layer": PHASE_COUPLED_LAYER_VERSION,
            "fusion": MASK_FUSION_VERSION,
            "stream_order": STREAM_ORDER,
            "head_order": HEAD_ORDER,
            "head_stream_policy": HEAD_STREAM_POLICY,
            "uses_qkv_attention": False,
            "variable_qubit_count": True,
            "head_active_mask": True,
            "inactive_head_outputs_zero": True,
            "masked_dense_rows_must_be_zero": True,
            "global_phase_canonicalization": True,
            "born_prediction_can_consume_born_input": False,
            "topology_prediction_can_copy_topology_input": False,
            "topology_supervised_target_available": False,
            "topology_loss_weight": 0.0,
        },
    )


def scientific_model_config_payload(config: TriQTOModelConfig) -> dict[str, Any]:
    """Return architecture-defining fields, excluding labels and initialization seed."""
    if not isinstance(config, TriQTOModelConfig):
        raise TypeError("config must be TriQTOModelConfig")
    payload = model_config_to_dict(config)
    payload.pop("model_name")
    payload.pop("initialization_seed")
    return payload


def model_architecture_id(config: TriQTOModelConfig) -> str:
    return make_deterministic_id(
        "modelarch",
        {
            "model_schema_id": model_schema_id(),
            "scientific_config": scientific_model_config_payload(config),
        },
    )


def model_config_id(config: TriQTOModelConfig) -> str:
    """Identify the complete constructor config, including label and initialization seed."""
    return make_deterministic_id("modelconfig", model_config_to_dict(config))


__all__ = [
    "model_architecture_id",
    "model_config_id",
    "model_schema_id",
    "scientific_model_config_payload",
]
