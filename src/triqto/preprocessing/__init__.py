"""Deterministic offline preprocessing for completed TriQTO Phase 7 datasets.

The preprocessing package is deliberately external to the learned model.  It
validates and canonicalizes immutable Phase 7 data, constructs auditable
identity/leakage relations and challenge splits, and publishes task-specific
training views under a fresh output root.
"""
from __future__ import annotations

from .config import (
    PreprocessingConfig,
    load_preprocessing_config,
    preprocessing_config_from_dict,
    preprocessing_config_to_dict,
    save_preprocessing_config,
)

__all__ = [
    "PreprocessingConfig",
    "load_preprocessing_config",
    "preprocessing_config_from_dict",
    "preprocessing_config_to_dict",
    "save_preprocessing_config",
]
