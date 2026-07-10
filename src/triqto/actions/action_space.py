"""Public bounded action-space contracts for Phase 9."""
from __future__ import annotations

from .constants import SUPPORTED_EDIT_TYPES
from .models import ActionCandidate, ActionEdit


def supported_edit_types() -> tuple[str, ...]:
    """Return the fixed ordered Phase 9 v1 primitive edit vocabulary."""
    return SUPPORTED_EDIT_TYPES


__all__ = ["ActionCandidate", "ActionEdit", "supported_edit_types"]
