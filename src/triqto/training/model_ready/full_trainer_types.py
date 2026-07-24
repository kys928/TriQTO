"""Public result contract for vectorized model-ready full training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FullTrainingResult:
    status: str
    output_root: Path
    summary: dict[str, Any]


__all__ = ["FullTrainingResult"]
