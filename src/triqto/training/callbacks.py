"""Small deterministic training-state callbacks without framework magic."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class EarlyStoppingState:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = -1
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 0:
            raise ValueError("patience must be a nonnegative integer")

    def update(self, loss: float, epoch: int) -> tuple[bool, bool]:
        if not math.isfinite(loss):
            raise ValueError("early-stopping loss must be finite")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        should_stop = self.patience > 0 and self.bad_epochs >= self.patience
        return improved, should_stop


__all__ = ["EarlyStoppingState"]
