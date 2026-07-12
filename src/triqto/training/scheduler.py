"""Serializable step-based learning-rate schedules."""
from __future__ import annotations

import math
from typing import Any

import torch

from .config import SchedulerConfig


class DeterministicLRScheduler:
    """A minimal schedule whose complete state is explicit and JSON-safe."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
        *,
        total_steps: int,
    ) -> None:
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        if config.warmup_steps >= total_steps and config.name == "warmup_cosine":
            raise ValueError("warmup_steps must be smaller than total_steps")
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.step_index = 0
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        self._apply()

    def _ratio(self, step_index: int) -> float:
        if self.config.name == "constant":
            return 1.0
        if step_index < self.config.warmup_steps:
            return float(step_index + 1) / float(max(self.config.warmup_steps, 1))
        span = max(self.total_steps - self.config.warmup_steps, 1)
        progress = min(max((step_index - self.config.warmup_steps) / span, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        minimum = self.config.minimum_learning_rate_ratio
        return minimum + (1.0 - minimum) * cosine

    def _apply(self) -> None:
        ratio = self._ratio(self.step_index)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * ratio

    def step(self) -> None:
        if self.step_index >= self.total_steps:
            raise RuntimeError("Scheduler stepped beyond configured total_steps")
        self.step_index += 1
        self._apply()

    @property
    def learning_rate(self) -> float:
        values = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(values) != 1:
            raise ValueError("Phase 14 expects one shared learning rate across parameter groups")
        return next(iter(values))

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "base_lrs": list(self.base_lrs),
            "scheduler_name": self.config.name,
            "warmup_steps": self.config.warmup_steps,
            "minimum_learning_rate_ratio": self.config.minimum_learning_rate_ratio,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("scheduler state must be a dictionary")
        expected = {
            "step_index", "total_steps", "base_lrs", "scheduler_name",
            "warmup_steps", "minimum_learning_rate_ratio",
        }
        if set(payload) != expected:
            raise ValueError("scheduler state key mismatch")
        if payload["total_steps"] != self.total_steps or payload["scheduler_name"] != self.config.name:
            raise ValueError("scheduler state does not match configured schedule")
        if payload["warmup_steps"] != self.config.warmup_steps:
            raise ValueError("scheduler warmup state mismatch")
        if float(payload["minimum_learning_rate_ratio"]) != self.config.minimum_learning_rate_ratio:
            raise ValueError("scheduler minimum ratio state mismatch")
        base = tuple(float(value) for value in payload["base_lrs"])
        if base != self.base_lrs:
            raise ValueError("scheduler base learning rates mismatch")
        step = payload["step_index"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > self.total_steps:
            raise ValueError("scheduler step_index is invalid")
        self.step_index = step
        self._apply()


__all__ = ["DeterministicLRScheduler"]
