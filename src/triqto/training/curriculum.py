"""Deterministic Phase 14 curriculum expansion."""
from __future__ import annotations

from dataclasses import dataclass

from .config import TrainingConfig


@dataclass(frozen=True, slots=True)
class EpochPlan:
    epoch: int
    stage_index: int
    stage_name: str
    stage_epoch: int
    tasks: tuple[str, ...]


def build_epoch_plan(config: TrainingConfig) -> tuple[EpochPlan, ...]:
    plans: list[EpochPlan] = []
    epoch = 0
    for stage_index, stage in enumerate(config.stages):
        for stage_epoch in range(stage.epochs):
            plans.append(
                EpochPlan(
                    epoch=epoch,
                    stage_index=stage_index,
                    stage_name=stage.name,
                    stage_epoch=stage_epoch,
                    tasks=stage.tasks,
                )
            )
            epoch += 1
    if len(plans) != config.total_epochs:
        raise RuntimeError("Curriculum expansion count mismatch")
    return tuple(plans)


__all__ = ["EpochPlan", "build_epoch_plan"]
