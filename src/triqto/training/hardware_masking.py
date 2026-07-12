"""Explicit hardware-mode invariants reused by Phase 14 adapters."""
from __future__ import annotations

from triqto.model import TriQTOBatch, TriQTOModelConfig


def validate_hardware_masked_batch(batch: TriQTOBatch, config: TriQTOModelConfig) -> None:
    batch.validate(config)
    hardware = batch.resolved_hardware_mask()
    if batch.hilbert is not None and bool((hardware & batch.hilbert.available_mask).any()):
        raise ValueError("Hardware-mode rows cannot contain Hilbert values")
    dependency = batch.resolved_topology_hilbert_dependency()
    if batch.topology is not None and bool((hardware & dependency & batch.topology.available_mask).any()):
        raise ValueError("Hardware-mode rows cannot contain Hilbert-dependent topology")


__all__ = ["validate_hardware_masked_batch"]
