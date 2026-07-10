"""Deterministic Phase 7 child seed hierarchy."""
from __future__ import annotations
import hashlib
from typing import Any
from triqto.core.ids import canonical_json

def derive_child_seed(base_seed: int, namespace: str, payload: Any) -> int:
    if not namespace: raise ValueError("namespace must be non-empty")
    digest = hashlib.sha256(canonical_json({"base_seed": int(base_seed), "namespace": namespace, "payload": payload}).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
