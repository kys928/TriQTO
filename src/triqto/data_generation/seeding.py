"""Deterministic Phase 7 child seed hierarchy."""
from __future__ import annotations

import hashlib
from typing import Any

from triqto.core.ids import canonical_json
from .specs import _require_int, _require_nonblank


def derive_child_seed(base_seed: int, namespace: str, payload: Any) -> int:
    """Derive a stable nonnegative 31-bit child seed without global random state."""
    checked_seed = _require_int(base_seed, "base_seed")
    checked_namespace = _require_nonblank(namespace, "namespace")
    digest = hashlib.sha256(
        canonical_json(
            {"base_seed": checked_seed, "namespace": checked_namespace, "payload": payload}
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
