"""Deterministic identifier helpers for TriQTO records.

TriQTO manifests use stable IDs so records produced on different machines can be
joined without relying on local file order.  The helpers in this module canonicalize
small JSON-serializable payloads and derive short SHA-256 identifiers with explicit
prefixes for the record family.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_ID_BYTES = 16


def _normalize_payload(payload: Any) -> Any:
    """Return a JSON-stable representation for deterministic hashing."""
    if is_dataclass(payload):
        return _normalize_payload(asdict(payload))
    if isinstance(payload, Enum):
        return payload.value
    if isinstance(payload, Path):
        return payload.as_posix()
    if isinstance(payload, (datetime, date)):
        return payload.isoformat()
    if isinstance(payload, dict):
        return {str(k): _normalize_payload(payload[k]) for k in sorted(payload)}
    if isinstance(payload, (list, tuple)):
        return [_normalize_payload(item) for item in payload]
    if isinstance(payload, set):
        return sorted(_normalize_payload(item) for item in payload)
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    raise TypeError(f"Unsupported payload type for deterministic ID: {type(payload)!r}")


def canonical_json(payload: Any) -> str:
    """Serialize a payload into canonical JSON for stable hashes."""
    return json.dumps(_normalize_payload(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def make_deterministic_id(prefix: str, payload: Any, *, digest_bytes: int = DEFAULT_ID_BYTES) -> str:
    """Create a deterministic ID using a human-readable prefix and SHA-256 digest."""
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("ID prefix must be non-empty and alphanumeric/underscore only.")
    if digest_bytes <= 0 or digest_bytes > hashlib.sha256().digest_size:
        raise ValueError("digest_bytes must be between 1 and 32 for SHA-256.")
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[: digest_bytes * 2]}"


def make_circuit_id(payload: Any) -> str:
    """Create a stable identifier for a circuit specification or circuit record."""
    return make_deterministic_id("circuit", payload)


def make_run_id(payload: Any) -> str:
    """Create a stable identifier for a simulation, evaluation, or hardware run."""
    return make_deterministic_id("run", payload)


def make_sample_id(payload: Any) -> str:
    """Create a stable identifier for a data-lake sample or training example."""
    return make_deterministic_id("sample", payload)


def make_topology_group_id(payload: Any) -> str:
    """Create a stable identifier for a topology point-cloud/alignment group."""
    return make_deterministic_id("topology", payload)
