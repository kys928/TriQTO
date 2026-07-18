"""Shared deterministic checkpoint primitives for expensive Phase 15.6 work units."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
import uuid
from typing import Any, Literal, TypeVar

import numpy as np

from triqto.graph.utils import strict_json_load, write_strict_json

ResumeMode = Literal["strict", "repair", "off"]
CheckpointRetention = Literal["phase", "campaign", "always"]
T = TypeVar("T")
CHECKPOINT_SCHEMA = "triqto.phase15_6.checkpoint.v2"


def normalize_resume_mode(value: str) -> ResumeMode:
    if value not in {"strict", "repair", "off"}:
        raise ValueError("resume_mode must be one of: strict, repair, off")
    return value  # type: ignore[return-value]


def normalize_checkpoint_retention(value: str) -> CheckpointRetention:
    if value not in {"phase", "campaign", "always"}:
        raise ValueError(
            "checkpoint_retention must be one of: phase, campaign, always"
        )
    return value  # type: ignore[return-value]


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Mapping[str, Any]) -> np.ndarray:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8).copy()


def decode_json_bytes(array: np.ndarray, name: str) -> dict[str, Any]:
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional uint8 array")
    payload = json.loads(array.tobytes().decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must decode to a JSON object")
    return payload


def atomic_write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **dict(arrays))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_artifact(path: str | Path, writer: Callable[[Path], None]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix
    temporary = target.parent / f".{target.stem}.tmp-{uuid.uuid4().hex}{suffix}"
    try:
        writer(temporary)
        if not temporary.is_file():
            raise RuntimeError(f"Checkpoint writer did not create {temporary}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _quarantine(paths: list[Path], root: Path, reason: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = root / "quarantine" / f"{stamp}-{uuid.uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if path.exists():
            shutil.move(str(path), destination / path.name)
    write_strict_json(destination / "reason.json", {"reason": reason})


def prepare_checkpoint_root(root: str | Path, mode: str) -> Path:
    resolved_mode = normalize_resume_mode(mode)
    target = Path(root)
    if resolved_mode == "off" and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_checkpoint_artifact(
    *,
    root: str | Path,
    phase: str,
    unit_id: str,
    stage: str,
    artifact_path: str | Path,
    marker_path: str | Path,
    identity: Mapping[str, Any],
    resume_mode: str,
    loader: Callable[[Path, Mapping[str, Any]], T],
) -> T | None:
    mode = normalize_resume_mode(resume_mode)
    artifact = Path(artifact_path)
    marker = Path(marker_path)
    if mode == "off":
        artifact.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        return None
    if not artifact.exists() and not marker.exists():
        return None
    try:
        if not artifact.is_file() or not marker.is_file():
            raise ValueError("checkpoint artifact/marker pair is incomplete")
        payload = strict_json_load(marker)
        if not isinstance(payload, dict):
            raise TypeError("checkpoint marker must be a JSON object")
        expected_identity_hash = canonical_json_hash(identity)
        required = {
            "schema": CHECKPOINT_SCHEMA,
            "complete": True,
            "phase": phase,
            "unit_id": unit_id,
            "stage": stage,
            "identity_hash": expected_identity_hash,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"checkpoint marker {key} mismatch: "
                    f"expected {expected!r}, got {payload.get(key)!r}"
                )
        expected_sha = payload.get("artifact_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError("checkpoint marker artifact_sha256 is invalid")
        actual_sha = sha256_file(artifact)
        if actual_sha != expected_sha:
            raise ValueError("checkpoint artifact SHA-256 mismatch")
        return loader(artifact, payload)
    except Exception as exc:
        if mode == "strict":
            raise RuntimeError(
                f"Invalid {phase} checkpoint for unit={unit_id} stage={stage}: {exc}"
            ) from exc
        _quarantine(
            [artifact, marker],
            Path(root),
            f"unit={unit_id} stage={stage}: {type(exc).__name__}: {exc}",
        )
        return None


def commit_checkpoint_artifact(
    *,
    phase: str,
    unit_id: str,
    stage: str,
    artifact_path: str | Path,
    marker_path: str | Path,
    identity: Mapping[str, Any],
    writer: Callable[[Path], None],
    validator: Callable[[Path], T],
    marker_metadata: Mapping[str, Any] | None = None,
) -> T:
    artifact = Path(artifact_path)
    marker = Path(marker_path)
    marker.unlink(missing_ok=True)
    atomic_write_artifact(artifact, writer)
    loaded = validator(artifact)
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "complete": True,
        "phase": phase,
        "unit_id": unit_id,
        "stage": stage,
        "identity_hash": canonical_json_hash(identity),
        "artifact_sha256": sha256_file(artifact),
    }
    if marker_metadata:
        payload["metadata"] = dict(marker_metadata)
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_strict_json(marker, payload)
    return loaded


def record_checkpoint_failure(
    *,
    root: str | Path,
    phase: str,
    unit_id: str,
    stage: str,
    error: BaseException,
    context: Mapping[str, Any] | None = None,
) -> Path:
    target = Path(root) / "failed_units" / f"{unit_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "triqto.phase15_6.checkpoint_failure.v1",
        "phase": phase,
        "unit_id": unit_id,
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
        "context": dict(context or {}),
    }
    write_strict_json(target, payload)
    return target


def clear_checkpoint_failure(root: str | Path, unit_id: str) -> None:
    (Path(root) / "failed_units" / f"{unit_id}.json").unlink(missing_ok=True)


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CheckpointRetention",
    "ResumeMode",
    "atomic_write_artifact",
    "atomic_write_npz",
    "canonical_json_hash",
    "clear_checkpoint_failure",
    "commit_checkpoint_artifact",
    "decode_json_bytes",
    "json_bytes",
    "load_checkpoint_artifact",
    "normalize_checkpoint_retention",
    "normalize_resume_mode",
    "prepare_checkpoint_root",
    "record_checkpoint_failure",
    "sha256_file",
]
