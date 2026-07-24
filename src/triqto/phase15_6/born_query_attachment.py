"""Immutable attachment of deployable Born query coordinates to model-ready data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np

BORN_QUERY_ATTACHMENT_SCHEMA = "triqto.phase12.born_query_attachment.v1"
BORN_QUERY_ATTACHMENT_VERSION = "triqto.phase12.born_query_attachment_runner.v1"
_QUERY_FIELD = "x_born_query_outcome_bitstrings"
_TARGET_SUPPORT_FIELD = "y_born_target_outcome_bitstrings"
_TARGET_PROBABILITY_FIELD = "y_born_target_probabilities"


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _safe_artifact(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"artifact_ref must be relative: {reference!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact_ref escapes source root: {reference!r}") from exc
    return resolved


def _validate_query_support(value: np.ndarray, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{field} must be a string array")
    flattened = array.reshape(-1)
    if flattened.size == 0:
        raise ValueError(f"{field} must not be empty")
    names = [str(item) for item in flattened.tolist()]
    widths = {len(item) for item in names}
    if len(widths) != 1 or next(iter(widths)) <= 0:
        raise ValueError(f"{field} bitstrings must have one positive width")
    if any(any(character not in {"0", "1"} for character in item) for item in names):
        raise ValueError(f"{field} contains a non-binary outcome")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicate outcomes")
    return flattened.copy()


def _rewrite_artifact(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    support_value = arrays.get(_TARGET_SUPPORT_FIELD)
    probability_value = arrays.get(_TARGET_PROBABILITY_FIELD)
    if support_value is None and probability_value is None:
        return False
    if support_value is None or probability_value is None:
        raise ValueError(f"{path} has incomplete Born target support")
    support = _validate_query_support(support_value, field=_TARGET_SUPPORT_FIELD)
    probabilities = np.asarray(probability_value).reshape(-1)
    if probabilities.size != support.size:
        raise ValueError(f"{path} Born target support/probability widths differ")
    existing = arrays.get(_QUERY_FIELD)
    if existing is not None:
        query = _validate_query_support(existing, field=_QUERY_FIELD)
        if not np.array_equal(query, support):
            raise ValueError(f"{path} existing query coordinates differ from target support")
        return False
    arrays[_QUERY_FIELD] = support
    temporary = path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def attach_born_query_coordinates(
    *,
    source_root: str | Path,
    output_parent: str | Path,
) -> dict[str, Any]:
    """Create a separate immutable product with explicit x_* Born query support."""
    source = Path(source_root).expanduser().resolve()
    output_base = Path(output_parent).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output_base == source or output_base.is_relative_to(source):
        raise ValueError("Born-query output must live outside the source root")

    source_manifest = source / "manifests" / "processed_item_manifest.parquet"
    source_completion = source / "preprocessed_complete.json"
    source_contract = source / "manifests" / "model_input_contract.json"
    for path in (source_manifest, source_completion, source_contract):
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = _read_json(source_completion)
    if completion.get("complete") is not True:
        raise ValueError("source preprocessing product is incomplete")
    if float(completion.get("lambda_top", 0.0)) != 0.0:
        raise ValueError("lambda_top must remain exactly 0.0")

    source_manifest_hash = _sha256_file(source_manifest)
    expected_hash = str(completion.get("processed_item_manifest_sha256") or "")
    if expected_hash and expected_hash != source_manifest_hash:
        raise ValueError("source processed manifest hash mismatch")
    identity = {
        "schema": BORN_QUERY_ATTACHMENT_SCHEMA,
        "version": BORN_QUERY_ATTACHMENT_VERSION,
        "source_manifest_sha256": source_manifest_hash,
    }
    run_token = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    final_root = output_base / f"phase12_queries_{run_token}"
    final_marker = final_root / "born_query_attachment_complete.json"
    if final_marker.is_file():
        marker = _read_json(final_marker)
        if marker.get("complete") is True:
            return {"status": "already_complete", "output_root": str(final_root), **marker}
    if final_root.exists():
        raise FileExistsError(f"incomplete Born-query product exists: {final_root}")

    output_base.mkdir(parents=True, exist_ok=True)
    staging = output_base / f".{final_root.name}.staging-{uuid.uuid4().hex}"
    shutil.copytree(source, staging, copy_function=_copy_link_or_copy)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        manifest_path = staging / "manifests" / "processed_item_manifest.parquet"
        table = pq.read_table(manifest_path)
        rows = table.to_pylist()
        if not rows:
            raise ValueError("source manifest is empty")
        attached_rows = 0
        born_target_rows = 0
        by_task: dict[str, int] = {}
        for row in rows:
            artifact = _safe_artifact(staging, str(row["artifact_ref"]))
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            with np.load(artifact, allow_pickle=False) as archive:
                has_target = _TARGET_SUPPORT_FIELD in archive.files
            if has_target:
                born_target_rows += 1
                by_task[str(row["task"])] = by_task.get(str(row["task"]), 0) + 1
            if _rewrite_artifact(artifact):
                attached_rows += 1
            row["content_hash"] = _sha256_file(artifact)

        temporary_manifest = manifest_path.with_name(
            f".{manifest_path.name}.tmp-{uuid.uuid4().hex}"
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary_manifest)
        os.replace(temporary_manifest, manifest_path)
        output_manifest_hash = _sha256_file(manifest_path)

        contract_path = staging / "manifests" / "model_input_contract.json"
        contract = _read_json(contract_path)
        contract["born_query_coordinates"] = {
            "field": _QUERY_FIELD,
            "role": "deployable output-coordinate query",
            "source": "target support names only",
            "target_probabilities_copied": False,
            "required_for_tasks": [
                "born_prediction",
                "joint_multitask",
                "hardware_masked",
            ],
        }
        _atomic_json(contract_path, contract)

        for marker_name in (
            "preprocessed_complete.json",
            "topology_attachment_complete.json",
        ):
            marker_path = staging / marker_name
            if not marker_path.is_file():
                continue
            marker = _read_json(marker_path)
            marker["processed_item_manifest_sha256"] = output_manifest_hash
            marker["born_query_attachment"] = {
                "schema": BORN_QUERY_ATTACHMENT_SCHEMA,
                "query_field": _QUERY_FIELD,
                "born_target_rows": born_target_rows,
                "rewritten_rows": attached_rows,
            }
            _atomic_json(marker_path, marker)

        source_manifest_hash_after = _sha256_file(source_manifest)
        if source_manifest_hash_after != source_manifest_hash:
            raise RuntimeError("source manifest changed during Born-query attachment")
        marker = {
            "schema": BORN_QUERY_ATTACHMENT_SCHEMA,
            "version": BORN_QUERY_ATTACHMENT_VERSION,
            "complete": True,
            "source_root": str(source),
            "source_manifest_sha256": source_manifest_hash,
            "processed_item_manifest_sha256": output_manifest_hash,
            "published_model_items": len(rows),
            "born_target_rows": born_target_rows,
            "rewritten_rows": attached_rows,
            "born_target_rows_by_task": dict(sorted(by_task.items())),
            "query_field": _QUERY_FIELD,
            "target_probabilities_copied": False,
            "source_mutated": False,
            "lambda_top": 0.0,
        }
        _atomic_json(staging / "born_query_attachment_complete.json", marker)
        os.replace(staging, final_root)
        return {"status": "complete", "output_root": str(final_root), **marker}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "BORN_QUERY_ATTACHMENT_SCHEMA",
    "BORN_QUERY_ATTACHMENT_VERSION",
    "attach_born_query_coordinates",
]
