"""Immutable content-addressed operational-action artifacts."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import shutil
import uuid

import numpy as np
from triqto.core.ids import canonical_json, make_deterministic_id
from .operational import OPERATIONAL_ACTION_SCHEMA, OperationalActionResult, operational_action_content_hash, operational_action_payload

OPERATIONAL_ACTION_DATASET_SCHEMA = "triqto.operational_action_dataset.v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_operational_action_result(result: OperationalActionResult) -> None:
    if not isinstance(result, OperationalActionResult):
        raise TypeError("result must be OperationalActionResult")
    if result.action_type not in {"basis_probe", "layout_selection", "routing_transpilation", "depth_reduction"}:
        raise ValueError("unsupported operational action type")
    if result.status not in {"accepted", "rejected", "no_op"}:
        raise ValueError("unsupported operational action status")
    if result.available != result.availability_mask or result.available != (result.status == "accepted"):
        raise ValueError("operational availability/status mismatch")
    if result.status != "accepted" and not result.rejection_reason:
        raise ValueError("rejected/no-op action requires a reason")
    if result.physical_hardware and result.required_evidence_tier != "physical_hardware":
        raise ValueError("physical action requires physical evidence tier")
    if result.privileged_information:
        raise ValueError("operational actions cannot use privileged inverse information")
    if result.content_hash != operational_action_content_hash(result):
        raise ValueError("operational action content hash mismatch")


def save_operational_action_result(path: str | Path, result: OperationalActionResult) -> Path:
    validate_operational_action_result(result)
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"operational action artifact exists: {target}")
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        _write_json(temporary, operational_action_payload(result, include_content_hash=True))
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_operational_action_result(path: str | Path) -> OperationalActionResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.pop("schema", None) != OPERATIONAL_ACTION_SCHEMA:
        raise ValueError("unsupported operational action artifact schema")
    expected = {field.name for field in fields(OperationalActionResult)}
    if set(payload) != expected:
        raise ValueError("operational action artifact key mismatch")
    result = OperationalActionResult(**payload)
    validate_operational_action_result(result)
    return result


def write_operational_action_dataset(output_root: str | Path, results: Iterable[OperationalActionResult], *, source_dataset_id: str, evidence_tier: str) -> dict[str, Any]:
    if not source_dataset_id.strip() or not evidence_tier.strip():
        raise ValueError("source_dataset_id and evidence_tier must be nonblank")
    ordered = sorted(results, key=lambda value: value.action_id)
    if not ordered or len({value.action_id for value in ordered}) != len(ordered):
        raise ValueError("operational dataset requires unique nonempty actions")
    for result in ordered:
        validate_operational_action_result(result)
        if evidence_tier == "mixed_offline":
            if result.physical_hardware or result.required_evidence_tier == "physical_hardware":
                raise ValueError("mixed_offline cannot contain hardware actions")
        elif result.required_evidence_tier != evidence_tier:
            raise ValueError("dataset evidence tier mismatch")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"operational output root exists: {root}")
    dataset_id = make_deterministic_id("operational_action_dataset", {
        "schema": OPERATIONAL_ACTION_DATASET_SCHEMA,
        "source_dataset_id": source_dataset_id,
        "evidence_tier": evidence_tier,
        "action_ids": [value.action_id for value in ordered],
        "content_hashes": [value.content_hash for value in ordered],
    })
    references = [f"actions/{value.action_id}.json" for value in ordered]
    family_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for result in ordered:
        family_counts[result.action_family] = family_counts.get(result.action_family, 0) + 1
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    summary = {
        "schema": OPERATIONAL_ACTION_DATASET_SCHEMA,
        "operational_action_dataset_id": dataset_id,
        "source_dataset_id": source_dataset_id,
        "evidence_tier": evidence_tier,
        "action_evidence_tiers": sorted({value.required_evidence_tier for value in ordered}),
        "physical_hardware": False,
        "action_count": len(ordered),
        "family_counts": dict(sorted(family_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "claim_scope": "operational engineering evidence; not logical correction success",
        "topology_loss_weight": 0.0,
    }
    managed = ["operational_action_summary.json", "phase12_compatible_action_arrays.npz", *references, "operational_action_complete.json"]
    manifest = {
        "schema": OPERATIONAL_ACTION_DATASET_SCHEMA,
        "operational_action_dataset_id": dataset_id,
        "source_dataset_id": source_dataset_id,
        "evidence_tier": evidence_tier,
        "action_evidence_tiers": summary["action_evidence_tiers"],
        "action_ids": [value.action_id for value in ordered],
        "action_content_hashes": [value.content_hash for value in ordered],
        "managed_files": managed,
    }
    manifest["manifest_content_hash"] = make_deterministic_id("operational_action_manifest", {"payload": canonical_json(manifest)})
    staging = root.parent / f".{root.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True)
        for reference, result in zip(references, ordered, strict=True):
            save_operational_action_result(staging / reference, result)
        from .operational_adapter import operational_actions_to_phase12_arrays
        np.savez_compressed(staging / "phase12_compatible_action_arrays.npz", **operational_actions_to_phase12_arrays(ordered))
        _write_json(staging / "operational_action_summary.json", summary)
        _write_json(staging / "operational_action_complete.json", manifest)
        staging.replace(root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"summary": summary, "manifest": manifest}


def load_operational_action_dataset(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest = json.loads((base / "operational_action_complete.json").read_text(encoding="utf-8"))
    content_hash = manifest.pop("manifest_content_hash", None)
    if content_hash != make_deterministic_id("operational_action_manifest", {"payload": canonical_json(manifest)}):
        raise ValueError("operational action manifest content hash mismatch")
    manifest["manifest_content_hash"] = content_hash
    actual = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}
    if actual != set(manifest.get("managed_files", [])):
        raise ValueError("operational action managed inventory mismatch")
    results = [load_operational_action_result(base / "actions" / f"{action_id}.json") for action_id in manifest["action_ids"]]
    for result, action_id, expected_hash in zip(results, manifest["action_ids"], manifest["action_content_hashes"], strict=True):
        if result.action_id != action_id or result.content_hash != expected_hash:
            raise ValueError("operational manifest/artifact mismatch")
    with np.load(base / "phase12_compatible_action_arrays.npz", allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    from .operational_adapter import operational_actions_to_phase12_arrays
    expected_arrays = operational_actions_to_phase12_arrays(results)
    if set(arrays) != set(expected_arrays) or any(not np.array_equal(arrays[name], expected_arrays[name]) for name in expected_arrays):
        raise ValueError("operational Phase 12 adapter artifact mismatch")
    summary = json.loads((base / "operational_action_summary.json").read_text(encoding="utf-8"))
    if summary.get("operational_action_dataset_id") != manifest.get("operational_action_dataset_id"):
        raise ValueError("operational summary/manifest mismatch")
    return {"manifest": manifest, "summary": summary, "results": results, "phase12_arrays": arrays}


__all__ = ["OPERATIONAL_ACTION_DATASET_SCHEMA", "load_operational_action_dataset", "load_operational_action_result", "save_operational_action_result", "validate_operational_action_result", "write_operational_action_dataset"]
