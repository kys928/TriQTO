"""Immutable topology artifacts derived only from validated latent extractions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import shutil
import uuid

import numpy as np
from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.training.latent_extraction import load_latent_extraction
from .latent import LatentTopologyConfig, compute_latent_topology

CHECKPOINT_LATENT_TOPOLOGY_SCHEMA = "triqto.checkpoint_latent_topology.v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_checkpoint_bound_latent_topology(*, latent_extraction_root: str | Path, output_root: str | Path, config: LatentTopologyConfig | None = None) -> dict[str, Any]:
    extracted = load_latent_extraction(latent_extraction_root)
    metadata = extracted["metadata"]
    if metadata.get("trained_checkpoint") is not True or metadata.get("topology_loss_weight") != 0.0:
        raise ValueError("checkpoint-bound topology requires trained coordinates and zero topology loss")
    result = compute_latent_topology(
        checkpoint_id=metadata["checkpoint_id"],
        checkpoint_content_hash=metadata["checkpoint_content_hash"],
        model_architecture_id=metadata["model_architecture_id"],
        model_config_id=metadata["model_config_id"],
        training_view_dataset_id=metadata["training_view_dataset_id"],
        latent_extraction_id=metadata["latent_extraction_id"],
        split=metadata["split"],
        head=metadata["head"],
        representation=metadata["representation"],
        point_ids=extracted["point_ids"],
        coordinates=extracted["coordinates"],
        config=config,
        evidence_tier=metadata["evidence_tier"],
    )
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"latent topology output exists: {output}")
    diagrams = {int(dimension): np.asarray(values, dtype=np.float64) for dimension, values in result["persistence_diagrams"].items()}
    public = dict(result)
    public.pop("persistence_diagrams")
    public["schema"] = CHECKPOINT_LATENT_TOPOLOGY_SCHEMA
    public["coordinate_source"] = "validated_latent_extraction_artifact"
    public["coordinate_source_identity"] = metadata["latent_extraction_id"]
    public["result_content_hash"] = make_deterministic_id("checkpoint_latent_topology_result", {"payload": canonical_json(public)})
    manifest = {
        "schema": CHECKPOINT_LATENT_TOPOLOGY_SCHEMA,
        "latent_topology_id": result["latent_topology_id"],
        "latent_extraction_id": metadata["latent_extraction_id"],
        "checkpoint_id": metadata["checkpoint_id"],
        "result_content_hash": public["result_content_hash"],
        "managed_files": ["latent_topology_result.json", "latent_topology_diagrams.npz", "latent_topology_complete.json"],
    }
    manifest["manifest_content_hash"] = make_deterministic_id("checkpoint_latent_topology_manifest", {"payload": canonical_json(manifest)})
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True)
        np.savez_compressed(staging / "latent_topology_diagrams.npz", **{f"h{dimension}": diagram for dimension, diagram in sorted(diagrams.items())})
        _write_json(staging / "latent_topology_result.json", public)
        _write_json(staging / "latent_topology_complete.json", manifest)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"manifest": manifest, "result": public, "persistence_diagrams": diagrams}


def load_checkpoint_bound_latent_topology(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest = json.loads((base / "latent_topology_complete.json").read_text(encoding="utf-8"))
    manifest_hash = manifest.pop("manifest_content_hash", None)
    if manifest_hash != make_deterministic_id("checkpoint_latent_topology_manifest", {"payload": canonical_json(manifest)}):
        raise ValueError("latent topology manifest content hash mismatch")
    manifest["manifest_content_hash"] = manifest_hash
    actual = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}
    if actual != set(manifest.get("managed_files", [])):
        raise ValueError("latent topology managed inventory mismatch")
    result = json.loads((base / "latent_topology_result.json").read_text(encoding="utf-8"))
    result_hash = result.pop("result_content_hash", None)
    if result_hash != make_deterministic_id("checkpoint_latent_topology_result", {"payload": canonical_json(result)}):
        raise ValueError("latent topology result content hash mismatch")
    result["result_content_hash"] = result_hash
    if result.get("checkpoint_bound") is not True or result.get("trained_checkpoint") is not True or result.get("diagnostic_only") is not True or result.get("topology_loss_weight") != 0.0:
        raise ValueError("latent topology claim boundary mismatch")
    with np.load(base / "latent_topology_diagrams.npz", allow_pickle=False) as payload:
        diagrams = {int(name.removeprefix("h")): np.asarray(payload[name], dtype=np.float64).copy() for name in payload.files}
    if set(diagrams) != set(result["metadata"]["config"]["homology_dimensions"]):
        raise ValueError("latent topology diagram dimensions mismatch")
    if result.get("latent_topology_id") != manifest.get("latent_topology_id"):
        raise ValueError("latent topology identity mismatch")
    return {"manifest": manifest, "result": result, "persistence_diagrams": diagrams}


__all__ = ["CHECKPOINT_LATENT_TOPOLOGY_SCHEMA", "load_checkpoint_bound_latent_topology", "run_checkpoint_bound_latent_topology"]
