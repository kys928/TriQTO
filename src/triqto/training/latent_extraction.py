"""Deterministic latent extraction from a validated trained Phase 14 checkpoint."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import shutil
import uuid

import numpy as np
import torch
import yaml

from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.model import TriQTOModel, model_config_from_dict, model_config_id
from triqto.model.constants import HEAD_ORDER
from triqto.training.checkpoints import load_training_checkpoint
from triqto.training.datamodule import collate_training_examples, load_training_examples
from triqto.training.models import TrainingDataSpec
from triqto.training.source import load_completed_training_view_dataset, verify_training_view_snapshot

LATENT_EXTRACTION_SCHEMA = "triqto.latent_extraction.v1"


@dataclass(frozen=True, slots=True)
class LatentExtractionConfig:
    split: str = "validation"
    tasks: tuple[str, ...] = ("diagnosis",)
    head: str = "diagnosis"
    representation: str = "head_latent"
    max_points: int = 4096
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        tasks = tuple(str(value) for value in self.tasks)
        if not tasks or len(set(tasks)) != len(tasks) or any(not value for value in tasks):
            raise ValueError("tasks must be unique nonblank strings")
        if self.head not in HEAD_ORDER:
            raise ValueError(f"head must be one of {HEAD_ORDER}")
        if self.representation not in {"head_latent", "graph_embedding"}:
            raise ValueError("representation must be head_latent or graph_embedding")
        for name, value in (("max_points", self.max_points), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive integer")
        object.__setattr__(self, "tasks", tasks)


def latent_extraction_config_to_dict(config: LatentExtractionConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["tasks"] = list(config.tasks)
    return payload


def load_latent_extraction_config(path: str | Path) -> LatentExtractionConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("latent extraction config must contain mapping")
    allowed = set(LatentExtractionConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    if set(payload) - allowed:
        raise ValueError(f"unknown latent extraction fields: {sorted(set(payload) - allowed)}")
    data = dict(payload)
    if "tasks" in data:
        data["tasks"] = tuple(data["tasks"])
    return LatentExtractionConfig(**data)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a mapping")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _coordinate_hash(point_ids: Sequence[str], coordinates: np.ndarray) -> str:
    values = np.ascontiguousarray(coordinates, dtype=np.float64)
    digest = hashlib.sha256(canonical_json(list(point_ids)).encode("utf-8"))
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return f"sha256:{digest.hexdigest()}"


def restore_checkpoint_for_latents(checkpoint: str | Path, *, expected_training_run_id: str | None = None) -> tuple[TriQTOModel, dict[str, Any], TrainingDataSpec]:
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    probe = load_training_checkpoint(path, expected_training_run_id=expected_training_run_id)
    if not isinstance(probe.get("checkpoint_id"), str) or not probe["checkpoint_id"].strip():
        raise ValueError("latent extraction requires nonblank checkpoint_id")
    if not isinstance(probe.get("content_hash"), str) or not probe["content_hash"].startswith("sha256:"):
        raise ValueError("latent extraction requires checkpoint content hash")
    if isinstance(probe.get("global_step"), bool) or not isinstance(probe.get("global_step"), int) or probe["global_step"] <= 0:
        raise ValueError("latent extraction rejects untrained/zero-step checkpoint")
    model_config = model_config_from_dict(probe["model_config"])
    spec = TrainingDataSpec.from_dict(probe["data_spec"])
    model = TriQTOModel(model_config).eval()
    restored = load_training_checkpoint(path, model=model, expected_training_run_id=expected_training_run_id)
    if restored["checkpoint_id"] != probe["checkpoint_id"] or restored["content_hash"] != probe["content_hash"]:
        raise ValueError("checkpoint identity changed during restore")
    return model, restored, spec


def _validate_training_root(root: Path, checkpoint: Mapping[str, Any], *, dataset_id: str, architecture_id: str, config_id: str) -> dict[str, Any]:
    complete, summary = _json(root / "training_complete.json"), _json(root / "training_summary.json")
    if complete.get("complete") is not True or summary.get("training_executed") is not True or summary.get("model_trained") is not True:
        raise ValueError("latent extraction requires completed trained Phase 14 run")
    if complete.get("global_step", 0) <= 0 or complete.get("training_run_id") != checkpoint.get("training_run_id"):
        raise ValueError("checkpoint/training run mismatch")
    if complete.get("training_view_dataset_id") != dataset_id or checkpoint.get("training_view_dataset_id") != dataset_id:
        raise ValueError("checkpoint/Phase 12 source mismatch")
    if complete.get("model_architecture_id") != architecture_id or complete.get("model_config_id") != config_id:
        raise ValueError("model identity mismatch")
    if complete.get("topology_loss_weight") != 0.0 or summary.get("topology_loss_weight") != 0.0:
        raise ValueError("topology loss must remain zero")
    if complete.get("test_split_used_for_optimization") is not False or summary.get("test_split_evaluated") is not False:
        raise ValueError("test split must remain untouched during training")
    return complete


def extract_latent_coordinates(model: TriQTOModel, examples: Sequence[Any], *, config: LatentExtractionConfig) -> tuple[tuple[str, ...], np.ndarray]:
    ordered = sorted(examples, key=lambda value: value.view_item_id)
    if len(ordered) < 2 or len(ordered) > config.max_points:
        raise ValueError("latent extraction point count is outside configured bounds")
    point_ids: list[str] = []
    parts: list[np.ndarray] = []
    head_index = HEAD_ORDER.index(config.head)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ordered), config.batch_size):
            group = ordered[start : start + config.batch_size]
            batch = collate_training_examples(group)
            active = batch.model_batch.resolved_head_active_mask()[:, head_index]
            if not bool(active.all()):
                raise ValueError(f"selected latent head {config.head!r} is inactive")
            output = model(batch.model_batch)
            latent = output.head_latents[:, head_index, :] if config.representation == "head_latent" else output.graph_embedding
            values = latent.detach().cpu().to(torch.float64).numpy()
            if values.shape[0] != len(group) or values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError("invalid latent coordinates")
            point_ids.extend(item.view_item_id for item in group)
            parts.append(values)
    ids = tuple(point_ids)
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("latent point IDs must be unique nonblank")
    return ids, np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)


def extract_checkpoint_latents(*, training_view_root: str | Path, training_root: str | Path, checkpoint: str | Path, output_root: str | Path, config: LatentExtractionConfig, phase7_root: str | Path | None = None) -> dict[str, Any]:
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"latent output exists: {output}")
    dataset = load_completed_training_view_dataset(training_view_root)
    verify_training_view_snapshot(dataset)
    model, checkpoint_meta, spec = restore_checkpoint_for_latents(checkpoint)
    if spec.training_view_dataset_id != dataset.training_view_dataset_id:
        raise ValueError("checkpoint data spec/Phase 12 mismatch")
    architecture_id, config_id = model.architecture_id, model_config_id(model.config)
    complete = _validate_training_root(Path(training_root), checkpoint_meta, dataset_id=dataset.training_view_dataset_id, architecture_id=architecture_id, config_id=config_id)
    examples = load_training_examples(dataset, tasks=config.tasks, split=config.split, spec=spec, phase7_root=phase7_root, allow_evaluation_splits=config.split == "test")
    if not examples or any(value.split != config.split for value in examples):
        raise ValueError("latent extraction found no exact selected-split examples")
    point_ids, coordinates = extract_latent_coordinates(model, examples, config=config)
    coordinate_hash = _coordinate_hash(point_ids, coordinates)
    metadata: dict[str, Any] = {
        "schema": LATENT_EXTRACTION_SCHEMA,
        "checkpoint_id": checkpoint_meta["checkpoint_id"],
        "checkpoint_content_hash": checkpoint_meta["content_hash"],
        "training_run_id": complete["training_run_id"],
        "trained": True,
        "trained_checkpoint": True,
        "model_architecture_id": architecture_id,
        "model_config_id": config_id,
        "training_view_dataset_id": dataset.training_view_dataset_id,
        "phase12_snapshot_hash": complete["phase12_snapshot_hash"],
        "split": config.split,
        "tasks": list(config.tasks),
        "head": config.head,
        "representation": config.representation,
        "point_ids": list(point_ids),
        "point_count": len(point_ids),
        "coordinate_dim": int(coordinates.shape[1]),
        "coordinate_hash": coordinate_hash,
        "config": latent_extraction_config_to_dict(config),
        "diagnostic_only": True,
        "physical_hardware": False,
        "evidence_tier": "trained_cpu_checkpoint",
        "topology_loss_weight": 0.0,
        "test_data_used_for_fitting": False,
    }
    identity_keys = ("schema", "checkpoint_id", "checkpoint_content_hash", "model_architecture_id", "model_config_id", "training_view_dataset_id", "phase12_snapshot_hash", "split", "tasks", "head", "representation", "point_ids", "coordinate_hash", "config")
    metadata["latent_extraction_id"] = make_deterministic_id("latent_extraction", {key: metadata[key] for key in identity_keys})
    metadata["metadata_content_hash"] = make_deterministic_id("latent_extraction_metadata", {"payload": canonical_json(metadata)})
    manifest = {
        "schema": LATENT_EXTRACTION_SCHEMA,
        "latent_extraction_id": metadata["latent_extraction_id"],
        "coordinate_hash": coordinate_hash,
        "metadata_content_hash": metadata["metadata_content_hash"],
        "managed_files": ["latent_coordinates.npz", "latent_metadata.json", "latent_complete.json"],
    }
    manifest["manifest_content_hash"] = make_deterministic_id("latent_extraction_manifest", {"payload": canonical_json(manifest)})
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True)
        np.savez_compressed(staging / "latent_coordinates.npz", point_ids=np.asarray(point_ids, dtype=str), coordinates=coordinates)
        _write_json(staging / "latent_metadata.json", metadata)
        _write_json(staging / "latent_complete.json", manifest)
        staging.replace(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verify_training_view_snapshot(dataset)
    return {"metadata": metadata, "manifest": manifest, "point_ids": point_ids, "coordinates": coordinates}


def load_latent_extraction(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    manifest = _json(base / "latent_complete.json")
    manifest_hash = manifest.pop("manifest_content_hash", None)
    if manifest_hash != make_deterministic_id("latent_extraction_manifest", {"payload": canonical_json(manifest)}):
        raise ValueError("latent extraction manifest content hash mismatch")
    manifest["manifest_content_hash"] = manifest_hash
    actual = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}
    if actual != set(manifest.get("managed_files", [])):
        raise ValueError("latent extraction managed inventory mismatch")
    metadata = _json(base / "latent_metadata.json")
    metadata_hash = metadata.pop("metadata_content_hash", None)
    if metadata_hash != make_deterministic_id("latent_extraction_metadata", {"payload": canonical_json(metadata)}):
        raise ValueError("latent extraction metadata content hash mismatch")
    metadata["metadata_content_hash"] = metadata_hash
    with np.load(base / "latent_coordinates.npz", allow_pickle=False) as payload:
        if set(payload.files) != {"point_ids", "coordinates"}:
            raise ValueError("latent coordinate array-name mismatch")
        point_ids = tuple(str(value) for value in payload["point_ids"].tolist())
        coordinates = np.asarray(payload["coordinates"], dtype=np.float64).copy()
    if coordinates.ndim != 2 or coordinates.shape[0] != len(point_ids) or not np.isfinite(coordinates).all():
        raise ValueError("latent coordinate shape/value mismatch")
    if len(point_ids) != len(set(point_ids)) or list(point_ids) != metadata.get("point_ids"):
        raise ValueError("latent point ordering/identity mismatch")
    coordinate_hash = _coordinate_hash(point_ids, coordinates)
    if coordinate_hash != metadata.get("coordinate_hash") or coordinate_hash != manifest.get("coordinate_hash"):
        raise ValueError("latent coordinate content hash mismatch")
    if metadata.get("trained_checkpoint") is not True or metadata.get("topology_loss_weight") != 0.0:
        raise ValueError("latent extraction claim boundary mismatch")
    return {"manifest": manifest, "metadata": metadata, "point_ids": point_ids, "coordinates": coordinates}


__all__ = ["LATENT_EXTRACTION_SCHEMA", "LatentExtractionConfig", "extract_checkpoint_latents", "extract_latent_coordinates", "latent_extraction_config_to_dict", "load_latent_extraction", "load_latent_extraction_config", "restore_checkpoint_for_latents"]
