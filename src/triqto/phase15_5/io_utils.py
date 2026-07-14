"""I/O and immutable-source helpers for Phase 15.5."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from triqto.model.constants import HEAD_ORDER
from triqto.training import collate_training_examples, load_training_examples
from triqto.training.models import TrainingDataSpec

def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a mapping")
    return payload

def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

def _managed_inventory(root: Path, references: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for reference in sorted(references):
        path = root / reference
        result.append({"reference": reference, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    return result

def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return first == second or first in second.parents or second in first.parents

def _validate_training_binding(training_root: Path, checkpoint: Mapping[str, Any], dataset_id: str, model: Any) -> dict[str, Any]:
    complete = _read_json(training_root / "training_complete.json")
    summary = _read_json(training_root / "training_summary.json")
    if complete.get("complete") is not True or summary.get("model_trained") is not True:
        raise ValueError("Phase 15.5 requires a completed trained Phase 14 run")
    if checkpoint.get("global_step", 0) <= 0 or complete.get("global_step", 0) <= 0:
        raise ValueError("Phase 15.5 rejects untrained/zero-step checkpoints")
    if checkpoint.get("training_run_id") != complete.get("training_run_id"):
        raise ValueError("Phase 15.5 checkpoint/training run mismatch")
    if checkpoint.get("training_view_dataset_id") != dataset_id or complete.get("training_view_dataset_id") != dataset_id:
        raise ValueError("Phase 15.5 checkpoint/Phase 12 mismatch")
    if complete.get("model_architecture_id") != model.architecture_id:
        raise ValueError("Phase 15.5 model architecture mismatch")
    if complete.get("topology_loss_weight") != 0.0 or summary.get("topology_loss_weight") != 0.0:
        raise ValueError("Phase 15.5 requires topology loss exactly zero")
    if complete.get("test_split_used_for_optimization") is not False or summary.get("test_split_evaluated") is not False:
        raise ValueError("Phase 15.5 requires untouched Phase 12 test rows")
    return complete

def _selected_examples(dataset: Any, spec: TrainingDataSpec, phase7_root: Path, split: str, maximum: int) -> list[Any]:
    examples = load_training_examples(
        dataset,
        tasks=("diagnosis",),
        split=split,
        spec=spec,
        phase7_root=phase7_root,
        allow_evaluation_splits=split == "test",
    )
    ordered = sorted(examples, key=lambda value: value.view_item_id)
    if not ordered:
        raise ValueError(f"Phase 15.5 found no diagnosis examples for split {split}")
    return ordered[:maximum]

def _latent_table(model: Any, examples: Sequence[Any]) -> dict[str, np.ndarray]:
    head_index = HEAD_ORDER.index("diagnosis")
    rows: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for example in sorted(examples, key=lambda value: value.view_item_id):
            batch = collate_training_examples([example])
            output = model(batch.model_batch)
            latent = output.head_latents[0, head_index, :].detach().cpu().to(torch.float64).numpy()
            if latent.ndim != 1 or not np.isfinite(latent).all():
                raise ValueError("Phase 15.5 extracted invalid diagnosis latent")
            if example.entity_id in rows:
                raise ValueError(f"duplicate Phase 15.5 entity_id {example.entity_id}")
            rows[example.entity_id] = np.ascontiguousarray(latent, dtype=np.float64)
    return rows
