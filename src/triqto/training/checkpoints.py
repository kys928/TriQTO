"""Safe, pickle-free, exact Phase 14 checkpoint persistence."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from triqto.core.ids import canonical_json
from triqto.model import (
    TriQTOModelConfig,
    load_architecture_state_dict_strict,
    model_config_to_dict,
    state_dict_signature,
)

from .config import TrainingConfig, training_config_to_dict
from .constants import TRAINING_CHECKPOINT_VERSION
from .identities import training_checkpoint_id
from .models import TrainingDataSpec
from .scheduler import DeterministicLRScheduler

_METADATA_ARRAY = "__metadata_json_utf8__"


class _TreeEncoder:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.counter = 0

    def encode(self, value: Any) -> Any:
        if isinstance(value, Tensor):
            name = f"tensor_{self.counter:08d}"
            self.counter += 1
            array = value.detach().cpu().contiguous().numpy()
            if array.dtype.kind == "O":
                raise TypeError("Checkpoint tensor cannot use object dtype")
            self.arrays[name] = array
            return {"__tensor__": name}
        if isinstance(value, np.ndarray):
            name = f"tensor_{self.counter:08d}"
            self.counter += 1
            array = np.ascontiguousarray(value)
            if array.dtype.kind == "O":
                raise TypeError("Checkpoint array cannot use object dtype")
            self.arrays[name] = array
            return {"__ndarray__": name}
        if isinstance(value, dict):
            return {
                "__dict__": [
                    [self.encode_key(key), self.encode(item)]
                    for key, item in value.items()
                ]
            }
        if isinstance(value, tuple):
            return {"__tuple__": [self.encode(item) for item in value]}
        if isinstance(value, list):
            return {"__list__": [self.encode(item) for item in value]}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Checkpoint metadata cannot contain non-finite floats")
            return value
        raise TypeError(f"Unsupported checkpoint tree type: {type(value)!r}")

    @staticmethod
    def encode_key(key: Any) -> dict[str, Any]:
        if key is None:
            return {"type": "none", "value": None}
        if isinstance(key, bool):
            return {"type": "bool", "value": key}
        if isinstance(key, int):
            return {"type": "int", "value": key}
        if isinstance(key, str):
            return {"type": "str", "value": key}
        raise TypeError(f"Unsupported checkpoint dictionary key: {type(key)!r}")


class _TreeDecoder:
    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.used: set[str] = set()

    def decode(self, value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"__tensor__"}:
            name = value["__tensor__"]
            self.used.add(name)
            return torch.from_numpy(self.arrays[name].copy())
        if isinstance(value, dict) and set(value) == {"__ndarray__"}:
            name = value["__ndarray__"]
            self.used.add(name)
            return self.arrays[name].copy()
        if isinstance(value, dict) and set(value) == {"__dict__"}:
            return {
                self.decode_key(key): self.decode(item)
                for key, item in value["__dict__"]
            }
        if isinstance(value, dict) and set(value) == {"__tuple__"}:
            return tuple(self.decode(item) for item in value["__tuple__"])
        if isinstance(value, dict) and set(value) == {"__list__"}:
            return [self.decode(item) for item in value["__list__"]]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError("Malformed checkpoint metadata tree")

    @staticmethod
    def decode_key(payload: Any) -> Any:
        if not isinstance(payload, dict) or set(payload) != {"type", "value"}:
            raise TypeError("Malformed checkpoint dictionary key")
        kind, value = payload["type"], payload["value"]
        if kind == "none" and value is None:
            return None
        if kind == "bool" and isinstance(value, bool):
            return value
        if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
            return value
        if kind == "str" and isinstance(value, str):
            return value
        raise TypeError("Malformed checkpoint dictionary key type")


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(payload) != expected:
        raise ValueError("RNG state key mismatch")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"])
    cuda = payload["torch_cuda"]
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(list(cuda))


def _content_hash(metadata_without_hash: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(canonical_json(metadata_without_hash).encode("utf-8"))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: DeterministicLRScheduler,
    training_config: TrainingConfig,
    model_config: TriQTOModelConfig,
    data_spec: TrainingDataSpec,
    training_schema_id: str,
    training_recipe_id: str,
    operational_config_id: str,
    training_run_id: str,
    epoch_completed: int,
    stage_index: int,
    global_step: int,
    best_validation_loss: float,
    best_epoch: int,
    kind: str,
) -> dict[str, Any]:
    if kind not in {"epoch", "best", "final"}:
        raise ValueError("checkpoint kind must be epoch, best, or final")
    for name, value in (
        ("epoch_completed", epoch_completed),
        ("stage_index", stage_index),
        ("global_step", global_step),
        ("best_epoch", best_epoch),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not math.isfinite(best_validation_loss):
        raise ValueError("best_validation_loss must be finite")
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    encoder = _TreeEncoder()
    tree = encoder.encode(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": capture_rng_state(),
        }
    )
    checkpoint_id = training_checkpoint_id(
        training_run_id,
        epoch_completed=epoch_completed,
        global_step=global_step,
        kind=kind,
    )
    metadata: dict[str, Any] = {
        "checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "checkpoint_id": checkpoint_id,
        "kind": kind,
        "training_schema_id": training_schema_id,
        "training_recipe_id": training_recipe_id,
        "operational_config_id": operational_config_id,
        "training_run_id": training_run_id,
        "training_view_dataset_id": data_spec.training_view_dataset_id,
        "training_config": training_config_to_dict(training_config),
        "model_config": model_config_to_dict(model_config),
        "data_spec": data_spec.to_dict(),
        "epoch_completed": epoch_completed,
        "stage_index": stage_index,
        "global_step": global_step,
        "best_validation_loss": float(best_validation_loss),
        "best_epoch": best_epoch,
        "model_state_signature": state_dict_signature(model),
        "optimizer_state_present": True,
        "scheduler_state_present": True,
        "rng_state_present": True,
        "tree": tree,
    }
    content_hash = _content_hash(metadata, encoder.arrays)
    metadata["content_hash"] = content_hash
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    arrays = dict(encoder.arrays)
    arrays[_METADATA_ARRAY] = np.frombuffer(metadata_bytes, dtype=np.uint8).copy()
    np.savez_compressed(target, **arrays)
    return {
        "checkpoint_id": checkpoint_id,
        "content_hash": content_hash,
        "model_state_signature": metadata["model_state_signature"],
    }


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: DeterministicLRScheduler | None = None,
    restore_rng: bool = False,
    expected_training_run_id: str | None = None,
) -> dict[str, Any]:
    target = Path(path)
    with np.load(target, allow_pickle=False) as payload:
        if _METADATA_ARRAY not in payload.files:
            raise ValueError("Checkpoint metadata array is missing")
        metadata_array = payload[_METADATA_ARRAY]
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise TypeError("Checkpoint metadata must be one-dimensional uint8")
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
        arrays = {
            name: payload[name].copy()
            for name in payload.files
            if name != _METADATA_ARRAY
        }
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a dictionary")
    content_hash = metadata.pop("content_hash", None)
    if content_hash != _content_hash(metadata, arrays):
        raise ValueError("Checkpoint logical content hash mismatch")
    metadata["content_hash"] = content_hash
    if metadata.get("checkpoint_version") != TRAINING_CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    if expected_training_run_id is not None and metadata.get("training_run_id") != expected_training_run_id:
        raise ValueError("Checkpoint training_run_id mismatch")
    decoder = _TreeDecoder(arrays)
    decoded = decoder.decode(metadata["tree"])
    if decoder.used != set(arrays):
        raise ValueError("Checkpoint contains unreferenced tensor arrays")
    if model is not None:
        load_architecture_state_dict_strict(model, decoded["model_state"])
        if state_dict_signature(model) != metadata["model_state_signature"]:
            raise ValueError("Restored model state signature mismatch")
    if optimizer is not None:
        optimizer.load_state_dict(decoded["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(decoded["scheduler_state"])
    if restore_rng:
        restore_rng_state(decoded["rng_state"])
    return {**metadata, "decoded_state": decoded}


__all__ = [
    "capture_rng_state",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_training_checkpoint",
]
