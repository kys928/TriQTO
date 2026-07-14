"""Deterministic family-conditioned operational policy for Phase 15.5."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from triqto.core.ids import canonical_json, make_deterministic_id

POLICY_SCHEMA = "triqto.phase15_5.operational_policy.v1"
FAMILY_NAMES = (
    "diagnostic_evidence_acquisition",
    "compilation_layout",
    "compilation_routing",
    "semantics_preserving_optimization",
)


@dataclass(frozen=True, slots=True)
class PolicyDataset:
    candidate_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    splits: tuple[str, ...]
    family_ids: np.ndarray
    context_features: np.ndarray
    candidate_features: np.ndarray
    utilities: np.ndarray
    available_mask: np.ndarray

    def validate(self) -> None:
        n = len(self.candidate_ids)
        if n == 0 or len(set(self.candidate_ids)) != n:
            raise ValueError("policy dataset requires unique candidate IDs")
        for name, values in (("group_ids", self.group_ids), ("split_group_ids", self.split_group_ids), ("splits", self.splits)):
            if len(values) != n or any(not value for value in values):
                raise ValueError(f"{name} must match candidate rows and be nonblank")
        if self.family_ids.shape != (n,) or self.family_ids.dtype != np.int64:
            raise ValueError("family_ids must be int64 with candidate shape")
        if np.any(self.family_ids < 0) or np.any(self.family_ids >= len(FAMILY_NAMES)):
            raise ValueError("family_ids contain out-of-range values")
        if self.context_features.ndim != 2 or self.context_features.shape[0] != n:
            raise ValueError("context_features must be two-dimensional with candidate rows")
        if self.candidate_features.ndim != 2 or self.candidate_features.shape[0] != n:
            raise ValueError("candidate_features must be two-dimensional with candidate rows")
        for name, values in (("context_features", self.context_features), ("candidate_features", self.candidate_features), ("utilities", self.utilities)):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.utilities.shape != (n,) or np.any(self.utilities < 0.0) or np.any(self.utilities > 1.0):
            raise ValueError("utilities must have candidate shape and lie in [0,1]")
        if self.available_mask.shape != (n,) or self.available_mask.dtype != np.bool_:
            raise ValueError("available_mask must be bool with candidate shape")
        if any(value not in {"train", "validation", "test"} for value in self.splits):
            raise ValueError("policy dataset split must be train, validation, or test")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, group_id in enumerate(self.group_ids):
            grouped[group_id].append(index)
        for group_id, indices in grouped.items():
            if len({int(self.family_ids[i]) for i in indices}) != 1 or len({self.splits[i] for i in indices}) != 1 or len({self.split_group_ids[i] for i in indices}) != 1:
                raise ValueError(f"policy group {group_id} mixes family/split identities")
            if not bool(self.available_mask[np.asarray(indices, dtype=np.int64)].any()):
                raise ValueError(f"policy group {group_id} has no available candidate")


class OperationalPolicy(nn.Module):
    def __init__(self, context_dim: int, candidate_dim: int, hidden_dim: int, family_count: int = len(FAMILY_NAMES)) -> None:
        super().__init__()
        if min(context_dim, candidate_dim, hidden_dim, family_count) <= 0:
            raise ValueError("policy dimensions must be positive")
        self.context_dim, self.candidate_dim, self.hidden_dim, self.family_count = map(int, (context_dim, candidate_dim, hidden_dim, family_count))
        self.network = nn.Sequential(
            nn.Linear(context_dim + candidate_dim + family_count, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: Tensor, candidate: Tensor, family_ids: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context tensor shape mismatch")
        if candidate.ndim != 2 or candidate.shape != (context.shape[0], self.candidate_dim):
            raise ValueError("candidate tensor shape mismatch")
        if family_ids.dtype != torch.long or family_ids.shape != (context.shape[0],):
            raise ValueError("family_ids must be int64 with row shape")
        family = torch.nn.functional.one_hot(family_ids, num_classes=self.family_count).to(context.dtype)
        return self.network(torch.cat((context, candidate, family), dim=1)).squeeze(1)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = values.mean(axis=0), values.std(axis=0)
    return mean.astype(np.float64), np.where(std > 1e-12, std, 1.0).astype(np.float64)


def _groups(dataset: PolicyDataset, split: str) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (group_id, row_split) in enumerate(zip(dataset.group_ids, dataset.splits, strict=True)):
        if row_split == split:
            grouped[group_id].append(index)
    return [np.asarray(grouped[key], dtype=np.int64) for key in sorted(grouped)]


def _loss(model: OperationalPolicy, tensors: Mapping[str, Tensor], indices: Tensor, mse_weight: float) -> Tensor:
    available = tensors["available"].index_select(0, indices)
    selected = indices[available]
    if selected.numel() == 0:
        raise ValueError("policy group has no available candidate")
    scores = model(tensors["context"].index_select(0, selected), tensors["candidate"].index_select(0, selected), tensors["family_ids"].index_select(0, selected))
    targets = tensors["utilities"].index_select(0, selected)
    best = torch.argmax(targets).reshape(1)
    ranking = torch.nn.functional.cross_entropy(scores.unsqueeze(0), best)
    return ranking + float(mse_weight) * torch.nn.functional.mse_loss(torch.sigmoid(scores), targets)


def _eval(model: OperationalPolicy, tensors: Mapping[str, Tensor], groups: Sequence[np.ndarray], mse_weight: float) -> float:
    if not groups:
        raise ValueError("policy evaluation split has no groups")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for group in groups:
            total += float(_loss(model, tensors, torch.as_tensor(group, dtype=torch.long), mse_weight).cpu())
    return total / len(groups)


def train_operational_policy(dataset: PolicyDataset, *, hidden_dim: int, epochs: int, learning_rate: float, weight_decay: float, utility_mse_weight: float, seed: int) -> dict[str, Any]:
    dataset.validate()
    if epochs <= 0 or hidden_dim <= 0:
        raise ValueError("policy epochs/hidden_dim must be positive")
    train_groups, validation_groups = _groups(dataset, "train"), _groups(dataset, "validation")
    if not train_groups or not validation_groups:
        raise ValueError("operational policy requires train and validation groups")
    train_rows = np.asarray([value == "train" for value in dataset.splits], dtype=np.bool_)
    context_mean, context_std = _normalization(dataset.context_features[train_rows])
    candidate_mean, candidate_std = _normalization(dataset.candidate_features[train_rows])
    tensors = {
        "context": torch.as_tensor((dataset.context_features - context_mean) / context_std, dtype=torch.float32),
        "candidate": torch.as_tensor((dataset.candidate_features - candidate_mean) / candidate_std, dtype=torch.float32),
        "family_ids": torch.as_tensor(dataset.family_ids, dtype=torch.long),
        "utilities": torch.as_tensor(dataset.utilities, dtype=torch.float32),
        "available": torch.as_tensor(dataset.available_mask, dtype=torch.bool),
    }
    _seed(seed)
    model = OperationalPolicy(dataset.context_features.shape[1], dataset.candidate_features.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, Tensor] | None = None
    best_validation = math.inf
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.stack([_loss(model, tensors, torch.as_tensor(group, dtype=torch.long), utility_mse_weight) for group in train_groups]).mean()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        validation_loss = _eval(model, tensors, validation_groups, utility_mse_weight)
        history.append({"epoch": epoch, "train_loss": float(train_loss.detach().cpu()), "validation_loss": validation_loss})
        if validation_loss < best_validation - 1e-12:
            best_validation, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None or best_epoch < 0 or not math.isfinite(best_validation):
        raise RuntimeError("operational policy failed to produce a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    return {"model": model.eval(), "history": history, "best_epoch": best_epoch, "best_validation_loss": best_validation, "context_mean": context_mean, "context_std": context_std, "candidate_mean": candidate_mean, "candidate_std": candidate_std}


def score_dataset(model: OperationalPolicy, dataset: PolicyDataset, *, context_mean: np.ndarray, context_std: np.ndarray, candidate_mean: np.ndarray, candidate_std: np.ndarray) -> np.ndarray:
    dataset.validate()
    context = torch.as_tensor((dataset.context_features - context_mean) / context_std, dtype=torch.float32)
    candidate = torch.as_tensor((dataset.candidate_features - candidate_mean) / candidate_std, dtype=torch.float32)
    family_ids = torch.as_tensor(dataset.family_ids, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        scores = model(context, candidate, family_ids).cpu().to(torch.float64).numpy()
    scores = np.asarray(scores, dtype=np.float64)
    scores[~dataset.available_mask] = -math.inf
    return scores


def _state_arrays(model: OperationalPolicy) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    mapping: dict[str, str] = {}
    for index, (name, tensor) in enumerate(sorted(model.state_dict().items())):
        key = f"tensor_{index:04d}"
        arrays[key] = tensor.detach().cpu().contiguous().numpy()
        mapping[name] = key
    return arrays, mapping


def _checkpoint_hash(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(canonical_json(dict(metadata)).encode("utf-8"))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return f"sha256:{digest.hexdigest()}"


def save_policy_checkpoint(path: str | Path, *, training_result: Mapping[str, Any], source_identity: Mapping[str, Any], config_identity: Mapping[str, Any]) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"policy checkpoint exists: {target}")
    model = training_result["model"]
    if not isinstance(model, OperationalPolicy):
        raise TypeError("training result model must be OperationalPolicy")
    arrays, mapping = _state_arrays(model)
    arrays.update({"context_mean": np.asarray(training_result["context_mean"], dtype=np.float64), "context_std": np.asarray(training_result["context_std"], dtype=np.float64), "candidate_mean": np.asarray(training_result["candidate_mean"], dtype=np.float64), "candidate_std": np.asarray(training_result["candidate_std"], dtype=np.float64)})
    metadata: dict[str, Any] = {"schema": POLICY_SCHEMA, "context_dim": model.context_dim, "candidate_dim": model.candidate_dim, "hidden_dim": model.hidden_dim, "family_names": list(FAMILY_NAMES), "state_mapping": mapping, "best_epoch": int(training_result["best_epoch"]), "best_validation_loss": float(training_result["best_validation_loss"]), "trained": True, "physical_hardware": False, "topology_loss_weight": 0.0, "source_identity": dict(source_identity), "config_identity": dict(config_identity)}
    metadata["policy_checkpoint_id"] = make_deterministic_id("phase155_policy", metadata)
    metadata["content_hash"] = _checkpoint_hash(metadata, arrays)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    target.with_suffix(".json").write_text(json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata


def load_policy_checkpoint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    metadata = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    expected = metadata.pop("content_hash", None)
    with np.load(target, allow_pickle=False) as payload:
        arrays = {name: payload[name].copy() for name in payload.files}
    if expected != _checkpoint_hash(metadata, arrays):
        raise ValueError("operational policy checkpoint content hash mismatch")
    metadata["content_hash"] = expected
    if metadata.get("schema") != POLICY_SCHEMA or metadata.get("trained") is not True:
        raise ValueError("operational policy checkpoint claim boundary mismatch")
    if metadata.get("physical_hardware") is not False or metadata.get("topology_loss_weight") != 0.0:
        raise ValueError("operational policy checkpoint hardware/topology boundary mismatch")
    model = OperationalPolicy(int(metadata["context_dim"]), int(metadata["candidate_dim"]), int(metadata["hidden_dim"]), len(metadata["family_names"]))
    state = {name: torch.from_numpy(arrays[key].copy()) for name, key in metadata["state_mapping"].items()}
    model.load_state_dict(state, strict=True)
    return {"model": model.eval(), "metadata": metadata, "context_mean": np.asarray(arrays["context_mean"], dtype=np.float64), "context_std": np.asarray(arrays["context_std"], dtype=np.float64), "candidate_mean": np.asarray(arrays["candidate_mean"], dtype=np.float64), "candidate_std": np.asarray(arrays["candidate_std"], dtype=np.float64)}


__all__ = ["FAMILY_NAMES", "OperationalPolicy", "POLICY_SCHEMA", "PolicyDataset", "load_policy_checkpoint", "save_policy_checkpoint", "score_dataset", "train_operational_policy"]
