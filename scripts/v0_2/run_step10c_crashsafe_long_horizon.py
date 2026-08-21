#!/usr/bin/env python3
"""Step 10C crash-safe longer-horizon dual-initialization benchmark.

Scientific change versus frozen Step 10B: max training ceiling 20 -> 40 only.
Engineering changes only: atomic best/resume checkpoints, exact optimizer/RNG
crash recovery, atomic progress heartbeat, wall-time and gradient-clipping
telemetry.  Final evaluation uses only the frozen fresh Step-10C outer cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
import run_step10_warmstart_vs_scratch as step10b

SCHEMA = "triqto.v0_2.step10c_crashsafe_long_horizon.v1"
OUTER_SCHEMA = "triqto.v0_2.step10c_fresh_outer_cohort.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10c_crashsafe_long_horizon.json"
DEFAULT_MIXTURE_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step10c_crashsafe_long_horizon")
DEFAULT_OUTER_PARENT = Path("/workspace/triqto-data/step10c_fresh_outer_cohort")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mixture-config", type=Path, default=DEFAULT_MIXTURE_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--mixture-product-dir", type=Path)
    parser.add_argument("--fresh-outer-product-dir", type=Path)
    parser.add_argument("--step9a-bundle-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Refuse to reuse any existing exact-identity resume checkpoint.",
    )
    return parser.parse_args()


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json_fsync(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path.parent)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temp.open("wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_dir(path.parent)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def _resolve_outer_product(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    pointer = DEFAULT_OUTER_PARENT / "current_product.json"
    if not pointer.is_file():
        raise RuntimeError(
            "fresh Step-10C outer current_product.json missing; generate/freeze outer cohort first"
        )
    payload = baseline.read_json(pointer)
    if payload.get("schema") != "triqto.v0_2.step10c_fresh_outer_current_product.v1":
        raise RuntimeError("unexpected Step-10C outer pointer schema")
    return Path(payload["product_dir"]).expanduser().resolve()


def _verify_outer_product(
    product: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], dict[int, int]]:
    complete = baseline.read_json(product / "dataset_complete.json")
    if complete.get("schema") != OUTER_SCHEMA or complete.get("status") != "COMPLETE_FROZEN_OUTER_VALIDATION":
        raise RuntimeError("fresh Step-10C outer product incomplete or wrong schema/status")
    if not bool(complete.get("outer_validation_only")) or bool(complete.get("step10b_outer_reused", True)):
        raise RuntimeError("fresh Step-10C outer scientific boundary failed")
    if bool(complete.get("model_evaluated_before_freeze", True)):
        raise RuntimeError("fresh Step-10C outer was not frozen before model evaluation")
    manifests = product / "manifests"
    for name, expected in complete.get("manifest_hashes", {}).items():
        if baseline.sha256_file(manifests / name) != expected:
            raise RuntimeError(f"fresh outer manifest hash mismatch: {name}")
    if baseline.sha256_file(product / "eda.json") != complete.get("eda_sha256"):
        raise RuntimeError("fresh outer EDA hash mismatch")
    eda = baseline.read_json(product / "eda.json")
    if eda.get("status") != "PASS" or bool(eda.get("model_evaluated", True)):
        raise RuntimeError("fresh outer EDA status/boundary failed")
    freshness = eda.get("freshness", {})
    if int(freshness.get("overlap_with_step10_original_graphs", -1)) != 0:
        raise RuntimeError("fresh outer overlaps Step-10 original graphs")
    if int(freshness.get("overlap_with_step10_bridge_graphs", -1)) != 0:
        raise RuntimeError("fresh outer overlaps Step-10 bridge graphs")
    if bool(freshness.get("exact_step9d_pilot_graph_present", True)):
        raise RuntimeError("fresh outer contains exact Step-9D pilot graph")

    original_rows = baseline.read_csv(manifests / "original_example_manifest.csv")
    bridge_rows = baseline.read_csv(manifests / "bridge_example_manifest.csv")
    if len(original_rows) != int(complete["original_example_count"]):
        raise RuntimeError("fresh original outer manifest count mismatch")
    if len(bridge_rows) != int(complete["bridge_example_count"]):
        raise RuntimeError("fresh bridge outer manifest count mismatch")
    parent_by_root: dict[int, int] = {}
    for row in bridge_rows:
        root = int(row["root_index"])
        parent = int(row["parent_group_index"])
        if root in parent_by_root and parent_by_root[root] != parent:
            raise RuntimeError("fresh bridge root maps to multiple parents")
        parent_by_root[root] = parent
    if len(set(parent_by_root.values())) != int(complete["bridge_parent_group_count"]):
        raise RuntimeError("fresh bridge parent-group count mismatch")
    return complete, original_rows, bridge_rows, parent_by_root


def _rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if torch.cuda.is_available() and snapshot.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _runtime_fingerprint(device: torch.device) -> dict[str, Any]:
    cuda_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device_type": str(device),
        "torch_num_threads": int(torch.get_num_threads()),
        "cuda_version_if_any": torch.version.cuda,
        "cuda_device_name_if_any": cuda_name,
    }


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return baseline.canonical_json(dict(left)) == baseline.canonical_json(dict(right))
    except (TypeError, ValueError):
        return False


def _assert_step10c_delta_only_horizon(config: Mapping[str, Any]) -> None:
    old = baseline.read_json(step10b.DEFAULT_CONFIG)
    if old.get("schema") != step10b.SCHEMA:
        raise RuntimeError("unexpected frozen Step-10B config")
    if config["architecture"]["variant"] != old["architecture"]["variant"]:
        raise RuntimeError("Step-10C architecture differs from Step-10B")
    if int(config["architecture"]["expected_trainable_parameter_count"]) != int(old["architecture"]["expected_trainable_parameter_count"]):
        raise RuntimeError("Step-10C parameter-count contract differs from Step-10B")
    frozen_keys = (
        "root_batch_size", "early_stopping_patience", "early_stopping_min_delta",
        "optimizer", "learning_rate", "weight_decay", "gradient_clip_norm",
        "effect_loss_weight", "mechanism_loss_weight", "domain_schedule",
    )
    for key in frozen_keys:
        if config["training"][key] != old["training"][key]:
            raise RuntimeError(f"Step-10C changed frozen training field {key}")
    if int(old["training"]["max_epochs"]) != 20 or int(config["training"]["max_epochs"]) != 40:
        raise RuntimeError("Step-10C horizon contract must be exactly 20 -> 40")
    if config["selection"]["eligible_checkpoint_order"] != old["selection"]["eligible_checkpoint_order"]:
        raise RuntimeError("Step-10C checkpoint-selection order differs from Step-10B")
    if config["evaluation"]["bridge_mechanism_gate"] != old["evaluation"]["bridge_mechanism_gate"]:
        raise RuntimeError("Step-10C bridge gate differs from Step-10B")


def _train_one_block_with_telemetry(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    block: step7.CachedBlock,
    *,
    device: torch.device,
    effect_class_weight: np.ndarray,
    mechanism_class_weight: np.ndarray,
    training: Mapping[str, Any],
) -> dict[str, float]:
    model.train()
    model_batch, targets = step7.move_cached_block(block, device)
    output = model(model_batch)
    effect_weight = torch.as_tensor(effect_class_weight, dtype=torch.float32, device=device)
    mechanism_weight = torch.as_tensor(mechanism_class_weight, dtype=torch.float32, device=device)

    effect_label = targets.effect_present.to(torch.long)
    per_effect = F.binary_cross_entropy_with_logits(
        output.effect_logit, targets.effect_present, reduction="none"
    )
    effect_loss = (per_effect * effect_weight.index_select(0, effect_label)).mean()
    mechanism_mask = targets.mechanism_loss_mask
    if not bool(mechanism_mask.any()):
        raise RuntimeError("Step-10C fit block unexpectedly has no mechanism-supervised rows")
    mechanism_label = targets.mechanism[mechanism_mask]
    per_mechanism = F.cross_entropy(
        output.mechanism_logits[mechanism_mask], mechanism_label, reduction="none"
    )
    mechanism_loss = (
        per_mechanism * mechanism_weight.index_select(0, mechanism_label)
    ).mean()
    total_loss = (
        float(training["effect_loss_weight"]) * effect_loss
        + float(training["mechanism_loss_weight"]) * mechanism_loss
    )
    if not torch.isfinite(total_loss):
        raise RuntimeError("non-finite Step-10C training loss")

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    preclip = float(step7.finite_gradient_norm(model))
    clip_limit = float(training["gradient_clip_norm"])
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
    postclip = float(step7.finite_gradient_norm(model))
    optimizer.step()
    return {
        "effect_loss": float(effect_loss.detach()),
        "mechanism_loss": float(mechanism_loss.detach()),
        "total_loss": float(total_loss.detach()),
        "preclip_gradient_norm": preclip,
        "postclip_gradient_norm": postclip,
        "clipped": float(preclip > clip_limit),
    }


def _domain_balanced_epoch_with_telemetry(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    original_blocks: Sequence[step7.CachedBlock],
    bridge_blocks: Sequence[step7.CachedBlock],
    *, seed: int, epoch: int, device: torch.device,
    effect_class_weight: np.ndarray, mechanism_class_weight: np.ndarray,
    training: Mapping[str, Any],
) -> dict[str, float]:
    if not original_blocks or not bridge_blocks:
        raise RuntimeError("Step-10C fit blocks missing a domain")
    started = time.monotonic()
    old_order = np.arange(len(original_blocks), dtype=np.int64)
    bridge_order = np.arange(len(bridge_blocks), dtype=np.int64)
    np.random.default_rng(seed * 1_000_003 + epoch * 97_409 + 11).shuffle(old_order)
    np.random.default_rng(seed * 1_000_003 + epoch * 97_409 + 29).shuffle(bridge_order)
    steps_per_domain = max(len(old_order), len(bridge_order))
    domains = ("original", "bridge") if epoch % 2 else ("bridge", "original")
    totals: dict[str, float] = defaultdict(float)
    max_preclip = 0.0
    optimizer_steps = 0

    for step_index in range(steps_per_domain):
        for domain in domains:
            if domain == "original":
                block = original_blocks[int(old_order[step_index % len(old_order)])]
            else:
                block = bridge_blocks[int(bridge_order[step_index % len(bridge_order)])]
            metrics = _train_one_block_with_telemetry(
                model, optimizer, block, device=device,
                effect_class_weight=effect_class_weight,
                mechanism_class_weight=mechanism_class_weight,
                training=training,
            )
            for key in (
                "effect_loss", "mechanism_loss", "total_loss",
                "preclip_gradient_norm", "postclip_gradient_norm", "clipped",
            ):
                totals[key] += float(metrics[key])
            max_preclip = max(max_preclip, float(metrics["preclip_gradient_norm"]))
            optimizer_steps += 1

    denominator = float(2 * steps_per_domain)
    return {
        "effect_loss": totals["effect_loss"] / denominator,
        "mechanism_loss": totals["mechanism_loss"] / denominator,
        "total_loss": totals["total_loss"] / denominator,
        "mean_preclip_gradient_norm": totals["preclip_gradient_norm"] / denominator,
        "max_preclip_gradient_norm": max_preclip,
        "fraction_optimizer_steps_clipped": totals["clipped"] / denominator,
        "mean_postclip_gradient_norm": totals["postclip_gradient_norm"] / denominator,
        "optimizer_steps": float(optimizer_steps),
        "optimizer_blocks_per_domain": float(steps_per_domain),
        "wall_time_seconds": float(time.monotonic() - started),
    }


def _resume_payload(
    *, identity: Mapping[str, Any], model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, completed_epoch: int, stale: int,
    best_eligible_key: tuple[float, float, float, float] | None,
    best_eligible_state: Mapping[str, torch.Tensor] | None,
    best_eligible_epoch: int | None,
    best_unconstrained_key: tuple[float, float, float, float] | None,
    best_unconstrained_state: Mapping[str, torch.Tensor] | None,
    best_unconstrained_epoch: int | None,
    history: Sequence[Mapping[str, Any]], run_complete: bool,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": "resume",
        "identity": dict(identity),
        "completed_epoch": int(completed_epoch),
        "stale": int(stale),
        "model_state_dict": _state_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eligible_key": best_eligible_key,
        "best_eligible_state": dict(best_eligible_state) if best_eligible_state is not None else None,
        "best_eligible_epoch": best_eligible_epoch,
        "best_unconstrained_key": best_unconstrained_key,
        "best_unconstrained_state": dict(best_unconstrained_state) if best_unconstrained_state is not None else None,
        "best_unconstrained_epoch": best_unconstrained_epoch,
        "history": [dict(row) for row in history],
        "rng_state": _rng_snapshot(),
        "run_complete": bool(run_complete),
    }


def _train_seed_crashsafe(
    *, initialization: str, seed: int, experiment: Mapping[str, Any],
    config: Mapping[str, Any], bundle: Path, original_fit: Sequence[step7.CachedBlock],
    bridge_fit: Sequence[step7.CachedBlock], original_selection: Sequence[step7.CachedBlock],
    bridge_selection: Sequence[step7.CachedBlock], floor: Mapping[str, float],
    effect_class_weight: np.ndarray, mechanism_class_weight: np.ndarray,
    device: torch.device, run_dir: Path, identity: Mapping[str, Any], allow_resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    resume_path = run_dir / "resume.pt"
    progress_path = run_dir / "progress.json"

    model = step10b._initialize_model(
        initialization=initialization, seed=seed, experiment=experiment,
        device=device, bundle=bundle, config=config,
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    min_delta = float(training["early_stopping_min_delta"])

    history: list[dict[str, Any]] = []
    best_eligible_key: tuple[float, float, float, float] | None = None
    best_eligible_state: dict[str, torch.Tensor] | None = None
    best_eligible_epoch: int | None = None
    best_unconstrained_key: tuple[float, float, float, float] | None = None
    best_unconstrained_state: dict[str, torch.Tensor] | None = None
    best_unconstrained_epoch: int | None = None
    stale = 0
    completed_epoch = 0
    resumed = False

    if resume_path.is_file():
        if not allow_resume:
            raise RuntimeError(f"resume checkpoint exists but --no-resume was requested: {resume_path}")
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("kind") != "resume":
            raise RuntimeError(f"invalid Step-10C resume checkpoint: {resume_path}")
        if not _identity_equal(identity, payload.get("identity", {})):
            raise RuntimeError(
                f"Step-10C resume identity mismatch for {initialization} seed={seed}; refusing continuation"
            )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed_epoch = int(payload["completed_epoch"])
        stale = int(payload["stale"])
        best_eligible_key = tuple(payload["best_eligible_key"]) if payload.get("best_eligible_key") is not None else None
        best_eligible_state = payload.get("best_eligible_state")
        best_eligible_epoch = payload.get("best_eligible_epoch")
        best_unconstrained_key = tuple(payload["best_unconstrained_key"]) if payload.get("best_unconstrained_key") is not None else None
        best_unconstrained_state = payload.get("best_unconstrained_state")
        best_unconstrained_epoch = payload.get("best_unconstrained_epoch")
        history = [dict(row) for row in payload.get("history", [])]
        _restore_rng(payload["rng_state"])
        resumed = True
        print(
            f"RESUME Step-10C {initialization} seed={seed} after epoch={completed_epoch} stale={stale}",
            flush=True,
        )
    else:
        original_pred = step7.predict_blocks(model, original_selection, device)
        bridge_pred = step7.predict_blocks(model, bridge_selection, device)
        original_summary = step7.selection_summary(original_pred)
        bridge_summary = step7.selection_summary(bridge_pred)
        eligible = step10b._eligible(original_summary, floor)
        key = step10b._candidate_key(original_summary, bridge_summary, 0)
        state = _state_cpu(model)
        best_unconstrained_key = key
        best_unconstrained_state = state
        best_unconstrained_epoch = 0
        improved = False
        if eligible:
            best_eligible_key = key
            best_eligible_state = state
            best_eligible_epoch = 0
            improved = True
            atomic_torch_save(best_path, {
                "schema": SCHEMA, "kind": "best", "identity": dict(identity),
                "epoch": 0, "candidate_key": key, "state_dict": state,
            })
        row = {
            "initialization": initialization,
            "seed": int(seed),
            "epoch": 0,
            "selection_eligible": bool(eligible),
            "original_selection_effect_ba": float(original_summary["effect_balanced_accuracy"]),
            "original_selection_mechanism_ba": float(original_summary["mechanism_balanced_accuracy"]),
            "bridge_selection_effect_ba": float(bridge_summary["effect_balanced_accuracy"]),
            "bridge_selection_mechanism_ba": float(bridge_summary["mechanism_balanced_accuracy"]),
            "retention_floor_effect_ba": float(floor["effect_balanced_accuracy"]),
            "retention_floor_mechanism_ba": float(floor["mechanism_balanced_accuracy"]),
            "eligible_checkpoint_improved": improved,
        }
        history.append(row)
        print(
            f"{initialization} seed={seed} epoch=00 "
            f"old_mech={row['original_selection_mechanism_ba']:.4f} "
            f"bridge_mech={row['bridge_selection_mechanism_ba']:.4f} "
            f"eligible={'YES' if eligible else 'NO'}",
            flush=True,
        )
        atomic_torch_save(
            resume_path,
            _resume_payload(
                identity=identity, model=model, optimizer=optimizer,
                completed_epoch=0, stale=stale,
                best_eligible_key=best_eligible_key,
                best_eligible_state=best_eligible_state,
                best_eligible_epoch=best_eligible_epoch,
                best_unconstrained_key=best_unconstrained_key,
                best_unconstrained_state=best_unconstrained_state,
                best_unconstrained_epoch=best_unconstrained_epoch,
                history=history, run_complete=False,
            ),
        )

    for epoch in range(completed_epoch + 1, max_epochs + 1):
        if best_eligible_state is not None and stale >= patience:
            break
        train_metrics = _domain_balanced_epoch_with_telemetry(
            model, optimizer, original_fit, bridge_fit,
            seed=seed, epoch=epoch, device=device,
            effect_class_weight=effect_class_weight,
            mechanism_class_weight=mechanism_class_weight,
            training=training,
        )
        original_pred = step7.predict_blocks(model, original_selection, device)
        bridge_pred = step7.predict_blocks(model, bridge_selection, device)
        original_summary = step7.selection_summary(original_pred)
        bridge_summary = step7.selection_summary(bridge_pred)
        eligible = step10b._eligible(original_summary, floor)
        key = step10b._candidate_key(original_summary, bridge_summary, epoch)
        state = _state_cpu(model)

        if step10b._better_key(key, best_unconstrained_key, min_delta):
            best_unconstrained_key = key
            best_unconstrained_state = state
            best_unconstrained_epoch = epoch

        improved = False
        if eligible and step10b._better_key(key, best_eligible_key, min_delta):
            best_eligible_key = key
            best_eligible_state = state
            best_eligible_epoch = epoch
            stale = 0
            improved = True
            atomic_torch_save(best_path, {
                "schema": SCHEMA, "kind": "best", "identity": dict(identity),
                "epoch": int(epoch), "candidate_key": key, "state_dict": state,
            })
        elif eligible and best_eligible_state is not None:
            stale += 1

        row: dict[str, Any] = {
            "initialization": initialization,
            "seed": int(seed),
            "epoch": int(epoch),
            "selection_eligible": bool(eligible),
            "original_selection_effect_ba": float(original_summary["effect_balanced_accuracy"]),
            "original_selection_mechanism_ba": float(original_summary["mechanism_balanced_accuracy"]),
            "bridge_selection_effect_ba": float(bridge_summary["effect_balanced_accuracy"]),
            "bridge_selection_mechanism_ba": float(bridge_summary["mechanism_balanced_accuracy"]),
            "retention_floor_effect_ba": float(floor["effect_balanced_accuracy"]),
            "retention_floor_mechanism_ba": float(floor["mechanism_balanced_accuracy"]),
            "eligible_checkpoint_improved": bool(improved),
        }
        row.update({f"train_{name}": float(value) for name, value in train_metrics.items()})
        history.append(row)
        completed_epoch = epoch

        atomic_torch_save(
            resume_path,
            _resume_payload(
                identity=identity, model=model, optimizer=optimizer,
                completed_epoch=completed_epoch, stale=stale,
                best_eligible_key=best_eligible_key,
                best_eligible_state=best_eligible_state,
                best_eligible_epoch=best_eligible_epoch,
                best_unconstrained_key=best_unconstrained_key,
                best_unconstrained_state=best_unconstrained_state,
                best_unconstrained_epoch=best_unconstrained_epoch,
                history=history, run_complete=False,
            ),
        )
        atomic_json_fsync(progress_path, {
            "schema": SCHEMA,
            "status": "RUNNING",
            "initialization": initialization,
            "seed": int(seed),
            "current_epoch": int(epoch),
            "max_epochs": max_epochs,
            "selection_eligible": bool(eligible),
            "stale_eligible_epochs": int(stale),
            "original_effect_ba": row["original_selection_effect_ba"],
            "original_mechanism_ba": row["original_selection_mechanism_ba"],
            "bridge_effect_ba": row["bridge_selection_effect_ba"],
            "bridge_mechanism_ba": row["bridge_selection_mechanism_ba"],
            "best_eligible_epoch": best_eligible_epoch,
            "best_eligible_bridge_mechanism_ba": None if best_eligible_key is None else float(best_eligible_key[0]),
            "epoch_wall_time_seconds": float(train_metrics["wall_time_seconds"]),
            "mean_preclip_gradient_norm": float(train_metrics["mean_preclip_gradient_norm"]),
            "max_preclip_gradient_norm": float(train_metrics["max_preclip_gradient_norm"]),
            "fraction_optimizer_steps_clipped": float(train_metrics["fraction_optimizer_steps_clipped"]),
            "mean_postclip_gradient_norm": float(train_metrics["mean_postclip_gradient_norm"]),
            "resume_used": resumed,
            "updated_unix": time.time(),
        })
        print(
            f"{initialization} seed={seed} epoch={epoch:02d} "
            f"old_mech={row['original_selection_mechanism_ba']:.4f} "
            f"bridge_mech={row['bridge_selection_mechanism_ba']:.4f} "
            f"eligible={'YES' if eligible else 'NO'} "
            f"best={best_eligible_epoch} clipped={train_metrics['fraction_optimizer_steps_clipped']:.3f}",
            flush=True,
        )

    selected_eligible = best_eligible_state is not None
    selected_state = best_eligible_state if selected_eligible else best_unconstrained_state
    selected_epoch = best_eligible_epoch if selected_eligible else best_unconstrained_epoch
    if selected_state is None or selected_epoch is None:
        raise RuntimeError(f"Step-10C {initialization} seed={seed} failed to select any state")

    atomic_torch_save(
        resume_path,
        _resume_payload(
            identity=identity, model=model, optimizer=optimizer,
            completed_epoch=int(max(int(row["epoch"]) for row in history)), stale=stale,
            best_eligible_key=best_eligible_key,
            best_eligible_state=best_eligible_state,
            best_eligible_epoch=best_eligible_epoch,
            best_unconstrained_key=best_unconstrained_key,
            best_unconstrained_state=best_unconstrained_state,
            best_unconstrained_epoch=best_unconstrained_epoch,
            history=history, run_complete=True,
        ),
    )

    model.load_state_dict(selected_state, strict=True)
    model.to(device)
    for row in history:
        row["selected_checkpoint"] = int(row["epoch"]) == int(selected_epoch)
    original_selected = step7.predict_blocks(model, original_selection, device)
    bridge_selected = step7.predict_blocks(model, bridge_selection, device)
    record = {
        "initialization": initialization,
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "selected_checkpoint_retention_eligible": bool(selected_eligible),
        "epochs_ran": int(max(int(row["epoch"]) for row in history)),
        "stopped_early": bool(max(int(row["epoch"]) for row in history) < max_epochs),
        "resumed_from_checkpoint": bool(resumed),
        "original_selection": step7.selection_summary(original_selected),
        "bridge_selection": step7.selection_summary(bridge_selected),
        "retention_floor": dict(floor),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    atomic_json_fsync(progress_path, {
        "schema": SCHEMA, "status": "COMPLETE",
        "initialization": initialization, "seed": int(seed),
        "epochs_ran": int(record["epochs_ran"]),
        "selected_epoch": int(selected_epoch),
        "selected_checkpoint_retention_eligible": bool(selected_eligible),
        "resume_used": bool(resumed), "updated_unix": time.time(),
    })
    return record, history, {name: value.detach().cpu().clone() for name, value in selected_state.items()}


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    mixture_config_path = args.mixture_config.expanduser().resolve()
    step7_path = args.step7_config.expanduser().resolve()
    config = baseline.read_json(config_path)
    mixture_cfg = baseline.read_json(mixture_config_path)
    experiment = baseline.read_json(step7_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_STEP10C_TRAINING_OUTCOME":
        raise RuntimeError("unexpected Step-10C training config schema/status")
    if mixture_cfg.get("schema") != step10b.MIXTURE_SCHEMA:
        raise RuntimeError("unexpected Step-10 mixture config schema")
    if experiment.get("schema") != step7.EXPERIMENT_SCHEMA:
        raise RuntimeError("unexpected Step-7 experiment config")
    _assert_step10c_delta_only_horizon(config)

    mixture_product = step10b._resolve_mixture_product(args.mixture_product_dir)
    mixture_complete, bridge_rows, _old_parent_by_root = step10b._verify_mixture_product(
        mixture_product, mixture_cfg
    )
    original_product, original_complete, original_rows = step10b._verify_original_product(
        mixture_product, mixture_cfg, experiment
    )
    outer_product = _resolve_outer_product(args.fresh_outer_product_dir)
    outer_complete, fresh_original_rows, fresh_bridge_rows, fresh_parent_by_root = _verify_outer_product(
        outer_product
    )
    bundle, bundle_complete = step10b._resolve_step9a_bundle(args.step9a_bundle_dir, config)
    seeds = [int(v) for v in config["seeds"]]
    if seeds != [int(v) for v in bundle_complete["seeds"]]:
        raise RuntimeError("Step-10C seeds differ from frozen Step-9A bundle seeds")

    original_fit_roots, original_selection_roots, _spent_original_outer_roots, original_by_root = step7.split_root_indices(
        original_rows, experiment
    )
    bridge_by_root = step10b._by_root(bridge_rows)
    bridge_fit_roots = step10b._bridge_roots(bridge_rows, "fit")
    bridge_selection_roots = step10b._bridge_roots(bridge_rows, "selection")

    fresh_original_by_root = step10b._by_root(fresh_original_rows)
    fresh_bridge_by_root = step10b._by_root(fresh_bridge_rows)
    fresh_original_roots = sorted(fresh_original_by_root)
    fresh_bridge_roots = sorted(fresh_bridge_by_root)

    device = step7.resolve_device(args.device)
    root_batch_size = int(config["training"]["root_batch_size"])
    print("STEP 10C CRASH-SAFE LONG-HORIZON — FRESH OUTER / NO QPU", flush=True)
    print(f"device: {device}", flush=True)
    print(
        f"fit/selection roots original={len(original_fit_roots)}/{len(original_selection_roots)} "
        f"bridge={len(bridge_fit_roots)}/{len(bridge_selection_roots)}",
        flush=True,
    )
    print(
        f"fresh outer roots original={len(fresh_original_roots)} bridge={len(fresh_bridge_roots)}",
        flush=True,
    )

    original_fit = step7.materialize_blocks(
        product=original_product, rows=original_rows, by_root=original_by_root,
        roots=original_fit_roots, root_batch_size=root_batch_size,
        label="step10c-original-fit", progress_every=args.progress_every,
    )
    original_selection = step7.materialize_blocks(
        product=original_product, rows=original_rows, by_root=original_by_root,
        roots=original_selection_roots, root_batch_size=root_batch_size,
        label="step10c-original-selection", progress_every=args.progress_every,
    )
    bridge_fit = step7.materialize_blocks(
        product=mixture_product, rows=bridge_rows, by_root=bridge_by_root,
        roots=bridge_fit_roots, root_batch_size=root_batch_size,
        label="step10c-bridge-fit", progress_every=args.progress_every,
    )
    bridge_selection = step7.materialize_blocks(
        product=mixture_product, rows=bridge_rows, by_root=bridge_by_root,
        roots=bridge_selection_roots, root_batch_size=root_batch_size,
        label="step10c-bridge-selection", progress_every=args.progress_every,
    )
    fresh_original_outer = step7.materialize_blocks(
        product=outer_product, rows=fresh_original_rows, by_root=fresh_original_by_root,
        roots=fresh_original_roots, root_batch_size=root_batch_size,
        label="step10c-fresh-original-outer", progress_every=args.progress_every,
    )
    fresh_bridge_outer = step7.materialize_blocks(
        product=outer_product, rows=fresh_bridge_rows, by_root=fresh_bridge_by_root,
        roots=fresh_bridge_roots, root_batch_size=root_batch_size,
        label="step10c-fresh-bridge-outer", progress_every=args.progress_every,
    )

    effect_class_weight, mechanism_class_weight = step7.class_weights(
        list(original_fit) + list(bridge_fit)
    )
    tolerance = float(config["selection"]["original_domain_retention_tolerance"])
    retention_floors: dict[int, dict[str, float]] = {}
    epoch0_warm_selection: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        model = step10b._initialize_model(
            initialization="warm_start", seed=seed, experiment=experiment,
            device=device, bundle=bundle, config=config,
        )
        summary = step7.selection_summary(step7.predict_blocks(model, original_selection, device))
        epoch0_warm_selection[seed] = summary
        retention_floors[seed] = {
            "mechanism_balanced_accuracy": float(summary["mechanism_balanced_accuracy"]) - tolerance,
            "effect_balanced_accuracy": float(summary["effect_balanced_accuracy"]) - tolerance,
        }
        del model

    benchmark_key_identity = {
        "schema": SCHEMA,
        "training_config_sha256": baseline.sha256_file(config_path),
        "mixture_config_sha256": baseline.sha256_file(mixture_config_path),
        "step7_config_sha256": baseline.sha256_file(step7_path),
        "runner_sha256": baseline.sha256_file(Path(__file__).resolve()),
        "mixture_product_id": mixture_complete["product_id"],
        "original_product_id": original_complete["product_id"],
        "fresh_outer_product_id": outer_complete["product_id"],
        "fresh_outer_dataset_complete_sha256": baseline.sha256_file(outer_product / "dataset_complete.json"),
        "step9a_bundle_id": bundle_complete["bundle_id"],
    }
    benchmark_identity = {
        **benchmark_key_identity,
        "execution_device": str(device),
        "runtime_fingerprint": _runtime_fingerprint(device),
    }
    benchmark_id = "benchmark_" + hashlib.sha256(
        baseline.canonical_json(benchmark_key_identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / benchmark_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite completed Step-10C benchmark {output}")
    work_root = output_parent / f".{benchmark_id}.work"
    work_root.mkdir(parents=True, exist_ok=True)

    selected_records: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    selected_states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    selection_predictions: dict[tuple[str, int, str], step7.PredictionSet] = {}
    outer_predictions: dict[tuple[str, int, str], step7.PredictionSet] = {}
    any_resumed = False

    for initialization in [str(v) for v in config["initializations"]]:
        for seed in seeds:
            print(f"\nTraining Step-10C initialization={initialization} seed={seed}", flush=True)
            run_identity = {
                "schema": SCHEMA,
                "training_config_sha256": benchmark_identity["training_config_sha256"],
                "runner_sha256": benchmark_identity["runner_sha256"],
                "mixture_product_id": mixture_complete["product_id"],
                "fresh_outer_product_id": outer_complete["product_id"],
                "step9a_bundle_id": bundle_complete["bundle_id"],
                "initialization": initialization,
                "seed": int(seed),
                "architecture": "late_concat",
                "trainable_parameter_count": int(config["architecture"]["expected_trainable_parameter_count"]),
                "execution_device": str(device),
                "runtime_fingerprint": _runtime_fingerprint(device),
            }
            record, history, selected_state = _train_seed_crashsafe(
                initialization=initialization, seed=seed, experiment=experiment,
                config=config, bundle=bundle, original_fit=original_fit,
                bridge_fit=bridge_fit, original_selection=original_selection,
                bridge_selection=bridge_selection, floor=retention_floors[seed],
                effect_class_weight=effect_class_weight,
                mechanism_class_weight=mechanism_class_weight, device=device,
                run_dir=work_root / f"{initialization}__seed{seed}",
                identity=run_identity, allow_resume=not args.no_resume,
            )
            key = f"{initialization}__seed{seed}"
            selected_records[key] = record
            histories.extend(history)
            selected_states[(initialization, seed)] = selected_state
            any_resumed = any_resumed or bool(record["resumed_from_checkpoint"])

            model = step10b._initialize_model(
                initialization=initialization, seed=seed, experiment=experiment,
                device=device, bundle=bundle, config=config,
            )
            model.load_state_dict(selected_state, strict=True)
            model.to(device)
            selection_predictions[(initialization, seed, "original")] = step7.predict_blocks(
                model, original_selection, device
            )
            selection_predictions[(initialization, seed, "bridge")] = step7.predict_blocks(
                model, bridge_selection, device
            )
            outer_predictions[(initialization, seed, "original")] = step7.predict_blocks(
                model, fresh_original_outer, device
            )
            outer_predictions[(initialization, seed, "bridge")] = step7.predict_blocks(
                model, fresh_bridge_outer, device
            )
            del model

    baseline_outer_members: list[step7.PredictionSet] = []
    for seed in seeds:
        model = step10b._initialize_model(
            initialization="warm_start", seed=seed, experiment=experiment,
            device=device, bundle=bundle, config=config,
        )
        baseline_outer_members.append(step7.predict_blocks(model, fresh_original_outer, device))
        del model
    step9a_baseline_outer = step10b._mean_prediction(baseline_outer_members)
    step9a_threshold = float(bundle_complete["deployment_effect_threshold"])

    replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    confidence = float(config["evaluation"]["confidence_level"])
    outer_metric_rows: list[dict[str, Any]] = []
    bootstrap_by_domain_task: dict[str, dict[str, dict[str, Mapping[str, np.ndarray]]]] = {
        "original": {"effect_detection": {}, "mechanism_diagnosis": {}, "integrated_diagnosis": {}},
        "bridge": {"effect_detection": {}, "mechanism_diagnosis": {}, "integrated_diagnosis": {}},
    }
    ensemble_predictions: dict[tuple[str, str], step7.PredictionSet] = {}
    ensemble_thresholds: dict[str, float] = {}

    baseline_rows, _ = step10b._bootstrap_domain(
        name="step9a_baseline", prediction=step9a_baseline_outer,
        threshold=step9a_threshold, replicates=replicates,
        seed=bootstrap_seed, confidence=confidence,
    )
    for row in baseline_rows:
        item = dict(row); item["domain"] = "original"; outer_metric_rows.append(item)

    for initialization in [str(v) for v in config["initializations"]]:
        original_sel_ensemble = step10b._mean_prediction(
            [selection_predictions[(initialization, seed, "original")] for seed in seeds]
        )
        bridge_sel_ensemble = step10b._mean_prediction(
            [selection_predictions[(initialization, seed, "bridge")] for seed in seeds]
        )
        threshold = step10b._selection_threshold(original_sel_ensemble, bridge_sel_ensemble)
        ensemble_thresholds[initialization] = threshold
        original_outer_ensemble = step10b._mean_prediction(
            [outer_predictions[(initialization, seed, "original")] for seed in seeds]
        )
        bridge_outer_ensemble = step10b._mean_prediction(
            [outer_predictions[(initialization, seed, "bridge")] for seed in seeds]
        )
        bridge_outer_grouped = step10b._replace_groups(bridge_outer_ensemble, fresh_parent_by_root)
        ensemble_predictions[(initialization, "original")] = original_outer_ensemble
        ensemble_predictions[(initialization, "bridge")] = bridge_outer_grouped
        for domain, prediction in (("original", original_outer_ensemble), ("bridge", bridge_outer_grouped)):
            rows, boots = step10b._bootstrap_domain(
                name=initialization, prediction=prediction, threshold=threshold,
                replicates=replicates, seed=bootstrap_seed, confidence=confidence,
            )
            for row in rows:
                item = dict(row); item["domain"] = domain; outer_metric_rows.append(item)
            for task, boot in boots.items():
                bootstrap_by_domain_task[domain][task][initialization] = boot

    paired_rows: list[dict[str, Any]] = []
    for domain in ("original", "bridge"):
        for task, task_boot in bootstrap_by_domain_task[domain].items():
            rows = baseline.paired_difference_rows(
                task, [("scratch", "warm_start")], task_boot, confidence
            )
            for row in rows:
                item = dict(row); item["domain"] = domain; paired_rows.append(item)

    lookup = {
        (str(row["domain"]), str(row["baseline"]), str(row["task"])): row
        for row in outer_metric_rows
    }
    baseline_old_effect = lookup[("original", "step9a_baseline", "effect_detection")]
    baseline_old_mech = lookup[("original", "step9a_baseline", "mechanism_diagnosis")]
    bridge_gate_cfg = config["evaluation"]["bridge_mechanism_gate"]
    retention_cfg = config["evaluation"]["original_retention_gate"]
    gates: dict[str, dict[str, Any]] = {}
    for initialization in [str(v) for v in config["initializations"]]:
        old_effect = lookup[("original", initialization, "effect_detection")]
        old_mech = lookup[("original", initialization, "mechanism_diagnosis")]
        bridge_mech = lookup[("bridge", initialization, "mechanism_diagnosis")]
        old_effect_drop = float(baseline_old_effect["balanced_accuracy"]) - float(old_effect["balanced_accuracy"])
        old_mech_drop = float(baseline_old_mech["balanced_accuracy"]) - float(old_mech["balanced_accuracy"])
        bridge_pass = bool(
            float(bridge_mech["balanced_accuracy"]) >= float(bridge_gate_cfg["balanced_accuracy_minimum"])
            and float(bridge_mech["balanced_accuracy_ci_low"]) >= float(bridge_gate_cfg["bootstrap_ci_lower_minimum"])
            and step10b._minimum_mechanism_recall(bridge_mech) >= float(bridge_gate_cfg["minimum_class_recall"])
        )
        retention_pass = bool(
            old_effect_drop <= float(retention_cfg["effect_balanced_accuracy_max_drop"])
            and old_mech_drop <= float(retention_cfg["mechanism_balanced_accuracy_max_drop"])
        )
        all_eligible = all(
            bool(selected_records[f"{initialization}__seed{seed}"]["selected_checkpoint_retention_eligible"])
            for seed in seeds
        )
        gates[initialization] = {
            "bridge_gate_pass": bridge_pass,
            "original_retention_gate_pass": retention_pass,
            "all_seed_selected_checkpoints_retention_eligible": all_eligible,
            "full_step10c_gate_pass": bool(bridge_pass and retention_pass and all_eligible),
            "original_effect_ba_drop_vs_step9a": old_effect_drop,
            "original_mechanism_ba_drop_vs_step9a": old_mech_drop,
            "bridge_mechanism_balanced_accuracy": float(bridge_mech["balanced_accuracy"]),
            "bridge_mechanism_ci_low": float(bridge_mech["balanced_accuracy_ci_low"]),
            "bridge_minimum_mechanism_recall": step10b._minimum_mechanism_recall(bridge_mech),
        }

    warm_pass = bool(gates["warm_start"]["full_step10c_gate_pass"])
    scratch_pass = bool(gates["scratch"]["full_step10c_gate_pass"])
    bridge_pair = next((
        row for row in paired_rows
        if row["domain"] == "bridge" and row["task"] == "mechanism_diagnosis"
        and row["metric"] == "balanced_accuracy"
    ), None)
    scratch_clear_advantage = bool(
        scratch_pass and warm_pass and bridge_pair is not None
        and float(bridge_pair["ci_low"]) > 0.01
        and float(gates["scratch"]["original_mechanism_ba_drop_vs_step9a"])
        <= float(gates["warm_start"]["original_mechanism_ba_drop_vs_step9a"]) + 0.01
    )
    if warm_pass and not scratch_clear_advantage:
        decision = "WARM_START_REUSE_PREFERRED_AFTER_DUAL_DOMAIN_GATE"
    elif scratch_pass and (not warm_pass or scratch_clear_advantage):
        decision = "SCRATCH_INITIALIZATION_PREFERRED_BY_FROZEN_COMPARISON"
    else:
        decision = "NO_INITIALIZATION_PASSES_FULL_DUAL_DOMAIN_GATE"

    staging = output_parent / f".{benchmark_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        checkpoint_hashes: dict[str, str] = {}
        for initialization in [str(v) for v in config["initializations"]]:
            for seed in seeds:
                name = f"{initialization}__seed{seed}.pt"
                path = staging / name
                record = selected_records[f"{initialization}__seed{seed}"]
                atomic_torch_save(path, {
                    "schema": SCHEMA,
                    "initialization": initialization,
                    "seed": int(seed),
                    "architecture": "late_concat",
                    "selected_epoch": int(record["selected_epoch"]),
                    "selected_checkpoint_retention_eligible": bool(record["selected_checkpoint_retention_eligible"]),
                    "source_step9a_bundle_id": bundle_complete["bundle_id"] if initialization == "warm_start" else None,
                    "optimizer_state_present": False,
                    "state_dict": selected_states[(initialization, seed)],
                })
                checkpoint_hashes[name] = baseline.sha256_file(path)

        baseline.write_csv(staging / "training_history.csv", histories)
        baseline.write_csv(staging / "outer_domain_metrics.csv", outer_metric_rows)
        baseline.write_csv(staging / "paired_initialization_differences.csv", paired_rows)
        baseline.atomic_json(staging / "model_selection.json", {
            "per_seed": selected_records,
            "warm_start_epoch0_original_selection": {str(seed): epoch0_warm_selection[seed] for seed in seeds},
            "retention_floors": {str(seed): retention_floors[seed] for seed in seeds},
            "ensemble_effect_thresholds": ensemble_thresholds,
        })
        baseline.atomic_json(staging / "decision.json", {
            "schema": SCHEMA,
            "decision": decision,
            "gates": gates,
            "bridge_scratch_minus_warm_start_mechanism_ba_paired": bridge_pair,
            "qpu_executed": False,
            "spent_confirmatory_cohort_accessed": False,
            "step10b_outer_evaluated": False,
            "architecture_changed": False,
            "optimizer_state_resumed_for_crash_recovery": bool(any_resumed),
            "fresh_run_optimizer_state_from_step9a_or_step10b_reused": False,
            "warm_start_state_dict_reused": True,
            "fresh_outer_validation_used_for_selection": False,
        })
        arrays: dict[str, np.ndarray] = {}
        for initialization in [str(v) for v in config["initializations"]]:
            for domain in ("original", "bridge"):
                prediction = ensemble_predictions[(initialization, domain)]
                prefix = f"{initialization}__{domain}"
                arrays[f"{prefix}__effect_truth"] = prediction.effect_truth
                arrays[f"{prefix}__mechanism_truth_all"] = prediction.mechanism_truth_all
                arrays[f"{prefix}__mechanism_mask"] = prediction.mechanism_mask
                arrays[f"{prefix}__effect_logits"] = prediction.effect_logits
                arrays[f"{prefix}__mechanism_logits"] = prediction.mechanism_logits
                arrays[f"{prefix}__bootstrap_group"] = prediction.root_indices
        np.savez_compressed(staging / "outer_predictions.npz", **arrays)

        files = [
            "training_history.csv", "outer_domain_metrics.csv",
            "paired_initialization_differences.csv", "model_selection.json",
            "decision.json", "outer_predictions.npz",
        ] + sorted(checkpoint_hashes)
        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "benchmark_id": benchmark_id,
            "identity": benchmark_identity,
            "decision": decision,
            "mixture_product_id": mixture_complete["product_id"],
            "original_product_id": original_complete["product_id"],
            "fresh_outer_product_id": outer_complete["product_id"],
            "fresh_outer_dataset_complete_sha256": baseline.sha256_file(outer_product / "dataset_complete.json"),
            "step9a_bundle_id": bundle_complete["bundle_id"],
            "architecture": "late_concat",
            "trainable_parameter_count": int(config["architecture"]["expected_trainable_parameter_count"]),
            "max_epochs": int(config["training"]["max_epochs"]),
            "any_run_resumed": bool(any_resumed),
            "qpu_executed": False,
            "step10b_outer_evaluated": False,
            "fresh_outer_used_for_selection": False,
            "file_hashes": {name: baseline.sha256_file(staging / name) for name in files},
        }
        baseline.atomic_json(staging / "benchmark_complete.json", completion)
        os.replace(staging, output)
        _fsync_dir(output_parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 10C CRASH-SAFE LONG-HORIZON COMPLETE\n")
    for initialization in ("warm_start", "scratch"):
        gate = gates[initialization]
        print(
            f"{initialization}: bridge_mech_BA={gate['bridge_mechanism_balanced_accuracy']:.4f} "
            f"CI_low={gate['bridge_mechanism_ci_low']:.4f} "
            f"min_recall={gate['bridge_minimum_mechanism_recall']:.4f} "
            f"old_mech_drop={gate['original_mechanism_ba_drop_vs_step9a']:.4f} "
            f"old_effect_drop={gate['original_effect_ba_drop_vs_step9a']:.4f} "
            f"full_gate={'PASS' if gate['full_step10c_gate_pass'] else 'FAIL'}"
        )
    print(f"DECISION GATE: {decision}")
    print(f"Any crash-recovery resume used: {'YES' if any_resumed else 'NO'}")
    print("Step-10B outer evaluated: NO")
    print("QPU executed: NO")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
