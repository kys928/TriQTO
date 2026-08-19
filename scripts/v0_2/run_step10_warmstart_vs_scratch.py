#!/usr/bin/env python3
"""Step 10B warm-start versus scratch benchmark on the frozen dual-domain mixture."""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7

SCHEMA = "triqto.v0_2.step10_warmstart_vs_scratch.v1"
MIXTURE_SCHEMA = "triqto.v0_2.step10_training_mixture.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10_warmstart_vs_scratch.json"
DEFAULT_MIXTURE_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_MIXTURE_PARENT = Path("/workspace/triqto-data/step10_training_mixture")
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step10_warmstart_vs_scratch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mixture-config", type=Path, default=DEFAULT_MIXTURE_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--mixture-product-dir", type=Path)
    parser.add_argument("--step9a-bundle-dir", type=Path)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def _resolve_mixture_product(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    pointer = DEFAULT_MIXTURE_PARENT / "current_product.json"
    if not pointer.is_file():
        raise RuntimeError(
            "Step-10 mixture current_product.json is missing; run Step 10A first "
            "or pass --mixture-product-dir"
        )
    payload = baseline.read_json(pointer)
    if payload.get("schema") != "triqto.v0_2.step10_current_product.v1":
        raise RuntimeError("unexpected Step-10 current-product pointer schema")
    return Path(payload["product_dir"]).expanduser().resolve()


def _verify_mixture_product(
    product: Path,
    mixture_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], dict[int, int]]:
    complete = baseline.read_json(product / "dataset_complete.json")
    if complete.get("schema") != MIXTURE_SCHEMA or complete.get("status") != "COMPLETE":
        raise RuntimeError("Step-10 mixture product is incomplete or has wrong schema")
    for key, expected in (
        ("bridge_clean_root_count", int(mixture_cfg["bridge"]["expected_clean_roots"])),
        ("bridge_example_count", int(mixture_cfg["bridge"]["expected_examples"])),
        ("bridge_fit_root_count", int(mixture_cfg["bridge"]["parent_split_rule"]["expected_fit_roots"])),
        ("bridge_selection_root_count", int(mixture_cfg["bridge"]["parent_split_rule"]["expected_selection_roots"])),
        ("bridge_outer_validation_root_count", int(mixture_cfg["bridge"]["parent_split_rule"]["expected_outer_validation_roots"])),
    ):
        if int(complete.get(key, -1)) != expected:
            raise RuntimeError(f"Step-10 mixture mismatch for {key}: {complete.get(key)} != {expected}")
    if bool(complete.get("exact_step9d_pilot_graph_present", True)):
        raise RuntimeError("Step-10 mixture unexpectedly contains the exact Step-9D pilot graph")
    manifests = product / "manifests"
    for name, expected_hash in complete.get("manifest_hashes", {}).items():
        if baseline.sha256_file(manifests / name) != expected_hash:
            raise RuntimeError(f"Step-10 mixture manifest hash mismatch: {name}")
    if baseline.sha256_file(product / "stage_validation.json") != complete.get("stage_validation_sha256"):
        raise RuntimeError("Step-10 stage validation hash mismatch")
    rows = baseline.read_csv(manifests / "bridge_example_manifest.csv")
    if len(rows) != int(complete["bridge_example_count"]):
        raise RuntimeError("Step-10 bridge example manifest count mismatch")
    parent_by_root: dict[int, int] = {}
    partition_by_root: dict[int, str] = {}
    for row in rows:
        root = int(row["root_index"])
        parent = int(row["parent_group_index"])
        partition = str(row["step10_partition"])
        if root in parent_by_root and parent_by_root[root] != parent:
            raise RuntimeError("bridge root maps to multiple parents")
        if root in partition_by_root and partition_by_root[root] != partition:
            raise RuntimeError("bridge root crosses partitions")
        parent_by_root[root] = parent
        partition_by_root[root] = partition
    return complete, rows, parent_by_root


def _verify_original_product(
    mixture_product: Path,
    mixture_cfg: Mapping[str, Any],
    step7_cfg: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], list[dict[str, str]]]:
    ref = baseline.read_json(mixture_product / "original_domain_reference.json")
    original = mixture_cfg["original_domain"]
    if ref.get("product_id") != original["product_id"]:
        raise RuntimeError("Step-10 original reference product mismatch")
    product = Path(ref["product_dir"]).expanduser().resolve()
    complete, rows = baseline.verify_source_product(product, step7_cfg)
    if complete["product_id"] != original["product_id"]:
        raise RuntimeError("verified original product does not match frozen Step-10 source")
    if baseline.sha256_file(product / "dataset_complete.json") != ref["dataset_complete_sha256"]:
        raise RuntimeError("original-domain dataset_complete hash changed")
    return product, complete, rows


def _by_root(rows: Sequence[Mapping[str, str]]) -> dict[int, list[int]]:
    output: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        output[int(row["root_index"])].append(index)
    return dict(output)


def _bridge_roots(rows: Sequence[Mapping[str, str]], partition: str) -> list[int]:
    by: dict[int, str] = {}
    for row in rows:
        root = int(row["root_index"])
        value = str(row["step10_partition"])
        if root in by and by[root] != value:
            raise RuntimeError("bridge root has inconsistent partition")
        by[root] = value
    return sorted(root for root, value in by.items() if value == partition)


def _resolve_step9a_bundle(
    explicit: Path | None,
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if explicit is not None:
        bundle = explicit.expanduser().resolve()
        candidates = [bundle]
    else:
        parent = Path(config["warm_start"]["default_bundle_parent"]).expanduser().resolve()
        candidates = sorted(
            path
            for path in parent.glob("deploy_*")
            if path.is_dir() and (path / "bundle_complete.json").is_file()
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one frozen Step-9A deployment bundle, found {len(candidates)}"
        )
    bundle = candidates[0]
    complete = baseline.read_json(bundle / "bundle_complete.json")
    if complete.get("status") != "DEPLOYMENT_REFIT_BUNDLE_FROZEN":
        raise RuntimeError("Step-9A bundle is not frozen")
    if complete.get("architecture") != "late_concat":
        raise RuntimeError("Step-9A warm-start architecture is not late_concat")
    for name, expected_hash in complete.get("file_hashes", {}).items():
        if baseline.sha256_file(bundle / name) != expected_hash:
            raise RuntimeError(f"Step-9A bundle file hash mismatch: {name}")
    return bundle, complete


def _load_warm_state(
    *,
    bundle: Path,
    seed: int,
    config: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    name = str(config["warm_start"]["checkpoint_names"][str(seed)])
    payload = torch.load(bundle / name, map_location="cpu", weights_only=False)
    if int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"Step-9A checkpoint seed mismatch for {name}")
    if payload.get("architecture") != "late_concat":
        raise RuntimeError(f"Step-9A checkpoint architecture mismatch for {name}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Step-9A checkpoint missing state_dict: {name}")
    return state


def _replace_groups(
    prediction: step7.PredictionSet,
    group_by_root: Mapping[int, int],
) -> step7.PredictionSet:
    groups = np.asarray(
        [int(group_by_root[int(root)]) for root in prediction.root_indices],
        dtype=np.int64,
    )
    return step7.PredictionSet(
        source_indices=prediction.source_indices.copy(),
        root_indices=groups,
        effect_truth=prediction.effect_truth.copy(),
        mechanism_truth_all=prediction.mechanism_truth_all.copy(),
        mechanism_mask=prediction.mechanism_mask.copy(),
        effect_logits=prediction.effect_logits.copy(),
        mechanism_logits=prediction.mechanism_logits.copy(),
    )


def _mean_prediction(
    predictions: Sequence[step7.PredictionSet],
) -> step7.PredictionSet:
    if not predictions:
        raise ValueError("cannot aggregate empty prediction list")
    ref = predictions[0]
    for index, pred in enumerate(predictions[1:], start=1):
        step7.assert_prediction_alignment(ref, pred, f"Step-10 ensemble member {index}")
    return step7.PredictionSet(
        source_indices=ref.source_indices.copy(),
        root_indices=ref.root_indices.copy(),
        effect_truth=ref.effect_truth.copy(),
        mechanism_truth_all=ref.mechanism_truth_all.copy(),
        mechanism_mask=ref.mechanism_mask.copy(),
        effect_logits=np.mean(
            np.stack([pred.effect_logits for pred in predictions]), axis=0
        ).astype(np.float32),
        mechanism_logits=np.mean(
            np.stack([pred.mechanism_logits for pred in predictions]), axis=0
        ).astype(np.float32),
    )


def _selection_threshold(
    original: step7.PredictionSet,
    bridge: step7.PredictionSet,
) -> float:
    truth = np.concatenate([original.effect_truth, bridge.effect_truth])
    logits = np.concatenate([original.effect_logits, bridge.effect_logits])
    threshold, _ = baseline.select_binary_threshold(
        truth.astype(np.int8), logits.astype(np.float64)
    )
    return float(threshold)


def _candidate_key(
    original_summary: Mapping[str, Any],
    bridge_summary: Mapping[str, Any],
    epoch: int,
) -> tuple[float, float, float, float]:
    original_mech = float(original_summary["mechanism_balanced_accuracy"])
    bridge_mech = float(bridge_summary["mechanism_balanced_accuracy"])
    return (
        bridge_mech,
        min(original_mech, bridge_mech),
        float(bridge_summary["effect_balanced_accuracy"]),
        -float(epoch),
    )


def _better_key(
    candidate: tuple[float, float, float, float],
    best: tuple[float, float, float, float] | None,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if candidate[0] > best[0] + min_delta:
        return True
    if abs(candidate[0] - best[0]) <= min_delta:
        return candidate[1:] > best[1:]
    return False


def _domain_balanced_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    original_blocks: Sequence[step7.CachedBlock],
    bridge_blocks: Sequence[step7.CachedBlock],
    *,
    seed: int,
    epoch: int,
    device: torch.device,
    effect_class_weight: np.ndarray,
    mechanism_class_weight: np.ndarray,
    training: Mapping[str, Any],
) -> dict[str, float]:
    if not original_blocks or not bridge_blocks:
        raise RuntimeError("Step-10 fit blocks missing a domain")
    old_order = np.arange(len(original_blocks), dtype=np.int64)
    bridge_order = np.arange(len(bridge_blocks), dtype=np.int64)
    np.random.default_rng(seed * 1_000_003 + epoch * 97_409 + 11).shuffle(old_order)
    np.random.default_rng(seed * 1_000_003 + epoch * 97_409 + 29).shuffle(bridge_order)
    steps_per_domain = max(len(old_order), len(bridge_order))
    totals: dict[str, float] = defaultdict(float)
    optimizer_steps = 0

    domains = ("original", "bridge") if epoch % 2 else ("bridge", "original")
    for step_index in range(steps_per_domain):
        for domain in domains:
            if domain == "original":
                block = original_blocks[int(old_order[step_index % len(old_order)])]
                local_seed = seed + 101
            else:
                block = bridge_blocks[int(bridge_order[step_index % len(bridge_order)])]
                local_seed = seed + 211
            metrics = step7.train_one_epoch(
                model,
                optimizer,
                [block],
                seed=local_seed,
                epoch=epoch,
                device=device,
                effect_class_weight=effect_class_weight,
                mechanism_class_weight=mechanism_class_weight,
                effect_loss_weight=float(training["effect_loss_weight"]),
                mechanism_loss_weight=float(training["mechanism_loss_weight"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            for name in (
                "effect_loss",
                "mechanism_loss",
                "total_loss",
                "mean_preclip_gradient_norm",
            ):
                totals[name] += float(metrics[name])
            optimizer_steps += int(metrics["optimizer_steps"])

    denominator = float(2 * steps_per_domain)
    return {
        "effect_loss": totals["effect_loss"] / denominator,
        "mechanism_loss": totals["mechanism_loss"] / denominator,
        "total_loss": totals["total_loss"] / denominator,
        "mean_preclip_gradient_norm": totals["mean_preclip_gradient_norm"] / denominator,
        "optimizer_steps": float(optimizer_steps),
        "optimizer_blocks_per_domain": float(steps_per_domain),
    }


def _initialize_model(
    *,
    initialization: str,
    seed: int,
    experiment: Mapping[str, Any],
    device: torch.device,
    bundle: Path,
    config: Mapping[str, Any],
) -> torch.nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = step7.instantiate_model("late_concat", seed, experiment, device)
    if initialization == "warm_start":
        state = _load_warm_state(bundle=bundle, seed=seed, config=config)
        model.load_state_dict(dict(state), strict=True)
    elif initialization != "scratch":
        raise ValueError(f"unknown Step-10 initialization {initialization!r}")
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count != int(config["architecture"]["expected_trainable_parameter_count"]):
        raise RuntimeError(f"Step-10 trainable parameter count changed: {count}")
    return model


def _eligible(
    original_summary: Mapping[str, Any],
    floor: Mapping[str, float],
) -> bool:
    return bool(
        float(original_summary["mechanism_balanced_accuracy"])
        >= float(floor["mechanism_balanced_accuracy"])
        and float(original_summary["effect_balanced_accuracy"])
        >= float(floor["effect_balanced_accuracy"])
    )


def _train_seed(
    *,
    initialization: str,
    seed: int,
    experiment: Mapping[str, Any],
    config: Mapping[str, Any],
    bundle: Path,
    original_fit: Sequence[step7.CachedBlock],
    bridge_fit: Sequence[step7.CachedBlock],
    original_selection: Sequence[step7.CachedBlock],
    bridge_selection: Sequence[step7.CachedBlock],
    floor: Mapping[str, float],
    effect_class_weight: np.ndarray,
    mechanism_class_weight: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.nn.Module]:
    model = _initialize_model(
        initialization=initialization,
        seed=seed,
        experiment=experiment,
        device=device,
        bundle=bundle,
        config=config,
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
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

    def consider(epoch: int, train_metrics: Mapping[str, float] | None) -> tuple[bool, dict[str, Any], dict[str, Any]]:
        nonlocal best_eligible_key, best_eligible_state, best_eligible_epoch
        nonlocal best_unconstrained_key, best_unconstrained_state, best_unconstrained_epoch, stale
        original_pred = step7.predict_blocks(model, original_selection, device)
        bridge_pred = step7.predict_blocks(model, bridge_selection, device)
        original_summary = step7.selection_summary(original_pred)
        bridge_summary = step7.selection_summary(bridge_pred)
        eligible = _eligible(original_summary, floor)
        key = _candidate_key(original_summary, bridge_summary, epoch)
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        if _better_key(key, best_unconstrained_key, min_delta):
            best_unconstrained_key = key
            best_unconstrained_state = state
            best_unconstrained_epoch = epoch

        improved = False
        if eligible and _better_key(key, best_eligible_key, min_delta):
            best_eligible_key = key
            best_eligible_state = state
            best_eligible_epoch = epoch
            stale = 0
            improved = True
        elif best_eligible_state is not None:
            stale += 1

        row: dict[str, Any] = {
            "initialization": initialization,
            "seed": seed,
            "epoch": epoch,
            "selection_eligible": eligible,
            "original_selection_effect_ba": float(original_summary["effect_balanced_accuracy"]),
            "original_selection_mechanism_ba": float(original_summary["mechanism_balanced_accuracy"]),
            "bridge_selection_effect_ba": float(bridge_summary["effect_balanced_accuracy"]),
            "bridge_selection_mechanism_ba": float(bridge_summary["mechanism_balanced_accuracy"]),
            "retention_floor_effect_ba": float(floor["effect_balanced_accuracy"]),
            "retention_floor_mechanism_ba": float(floor["mechanism_balanced_accuracy"]),
            "eligible_checkpoint_improved": improved,
        }
        if train_metrics is not None:
            row.update({f"train_{key}": float(value) for key, value in train_metrics.items()})
        history.append(row)
        print(
            f"{initialization} seed={seed} epoch={epoch:02d} "
            f"old_mech={row['original_selection_mechanism_ba']:.4f} "
            f"bridge_mech={row['bridge_selection_mechanism_ba']:.4f} "
            f"eligible={'YES' if eligible else 'NO'}",
            flush=True,
        )
        return eligible, original_summary, bridge_summary

    consider(0, None)

    for epoch in range(1, max_epochs + 1):
        train_metrics = _domain_balanced_epoch(
            model,
            optimizer,
            original_fit,
            bridge_fit,
            seed=seed,
            epoch=epoch,
            device=device,
            effect_class_weight=effect_class_weight,
            mechanism_class_weight=mechanism_class_weight,
            training=training,
        )
        consider(epoch, train_metrics)
        if best_eligible_state is not None and stale >= patience:
            break

    selected_eligible = best_eligible_state is not None
    selected_state = best_eligible_state if selected_eligible else best_unconstrained_state
    selected_epoch = best_eligible_epoch if selected_eligible else best_unconstrained_epoch
    if selected_state is None or selected_epoch is None:
        raise RuntimeError(f"Step-10 {initialization} seed={seed} failed to select any state")
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
        "original_selection": step7.selection_summary(original_selected),
        "bridge_selection": step7.selection_summary(bridge_selected),
        "retention_floor": dict(floor),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    return record, history, model


def _bootstrap_domain(
    *,
    name: str,
    prediction: step7.PredictionSet,
    threshold: float,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, np.ndarray]]]:
    rows, boots, _ = step7.bootstrap_rows(
        name=name,
        truth=prediction,
        effect_logits=prediction.effect_logits,
        mechanism_logits=prediction.mechanism_logits,
        threshold=threshold,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    return rows, boots


def _minimum_mechanism_recall(row: Mapping[str, Any]) -> float:
    return min(
        float(row["recall__rz_drift"]),
        float(row["recall__rx_overrotation"]),
        float(row["recall__ry_overrotation"]),
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    mixture_config_path = args.mixture_config.expanduser().resolve()
    step7_path = args.step7_config.expanduser().resolve()
    config = baseline.read_json(config_path)
    mixture_cfg = baseline.read_json(mixture_config_path)
    experiment = baseline.read_json(step7_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_STEP10_TRAINING_OUTCOME":
        raise RuntimeError("unexpected Step-10 training config schema/status")
    if mixture_cfg.get("schema") != MIXTURE_SCHEMA:
        raise RuntimeError("unexpected Step-10 mixture config schema")
    if experiment.get("schema") != step7.EXPERIMENT_SCHEMA:
        raise RuntimeError("unexpected Step-7 experiment config")

    mixture_product = _resolve_mixture_product(args.mixture_product_dir)
    mixture_complete, bridge_rows, parent_by_root = _verify_mixture_product(
        mixture_product, mixture_cfg
    )
    original_product, original_complete, original_rows = _verify_original_product(
        mixture_product, mixture_cfg, experiment
    )
    bundle, bundle_complete = _resolve_step9a_bundle(args.step9a_bundle_dir, config)
    seeds = [int(v) for v in config["seeds"]]
    if seeds != [int(v) for v in bundle_complete["seeds"]]:
        raise RuntimeError("Step-10 seeds differ from frozen Step-9A bundle seeds")

    original_fit_roots, original_selection_roots, original_outer_roots, original_by_root = step7.split_root_indices(
        original_rows, experiment
    )
    bridge_by_root = _by_root(bridge_rows)
    bridge_fit_roots = _bridge_roots(bridge_rows, "fit")
    bridge_selection_roots = _bridge_roots(bridge_rows, "selection")
    bridge_outer_roots = _bridge_roots(bridge_rows, "outer_validation")

    device = step7.resolve_device(args.device)
    root_batch_size = int(config["training"]["root_batch_size"])
    print("STEP 10B WARM-START VS SCRATCH — NO QPU / SAME ARCHITECTURE", flush=True)
    print(f"device: {device}", flush=True)
    print(
        "roots original fit/sel/outer="
        f"{len(original_fit_roots)}/{len(original_selection_roots)}/{len(original_outer_roots)} "
        "bridge fit/sel/outer="
        f"{len(bridge_fit_roots)}/{len(bridge_selection_roots)}/{len(bridge_outer_roots)}",
        flush=True,
    )

    original_fit = step7.materialize_blocks(
        product=original_product,
        rows=original_rows,
        by_root=original_by_root,
        roots=original_fit_roots,
        root_batch_size=root_batch_size,
        label="step10-original-fit",
        progress_every=args.progress_every,
    )
    original_selection = step7.materialize_blocks(
        product=original_product,
        rows=original_rows,
        by_root=original_by_root,
        roots=original_selection_roots,
        root_batch_size=root_batch_size,
        label="step10-original-selection",
        progress_every=args.progress_every,
    )
    original_outer = step7.materialize_blocks(
        product=original_product,
        rows=original_rows,
        by_root=original_by_root,
        roots=original_outer_roots,
        root_batch_size=root_batch_size,
        label="step10-original-outer",
        progress_every=args.progress_every,
    )
    bridge_fit = step7.materialize_blocks(
        product=mixture_product,
        rows=bridge_rows,
        by_root=bridge_by_root,
        roots=bridge_fit_roots,
        root_batch_size=root_batch_size,
        label="step10-bridge-fit",
        progress_every=args.progress_every,
    )
    bridge_selection = step7.materialize_blocks(
        product=mixture_product,
        rows=bridge_rows,
        by_root=bridge_by_root,
        roots=bridge_selection_roots,
        root_batch_size=root_batch_size,
        label="step10-bridge-selection",
        progress_every=args.progress_every,
    )
    bridge_outer = step7.materialize_blocks(
        product=mixture_product,
        rows=bridge_rows,
        by_root=bridge_by_root,
        roots=bridge_outer_roots,
        root_batch_size=root_batch_size,
        label="step10-bridge-outer",
        progress_every=args.progress_every,
    )

    effect_class_weight, mechanism_class_weight = step7.class_weights(
        list(original_fit) + list(bridge_fit)
    )

    tolerance = float(config["selection"]["original_domain_retention_tolerance"])
    retention_floors: dict[int, dict[str, float]] = {}
    epoch0_warm_selection: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        model = _initialize_model(
            initialization="warm_start",
            seed=seed,
            experiment=experiment,
            device=device,
            bundle=bundle,
            config=config,
        )
        prediction = step7.predict_blocks(model, original_selection, device)
        summary = step7.selection_summary(prediction)
        epoch0_warm_selection[seed] = summary
        retention_floors[seed] = {
            "mechanism_balanced_accuracy": float(summary["mechanism_balanced_accuracy"]) - tolerance,
            "effect_balanced_accuracy": float(summary["effect_balanced_accuracy"]) - tolerance,
        }
        del model

    selected_records: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    selected_models: dict[tuple[str, int], torch.nn.Module] = {}
    selection_predictions: dict[tuple[str, int, str], step7.PredictionSet] = {}
    outer_predictions: dict[tuple[str, int, str], step7.PredictionSet] = {}

    for initialization in [str(v) for v in config["initializations"]]:
        for seed in seeds:
            print(f"\nTraining Step-10 initialization={initialization} seed={seed}", flush=True)
            record, history, model = _train_seed(
                initialization=initialization,
                seed=seed,
                experiment=experiment,
                config=config,
                bundle=bundle,
                original_fit=original_fit,
                bridge_fit=bridge_fit,
                original_selection=original_selection,
                bridge_selection=bridge_selection,
                floor=retention_floors[seed],
                effect_class_weight=effect_class_weight,
                mechanism_class_weight=mechanism_class_weight,
                device=device,
            )
            key = f"{initialization}__seed{seed}"
            selected_records[key] = record
            histories.extend(history)
            selected_models[(initialization, seed)] = model
            selection_predictions[(initialization, seed, "original")] = step7.predict_blocks(
                model, original_selection, device
            )
            selection_predictions[(initialization, seed, "bridge")] = step7.predict_blocks(
                model, bridge_selection, device
            )
            outer_predictions[(initialization, seed, "original")] = step7.predict_blocks(
                model, original_outer, device
            )
            outer_predictions[(initialization, seed, "bridge")] = step7.predict_blocks(
                model, bridge_outer, device
            )

    baseline_outer_members: list[step7.PredictionSet] = []
    for seed in seeds:
        model = _initialize_model(
            initialization="warm_start",
            seed=seed,
            experiment=experiment,
            device=device,
            bundle=bundle,
            config=config,
        )
        baseline_outer_members.append(step7.predict_blocks(model, original_outer, device))
        del model
    step9a_baseline_outer = _mean_prediction(baseline_outer_members)
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

    baseline_rows, _baseline_boot = _bootstrap_domain(
        name="step9a_baseline",
        prediction=step9a_baseline_outer,
        threshold=step9a_threshold,
        replicates=replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    for row in baseline_rows:
        item = dict(row)
        item["domain"] = "original"
        outer_metric_rows.append(item)

    for initialization in [str(v) for v in config["initializations"]]:
        original_selection_ensemble = _mean_prediction(
            [selection_predictions[(initialization, seed, "original")] for seed in seeds]
        )
        bridge_selection_ensemble = _mean_prediction(
            [selection_predictions[(initialization, seed, "bridge")] for seed in seeds]
        )
        threshold = _selection_threshold(
            original_selection_ensemble, bridge_selection_ensemble
        )
        ensemble_thresholds[initialization] = threshold

        original_outer_ensemble = _mean_prediction(
            [outer_predictions[(initialization, seed, "original")] for seed in seeds]
        )
        bridge_outer_ensemble = _mean_prediction(
            [outer_predictions[(initialization, seed, "bridge")] for seed in seeds]
        )
        bridge_outer_grouped = _replace_groups(bridge_outer_ensemble, parent_by_root)
        ensemble_predictions[(initialization, "original")] = original_outer_ensemble
        ensemble_predictions[(initialization, "bridge")] = bridge_outer_grouped

        for domain, prediction in (
            ("original", original_outer_ensemble),
            ("bridge", bridge_outer_grouped),
        ):
            rows, boots = _bootstrap_domain(
                name=initialization,
                prediction=prediction,
                threshold=threshold,
                replicates=replicates,
                seed=bootstrap_seed,
                confidence=confidence,
            )
            for row in rows:
                item = dict(row)
                item["domain"] = domain
                outer_metric_rows.append(item)
            for task, boot in boots.items():
                bootstrap_by_domain_task[domain][task][initialization] = boot

    paired_rows: list[dict[str, Any]] = []
    for domain in ("original", "bridge"):
        for task, task_boot in bootstrap_by_domain_task[domain].items():
            rows = baseline.paired_difference_rows(
                task,
                [("scratch", "warm_start")],
                task_boot,
                confidence,
            )
            for row in rows:
                item = dict(row)
                item["domain"] = domain
                paired_rows.append(item)

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
            float(bridge_mech["balanced_accuracy"])
            >= float(bridge_gate_cfg["balanced_accuracy_minimum"])
            and float(bridge_mech["balanced_accuracy_ci_low"])
            >= float(bridge_gate_cfg["bootstrap_ci_lower_minimum"])
            and _minimum_mechanism_recall(bridge_mech)
            >= float(bridge_gate_cfg["minimum_class_recall"])
        )
        retention_pass = bool(
            old_effect_drop <= float(retention_cfg["effect_balanced_accuracy_max_drop"])
            and old_mech_drop <= float(retention_cfg["mechanism_balanced_accuracy_max_drop"])
        )
        all_seed_checkpoints_eligible = all(
            bool(selected_records[f"{initialization}__seed{seed}"]["selected_checkpoint_retention_eligible"])
            for seed in seeds
        )
        gates[initialization] = {
            "bridge_gate_pass": bridge_pass,
            "original_retention_gate_pass": retention_pass,
            "all_seed_selected_checkpoints_retention_eligible": all_seed_checkpoints_eligible,
            "full_step10_gate_pass": bool(
                bridge_pass and retention_pass and all_seed_checkpoints_eligible
            ),
            "original_effect_ba_drop_vs_step9a": old_effect_drop,
            "original_mechanism_ba_drop_vs_step9a": old_mech_drop,
            "bridge_mechanism_balanced_accuracy": float(bridge_mech["balanced_accuracy"]),
            "bridge_mechanism_ci_low": float(bridge_mech["balanced_accuracy_ci_low"]),
            "bridge_minimum_mechanism_recall": _minimum_mechanism_recall(bridge_mech),
        }

    warm_pass = bool(gates["warm_start"]["full_step10_gate_pass"])
    scratch_pass = bool(gates["scratch"]["full_step10_gate_pass"])
    bridge_pair = next(
        (
            row
            for row in paired_rows
            if row["domain"] == "bridge"
            and row["task"] == "mechanism_diagnosis"
            and row["metric"] == "balanced_accuracy"
        ),
        None,
    )
    scratch_clear_advantage = bool(
        scratch_pass
        and warm_pass
        and bridge_pair is not None
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

    identity = {
        "schema": SCHEMA,
        "training_config_sha256": baseline.sha256_file(config_path),
        "mixture_config_sha256": baseline.sha256_file(mixture_config_path),
        "step7_config_sha256": baseline.sha256_file(step7_path),
        "runner_sha256": baseline.sha256_file(Path(__file__).resolve()),
        "mixture_product_id": mixture_complete["product_id"],
        "original_product_id": original_complete["product_id"],
        "step9a_bundle_id": bundle_complete["bundle_id"],
    }
    benchmark_id = "benchmark_" + hashlib.sha256(
        baseline.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / benchmark_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing Step-10 benchmark {output}")
    staging = output_parent / f".{benchmark_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        checkpoint_hashes: dict[str, str] = {}
        for initialization in [str(v) for v in config["initializations"]]:
            for seed in seeds:
                name = f"{initialization}__seed{seed}.pt"
                path = staging / name
                record = selected_records[f"{initialization}__seed{seed}"]
                torch.save(
                    {
                        "schema": SCHEMA,
                        "initialization": initialization,
                        "seed": int(seed),
                        "architecture": "late_concat",
                        "selected_epoch": int(record["selected_epoch"]),
                        "selected_checkpoint_retention_eligible": bool(
                            record["selected_checkpoint_retention_eligible"]
                        ),
                        "source_step9a_bundle_id": bundle_complete["bundle_id"]
                        if initialization == "warm_start"
                        else None,
                        "optimizer_state_present": False,
                        "state_dict": {
                            key: value.detach().cpu().clone()
                            for key, value in selected_models[(initialization, seed)].state_dict().items()
                        },
                    },
                    path,
                )
                checkpoint_hashes[name] = baseline.sha256_file(path)

        baseline.write_csv(staging / "training_history.csv", histories)
        baseline.write_csv(staging / "outer_domain_metrics.csv", outer_metric_rows)
        baseline.write_csv(staging / "paired_initialization_differences.csv", paired_rows)
        baseline.atomic_json(
            staging / "model_selection.json",
            {
                "per_seed": selected_records,
                "warm_start_epoch0_original_selection": {
                    str(seed): epoch0_warm_selection[seed] for seed in seeds
                },
                "retention_floors": {
                    str(seed): retention_floors[seed] for seed in seeds
                },
                "ensemble_effect_thresholds": ensemble_thresholds,
            },
        )
        baseline.atomic_json(
            staging / "decision.json",
            {
                "schema": SCHEMA,
                "decision": decision,
                "gates": gates,
                "bridge_scratch_minus_warm_start_mechanism_ba_paired": bridge_pair,
                "qpu_executed": False,
                "spent_confirmatory_cohort_accessed": False,
                "architecture_changed": False,
                "optimizer_state_resumed": False,
                "warm_start_state_dict_reused": True,
                "outer_validation_used_for_selection": False,
            },
        )

        payload: dict[str, np.ndarray] = {}
        for initialization in [str(v) for v in config["initializations"]]:
            for domain in ("original", "bridge"):
                prediction = ensemble_predictions[(initialization, domain)]
                prefix = f"{initialization}__{domain}"
                payload[f"{prefix}__effect_truth"] = prediction.effect_truth
                payload[f"{prefix}__mechanism_truth_all"] = prediction.mechanism_truth_all
                payload[f"{prefix}__mechanism_mask"] = prediction.mechanism_mask
                payload[f"{prefix}__effect_logits"] = prediction.effect_logits
                payload[f"{prefix}__mechanism_logits"] = prediction.mechanism_logits
                payload[f"{prefix}__bootstrap_group"] = prediction.root_indices
        np.savez_compressed(staging / "outer_predictions.npz", **payload)

        files = [
            "training_history.csv",
            "outer_domain_metrics.csv",
            "paired_initialization_differences.csv",
            "model_selection.json",
            "decision.json",
            "outer_predictions.npz",
        ] + sorted(checkpoint_hashes)
        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "benchmark_id": benchmark_id,
            "identity": identity,
            "decision": decision,
            "mixture_product_id": mixture_complete["product_id"],
            "original_product_id": original_complete["product_id"],
            "step9a_bundle_id": bundle_complete["bundle_id"],
            "architecture": "late_concat",
            "trainable_parameter_count_each": int(
                config["architecture"]["expected_trainable_parameter_count"]
            ),
            "seeds": seeds,
            "initializations": list(config["initializations"]),
            "checkpoint_hashes": checkpoint_hashes,
            "qpu_executed": False,
            "architecture_changed": False,
            "optimizer_state_resumed": False,
            "outer_validation_used_for_selection": False,
            "file_hashes": {
                name: baseline.sha256_file(staging / name) for name in files
            },
        }
        baseline.atomic_json(staging / "benchmark_complete.json", completion)
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 10B WARM-START VS SCRATCH COMPLETE\n")
    for initialization in ("warm_start", "scratch"):
        gate = gates[initialization]
        print(
            f"{initialization}: bridge_mech_BA="
            f"{gate['bridge_mechanism_balanced_accuracy']:.4f} "
            f"CI_low={gate['bridge_mechanism_ci_low']:.4f} "
            f"min_recall={gate['bridge_minimum_mechanism_recall']:.4f} "
            f"old_mech_drop={gate['original_mechanism_ba_drop_vs_step9a']:.4f} "
            f"old_effect_drop={gate['original_effect_ba_drop_vs_step9a']:.4f} "
            f"full_gate={'PASS' if gate['full_step10_gate_pass'] else 'FAIL'}"
        )
    print(f"DECISION GATE: {decision}")
    print("QPU executed: NO")
    print("Architecture changed: NO")
    print("Optimizer state resumed: NO")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
