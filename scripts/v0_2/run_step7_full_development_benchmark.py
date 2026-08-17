#!/usr/bin/env python3
"""Run the frozen Step 7 full neural development benchmark."""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import benchmark_step6_cheap_baselines as baseline
import smoke_step7_structured_diagnostic_model as smoke_runner
from triqto.model.contracts import GraphTensorBatch
from triqto.step7.contracts import DiagnosticTensorBatch, Step7ModelBatch, Step7Targets
from triqto.step7.graph_adapter import batch_from_step5_examples
from triqto.step7.model import Step7DiagnosticModel


EXECUTION_SCHEMA = "triqto.v0_2.step7_full_development_benchmark.v1"
EXPERIMENT_SCHEMA = "triqto.v0_2.step7_structured_diagnostic_model.v1"
DEFAULT_EXECUTION_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_full_development_benchmark.json"
DEFAULT_EXPERIMENT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"


@dataclass(slots=True)
class CachedBlock:
    root_indices: np.ndarray
    source_indices: np.ndarray
    batch: Step7ModelBatch
    targets: Step7Targets


@dataclass(slots=True)
class PredictionSet:
    source_indices: np.ndarray
    root_indices: np.ndarray
    effect_truth: np.ndarray
    mechanism_truth_all: np.ndarray
    mechanism_mask: np.ndarray
    effect_logits: np.ndarray
    mechanism_logits: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXECUTION_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--step6a-dir", type=Path)
    parser.add_argument("--step6b-dir", type=Path)
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _raw_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def verify_smoke(smoke_dir: Path, execution: Mapping[str, Any], experiment_path: Path) -> dict[str, Any]:
    expected = execution["required_smoke"]
    complete_path = smoke_dir / "smoke_complete.json"
    if not complete_path.is_file():
        raise RuntimeError(f"missing required Step 7 smoke marker: {complete_path}")
    complete = baseline.read_json(complete_path)
    if complete.get("smoke_id") != expected["smoke_id"]:
        raise RuntimeError("Step 7 smoke ID mismatch")
    if complete.get("decision") != expected["decision"] or bool(complete.get("scientific_metric_claim")):
        raise RuntimeError("Step 7 smoke did not satisfy the frozen non-scientific pass gate")
    smoke_product = complete.get("source_product_id", complete.get("identity", {}).get("source_product_id"))
    if smoke_product != execution["source_product_id"]:
        raise RuntimeError("Step 7 smoke source product mismatch")
    recorded_experiment = complete.get("identity", {}).get("experiment_config_sha256")
    actual_experiment = baseline.sha256_file(experiment_path)
    if recorded_experiment != actual_experiment or actual_experiment != execution["experiment_config"]["sha256"]:
        raise RuntimeError("Step 7 experiment config no longer matches the accepted smoke")
    uploaded_hash = expected.get("uploaded_smoke_complete_sha256")
    if uploaded_hash and _raw_sha256(complete_path) != uploaded_hash:
        raise RuntimeError("local smoke_complete.json does not match the independently archived uploaded marker")
    return complete


def split_root_indices(rows: Sequence[dict[str, str]], experiment: Mapping[str, Any]) -> tuple[list[int], list[int], list[int], dict[int, list[int]]]:
    by_root: dict[int, list[int]] = defaultdict(list)
    split_by_root: dict[int, str] = {}
    occurrence_by_root: dict[int, int] = {}
    for source_index, row in enumerate(rows):
        root = int(row["root_index"])
        by_root[root].append(source_index)
        split_by_root[root] = row["split"]
        occurrence_by_root[root] = int(row["family_occurrence_index"])
    for root, indices in by_root.items():
        if len(indices) != 13:
            raise RuntimeError(f"Step 7 root {root} has {len(indices)} examples instead of 13")
        if len({rows[index]["split"] for index in indices}) != 1:
            raise RuntimeError(f"Step 7 root {root} crosses source splits")
    fit = sorted(root for root in by_root if split_by_root[root] == "train" and occurrence_by_root[root] % 5 in {1, 2, 3})
    selection = sorted(root for root in by_root if split_by_root[root] == "train" and occurrence_by_root[root] % 5 == 4)
    outer = sorted(root for root in by_root if split_by_root[root] == "validation")
    expected = experiment["development_split"]
    for values, wanted, name in (
        (fit, int(expected["expected_fit_roots"]), "fit"),
        (selection, int(expected["expected_selection_roots"]), "selection"),
        (outer, int(expected["expected_outer_validation_roots"]), "outer validation"),
    ):
        if len(values) != wanted:
            raise RuntimeError(f"Step 7 {name} root count {len(values)} != frozen {wanted}")
    if any(occurrence_by_root[root] % 5 != 0 for root in outer):
        raise RuntimeError("outer development validation does not satisfy frozen family-occurrence rule")
    if set(fit) & set(selection) or set(fit) & set(outer) or set(selection) & set(outer):
        raise RuntimeError("Step 7 development root partitions overlap")
    return fit, selection, outer, by_root


def root_blocks(roots: Sequence[int], root_batch_size: int) -> list[list[int]]:
    if root_batch_size <= 0:
        raise ValueError("root_batch_size must be positive")
    ordered = list(roots)
    return [ordered[start:start + root_batch_size] for start in range(0, len(ordered), root_batch_size)]


def materialize_blocks(*, product: Path, rows: Sequence[dict[str, str]], by_root: Mapping[int, Sequence[int]], roots: Sequence[int], root_batch_size: int, label: str, progress_every: int) -> list[CachedBlock]:
    blocks: list[CachedBlock] = []
    touched = 0
    next_progress = max(1, progress_every)
    for root_group in root_blocks(roots, root_batch_size):
        source_indices: list[int] = []
        root_indices: list[int] = []
        examples: list[dict[str, np.ndarray]] = []
        for root in root_group:
            for source_index in by_root[root]:
                row = rows[source_index]
                examples.append(smoke_runner.load_example(product, row))
                source_indices.append(source_index)
                root_indices.append(root)
                touched += 1
                if touched >= next_progress:
                    print(f"Materialized {label}: {touched} artifacts", flush=True)
                    next_progress += max(1, progress_every)
        batch, targets = batch_from_step5_examples(examples, device="cpu")
        targets.validate(batch.graph.graph_count, batch.graph.node_features.device)
        blocks.append(CachedBlock(root_indices=np.asarray(root_indices, dtype=np.int64), source_indices=np.asarray(source_indices, dtype=np.int64), batch=batch, targets=targets))
    expected_examples = len(roots) * 13
    if touched != expected_examples:
        raise RuntimeError(f"materialized {label} examples {touched} != {expected_examples}")
    return blocks


def _to_device_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device=device, non_blocking=False)


def move_cached_block(block: CachedBlock, device: torch.device) -> tuple[Step7ModelBatch, Step7Targets]:
    graph = block.batch.graph
    graph_moved = GraphTensorBatch(
        node_features=_to_device_tensor(graph.node_features, device), edge_index=_to_device_tensor(graph.edge_index, device),
        edge_features=_to_device_tensor(graph.edge_features, device), edge_event_index=_to_device_tensor(graph.edge_event_index, device),
        gate_features=_to_device_tensor(graph.gate_features, device), gate_qubit_ptr=_to_device_tensor(graph.gate_qubit_ptr, device),
        gate_qubit_indices=_to_device_tensor(graph.gate_qubit_indices, device), node_batch=_to_device_tensor(graph.node_batch, device),
        gate_batch=_to_device_tensor(graph.gate_batch, device), graph_count=graph.graph_count,
    )
    diagnostic = block.batch.diagnostic
    diagnostic_moved = DiagnosticTensorBatch(
        local_values=_to_device_tensor(diagnostic.local_values, device), pair_values=_to_device_tensor(diagnostic.pair_values, device),
        pair_index=_to_device_tensor(diagnostic.pair_index, device), pair_batch=_to_device_tensor(diagnostic.pair_batch, device),
        global_parity=_to_device_tensor(diagnostic.global_parity, device), basis_codes=_to_device_tensor(diagnostic.basis_codes, device),
        observed_shots=_to_device_tensor(diagnostic.observed_shots, device), reference_shots=_to_device_tensor(diagnostic.reference_shots, device),
        reference_available_mask=_to_device_tensor(diagnostic.reference_available_mask, device), reference_kind_code=_to_device_tensor(diagnostic.reference_kind_code, device),
        available_mask=_to_device_tensor(diagnostic.available_mask, device),
    )
    targets = Step7Targets(
        effect_present=_to_device_tensor(block.targets.effect_present, device), mechanism=_to_device_tensor(block.targets.mechanism, device),
        mechanism_loss_mask=_to_device_tensor(block.targets.mechanism_loss_mask, device),
    )
    return Step7ModelBatch(graph=graph_moved, diagnostic=diagnostic_moved), targets


def class_weights(blocks: Sequence[CachedBlock]) -> tuple[np.ndarray, np.ndarray]:
    effect = np.concatenate([block.targets.effect_present.detach().cpu().numpy().astype(np.int64) for block in blocks])
    mechanism_parts = []
    for block in blocks:
        mask = block.targets.mechanism_loss_mask.detach().cpu().numpy().astype(bool)
        mechanism = block.targets.mechanism.detach().cpu().numpy().astype(np.int64)
        mechanism_parts.append(mechanism[mask])
    mechanism = np.concatenate(mechanism_parts)
    effect_counts = np.bincount(effect, minlength=2).astype(np.float64)
    mechanism_counts = np.bincount(mechanism, minlength=3).astype(np.float64)
    if np.any(effect_counts <= 0) or np.any(mechanism_counts <= 0):
        raise RuntimeError("Step 7 fit partition is missing a supervised class")
    return (len(effect) / (2.0 * effect_counts)).astype(np.float32), (len(mechanism) / (3.0 * mechanism_counts)).astype(np.float32)


def finite_gradient_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    seen = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("non-finite Step 7 full-run gradient")
        total += parameter.grad.detach().to(dtype=torch.float64).square().sum().cpu()
        seen += parameter.grad.numel()
    if seen == 0:
        raise RuntimeError("Step 7 full run produced no gradients")
    return float(torch.sqrt(total))


def train_one_epoch(model: Step7DiagnosticModel, optimizer: torch.optim.Optimizer, blocks: Sequence[CachedBlock], *, seed: int, epoch: int, device: torch.device, effect_class_weight: np.ndarray, mechanism_class_weight: np.ndarray, effect_loss_weight: float, mechanism_loss_weight: float, gradient_clip_norm: float) -> dict[str, float]:
    model.train()
    order = np.arange(len(blocks), dtype=np.int64)
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(epoch) * 97_409)
    rng.shuffle(order)
    effect_weight_tensor = torch.as_tensor(effect_class_weight, dtype=torch.float32, device=device)
    mechanism_weight_tensor = torch.as_tensor(mechanism_class_weight, dtype=torch.float32, device=device)
    effect_sum = mechanism_sum = total_sum = gradient_norm_sum = 0.0
    effect_count = mechanism_count = steps = 0
    for block_index in order.tolist():
        model_batch, targets = move_cached_block(blocks[block_index], device)
        output = model(model_batch)
        effect_label = targets.effect_present.to(torch.long)
        per_effect = F.binary_cross_entropy_with_logits(output.effect_logit, targets.effect_present, reduction="none")
        effect_loss = (per_effect * effect_weight_tensor.index_select(0, effect_label)).mean()
        mechanism_mask = targets.mechanism_loss_mask
        if not bool(mechanism_mask.any()):
            raise RuntimeError("Step 7 fit block unexpectedly has no mechanism-supervised rows")
        mechanism_label = targets.mechanism[mechanism_mask]
        per_mechanism = F.cross_entropy(output.mechanism_logits[mechanism_mask], mechanism_label, reduction="none")
        mechanism_loss = (per_mechanism * mechanism_weight_tensor.index_select(0, mechanism_label)).mean()
        total_loss = float(effect_loss_weight) * effect_loss + float(mechanism_loss_weight) * mechanism_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError("non-finite Step 7 training loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = finite_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
        optimizer.step()
        batch_effect_count = int(targets.effect_present.numel())
        batch_mechanism_count = int(mechanism_mask.sum())
        effect_sum += float(effect_loss.detach()) * batch_effect_count
        mechanism_sum += float(mechanism_loss.detach()) * batch_mechanism_count
        total_sum += float(total_loss.detach()) * batch_effect_count
        effect_count += batch_effect_count
        mechanism_count += batch_mechanism_count
        gradient_norm_sum += gradient_norm
        steps += 1
    return {
        "effect_loss": effect_sum / max(1, effect_count), "mechanism_loss": mechanism_sum / max(1, mechanism_count),
        "total_loss": total_sum / max(1, effect_count), "mean_preclip_gradient_norm": gradient_norm_sum / max(1, steps),
        "optimizer_steps": float(steps),
    }


@torch.no_grad()
def predict_blocks(model: Step7DiagnosticModel, blocks: Sequence[CachedBlock], device: torch.device) -> PredictionSet:
    model.eval()
    source_parts: list[np.ndarray] = []; root_parts: list[np.ndarray] = []; effect_truth_parts: list[np.ndarray] = []
    mechanism_truth_parts: list[np.ndarray] = []; mask_parts: list[np.ndarray] = []; effect_logit_parts: list[np.ndarray] = []
    mechanism_logit_parts: list[np.ndarray] = []
    for block in blocks:
        model_batch, targets = move_cached_block(block, device)
        output = model(model_batch)
        source_parts.append(block.source_indices); root_parts.append(block.root_indices)
        effect_truth_parts.append(targets.effect_present.detach().cpu().numpy().astype(np.int8))
        mechanism_truth_parts.append(targets.mechanism.detach().cpu().numpy().astype(np.int8))
        mask_parts.append(targets.mechanism_loss_mask.detach().cpu().numpy().astype(bool))
        effect_logit_parts.append(output.effect_logit.detach().cpu().numpy().astype(np.float32))
        mechanism_logit_parts.append(output.mechanism_logits.detach().cpu().numpy().astype(np.float32))
    source = np.concatenate(source_parts); order = np.argsort(source, kind="stable")
    return PredictionSet(
        source_indices=source[order], root_indices=np.concatenate(root_parts)[order], effect_truth=np.concatenate(effect_truth_parts)[order],
        mechanism_truth_all=np.concatenate(mechanism_truth_parts)[order], mechanism_mask=np.concatenate(mask_parts)[order],
        effect_logits=np.concatenate(effect_logit_parts)[order], mechanism_logits=np.concatenate(mechanism_logit_parts, axis=0)[order],
    )


def _metrics_binary(truth: np.ndarray, logits: np.ndarray, threshold: float | None = None) -> tuple[float, dict[str, Any], np.ndarray]:
    if threshold is None:
        threshold, metrics = baseline.select_binary_threshold(truth.astype(np.int64), logits.astype(np.float64))
    else:
        pred = (logits >= float(threshold)).astype(np.int8)
        metrics = baseline.metrics_from_cm(baseline.confusion_matrix(truth, pred, 2))
    pred = (logits >= float(threshold)).astype(np.int8)
    metrics = dict(metrics); metrics["roc_auc"] = baseline.binary_auc(truth, logits)
    return float(threshold), metrics, pred


def _metrics_mechanism(truth: np.ndarray, mask: np.ndarray, logits: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    supervised_truth = truth[mask].astype(np.int64); supervised_logits = logits[mask]
    pred = np.argmax(supervised_logits, axis=1).astype(np.int8)
    metrics = dict(baseline.metrics_from_cm(baseline.confusion_matrix(supervised_truth, pred, 3)))
    metrics["macro_ovr_roc_auc"] = baseline.macro_ovr_auc(supervised_truth, supervised_logits, 3)
    return metrics, pred, np.argmax(logits, axis=1).astype(np.int8)


def selection_summary(prediction: PredictionSet) -> dict[str, Any]:
    threshold, effect_metrics, _ = _metrics_binary(prediction.effect_truth, prediction.effect_logits)
    mechanism_metrics, _, _ = _metrics_mechanism(prediction.mechanism_truth_all, prediction.mechanism_mask, prediction.mechanism_logits)
    return {
        "effect_threshold": threshold, "effect_balanced_accuracy": float(effect_metrics["balanced_accuracy"]),
        "effect_macro_f1": float(effect_metrics["macro_f1"]), "mechanism_balanced_accuracy": float(mechanism_metrics["balanced_accuracy"]),
        "mechanism_macro_f1": float(mechanism_metrics["macro_f1"]),
    }


def checkpoint_is_better(candidate: Mapping[str, Any], best: Mapping[str, Any] | None, min_delta: float) -> bool:
    if best is None:
        return True
    candidate_mech = float(candidate["mechanism_balanced_accuracy"]); best_mech = float(best["mechanism_balanced_accuracy"])
    if candidate_mech > best_mech + min_delta:
        return True
    if abs(candidate_mech - best_mech) <= min_delta:
        return float(candidate["effect_balanced_accuracy"]) > float(best["effect_balanced_accuracy"]) + min_delta
    return False


def instantiate_model(variant: str, seed: int, experiment: Mapping[str, Any], device: torch.device) -> Step7DiagnosticModel:
    settings = experiment["model"]
    model = Step7DiagnosticModel(
        variant=variant, hidden_dim=int(settings["hidden_dim"]), graph_message_passing_layers=int(settings["graph_message_passing_layers"]),
        residual_mlp_layers=int(settings["residual_mlp_layers"]), dropout=float(settings["dropout"]),
        layer_norm_eps=float(settings["layer_norm_eps"]), initialization_seed=int(seed),
    ).to(device)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if actual_parameters != 453_829:
        raise RuntimeError(f"Step 7 fairness contract failed for {variant}: {actual_parameters} != 453829")
    return model


def train_variant_seed(*, variant: str, seed: int, experiment: Mapping[str, Any], fit_blocks: Sequence[CachedBlock], selection_blocks: Sequence[CachedBlock], outer_blocks: Sequence[CachedBlock], effect_class_weight: np.ndarray, mechanism_class_weight: np.ndarray, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], PredictionSet, PredictionSet]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = instantiate_model(variant, seed, experiment, device)
    training = experiment["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    max_epochs = int(training["max_epochs"]); patience = int(training["early_stopping_patience"]); min_delta = float(training["early_stopping_min_delta"])
    best_summary: dict[str, Any] | None = None; best_state: dict[str, torch.Tensor] | None = None; best_epoch: int | None = None; stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, optimizer, fit_blocks, seed=seed, epoch=epoch, device=device, effect_class_weight=effect_class_weight,
            mechanism_class_weight=mechanism_class_weight, effect_loss_weight=float(training["effect_loss_weight"]),
            mechanism_loss_weight=float(training["mechanism_loss_weight"]), gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        selection_prediction = predict_blocks(model, selection_blocks, device); summary = selection_summary(selection_prediction)
        row = {"variant": variant, "seed": seed, "epoch": epoch, **train_metrics, **{f"selection_{key}": value for key, value in summary.items()}, "selected_checkpoint": False}
        history.append(row)
        print(f"{variant} seed={seed} epoch={epoch} train_loss={train_metrics['total_loss']:.4f} sel_mech_BA={summary['mechanism_balanced_accuracy']:.4f} sel_effect_BA={summary['effect_balanced_accuracy']:.4f}", flush=True)
        if checkpoint_is_better(summary, best_summary, min_delta):
            best_summary = dict(summary); best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_summary is None or best_epoch is None:
        raise RuntimeError(f"Step 7 run failed to select a checkpoint for {variant} seed={seed}")
    for row in history:
        if int(row["epoch"]) == best_epoch:
            row["selected_checkpoint"] = True
    model.load_state_dict(best_state); model.to(device)
    selection_prediction = predict_blocks(model, selection_blocks, device); outer_prediction = predict_blocks(model, outer_blocks, device)
    final_threshold, final_effect_metrics, _ = _metrics_binary(selection_prediction.effect_truth, selection_prediction.effect_logits)
    final_mechanism_metrics, _, _ = _metrics_mechanism(selection_prediction.mechanism_truth_all, selection_prediction.mechanism_mask, selection_prediction.mechanism_logits)
    selected = {
        "variant": variant, "seed": seed, "selected_epoch": best_epoch, "selection_effect_threshold": final_threshold,
        "selection_effect_balanced_accuracy": float(final_effect_metrics["balanced_accuracy"]), "selection_effect_macro_f1": float(final_effect_metrics["macro_f1"]),
        "selection_mechanism_balanced_accuracy": float(final_mechanism_metrics["balanced_accuracy"]),
        "selection_mechanism_macro_f1": float(final_mechanism_metrics["macro_f1"]), "epochs_ran": len(history),
        "stopped_early": len(history) < max_epochs, "trainable_parameter_count": 453_829,
    }
    return selected, history, selection_prediction, outer_prediction


def assert_prediction_alignment(reference: PredictionSet, candidate: PredictionSet, label: str) -> None:
    for name in ("source_indices", "root_indices", "effect_truth", "mechanism_truth_all", "mechanism_mask"):
        if not np.array_equal(getattr(reference, name), getattr(candidate, name)):
            raise RuntimeError(f"Step 7 prediction alignment mismatch for {label}: {name}")


def simple_metric_rows(*, prefix: str, truth: PredictionSet, effect_logits: np.ndarray, mechanism_logits: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    _, effect_metrics, effect_pred = _metrics_binary(truth.effect_truth, effect_logits, threshold)
    mechanism_metrics, _, mechanism_pred_all = _metrics_mechanism(truth.mechanism_truth_all, truth.mechanism_mask, mechanism_logits)
    integrated_truth = np.where(truth.effect_truth == 0, 0, truth.mechanism_truth_all.astype(np.int64) + 1).astype(np.int8)
    integrated_pred = np.where(effect_pred == 0, 0, mechanism_pred_all.astype(np.int64) + 1).astype(np.int8)
    integrated_metrics = baseline.metrics_from_cm(baseline.confusion_matrix(integrated_truth, integrated_pred, 4))
    rows = []
    for task, metrics in (("effect_detection", effect_metrics), ("mechanism_diagnosis", mechanism_metrics), ("integrated_diagnosis", integrated_metrics)):
        row = {"task": task, "baseline": prefix}; row.update(metrics)
        if task == "effect_detection": row["roc_auc"] = effect_metrics["roc_auc"]; row["selected_threshold"] = threshold
        if task == "mechanism_diagnosis": row["macro_ovr_roc_auc"] = mechanism_metrics["macro_ovr_roc_auc"]
        rows.append(row)
    return rows


def bootstrap_rows(*, name: str, truth: PredictionSet, effect_logits: np.ndarray, mechanism_logits: np.ndarray, threshold: float, replicates: int, seed: int, confidence: float) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, np.ndarray]], dict[str, np.ndarray]]:
    _, _, effect_pred = _metrics_binary(truth.effect_truth, effect_logits, threshold)
    _, mechanism_pred, mechanism_pred_all = _metrics_mechanism(truth.mechanism_truth_all, truth.mechanism_mask, mechanism_logits)
    mechanism_truth = truth.mechanism_truth_all[truth.mechanism_mask]; mechanism_groups = truth.root_indices[truth.mechanism_mask]
    integrated_truth = np.where(truth.effect_truth == 0, 0, truth.mechanism_truth_all.astype(np.int64) + 1).astype(np.int8)
    integrated_pred = np.where(effect_pred == 0, 0, mechanism_pred_all.astype(np.int64) + 1).astype(np.int8)
    effect_row, effect_boot = baseline.evaluation_row(task="effect_detection", baseline=name, privileged=False, y_true=truth.effect_truth, y_pred=effect_pred, scores=effect_logits, groups=truth.root_indices, class_names=("no_effect", "effect"), n_train=0, selected_lambda=None, selected_threshold=threshold, bootstrap_replicates=replicates, bootstrap_seed=seed, confidence=confidence)
    mechanism_row, mechanism_boot = baseline.evaluation_row(task="mechanism_diagnosis", baseline=name, privileged=False, y_true=mechanism_truth, y_pred=mechanism_pred, scores=mechanism_logits[truth.mechanism_mask], groups=mechanism_groups, class_names=baseline.MECHANISM_NAMES, n_train=0, selected_lambda=None, selected_threshold=None, bootstrap_replicates=replicates, bootstrap_seed=seed, confidence=confidence)
    integrated_row, integrated_boot = baseline.evaluation_row(task="integrated_diagnosis", baseline=name, privileged=False, y_true=integrated_truth, y_pred=integrated_pred, scores=None, groups=truth.root_indices, class_names=baseline.INTEGRATED_NAMES, n_train=0, selected_lambda=None, selected_threshold=threshold, bootstrap_replicates=replicates, bootstrap_seed=seed, confidence=confidence)
    return [effect_row, mechanism_row, integrated_row], {"effect_detection": effect_boot, "mechanism_diagnosis": mechanism_boot, "integrated_diagnosis": integrated_boot}, {"effect_pred": effect_pred, "mechanism_pred_all": mechanism_pred_all, "integrated_pred": integrated_pred}


def verify_step6_predictions(step6a_dir: Path, step6b_dir: Path, outer: PredictionSet, source_product_id: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    complete_a = baseline.read_json(step6a_dir / "benchmark_complete.json"); complete_b = baseline.read_json(step6b_dir / "closure_complete.json")
    if complete_a.get("benchmark_id") != "benchmark_383d4c3070350f0bef6fdb23" or complete_b.get("closure_id") != "closure_e7de7cdd47142287352f8de8":
        raise RuntimeError("unexpected Step 6 reference identity")
    if complete_a.get("source_product_id") != source_product_id or complete_b.get("source_product_id") != source_product_id:
        raise RuntimeError("Step 6 reference source product mismatch")
    for directory, marker in ((step6a_dir, complete_a), (step6b_dir, complete_b)):
        for name, expected_hash in marker.get("file_hashes", {}).items():
            if baseline.sha256_file(directory / name) != expected_hash:
                raise RuntimeError(f"Step 6 reference hash mismatch: {directory.name}/{name}")
    with np.load(step6a_dir / "validation_predictions.npz", allow_pickle=False) as payload:
        prior_a = {key: payload[key] for key in payload.files}
    with np.load(step6b_dir / "validation_predictions.npz", allow_pickle=False) as payload:
        prior_b = {key: payload[key] for key in payload.files}
    for label, prior in (("Step 6A", prior_a), ("Step 6B", prior_b)):
        for key, expected in {"validation_root_index": outer.root_indices, "effect_truth": outer.effect_truth, "mechanism_truth_all": outer.mechanism_truth_all, "mechanism_loss_mask": outer.mechanism_mask}.items():
            if key not in prior or not np.array_equal(np.asarray(prior[key]), expected):
                raise RuntimeError(f"{label} validation alignment failed for {key}")
    return prior_a, prior_b


def step6_bootstraps(*, prior_a: Mapping[str, np.ndarray], prior_b: Mapping[str, np.ndarray], outer: PredictionSet, replicates: int, seed: int, confidence: float) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    effect_pred = np.asarray(prior_b["effect__diag_snr_proxy_threshold__pred"], dtype=np.int8)
    mechanism_pred_all = np.asarray(prior_a["mechanism__diag_full_context_graph__pred_all"], dtype=np.int8)
    _, effect_boot = baseline.evaluation_row(task="effect_detection", baseline="diag_snr_proxy_threshold", privileged=False, y_true=outer.effect_truth, y_pred=effect_pred, scores=None, groups=outer.root_indices, class_names=("no_effect", "effect"), n_train=0, selected_lambda=None, selected_threshold=None, bootstrap_replicates=replicates, bootstrap_seed=seed, confidence=confidence)
    _, mechanism_boot = baseline.evaluation_row(task="mechanism_diagnosis", baseline="diag_full_context_graph", privileged=False, y_true=outer.mechanism_truth_all[outer.mechanism_mask], y_pred=mechanism_pred_all[outer.mechanism_mask], scores=None, groups=outer.root_indices[outer.mechanism_mask], class_names=baseline.MECHANISM_NAMES, n_train=0, selected_lambda=None, selected_threshold=None, bootstrap_replicates=replicates, bootstrap_seed=seed, confidence=confidence)
    return effect_boot, mechanism_boot


def stratified_metric_rows(*, name: str, outer: PredictionSet, rows: Sequence[dict[str, str]], effect_pred: np.ndarray, mechanism_pred_all: np.ndarray, strata: Sequence[str], minimum: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []; contexts = [rows[int(index)] for index in outer.source_indices]
    for stratum in strata:
        values = np.asarray([str(row[stratum]) for row in contexts], dtype=object)
        for value in sorted(set(values.tolist())):
            local = values == value; n = int(np.sum(local))
            if n >= minimum:
                truth = outer.effect_truth[local]; unique = np.unique(truth)
                row: dict[str, Any] = {"task": "effect_detection", "baseline": name, "stratum": stratum, "value": value, "n_examples": n, "balanced_accuracy_defined": bool(len(unique) == 2)}
                if len(unique) == 2: row.update(baseline.metrics_from_cm(baseline.confusion_matrix(truth, effect_pred[local], 2)))
                output.append(row)
            mechanism_local = local & outer.mechanism_mask; mechanism_n = int(np.sum(mechanism_local))
            if mechanism_n >= minimum:
                truth_m = outer.mechanism_truth_all[mechanism_local]; unique_m = np.unique(truth_m)
                row_m: dict[str, Any] = {"task": "mechanism_diagnosis", "baseline": name, "stratum": stratum, "value": value, "n_examples": mechanism_n, "balanced_accuracy_defined": bool(len(unique_m) == 3)}
                if len(unique_m) == 3: row_m.update(baseline.metrics_from_cm(baseline.confusion_matrix(truth_m, mechanism_pred_all[mechanism_local], 3)))
                output.append(row_m)
    return output


def main() -> None:
    args = parse_args(); execution_path = args.config.expanduser().resolve(); experiment_path = args.experiment_config.expanduser().resolve()
    execution = baseline.read_json(execution_path); experiment = baseline.read_json(experiment_path)
    if execution.get("schema") != EXECUTION_SCHEMA or execution.get("status") != "FROZEN_AFTER_SMOKE_BEFORE_FULL_NEURAL_OUTCOME": raise RuntimeError("unexpected Step 7 full benchmark execution config")
    if experiment.get("schema") != EXPERIMENT_SCHEMA or experiment.get("status") != "FROZEN_BEFORE_STEP7_NEURAL_OUTCOME": raise RuntimeError("unexpected Step 7 experiment config")
    if baseline.sha256_file(experiment_path) != execution["experiment_config"]["sha256"]: raise RuntimeError("Step 7 frozen experiment config hash changed after smoke")
    product = args.product_dir.expanduser().resolve() if args.product_dir else Path(experiment["source_dataset"]["default_product_dir"]).expanduser().resolve()
    smoke_dir = args.smoke_dir.expanduser().resolve() if args.smoke_dir else Path(execution["required_smoke"]["default_dir"]).expanduser().resolve()
    step6a_dir = args.step6a_dir.expanduser().resolve() if args.step6a_dir else Path(execution["step6_pairing"]["linear_default_dir"]).expanduser().resolve()
    step6b_dir = args.step6b_dir.expanduser().resolve() if args.step6b_dir else Path(execution["step6_pairing"]["nonlinear_default_dir"]).expanduser().resolve()
    output_parent = args.output_parent.expanduser().resolve() if args.output_parent else Path(execution["outputs"]["default_parent"]).expanduser().resolve()
    verify_smoke(smoke_dir, execution, experiment_path)
    source_complete, rows = baseline.verify_source_product(product, experiment)
    if source_complete["product_id"] != execution["source_product_id"]: raise RuntimeError("Step 7 source product does not match frozen execution config")
    fit_roots, selection_roots, outer_roots, by_root = split_root_indices(rows, experiment); root_batch_size = int(execution["execution"]["root_batch_size"])
    print("Materializing and SHA-verifying frozen Step 5 artifacts exactly once...", flush=True)
    fit_blocks = materialize_blocks(product=product, rows=rows, by_root=by_root, roots=fit_roots, root_batch_size=root_batch_size, label="fit", progress_every=args.progress_every)
    selection_blocks = materialize_blocks(product=product, rows=rows, by_root=by_root, roots=selection_roots, root_batch_size=root_batch_size, label="selection", progress_every=args.progress_every)
    outer_blocks = materialize_blocks(product=product, rows=rows, by_root=by_root, roots=outer_roots, root_batch_size=root_batch_size, label="outer-validation", progress_every=args.progress_every)
    effect_class_weight, mechanism_class_weight = class_weights(fit_blocks); device = resolve_device(args.device); print(f"Step 7 training device: {device}", flush=True)
    training = experiment["training"]; primary_variants = list(experiment["variants"]["primary"]); ablation_variants = list(experiment["variants"]["predeclared_ablations"])
    primary_seeds = [int(seed) for seed in training["primary_seeds"]]; ablation_seeds = [int(seed) for seed in training["ablation_seeds"]]
    run_count = len(primary_variants) * len(primary_seeds) + len(ablation_variants) * len(ablation_seeds)
    if run_count != int(execution["runs"]["expected_model_runs"]): raise RuntimeError("Step 7 expected model-run count changed")
    selections: dict[str, Any] = {}; histories: list[dict[str, Any]] = []; selection_predictions: dict[tuple[str, int], PredictionSet] = {}; outer_predictions: dict[tuple[str, int], PredictionSet] = {}; reference_outer: PredictionSet | None = None
    for variant in primary_variants + ablation_variants:
        seeds = primary_seeds if variant in primary_variants else ablation_seeds
        for seed in seeds:
            print(f"\nTraining Step 7 variant={variant} seed={seed}", flush=True)
            selected, history, selection_prediction, outer_prediction = train_variant_seed(variant=variant, seed=seed, experiment=experiment, fit_blocks=fit_blocks, selection_blocks=selection_blocks, outer_blocks=outer_blocks, effect_class_weight=effect_class_weight, mechanism_class_weight=mechanism_class_weight, device=device)
            selections[f"{variant}__seed{seed}"] = selected; histories.extend(history); selection_predictions[(variant, seed)] = selection_prediction; outer_predictions[(variant, seed)] = outer_prediction
            if reference_outer is None: reference_outer = outer_prediction
            else: assert_prediction_alignment(reference_outer, outer_prediction, f"{variant} seed {seed}")
    assert reference_outer is not None

    replicates = int(experiment["evaluation"]["clean_root_bootstrap_replicates"]); bootstrap_seed = int(experiment["evaluation"]["bootstrap_seed"]); confidence = float(experiment["evaluation"]["confidence_level"])
    aggregate_metrics: list[dict[str, Any]] = []; seed_metrics: list[dict[str, Any]] = []; ablation_metrics: list[dict[str, Any]] = []; stratified_metrics: list[dict[str, Any]] = []; paired_rows: list[dict[str, Any]] = []
    primary_boot: dict[str, dict[str, Mapping[str, np.ndarray]]] = {"effect_detection": {}, "mechanism_diagnosis": {}, "integrated_diagnosis": {}}
    aggregate_payload: dict[str, dict[str, np.ndarray | float]] = {}
    for variant in primary_variants:
        selection_set = selection_predictions[(variant, primary_seeds[0])]; outer_set = outer_predictions[(variant, primary_seeds[0])]
        for seed in primary_seeds[1:]:
            assert_prediction_alignment(selection_set, selection_predictions[(variant, seed)], f"selection {variant} seed {seed}"); assert_prediction_alignment(outer_set, outer_predictions[(variant, seed)], f"outer {variant} seed {seed}")
        selection_effect_logits = np.mean(np.stack([selection_predictions[(variant, seed)].effect_logits for seed in primary_seeds]), axis=0)
        outer_effect_logits = np.mean(np.stack([outer_predictions[(variant, seed)].effect_logits for seed in primary_seeds]), axis=0)
        outer_mechanism_logits = np.mean(np.stack([outer_predictions[(variant, seed)].mechanism_logits for seed in primary_seeds]), axis=0)
        ensemble_threshold, _, _ = _metrics_binary(selection_set.effect_truth, selection_effect_logits)
        selections[f"{variant}__ensemble"] = {"selection_effect_threshold": ensemble_threshold, "aggregation": "mean_logits_across_frozen_primary_seeds", "seeds": primary_seeds}
        rows_out, boots, predictions = bootstrap_rows(name=variant, truth=outer_set, effect_logits=outer_effect_logits, mechanism_logits=outer_mechanism_logits, threshold=ensemble_threshold, replicates=replicates, seed=bootstrap_seed, confidence=confidence)
        aggregate_metrics.extend(rows_out)
        for task, boot in boots.items(): primary_boot[task][variant] = boot
        aggregate_payload[variant] = {"effect_logits": outer_effect_logits.astype(np.float32), "mechanism_logits": outer_mechanism_logits.astype(np.float32), "effect_pred": predictions["effect_pred"].astype(np.int8), "mechanism_pred_all": predictions["mechanism_pred_all"].astype(np.int8), "threshold": float(ensemble_threshold)}
        stratified_metrics.extend(stratified_metric_rows(name=variant, outer=outer_set, rows=rows, effect_pred=predictions["effect_pred"], mechanism_pred_all=predictions["mechanism_pred_all"], strata=experiment["evaluation"]["strata"], minimum=int(experiment["evaluation"]["minimum_stratum_examples"])))
        for seed in primary_seeds:
            threshold = float(selections[f"{variant}__seed{seed}"]["selection_effect_threshold"])
            seed_metrics.extend(simple_metric_rows(prefix=f"{variant}__seed{seed}", truth=outer_predictions[(variant, seed)], effect_logits=outer_predictions[(variant, seed)].effect_logits, mechanism_logits=outer_predictions[(variant, seed)].mechanism_logits, threshold=threshold))
    configured_pairs = [tuple(pair) for pair in experiment["evaluation"]["primary_paired_comparisons"]]
    for task, boot in primary_boot.items(): paired_rows.extend(baseline.paired_difference_rows(task, configured_pairs, boot, confidence))

    structured_seed_name = f"structured_interaction__seed{ablation_seeds[0]}"; structured_seed = outer_predictions[("structured_interaction", ablation_seeds[0])]; structured_threshold = float(selections[structured_seed_name]["selection_effect_threshold"])
    structured_rows, structured_boots, _ = bootstrap_rows(name=structured_seed_name, truth=structured_seed, effect_logits=structured_seed.effect_logits, mechanism_logits=structured_seed.mechanism_logits, threshold=structured_threshold, replicates=replicates, seed=bootstrap_seed, confidence=confidence)
    ablation_metrics.extend(structured_rows); ablation_boot: dict[str, dict[str, Mapping[str, np.ndarray]]] = {task: {structured_seed_name: boot} for task, boot in structured_boots.items()}
    for variant in ablation_variants:
        seed = ablation_seeds[0]; outer_set = outer_predictions[(variant, seed)]; threshold = float(selections[f"{variant}__seed{seed}"]["selection_effect_threshold"])
        rows_out, boots, _ = bootstrap_rows(name=f"{variant}__seed{seed}", truth=outer_set, effect_logits=outer_set.effect_logits, mechanism_logits=outer_set.mechanism_logits, threshold=threshold, replicates=replicates, seed=bootstrap_seed, confidence=confidence)
        ablation_metrics.extend(rows_out)
        for task, boot in boots.items(): ablation_boot.setdefault(task, {})[f"{variant}__seed{seed}"] = boot
    ablation_pairs = [(structured_seed_name, f"{variant}__seed{ablation_seeds[0]}") for variant in ablation_variants]
    for task, boot in ablation_boot.items(): paired_rows.extend(baseline.paired_difference_rows(task, ablation_pairs, boot, confidence))

    prior_a, prior_b = verify_step6_predictions(step6a_dir, step6b_dir, reference_outer, source_complete["product_id"])
    step6_effect_boot, step6_mechanism_boot = step6_bootstraps(prior_a=prior_a, prior_b=prior_b, outer=reference_outer, replicates=replicates, seed=bootstrap_seed, confidence=confidence)
    effect_pair_boot = dict(primary_boot["effect_detection"]); effect_pair_boot["diag_snr_proxy_threshold"] = step6_effect_boot
    mechanism_pair_boot = dict(primary_boot["mechanism_diagnosis"]); mechanism_pair_boot["diag_full_context_graph"] = step6_mechanism_boot
    paired_rows.extend(baseline.paired_difference_rows("effect_detection", [("structured_interaction", "diag_snr_proxy_threshold")], effect_pair_boot, confidence))
    paired_rows.extend(baseline.paired_difference_rows("mechanism_diagnosis", [("structured_interaction", "diag_full_context_graph")], mechanism_pair_boot, confidence))
    paired_lookup = {(row["task"], row["left"], row["right"], row["metric"]): row for row in paired_rows}
    architecture_row = paired_lookup[("mechanism_diagnosis", "structured_interaction", "late_concat", "balanced_accuracy")]
    diagnostic_row = paired_lookup[("mechanism_diagnosis", "structured_interaction", "diagnostic_only", "balanced_accuracy")]
    graph_row = paired_lookup[("mechanism_diagnosis", "structured_interaction", "graph_only", "balanced_accuracy")]
    effect_row = paired_lookup[("effect_detection", "structured_interaction", "diag_snr_proxy_threshold", "balanced_accuracy")]
    decision_flags = {
        "structured_interaction_architecture_signal": float(architecture_row["ci_low"]) > 0.0,
        "structured_beats_diagnostic_only_ci": float(diagnostic_row["ci_low"]) > 0.0,
        "structured_beats_graph_only_ci": float(graph_row["ci_low"]) > 0.0,
        "effect_path_noninferior_to_step6_snr_proxy": float(effect_row["ci_low"]) >= -0.01,
        "no_confirmation_claim_from_step7": True, "new_confirmatory_cohort_required": True,
    }

    identity = {"schema": EXECUTION_SCHEMA, "execution_config_sha256": baseline.sha256_file(execution_path), "experiment_config_sha256": baseline.sha256_file(experiment_path), "runner_sha256": baseline.sha256_file(Path(__file__).resolve()), "source_product_id": source_complete["product_id"], "smoke_id": execution["required_smoke"]["smoke_id"]}
    benchmark_id = "benchmark_" + hashlib.sha256(baseline.canonical_json(identity).encode("utf-8")).hexdigest()[:24]
    output_parent.mkdir(parents=True, exist_ok=True); output = output_parent / benchmark_id
    if output.exists(): raise RuntimeError(f"refusing to overwrite existing Step 7 benchmark {output}")
    staging = output_parent / f".{benchmark_id}.staging-{uuid.uuid4().hex}"; staging.mkdir()
    baseline.write_csv(staging / "aggregate_metrics.csv", aggregate_metrics); baseline.write_csv(staging / "seed_metrics.csv", seed_metrics); baseline.write_csv(staging / "paired_differences.csv", paired_rows)
    baseline.write_csv(staging / "ablation_metrics.csv", ablation_metrics); baseline.write_csv(staging / "stratified_metrics.csv", stratified_metrics); baseline.write_csv(staging / "training_history.csv", histories)
    baseline.atomic_json(staging / "model_selection.json", selections)
    baseline.atomic_json(staging / "decision.json", {"schema": EXECUTION_SCHEMA, "decision": experiment["completion"]["decision_on_completed_run"], "evidence_status": experiment["evidence_status"], "interpretation_flags": decision_flags, "outer_validation_used_for_selection": False, "historical_v0_1_test_accessed": False, "spent_confirmatory_cohort_accessed": False, "new_confirmatory_cohort_accessed": False, "hardware_executed": False, "step8_automatically_unlocked": False})
    payload: dict[str, np.ndarray] = {"validation_source_index": reference_outer.source_indices.astype(np.int64), "validation_root_index": reference_outer.root_indices.astype(np.int64), "effect_truth": reference_outer.effect_truth.astype(np.int8), "mechanism_truth_all": reference_outer.mechanism_truth_all.astype(np.int8), "mechanism_loss_mask": reference_outer.mechanism_mask.astype(bool)}
    for variant, values in aggregate_payload.items():
        payload[f"effect__{variant}__logit"] = np.asarray(values["effect_logits"]); payload[f"effect__{variant}__pred"] = np.asarray(values["effect_pred"])
        payload[f"mechanism__{variant}__logits"] = np.asarray(values["mechanism_logits"]); payload[f"mechanism__{variant}__pred_all"] = np.asarray(values["mechanism_pred_all"])
    for (variant, seed), prediction in outer_predictions.items():
        payload[f"effect__{variant}__seed{seed}__logit"] = prediction.effect_logits.astype(np.float32); payload[f"mechanism__{variant}__seed{seed}__logits"] = prediction.mechanism_logits.astype(np.float32)
    np.savez_compressed(staging / "validation_predictions.npz", **payload)
    files = ["aggregate_metrics.csv", "seed_metrics.csv", "paired_differences.csv", "ablation_metrics.csv", "stratified_metrics.csv", "training_history.csv", "model_selection.json", "decision.json", "validation_predictions.npz"]
    completion = {"schema": EXECUTION_SCHEMA, "status": "COMPLETE", "benchmark_id": benchmark_id, "identity": identity, "source_product_id": source_complete["product_id"], "smoke_id": execution["required_smoke"]["smoke_id"], "fit_roots": len(fit_roots), "selection_roots": len(selection_roots), "outer_development_validation_roots": len(outer_roots), "model_runs": run_count, "primary_variants": primary_variants, "primary_seeds": primary_seeds, "ablation_variants": ablation_variants, "ablation_seeds": ablation_seeds, "interpretation_flags": decision_flags, "evidence_status": experiment["evidence_status"], "historical_v0_1_test_accessed": False, "spent_confirmatory_cohort_accessed": False, "new_confirmatory_cohort_accessed": False, "hardware_executed": False, "step8_automatically_unlocked": False, "file_hashes": {name: baseline.sha256_file(staging / name) for name in files}}
    baseline.atomic_json(staging / "benchmark_complete.json", completion); os.replace(staging, output)
    metric_lookup = {(row["task"], row["baseline"]): row for row in aggregate_metrics}
    print("\nTRIQTO STEP 7 FULL DEVELOPMENT BENCHMARK COMPLETE\n"); print(f"Decision: {experiment['completion']['decision_on_completed_run']}")
    for variant in primary_variants:
        print(f"{variant}: effect_BA={float(metric_lookup[('effect_detection', variant)]['balanced_accuracy']):.4f} mechanism_BA={float(metric_lookup[('mechanism_diagnosis', variant)]['balanced_accuracy']):.4f} integrated_BA={float(metric_lookup[('integrated_diagnosis', variant)]['balanced_accuracy']):.4f}")
    print("Structured-vs-late mechanism architecture signal: " + ("YES" if decision_flags["structured_interaction_architecture_signal"] else "NO"))
    print("Structured effect noninferior to Step 6 SNR proxy: " + ("YES" if decision_flags["effect_path_noninferior_to_step6_snr_proxy"] else "NO"))
    print("Evidence status: DEVELOPMENT, NOT CONFIRMATORY"); print("Historical v0.1 test accessed: NO"); print("Spent confirmatory cohort accessed: NO"); print("New confirmatory cohort accessed: NO"); print("Step 8 automatically unlocked: NO"); print(f"Results: {output}")


if __name__ == "__main__":
    main()
