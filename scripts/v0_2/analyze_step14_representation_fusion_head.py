#!/usr/bin/env python3
"""Post-Step-14 representation/fusion/head decomposition on fixed development data.

This is a diagnostic probe only. It never updates the frozen TriQTO checkpoints,
never uses Step-14 simulator outer or future-hardware reserve data, and never
uses QPU evidence. Small probe heads are trained with a fully frozen recipe on
fit partitions and evaluated on the already-defined selection partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
import run_step14_cross_motif_training as step14
from triqto.model.tensor_ops import segment_mean
from triqto.step7.model import Step7DiagnosticModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
MIXTURE_CONFIG = ROOT / "configs/v0_2/step10_training_mixture.json"
STEP7_CONFIG = ROOT / "configs/v0_2/step7_structured_diagnostic_model.json"
TRAINING_PARENT = Path("/workspace/triqto-data/step14_cross_motif_training")
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_representation_decomposition")
SCHEMA = "triqto.v0_2.step14_representation_fusion_head_decomposition.v1"
STAGES = (
    "fused_representation_64",
    "prefusion_global_128",
    "prediagnostic_global_256",
    "aligned_prepool_640",
)
PROBE_SPEC = {
    "hidden_dim": 64,
    "dropout": 0.1,
    "epochs": 25,
    "optimizer": "AdamW",
    "learning_rate": 0.003,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "standardization": "fit_mean_std_per_seed_and_stage",
    "sample_weighting": "equal_total_weight_per_fit_domain_and_mechanism_class",
    "selection_used_for_training_or_early_stopping": False,
    "meaningful_delta_minimum": 0.05,
    "bootstrap_replicates": 1000,
    "bootstrap_unit": "cross_motif_family_id",
    "bootstrap_seed": 2026090101,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_training_freeze(run_id: str, freeze_sha: str) -> tuple[Path, dict[str, Any]]:
    pointer_path = TRAINING_PARENT / "current_training_run.json"
    pointer = read_json(pointer_path)
    if pointer.get("schema") != "triqto.v0_2.step14_current_training_run.v1":
        raise RuntimeError("unexpected Step-14 current training pointer schema")
    if str(pointer.get("run_id")) != run_id:
        raise RuntimeError("requested decomposition training run is not the frozen current run")
    if str(pointer.get("selection_freeze_sha256")) != freeze_sha:
        raise RuntimeError("requested selection-freeze hash does not match current frozen run")
    run_dir = Path(str(pointer.get("run_dir", ""))).expanduser().resolve()
    if run_dir.parent != TRAINING_PARENT.resolve() or run_dir.name != run_id:
        raise RuntimeError("unsafe Step-14 training run directory")
    freeze_path = run_dir / "selection_freeze.json"
    if not freeze_path.is_file() or baseline.sha256_file(freeze_path) != freeze_sha:
        raise RuntimeError("Step-14 selection freeze is missing or hash-mismatched")
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN_BEFORE_ANY_STEP14_OUTER_MATERIALIZATION":
        raise RuntimeError("selection marker is not the pre-outer frozen marker")
    if not bool(freeze.get("all_three_seed_checkpoints_frozen")):
        raise RuntimeError("selection marker does not freeze all three checkpoints")
    if [int(v) for v in freeze.get("seeds", [])] != [1701, 1702, 1703]:
        raise RuntimeError("unexpected Step-14 frozen seed set")
    return run_dir, freeze


def load_selected_model(
    run_dir: Path,
    freeze: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> Step7DiagnosticModel:
    record = freeze["selected_seed_records"][str(seed)]
    checkpoint = run_dir / str(record["checkpoint"])
    wanted = str(record["checkpoint_sha256"])
    if baseline.sha256_file(checkpoint) != wanted:
        raise RuntimeError(f"selected Step-14 checkpoint hash mismatch for seed {seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "triqto.v0_2.step14_training_run.v1":
        raise RuntimeError(f"unexpected selected checkpoint schema for seed {seed}")
    if int(payload.get("seed", -1)) != seed or payload.get("architecture") != "late_concat":
        raise RuntimeError(f"selected checkpoint metadata mismatch for seed {seed}")
    model = Step7DiagnosticModel(variant="late_concat", initialization_seed=seed)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def ba_from_logits(truth: np.ndarray, logits: np.ndarray) -> float:
    guess = np.argmax(logits, axis=1).astype(np.int64)
    cm = baseline.confusion_matrix(truth.astype(np.int64), guess, 3)
    return float(baseline.metrics_from_cm(cm)["balanced_accuracy"])


def recalls_from_logits(truth: np.ndarray, logits: np.ndarray) -> list[float]:
    guess = np.argmax(logits, axis=1).astype(np.int64)
    cm = baseline.confusion_matrix(truth.astype(np.int64), guess, 3)
    out: list[float] = []
    for cls in range(3):
        denom = float(np.sum(cm[cls]))
        out.append(float(cm[cls, cls] / denom) if denom else 0.0)
    return out


def _extract_block_features(
    model: Step7DiagnosticModel,
    model_batch: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    local, pair, parity, diagnostic_graph = model.diagnostic_encoder(model_batch)
    graph_output = model.graph_encoder(model_batch.graph)
    graph_embedding = graph_output.graph_embedding
    count = model_batch.graph.graph_count
    available = model_batch.diagnostic.available_mask.to(dtype=graph_embedding.dtype).unsqueeze(1)

    pooled_local = segment_mean(local, model_batch.graph.node_batch, count) * available
    if pair.shape[0]:
        pooled_pair = segment_mean(pair, model_batch.diagnostic.pair_batch, count) * available
    else:
        pooled_pair = graph_embedding.new_zeros(graph_embedding.shape)
    parity_masked = parity * available

    prefusion = torch.cat((graph_embedding, diagnostic_graph), dim=1)
    fused = model.late_concat_fusion(prefusion)
    prediagnostic = torch.cat((graph_embedding, pooled_local, pooled_pair, parity_masked), dim=1)

    local_product = segment_mean(
        graph_output.node_embeddings * local,
        model_batch.graph.node_batch,
        count,
    ) * available
    local_absdiff = segment_mean(
        (graph_output.node_embeddings - local).abs(),
        model_batch.graph.node_batch,
        count,
    ) * available

    if pair.shape[0]:
        left = graph_output.node_embeddings.index_select(0, model_batch.diagnostic.pair_index[0])
        right = graph_output.node_embeddings.index_select(0, model_batch.diagnostic.pair_index[1])
        pair_context = 0.5 * (left + right)
        pair_product = segment_mean(
            pair_context * pair,
            model_batch.diagnostic.pair_batch,
            count,
        ) * available
        pair_absdiff = segment_mean(
            (pair_context - pair).abs(),
            model_batch.diagnostic.pair_batch,
            count,
        ) * available
    else:
        pair_product = graph_embedding.new_zeros(graph_embedding.shape)
        pair_absdiff = graph_embedding.new_zeros(graph_embedding.shape)

    global_product = graph_embedding * parity_masked
    global_absdiff = (graph_embedding - parity_masked).abs() * available
    aligned = torch.cat(
        (
            prediagnostic,
            local_product,
            local_absdiff,
            pair_product,
            pair_absdiff,
            global_product,
            global_absdiff,
        ),
        dim=1,
    )
    if aligned.shape[1] != 640:
        raise RuntimeError(f"aligned decomposition representation has width {aligned.shape[1]} != 640")

    features = {
        "fused_representation_64": fused,
        "prefusion_global_128": prefusion,
        "prediagnostic_global_256": prediagnostic,
        "aligned_prepool_640": aligned,
    }
    frozen_logits = model.mechanism_head(fused)
    return features, frozen_logits


@torch.no_grad()
def extract_domain(
    model: Step7DiagnosticModel,
    blocks: Sequence[step7.CachedBlock],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    feature_parts: dict[str, list[np.ndarray]] = {name: [] for name in STAGES}
    truth_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    root_parts: list[np.ndarray] = []
    frozen_parts: list[np.ndarray] = []
    for block in blocks:
        model_batch, targets = step7.move_cached_block(block, device)
        features, frozen_logits = _extract_block_features(model, model_batch)
        mask_t = targets.mechanism_loss_mask
        mask = mask_t.detach().cpu().numpy().astype(bool)
        truth_parts.append(targets.mechanism.detach().cpu().numpy().astype(np.int64)[mask])
        source_parts.append(block.source_indices[mask])
        root_parts.append(block.root_indices[mask])
        frozen_parts.append(frozen_logits[mask_t].detach().cpu().numpy().astype(np.float32))
        for name in STAGES:
            feature_parts[name].append(features[name][mask_t].detach().cpu().numpy().astype(np.float32))
    source = np.concatenate(source_parts)
    order = np.argsort(source, kind="stable")
    return {
        "source_indices": source[order],
        "root_indices": np.concatenate(root_parts)[order],
        "truth": np.concatenate(truth_parts)[order],
        "frozen_logits": np.concatenate(frozen_parts, axis=0)[order],
        "features": {
            name: np.concatenate(parts, axis=0)[order]
            for name, parts in feature_parts.items()
        },
    }


class ProbeHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(PROBE_SPEC["hidden_dim"])),
            nn.GELU(),
            nn.Dropout(float(PROBE_SPEC["dropout"])),
            nn.Linear(int(PROBE_SPEC["hidden_dim"]), 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_sample_weights(truth: np.ndarray, domains: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(truth), dtype=np.float32)
    for domain in sorted(np.unique(domains).tolist()):
        for cls in range(3):
            mask = (domains == domain) & (truth == cls)
            count = int(np.sum(mask))
            if count <= 0:
                raise RuntimeError(f"probe fit data missing domain={domain} class={cls}")
            weights[mask] = 1.0 / float(count)
    weights *= float(len(weights)) / float(np.sum(weights))
    return weights


def train_probe(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_domains: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[ProbeHead, np.ndarray, np.ndarray, dict[str, float]]:
    mean = fit_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = fit_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std >= 1e-6, std, 1.0).astype(np.float32)
    x = torch.as_tensor((fit_x - mean) / std, dtype=torch.float32, device=device)
    y = torch.as_tensor(fit_y, dtype=torch.long, device=device)
    w = torch.as_tensor(make_sample_weights(fit_y, fit_domains), dtype=torch.float32, device=device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    probe = ProbeHead(fit_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(PROBE_SPEC["learning_rate"]),
        weight_decay=float(PROBE_SPEC["weight_decay"]),
    )
    final_loss = float("nan")
    for _epoch in range(int(PROBE_SPEC["epochs"])):
        probe.train()
        logits = probe(x)
        per = F.cross_entropy(logits, y, reduction="none")
        loss = (per * w).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite decomposition probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), float(PROBE_SPEC["gradient_clip_norm"]))
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    probe.eval()
    with torch.no_grad():
        fit_logits = probe(x).detach().cpu().numpy().astype(np.float32)
    return probe, mean, std, {
        "final_weighted_fit_loss": final_loss,
        "fit_balanced_accuracy": ba_from_logits(fit_y, fit_logits),
    }


@torch.no_grad()
def apply_probe(
    probe: ProbeHead,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.as_tensor((x - mean) / std, dtype=torch.float32, device=device)
    return probe(tensor).detach().cpu().numpy().astype(np.float32)


def metric_record(truth: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    recalls = recalls_from_logits(truth, logits)
    return {
        "mechanism_balanced_accuracy": ba_from_logits(truth, logits),
        "mechanism_recall": recalls,
        "minimum_mechanism_recall": min(recalls),
        "example_count": int(len(truth)),
    }


def bootstrap_delta(
    truth: np.ndarray,
    candidate_logits: np.ndarray,
    reference_logits: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique.tolist()}
    rng = np.random.default_rng(int(PROBE_SPEC["bootstrap_seed"]))
    values = np.empty(int(PROBE_SPEC["bootstrap_replicates"]), dtype=np.float64)
    for rep in range(len(values)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_indices[group] for group in sampled.tolist()])
        values[rep] = (
            ba_from_logits(truth[idx], candidate_logits[idx])
            - ba_from_logits(truth[idx], reference_logits[idx])
        )
    delta = ba_from_logits(truth, candidate_logits) - ba_from_logits(truth, reference_logits)
    return {
        "mean_delta": float(delta),
        "bootstrap_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "bootstrap_replicates": int(len(values)),
        "bootstrap_unit_count": int(len(unique)),
    }


def main() -> None:
    args = parse_args()
    if not args.selection_freeze_sha256.startswith("sha256:") or len(args.selection_freeze_sha256) != 71:
        raise ValueError("selection freeze must be sha256:<64-hex>")
    if args.progress_every < 1 or args.progress_every > 100000:
        raise ValueError("progress_every must be between 1 and 100000")

    cfg = read_json(CONFIG)
    step14.assert_contract(cfg)
    run_dir, freeze = verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)
    device = step7.resolve_device(args.device)
    print("TRIQTO STEP 14 REPRESENTATION/FUSION/HEAD DECOMPOSITION", flush=True)
    print("MAIN MODEL FROZEN — DEVELOPMENT FIT/SELECTION ONLY — NO OUTER / NO QPU", flush=True)

    mixture_cfg = read_json(MIXTURE_CONFIG)
    experiment = read_json(STEP7_CONFIG)
    cross_product = step14.resolve_cross_product(None)
    cross_rows, cross_by, cross_fit_roots, cross_sel_roots = step14.verify_cross_product(cross_product, cfg)
    step10_product = Path(cfg["legacy_training_domains"]["default_product_dir"]).expanduser().resolve()
    (
        original_product,
        original_rows,
        original_by,
        original_fit_roots,
        original_sel_roots,
        bridge_rows,
        bridge_by,
        bridge_fit_roots,
        bridge_sel_roots,
    ) = step14.load_legacy(step10_product, mixture_cfg, experiment)

    batch_size = int(cfg["training"]["root_batch_size"])
    def mat(product: Path, rows: Sequence[dict[str, str]], by: Mapping[int, Sequence[int]], roots: Sequence[int], label: str):
        return step7.materialize_blocks(
            product=product,
            rows=rows,
            by_root=by,
            roots=roots,
            root_batch_size=batch_size,
            label=label,
            progress_every=args.progress_every,
        )

    fit_blocks = {
        "legacy_original": mat(original_product, original_rows, original_by, original_fit_roots, "decomp-original-fit"),
        "legacy_bridge": mat(step10_product, bridge_rows, bridge_by, bridge_fit_roots, "decomp-bridge-fit"),
        "cross_motif": mat(cross_product, cross_rows, cross_by, cross_fit_roots, "decomp-cross-fit"),
    }
    sel_blocks = {
        "legacy_original": mat(original_product, original_rows, original_by, original_sel_roots, "decomp-original-selection"),
        "legacy_bridge": mat(step10_product, bridge_rows, bridge_by, bridge_sel_roots, "decomp-bridge-selection"),
        "cross_motif": mat(cross_product, cross_rows, cross_by, cross_sel_roots, "decomp-cross-selection"),
    }

    identity = {
        "schema": SCHEMA,
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "protocol_config_sha256": baseline.sha256_file(CONFIG),
        "cross_dataset_complete_sha256": baseline.sha256_file(cross_product / "dataset_complete.json"),
        "step10_dataset_complete_sha256": baseline.sha256_file(step10_product / "dataset_complete.json"),
        "selected_checkpoint_hashes": freeze["checkpoint_hashes"],
        "probe_spec": PROBE_SPEC,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_accessed": False,
        "main_model_weights_updated": False,
    }
    decomposition_id = "decomposition_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    out_dir = output_parent / decomposition_id
    complete_path = out_dir / "decomposition_complete.json"
    if complete_path.is_file():
        complete = read_json(complete_path)
        if complete.get("identity") == identity and complete.get("status") == "COMPLETE_FROZEN_DECOMPOSITION":
            print("Exact decomposition product already complete:", out_dir, flush=True)
            return
        raise RuntimeError("decomposition output directory exists with mismatched identity")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_results: dict[str, Any] = {}
    ensemble_logits: dict[str, dict[str, list[np.ndarray]]] = {
        domain: {"frozen_head": [], **{stage: [] for stage in STAGES}}
        for domain in sel_blocks
    }
    selection_truth: dict[str, np.ndarray] = {}
    selection_sources: dict[str, np.ndarray] = {}

    for seed in (1701, 1702, 1703):
        print(f"Extracting frozen representations for seed {seed}", flush=True)
        model = load_selected_model(run_dir, freeze, seed, device)
        fit = {name: extract_domain(model, blocks, device) for name, blocks in fit_blocks.items()}
        selection = {name: extract_domain(model, blocks, device) for name, blocks in sel_blocks.items()}

        for domain in selection:
            if domain in selection_truth:
                if not np.array_equal(selection_truth[domain], selection[domain]["truth"]):
                    raise RuntimeError(f"selection truth misalignment across seeds for {domain}")
                if not np.array_equal(selection_sources[domain], selection[domain]["source_indices"]):
                    raise RuntimeError(f"selection source-index misalignment across seeds for {domain}")
            else:
                selection_truth[domain] = selection[domain]["truth"].copy()
                selection_sources[domain] = selection[domain]["source_indices"].copy()
            ensemble_logits[domain]["frozen_head"].append(selection[domain]["frozen_logits"])

        fit_truth = np.concatenate([fit[name]["truth"] for name in fit_blocks])
        fit_domain_ids = np.concatenate([
            np.full(len(fit[name]["truth"]), idx, dtype=np.int64)
            for idx, name in enumerate(fit_blocks)
        ])
        per_stage: dict[str, Any] = {}
        for stage_index, stage in enumerate(STAGES):
            fit_x = np.concatenate([fit[name]["features"][stage] for name in fit_blocks], axis=0)
            probe_seed = 500_000 + seed * 10 + stage_index
            probe, mean, std, training_metrics = train_probe(
                fit_x,
                fit_truth,
                fit_domain_ids,
                seed=probe_seed,
                device=device,
            )
            stage_selection: dict[str, Any] = {}
            for domain in selection:
                logits = apply_probe(probe, selection[domain]["features"][stage], mean, std, device)
                ensemble_logits[domain][stage].append(logits)
                stage_selection[domain] = metric_record(selection[domain]["truth"], logits)
            per_stage[stage] = {
                "probe_seed": probe_seed,
                "training": training_metrics,
                "selection": stage_selection,
            }
            print(
                f"seed={seed} stage={stage} cross_selection_ba="
                f"{stage_selection['cross_motif']['mechanism_balanced_accuracy']:.4f}",
                flush=True,
            )
        seed_results[str(seed)] = {
            "frozen_head_selection": {
                domain: metric_record(selection[domain]["truth"], selection[domain]["frozen_logits"])
                for domain in selection
            },
            "probe_stages": per_stage,
        }
        del model, fit, selection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ensemble_metrics: dict[str, Any] = {}
    averaged: dict[str, dict[str, np.ndarray]] = {}
    for domain in ensemble_logits:
        averaged[domain] = {}
        ensemble_metrics[domain] = {}
        for name, parts in ensemble_logits[domain].items():
            logits = np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)
            averaged[domain][name] = logits
            ensemble_metrics[domain][name] = metric_record(selection_truth[domain], logits)

    cross_sources = selection_sources["cross_motif"]
    cross_groups = np.asarray([cross_rows[int(index)]["family_id"] for index in cross_sources], dtype=object)
    comparisons = {
        "mechanism_head_or_joint_training": ("fused_representation_64", "frozen_head"),
        "late_concat_fusion_compression": ("prefusion_global_128", "fused_representation_64"),
        "diagnostic_global_compression": ("prediagnostic_global_256", "prefusion_global_128"),
        "missing_local_graph_diagnostic_alignment": ("aligned_prepool_640", "prediagnostic_global_256"),
    }
    attribution: dict[str, Any] = {}
    passed_signals: list[str] = []
    for name, (candidate, reference) in comparisons.items():
        record = bootstrap_delta(
            selection_truth["cross_motif"],
            averaged["cross_motif"][candidate],
            averaged["cross_motif"][reference],
            cross_groups,
        )
        record["candidate"] = candidate
        record["reference"] = reference
        record["signal_gate"] = bool(
            record["mean_delta"] >= float(PROBE_SPEC["meaningful_delta_minimum"])
            and record["bootstrap_ci"][0] > 0.0
        )
        if record["signal_gate"]:
            passed_signals.append(name)
        attribution[name] = record

    if not passed_signals:
        interpretation = "NO_SINGLE_DOWNSTREAM_BOTTLENECK_ISOLATED__UPSTREAM_OR_DISTRIBUTED_REPRESENTATION_LIMIT"
        primary_signal = None
    elif len(passed_signals) == 1:
        primary_signal = passed_signals[0]
        interpretation = "SINGLE_PRIMARY_BOTTLENECK_SIGNAL__" + primary_signal.upper()
    else:
        primary_signal = max(passed_signals, key=lambda key: float(attribution[key]["mean_delta"]))
        interpretation = "MULTIPLE_SERIAL_BOTTLENECK_SIGNALS__PRIMARY_" + primary_signal.upper()

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_DECOMPOSITION",
        "decomposition_id": decomposition_id,
        "identity": identity,
        "probe_spec": PROBE_SPEC,
        "seed_results": seed_results,
        "selection_ensemble_metrics": ensemble_metrics,
        "cross_motif_attribution": attribution,
        "passed_signals": passed_signals,
        "primary_signal": primary_signal,
        "interpretation": interpretation,
        "scientific_boundaries": {
            "main_model_retrained": False,
            "selection_used_for_probe_training_or_early_stopping": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
            "probe_outputs_select_no_deployment_checkpoint": True,
        },
    }
    result_path = out_dir / "decomposition_result.json"
    atomic_json(result_path, result)
    complete = {
        "schema": SCHEMA,
        "status": "COMPLETE_FROZEN_DECOMPOSITION",
        "decomposition_id": decomposition_id,
        "identity": identity,
        "decomposition_result_sha256": baseline.sha256_file(result_path),
        "interpretation": interpretation,
        "primary_signal": primary_signal,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    }
    atomic_json(complete_path, complete)
    atomic_json(output_parent / "current_decomposition.json", {
        "schema": "triqto.v0_2.step14_current_representation_decomposition.v1",
        "status": "COMPLETE_FROZEN_DECOMPOSITION",
        "decomposition_id": decomposition_id,
        "decomposition_dir": str(out_dir),
        "decomposition_result_sha256": baseline.sha256_file(result_path),
        "decomposition_complete_sha256": baseline.sha256_file(complete_path),
        "training_run_id": args.training_run_id,
        "selection_freeze_sha256": args.selection_freeze_sha256,
        "primary_signal": primary_signal,
        "interpretation": interpretation,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    })

    print("\nDECOMPOSITION COMPLETE", flush=True)
    print("Interpretation:", interpretation, flush=True)
    print("Primary signal:", primary_signal, flush=True)
    for name, record in attribution.items():
        print(
            f"{name}: delta={record['mean_delta']:.4f} "
            f"CI=[{record['bootstrap_ci'][0]:.4f},{record['bootstrap_ci'][1]:.4f}] "
            f"gate={'PASS' if record['signal_gate'] else 'NO'}",
            flush=True,
        )
    print("Result:", result_path, flush=True)


if __name__ == "__main__":
    main()
