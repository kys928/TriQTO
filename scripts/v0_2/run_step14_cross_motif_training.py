#!/usr/bin/env python3
"""Run the frozen Step-14 warm-start cross-motif training intervention.

Consumes only legacy fit/selection plus the frozen Step-14 development product.
Freezes three selected checkpoints and one ensemble effect threshold before any
Step-14 outer cohort may be materialized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
import run_step10_warmstart_vs_scratch as step10b
from triqto.step7.model import Step7DiagnosticModel

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "triqto.v0_2.step14_training_run.v1"
DEFAULT_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
DEFAULT_MIXTURE_CONFIG = ROOT / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = ROOT / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_CROSS_PARENT = Path("/workspace/triqto-data/step14_cross_motif_dataset")
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step14_cross_motif_training")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--mixture-config", type=Path, default=DEFAULT_MIXTURE_CONFIG)
    p.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    p.add_argument("--step10-product-dir", type=Path)
    p.add_argument("--cross-motif-product-dir", type=Path)
    p.add_argument("--step10c-benchmark-dir", type=Path)
    p.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def assert_contract(cfg: Mapping[str, Any]) -> None:
    if cfg.get("schema") != "triqto.v0_2.step14_cross_motif_generalization_training.v1" or cfg.get("status") != "FROZEN_BEFORE_STEP14_DATASET_GENERATION":
        raise RuntimeError("unexpected Step-14 protocol schema/status")
    t = cfg["training"]
    expected = {"root_batch_size": 32, "max_epochs": 30, "early_stopping_patience": 5,
                "learning_rate": 0.0001, "weight_decay": 0.0001, "gradient_clip_norm": 1.0,
                "effect_loss_weight": 1.0, "mechanism_loss_weight": 1.0}
    for key, value in expected.items():
        if float(t[key]) != float(value): raise RuntimeError(f"Step-14 training contract drift: {key}")
    if t["optimizer"] != "AdamW" or t["learning_rate_schedule"] != "constant" or [int(v) for v in t["seeds"]] != [1701,1702,1703]:
        raise RuntimeError("Step-14 optimizer/schedule/seed drift")
    if cfg["architecture"]["variant"] != "late_concat" or int(cfg["architecture"]["expected_trainable_parameter_count"]) != 453829:
        raise RuntimeError("Step-14 architecture drift")


def by_root(rows: Sequence[dict[str, str]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(rows): out[int(row["root_index"])].append(i)
    if any(len(v) != 13 for v in out.values()): raise RuntimeError("expected exactly 13 examples/root")
    return dict(out)


def resolve_cross_product(path: Path | None) -> Path:
    if path is not None: return path.expanduser().resolve()
    pointer = DEFAULT_CROSS_PARENT / "current_development_product.json"
    if not pointer.is_file(): raise RuntimeError("Step-14 development product pointer missing")
    return Path(read_json(pointer)["product_dir"]).expanduser().resolve()


def verify_cross_product(product: Path, cfg: Mapping[str, Any]):
    complete = read_json(product / "dataset_complete.json")
    if complete.get("schema") != "triqto.v0_2.step14_cross_motif_dataset.v1" or complete.get("status") != "COMPLETE_FROZEN_DEVELOPMENT":
        raise RuntimeError("Step-14 development product incomplete/wrong schema")
    if bool(complete.get("model_evaluated_before_freeze", True)) or bool(complete.get("qpu_executed", True)) or bool(complete.get("future_hardware_reserve_materialized", True)):
        raise RuntimeError("Step-14 development product violated frozen boundary")
    manifests = product / "manifests"
    for name, wanted in complete["manifest_hashes"].items():
        if baseline.sha256_file(manifests / name) != wanted: raise RuntimeError(f"Step-14 manifest hash mismatch: {name}")
    rows = baseline.read_csv(manifests / "example_manifest.csv"); grouped = by_root(rows)
    fit = sorted(root for root, idx in grouped.items() if {rows[i]["step14_partition"] for i in idx} == {"fit"})
    sel = sorted(root for root, idx in grouped.items() if {rows[i]["step14_partition"] for i in idx} == {"selection"})
    e = cfg["cross_motif_dataset"]["expected_counts"]
    if len(fit) != int(e["fit_roots"]) or len(sel) != int(e["selection_roots"]): raise RuntimeError("Step-14 development split count drift")
    if any({rows[i]["step14_partition"] for i in idx} - {"fit","selection"} for idx in grouped.values()): raise RuntimeError("outer/reserve rows present before training")
    return rows, grouped, fit, sel


def load_legacy(product: Path, mixture_cfg: Mapping[str, Any], experiment: Mapping[str, Any]):
    complete, bridge_rows, _ = step10b._verify_mixture_product(product, mixture_cfg)
    original_product, _, original_rows = step10b._verify_original_product(product, mixture_cfg, experiment)
    if complete.get("product_id") != "product_0f7112597501f7ea5fbe123b": raise RuntimeError("unexpected Step-10 mixture product")
    ofit, osel, _outer, oby = step7.split_root_indices(original_rows, experiment)
    bby = by_root(bridge_rows); bfit = step10b._bridge_roots(bridge_rows, "fit"); bsel = step10b._bridge_roots(bridge_rows, "selection")
    return original_product, original_rows, oby, ofit, osel, bridge_rows, bby, bfit, bsel


def instantiate(seed: int, cfg: Mapping[str, Any], benchmark: Path, device: torch.device) -> Step7DiagnosticModel:
    filename = str(cfg["warm_start"]["checkpoint_names"][str(seed)]); path = benchmark / filename
    if baseline.sha256_file(path) != str(cfg["warm_start"]["checkpoint_sha256"][filename]): raise RuntimeError(f"Step-10C checkpoint hash mismatch: {filename}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("initialization") != "warm_start" or payload.get("architecture") != "late_concat" or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"Step-10C checkpoint metadata mismatch: {filename}")
    model = Step7DiagnosticModel(variant="late_concat", initialization_seed=seed); model.load_state_dict(payload["state_dict"], strict=True); model.to(device)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 453829: raise RuntimeError("trainable parameter count drift")
    return model


def min_recall(pred: step7.PredictionSet) -> tuple[float, list[float]]:
    mask = pred.mechanism_mask; truth = pred.mechanism_truth_all[mask].astype(np.int64); guess = np.argmax(pred.mechanism_logits[mask], axis=1)
    cm = baseline.confusion_matrix(truth, guess, 3); recalls = [float(cm[i,i]/np.sum(cm[i])) if np.sum(cm[i]) else 0.0 for i in range(3)]
    return min(recalls), recalls


def selection_metrics(model: Step7DiagnosticModel, domains: Mapping[str, Sequence[step7.CachedBlock]], device: torch.device) -> dict[str, Any]:
    out = {}
    for name, blocks in domains.items():
        pred = step7.predict_blocks(model, blocks, device); summary = step7.selection_summary(pred); mr, recalls = min_recall(pred)
        out[name] = {**summary, "minimum_mechanism_recall": mr, "mechanism_recall": recalls}
    return out


def eligible(metrics: Mapping[str, Any], base: Mapping[str, Any]) -> bool:
    return (float(metrics["legacy_original"]["mechanism_balanced_accuracy"]) >= float(base["legacy_original"]["mechanism_balanced_accuracy"]) - 0.02 and
            float(metrics["legacy_bridge"]["mechanism_balanced_accuracy"]) >= float(base["legacy_bridge"]["mechanism_balanced_accuracy"]) - 0.02)


def candidate_key(metrics: Mapping[str, Any], epoch: int) -> tuple[float,float,float,int]:
    return (float(metrics["cross_motif"]["mechanism_balanced_accuracy"]), float(metrics["cross_motif"]["minimum_mechanism_recall"]),
            min(float(metrics["legacy_original"]["mechanism_balanced_accuracy"]), float(metrics["legacy_bridge"]["mechanism_balanced_accuracy"])), -int(epoch))


def better(key, best, delta: float) -> bool:
    if best is None: return True
    for a,b in zip(key[:3], best[:3]):
        if a > b + delta: return True
        if a < b - delta: return False
    return key[3] > best[3]


def train_epoch(model, optimizer, domains, seed, epoch, device, effect_weight, mechanism_weight, cfg):
    names = ("legacy_original","legacy_bridge","cross_motif"); orders = {}
    for j,name in enumerate(names):
        order = np.arange(len(domains[name]), dtype=np.int64); np.random.default_rng(seed*1_000_003 + epoch*97_409 + 101*j).shuffle(order); orders[name] = order
    steps = max(len(v) for v in orders.values()); totals = defaultdict(float); updates = 0; rotated = names[epoch%3:] + names[:epoch%3]
    t = cfg["training"]
    for i in range(steps):
        for name in rotated:
            block = domains[name][int(orders[name][i % len(orders[name])])]
            m = step7.train_one_epoch(model, optimizer, [block], seed=seed + {"legacy_original":11,"legacy_bridge":23,"cross_motif":37}[name], epoch=epoch,
                device=device, effect_class_weight=effect_weight, mechanism_class_weight=mechanism_weight,
                effect_loss_weight=float(t["effect_loss_weight"]), mechanism_loss_weight=float(t["mechanism_loss_weight"]), gradient_clip_norm=float(t["gradient_clip_norm"]))
            for k in ("effect_loss","mechanism_loss","total_loss","mean_preclip_gradient_norm"): totals[k] += float(m[k])
            updates += 1
    return {**{k: totals[k]/updates for k in totals}, "optimizer_steps": float(updates), "optimizer_blocks_per_domain": float(steps)}


def ensemble(models, blocks, device):
    preds = [step7.predict_blocks(m, blocks, device) for m in models]; ref = preds[0]
    for p in preds[1:]: step7.assert_prediction_alignment(ref, p, "Step-14 ensemble")
    return step7.PredictionSet(source_indices=ref.source_indices, root_indices=ref.root_indices, effect_truth=ref.effect_truth,
        mechanism_truth_all=ref.mechanism_truth_all, mechanism_mask=ref.mechanism_mask,
        effect_logits=np.mean(np.stack([p.effect_logits for p in preds]), axis=0), mechanism_logits=np.mean(np.stack([p.mechanism_logits for p in preds]), axis=0))


def binary_stats(pred: step7.PredictionSet, threshold: float) -> tuple[float,float]:
    guess = (pred.effect_logits >= threshold).astype(np.int8); cm = baseline.confusion_matrix(pred.effect_truth, guess, 2)
    ba = float(baseline.metrics_from_cm(cm)["balanced_accuracy"]); return ba, float(cm[0,1]/np.sum(cm[0])) if np.sum(cm[0]) else 0.0


def select_threshold(preds: Mapping[str, step7.PredictionSet], step10c_threshold: float):
    u = np.unique(np.concatenate([p.effect_logits for p in preds.values()]).astype(np.float64))
    candidates = u if len(u) == 1 else np.concatenate(([u[0]-1e-6], (u[:-1]+u[1:])/2, [u[-1]+1e-6]))
    best = None
    for th in candidates.tolist():
        stats = [binary_stats(p, th) for p in preds.values()]; bas = [x[0] for x in stats]; fprs = [x[1] for x in stats]
        key = (float(np.mean(bas)), -float(np.mean(fprs)), -abs(float(th)-step10c_threshold), -float(th))
        if best is None or key > best[0]: best = (key, float(th), bas, fprs)
    assert best is not None
    return best[1], {"objective_macro_effect_balanced_accuracy": best[0][0], "domain_balanced_accuracy": dict(zip(preds.keys(), best[2])),
                     "domain_false_positive_rate": dict(zip(preds.keys(), best[3])), "step10c_threshold_reference": step10c_threshold}


def main() -> None:
    args = parse_args(); config_path = args.config.expanduser().resolve(); cfg = read_json(config_path); assert_contract(cfg)
    mixture_cfg = read_json(args.mixture_config.expanduser().resolve()); experiment = read_json(args.step7_config.expanduser().resolve())
    cross_product = resolve_cross_product(args.cross_motif_product_dir); cross_rows, cross_by, cross_fit_roots, cross_sel_roots = verify_cross_product(cross_product, cfg)
    step10_product = (args.step10_product_dir or Path(cfg["legacy_training_domains"]["default_product_dir"])).expanduser().resolve()
    original_product, original_rows, original_by, ofit, osel, bridge_rows, bridge_by, bfit, bsel = load_legacy(step10_product, mixture_cfg, experiment)
    benchmark = (args.step10c_benchmark_dir or Path(cfg["warm_start"]["default_benchmark_dir"])).expanduser().resolve()
    if baseline.sha256_file(benchmark / "model_selection.json") != str(cfg["warm_start"]["model_selection_sha256"]): raise RuntimeError("Step-10C model-selection hash mismatch")
    step10c_threshold = float(read_json(benchmark / "model_selection.json")["ensemble_effect_thresholds"]["warm_start"])
    device = step7.resolve_device(args.device); batch = int(cfg["training"]["root_batch_size"])
    print("TRIQTO STEP 14 TRAINING — FIT/SELECTION ONLY; NO OUTER / NO QPU", flush=True)
    def mat(product, rows, by, roots, label): return step7.materialize_blocks(product=product, rows=rows, by_root=by, roots=roots, root_batch_size=batch, label=label, progress_every=args.progress_every)
    original_fit, original_sel = mat(original_product, original_rows, original_by, ofit, "step14-original-fit"), mat(original_product, original_rows, original_by, osel, "step14-original-selection")
    bridge_fit, bridge_sel = mat(step10_product, bridge_rows, bridge_by, bfit, "step14-bridge-fit"), mat(step10_product, bridge_rows, bridge_by, bsel, "step14-bridge-selection")
    cross_fit, cross_sel = mat(cross_product, cross_rows, cross_by, cross_fit_roots, "step14-cross-fit"), mat(cross_product, cross_rows, cross_by, cross_sel_roots, "step14-cross-selection")
    fit_domains = {"legacy_original": original_fit, "legacy_bridge": bridge_fit, "cross_motif": cross_fit}; sel_domains = {"legacy_original": original_sel, "legacy_bridge": bridge_sel, "cross_motif": cross_sel}
    effect_weight, mechanism_weight = step7.class_weights(original_fit + bridge_fit + cross_fit)
    identity = {"schema": SCHEMA, "protocol_config_sha256": baseline.sha256_file(config_path), "cross_dataset_sha256": baseline.sha256_file(cross_product/"dataset_complete.json"),
                "step10_dataset_sha256": baseline.sha256_file(step10_product/"dataset_complete.json"), "step10c_model_selection_sha256": baseline.sha256_file(benchmark/"model_selection.json"),
                "outer_accessed": False, "qpu_accessed": False}
    run_id = "training_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]; run_dir = args.output_parent.expanduser().resolve()/run_id; run_dir.mkdir(parents=True, exist_ok=True)
    selected_models, records, history = [], {}, []
    for seed in (1701,1702,1703):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model = instantiate(seed, cfg, benchmark, device); base_metrics = selection_metrics(model, sel_domains, device)
        t = cfg["training"]; optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["learning_rate"]), weight_decay=float(t["weight_decay"]))
        best_key = best_state = best_epoch = None; stale = 0
        for epoch in range(1, int(t["max_epochs"])+1):
            tm = train_epoch(model, optimizer, fit_domains, seed, epoch, device, effect_weight, mechanism_weight, cfg); metrics = selection_metrics(model, sel_domains, device)
            ok = eligible(metrics, base_metrics); key = candidate_key(metrics, epoch); improved = ok and better(key, best_key, float(t["early_stopping_min_delta"]))
            if improved: best_key, best_epoch, best_state, stale = key, epoch, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
            elif best_state is not None: stale += 1
            history.append({"seed": seed, "epoch": epoch, "eligible": ok, "selected_checkpoint": False, "train": tm, "selection": metrics, "baseline_step10c_selection": base_metrics})
            print(f"seed={seed} epoch={epoch:02d} cross_mech={metrics['cross_motif']['mechanism_balanced_accuracy']:.4f} cross_min_recall={metrics['cross_motif']['minimum_mechanism_recall']:.4f} eligible={'YES' if ok else 'NO'}", flush=True)
            if best_state is not None and stale >= int(t["early_stopping_patience"]): break
        if best_state is None or best_epoch is None: raise RuntimeError(f"Step-14 seed {seed} produced no retention-eligible checkpoint")
        model.load_state_dict(best_state, strict=True); model.to(device); model.eval()
        for row in history:
            if row["seed"] == seed and row["epoch"] == best_epoch: row["selected_checkpoint"] = True
        selected = selection_metrics(model, sel_domains, device); checkpoint = run_dir / f"step14__seed{seed}.pt"
        torch.save({"schema": SCHEMA, "seed": seed, "architecture": "late_concat", "state_dict": best_state,
                    "warm_start_source": cfg["warm_start"]["checkpoint_names"][str(seed)], "optimizer_state_reused": False,
                    "selected_epoch": best_epoch, "selected_checkpoint_retention_eligible": True, "selection": selected,
                    "baseline_step10c_selection": base_metrics}, checkpoint)
        records[str(seed)] = {"selected_epoch": best_epoch, "checkpoint": checkpoint.name, "checkpoint_sha256": baseline.sha256_file(checkpoint), "selection": selected, "baseline_step10c_selection": base_metrics}; selected_models.append(model)
    domain_preds = {name: ensemble(selected_models, blocks, device) for name,blocks in sel_domains.items()}; threshold, threshold_metrics = select_threshold(domain_preds, step10c_threshold)
    freeze = {"schema": "triqto.v0_2.step14_selection_freeze.v1", "status": "FROZEN_BEFORE_ANY_STEP14_OUTER_MATERIALIZATION",
              "protocol_config_sha256": baseline.sha256_file(config_path), "training_run_id": run_id, "all_three_seed_checkpoints_frozen": True,
              "seeds": [1701,1702,1703], "selected_seed_records": records, "checkpoint_hashes": {r["checkpoint"]:r["checkpoint_sha256"] for r in records.values()},
              "ensemble_effect_threshold": threshold, "threshold_selection": threshold_metrics, "step10c_threshold_reference": step10c_threshold,
              "outer_accessed_before_freeze": False, "step12_data_used_for_selection": False, "qpu_accessed": False}
    atomic_json(run_dir/"selection_freeze.json", freeze); atomic_json(run_dir/"training_history.json", {"schema": SCHEMA, "history": history})
    atomic_json(run_dir/"training_complete.json", {"schema": SCHEMA, "status": "COMPLETE_SELECTION_FROZEN_BEFORE_OUTER", "run_id": run_id,
                "identity": identity, "selection_freeze_sha256": baseline.sha256_file(run_dir/"selection_freeze.json"), "outer_accessed": False, "qpu_executed": False})
    atomic_json(args.output_parent.expanduser().resolve()/"current_training_run.json", {"schema":"triqto.v0_2.step14_current_training_run.v1", "run_id":run_id, "run_dir":str(run_dir), "selection_freeze_sha256":baseline.sha256_file(run_dir/"selection_freeze.json")})
    print("\nTRIQTO STEP 14 TRAINING COMPLETE — SELECTION FROZEN BEFORE OUTER"); print("Run:", run_id); print("Selected epochs:", {s:records[str(s)]["selected_epoch"] for s in (1701,1702,1703)}); print("Ensemble effect threshold:", threshold); print("Outer accessed: NO"); print("QPU executed: NO"); print("Selection freeze:", run_dir/"selection_freeze.json")


if __name__ == "__main__":
    main()
