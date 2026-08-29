#!/usr/bin/env python3
"""Evaluate frozen Step-10C checkpoints on Step-14 selection data before training.

This is a read-only model-evaluation stage with respect to all scientific datasets.
It consumes only already-materialized legacy selection partitions and the frozen
Step-14 development selection partition. It does not touch Step-14 outer/reserve
cohorts, does not train, and does not access a QPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import benchmark_step6_cheap_baselines as baseline
import run_step14_cross_motif_training as train
import run_step7_full_development_benchmark as step7

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
DEFAULT_MIXTURE_CONFIG = ROOT / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = ROOT / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step14_cross_motif_training")
SCHEMA = "triqto.v0_2.step14_pretraining_baseline.v1"


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


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = train.read_json(config_path)
    train.assert_contract(cfg)
    mixture_cfg = train.read_json(args.mixture_config.expanduser().resolve())
    experiment = train.read_json(args.step7_config.expanduser().resolve())

    cross_product = train.resolve_cross_product(args.cross_motif_product_dir)
    cross_rows, cross_by, _cross_fit_roots, cross_sel_roots = train.verify_cross_product(
        cross_product, cfg
    )
    step10_product = (
        args.step10_product_dir
        or Path(cfg["legacy_training_domains"]["default_product_dir"])
    ).expanduser().resolve()
    (
        original_product,
        original_rows,
        original_by,
        _ofit,
        osel,
        bridge_rows,
        bridge_by,
        _bfit,
        bsel,
    ) = train.load_legacy(step10_product, mixture_cfg, experiment)

    benchmark = (
        args.step10c_benchmark_dir
        or Path(cfg["warm_start"]["default_benchmark_dir"])
    ).expanduser().resolve()
    model_selection_path = benchmark / "model_selection.json"
    if baseline.sha256_file(model_selection_path) != str(
        cfg["warm_start"]["model_selection_sha256"]
    ):
        raise RuntimeError("Step-10C model-selection hash mismatch")

    device = step7.resolve_device(args.device)
    batch = int(cfg["training"]["root_batch_size"])
    print(
        "TRIQTO STEP 14 PRETRAINING BASELINE — SELECTION ONLY; NO TRAINING / NO OUTER / NO QPU",
        flush=True,
    )

    def mat(product, rows, by, roots, label):
        return step7.materialize_blocks(
            product=product,
            rows=rows,
            by_root=by,
            roots=roots,
            root_batch_size=batch,
            label=label,
            progress_every=args.progress_every,
        )

    selection_domains = {
        "legacy_original": mat(
            original_product,
            original_rows,
            original_by,
            osel,
            "step14-baseline-original-selection",
        ),
        "legacy_bridge": mat(
            step10_product,
            bridge_rows,
            bridge_by,
            bsel,
            "step14-baseline-bridge-selection",
        ),
        "cross_motif": mat(
            cross_product,
            cross_rows,
            cross_by,
            cross_sel_roots,
            "step14-baseline-cross-selection",
        ),
    }

    per_seed: dict[str, Any] = {}
    for seed in (1701, 1702, 1703):
        model = train.instantiate(seed, cfg, benchmark, device)
        metrics = train.selection_metrics(model, selection_domains, device)
        filename = str(cfg["warm_start"]["checkpoint_names"][str(seed)])
        per_seed[str(seed)] = {
            "checkpoint": filename,
            "checkpoint_sha256": baseline.sha256_file(benchmark / filename),
            "selection": metrics,
        }
        print(
            f"seed={seed} cross_mech={metrics['cross_motif']['mechanism_balanced_accuracy']:.6f} "
            f"cross_min_recall={metrics['cross_motif']['minimum_mechanism_recall']:.6f}",
            flush=True,
        )

    identity = {
        "schema": SCHEMA,
        "protocol_config_sha256": baseline.sha256_file(config_path),
        "cross_dataset_sha256": baseline.sha256_file(cross_product / "dataset_complete.json"),
        "step10_dataset_sha256": baseline.sha256_file(step10_product / "dataset_complete.json"),
        "step10c_model_selection_sha256": baseline.sha256_file(model_selection_path),
        "selection_domains": ["legacy_original", "legacy_bridge", "cross_motif"],
        "seeds": [1701, 1702, 1703],
        "training_performed": False,
        "outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_accessed": False,
    }
    baseline_id = "pretraining_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    baseline_dir = output_parent / "pretraining_baselines" / baseline_id
    baseline_dir.mkdir(parents=True, exist_ok=True)
    output_path = baseline_dir / "pretraining_baseline.json"
    payload = {
        **identity,
        "status": "COMPLETE_PRETRAINING_SELECTION_BASELINE",
        "baseline_id": baseline_id,
        "per_seed": per_seed,
    }
    train.atomic_json(output_path, payload)
    train.atomic_json(
        output_parent / "current_pretraining_baseline.json",
        {
            "schema": "triqto.v0_2.step14_current_pretraining_baseline.v1",
            "baseline_id": baseline_id,
            "baseline_path": str(output_path),
            "pretraining_baseline_sha256": baseline.sha256_file(output_path),
        },
    )

    print("\nTRIQTO STEP 14 PRETRAINING BASELINE COMPLETE", flush=True)
    print("Baseline:", baseline_id, flush=True)
    print("Training performed: NO", flush=True)
    print("Outer accessed: NO", flush=True)
    print("Future-hardware reserve accessed: NO", flush=True)
    print("QPU accessed: NO", flush=True)
    print("Artifact:", output_path, flush=True)


if __name__ == "__main__":
    main()
