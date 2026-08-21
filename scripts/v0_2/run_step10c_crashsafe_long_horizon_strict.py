#!/usr/bin/env python3
"""Authoritative Step-10C entrypoint with strict fresh-outer sequencing.

This module reuses the frozen Step-10C crash-safe training/checkpoint helpers,
but enforces one additional reviewer-facing boundary: all six per-seed training
and selection trajectories must finish before any fresh-outer NPZ artifact is
materialized or any model forward pass is run on the fresh outer cohort.

Scientific settings are unchanged.  This is sequencing/provenance hardening.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import run_step10c_crashsafe_long_horizon as impl

baseline = impl.baseline
step7 = impl.step7
step10b = impl.step10b
SCHEMA = impl.SCHEMA


def _write_outer_boundary_marker(
    path: Path,
    *,
    benchmark_id: str,
    outer_product_id: str,
    selected_records: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        f"{initialization}__seed{seed}"
        for initialization in ("warm_start", "scratch")
        for seed in (1701, 1702, 1703)
    }
    observed = set(selected_records)
    if observed != expected:
        raise RuntimeError(
            "fresh outer access blocked: all six Step-10C selections must be complete "
            f"before outer materialization; observed={sorted(observed)}"
        )
    if not all("selected_epoch" in selected_records[key] for key in sorted(expected)):
        raise RuntimeError("fresh outer access blocked: selected epoch missing")
    marker = {
        "schema": "triqto.v0_2.step10c_outer_evaluation_boundary.v1",
        "benchmark_id": benchmark_id,
        "fresh_outer_product_id": outer_product_id,
        "selected_run_count_before_outer_materialization": 6,
        "all_six_seed_selections_complete": True,
        "fresh_outer_materialized_during_training": False,
        "fresh_outer_used_for_epoch_selection": False,
        "fresh_outer_used_for_threshold_selection": False,
        "selected_epochs": {
            key: int(selected_records[key]["selected_epoch"]) for key in sorted(expected)
        },
        "started_unix": time.time(),
    }
    impl.atomic_json_fsync(path, marker)
    return marker


def main() -> None:
    args = impl.parse_args()
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
    impl._assert_step10c_delta_only_horizon(config)

    mixture_product = step10b._resolve_mixture_product(args.mixture_product_dir)
    mixture_complete, bridge_rows, _old_parent_by_root = step10b._verify_mixture_product(
        mixture_product, mixture_cfg
    )
    original_product, original_complete, original_rows = step10b._verify_original_product(
        mixture_product, mixture_cfg, experiment
    )
    outer_product = impl._resolve_outer_product(args.fresh_outer_product_dir)
    outer_complete, fresh_original_rows, fresh_bridge_rows, fresh_parent_by_root = impl._verify_outer_product(
        outer_product
    )
    bundle, bundle_complete = step10b._resolve_step9a_bundle(args.step9a_bundle_dir, config)
    seeds = [int(value) for value in config["seeds"]]
    if seeds != [int(value) for value in bundle_complete["seeds"]]:
        raise RuntimeError("Step-10C seeds differ from frozen Step-9A bundle seeds")

    # Historical source manifests may be read for integrity/split filtering, but
    # spent Step-10B outer artifacts are never materialized.
    original_fit_roots, original_selection_roots, _spent_original_outer_roots, original_by_root = step7.split_root_indices(
        original_rows, experiment
    )
    bridge_by_root = step10b._by_root(bridge_rows)
    bridge_fit_roots = step10b._bridge_roots(bridge_rows, "fit")
    bridge_selection_roots = step10b._bridge_roots(bridge_rows, "selection")

    # Fresh outer manifests are verified before training, but their NPZ artifacts
    # are deliberately not materialized until after all six selections finish.
    fresh_original_by_root = step10b._by_root(fresh_original_rows)
    fresh_bridge_by_root = step10b._by_root(fresh_bridge_rows)
    fresh_original_roots = sorted(fresh_original_by_root)
    fresh_bridge_roots = sorted(fresh_bridge_by_root)

    device = step7.resolve_device(args.device)
    root_batch_size = int(config["training"]["root_batch_size"])
    print("STEP 10C STRICT CRASH-SAFE LONG-HORIZON — FRESH OUTER / NO QPU", flush=True)
    print(f"device: {device}", flush=True)
    print(
        f"fit/selection roots original={len(original_fit_roots)}/{len(original_selection_roots)} "
        f"bridge={len(bridge_fit_roots)}/{len(bridge_selection_roots)}",
        flush=True,
    )
    print(
        f"fresh outer roots verified but NOT materialized: "
        f"original={len(fresh_original_roots)} bridge={len(fresh_bridge_roots)}",
        flush=True,
    )

    original_fit = step7.materialize_blocks(
        product=original_product,
        rows=original_rows,
        by_root=original_by_root,
        roots=original_fit_roots,
        root_batch_size=root_batch_size,
        label="step10c-original-fit",
        progress_every=args.progress_every,
    )
    original_selection = step7.materialize_blocks(
        product=original_product,
        rows=original_rows,
        by_root=original_by_root,
        roots=original_selection_roots,
        root_batch_size=root_batch_size,
        label="step10c-original-selection",
        progress_every=args.progress_every,
    )
    bridge_fit = step7.materialize_blocks(
        product=mixture_product,
        rows=bridge_rows,
        by_root=bridge_by_root,
        roots=bridge_fit_roots,
        root_batch_size=root_batch_size,
        label="step10c-bridge-fit",
        progress_every=args.progress_every,
    )
    bridge_selection = step7.materialize_blocks(
        product=mixture_product,
        rows=bridge_rows,
        by_root=bridge_by_root,
        roots=bridge_selection_roots,
        root_batch_size=root_batch_size,
        label="step10c-bridge-selection",
        progress_every=args.progress_every,
    )

    effect_class_weight, mechanism_class_weight = step7.class_weights(
        list(original_fit) + list(bridge_fit)
    )
    tolerance = float(config["selection"]["original_domain_retention_tolerance"])
    retention_floors: dict[int, dict[str, float]] = {}
    epoch0_warm_selection: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        model = step10b._initialize_model(
            initialization="warm_start",
            seed=seed,
            experiment=experiment,
            device=device,
            bundle=bundle,
            config=config,
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
        "implementation_runner_sha256": baseline.sha256_file(Path(impl.__file__).resolve()),
        "mixture_product_id": mixture_complete["product_id"],
        "original_product_id": original_complete["product_id"],
        "fresh_outer_product_id": outer_complete["product_id"],
        "fresh_outer_dataset_complete_sha256": baseline.sha256_file(
            outer_product / "dataset_complete.json"
        ),
        "step9a_bundle_id": bundle_complete["bundle_id"],
    }
    benchmark_identity = {
        **benchmark_key_identity,
        "execution_device": str(device),
        "runtime_fingerprint": impl._runtime_fingerprint(device),
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
    selected_states: dict[tuple[str, int], dict[str, Any]] = {}
    selection_predictions: dict[tuple[str, int, str], step7.PredictionSet] = {}
    any_resumed = False

    # TRAINING/SELECTION PHASE ONLY.  Fresh outer NPZ artifacts are unavailable
    # to this loop by construction.
    for initialization in [str(value) for value in config["initializations"]]:
        for seed in seeds:
            print(
                f"\nTraining Step-10C initialization={initialization} seed={seed}",
                flush=True,
            )
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
                "trainable_parameter_count": int(
                    config["architecture"]["expected_trainable_parameter_count"]
                ),
                "execution_device": str(device),
                "runtime_fingerprint": impl._runtime_fingerprint(device),
            }
            record, history, selected_state = impl._train_seed_crashsafe(
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
                run_dir=work_root / f"{initialization}__seed{seed}",
                identity=run_identity,
                allow_resume=not args.no_resume,
            )
            key = f"{initialization}__seed{seed}"
            selected_records[key] = record
            histories.extend(history)
            selected_states[(initialization, seed)] = selected_state
            any_resumed = any_resumed or bool(record["resumed_from_checkpoint"])

            model = step10b._initialize_model(
                initialization=initialization,
                seed=seed,
                experiment=experiment,
                device=device,
                bundle=bundle,
                config=config,
            )
            model.load_state_dict(selected_state, strict=True)
            model.to(device)
            selection_predictions[(initialization, seed, "original")] = step7.predict_blocks(
                model, original_selection, device
            )
            selection_predictions[(initialization, seed, "bridge")] = step7.predict_blocks(
                model, bridge_selection, device
            )
            del model

    # Freeze the selection-complete boundary before touching a fresh-outer NPZ.
    boundary_marker = _write_outer_boundary_marker(
        work_root / "outer_evaluation_started.json",
        benchmark_id=benchmark_id,
        outer_product_id=outer_complete["product_id"],
        selected_records=selected_records,
    )
    print(
        "All six checkpoint selections complete. Fresh outer materialization now unlocked.",
        flush=True,
    )

    # OUTER EVALUATION PHASE.  No optimizer step, checkpoint choice, epoch choice,
    # or threshold choice may depend on these artifacts.
    fresh_original_outer = step7.materialize_blocks(
        product=outer_product,
        rows=fresh_original_rows,
        by_root=fresh_original_by_root,
        roots=fresh_original_roots,
        root_batch_size=root_batch_size,
        label="step10c-fresh-original-outer",
        progress_every=args.progress_every,
    )
    fresh_bridge_outer = step7.materialize_blocks(
        product=outer_product,
        rows=fresh_bridge_rows,
        by_root=fresh_bridge_by_root,
        roots=fresh_bridge_roots,
        root_batch_size=root_batch_size,
        label="step10c-fresh-bridge-outer",
        progress_every=args.progress_every,
    )

    outer_predictions: dict[tuple[str, int, str], step7.PredictionSet] = {}
    for initialization in [str(value) for value in config["initializations"]]:
        for seed in seeds:
            model = step10b._initialize_model(
                initialization=initialization,
                seed=seed,
                experiment=experiment,
                device=device,
                bundle=bundle,
                config=config,
            )
            model.load_state_dict(selected_states[(initialization, seed)], strict=True)
            model.to(device)
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
            initialization="warm_start",
            seed=seed,
            experiment=experiment,
            device=device,
            bundle=bundle,
            config=config,
        )
        baseline_outer_members.append(
            step7.predict_blocks(model, fresh_original_outer, device)
        )
        del model
    step9a_baseline_outer = step10b._mean_prediction(baseline_outer_members)
    step9a_threshold = float(bundle_complete["deployment_effect_threshold"])

    replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    confidence = float(config["evaluation"]["confidence_level"])
    outer_metric_rows: list[dict[str, Any]] = []
    bootstrap_by_domain_task: dict[str, dict[str, dict[str, Mapping[str, np.ndarray]]]] = {
        "original": {
            "effect_detection": {},
            "mechanism_diagnosis": {},
            "integrated_diagnosis": {},
        },
        "bridge": {
            "effect_detection": {},
            "mechanism_diagnosis": {},
            "integrated_diagnosis": {},
        },
    }
    ensemble_predictions: dict[tuple[str, str], step7.PredictionSet] = {}
    ensemble_thresholds: dict[str, float] = {}

    baseline_rows, _ = step10b._bootstrap_domain(
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

    for initialization in [str(value) for value in config["initializations"]]:
        original_sel_ensemble = step10b._mean_prediction(
            [selection_predictions[(initialization, seed, "original")] for seed in seeds]
        )
        bridge_sel_ensemble = step10b._mean_prediction(
            [selection_predictions[(initialization, seed, "bridge")] for seed in seeds]
        )
        threshold = step10b._selection_threshold(
            original_sel_ensemble, bridge_sel_ensemble
        )
        ensemble_thresholds[initialization] = threshold

        original_outer_ensemble = step10b._mean_prediction(
            [outer_predictions[(initialization, seed, "original")] for seed in seeds]
        )
        bridge_outer_ensemble = step10b._mean_prediction(
            [outer_predictions[(initialization, seed, "bridge")] for seed in seeds]
        )
        bridge_outer_grouped = step10b._replace_groups(
            bridge_outer_ensemble, fresh_parent_by_root
        )
        ensemble_predictions[(initialization, "original")] = original_outer_ensemble
        ensemble_predictions[(initialization, "bridge")] = bridge_outer_grouped

        for domain, prediction in (
            ("original", original_outer_ensemble),
            ("bridge", bridge_outer_grouped),
        ):
            rows, boots = step10b._bootstrap_domain(
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
    for initialization in [str(value) for value in config["initializations"]]:
        old_effect = lookup[("original", initialization, "effect_detection")]
        old_mech = lookup[("original", initialization, "mechanism_diagnosis")]
        bridge_mech = lookup[("bridge", initialization, "mechanism_diagnosis")]
        old_effect_drop = (
            float(baseline_old_effect["balanced_accuracy"])
            - float(old_effect["balanced_accuracy"])
        )
        old_mech_drop = (
            float(baseline_old_mech["balanced_accuracy"])
            - float(old_mech["balanced_accuracy"])
        )
        bridge_pass = bool(
            float(bridge_mech["balanced_accuracy"])
            >= float(bridge_gate_cfg["balanced_accuracy_minimum"])
            and float(bridge_mech["balanced_accuracy_ci_low"])
            >= float(bridge_gate_cfg["bootstrap_ci_lower_minimum"])
            and step10b._minimum_mechanism_recall(bridge_mech)
            >= float(bridge_gate_cfg["minimum_class_recall"])
        )
        retention_pass = bool(
            old_effect_drop
            <= float(retention_cfg["effect_balanced_accuracy_max_drop"])
            and old_mech_drop
            <= float(retention_cfg["mechanism_balanced_accuracy_max_drop"])
        )
        all_eligible = all(
            bool(
                selected_records[f"{initialization}__seed{seed}"][
                    "selected_checkpoint_retention_eligible"
                ]
            )
            for seed in seeds
        )
        gates[initialization] = {
            "bridge_gate_pass": bridge_pass,
            "original_retention_gate_pass": retention_pass,
            "all_seed_selected_checkpoints_retention_eligible": all_eligible,
            "full_step10c_gate_pass": bool(
                bridge_pass and retention_pass and all_eligible
            ),
            "original_effect_ba_drop_vs_step9a": old_effect_drop,
            "original_mechanism_ba_drop_vs_step9a": old_mech_drop,
            "bridge_mechanism_balanced_accuracy": float(
                bridge_mech["balanced_accuracy"]
            ),
            "bridge_mechanism_ci_low": float(
                bridge_mech["balanced_accuracy_ci_low"]
            ),
            "bridge_minimum_mechanism_recall": step10b._minimum_mechanism_recall(
                bridge_mech
            ),
        }

    warm_pass = bool(gates["warm_start"]["full_step10c_gate_pass"])
    scratch_pass = bool(gates["scratch"]["full_step10c_gate_pass"])
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
        <= float(gates["warm_start"]["original_mechanism_ba_drop_vs_step9a"])
        + 0.01
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
        for initialization in [str(value) for value in config["initializations"]]:
            for seed in seeds:
                name = f"{initialization}__seed{seed}.pt"
                path = staging / name
                record = selected_records[f"{initialization}__seed{seed}"]
                impl.atomic_torch_save(
                    path,
                    {
                        "schema": SCHEMA,
                        "initialization": initialization,
                        "seed": int(seed),
                        "architecture": "late_concat",
                        "selected_epoch": int(record["selected_epoch"]),
                        "selected_checkpoint_retention_eligible": bool(
                            record["selected_checkpoint_retention_eligible"]
                        ),
                        "source_step9a_bundle_id": (
                            bundle_complete["bundle_id"]
                            if initialization == "warm_start"
                            else None
                        ),
                        "optimizer_state_present": False,
                        "state_dict": selected_states[(initialization, seed)],
                    },
                )
                checkpoint_hashes[name] = baseline.sha256_file(path)

        baseline.write_csv(staging / "training_history.csv", histories)
        baseline.write_csv(staging / "outer_domain_metrics.csv", outer_metric_rows)
        baseline.write_csv(
            staging / "paired_initialization_differences.csv", paired_rows
        )
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
            staging / "evaluation_boundary.json",
            {
                **boundary_marker,
                "fresh_outer_materialized_after_all_six_selection_runs": True,
                "outer_metric_computation_after_boundary": True,
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
                "step10b_outer_evaluated": False,
                "architecture_changed": False,
                "optimizer_state_resumed_for_crash_recovery": bool(any_resumed),
                "fresh_run_optimizer_state_from_step9a_or_step10b_reused": False,
                "warm_start_state_dict_reused": True,
                "fresh_outer_validation_used_for_selection": False,
                "fresh_outer_materialized_only_after_all_six_selections": True,
            },
        )

        arrays: dict[str, np.ndarray] = {}
        for initialization in [str(value) for value in config["initializations"]]:
            for domain in ("original", "bridge"):
                prediction = ensemble_predictions[(initialization, domain)]
                prefix = f"{initialization}__{domain}"
                arrays[f"{prefix}__effect_truth"] = prediction.effect_truth
                arrays[f"{prefix}__mechanism_truth_all"] = (
                    prediction.mechanism_truth_all
                )
                arrays[f"{prefix}__mechanism_mask"] = prediction.mechanism_mask
                arrays[f"{prefix}__effect_logits"] = prediction.effect_logits
                arrays[f"{prefix}__mechanism_logits"] = prediction.mechanism_logits
                arrays[f"{prefix}__bootstrap_group"] = prediction.root_indices
        np.savez_compressed(staging / "outer_predictions.npz", **arrays)

        files = [
            "training_history.csv",
            "outer_domain_metrics.csv",
            "paired_initialization_differences.csv",
            "model_selection.json",
            "evaluation_boundary.json",
            "decision.json",
            "outer_predictions.npz",
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
            "fresh_outer_dataset_complete_sha256": baseline.sha256_file(
                outer_product / "dataset_complete.json"
            ),
            "step9a_bundle_id": bundle_complete["bundle_id"],
            "architecture": "late_concat",
            "trainable_parameter_count": int(
                config["architecture"]["expected_trainable_parameter_count"]
            ),
            "max_epochs": int(config["training"]["max_epochs"]),
            "any_run_resumed": bool(any_resumed),
            "qpu_executed": False,
            "step10b_outer_evaluated": False,
            "fresh_outer_used_for_selection": False,
            "fresh_outer_materialized_after_all_six_selections": True,
            "file_hashes": {
                name: baseline.sha256_file(staging / name) for name in files
            },
        }
        baseline.atomic_json(staging / "benchmark_complete.json", completion)
        os.replace(staging, output)
        impl._fsync_dir(output_parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 10C STRICT CRASH-SAFE LONG-HORIZON COMPLETE\n")
    for initialization in ("warm_start", "scratch"):
        gate = gates[initialization]
        print(
            f"{initialization}: bridge_mech_BA="
            f"{gate['bridge_mechanism_balanced_accuracy']:.4f} "
            f"CI_low={gate['bridge_mechanism_ci_low']:.4f} "
            f"min_recall={gate['bridge_minimum_mechanism_recall']:.4f} "
            f"old_mech_drop={gate['original_mechanism_ba_drop_vs_step9a']:.4f} "
            f"old_effect_drop={gate['original_effect_ba_drop_vs_step9a']:.4f} "
            f"full_gate={'PASS' if gate['full_step10c_gate_pass'] else 'FAIL'}"
        )
    print(f"DECISION GATE: {decision}")
    print(f"Any crash-recovery resume used: {'YES' if any_resumed else 'NO'}")
    print("Fresh outer materialized before all six selections: NO")
    print("Step-10B outer evaluated: NO")
    print("QPU executed: NO")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
