#!/usr/bin/env python3
"""Final simulator-development LR refinement before IBM hardware.

Step 10D is development-selection only. It never materializes or evaluates any
outer cohort and never executes a QPU. The only change to the Step-10C
warm-start training trajectory is LR=3e-4 for epochs 1-20 and LR=1e-4 for
21-40. After this benchmark, the frozen protocol requires moving to hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import benchmark_step6_cheap_baselines as baseline
import run_step7_full_development_benchmark as step7
import run_step10_warmstart_vs_scratch as step10b
import run_step10c_crashsafe_long_horizon as step10c

SCHEMA = "triqto.v0_2.step10d_final_simulator_lr_refinement.v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10d_final_simulator_lr_refinement.json"
DEFAULT_MIXTURE_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step10_training_mixture.json"
DEFAULT_STEP7_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step7_structured_diagnostic_model.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step10d_final_simulator_lr_refinement")
DEFAULT_STEP10C_BENCHMARK = Path(
    "/workspace/triqto-data/step10c_crashsafe_long_horizon/benchmark_f9478da45d68795655259054"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mixture-config", type=Path, default=DEFAULT_MIXTURE_CONFIG)
    parser.add_argument("--step7-config", type=Path, default=DEFAULT_STEP7_CONFIG)
    parser.add_argument("--mixture-product-dir", type=Path)
    parser.add_argument("--step9a-bundle-dir", type=Path)
    parser.add_argument("--step10c-benchmark-dir", type=Path, default=DEFAULT_STEP10C_BENCHMARK)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _lr_for_epoch(training: Mapping[str, Any], epoch: int) -> float:
    if epoch < 1:
        raise ValueError("training epoch must be >= 1")
    matches = [
        row for row in training["learning_rate_schedule"]
        if int(row["epoch_start"]) <= epoch <= int(row["epoch_end"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Step-10D LR schedule does not uniquely cover epoch {epoch}")
    return float(matches[0]["learning_rate"])


def _assert_frozen_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_STEP10D_OUTCOME":
        raise RuntimeError("unexpected Step-10D config schema/status")
    if not bool(config.get("final_simulator_intervention_before_ibm_hardware")):
        raise RuntimeError("Step-10D final-simulator hard stop is not frozen")
    if config["architecture"]["variant"] != "late_concat":
        raise RuntimeError("Step-10D architecture must remain late_concat")
    if int(config["architecture"]["expected_trainable_parameter_count"]) != 453829:
        raise RuntimeError("Step-10D parameter-count contract changed")
    if config["development_candidate"]["initialization"] != "warm_start":
        raise RuntimeError("Step-10D must refine warm_start only")
    if bool(config["development_candidate"].get("scratch_rerun")):
        raise RuntimeError("Step-10D scratch rerun is forbidden")

    old = baseline.read_json(step10c.DEFAULT_CONFIG)
    frozen_training_keys = (
        "root_batch_size", "max_epochs", "early_stopping_patience",
        "early_stopping_min_delta", "optimizer", "weight_decay",
        "gradient_clip_norm", "effect_loss_weight", "mechanism_loss_weight",
        "domain_schedule",
    )
    for key in frozen_training_keys:
        if config["training"][key] != old["training"][key]:
            raise RuntimeError(f"Step-10D changed frozen training field {key}")
    expected_schedule = [
        {"epoch_start": 1, "epoch_end": 20, "learning_rate": 0.0003},
        {"epoch_start": 21, "epoch_end": 40, "learning_rate": 0.0001},
    ]
    if config["training"]["learning_rate_schedule"] != expected_schedule:
        raise RuntimeError("Step-10D LR schedule differs from frozen 3e-4 -> 1e-4 contract")
    if not bool(config["outer_and_qpu_boundary"]["hardware_stage_proceeds_regardless_of_step10d_improvement"]):
        raise RuntimeError("Step-10D must proceed to hardware regardless of outcome")
    if config["outer_and_qpu_boundary"]["step10c_spent_outer_access"] != "FORBIDDEN":
        raise RuntimeError("Step-10C spent outer access must be forbidden")


def _verify_step10c_reference(
    benchmark: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    benchmark = benchmark.expanduser().resolve()
    complete = baseline.read_json(benchmark / "benchmark_complete.json")
    ref = config["step10c_reference"]
    if complete.get("status") != "COMPLETE":
        raise RuntimeError("Step-10C reference benchmark is incomplete")
    if complete.get("benchmark_id") != ref["benchmark_id"]:
        raise RuntimeError("Step-10C benchmark identity mismatch")
    if complete.get("decision") != ref["official_decision"]:
        raise RuntimeError("Step-10C frozen decision mismatch")
    if bool(complete.get("qpu_executed", True)):
        raise RuntimeError("unexpected QPU execution in Step-10C reference")

    states: dict[int, dict[str, torch.Tensor]] = {}
    for seed_text, name in config["step10c_candidate_reference"]["checkpoint_names"].items():
        seed = int(seed_text)
        path = benchmark / name
        expected = complete.get("file_hashes", {}).get(name)
        if expected is None or baseline.sha256_file(path) != expected:
            raise RuntimeError(f"Step-10C checkpoint hash mismatch: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("initialization") != "warm_start" or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"Step-10C warm checkpoint metadata mismatch: {name}")
        if payload.get("architecture") != "late_concat":
            raise RuntimeError(f"Step-10C warm checkpoint architecture mismatch: {name}")
        if not bool(payload.get("selected_checkpoint_retention_eligible")):
            raise RuntimeError(f"Step-10C warm checkpoint is not retention-eligible: {name}")
        states[seed] = {
            key: value.detach().cpu().clone()
            for key, value in payload["state_dict"].items()
        }
    return complete, states


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
        "model_state_dict": step10c._state_cpu(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eligible_key": best_eligible_key,
        "best_eligible_state": dict(best_eligible_state) if best_eligible_state is not None else None,
        "best_eligible_epoch": best_eligible_epoch,
        "best_unconstrained_key": best_unconstrained_key,
        "best_unconstrained_state": dict(best_unconstrained_state) if best_unconstrained_state is not None else None,
        "best_unconstrained_epoch": best_unconstrained_epoch,
        "history": [dict(row) for row in history],
        "rng_state": step10c._rng_snapshot(),
        "run_complete": bool(run_complete),
    }


def _train_seed(
    *, seed: int, experiment: Mapping[str, Any], config: Mapping[str, Any],
    bundle: Path, original_fit: Sequence[step7.CachedBlock],
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
        initialization="warm_start", seed=seed, experiment=experiment,
        device=device, bundle=bundle, config=config,
    )
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=_lr_for_epoch(training, 1),
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    min_delta = float(training["early_stopping_min_delta"])

    history: list[dict[str, Any]] = []
    best_eligible_key = None
    best_eligible_state = None
    best_eligible_epoch = None
    best_unconstrained_key = None
    best_unconstrained_state = None
    best_unconstrained_epoch = None
    stale = 0
    completed_epoch = 0
    resumed = False

    if resume_path.is_file():
        if not allow_resume:
            raise RuntimeError(f"resume checkpoint exists but --no-resume was requested: {resume_path}")
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("kind") != "resume":
            raise RuntimeError(f"invalid Step-10D resume checkpoint: {resume_path}")
        if not step10c._identity_equal(identity, payload.get("identity", {})):
            raise RuntimeError(f"Step-10D resume identity mismatch seed={seed}")
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
        step10c._restore_rng(payload["rng_state"])
        resumed = True
        print(f"RESUME Step-10D warm_start seed={seed} after epoch={completed_epoch} stale={stale}", flush=True)
    else:
        original_pred = step7.predict_blocks(model, original_selection, device)
        bridge_pred = step7.predict_blocks(model, bridge_selection, device)
        original_summary = step7.selection_summary(original_pred)
        bridge_summary = step7.selection_summary(bridge_pred)
        eligible = step10b._eligible(original_summary, floor)
        key = step10b._candidate_key(original_summary, bridge_summary, 0)
        state = step10c._state_cpu(model)
        best_unconstrained_key = key
        best_unconstrained_state = state
        best_unconstrained_epoch = 0
        improved = False
        if eligible:
            best_eligible_key = key
            best_eligible_state = state
            best_eligible_epoch = 0
            improved = True
            step10c.atomic_torch_save(best_path, {
                "schema": SCHEMA, "kind": "best", "identity": dict(identity),
                "epoch": 0, "candidate_key": key, "state_dict": state,
            })
        history.append({
            "initialization": "warm_start", "seed": seed, "epoch": 0,
            "learning_rate": None,
            "selection_eligible": bool(eligible),
            "original_selection_effect_ba": float(original_summary["effect_balanced_accuracy"]),
            "original_selection_mechanism_ba": float(original_summary["mechanism_balanced_accuracy"]),
            "bridge_selection_effect_ba": float(bridge_summary["effect_balanced_accuracy"]),
            "bridge_selection_mechanism_ba": float(bridge_summary["mechanism_balanced_accuracy"]),
            "retention_floor_effect_ba": float(floor["effect_balanced_accuracy"]),
            "retention_floor_mechanism_ba": float(floor["mechanism_balanced_accuracy"]),
            "eligible_checkpoint_improved": bool(improved),
        })
        step10c.atomic_torch_save(
            resume_path,
            _resume_payload(
                identity=identity, model=model, optimizer=optimizer,
                completed_epoch=0, stale=stale,
                best_eligible_key=best_eligible_key, best_eligible_state=best_eligible_state,
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
        lr = _lr_for_epoch(training, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        train_metrics = step10c._domain_balanced_epoch_with_telemetry(
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
        state = step10c._state_cpu(model)

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
            step10c.atomic_torch_save(best_path, {
                "schema": SCHEMA, "kind": "best", "identity": dict(identity),
                "epoch": int(epoch), "candidate_key": key, "state_dict": state,
            })
        elif eligible and best_eligible_state is not None:
            stale += 1

        row: dict[str, Any] = {
            "initialization": "warm_start", "seed": seed, "epoch": int(epoch),
            "learning_rate": float(lr),
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

        step10c.atomic_torch_save(
            resume_path,
            _resume_payload(
                identity=identity, model=model, optimizer=optimizer,
                completed_epoch=completed_epoch, stale=stale,
                best_eligible_key=best_eligible_key, best_eligible_state=best_eligible_state,
                best_eligible_epoch=best_eligible_epoch,
                best_unconstrained_key=best_unconstrained_key,
                best_unconstrained_state=best_unconstrained_state,
                best_unconstrained_epoch=best_unconstrained_epoch,
                history=history, run_complete=False,
            ),
        )
        step10c.atomic_json_fsync(progress_path, {
            "schema": SCHEMA, "status": "RUNNING", "seed": seed,
            "current_epoch": epoch, "max_epochs": max_epochs,
            "learning_rate": float(lr), "selection_eligible": bool(eligible),
            "stale_eligible_epochs": int(stale), "best_eligible_epoch": best_eligible_epoch,
            "bridge_mechanism_ba": row["bridge_selection_mechanism_ba"],
            "original_mechanism_ba": row["original_selection_mechanism_ba"],
            "fraction_optimizer_steps_clipped": float(train_metrics["fraction_optimizer_steps_clipped"]),
            "mean_preclip_gradient_norm": float(train_metrics["mean_preclip_gradient_norm"]),
            "mean_postclip_gradient_norm": float(train_metrics["mean_postclip_gradient_norm"]),
            "epoch_wall_time_seconds": float(train_metrics["wall_time_seconds"]),
            "resume_used": resumed, "updated_unix": time.time(),
        })
        print(
            f"warm_start seed={seed} epoch={epoch:02d} lr={lr:.1e} "
            f"old_mech={row['original_selection_mechanism_ba']:.4f} "
            f"bridge_mech={row['bridge_selection_mechanism_ba']:.4f} "
            f"eligible={'YES' if eligible else 'NO'} best={best_eligible_epoch}",
            flush=True,
        )

    selected_eligible = best_eligible_state is not None
    selected_state = best_eligible_state if selected_eligible else best_unconstrained_state
    selected_epoch = best_eligible_epoch if selected_eligible else best_unconstrained_epoch
    if selected_state is None or selected_epoch is None:
        raise RuntimeError(f"Step-10D seed={seed} failed to select any state")

    step10c.atomic_torch_save(
        resume_path,
        _resume_payload(
            identity=identity, model=model, optimizer=optimizer,
            completed_epoch=int(max(int(row["epoch"]) for row in history)), stale=stale,
            best_eligible_key=best_eligible_key, best_eligible_state=best_eligible_state,
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
        "initialization": "warm_start", "seed": seed,
        "selected_epoch": int(selected_epoch),
        "selected_checkpoint_retention_eligible": bool(selected_eligible),
        "epochs_ran": int(max(int(row["epoch"]) for row in history)),
        "stopped_early": bool(max(int(row["epoch"]) for row in history) < max_epochs),
        "resumed_from_checkpoint": bool(resumed),
        "original_selection": step7.selection_summary(original_selected),
        "bridge_selection": step7.selection_summary(bridge_selected),
        "retention_floor": dict(floor),
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    step10c.atomic_json_fsync(progress_path, {
        "schema": SCHEMA, "status": "COMPLETE", "seed": seed,
        "epochs_ran": record["epochs_ran"], "selected_epoch": int(selected_epoch),
        "selected_checkpoint_retention_eligible": bool(selected_eligible),
        "resume_used": bool(resumed), "updated_unix": time.time(),
    })
    return record, history, {k: v.detach().cpu().clone() for k, v in selected_state.items()}


def _ensemble_prediction_from_states(
    states: Mapping[int, Mapping[str, torch.Tensor]],
    *, experiment: Mapping[str, Any], config: Mapping[str, Any], bundle: Path,
    blocks: Sequence[step7.CachedBlock], device: torch.device,
) -> step7.PredictionSet:
    members: list[step7.PredictionSet] = []
    for seed in sorted(states):
        model = step10b._initialize_model(
            initialization="warm_start", seed=int(seed), experiment=experiment,
            device=device, bundle=bundle, config=config,
        )
        model.load_state_dict(dict(states[seed]), strict=True)
        model.to(device)
        members.append(step7.predict_blocks(model, blocks, device))
        del model
    return step10b._mean_prediction(members)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    mixture_config_path = args.mixture_config.expanduser().resolve()
    step7_path = args.step7_config.expanduser().resolve()
    config = baseline.read_json(config_path)
    mixture_cfg = baseline.read_json(mixture_config_path)
    experiment = baseline.read_json(step7_path)
    _assert_frozen_contract(config)
    if mixture_cfg.get("schema") != step10b.MIXTURE_SCHEMA:
        raise RuntimeError("unexpected Step-10 mixture config schema")
    if experiment.get("schema") != step7.EXPERIMENT_SCHEMA:
        raise RuntimeError("unexpected Step-7 experiment config")

    mixture_product = step10b._resolve_mixture_product(args.mixture_product_dir)
    mixture_complete, bridge_rows, _ = step10b._verify_mixture_product(mixture_product, mixture_cfg)
    original_product, original_complete, original_rows = step10b._verify_original_product(
        mixture_product, mixture_cfg, experiment
    )
    bundle, bundle_complete = step10b._resolve_step9a_bundle(args.step9a_bundle_dir, config)
    seeds = [int(v) for v in config["development_candidate"]["seeds"]]
    if seeds != [1701, 1702, 1703]:
        raise RuntimeError("unexpected Step-10D seeds")
    if seeds != [int(v) for v in bundle_complete["seeds"]]:
        raise RuntimeError("Step-10D seeds differ from Step-9A bundle")

    step10c_benchmark = args.step10c_benchmark_dir.expanduser().resolve()
    step10c_complete, step10c_states = _verify_step10c_reference(step10c_benchmark, config)

    original_fit_roots, original_selection_roots, _spent_outer, original_by_root = step7.split_root_indices(
        original_rows, experiment
    )
    bridge_by_root = step10b._by_root(bridge_rows)
    bridge_fit_roots = step10b._bridge_roots(bridge_rows, "fit")
    bridge_selection_roots = step10b._bridge_roots(bridge_rows, "selection")

    device = step7.resolve_device(args.device)
    root_batch_size = int(config["training"]["root_batch_size"])
    print("STEP 10D FINAL SIMULATOR LR REFINEMENT — SELECTION ONLY / NO OUTER / NO QPU", flush=True)
    print(f"device: {device}", flush=True)
    print("FINAL SIMULATOR INTERVENTION BEFORE IBM HARDWARE: YES", flush=True)
    print("LR epochs 1-20: 3e-4 | epochs 21-40: 1e-4", flush=True)

    original_fit = step7.materialize_blocks(
        product=original_product, rows=original_rows, by_root=original_by_root,
        roots=original_fit_roots, root_batch_size=root_batch_size,
        label="step10d-original-fit", progress_every=args.progress_every,
    )
    original_selection = step7.materialize_blocks(
        product=original_product, rows=original_rows, by_root=original_by_root,
        roots=original_selection_roots, root_batch_size=root_batch_size,
        label="step10d-original-selection", progress_every=args.progress_every,
    )
    bridge_fit = step7.materialize_blocks(
        product=mixture_product, rows=bridge_rows, by_root=bridge_by_root,
        roots=bridge_fit_roots, root_batch_size=root_batch_size,
        label="step10d-bridge-fit", progress_every=args.progress_every,
    )
    bridge_selection = step7.materialize_blocks(
        product=mixture_product, rows=bridge_rows, by_root=bridge_by_root,
        roots=bridge_selection_roots, root_batch_size=root_batch_size,
        label="step10d-bridge-selection", progress_every=args.progress_every,
    )
    effect_class_weight, mechanism_class_weight = step7.class_weights(list(original_fit) + list(bridge_fit))

    tolerance = float(config["selection"]["original_domain_retention_tolerance"])
    retention_floors: dict[int, dict[str, float]] = {}
    epoch0_original_members: list[step7.PredictionSet] = []
    epoch0_per_seed: dict[str, Any] = {}
    for seed in seeds:
        model = step10b._initialize_model(
            initialization="warm_start", seed=seed, experiment=experiment,
            device=device, bundle=bundle, config=config,
        )
        pred = step7.predict_blocks(model, original_selection, device)
        summary = step7.selection_summary(pred)
        epoch0_original_members.append(pred)
        epoch0_per_seed[str(seed)] = summary
        retention_floors[seed] = {
            "mechanism_balanced_accuracy": float(summary["mechanism_balanced_accuracy"]) - tolerance,
            "effect_balanced_accuracy": float(summary["effect_balanced_accuracy"]) - tolerance,
        }
        del model
    epoch0_original_ensemble = step10b._mean_prediction(epoch0_original_members)
    epoch0_ensemble_summary = step7.selection_summary(epoch0_original_ensemble)
    ensemble_floor = {
        "mechanism_balanced_accuracy": float(epoch0_ensemble_summary["mechanism_balanced_accuracy"]) - tolerance,
        "effect_balanced_accuracy": float(epoch0_ensemble_summary["effect_balanced_accuracy"]) - tolerance,
    }

    key_identity = {
        "schema": SCHEMA,
        "training_config_sha256": baseline.sha256_file(config_path),
        "mixture_config_sha256": baseline.sha256_file(mixture_config_path),
        "step7_config_sha256": baseline.sha256_file(step7_path),
        "runner_sha256": baseline.sha256_file(Path(__file__).resolve()),
        "mixture_product_id": mixture_complete["product_id"],
        "original_product_id": original_complete["product_id"],
        "step9a_bundle_id": bundle_complete["bundle_id"],
        "reference_step10c_benchmark_id": step10c_complete["benchmark_id"],
    }
    benchmark_id = "benchmark_" + hashlib.sha256(
        baseline.canonical_json(key_identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / benchmark_id
    if output.exists():
        raise RuntimeError(f"refusing to overwrite completed Step-10D benchmark {output}")
    work_root = output_parent / f".{benchmark_id}.work"
    work_root.mkdir(parents=True, exist_ok=True)

    selected_records: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    selected_states: dict[int, dict[str, torch.Tensor]] = {}
    any_resumed = False
    runtime = step10c._runtime_fingerprint(device)
    for seed in seeds:
        run_identity = {
            **key_identity,
            "seed": seed,
            "initialization": "warm_start",
            "architecture": "late_concat",
            "trainable_parameter_count": 453829,
            "execution_device": str(device),
            "runtime_fingerprint": runtime,
        }
        print(f"\nTraining Step-10D warm_start seed={seed}", flush=True)
        record, history, state = _train_seed(
            seed=seed, experiment=experiment, config=config, bundle=bundle,
            original_fit=original_fit, bridge_fit=bridge_fit,
            original_selection=original_selection, bridge_selection=bridge_selection,
            floor=retention_floors[seed], effect_class_weight=effect_class_weight,
            mechanism_class_weight=mechanism_class_weight, device=device,
            run_dir=work_root / f"warm_start__seed{seed}",
            identity=run_identity, allow_resume=not args.no_resume,
        )
        selected_records[f"warm_start__seed{seed}"] = record
        histories.extend(history)
        selected_states[seed] = state
        any_resumed = any_resumed or bool(record["resumed_from_checkpoint"])

    step10d_original = _ensemble_prediction_from_states(
        selected_states, experiment=experiment, config=config, bundle=bundle,
        blocks=original_selection, device=device,
    )
    step10d_bridge = _ensemble_prediction_from_states(
        selected_states, experiment=experiment, config=config, bundle=bundle,
        blocks=bridge_selection, device=device,
    )
    step10c_original = _ensemble_prediction_from_states(
        step10c_states, experiment=experiment, config=config, bundle=bundle,
        blocks=original_selection, device=device,
    )
    step10c_bridge = _ensemble_prediction_from_states(
        step10c_states, experiment=experiment, config=config, bundle=bundle,
        blocks=bridge_selection, device=device,
    )
    d_old = step7.selection_summary(step10d_original)
    d_bridge = step7.selection_summary(step10d_bridge)
    c_old = step7.selection_summary(step10c_original)
    c_bridge = step7.selection_summary(step10c_bridge)

    all_seed_eligible = all(
        bool(selected_records[f"warm_start__seed{seed}"]["selected_checkpoint_retention_eligible"])
        for seed in seeds
    )
    ensemble_retention = bool(
        float(d_old["mechanism_balanced_accuracy"]) >= ensemble_floor["mechanism_balanced_accuracy"]
        and float(d_old["effect_balanced_accuracy"]) >= ensemble_floor["effect_balanced_accuracy"]
    )
    delta = float(d_bridge["mechanism_balanced_accuracy"]) - float(c_bridge["mechanism_balanced_accuracy"])
    min_improvement = float(config["hardware_candidate_decision"]["minimum_material_improvement"])
    step10d_primary = bool(all_seed_eligible and ensemble_retention and delta > min_improvement)
    primary = "step10d_warm_start" if step10d_primary else "step10c_warm_start"

    staging = output_parent / f".{benchmark_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        checkpoint_hashes: dict[str, str] = {}
        for seed in seeds:
            name = f"warm_start__seed{seed}.pt"
            path = staging / name
            record = selected_records[f"warm_start__seed{seed}"]
            step10c.atomic_torch_save(path, {
                "schema": SCHEMA,
                "initialization": "warm_start",
                "seed": seed,
                "architecture": "late_concat",
                "selected_epoch": int(record["selected_epoch"]),
                "selected_checkpoint_retention_eligible": bool(record["selected_checkpoint_retention_eligible"]),
                "source_step9a_bundle_id": bundle_complete["bundle_id"],
                "optimizer_state_present": False,
                "state_dict": selected_states[seed],
            })
            checkpoint_hashes[name] = baseline.sha256_file(path)

        baseline.write_csv(staging / "training_history.csv", histories)
        baseline.atomic_json(staging / "model_selection.json", {
            "per_seed": selected_records,
            "step9a_epoch0_original_selection": epoch0_per_seed,
            "retention_floors": {str(seed): retention_floors[seed] for seed in seeds},
            "step9a_epoch0_original_ensemble": epoch0_ensemble_summary,
            "ensemble_retention_floor": ensemble_floor,
        })
        decision = {
            "schema": SCHEMA,
            "primary_hardware_candidate": primary,
            "step10d_becomes_primary": step10d_primary,
            "minimum_material_improvement": min_improvement,
            "bridge_mechanism_ba_delta_step10d_minus_step10c": delta,
            "all_step10d_seed_checkpoints_retention_eligible": all_seed_eligible,
            "step10d_ensemble_retention_pass": ensemble_retention,
            "step10c_warm_selection": {"original": c_old, "bridge": c_bridge},
            "step10d_warm_selection": {"original": d_old, "bridge": d_bridge},
            "step10c_outer_accessed": False,
            "new_outer_accessed": False,
            "qpu_executed": False,
            "final_simulator_intervention_complete": True,
            "next_stage": "FREEZE_AND_EXECUTE_EXPLORATORY_IBM_HARDWARE_TRANSFER_PILOT",
            "further_simulator_tuning_before_hardware_permitted": False,
        }
        baseline.atomic_json(staging / "hardware_candidate_decision.json", decision)
        files = ["training_history.csv", "model_selection.json", "hardware_candidate_decision.json"] + sorted(checkpoint_hashes)
        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "benchmark_id": benchmark_id,
            "identity": {**key_identity, "execution_device": str(device), "runtime_fingerprint": runtime},
            "primary_hardware_candidate": primary,
            "final_simulator_intervention_complete": True,
            "hardware_stage_next_regardless_of_outcome": True,
            "step10c_outer_accessed": False,
            "new_outer_accessed": False,
            "qpu_executed": False,
            "architecture": "late_concat",
            "trainable_parameter_count": 453829,
            "any_run_resumed": any_resumed,
            "file_hashes": {name: baseline.sha256_file(staging / name) for name in files},
        }
        baseline.atomic_json(staging / "benchmark_complete.json", completion)
        os.replace(staging, output)
        step10c._fsync_dir(output_parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP 10D FINAL SIMULATOR LR REFINEMENT COMPLETE\n")
    print(f"Step-10C warm bridge selection mechanism BA: {float(c_bridge['mechanism_balanced_accuracy']):.6f}")
    print(f"Step-10D warm bridge selection mechanism BA: {float(d_bridge['mechanism_balanced_accuracy']):.6f}")
    print(f"Delta Step10D-Step10C: {delta:+.6f}")
    print(f"PRIMARY HARDWARE CANDIDATE: {primary}")
    print("Step-10C outer accessed: NO")
    print("New outer accessed: NO")
    print("QPU executed: NO")
    print("FURTHER SIMULATOR TUNING BEFORE HARDWARE: FORBIDDEN BY FROZEN PROTOCOL")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
