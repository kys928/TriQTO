"""Offline Phase 15.5 orchestration and immutable publication."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import os, shutil, uuid
import numpy as np
from triqto.core.ids import canonical_json, make_deterministic_id
from triqto.graph import load_completed_phase7_dataset, snapshot_managed_files
from triqto.training import load_completed_training_view_dataset, verify_training_view_snapshot
from triqto.training.latent_extraction import restore_checkpoint_for_latents
from .benchmark import _benchmark, _policy_dataset_arrays
from .config import Phase155Config, phase155_config_to_dict
from .constants import CANDIDATE_FEATURE_NAMES, CONTEXT_SUMMARY_NAMES, PHASE155_SCHEMA
from .dataset_builder import _build_rows
from .io_utils import _latent_table, _managed_inventory, _paths_overlap, _read_json, _selected_examples, _sha256, _validate_training_binding, _write_json, _write_jsonl
from .policy import FAMILY_NAMES, load_policy_checkpoint, save_policy_checkpoint, score_dataset, train_operational_policy

def run_phase15_5(
    *,
    phase7_root: str | Path,
    training_view_root: str | Path,
    training_root: str | Path,
    checkpoint: str | Path,
    output_root: str | Path,
    config: Phase155Config,
) -> dict[str, Any]:
    """Execute the immutable offline Phase 15.5 workflow."""
    if not isinstance(config, Phase155Config):
        raise TypeError("config must be Phase155Config")
    phase7_path = Path(phase7_root)
    view_path = Path(training_view_root)
    training_path = Path(training_root)
    checkpoint_path = Path(checkpoint)
    output = Path(output_root)
    for source in (phase7_path, view_path, training_path, checkpoint_path):
        if _paths_overlap(output, source):
            raise ValueError("Phase 15.5 output must not overlap immutable sources")
    if output.exists():
        raise FileExistsError(f"Phase 15.5 output already exists: {output}")
    phase7 = load_completed_phase7_dataset(phase7_path)
    training_view = load_completed_training_view_dataset(view_path)
    verify_training_view_snapshot(training_view)
    model, checkpoint_meta, spec = restore_checkpoint_for_latents(checkpoint_path)
    if spec.training_view_dataset_id != training_view.training_view_dataset_id:
        raise ValueError("Phase 15.5 checkpoint data spec/Phase 12 mismatch")
    training_complete = _validate_training_binding(training_path, checkpoint_meta, training_view.training_view_dataset_id, model)
    examples_by_split = {
        split: _selected_examples(training_view, spec, phase7_path, split, config.max_samples_per_split)
        for split in ("train", "validation", "test")
    }
    all_examples = [value for split in ("train", "validation", "test") for value in examples_by_split[split]]
    latents = _latent_table(model, all_examples)
    evidence_rows, metadata_rows, dataset, context_names, candidate_names = _build_rows(
        phase7=phase7,
        examples_by_split=examples_by_split,
        latents=latents,
        config=config,
    )
    source_identity = {
        "phase7_scientific_generation_id": phase7.source_scientific_generation_id,
        "phase7_snapshot_hash": phase7.source_snapshot.aggregate_sha256,
        "training_view_dataset_id": training_view.training_view_dataset_id,
        "phase12_snapshot_hash": training_view.snapshot.aggregate_sha256,
        "training_run_id": training_complete["training_run_id"],
        "checkpoint_id": checkpoint_meta["checkpoint_id"],
        "checkpoint_content_hash": checkpoint_meta["content_hash"],
        "model_architecture_id": model.architecture_id,
    }
    config_payload = phase155_config_to_dict(config)
    run_id = make_deterministic_id("phase155_run", {"schema": PHASE155_SCHEMA, "source_identity": source_identity, "config": config_payload})
    training_result = train_operational_policy(
        dataset,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        utility_mse_weight=config.utility_mse_weight,
        seed=config.seed,
    )
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True)
        _write_json(staging / "phase15_5_config.json", config_payload)
        _write_jsonl(staging / "noisy_evidence.jsonl", evidence_rows)
        _write_jsonl(staging / "operational_supervision.jsonl", metadata_rows)
        np.savez_compressed(staging / "policy_dataset.npz", **_policy_dataset_arrays(dataset))
        _write_json(staging / "policy_dataset_schema.json", {"context_feature_names": list(context_names), "candidate_feature_names": list(candidate_names), "family_names": list(FAMILY_NAMES), "simulation_only_supervision": True, "clean_pair_excluded_from_model_inputs": True})
        _write_json(staging / "training_history.json", {"history": training_result["history"], "best_epoch": training_result["best_epoch"], "best_validation_loss": training_result["best_validation_loss"], "train_split_used_for_optimization": True, "validation_split_used_for_model_selection": True, "test_split_used_for_optimization": False})
        checkpoint_metadata = save_policy_checkpoint(
            staging / "operational_policy_checkpoint.npz",
            training_result=training_result,
            source_identity=source_identity,
            config_identity={"phase15_5_run_id": run_id, "config": config_payload},
        )
        loaded = load_policy_checkpoint(staging / "operational_policy_checkpoint.npz")
        scores = score_dataset(
            loaded["model"],
            dataset,
            context_mean=loaded["context_mean"],
            context_std=loaded["context_std"],
            candidate_mean=loaded["candidate_mean"],
            candidate_std=loaded["candidate_std"],
        )
        report = _benchmark(dataset, scores, metadata_rows, config=config)
        report.update(
            {
                "schema": PHASE155_SCHEMA,
                "phase15_5_run_id": run_id,
                "policy_checkpoint_id": checkpoint_metadata["policy_checkpoint_id"],
                "source_identity": source_identity,
                "sample_counts": {split: len(examples_by_split[split]) for split in ("train", "validation", "test")},
                "candidate_count": len(dataset.candidate_ids),
                "available_candidate_count": int(dataset.available_mask.sum()),
                "noise_profile_count": len(config.noise_profiles),
                "action_family_count": len(FAMILY_NAMES),
                "operational_policy_trained": True,
                "matched_operational_supervision": True,
                "supervision_uses_privileged_clean_pairs": True,
                "model_inputs_exclude_clean_pair_targets": True,
                "fake_backend_fixture_only": True,
                "density_matrix_evidence_present": config.include_density_matrix,
                "physical_hardware": False,
                "topology_loss_weight": 0.0,
            }
        )
        _write_json(staging / "benchmark_report.json", report)
        managed = [
            "benchmark_report.json",
            "noisy_evidence.jsonl",
            "operational_policy_checkpoint.json",
            "operational_policy_checkpoint.npz",
            "operational_supervision.jsonl",
            "phase15_5_config.json",
            "policy_dataset.npz",
            "policy_dataset_schema.json",
            "training_history.json",
        ]
        inventory = _managed_inventory(staging, managed)
        completion = {
            "schema": PHASE155_SCHEMA,
            "complete": True,
            "phase15_5_run_id": run_id,
            "policy_checkpoint_id": checkpoint_metadata["policy_checkpoint_id"],
            "source_identity": source_identity,
            "managed_files": managed + ["phase15_5_complete.json"],
            "managed_inventory": inventory,
            "physical_hardware": False,
            "topology_loss_weight": 0.0,
            "test_split_used_for_optimization": False,
            "research_quality_claim": False,
        }
        completion["completion_content_hash"] = make_deterministic_id("phase155_completion", {"payload": canonical_json(completion)})
        _write_json(staging / "phase15_5_complete.json", completion)
        if output.exists():
            raise FileExistsError(f"Phase 15.5 output appeared during publication: {output}")
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    verify_training_view_snapshot(training_view)
    actual_phase7 = snapshot_managed_files(phase7_path, tuple(entry.reference for entry in phase7.source_snapshot.entries))
    if actual_phase7 != phase7.source_snapshot:
        raise RuntimeError("Phase 7 source changed during Phase 15.5")
    return {"summary": report, "completion": completion, "output_root": output}

def load_phase15_5_result(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    completion = _read_json(base / "phase15_5_complete.json")
    claimed_hash = completion.pop("completion_content_hash", None)
    if claimed_hash != make_deterministic_id("phase155_completion", {"payload": canonical_json(completion)}):
        raise ValueError("Phase 15.5 completion content hash mismatch")
    completion["completion_content_hash"] = claimed_hash
    managed = completion.get("managed_files")
    if not isinstance(managed, list) or sorted(managed) != sorted({path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}):
        raise ValueError("Phase 15.5 managed inventory mismatch")
    inventory = completion.get("managed_inventory")
    if not isinstance(inventory, list):
        raise TypeError("Phase 15.5 managed_inventory must be a list")
    for entry in inventory:
        if _sha256(base / entry["reference"]) != entry["sha256"] or (base / entry["reference"]).stat().st_size != entry["size_bytes"]:
            raise ValueError("Phase 15.5 managed file hash/size mismatch")
    report = _read_json(base / "benchmark_report.json")
    if report.get("phase15_5_run_id") != completion.get("phase15_5_run_id"):
        raise ValueError("Phase 15.5 report/completion identity mismatch")
    if report.get("physical_hardware") is not False or report.get("topology_loss_weight") != 0.0:
        raise ValueError("Phase 15.5 claim boundary mismatch")
    checkpoint = load_policy_checkpoint(base / "operational_policy_checkpoint.npz")
    if checkpoint["metadata"].get("policy_checkpoint_id") != completion.get("policy_checkpoint_id"):
        raise ValueError("Phase 15.5 checkpoint/completion identity mismatch")
    return {"completion": completion, "report": report, "checkpoint": checkpoint}

__all__ = ["CANDIDATE_FEATURE_NAMES", "CONTEXT_SUMMARY_NAMES", "PHASE155_SCHEMA", "load_phase15_5_result", "run_phase15_5"]
