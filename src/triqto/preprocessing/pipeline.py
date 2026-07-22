"""End-to-end immutable offline preprocessing pipeline for completed Phase 7 data."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

from .artifacts import (
    atomic_write_json, atomic_write_text, create_staging_directory,
    discard_staging_directory, output_inventory, publish_staging_directory,
    write_jsonl, write_parquet,
)
from .balancing import build_training_weights
from .config import PreprocessingConfig, preprocessing_config_to_dict
from .constants import (
    CANONICALIZATION_VERSION, LABEL_AUDIT_VERSION, PREPROCESSING_SCHEMA_VERSION,
    SEVERITY_POLICY_VERSION, SPLIT_POLICY_VERSION,
)
from .grouping import build_duplicate_relations, build_hard_negative_pairs, build_leakage_relations
from .hashing import sha256_payload
from .io import inventory_digest, inventory_files, load_phase7_source, verify_inventory_unchanged
from .outliers import detect_outliers
from .records import ProcessedSample
from .reporting import (
    build_distribution_report, build_duplicate_report, build_health_report,
    build_label_audit_report, build_leakage_report, build_outlier_report, render_html_report,
)
from .sample_context import _quarantined_sample
from .sample_processor import _process_sample
from .scaling import fit_semantic_scalers
from .splits import build_challenge_splits
from .validation import ValidationCollector
from .views import _flat_training_row, _task_view_rows

ProgressCallback = Callable[[dict[str, Any]], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_environment() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def _dependency_versions() -> dict[str, str | None]:
    names = (
        "triqto",
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "networkx",
        "pyyaml",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def preprocess_phase7_dataset(
    *,
    phase7_root: str | Path,
    output_root: str | Path,
    config: PreprocessingConfig,
    run_id: str | None = None,
    dry_run: bool = False,
    validation_only: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config.validate()
    source_path = Path(phase7_root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    if source_path == output_path or source_path in output_path.parents:
        raise ValueError("preprocessing output must not be the raw Phase 7 root or its child")
    if output_path in source_path.parents:
        raise ValueError("raw Phase 7 root must not be nested inside preprocessing output")

    started_wall = _utc_now()
    started_monotonic = time.monotonic()
    source = load_phase7_source(source_path)
    raw_inventory_digest = inventory_digest(source.inventory)
    config_payload = preprocessing_config_to_dict(config)
    run_identity = sha256_payload(
        "preprocessing_run",
        {
            "raw_inventory_digest": raw_inventory_digest,
            "scientific_generation_id": source.completion_marker.get(
                "scientific_generation_id"
            ),
            "config": config_payload,
        },
        config=config,
    )
    derived_run_id = f"preprocess_{run_identity[:24]}"
    if run_id is None:
        run_id = derived_run_id
    elif (
        not isinstance(run_id, str)
        or not run_id.strip()
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id)
    ):
        raise ValueError("explicit run_id must contain only letters, numbers, '_' or '-'")
    plan = {
        "run_id": run_id,
        "phase7_root": source_path.as_posix(),
        "output_root": output_path.as_posix(),
        "raw_file_count": len(source.inventory),
        "sample_count": len(source.samples),
        "raw_inventory_digest": raw_inventory_digest,
        "validation_only": validation_only,
        "dry_run": dry_run,
        "challenge_splits": list(config.splits.challenge_splits),
    }
    if dry_run:
        return {"status": "dry_run", "plan": plan}

    staging, final = create_staging_directory(output_path, run_id)
    processed: list[ProcessedSample] = []
    try:
        atomic_write_json(staging / "manifests" / "run_plan.json", plan)
        atomic_write_json(
            staging / "manifests" / "preprocessing_config.json", config_payload
        )
        write_parquet(
            staging / "indices" / "raw_file_inventory.parquet",
            [record.to_dict() for record in source.inventory],
        )
        for index, sample in enumerate(source.samples, start=1):
            try:
                item = _process_sample(source, sample, config)
            except Exception as exc:  # preserve evidence rather than aborting all records
                collector = ValidationCollector()
                collector.add(
                    "pipeline.unexpected_record_error",
                    "error",
                    f"sample_id={sample.sample_id}",
                    f"{type(exc).__name__}: {exc}",
                    "record processing must complete",
                    "quarantine",
                )
                item = _quarantined_sample(
                    sample,
                    source_locator=(
                        f"manifests/sample_manifest.parquet#sample_id={sample.sample_id}"
                    ),
                    collector=collector,
                    config=config,
                    reason="unexpected_internal_error",
                )
            processed.append(item)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "sample_preprocessing",
                        "completed": index,
                        "total": len(source.samples),
                        "accepted": sum(entry.accepted for entry in processed),
                        "quarantined": sum(not entry.accepted for entry in processed),
                    }
                )

        accepted = [sample for sample in processed if sample.accepted]
        quarantined = [sample for sample in processed if not sample.accepted]
        write_parquet(
            staging / "validated" / "accepted_samples.parquet",
            [sample.summary_dict() for sample in accepted],
        )
        write_jsonl(
            staging / "validated" / "accepted_samples.jsonl",
            [sample.summary_dict() for sample in accepted],
        )
        write_parquet(
            staging / "quarantine" / "quarantined_samples.parquet",
            [sample.summary_dict() for sample in quarantined],
        )
        write_jsonl(
            staging / "quarantine" / "quarantined_samples.jsonl",
            [sample.summary_dict() for sample in quarantined],
        )
        write_jsonl(
            staging / "canonical" / "canonical_records.jsonl",
            [
                {
                    "sample_id": sample.sample_id,
                    "canonical_payload": sample.canonical_payload,
                    "hashes": sample.hashes.to_dict(),
                    "audit_flags": sample.audit_flags,
                }
                for sample in accepted
            ],
        )
        write_parquet(
            staging / "indices" / "hash_index.parquet",
            [
                {"sample_id": sample.sample_id, **sample.hashes.to_dict()}
                for sample in accepted
            ],
        )

        duplicates = build_duplicate_relations(processed)
        leakage_relations = build_leakage_relations(processed)
        hard_negatives = build_hard_negative_pairs(
            processed,
            maximum_per_category=config.grouping.hard_negative_max_pairs_per_category,
        )
        write_parquet(
            staging / "groups" / "duplicate_groups.parquet",
            [record.to_dict() for record in duplicates],
        )
        write_parquet(
            staging / "groups" / "leakage_relations.parquet",
            [record.to_dict() for record in leakage_relations],
        )
        write_parquet(
            staging / "groups" / "hard_negative_pairs.parquet",
            [record.to_dict() for record in hard_negatives],
        )
        write_parquet(
            staging / "groups" / "counterfactual_sets.parquet",
            [
                {
                    "counterfactual_set_id": sample.hashes.counterfactual_set_hash,
                    "sample_id": sample.sample_id,
                    "clean_circuit_id": sample.clean_circuit_id,
                    "distorted_circuit_id": sample.distorted_circuit_id,
                    "candidate_ids": [],
                    "candidate_status": "not_available_in_phase7_input",
                    "pre_correction_effect_score": sample.combined_effect_score,
                    "candidate_ranking_target": None,
                }
                for sample in accepted
            ],
        )

        health_report = build_health_report(processed)
        duplicate_report = build_duplicate_report(duplicates)
        label_report = build_label_audit_report(processed)
        if validation_only:
            split_result = None
            outlier_records = detect_outliers(processed, config.outliers)
            distribution_report: dict[str, Any] = {}
            leakage_report: dict[str, Any] = {
                "status": "not_run_in_validation_only_mode"
            }
            weights: list[dict[str, Any]] = []
            scalers: dict[str, Any] = {}
        else:
            split_result = build_challenge_splits(
                processed, leakage_relations, config.splits
            )
            write_parquet(
                staging / "splits" / "assignments.parquet",
                [record.to_dict() for record in split_result.assignments],
            )
            atomic_write_json(
                staging / "splits" / "split_status.json",
                [record.to_dict() for record in split_result.statuses],
            )
            atomic_write_json(
                staging / "splits" / "leakage_violations.json",
                split_result.leakage_violations,
            )
            distribution_report = build_distribution_report(
                processed, split_result.assignments
            )
            leakage_report = build_leakage_report(
                split_result.statuses,
                split_result.leakage_violations,
            )
            baseline_status = next(
                (
                    item
                    for item in split_result.statuses
                    if item.split_name == "grouped_baseline" and item.status == "valid"
                ),
                None,
            )
            baseline_assignments = (
                [
                    item
                    for item in split_result.assignments
                    if item.split_name == "grouped_baseline"
                ]
                if baseline_status is not None
                else []
            )
            train_ids = {
                item.sample_id
                for item in baseline_assignments
                if item.partition == "train"
            }
            outlier_records = detect_outliers(processed, config.outliers)
            train_fit_outliers = detect_outliers(
                processed,
                config.outliers,
                fit_sample_ids=train_ids if train_ids else None,
            )
            write_parquet(
                staging / "reports" / "outliers_dataset_audit.parquet",
                [record.to_dict() for record in outlier_records],
            )
            write_parquet(
                staging / "reports" / "outliers_train_fit.parquet",
                [record.to_dict() for record in train_fit_outliers],
            )
            weights = (
                build_training_weights(
                    processed,
                    split_result.assignments,
                    split_name="grouped_baseline",
                    config=config.balancing,
                )
                if baseline_status is not None
                else []
            )
            write_parquet(
                staging / "training_views" / "grouped_baseline_weights.parquet",
                weights,
            )
            flat_by_id = {
                sample.sample_id: _flat_training_row(sample) for sample in accepted
            }
            training_rows = [flat_by_id[sample_id] for sample_id in sorted(train_ids)]
            feature_specs: dict[str, tuple[str, str]] = {
                "n_qubits": ("depth_counts", "log1p_robust"),
                "depth": ("depth_counts", "log1p_robust"),
                "one_qubit_event_count": ("depth_counts", "log1p_robust"),
                "two_qubit_event_count": ("depth_counts", "log1p_robust"),
                "combined_effect_score": ("distances", "identity"),
                "effect_hellinger": ("distances", "identity"),
                "effect_jensen_shannon_distance": ("distances", "identity"),
                "effect_fubini_study": ("distances", "identity"),
            }
            for row in training_rows:
                for name in row:
                    if name.startswith("angle_"):
                        feature_specs[name] = ("angles", "sincos")
            scalers = (
                fit_semantic_scalers(training_rows, feature_specs=feature_specs)
                if training_rows
                else {}
            )
            atomic_write_json(
                staging / "scalers" / "grouped_baseline_scalers.json", scalers
            )
            weight_by_id = {
                row["sample_id"]: float(row["clipped_weight"]) for row in weights
            }
            valid_split_names = [
                status.split_name
                for status in split_result.statuses
                if status.status == "valid"
            ]
            for split_name in valid_split_names:
                views = _task_view_rows(
                    processed,
                    split_result.assignments,
                    split_name,
                    weight_by_id if split_name == "grouped_baseline" else {},
                    scalers,
                )
                for view_name, rows in views.items():
                    root = staging / "training_views" / split_name
                    write_parquet(root / f"{view_name}.parquet", rows)
                    atomic_write_json(
                        root / f"{view_name}.manifest.json",
                        {
                            "view_name": view_name,
                            "source_split": split_name,
                            "row_count": len(rows),
                            "partitions": dict(
                                sorted(Counter(row["partition"] for row in rows).items())
                            ),
                            "parent_id_field": "parent_sample_id",
                            "mask_policy": "explicit_missingness_and_hilbert_masks",
                            "scaler_policy": "fit_on_grouped_baseline_train_only",
                        },
                    )

        outlier_report = build_outlier_report(outlier_records)
        reports = {
            "dataset_health": health_report,
            "duplicates": duplicate_report,
            "label_audit": label_report,
            "leakage": leakage_report,
            "distribution_shift": distribution_report,
            "outliers": outlier_report,
        }
        for name, payload in reports.items():
            atomic_write_json(staging / "reports" / f"{name}.json", payload)
        if config.reports.html:
            atomic_write_text(
                staging / "reports" / "preprocessing_report.html",
                render_html_report("TriQTO Dataset Preprocessing Report", reports),
            )

        verify_inventory_unchanged(source.inventory, inventory_files(source.root))
        preliminary_outputs = output_inventory(staging)
        ended_wall = _utc_now()
        manifest = {
            "complete": False,
            "run_id": run_id,
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "preprocessing_version": config.preprocessing_version,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "label_audit_version": LABEL_AUDIT_VERSION,
            "severity_policy_version": SEVERITY_POLICY_VERSION,
            "start_timestamp": started_wall,
            "end_timestamp": ended_wall,
            "runtime_seconds": time.monotonic() - started_monotonic,
            "raw_dataset": {
                "root": source.root.as_posix(),
                "scientific_generation_id": source.completion_marker.get(
                    "scientific_generation_id"
                ),
                "config_id": source.completion_marker.get("config_id"),
                "raw_inventory_digest": raw_inventory_digest,
                "raw_file_count": len(source.inventory),
            },
            "config": config_payload,
            "record_counts": {
                "input": len(processed),
                "accepted": sum(sample.accepted for sample in processed),
                "quarantined": sum(not sample.accepted for sample in processed),
                "duplicate_groups": len(duplicates),
                "leakage_relations": len(leakage_relations),
                "hard_negative_pairs": len(hard_negatives),
                "split_assignments": 0
                if split_result is None
                else len(split_result.assignments),
            },
            "repairs": {
                "count": sum(
                    finding.repair_applied
                    for sample in processed
                    for finding in sample.findings
                ),
                "policy": "tiny_numeric_deviation_only; original raw data immutable",
            },
            "scientific_boundaries": {
                "preprocessing_proves_generalization": False,
                "preprocessing_removes_nonidentifiability": False,
                "preprocessing_discovers_causal_error_source": False,
                "layout_renaming_is_not_structural_layout_generalization": True,
                "phase7_candidate_actions_available": False,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": _dependency_versions(),
                **_git_environment(),
            },
            "outputs_before_manifest": preliminary_outputs,
            "validation_only": validation_only,
        }
        atomic_write_json(staging / "manifests" / "run_manifest.json", manifest)
        final_outputs = output_inventory(staging)
        completion = {
            "complete": True,
            "run_id": run_id,
            "schema_version": PREPROCESSING_SCHEMA_VERSION,
            "raw_inventory_digest": raw_inventory_digest,
            "accepted_count": sum(sample.accepted for sample in processed),
            "quarantine_count": sum(not sample.accepted for sample in processed),
            "managed_files": [row["relative_path"] for row in final_outputs],
            "managed_file_hashes": {
                row["relative_path"]: row["sha256"] for row in final_outputs
            },
        }
        atomic_write_json(staging / "preprocessing_complete.json", completion)
        publish_staging_directory(staging, final)
        return {
            "status": "complete",
            "run_id": run_id,
            "output_root": final.as_posix(),
            "accepted_count": completion["accepted_count"],
            "quarantine_count": completion["quarantine_count"],
            "validation_only": validation_only,
        }
    except Exception:
        discard_staging_directory(staging)
        raise

