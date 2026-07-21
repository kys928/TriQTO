"""Dedicated fast Phase 12 runner for a completed Phase 7/8/9/11 workspace."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import threading
import time
from typing import Any

from triqto.training_views import load_training_view_config

from .action_candidate_capacity import resolve_action_candidate_capacity
from .campaign import (
    PHASE156_DATA_COMPLETE_SCHEMA,
    _atomic_write_json,
    _load_prepared_plan,
    _read_json,
    _update_state,
    _verify_source_configs,
    _workspace_lock,
)
from .fast_phase12 import build_training_view_result_fast
from .resumable import normalize_checkpoint_retention, normalize_resume_mode
from .resumable_phase12 import write_training_view_dataset_resumable


def _require_complete(root: Path, marker_name: str, phase_name: str) -> dict[str, Any]:
    marker = root / marker_name
    if not marker.is_file():
        raise FileNotFoundError(
            f"{phase_name} must be complete before running Phase 12 separately: {marker}"
        )
    payload = _read_json(marker)
    if payload.get("complete") is not True:
        raise ValueError(f"{phase_name} completion marker is not complete")
    return payload


def _effective_training_config(
    *,
    plan: dict[str, Any],
    phase9: Path,
    resolution_path: Path,
):
    requested = load_training_view_config(
        plan["source_configs"]["training_view"]["absolute_path"]
    )
    phase9_marker = _read_json(phase9 / "action_complete.json")
    resolution: dict[str, Any] | None = None
    if resolution_path.is_file():
        candidate = _read_json(resolution_path)
        reusable = (
            candidate.get("phase9_action_engine_id") == phase9_marker["action_engine_id"]
            and candidate.get("requested_max_candidates_per_item")
            == requested.max_candidates_per_item
            and candidate.get("requested_max_source_refs_per_item")
            == requested.max_source_refs_per_item
            and isinstance(candidate.get("effective_max_candidates_per_item"), int)
            and isinstance(candidate.get("effective_max_source_refs_per_item"), int)
        )
        if reusable:
            resolution = candidate
            print(
                "[Phase 12][action-capacity] reusing validated capacity resolution | "
                f"max_candidates={resolution['effective_max_candidates_per_item']:,} | "
                f"max_source_refs={resolution['effective_max_source_refs_per_item']:,}",
                flush=True,
            )

    if resolution is None:
        effective_candidates, effective_refs, resolution = (
            resolve_action_candidate_capacity(
                phase9,
                requested.max_candidates_per_item,
                requested.max_source_refs_per_item,
            )
        )
        resolution = {
            **resolution,
            "campaign_id": plan["campaign_id"],
            "phase9_action_engine_id": phase9_marker["action_engine_id"],
        }
        _atomic_write_json(resolution_path, resolution)
    else:
        effective_candidates = int(resolution["effective_max_candidates_per_item"])
        effective_refs = int(resolution["effective_max_source_refs_per_item"])

    return replace(
        requested,
        max_candidates_per_item=effective_candidates,
        max_source_refs_per_item=effective_refs,
    ), resolution


class _Phase12Progress:
    def __init__(self, path: Path, workers: int) -> None:
        self.path = path
        self.workers = workers
        self.lock = threading.Lock()

    def __call__(self, payload: dict[str, Any]) -> None:
        with self.lock:
            event = payload.get("event")
            if event == "plan":
                record = {
                    "schema": "triqto.phase15_6.phase12_progress.v1",
                    "stage": "plan",
                    "task_count": int(payload["task_count"]),
                    "shard_count": int(payload["shard_count"]),
                    "workers": self.workers,
                }
                _atomic_write_json(self.path, record)
                print(
                    "[Phase 12] plan | "
                    f"tasks={record['task_count']} | hash_shards={record['shard_count']} | "
                    f"workers={self.workers}",
                    flush=True,
                )
                return
            if event != "logical_shard":
                return

            eta = payload.get("eta_seconds")
            record = {
                "schema": "triqto.phase15_6.phase12_progress.v1",
                "stage": "logical_shards",
                "task": str(payload["task"]),
                "shard_index": int(payload["shard_index"]),
                "completed_shards": int(payload["completed_shards"]),
                "total_shards": int(payload["total_shards"]),
                "completed_entities": int(payload.get("completed_entities", 0)),
                "total_entities": int(payload.get("total_entities", 0)),
                "status": str(payload["status"]),
                "resumed_shards": int(payload.get("resumed_shards", 0)),
                "workers": int(payload.get("workers", self.workers)),
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
                "eta_seconds": None if eta is None else float(eta),
            }
            _atomic_write_json(self.path, record)
            completed = record["completed_shards"]
            total = record["total_shards"]
            if completed == total or completed % 5 == 0:
                eta_text = (
                    "unknown"
                    if record["eta_seconds"] is None
                    else f"{record['eta_seconds'] / 60.0:.1f}m"
                )
                print(
                    "[Phase 12] "
                    f"task={record['task']} | shards={completed}/{total} | "
                    f"entities={record['completed_entities']:,}/"
                    f"{record['total_entities']:,} | status={record['status']} | "
                    f"resumed={record['resumed_shards']} | workers={record['workers']} | "
                    f"ETA≈{eta_text}",
                    flush=True,
                )


def run_phase12_only(
    *,
    workspace: str | Path,
    phase12_workers: int = 4,
    phase12_shards: int = 256,
    resume_mode: str = "strict",
    checkpoint_retention: str = "campaign",
) -> dict[str, Any]:
    """Finish and publish Phase 12 without rerunning the wider data-stage wrapper."""
    resolved_resume_mode = normalize_resume_mode(resume_mode)
    resolved_retention = normalize_checkpoint_retention(checkpoint_retention)
    if isinstance(phase12_workers, bool) or not isinstance(phase12_workers, int):
        raise TypeError("phase12_workers must be an integer and not bool")
    if phase12_workers < 1 or phase12_workers > 32:
        raise ValueError("phase12_workers must be in [1, 32]")
    if isinstance(phase12_shards, bool) or not isinstance(phase12_shards, int):
        raise TypeError("phase12_shards must be an integer and not bool")
    if phase12_shards < 1:
        raise ValueError("phase12_shards must be positive")

    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    _verify_source_configs(plan)
    data = target / "data"
    phase7 = data / "phase7"
    phase8 = data / "phase8"
    phase9 = data / "phase9"
    phase11 = data / "phase11"
    phase12 = data / "phase12"
    checkpoints = data / ".phase12-checkpoints"
    progress_path = checkpoints / "progress.json"
    completion_path = data / "phase15_6_data_complete.json"
    resolution_path = data / "action_candidate_capacity_resolution.json"

    phase7_marker = _require_complete(phase7, "dataset_complete.json", "Phase 7")
    phase8_marker = _require_complete(phase8, "graph_complete.json", "Phase 8")
    phase9_marker = _require_complete(phase9, "action_complete.json", "Phase 9")
    _require_complete(phase11, "topology_complete.json", "Phase 11")

    if completion_path.is_file():
        completion = _read_json(completion_path)
        if completion.get("campaign_id") != plan["campaign_id"]:
            raise ValueError("data completion/campaign identity mismatch")
        return completion
    if (phase12 / "training_view_complete.json").is_file():
        print("[Phase 12] already published; finalizing campaign completion", flush=True)
    elif phase12.exists():
        raise FileExistsError(
            f"partial Phase 12 output exists without completion marker: {phase12}. "
            "Phase 12 publication is atomic; move this unexpected directory aside before retrying."
        )

    effective_config, action_resolution = _effective_training_config(
        plan=plan,
        phase9=phase9,
        resolution_path=resolution_path,
    )
    progress = _Phase12Progress(progress_path, phase12_workers)

    with _workspace_lock(target, "optimized-data"):
        if not (phase12 / "training_view_complete.json").is_file():
            started = time.monotonic()
            print(
                "[Phase 12] dedicated fast runner starting | "
                f"workers={phase12_workers} | shards={phase12_shards} | "
                "policy=lossless_no_sampling_no_truncation",
                flush=True,
            )
            views = build_training_view_result_fast(
                phase7,
                phase8,
                phase9,
                phase11,
                checkpoints,
                effective_config,
                shard_count=phase12_shards,
                resume_mode=resolved_resume_mode,
                progress_callback=progress,
                workers=phase12_workers,
            )
            write_training_view_dataset_resumable(
                views,
                phase12,
                checkpoints,
                progress_callback=progress,
                resume_mode=resolved_resume_mode,
            )
            print(
                f"[Phase 12] complete in {(time.monotonic() - started) / 60.0:.2f} minutes",
                flush=True,
            )
            if resolved_retention == "phase" and checkpoints.exists():
                shutil.rmtree(checkpoints)
                print("[Phase 12] checkpoints removed by retention policy", flush=True)

        phase12_marker = _read_json(phase12 / "training_view_complete.json")
        topology_capacity = (
            _read_json(data / "topology_capacity_resolution.json")
            if (data / "topology_capacity_resolution.json").is_file()
            else None
        )
        topology_group_count = (
            _read_json(data / "topology_group_count_resolution.json")
            if (data / "topology_group_count_resolution.json").is_file()
            else None
        )
        completion = {
            "schema": PHASE156_DATA_COMPLETE_SCHEMA,
            "complete": True,
            "campaign_id": plan["campaign_id"],
            "phase7_generation_id": phase7_marker["scientific_generation_id"],
            "phase8_graph_conversion_id": phase8_marker["graph_conversion_id"],
            "phase9_action_engine_id": phase9_marker["action_engine_id"],
            "phase12_training_view_dataset_id": phase12_marker[
                "training_view_dataset_id"
            ],
            "topology_capacity_resolution": topology_capacity,
            "topology_group_count_resolution": topology_group_count,
            "action_candidate_capacity_resolution": action_resolution,
            "restartability": {
                "phase7": "phase_boundary",
                "phase8": "phase_boundary",
                "phase9": "deterministic_shard",
                "phase11": (
                    "point_cloud_distance_per_manifold_persistence_final_group"
                ),
                "phase12": (
                    "parallel_deterministic_task_entity_hash_shard_and_item_artifact"
                ),
                "resume_mode": resolved_resume_mode,
                "checkpoint_retention": resolved_retention,
                "phase12_shards": phase12_shards,
                "phase12_workers": phase12_workers,
                "fast_existing_item_reuse": True,
            },
            "source_config_hashes": {
                name: record["sha256"]
                for name, record in sorted(plan["source_configs"].items())
            },
            "physical_hardware": False,
            "topology_loss_weight": 0.0,
        }
        _atomic_write_json(completion_path, completion)
        _update_state(target, data_complete=True)
        print("[Phase 15.6 data] complete", flush=True)
        return completion


__all__ = ["run_phase12_only"]
