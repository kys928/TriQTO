"""Lossless capacity wrapper for the optimized Phase 15.6 data pipeline.

This module keeps the established Phase 7/8/9/11/12 implementation intact while
resolving the real Phase 11 group count and Phase 12 variable action-set capacity
before those operational configs are created. It also adds durable Phase 11 terminal
telemetry and rolling runtime ETAs.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any

from . import optimized_data as _base
from .action_candidate_capacity import resolve_action_candidate_capacity
from .campaign import (
    _atomic_write_json,
    _campaign_values,
    _load_prepared_plan,
    _read_json,
)
from .config import Phase156CampaignConfig
from .topology_group_count import resolve_topology_group_count


_GROUP_STARTED: dict[str, float] = {}
_GROUP_DURATIONS: list[float] = []


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(
        path,
        {
            "schema": "triqto.phase15_6.phase11_progress.v1",
            **payload,
        },
    )


def run_optimized_data_stage(
    *,
    workspace: str | Path,
    workers: int | None = None,
    phase11_workers: int = 1,
    phase12_shards: int = 256,
    resume_mode: str = "strict",
    checkpoint_retention: str = "campaign",
) -> dict[str, Any]:
    """Run the optimized pipeline with lossless capacity expansion and telemetry."""
    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    campaign = Phase156CampaignConfig(**_campaign_values(plan["campaign_config"]))
    build = campaign.data_build
    data_root = target / "data"
    phase7 = data_root / "phase7"
    phase9 = data_root / "phase9"
    topology_resolution_path = data_root / "topology_group_count_resolution.json"
    action_resolution_path = data_root / "action_candidate_capacity_resolution.json"
    progress_path = data_root / ".phase11-checkpoints" / "progress.json"
    training_config_path = Path(
        plan["source_configs"]["training_view"]["absolute_path"]
    ).expanduser().resolve()

    _GROUP_STARTED.clear()
    _GROUP_DURATIONS.clear()
    topology_resolution_cache: dict[str, Any] = {}
    action_resolution_cache: dict[str, Any] = {}

    original_training_config_loader = _base.load_training_view_config
    original_phase12_builder = _base.build_training_view_result_resumable

    def ensure_topology_resolution() -> tuple[int, dict[str, Any]]:
        cached = topology_resolution_cache.get("payload")
        if isinstance(cached, dict):
            return int(cached["effective_max_groups"]), cached
        _write_progress(
            progress_path,
            {
                "stage": "resolving_group_count",
                "detail": "scanning immutable Phase 7/9 manifests without hydrating artifacts",
            },
        )
        effective, payload = resolve_topology_group_count(
            phase7,
            phase9,
            build.topology_max_groups,
            min_points=build.topology_min_points,
        )
        payload = {
            **payload,
            "campaign_id": plan["campaign_id"],
            "phase7_generation_id": _read_json(phase7 / "dataset_complete.json")[
                "scientific_generation_id"
            ],
            "phase9_action_engine_id": _read_json(phase9 / "action_complete.json")[
                "action_engine_id"
            ],
        }
        _atomic_write_json(topology_resolution_path, payload)
        topology_resolution_cache["payload"] = payload
        if payload["auto_expanded"]:
            print(
                "[Phase 11] topology group capacity auto-expanded losslessly: "
                f"{payload['requested_max_groups']:,} -> "
                f"{payload['effective_max_groups']:,} groups",
                flush=True,
            )
        print(
            "[Phase 11][preflight-plan] "
            f"group_kinds={payload['group_kind_counts']} | "
            f"total_groups={payload['required_group_count']:,} | "
            f"total_points={payload['total_group_point_count']:,} | "
            f"sum_squared_group_sizes={payload['sum_squared_group_sizes']:,} | "
            f"max_group_sizes={payload['maximum_group_sizes']} | "
            "policy=no_sampling_no_truncation",
            flush=True,
        )
        _write_progress(
            progress_path,
            {
                "stage": "group_count_ready",
                "total_groups": int(payload["required_group_count"]),
                "total_group_point_count": int(payload["total_group_point_count"]),
                "sum_squared_group_sizes": int(payload["sum_squared_group_sizes"]),
            },
        )
        return effective, payload

    def requested_training_config():
        cached = action_resolution_cache.get("requested_config")
        if cached is not None:
            return cached
        config = original_training_config_loader(training_config_path)
        action_resolution_cache["requested_config"] = config
        return config

    def ensure_action_resolution():
        cached = action_resolution_cache.get("payload")
        effective_config = action_resolution_cache.get("effective_config")
        if isinstance(cached, dict) and effective_config is not None:
            return effective_config, cached

        requested_config = requested_training_config()
        effective_candidates, effective_refs, payload = (
            resolve_action_candidate_capacity(
                phase9,
                requested_config.max_candidates_per_item,
                requested_config.max_source_refs_per_item,
            )
        )
        payload = {
            **payload,
            "campaign_id": plan["campaign_id"],
            "phase9_action_engine_id": _read_json(phase9 / "action_complete.json")[
                "action_engine_id"
            ],
        }
        effective_config = replace(
            requested_config,
            max_candidates_per_item=effective_candidates,
            max_source_refs_per_item=effective_refs,
        )
        _atomic_write_json(action_resolution_path, payload)
        action_resolution_cache["payload"] = payload
        action_resolution_cache["effective_config"] = effective_config
        if payload["candidate_capacity_auto_expanded"]:
            print(
                "[Phase 12] action candidate capacity auto-expanded losslessly: "
                f"{payload['requested_max_candidates_per_item']:,} -> "
                f"{payload['effective_max_candidates_per_item']:,} candidates/item",
                flush=True,
            )
        if payload["source_ref_capacity_auto_expanded"]:
            print(
                "[Phase 12] action source-reference capacity auto-expanded losslessly: "
                f"{payload['requested_max_source_refs_per_item']:,} -> "
                f"{payload['effective_max_source_refs_per_item']:,} refs/item",
                flush=True,
            )
        print(
            "[Phase 12][preflight-plan] "
            f"samples={payload['sample_count']:,} | "
            f"candidate_rows={payload['candidate_row_count']:,} | "
            f"max_candidates={payload['required_max_candidates_per_item']:,} | "
            f"effective_source_refs={payload['effective_max_source_refs_per_item']:,} | "
            "policy=no_sampling_no_truncation",
            flush=True,
        )
        return effective_config, payload

    original_config_factory = _base.TopologyAuditConfig
    original_progress = _base._phase11_progress

    def topology_config_factory(*args: Any, **kwargs: Any):
        effective, _payload = ensure_topology_resolution()
        kwargs["max_groups"] = effective
        return original_config_factory(*args, **kwargs)

    def training_config_loader(path: str | Path):
        resolved = Path(path).expanduser().resolve()
        if resolved != training_config_path:
            return original_training_config_loader(path)
        effective_config, _payload = ensure_action_resolution()
        return effective_config

    def phase12_builder(*args: Any, **kwargs: Any):
        _effective_config, payload = ensure_action_resolution()
        checkpoint_root = None
        if len(args) >= 5:
            checkpoint_root = Path(args[4])
        elif "checkpoint_root" in kwargs:
            checkpoint_root = Path(kwargs["checkpoint_root"])
        active_resume_mode = str(kwargs.get("resume_mode", "strict"))
        has_existing_checkpoints = False
        if checkpoint_root is not None and checkpoint_root.is_dir():
            try:
                has_existing_checkpoints = next(checkpoint_root.iterdir(), None) is not None
            except OSError:
                has_existing_checkpoints = True
        if (
            payload["auto_expanded"]
            and active_resume_mode == "strict"
            and has_existing_checkpoints
        ):
            kwargs["resume_mode"] = "repair"
            print(
                "[Phase 12] action capacity changed after earlier checkpoints; "
                "migrating Phase 12 logical-shard identities in repair mode. "
                "Content-addressed item artifacts remain reusable.",
                flush=True,
            )
        return original_phase12_builder(*args, **kwargs)

    def phase11_progress(payload: dict[str, Any]) -> None:
        original_progress(payload)
        event = str(payload.get("event", "unknown"))
        if event == "plan":
            _write_progress(
                progress_path,
                {
                    "stage": "group_plan_ready",
                    "total_groups": int(payload["total_groups"]),
                    "workers": int(payload["workers"]),
                    "estimated_checkpoint_bytes": int(
                        payload["estimated_checkpoint_bytes"]
                    ),
                    "execution_order": "deterministic_smallest_groups_first",
                },
            )
            return

        group_key = str(payload.get("group_key", ""))
        now = time.monotonic()
        if group_key and group_key not in _GROUP_STARTED:
            _GROUP_STARTED[group_key] = now
        status = {
            "stage": str(payload["stage"]),
            "stage_status": str(payload["stage_status"]),
            "current_group_index": int(payload["current_group_index"]),
            "total_groups": int(payload["total_groups"]),
            "group_kind": str(payload["group_kind"]),
            "group_key": group_key,
            "point_count": int(payload["point_count"]),
            "resumed_stages": int(payload.get("resumed_stages", 0)),
        }
        if payload.get("elapsed_seconds") is not None:
            status["stage_elapsed_seconds"] = float(payload["elapsed_seconds"])

        if payload.get("stage_status") == "group_complete" and group_key:
            started = _GROUP_STARTED.pop(group_key, now)
            group_elapsed = max(now - started, 0.0)
            _GROUP_DURATIONS.append(group_elapsed)
            completed = int(
                payload.get("completed_groups", payload["current_group_index"])
            )
            total = int(payload["total_groups"])
            active_workers = max(1, int(payload.get("workers", phase11_workers)))
            recent = _GROUP_DURATIONS[-25:]
            recent_mean = sum(recent) / len(recent)
            overall_mean = sum(_GROUP_DURATIONS) / len(_GROUP_DURATIONS)
            eta_seconds = recent_mean * max(total - completed, 0) / active_workers
            status.update(
                {
                    "completed_groups": completed,
                    "group_elapsed_seconds": group_elapsed,
                    "recent_mean_group_seconds": recent_mean,
                    "overall_mean_group_seconds": overall_mean,
                    "eta_seconds": eta_seconds,
                }
            )
            print(
                "[Phase 11][ETA] "
                f"groups={completed:,}/{total:,} | "
                f"last={group_elapsed/60.0:.2f}m | "
                f"recent_mean={recent_mean/60.0:.2f}m | "
                f"overall_mean={overall_mean/60.0:.2f}m | "
                f"ETA≈{eta_seconds/3600.0:.2f}h | workers={active_workers}",
                flush=True,
            )
        _write_progress(progress_path, status)

    _base.TopologyAuditConfig = topology_config_factory
    _base._phase11_progress = phase11_progress
    _base.load_training_view_config = training_config_loader
    _base.build_training_view_result_resumable = phase12_builder
    try:
        if (phase7 / "dataset_complete.json").is_file() and (
            phase9 / "action_complete.json"
        ).is_file():
            ensure_topology_resolution()
            ensure_action_resolution()
        result = _base.run_optimized_data_stage(
            workspace=target,
            workers=workers,
            phase11_workers=phase11_workers,
            phase12_shards=phase12_shards,
            resume_mode=resume_mode,
            checkpoint_retention=checkpoint_retention,
        )
    finally:
        _base.TopologyAuditConfig = original_config_factory
        _base._phase11_progress = original_progress
        _base.load_training_view_config = original_training_config_loader
        _base.build_training_view_result_resumable = original_phase12_builder

    completion_path = data_root / "phase15_6_data_complete.json"
    if topology_resolution_path.is_file():
        topology_resolution = _read_json(topology_resolution_path)
        result = {
            **result,
            "topology_group_count_resolution": topology_resolution,
        }
        if completion_path.is_file():
            completion = _read_json(completion_path)
            completion["topology_group_count_resolution"] = topology_resolution
            _atomic_write_json(completion_path, completion)
    if action_resolution_path.is_file():
        action_resolution = _read_json(action_resolution_path)
        result = {
            **result,
            "action_candidate_capacity_resolution": action_resolution,
        }
        if completion_path.is_file():
            completion = _read_json(completion_path)
            completion["action_candidate_capacity_resolution"] = action_resolution
            _atomic_write_json(completion_path, completion)
    return result


__all__ = ["run_optimized_data_stage"]
