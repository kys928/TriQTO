"""Lossless Phase 11 group-capacity wrapper for the optimized data pipeline.

This module keeps the established Phase 7/8/9/11/12 implementation intact while
resolving the real Phase 11 group count immediately before the topology config is
created. It also adds durable Phase 11 terminal telemetry and rolling runtime ETAs.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from . import optimized_data as _base
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
    """Run the optimized pipeline with lossless group-count expansion and telemetry."""
    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    campaign = Phase156CampaignConfig(**_campaign_values(plan["campaign_config"]))
    build = campaign.data_build
    data_root = target / "data"
    phase7 = data_root / "phase7"
    phase9 = data_root / "phase9"
    resolution_path = data_root / "topology_group_count_resolution.json"
    progress_path = data_root / ".phase11-checkpoints" / "progress.json"

    _GROUP_STARTED.clear()
    _GROUP_DURATIONS.clear()
    resolution_cache: dict[str, Any] = {}

    def ensure_resolution() -> tuple[int, dict[str, Any]]:
        cached = resolution_cache.get("payload")
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
        _atomic_write_json(resolution_path, payload)
        resolution_cache["payload"] = payload
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

    original_config_factory = _base.TopologyAuditConfig
    original_progress = _base._phase11_progress

    def topology_config_factory(*args: Any, **kwargs: Any):
        effective, _payload = ensure_resolution()
        kwargs["max_groups"] = effective
        return original_config_factory(*args, **kwargs)

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
    try:
        if (phase7 / "dataset_complete.json").is_file() and (
            phase9 / "action_complete.json"
        ).is_file():
            ensure_resolution()
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

    if resolution_path.is_file():
        resolution = _read_json(resolution_path)
        result = {**result, "topology_group_count_resolution": resolution}
        completion_path = data_root / "phase15_6_data_complete.json"
        if completion_path.is_file():
            completion = _read_json(completion_path)
            completion["topology_group_count_resolution"] = resolution
            _atomic_write_json(completion_path, completion)
    return result


__all__ = ["run_optimized_data_stage"]
