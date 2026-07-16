"""Resumable optimized Phase 15.6 data construction.

Each phase publishes independently and is skipped on restart when its strict
completion marker is present.  Later failures therefore never discard already
validated Phase 7/8/9/11 work.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from triqto.actions.config import ActionEngineConfig
from triqto.actions.parallel_pipeline import build_action_engine_result_parallel
from triqto.actions.sharded_artifacts import write_sharded_action_dataset
from triqto.data_generation import generate_dataset, load_generation_config, write_dataset
from triqto.graph import (
    GraphConversionConfig,
    convert_completed_dataset_to_graphs,
    write_graph_dataset,
)
from triqto.topology import (
    TopologyAuditConfig,
    build_topology_audit_result,
    write_topology_dataset,
)
from triqto.training_views import (
    build_training_view_result,
    load_training_view_config,
    write_training_view_dataset,
)

from .campaign import (
    PHASE156_DATA_COMPLETE_SCHEMA,
    _atomic_write_json,
    _campaign_values,
    _load_prepared_plan,
    _read_json,
    _update_state,
    _verify_source_configs,
    _workspace_lock,
)
from .config import Phase156CampaignConfig


def _require_complete_or_absent(root: Path, marker_name: str, phase_name: str) -> bool:
    marker = root / marker_name
    if marker.is_file():
        payload = _read_json(marker)
        if payload.get("complete") is not True:
            raise ValueError(f"{phase_name} completion marker is not complete")
        return True
    if root.exists():
        raise FileExistsError(
            f"partial {phase_name} root exists without {marker_name}: {root}. "
            "Move it aside or remove it only after confirming no writer is active."
        )
    return False


def _progress_line(payload: dict[str, Any]) -> None:
    eta = payload.get("eta_seconds")
    eta_text = "unknown" if eta is None else f"{float(eta) / 60.0:.1f}m"
    print(
        "[Phase 9] "
        f"{payload['completed_samples']}/{payload['total_samples']} samples | "
        f"{payload['candidate_count']} candidates | "
        f"{payload['samples_per_second']:.3f} samples/s | "
        f"ETA {eta_text} | workers={payload['workers']}",
        flush=True,
    )


def _timed_start(name: str) -> float:
    print(f"[{name}] starting", flush=True)
    return time.monotonic()


def _timed_done(name: str, started: float) -> None:
    print(f"[{name}] complete in {(time.monotonic() - started) / 60.0:.2f} minutes", flush=True)


def run_optimized_data_stage(
    *,
    workspace: str | Path,
    workers: int | None = None,
) -> dict[str, Any]:
    """Build Phase 7/8/9/11/12 with resume, progress, and ZIP-sharded Phase 9."""
    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    _verify_source_configs(plan)
    config = Phase156CampaignConfig(**_campaign_values(plan["campaign_config"]))
    build = config.data_build

    final_root = target / "data"
    completion_path = final_root / "phase15_6_data_complete.json"
    if completion_path.is_file():
        completion = _read_json(completion_path)
        if completion.get("campaign_id") != plan["campaign_id"]:
            raise ValueError("data completion/campaign identity mismatch")
        return completion

    phase7 = final_root / "phase7"
    phase8 = final_root / "phase8"
    phase9 = final_root / "phase9"
    phase11 = final_root / "phase11"
    phase12 = final_root / "phase12"

    with _workspace_lock(target, "optimized-data"):
        final_root.mkdir(parents=True, exist_ok=True)

        if _require_complete_or_absent(phase7, "dataset_complete.json", "Phase 7"):
            print("[Phase 7] already complete; resuming", flush=True)
        else:
            started = _timed_start("Phase 7")
            generation = generate_dataset(
                load_generation_config(plan["source_configs"]["generation"]["absolute_path"])
            )
            write_dataset(generation, phase7)
            _timed_done("Phase 7", started)

        if _require_complete_or_absent(phase8, "graph_complete.json", "Phase 8"):
            print("[Phase 8] already complete; resuming", flush=True)
        else:
            started = _timed_start("Phase 8")
            graph = convert_completed_dataset_to_graphs(
                phase7,
                GraphConversionConfig(
                    include_supplemental_counts=build.include_supplemental_counts
                ),
            )
            write_graph_dataset(graph, phase8)
            _timed_done("Phase 8", started)

        if _require_complete_or_absent(phase9, "action_complete.json", "Phase 9"):
            print("[Phase 9] already complete; resuming", flush=True)
        else:
            started = _timed_start("Phase 9")
            actions = build_action_engine_result_parallel(
                phase7,
                phase8,
                ActionEngineConfig(
                    candidate_magnitudes=build.action_candidate_magnitudes,
                    max_candidates_per_sample=build.max_candidates_per_sample,
                    max_edits_per_action=build.max_edits_per_action,
                ),
                workers=workers,
                progress_callback=_progress_line,
            )
            print("[Phase 9] computation complete; writing compressed shards", flush=True)
            write_sharded_action_dataset(actions, phase9)
            _timed_done("Phase 9", started)

        if _require_complete_or_absent(phase11, "topology_complete.json", "Phase 11"):
            print("[Phase 11] already complete; resuming", flush=True)
        else:
            started = _timed_start("Phase 11")
            topology = build_topology_audit_result(
                phase7,
                phase8,
                phase9,
                TopologyAuditConfig(
                    min_points=build.topology_min_points,
                    betti_grid_size=build.topology_betti_grid_size,
                    top_k_lifetimes=build.topology_top_k_lifetimes,
                    max_points_per_group=build.topology_max_points_per_group,
                    max_groups=build.topology_max_groups,
                    max_statevector_amplitudes=build.topology_max_statevector_amplitudes,
                    include_hilbert=build.topology_include_hilbert,
                ),
            )
            write_topology_dataset(topology, phase11)
            _timed_done("Phase 11", started)

        if _require_complete_or_absent(
            phase12,
            "training_view_complete.json",
            "Phase 12",
        ):
            print("[Phase 12] already complete; resuming", flush=True)
        else:
            started = _timed_start("Phase 12")
            views = build_training_view_result(
                phase7,
                phase8,
                phase9,
                phase11,
                load_training_view_config(
                    plan["source_configs"]["training_view"]["absolute_path"]
                ),
            )
            write_training_view_dataset(views, phase12)
            _timed_done("Phase 12", started)

        phase7_marker = _read_json(phase7 / "dataset_complete.json")
        phase8_marker = _read_json(phase8 / "graph_complete.json")
        phase9_marker = _read_json(phase9 / "action_complete.json")
        phase12_marker = _read_json(phase12 / "training_view_complete.json")
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


__all__ = ["run_optimized_data_stage"]
