"""Resumable, immutable Phase 15.6 campaign orchestration.

The campaign runner prepares and executes repository-supported stages. It does not
submit cloud jobs or physical-hardware work, and it never writes generated
research artifacts into the repository.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
from typing import Any, Iterator
import uuid

from triqto.actions import ActionEngineConfig, build_action_engine_result, write_action_dataset
from triqto.data_generation import generate_dataset, load_generation_config, write_dataset
from triqto.graph import GraphConversionConfig, convert_completed_dataset_to_graphs, write_graph_dataset
from triqto.model import load_model_config
from triqto.phase15_5 import load_phase155_config, load_phase15_5_result, run_phase15_5
from triqto.topology import TopologyAuditConfig, build_topology_audit_result, write_topology_dataset
from triqto.training import load_training_config, run_training
from triqto.training_views import (
    build_training_view_result,
    load_training_view_config,
    write_training_view_dataset,
)

from .config import Phase156CampaignConfig
from .planner import build_campaign_plan, plan_json

PHASE156_DATA_COMPLETE_SCHEMA = "triqto.phase15_6.data_complete.v1"
PHASE156_SEED_COMPLETE_SCHEMA = "triqto.phase15_6.seed_complete.v1"
PHASE156_AGGREGATE_SCHEMA = "triqto.phase15_6.aggregate.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _assert_external_workspace(repo_root: Path, workspace: Path) -> None:
    root = repo_root.resolve()
    target = workspace.resolve()
    if target == root or target.is_relative_to(root):
        raise ValueError(
            "Phase 15.6 generated artifacts must live outside the Git repository; "
            f"received workspace={target}"
        )


@contextmanager
def _workspace_lock(workspace: Path, stage: str) -> Iterator[None]:
    lock = workspace / f".phase15_6_{stage}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Phase 15.6 stage lock already exists: {lock}. "
            "Another writer may be active; remove it only after confirming no process is running."
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\nstage={stage}\n".encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        if lock.exists():
            lock.unlink()


def _load_prepared_plan(workspace: Path) -> dict[str, Any]:
    plan_path = workspace / "campaign_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(
            f"campaign is not prepared: {plan_path}; run the prepare stage first"
        )
    plan = _read_json(plan_path)
    if plan.get("schema") != "triqto.phase15_6.plan.v1":
        raise ValueError("unsupported or malformed Phase 15.6 campaign plan")
    if not isinstance(plan.get("campaign_id"), str):
        raise ValueError("Phase 15.6 campaign plan has no campaign_id")
    return plan


def prepare_campaign(
    *,
    repo_root: str | Path,
    workspace: str | Path,
    config: Phase156CampaignConfig,
) -> dict[str, Any]:
    """Create or verify an immutable campaign plan and source-config snapshots."""
    root = Path(repo_root).resolve()
    target = Path(workspace).expanduser().resolve()
    _assert_external_workspace(root, target)
    plan = build_campaign_plan(repo_root=root, config=config)
    plan_path = target / "campaign_plan.json"
    if plan_path.exists():
        existing = _read_json(plan_path)
        if existing.get("campaign_id") != plan["campaign_id"]:
            raise FileExistsError(
                "workspace already belongs to a different Phase 15.6 campaign"
            )
        return existing
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"unprepared Phase 15.6 workspace is not empty: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    snapshots = target / "source_config_snapshots"
    snapshots.mkdir()
    inventory: list[dict[str, Any]] = []
    for name, record in sorted(plan["source_configs"].items()):
        source = Path(record["absolute_path"])
        suffix = "".join(source.suffixes) or ".config"
        destination = snapshots / f"{name}{suffix}"
        shutil.copy2(source, destination)
        inventory.append(
            {
                "name": name,
                "reference": destination.relative_to(target).as_posix(),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    plan["source_snapshot_inventory"] = inventory
    plan_path.write_text(plan_json(plan), encoding="utf-8")
    _atomic_write_json(
        target / "campaign_state.json",
        {
            "schema": "triqto.phase15_6.state.v1",
            "campaign_id": plan["campaign_id"],
            "prepared": True,
            "data_complete": False,
            "completed_training_seeds": [],
            "completed_evaluation_seeds": [],
            "aggregate_complete": False,
            "physical_hardware": False,
            "topology_loss_weight": 0.0,
        },
    )
    return plan


def _verify_source_configs(plan: dict[str, Any]) -> None:
    for name, record in plan["source_configs"].items():
        path = Path(record["absolute_path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(
                f"Phase 15.6 source config changed after preparation: {name}"
            )


def run_data_stage(*, workspace: str | Path) -> dict[str, Any]:
    """Build one shared immutable Phase 7/8/9/11/12 research universe."""
    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    _verify_source_configs(plan)
    final_root = target / "data"
    completion_path = final_root / "phase15_6_data_complete.json"
    if completion_path.is_file():
        completion = _read_json(completion_path)
        if completion.get("campaign_id") != plan["campaign_id"]:
            raise ValueError("data completion/campaign identity mismatch")
        return completion
    if final_root.exists():
        raise FileExistsError(
            f"partial Phase 15.6 data root exists without completion marker: {final_root}"
        )

    config = Phase156CampaignConfig(**_campaign_values(plan["campaign_config"]))
    build = config.data_build
    staging = target / f".data.staging-{uuid.uuid4().hex}"
    with _workspace_lock(target, "data"):
        try:
            phase7 = staging / "phase7"
            phase8 = staging / "phase8"
            phase9 = staging / "phase9"
            phase11 = staging / "phase11"
            phase12 = staging / "phase12"

            generation = generate_dataset(
                load_generation_config(plan["source_configs"]["generation"]["absolute_path"])
            )
            write_dataset(generation, phase7)
            graph = convert_completed_dataset_to_graphs(
                phase7,
                GraphConversionConfig(
                    include_supplemental_counts=build.include_supplemental_counts
                ),
            )
            write_graph_dataset(graph, phase8)
            actions = build_action_engine_result(
                phase7,
                phase8,
                ActionEngineConfig(
                    candidate_magnitudes=build.action_candidate_magnitudes,
                    max_candidates_per_sample=build.max_candidates_per_sample,
                    max_edits_per_action=build.max_edits_per_action,
                ),
            )
            write_action_dataset(actions, phase9)
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

            completion = {
                "schema": PHASE156_DATA_COMPLETE_SCHEMA,
                "complete": True,
                "campaign_id": plan["campaign_id"],
                "phase7_generation_id": generation.scientific_generation_id,
                "phase8_graph_conversion_id": graph.graph_conversion_id,
                "phase9_action_engine_id": actions.action_engine_id,
                "phase12_training_view_dataset_id": views.training_view_dataset_id,
                "source_config_hashes": {
                    name: record["sha256"]
                    for name, record in sorted(plan["source_configs"].items())
                },
                "physical_hardware": False,
                "topology_loss_weight": 0.0,
            }
            _atomic_write_json(staging / "phase15_6_data_complete.json", completion)
            os.replace(staging, final_root)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    _update_state(target, data_complete=True)
    return completion


def _campaign_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce JSON plan values through the public strict config constructor."""
    values = dict(payload)
    values["training_seeds"] = tuple(values["training_seeds"])
    build = dict(values["data_build"])
    build["action_candidate_magnitudes"] = tuple(build["action_candidate_magnitudes"])
    values["data_build"] = build
    values["pod_requirements"] = dict(values["pod_requirements"])
    return values


def _selected_seeds(plan: dict[str, Any], seed: int | None) -> tuple[int, ...]:
    configured = tuple(int(value) for value in plan["campaign_config"]["training_seeds"])
    if seed is None:
        return configured
    if seed not in configured:
        raise ValueError(f"seed {seed} is not configured for this campaign")
    return (seed,)


def _final_checkpoint(result: Any) -> Any:
    finals = [checkpoint for checkpoint in result.checkpoints if checkpoint.kind == "final"]
    if len(finals) != 1:
        raise RuntimeError("Phase 14 training did not publish exactly one final checkpoint")
    return finals[0]


def run_training_stage(
    *,
    workspace: str | Path,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Train one or every configured Phase 14 seed against the shared Phase 12 data."""
    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    _verify_source_configs(plan)
    data_completion = _read_json(target / "data" / "phase15_6_data_complete.json")
    if data_completion.get("campaign_id") != plan["campaign_id"]:
        raise ValueError("Phase 15.6 data/campaign identity mismatch")
    config = Phase156CampaignConfig(**_campaign_values(plan["campaign_config"]))
    summaries: list[dict[str, Any]] = []

    with _workspace_lock(target, "train"):
        for training_seed in _selected_seeds(plan, seed):
            seed_root = target / "runs" / f"seed-{training_seed}"
            summary_path = seed_root / "phase15_6_seed_complete.json"
            if summary_path.is_file():
                summary = _read_json(summary_path)
                if summary.get("campaign_id") != plan["campaign_id"]:
                    raise ValueError("seed completion/campaign identity mismatch")
                summaries.append(summary)
                continue
            phase14_root = seed_root / "phase14"
            if phase14_root.exists():
                raise FileExistsError(
                    f"partial Phase 14 output exists without seed completion: {phase14_root}"
                )
            training_config = load_training_config(
                plan["source_configs"]["training"]["absolute_path"]
            )
            training_config = replace(
                training_config,
                run_name=f"{config.campaign_name}_seed_{training_seed}",
                seed=training_seed,
                device=config.execution_device,
            )
            model_config = load_model_config(
                plan["source_configs"]["model"]["absolute_path"]
            )
            model_config = replace(model_config, initialization_seed=training_seed)
            result = run_training(
                training_view_root=target / "data" / "phase12",
                output_root=phase14_root,
                training_config=training_config,
                model_config=model_config,
                phase7_root=target / "data" / "phase7",
            )
            checkpoint = _final_checkpoint(result)
            summary = {
                "schema": PHASE156_SEED_COMPLETE_SCHEMA,
                "complete": True,
                "campaign_id": plan["campaign_id"],
                "training_seed": training_seed,
                "training_run_id": result.training_run_id,
                "model_architecture_id": result.model_architecture_id,
                "training_view_dataset_id": result.training_view_dataset_id,
                "best_epoch": result.best_epoch,
                "best_validation_loss": result.best_validation_loss,
                "final_epoch": result.final_epoch,
                "global_step": result.global_step,
                "stopped_early": result.stopped_early,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_content_hash": checkpoint.content_hash,
                "checkpoint_reference": str(
                    (phase14_root / checkpoint.artifact_ref).relative_to(target)
                ),
                "requested_device": config.execution_device,
                "test_split_used_for_optimization": False,
                "physical_hardware": False,
                "topology_loss_weight": 0.0,
            }
            _atomic_write_json(summary_path, summary)
            summaries.append(summary)
    _update_state(
        target,
        completed_training_seeds=sorted(
            summary["training_seed"] for summary in summaries
        ),
        merge_seed_list=True,
    )
    return summaries


def run_evaluation_stage(
    *,
    workspace: str | Path,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run Phase 15.5 grouped benchmarking for trained campaign seeds."""
    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    _verify_source_configs(plan)
    config = Phase156CampaignConfig(**_campaign_values(plan["campaign_config"]))
    if not config.run_phase15_5:
        raise ValueError("Phase 15.5 evaluation is disabled in this campaign")
    reports: list[dict[str, Any]] = []

    with _workspace_lock(target, "evaluate"):
        for training_seed in _selected_seeds(plan, seed):
            seed_root = target / "runs" / f"seed-{training_seed}"
            seed_summary = _read_json(seed_root / "phase15_6_seed_complete.json")
            output = seed_root / "phase15_5"
            if output.exists():
                loaded = load_phase15_5_result(output)
                reports.append(loaded["report"])
                continue
            phase155_config = load_phase155_config(
                plan["source_configs"]["phase15_5"]["absolute_path"]
            )
            phase155_config = replace(phase155_config, seed=training_seed)
            result = run_phase15_5(
                phase7_root=target / "data" / "phase7",
                training_view_root=target / "data" / "phase12",
                training_root=seed_root / "phase14",
                checkpoint=target / seed_summary["checkpoint_reference"],
                output_root=output,
                config=phase155_config,
            )
            reports.append(result["summary"])
    _update_state(
        target,
        completed_evaluation_seeds=sorted(_selected_seeds(plan, seed)),
        merge_seed_list=True,
    )
    return reports


def aggregate_campaign(*, workspace: str | Path) -> dict[str, Any]:
    """Aggregate seed-level reports without inventing a paper-level claim."""
    target = Path(workspace).expanduser().resolve()
    plan = _load_prepared_plan(target)
    output = target / "aggregate" / "cross_seed_summary.json"
    if output.is_file():
        payload = _read_json(output)
        if payload.get("campaign_id") != plan["campaign_id"]:
            raise ValueError("aggregate/campaign identity mismatch")
        return payload
    reports: list[tuple[int, dict[str, Any]]] = []
    for seed in _selected_seeds(plan, None):
        report_path = target / "runs" / f"seed-{seed}" / "phase15_5" / "benchmark_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"missing Phase 15.5 report for seed {seed}: {report_path}"
            )
        reports.append((seed, _read_json(report_path)))

    metric_names = ("random_control", "no_op_control", "family_heuristic")
    improvements: dict[str, list[float]] = {name: [] for name in metric_names}
    trained_utilities: list[float] = []
    trained_regrets: list[float] = []
    per_seed: list[dict[str, Any]] = []
    for seed, report in reports:
        trained = report["aggregate"]["trained_policy"]
        trained_utilities.append(float(trained["mean_selected_utility"]))
        trained_regrets.append(float(trained["mean_regret"]))
        deltas = {
            name: float(report["trained_minus_controls"][name])
            for name in metric_names
        }
        for name, value in deltas.items():
            improvements[name].append(value)
        per_seed.append(
            {
                "seed": seed,
                "phase15_5_run_id": report["phase15_5_run_id"],
                "policy_checkpoint_id": report["policy_checkpoint_id"],
                "test_group_count": report["test_group_count"],
                "trained_mean_selected_utility": trained_utilities[-1],
                "trained_mean_regret": trained_regrets[-1],
                "trained_minus_controls": deltas,
            }
        )

    def summary(values: list[float]) -> dict[str, Any]:
        return {
            "mean": statistics.fmean(values),
            "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values),
            "maximum": max(values),
            "positive_seed_fraction": sum(value > 0.0 for value in values) / len(values),
            "seed_count": len(values),
        }

    payload = {
        "schema": PHASE156_AGGREGATE_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "campaign_name": plan["campaign_config"]["campaign_name"],
        "seed_count": len(reports),
        "per_seed": per_seed,
        "trained_mean_selected_utility": summary(trained_utilities),
        "trained_mean_regret": summary(trained_regrets),
        "trained_minus_controls": {
            name: summary(values) for name, values in improvements.items()
        },
        "diagnostic_success_gates": {
            "positive_vs_no_op_all_seeds": all(
                value > 0.0 for value in improvements["no_op_control"]
            ),
            "positive_vs_random_all_seeds": all(
                value > 0.0 for value in improvements["random_control"]
            ),
            "positive_vs_family_heuristic_majority": (
                sum(value > 0.0 for value in improvements["family_heuristic"])
                > len(reports) / 2
            ),
            "gate_status_is_not_a_research_claim": True,
        },
        "claim_boundaries": {
            "research_quality_claim": False,
            "physical_hardware": False,
            "broad_ood_claim": False,
            "calibrated_uncertainty_claim": False,
            "topology_benefit_claim": False,
            "topology_loss_weight": 0.0,
        },
    }
    _atomic_write_json(output, payload)
    _update_state(target, aggregate_complete=True)
    return payload


def _update_state(
    workspace: Path,
    *,
    merge_seed_list: bool = False,
    **updates: Any,
) -> None:
    path = workspace / "campaign_state.json"
    state = _read_json(path)
    for key, value in updates.items():
        if merge_seed_list and key in {
            "completed_training_seeds",
            "completed_evaluation_seeds",
        }:
            existing = {int(item) for item in state.get(key, [])}
            existing.update(int(item) for item in value)
            state[key] = sorted(existing)
        else:
            state[key] = value
    _atomic_write_json(path, state)


__all__ = [
    "PHASE156_AGGREGATE_SCHEMA",
    "PHASE156_DATA_COMPLETE_SCHEMA",
    "PHASE156_SEED_COMPLETE_SCHEMA",
    "aggregate_campaign",
    "prepare_campaign",
    "run_data_stage",
    "run_evaluation_stage",
    "run_training_stage",
]
