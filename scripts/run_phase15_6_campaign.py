#!/usr/bin/env python3
"""Prepare or execute a resumable Phase 15.6 research campaign on a user-managed pod."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triqto.phase15_6 import (  # noqa: E402
    Phase156CampaignConfig,
    aggregate_campaign,
    inspect_phase156_environment,
    load_phase156_config,
    prepare_campaign,
    run_data_stage,
    run_evaluation_stage,
    run_training_stage,
)


def _json(value: object) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def _prepared_config(workspace: Path) -> Phase156CampaignConfig:
    plan = json.loads((workspace / "campaign_plan.json").read_text(encoding="utf-8"))
    payload = dict(plan["campaign_config"])
    payload["training_seeds"] = tuple(payload["training_seeds"])
    build = dict(payload["data_build"])
    build["action_candidate_magnitudes"] = tuple(build["action_candidate_magnitudes"])
    payload["data_build"] = build
    return Phase156CampaignConfig(**payload)


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="External persistent volume path; must be outside the Git checkout.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create or verify an immutable campaign plan.")
    _add_workspace(prepare_parser)
    prepare_parser.add_argument("--config", default="configs/experiments/phase15_6_research_pilot.json")
    prepare_parser.add_argument("--repo-root", default=str(ROOT))

    preflight_parser = subparsers.add_parser("preflight", help="Check CPU, memory, disk, package, and CUDA requirements.")
    _add_workspace(preflight_parser)

    data_parser = subparsers.add_parser("data", help="Generate Phase 7 and build Phase 8/9/11/12 once.")
    _add_workspace(data_parser)

    train_parser = subparsers.add_parser("train", help="Train every configured seed or one selected seed.")
    _add_workspace(train_parser)
    train_parser.add_argument("--seed", type=int)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run Phase 15.5 evaluation for trained seeds.")
    _add_workspace(evaluate_parser)
    evaluate_parser.add_argument("--seed", type=int)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate completed seed-level benchmark reports.")
    _add_workspace(aggregate_parser)

    all_parser = subparsers.add_parser("all", help="Execute prepare, preflight, data, train, evaluate, and aggregate.")
    _add_workspace(all_parser)
    all_parser.add_argument("--config", default="configs/experiments/phase15_6_research_pilot.json")
    all_parser.add_argument("--repo-root", default=str(ROOT))

    args = parser.parse_args()
    workspace = Path(args.workspace).expanduser().resolve()

    if args.command == "prepare":
        config = load_phase156_config(args.config)
        _json(prepare_campaign(repo_root=args.repo_root, workspace=workspace, config=config))
        return

    if args.command == "preflight":
        config = _prepared_config(workspace)
        report = inspect_phase156_environment(
            workspace=workspace,
            requirements=config.pod_requirements,
            training_device=config.execution_device,
        )
        _json(report)
        if not report["ready"]:
            raise SystemExit(2)
        return

    if args.command == "data":
        _json(run_data_stage(workspace=workspace))
        return

    if args.command == "train":
        _json(run_training_stage(workspace=workspace, seed=args.seed))
        return

    if args.command == "evaluate":
        _json(run_evaluation_stage(workspace=workspace, seed=args.seed))
        return

    if args.command == "aggregate":
        _json(aggregate_campaign(workspace=workspace))
        return

    config = load_phase156_config(args.config)
    plan = prepare_campaign(repo_root=args.repo_root, workspace=workspace, config=config)
    report = inspect_phase156_environment(
        workspace=workspace,
        requirements=config.pod_requirements,
        training_device=config.execution_device,
    )
    if not report["ready"]:
        _json({"campaign_id": plan["campaign_id"], "environment": report})
        raise SystemExit(2)
    data = run_data_stage(workspace=workspace)
    training = run_training_stage(workspace=workspace)
    evaluation = run_evaluation_stage(workspace=workspace) if config.run_phase15_5 else []
    aggregate = aggregate_campaign(workspace=workspace) if config.run_phase15_5 else None
    _json({
        "campaign_id": plan["campaign_id"],
        "data": data,
        "training": training,
        "evaluation_run_ids": [value["phase15_5_run_id"] for value in evaluation],
        "aggregate": aggregate,
    })


if __name__ == "__main__":
    main()
