#!/usr/bin/env python3
"""Guarded Step-9D v2 planner/executor.

This wrapper deliberately reuses the frozen v1 pilot construction/execution code
while adding two fail-closed boundaries before any physical QPU submission:

1. exact Qiskit/Aer/Runtime version matching;
2. one explicit IBM Quantum Open Plan instance frozen into the plan identity.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEGACY_RUNNER = ROOT / "scripts" / "v0_2" / "run_step9d_exploratory_qpu_pilot.py"
DEFAULT_CONFIG = ROOT / "configs" / "v0_2" / "step9d_exploratory_qpu_pilot_v2.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step9d_exploratory_qpu_pilot")
SCHEMA = "triqto.v0_2.step9d_exploratory_qpu_pilot.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deployment-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--backend-name", type=str)
    parser.add_argument("--instance-name", type=str)
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--execute-physical-qpu", action="store_true")
    parser.add_argument("--confirmation-token", type=str)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_frozen_versions(config: dict[str, Any]) -> dict[str, str]:
    expected = config["software_environment"]
    actual = {
        "qiskit": importlib.metadata.version("qiskit"),
        "qiskit_aer": importlib.metadata.version("qiskit-aer"),
        "qiskit_ibm_runtime": importlib.metadata.version("qiskit-ibm-runtime"),
    }
    for key, observed in actual.items():
        wanted = str(expected[key])
        if observed != wanted:
            raise RuntimeError(
                f"Step-9D v2 software drift: {key}={observed}, expected {wanted}. "
                "Use the frozen .venv-step9d environment before planning or execution."
            )
    return actual


def select_open_instance(instances: list[Any], requested_name: str | None) -> dict[str, str]:
    open_instances = [
        dict(row)
        for row in instances
        if isinstance(row, dict) and str(row.get("plan", "")).strip().lower() == "open"
    ]
    if requested_name:
        matches = [
            row
            for row in open_instances
            if requested_name in {str(row.get("name", "")), str(row.get("crn", ""))}
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"--instance-name {requested_name!r} did not identify exactly one Open Plan instance"
            )
        return matches[0]
    if len(open_instances) != 1:
        names = [str(row.get("name", row.get("crn", "unknown"))) for row in open_instances]
        raise RuntimeError(
            "Step-9D v2 requires one explicit Open Plan instance. "
            f"Found {len(open_instances)} Open instances: {names}. "
            "Re-run with --instance-name <name>."
        )
    return open_instances[0]


def load_legacy_runner() -> Any:
    spec = importlib.util.spec_from_file_location("triqto_step9d_v1_runner", LEGACY_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load frozen Step-9D v1 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_instance_service(instance: dict[str, str]) -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    crn = str(instance.get("crn", "")).strip()
    if not crn:
        raise RuntimeError("Open Plan instance has no CRN")
    service = QiskitRuntimeService(instance=crn)
    active = str(service.active_instance())
    if active != crn:
        raise RuntimeError(f"IBM Runtime active instance mismatch: {active} != {crn}")
    return service


def configure_legacy(legacy: Any, config: dict[str, Any], service: Any, instance: dict[str, str]) -> None:
    legacy.SCHEMA = SCHEMA
    legacy._make_service = lambda: service
    original_snapshot = legacy.backend_snapshot
    original_identity = legacy._plan_identity

    def guarded_snapshot(backend: Any, candidate: Any, ranking: list[dict[str, Any]]) -> dict[str, Any]:
        value = original_snapshot(backend, candidate, ranking)
        value["instance"] = {
            "crn": str(instance["crn"]),
            "name": str(instance.get("name", "")),
            "plan": str(instance.get("plan", "")),
        }
        return value

    def guarded_identity(config_path: Path, ensemble: Any, snapshot: dict[str, Any], metadata: list[dict[str, Any]]) -> dict[str, Any]:
        value = original_identity(config_path, ensemble, snapshot, metadata)
        value["instance_crn"] = str(instance["crn"])
        value["instance_name"] = str(instance.get("name", ""))
        value["instance_plan"] = str(instance.get("plan", ""))
        value["software_versions"] = verify_frozen_versions(config)
        return value

    legacy.backend_snapshot = guarded_snapshot
    legacy._plan_identity = guarded_identity


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    if config.get("schema") != SCHEMA or config.get("status") != "FROZEN_BEFORE_PHYSICAL_QPU_EXECUTION":
        raise RuntimeError("Step-9D v2 contract is not frozen")
    versions = verify_frozen_versions(config)

    from qiskit_ibm_runtime import QiskitRuntimeService

    discovery = QiskitRuntimeService()
    instance = select_open_instance(list(discovery.instances()), args.instance_name)
    if str(instance.get("plan", "")).lower() != "open":
        raise RuntimeError("Step-9D v2 paid-plan execution is forbidden")
    service = make_instance_service(instance)
    legacy = load_legacy_runner()
    configure_legacy(legacy, config, service, instance)

    device = legacy.resolve_device(args.device)
    bundle_dir = args.deployment_bundle_dir.expanduser().resolve()
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)

    if args.execute_physical_qpu:
        expected = str(config["execution"]["explicit_confirmation_token"])
        if args.confirmation_token != expected:
            raise RuntimeError(f"physical QPU execution requires --confirmation-token {expected}")
        if args.plan_file is None:
            raise RuntimeError("physical QPU execution requires --plan-file from a prior Step-9D v2 plan")
        plan_file = args.plan_file.expanduser().resolve()
        plan = read_json(plan_file)
        identity = plan.get("identity", {})
        if identity.get("instance_crn") != str(instance["crn"]):
            raise RuntimeError("saved plan instance CRN does not match the active Open Plan instance")
        if str(identity.get("instance_plan", "")).lower() != "open":
            raise RuntimeError("saved plan is not bound to an Open Plan instance")
        if identity.get("software_versions") != versions:
            raise RuntimeError("saved plan software-version identity does not match the current environment")
        if identity.get("config_sha256") != legacy.sha256_file(config_path):
            raise RuntimeError("Step-9D v2 config changed since planning; create a fresh plan")
        legacy.execute_plan(
            config_path=config_path,
            config=config,
            bundle_dir=bundle_dir,
            plan_file=plan_file,
            device=device,
        )
    else:
        if args.plan_file is not None:
            raise RuntimeError("--plan-file is only valid with --execute-physical-qpu")
        plan_file = legacy.make_plan(
            config_path=config_path,
            config=config,
            bundle_dir=bundle_dir,
            output_parent=output_parent,
            backend_name=args.backend_name,
            device=device,
        )
        plan = read_json(plan_file)
        snapshot = read_json(plan_file.parent / "backend_snapshot.json")
        if plan["identity"]["instance_crn"] != str(instance["crn"]):
            raise RuntimeError("Step-9D v2 failed to freeze the explicit instance CRN")
        if str(plan["identity"]["instance_plan"]).lower() != "open":
            raise RuntimeError("Step-9D v2 plan is not bound to Open Plan")
        print(f"IBM instance: {instance.get('name', '')} | plan={instance.get('plan', '')}")
        print(f"Frozen software: {versions}")
        print("Paid-plan execution allowed: NO")


if __name__ == "__main__":
    main()
