from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "runpod_step15_frame_ranking.py"
WORKER = ROOT / "scripts" / "runpod_step15_worker.py"
WORKFLOW = ROOT / ".github" / "workflows" / "runpod-step15-frame-ranking.yml"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_step15_launcher_requires_immutable_sha_image(tmp_path: Path) -> None:
    launcher = load_module("runpod_step15_launcher_contract", LAUNCHER)
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps(
            {
                "operation": "fit_and_verify",
                "image_name": "ghcr.io/kys928/triqto-runpod:sha-" + "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    assert launcher.load_request(good)["operation"] == "fit_and_verify"

    for image in (
        "ghcr.io/kys928/triqto-runpod:latest",
        "ghcr.io/kys928/triqto-runpod:sha-deadbeef",
        "docker.io/kys928/triqto-runpod:sha-" + "a" * 40,
    ):
        bad = tmp_path / (image.split(":")[-1] + ".json")
        bad.write_text(json.dumps({"operation": "fit_and_verify", "image_name": image}), encoding="utf-8")
        with pytest.raises(ValueError):
            launcher.load_request(bad)


def test_step15_launcher_only_allows_two_scientific_operations(tmp_path: Path) -> None:
    launcher = load_module("runpod_step15_launcher_ops", LAUNCHER)
    image = "ghcr.io/kys928/triqto-runpod:sha-" + "b" * 40
    for operation in ("fit_and_verify", "evaluate_spent_holdout"):
        path = tmp_path / f"{operation}.json"
        path.write_text(json.dumps({"operation": operation, "image_name": image}), encoding="utf-8")
        assert launcher.load_request(path)["operation"] == operation
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"operation": "fit_on_holdout", "image_name": image}), encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.load_request(bad)


def test_step15_worker_separates_fit_verify_from_spent_holdout_evaluation() -> None:
    worker = load_module("runpod_step15_worker_contract", WORKER)
    fit = worker.build_commands(
        {"task": {"runner": "step15_frame_ranking", "operation": "fit_and_verify"}}
    )
    fit_joined = " ".join(token for _name, command in fit for token in command)
    assert "fit_step15_frame_ranker.py" in fit_joined
    assert "verify_step15_frame_ranker_freeze.py" in fit_joined
    assert "evaluate_step15_spent_holdout.py" not in fit_joined

    audit = worker.build_commands(
        {"task": {"runner": "step15_frame_ranking", "operation": "evaluate_spent_holdout"}}
    )
    names = [name for name, _command in audit]
    audit_joined = " ".join(token for _name, command in audit for token in command)
    assert names[0] == "pre_evaluation_verify"
    assert "verify_step15_frame_ranker_freeze.py" in audit_joined
    assert "evaluate_step15_spent_holdout.py" in audit_joined
    assert audit_joined.index("verify_step15_frame_ranker_freeze.py") < audit_joined.index("evaluate_step15_spent_holdout.py")


def test_step15_runpod_surface_is_typed_and_compiles() -> None:
    compile(LAUNCHER.read_text(encoding="utf-8"), str(LAUNCHER), "exec")
    compile(WORKER.read_text(encoding="utf-8"), str(WORKER), "exec")
    launch_source = LAUNCHER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert '"runner": "step15_frame_ranking"' in launch_source
    assert '"cloudType": "SECURE"' in launch_source
    assert '"lifecycle": {"mode": "detached"}' in launch_source
    assert "sha-[0-9a-f]{40}" in launch_source
    assert "runpod_step15_frame_ranking.py" in workflow
    assert "runpod/step15/*.json" in workflow
