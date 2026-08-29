from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "runpod_worker.py"
LAUNCHER = ROOT / "scripts" / "runpod_step14_development.py"
WORKSPACE = "/workspace/triqto-data/step14_cross_motif_dataset"


def load_worker():
    spec = importlib.util.spec_from_file_location("runpod_worker_step14_contract", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step14_worker_builds_development_only_command() -> None:
    worker = load_worker()
    job = {
        "task": {
            "runner": "step14_cross_motif",
            "command": "generate_development",
            "workspace": WORKSPACE,
            "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
            "progress_every": 25,
        }
    }
    command = worker.build_step14_cross_motif_command(job)
    joined = " ".join(command)
    assert "generate_step14_cross_motif_dataset.py" in joined
    assert "--mode development" in joined
    assert f"--output-parent {WORKSPACE}" in joined
    assert "simulator_outer" not in joined
    assert "selection-freeze" not in joined


def test_step14_worker_rejects_outer_or_selection_freeze() -> None:
    worker = load_worker()
    base = {
        "runner": "step14_cross_motif",
        "command": "generate_development",
        "workspace": WORKSPACE,
    }
    with pytest.raises(ValueError):
        worker.build_step14_cross_motif_command({"task": {**base, "mode": "simulator_outer"}})
    with pytest.raises(ValueError):
        worker.build_step14_cross_motif_command({"task": {**base, "selection_freeze": "/tmp/freeze.json"}})
    with pytest.raises(ValueError):
        worker.build_step14_cross_motif_command({"task": {**base, "command": "generate_outer"}})


def test_step14_launcher_and_workflow_are_typed_and_compile() -> None:
    compile(LAUNCHER.read_text(encoding="utf-8"), str(LAUNCHER), "exec")
    source = LAUNCHER.read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "runpod-step14-development.yml").read_text(encoding="utf-8")
    assert '"runner": "step14_cross_motif"' in source
    assert '"command": "generate_development"' in source
    assert '"workspace": WORKSPACE' in source
    assert '"lifecycle": {"mode": "detached"}' in source
    assert 'cloudType": "SECURE"' in source
    assert "runpod_step14_development.py" in workflow
    assert "runpod/step14/*.json" in workflow
