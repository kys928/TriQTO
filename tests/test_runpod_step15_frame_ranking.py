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
    for operation in ("fit_and_verify", "evaluate_spent_holdout", "diagnose_step14_replay"):
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
    assert "fit_step15_frame_ranker_true_positive.py" in fit_joined
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
    diagnose = worker.build_commands(
        {"task": {"runner": "step15_frame_ranking", "operation": "diagnose_step14_replay"}}
    )
    assert [name for name, _cmd in diagnose] == ["pre_diagnostic_verify", "step14_replay_diagnostic"]
    assert "diagnose_step14_phasea_replay.py" in " ".join(diagnose[1][1])


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


def test_step15_fail_closed_restart_guard_preserves_original_failure(monkeypatch, tmp_path: Path) -> None:
    worker = load_module("runpod_step15_worker_restart_contract", WORKER)
    job = {
        "id": "step15-restart-unit",
        "_control_run_id": "run-100-deadbeef",
        "task": {"runner": "step15_frame_ranking", "operation": "diagnose_step14_replay"},
        "provenance": {"image_name": "ghcr.io/kys928/triqto-runpod:sha-" + "f" * 40},
    }
    monkeypatch.setattr(worker.common, "load_job", lambda: job)
    monkeypatch.setattr(worker.common, "CONTROL_ROOT", tmp_path)
    monkeypatch.setattr(worker.common, "gpu_snapshot", lambda: {"output": "fake CPU/CUDA"})
    calls = []
    def failed_subprocess(command, log_path):
        calls.append(list(command))
        log_path.write_text("scientific failure", encoding="utf-8")
        return 9
    monkeypatch.setattr(worker.common, "run_subprocess", failed_subprocess)
    assert worker.run() == 9
    run_dir = tmp_path / job["id"] / job["_control_run_id"]
    status = (run_dir / "status.json").read_bytes()
    assert json.loads(status)["state"] == "failed"
    assert len(calls) == 1
    def interrupt_sleep(_delay):
        raise RuntimeError("restart guard entered sleep")
    monkeypatch.setattr(worker.time, "sleep", interrupt_sleep)
    with pytest.raises(RuntimeError, match="restart guard entered sleep"):
        worker.run()
    assert (run_dir / "status.json").read_bytes() == status
    assert len(calls) == 1


def test_step15_fail_closed_restart_guard_marks_unfinished_worker_failed(monkeypatch, tmp_path: Path) -> None:
    worker = load_module("runpod_step15_worker_crash_contract", WORKER)
    job = {
        "id": "step15-crash-unit",
        "_control_run_id": "run-101-deadbeef",
        "task": {"runner": "step15_frame_ranking", "operation": "diagnose_step14_replay"},
        "provenance": {"image_name": "ghcr.io/kys928/triqto-runpod:sha-" + "d" * 40},
    }
    monkeypatch.setattr(worker.common, "load_job", lambda: job)
    monkeypatch.setattr(worker.common, "CONTROL_ROOT", tmp_path)
    monkeypatch.setattr(worker.common, "gpu_snapshot", lambda: {"output": "fake"})
    def abrupt_crash(_command, _log_path):
        raise KeyboardInterrupt("simulated abrupt failure")
    monkeypatch.setattr(worker.common, "run_subprocess", abrupt_crash)
    with pytest.raises(KeyboardInterrupt):
        worker.run()
    run_dir = tmp_path / job["id"] / job["_control_run_id"]
    status_path = run_dir / "status.json"
    # Simulate the host stopping after startup but before terminal status.
    json_status = json.loads(status_path.read_text())
    json_status["state"] = "running"
    status_path.write_text(json.dumps(json_status), encoding="utf-8")
    monkeypatch.setattr(worker.time, "sleep", lambda _delay: (_ for _ in ()).throw(RuntimeError("guarded")))
    with pytest.raises(RuntimeError, match="guarded"):
        worker.run()
    final = json.loads(status_path.read_text())
    assert final["state"] == "failed"
    assert final["reason"] == "restarted_before_terminal_scientific_status"


def test_replay_diagnostic_source_compiles_and_has_no_truth_dependency() -> None:
    path = ROOT / "scripts" / "v0_2" / "diagnose_step14_phasea_replay.py"
    content = path.read_text(encoding="utf-8")
    compile(content, str(path), "exec")
    assert "KNOWN_DISAGREEMENTS" in content
    assert '"sealed_truth_accessed":False' in content
    assert '("cpu","cuda")' in content
    assert "diagnostic_result.json" in content
