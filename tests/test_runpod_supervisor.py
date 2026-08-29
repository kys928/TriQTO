from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

supervisor = importlib.import_module("runpod_supervisor")


def test_supervisor_exits_when_no_active_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"reconcile": 0}

    def reconcile() -> None:
        calls["reconcile"] += 1

    monkeypatch.setattr(supervisor.control, "reconcile_detached", reconcile)
    monkeypatch.setattr(supervisor.control, "list_active_records", lambda: [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["runpod_supervisor.py", "--poll-seconds", "10", "--lease-minutes", "1"],
    )

    supervisor.main()

    assert calls["reconcile"] == 1


def test_supervisor_requests_handoff_without_terminating_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_values = iter([0.0, 61.0])
    active = [("triqto-control/active/run-test.json", {"control_run_id": "run-test"})]

    monkeypatch.setattr(supervisor.control, "reconcile_detached", lambda: None)
    monkeypatch.setattr(supervisor.control, "list_active_records", lambda: active)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runpod_supervisor.py", "--poll-seconds", "10", "--lease-minutes", "1"],
    )

    with pytest.raises(SystemExit) as excinfo:
        supervisor.main()

    assert excinfo.value.code == supervisor.HANDOFF_EXIT_CODE
