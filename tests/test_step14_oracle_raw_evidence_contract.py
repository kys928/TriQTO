from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "v0_2"))

import analyze_step14_oracle_raw_evidence_ceiling as oracle  # noqa: E402
import run_step14_oracle_raw_evidence_ceiling as oracle_runner  # noqa: E402
import runpod_reconcile_hardened as hardened  # noqa: E402
import runpod_step14_training_worker as worker  # noqa: E402


def _loaded_graph() -> dict[str, np.ndarray]:
    return {
        "x__layout_logical_to_physical": np.asarray([0, 1, 2], dtype=np.int64),
        "x__graph_gate_names": np.asarray(["h", "cx", "ry", "cz", "rz"]),
        "x__graph_gate_qubit_ptr": np.asarray([0, 1, 3, 4, 6, 7], dtype=np.int64),
        "x__graph_gate_qubit_indices": np.asarray([0, 0, 1, 1, 1, 2, 2], dtype=np.int64),
        "x__graph_gate_parameter_ptr": np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64),
    }


def _root(**updates: object) -> dict[str, str]:
    value = {
        "operation_signature": json.dumps(["h:q0", "cx:q0-q1", "ry:q1", "cz:q1-q2", "rz:q2"]),
        "affected_qubit": "1",
        "injection_boundary_rank": "3",
        "injection_context_class": "inter_entangling",
    }
    value.update({key: str(item) for key, item in updates.items()})
    return value


def test_ordinary_context_is_independent_of_privileged_oracle_fields() -> None:
    loaded = _loaded_graph()
    reference = oracle.ordinary_circuit_context(_root(), loaded)
    changed = oracle.ordinary_circuit_context(
        _root(affected_qubit=2, injection_boundary_rank=1, injection_context_class="pre_entangling"),
        loaded,
    )
    np.testing.assert_array_equal(reference, changed)


def test_oracle_local_context_changes_with_true_boundary() -> None:
    localized = np.asarray([0.1, -0.2, 0.3, 0.05, 0.04, -0.01], dtype=np.float64)
    before = oracle.local_insertion_context(_root(injection_boundary_rank=2), localized)
    after = oracle.local_insertion_context(_root(injection_boundary_rank=4), localized)
    assert before.shape == after.shape
    assert not np.array_equal(before, after)


def test_frozen_generator_seventeen_gate_signature_is_supported() -> None:
    # Exact shape that reproduced the failed RunPod attempt at frozen root 32.
    signature = json.dumps([
        "ry:q2", "rx:q0", "cx:q2-q0", "rz:q0", "cz:q2-q1", "h:q2",
        "cx:q2-q3", "h:q0", "rx:q0", "cz:q2-q0", "rz:q0", "h:q4",
        "cz:q2-q3", "rx:q2", "rx:q4", "cx:q2-q0", "h:q2",
    ])
    original = oracle.MAX_GATES
    try:
        oracle_runner.apply_frozen_support_bound()
        parsed = oracle.parse_operation_signature(signature)
        assert len(parsed) == 17
        assert oracle.MAX_GATES == oracle_runner.FROZEN_STEP14_MAX_REFERENCE_OPERATIONS
    finally:
        oracle.MAX_GATES = original


def test_oracle_stage_is_explicitly_typed_and_post_selection_only() -> None:
    job = {
        "task": {
            "runner": "step14_training_selection",
            "command": "decompose_oracle_raw_evidence",
            "workspace": "/workspace/triqto-data/step14_cross_motif_training",
            "config": "configs/v0_2/step14_cross_motif_generalization_training.json",
            "progress_every": 1000,
            "expected_training_run_id": "training_18e0b4ed6e685af30b6c4a35",
            "expected_selection_freeze_sha256": "sha256:" + "a" * 64,
        }
    }
    command = worker.build_command(job)
    joined = " ".join(command)
    assert "run_step14_oracle_raw_evidence_ceiling.py" in joined
    assert "--training-run-id training_18e0b4ed6e685af30b6c4a35" in joined
    assert "--selection-freeze-sha256 sha256:" in joined
    assert "outer" not in joined
    assert "qpu" not in joined.lower()


def test_terminal_worker_status_survives_container_restart(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    log_path = tmp_path / "worker.log"
    terminal = {
        "state": "failed",
        "returncode": 1,
        "log_path": str(log_path),
        "completed_at": "2026-09-02T08:35:41+00:00",
    }
    status_path.write_text(json.dumps(terminal), encoding="utf-8")
    log_path.write_text("original traceback\n", encoding="utf-8")

    recovered = worker._existing_terminal_status(status_path)

    assert recovered == terminal
    assert worker._terminal_returncode(recovered) == 1
    assert log_path.read_text(encoding="utf-8") == "original traceback\n"


def test_hardened_reconciler_deletes_only_already_exited_stale_pod(monkeypatch) -> None:
    record = {
        "job_id": "science",
        "control_run_id": "run-test",
        "pod_id": "pod-test",
        "status_key": "triqto-control/runs/science/run-test/status.json",
    }
    monkeypatch.setattr(hardened.control, "list_active_records", lambda: [("triqto-control/active/run-test.json", record)])
    monkeypatch.setattr(hardened, "read_status", lambda _key: {"state": "running"})
    monkeypatch.setattr(hardened, "pod_desired_status", lambda _pod: "EXITED")
    deleted: list[str] = []
    archived: list[dict] = []
    monkeypatch.setattr(hardened.control, "delete_pod", lambda pod, best_effort=True: deleted.append(pod) or True)
    monkeypatch.setattr(hardened.control, "archive_and_remove", lambda _key, _record, terminal: archived.append(terminal))

    hardened.reconcile_detached()

    assert deleted == ["pod-test"]
    assert archived[0]["state"] == "pod_terminal_with_stale_worker_status"
    assert archived[0]["worker_state"] == "running"
    assert archived[0]["pod_desired_status"] == "EXITED"


def test_hardened_reconciler_never_age_kills_running_pod(monkeypatch) -> None:
    record = {
        "job_id": "science",
        "control_run_id": "run-test",
        "pod_id": "pod-test",
        "status_key": "triqto-control/runs/science/run-test/status.json",
        "launched_at": "2000-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(hardened.control, "list_active_records", lambda: [("triqto-control/active/run-test.json", record)])
    monkeypatch.setattr(hardened, "read_status", lambda _key: {"state": "running"})
    monkeypatch.setattr(hardened, "pod_desired_status", lambda _pod: "RUNNING")
    monkeypatch.setattr(hardened.control, "delete_pod", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not delete RUNNING Pod")))
    monkeypatch.setattr(hardened.control, "archive_and_remove", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not archive RUNNING Pod")))

    hardened.reconcile_detached()
