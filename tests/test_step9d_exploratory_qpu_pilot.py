from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qiskit.providers.fake_provider import GenericBackendV2

from triqto.hardware.qpu_pilot import (
    best_connected_three_qubit_chain,
    build_pilot_cases,
    compile_pilot_programs,
    descriptive_pilot_metrics,
    select_backend_and_chain,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "v0_2" / "step9d_exploratory_qpu_pilot_v2.json"
V1_CONFIG = ROOT / "configs" / "v0_2" / "step9d_exploratory_qpu_pilot.json"
V2_RUNNER = ROOT / "scripts" / "v0_2" / "run_step9d_exploratory_qpu_pilot_v2.py"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_v2_runner_module():
    spec = importlib.util.spec_from_file_location("step9d_v2_test_module", V2_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyTarget:
    def __init__(self, mapping):
        self.mapping = mapping
        self.operation_names = set(mapping)

    def __getitem__(self, name):
        return self.mapping[name]


class DummyBackend:
    def __init__(self, name: str, edge_error: float, readout_error: float, pending_jobs: int = 0):
        self.name = name
        self.backend_version = "1.0.0"
        self.num_qubits = 5
        self.processor_type = {"family": "test"}
        props = lambda error: SimpleNamespace(error=error)
        self.target = DummyTarget(
            {
                "cz": {
                    (0, 1): props(edge_error),
                    (1, 0): props(edge_error + 0.001),
                    (1, 2): props(edge_error + 0.002),
                    (2, 1): props(edge_error + 0.003),
                    (2, 3): props(edge_error + 0.02),
                },
                "measure": {
                    (0,): props(readout_error),
                    (1,): props(readout_error + 0.001),
                    (2,): props(readout_error + 0.002),
                    (3,): props(readout_error + 0.003),
                },
            }
        )
        self._status = SimpleNamespace(
            operational=True, status_msg="active", pending_jobs=pending_jobs
        )

    def status(self):
        return self._status


class DummyService:
    def __init__(self, backends):
        self._backends = list(backends)

    def backends(self, **kwargs):
        return list(self._backends)

    def backend(self, name):
        for backend in self._backends:
            if backend.name == name:
                return backend
        raise KeyError(name)


def test_step9d_v1_is_superseded_and_v2_is_frozen_open_only() -> None:
    v1 = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    assert v1["status"] == "SUPERSEDED_BEFORE_PHYSICAL_QPU_EXECUTION"
    assert v1["superseded_by"].endswith("step9d_exploratory_qpu_pilot_v2.json")

    cfg = load_config()
    assert cfg["schema"] == "triqto.v0_2.step9d_exploratory_qpu_pilot.v2"
    assert cfg["status"] == "FROZEN_BEFORE_PHYSICAL_QPU_EXECUTION"
    assert cfg["instance_policy"]["allowed_plans"] == ["open"]
    assert cfg["instance_policy"]["paid_plan_execution_allowed"] is False
    assert cfg["software_environment"] == {
        "fail_closed_on_version_drift": True,
        "qiskit": "2.1.2",
        "qiskit_aer": "0.17.1",
        "qiskit_ibm_runtime": "0.40.1",
    }
    assert cfg["scientific_boundaries"]["exploratory_only"] is True
    assert cfg["scientific_boundaries"]["confirmatory_claim"] is False
    execution = cfg["execution"]
    assert execution["case_count"] == 12
    assert execution["programs_per_case"] == 6
    assert execution["total_programs"] == 72
    assert execution["shots_per_program"] == 4096
    assert execution["total_executions"] == 72 * 4096
    assert execution["max_execution_time_seconds"] == 300
    assert execution["explicit_confirmation_token"] == "STEP9D_EXPLORATORY_QPU"


def test_v2_open_instance_selection_rejects_paid_and_ambiguous() -> None:
    runner = load_v2_runner_module()
    rows = [
        {"name": "free-main", "crn": "crn:open", "plan": "open"},
        {"name": "paid", "crn": "crn:paid", "plan": "pay-as-you-go"},
    ]
    assert runner.select_open_instance(rows, None)["crn"] == "crn:open"
    assert runner.select_open_instance(rows, "free-main")["crn"] == "crn:open"
    with pytest.raises(RuntimeError, match="did not identify exactly one Open Plan"):
        runner.select_open_instance(rows, "paid")
    with pytest.raises(RuntimeError, match="requires one explicit Open Plan instance"):
        runner.select_open_instance(rows + [{"name": "free-2", "crn": "crn:open2", "plan": "Open"}], None)


def test_v2_version_guard_matches_frozen_environment() -> None:
    runner = load_v2_runner_module()
    actual = runner.verify_frozen_versions(load_config())
    assert actual == {
        "qiskit": "2.1.2",
        "qiskit_aer": "0.17.1",
        "qiskit_ibm_runtime": "0.40.1",
    }


def test_backend_selection_prefers_calibration_quality_before_queue() -> None:
    better = DummyBackend("ibm_better", edge_error=0.006, readout_error=0.02, pending_jobs=20)
    busier_but_worse = DummyBackend("ibm_worse", edge_error=0.02, readout_error=0.01, pending_jobs=0)
    service = DummyService([busier_but_worse, better])
    backend, candidate, ranking = select_backend_and_chain(service)
    assert backend.name == "ibm_better"
    assert candidate.physical_chain == (0, 1, 2)
    assert candidate.score[0] < best_connected_three_qubit_chain(busier_but_worse).score[0]
    assert ranking[0]["backend_name"] == "ibm_better"


def test_error_equal_one_is_rejected_as_stale() -> None:
    backend = DummyBackend("ibm_stale", edge_error=1.0, readout_error=0.01)
    with pytest.raises(RuntimeError, match="no usable calibrated"):
        best_connected_three_qubit_chain(backend)


def test_pilot_case_matrix_is_three_families_times_clean_plus_three_mechanisms() -> None:
    cfg = load_config()
    cases = build_pilot_cases(cfg, [10, 11, 12])
    assert len(cases) == 12
    families = {case.family for case in cases}
    assert families == {"bell_like", "ghz", "phase_interference"}
    for family in families:
        local = [case for case in cases if case.family == family]
        assert {case.condition for case in local} == {
            "clean",
            "rz_drift",
            "rx_overrotation",
            "ry_overrotation",
        }
        clean = next(case for case in local if case.condition == "clean")
        assert clean.expected_effect is False
        assert clean.expected_mechanism is None
        assert clean.strength is None
        for distorted in [case for case in local if case.condition != "clean"]:
            assert distorted.expected_effect is True
            assert distorted.strength == pytest.approx(0.15)
    assert next(case for case in cases if case.family == "ghz").physical_layout == (10, 11, 12)
    assert next(case for case in cases if case.family == "bell_like").physical_layout == (10, 11)


def test_all_72_programs_compile_on_local_backend_with_no_routing_permutation() -> None:
    cfg = load_config()
    backend = GenericBackendV2(
        num_qubits=5,
        basis_gates=["id", "rz", "sx", "x", "cx"],
        coupling_map=[
            [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]
        ],
        seed=2026081711,
        noise_info=False,
    )
    cases = build_pilot_cases(cfg, [0, 1, 2])
    programs, metadata = compile_pilot_programs(
        cases,
        backend,
        optimization_level=1,
        seed_transpiler=17091,
        require_no_routing_permutation=True,
    )
    assert len(programs) == 72
    assert len(metadata) == 72
    assert all("meas" in {register.name for register in circuit.cregs} for circuit in programs)
    assert all(
        row["routing_permutation"] is None
        or row["routing_permutation"] == list(range(len(row["routing_permutation"])))
        for row in metadata
    )


def test_descriptive_metrics_and_v2_runner_remain_exploratory_and_confirmed() -> None:
    rows = [
        {"expected_effect": False, "expected_mechanism": None, "prediction": {"effect_present": True, "mechanism_prediction": "rz_drift"}},
        {"expected_effect": True, "expected_mechanism": "rz_drift", "prediction": {"effect_present": True, "mechanism_prediction": "rz_drift"}},
        {"expected_effect": True, "expected_mechanism": "rx_overrotation", "prediction": {"effect_present": False, "mechanism_prediction": "ry_overrotation"}},
        {"expected_effect": True, "expected_mechanism": "ry_overrotation", "prediction": {"effect_present": True, "mechanism_prediction": "ry_overrotation"}},
    ]
    metrics = descriptive_pilot_metrics(rows, ["rz_drift", "rx_overrotation", "ry_overrotation"])
    assert metrics["clean_effect_false_positive_count"] == 1
    assert metrics["distorted_effect_detection_count"] == 2
    assert metrics["distorted_mechanism_correct_count"] == 2
    assert metrics["confirmatory_interpretation_allowed"] is False

    source = V2_RUNNER.read_text(encoding="utf-8")
    assert "--execute-physical-qpu" in source
    assert "--confirmation-token" in source
    assert "instance_crn" in source
    assert "paid-plan execution is forbidden" in source
    assert "software drift" in source
