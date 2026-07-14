from __future__ import annotations

import pytest

from triqto.hardware import HardwareJobSpec, RuntimeSubmissionError, collect_hardware_result, require_runtime_environment, submit_hardware_job


class FakeClient:
    def __init__(self) -> None:
        self.submitted = False

    def submit(self, spec: HardwareJobSpec) -> str:
        self.submitted = True
        return "job-123"

    def result(self, job_id: str):
        return {"backend_id": "backend_1", "backend_name": "ibm_fake_physical", "counts": {"0": 3, "1": 1}, "shots": 4}


def spec(**overrides) -> HardwareJobSpec:
    values = dict(backend_name="ibm_fake_physical", circuit_id="circuit_1", backend_id="backend_1", shots=4, measurement_bases=("Z",), confirmation_token=None)
    values.update(overrides)
    return HardwareJobSpec(**values)


def test_credentials_required_without_exposing_token() -> None:
    with pytest.raises(RuntimeSubmissionError, match="credentials"):
        require_runtime_environment({})
    assert require_runtime_environment({"QISKIT_IBM_TOKEN": "not-printed"}) == "configured"


def test_submission_requires_confirmation_before_client_call() -> None:
    client = FakeClient()
    with pytest.raises(RuntimeSubmissionError, match="confirmation"):
        submit_hardware_job(spec(), client, confirm=False, env={"QISKIT_IBM_TOKEN": "x"})
    assert client.submitted is False


def test_confirmed_submission_and_collection_with_backend_binding() -> None:
    client = FakeClient()
    job_id = submit_hardware_job(spec(confirmation_token="SUBMIT_PHYSICAL_HARDWARE"), client, confirm=True, env={"QISKIT_IBM_TOKEN": "x"})
    assert job_id == "job-123"
    record = collect_hardware_result(spec(), client, job_id)
    assert record.shots_realized == 4
    assert record.metadata["hardware_mode_hilbert_masked"] is True


def test_physical_specs_reject_simulator_only_fields() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        spec(metadata={"statevector": [1, 0]})
