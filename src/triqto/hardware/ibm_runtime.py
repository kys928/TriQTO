"""Credential-gated IBM Runtime adapter boundary."""
from __future__ import annotations

import os
from typing import Any, Protocol

from .hardware_schema import HardwareJobSpec, HardwareResultRecord


class RuntimeSubmissionError(RuntimeError):
    pass


class RuntimeClient(Protocol):
    def submit(self, spec: HardwareJobSpec) -> str: ...
    def result(self, job_id: str) -> dict[str, Any]: ...


def require_runtime_environment(env: dict[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    token = values.get("QISKIT_IBM_TOKEN") or values.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise RuntimeSubmissionError("IBM Runtime credentials are not configured")
    return "configured"


def submit_hardware_job(spec: HardwareJobSpec, client: RuntimeClient, *, confirm: bool = False, env: dict[str, str] | None = None) -> str:
    if spec.confirmation_token != "SUBMIT_PHYSICAL_HARDWARE" or not confirm:
        raise RuntimeSubmissionError("physical hardware submission requires explicit confirmation")
    require_runtime_environment(env)
    try:
        return client.submit(spec)
    except Exception as exc:  # pragma: no cover - client dependent
        raise RuntimeSubmissionError("hardware job submission failed") from exc


def collect_hardware_result(spec: HardwareJobSpec, client: RuntimeClient, job_id: str) -> HardwareResultRecord:
    try:
        raw = client.result(job_id)
    except Exception as exc:  # pragma: no cover - client dependent
        raise RuntimeSubmissionError("hardware result collection failed") from exc
    if raw.get("backend_id") != spec.backend_id or raw.get("backend_name") != spec.backend_name:
        raise RuntimeSubmissionError("backend identity drift detected")
    counts = {str(k): int(v) for k, v in dict(raw.get("counts", {})).items()}
    return HardwareResultRecord(
        job_spec_id=spec.job_spec_id,
        backend_id=spec.backend_id,
        backend_name=spec.backend_name,
        job_id=job_id,
        shots_requested=spec.shots,
        shots_realized=int(raw.get("shots", sum(counts.values()))),
        counts=counts,
        metadata={"hardware_mode_hilbert_masked": True, "schema_source": "runtime_adapter"},
    )


def describe_contract() -> str:
    return "IBM Runtime adapter is credential-gated, confirmation-gated, and tested only with doubles."


__all__ = ["RuntimeClient", "RuntimeSubmissionError", "collect_hardware_result", "require_runtime_environment", "submit_hardware_job"]
