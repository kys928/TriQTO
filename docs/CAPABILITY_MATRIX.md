# TriQTO capability matrix

TriQTO currently provides an offline, deterministic research scaffold. It must not be described as hardware-validated, OOD-generalizing, uncertainty-calibrated, or topology-validated.

| Capability | Status | Notes |
| --- | --- | --- |
| Ideal statevector / Born simulation | Implemented, offline-tested | Simulator-only evidence tier. |
| Sampled ideal shots | Implemented where existing tests cover it | Offline only. |
| Noisy Aer shots / density simulation | Intentionally unsupported in active configs | Future work; configs must be marked unsupported until implemented and tested. |
| Fake-backend / transpilation evidence | Intentionally unsupported in active configs | Future work; no fabricated backend features. |
| IBM Runtime ingestion | Credential-gated placeholder only | Not run by default; no credentials in tests. |
| Topology loss | Diagnostic/audit only | `topology_loss_weight` remains `0.0` by default. |
| Phase 15 OOD/generalization claims | Unsupported | IID results must not be labeled as OOD generalization. |
| Per-example calibrated uncertainty | Unsupported | Existing uncertainty outputs are not validated calibration evidence. |

## Dependency matrix

Supported default environment: Python 3.11 on CPU with dependencies pinned by `requirements-cpu.txt` and `constraints/cpu.txt`.

Install default CPU profile:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints/cpu.txt
python -m pip install -e .
```

Optional GPU profile is isolated in `requirements-gpu.txt` and `constraints/gpu.txt`; it is never used by default CI.
