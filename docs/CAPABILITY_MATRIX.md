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

## Measurement and identifiability update

Basis-conditioned ideal simulator evidence now records a first-class measurement context `M` for Z/X/Y per-qubit bases and stores independently normalized `p(y | M)` distributions. Diagnosis/action supervision is masked for unidentifiable targets such as marker-only distortions and computational-basis phase blindness. This is integrated for the ideal simulator/data-generation/graph path only; noisy, fake-backend, and hardware modes remain outside the data lake until separately implemented and tested.


## OOD split and Phase 15 identity update

Deterministic standalone utilities now support audited family, qubit-count, distortion-type, and backend-ID holdout assignments, with IID splits labeled `iid_test`. Baseline comparison identity includes task, view, ablation, and execution/evidence mode to avoid artifact collisions. These utilities are executable and tested, but no empirical OOD result is claimed until a full Phase 15 run consumes produced artifacts.


## Metric and global-phase update

Pure-state fidelity/Fubini–Study, density-matrix fidelity, trace distance, purity, Bures distance, and finite-difference pure-state QGT/QFI are standalone executable utilities with analytical tests. Unsupported or nonphysical domains fail validation. The Hilbert encoder no longer uses largest-amplitude argmax global-phase anchoring.


## Noisy and density simulator update

Standalone seeded Aer noisy-shot and density-matrix helpers are executable for small circuits with explicit noise-model identities and validation. These APIs are not yet integrated into the Phase 7 data lake, Phase 12 views, fake-backend evidence, or hardware ingestion path.


## Uncertainty update

Per-example masked uncertainty losses and direct uncertainty/error diagnostics are executable and tested. They are not evidence of calibrated uncertainty until enabled in a trained checkpoint and evaluated on produced IID/OOD artifacts.


## Backend/transpilation evidence update

A stable local fake-backend fixture and deterministic transpilation evidence API are executable and tested offline. Missing calibration, duration, readout-error, gate-error, and T1/T2 features are represented with availability masks and missing reasons, never fabricated zeros. This remains standalone evidence until Phase 12/15 backend-view integration is completed.


## Hardware Runtime boundary update

A credential-gated IBM Runtime boundary is implemented with typed hardware job/result records, explicit confirmation before submission, backend identity drift checks, shot/count validation, and rejection of simulator-only fields in physical artifacts. Tests use credential-free doubles only; no hardware execution is performed.
