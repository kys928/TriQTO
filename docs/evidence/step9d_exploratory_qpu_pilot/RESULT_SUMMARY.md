# Step 9D v2 — completed exploratory IBM-QPU pilot

Status: **COMPLETE — EXPLORATORY ONLY, NOT CONFIRMATORY**

This directory freezes the first completed physical IBM-QPU transfer pilot for TriQTO Step 9D v2. The run used the frozen Step-9A deployment ensemble and the frozen Step-9D v2 protocol. No retraining, checkpoint selection, architecture change, threshold change, mitigation adaptation, or post-hoc tuning occurred before or during execution.

## Execution identity

- Plan ID: `qpuplan_0a679d301262372d3db8c3b5`
- IBM Runtime job ID: `da2oii43jnrc73ae6m6g`
- Backend: `ibm_marrakesh` version `1.0.21`
- Processor: `Heron r2`
- Physical chain: `[13, 14, 15]`
- Two-qubit gate: `cz`
- Calibration timestamp used for planning: `2026-08-19T10:05:38+00:00`
- Cases: 12
- Programs: 72
- Shots/program: 4096
- Total executions: 294912
- IBM-reported quantum seconds: 80
- Runtime stack: `qiskit_ibm_runtime-0.40.1,qiskit-2.1.2*,qiskit_aer-0.17.1*`
- Instance: `open-instance`, plan `open`
- IBM instance CRN is intentionally not committed to the public repository; its SHA-256 is `34ac4d57ceac894bb2555767b9860326064a9c72479cb6d9eed72fee5d7888a2`.

## Descriptive results

- Clean effect false positives: **0/3**
- Distorted effect detections: **7/9 (77.8%)**
- Distorted mechanism correct: **4/9 (44.4%)**
- Confirmatory interpretation allowed: **NO**

### By circuit family

| Family | Effect detection | Mechanism correct |
|---|---:|---:|
| bell_like | 1/3 | 2/3 |
| ghz | 3/3 | 2/3 |
| phase_interference | 3/3 | 0/3 |

## Integrity

The uploaded completed-result ZIP had SHA-256:

`2a07be8befbed43691a96a48492565b4d38a316d7a12f6725a85609656ea63b2`

All file hashes recorded in `pilot_complete.json` were independently recomputed and matched. The original raw `pilot_plan.json`, `backend_snapshot.json`, and `submitted_circuits.qpy` are not committed here because the raw plan/snapshot contain the IBM service CRN. Their exact SHA-256 identities are retained in `ARTIFACT_HASHES.sha256`, so the original bundle remains independently verifiable. A sanitized backend snapshot is included instead.

## Scientific interpretation boundary

This run supports only an exploratory transfer statement: a frozen simulator-developed diagnostic pipeline produced nontrivial real-hardware behavior on one IBM backend, one three-qubit chain, three circuit families, and one controlled-distortion strength. It does **not** establish confirmatory real-QPU robustness, universal mechanism identification, hardware generalization, quantum advantage, or a validated correction policy.

Post-hoc analysis of failure modes is permitted only after this archive is frozen and must not be retroactively described as part of the pre-registered exploratory run.
