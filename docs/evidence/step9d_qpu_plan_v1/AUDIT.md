# Step 9D v1 QPU plan audit — unsubmitted and superseded

Plan: `qpuplan_2383b79d848901968a9a8b37`

Status: **AUDITED / NO PHYSICAL QPU SUBMISSION / SUPERSEDED BEFORE EXECUTION**

The plan was generated on 2026-08-18 after successful IBM Runtime authentication. It selected `ibm_marrakesh` backend version `1.0.21`, Heron revision 2, physical chain `[34, 33, 39]`, using CZ edges with calibrated errors `0.0015942045695873375` and `0.001581793368924389`. Readout errors on the selected chain were `0.0042724609375`, `0.0169677734375`, and `0.0128173828125`.

The uploaded plan archive was independently audited:

- ZIP SHA-256: `12d3e2a98441c2dcb81168faea524116597217aa8860e95539ea0e42ec0b73a5`
- plan status: `STEP9D_QPU_PLAN_READY_NOT_SUBMITTED`
- `physical_qpu_submitted`: `false`
- cases: 12
- programs: 72
- shots/program: 4096
- total Sampler executions: 294912
- every case contains exactly reference/observed Z/X/Y programs
- all 72 transpiled routing permutations are identity
- depth range: 8–12
- size range: 12–24
- recorded compiled-program metadata SHA-256: `sha256:5851ea027151595c1b2b96c74e0bac7691e15db8f117517e9229b900e286655c`
- independently recomputed metadata SHA-256: exact match

Quality-first candidate ranking in the plan:

1. `ibm_marrakesh`: worst 2Q error `0.0015942045695873375`, mean 2Q error `0.0015879989692558633`, mean readout error `0.0113525390625`, 13 pending jobs.
2. `ibm_fez`: worst 2Q error `0.0018920877359271948`, mean 2Q error `0.00182637044450909`, mean readout error `0.013916015625`, 24196 pending jobs.
3. `ibm_kingston`: worst 2Q error `0.00406102581685458`, mean 2Q error `0.0038580519088009108`, mean readout error `0.01904296875`, 7 pending jobs.

## Why v1 was superseded

No physical-QPU access had started, so the protocol could still be safely hardened without spending a hardware attempt.

Two deployment-safety gaps were identified:

1. v1 relied on the caller to use the frozen package versions but did not fail closed if Qiskit/Aer/Runtime versions drifted.
2. the saved IBM account had no explicit instance. With Runtime 0.40.1, planning could see resources across multiple instances and the plan did not freeze the IBM instance CRN/plan. This leaves avoidable ambiguity about paid versus free instance selection.

Step 9D v2 therefore requires exact package versions, permits Open Plan only, freezes the exact instance CRN/name/plan into the plan identity, and lowers the maximum QPU execution-time ceiling from 900 seconds to 300 seconds.

The v1 plan must **not** be executed. It remains useful as an audit record of the initial hardware-quality selection.
