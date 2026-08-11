# Phase/amplitude label-semantics audit

## Status

This is a frozen **development audit** for Step 2 of the post-Phase-15.6 roadmap.

It does not alter the model, retrain a classifier, rewrite labels, or access the historical v0.1 test. Its purpose is to decide whether the coarse names `phase_like` and `amplitude_like` are physically stable descriptions of the observed final-state effect across circuit/state contexts.

The frozen configuration is:

`configs/v0_2/phase_amplitude_label_semantics_audit.json`

The runner is:

`scripts/v0_2/audit_phase_amplitude_label_semantics.py`

## Why this audit exists

The previous probe assigned the binary target from the coarse label. In practice, the development products used labels such as RZ drift under `phase_like` and RX/RY overrotation under `amplitude_like`.

Those mechanism names are physically meaningful, but the coarse phenotype names need not be universally valid. A gate-axis perturbation can propagate through later circuit operations, and its final effect depends on the state on which it acts and on the remainder of the circuit. The audit therefore separates:

1. **injected mechanism** — what perturbation was applied;
2. **observed phenomenology** — what happened to amplitude magnitudes/populations and relative-phase/interference structure in the final state.

No mechanism label is changed by this audit.

## Primary physical decomposition

For normalized clean and distorted pure states `psi` and `phi`, let

- `p_i = |psi_i|^2`;
- `q_i = |phi_i|^2`;
- `BC = sum_i sqrt(p_i q_i)`;
- `Q = |<psi|phi>|`.

Then exactly,

`1 - Q = (1 - BC) + (BC - Q)`.

Both components are non-negative (up to floating-point tolerance):

- `population_component = 1 - BC` measures disagreement in computational-basis amplitude magnitudes/populations;
- `phase_component = BC - Q` measures the additional overlap loss caused by relative-phase/interference disagreement after magnitude agreement is accounted for.

The decomposition is invariant to global phase.

The continuous primary diagnostic is

`dominance_log_ratio = ln((phase_component + epsilon) / (population_component + epsilon))`.

The frozen semantic expectation is:

- `phase_like`: positive log-ratio;
- `amplitude_like`: negative log-ratio.

This sign test is primary. A secondary descriptive phenotype uses a pre-frozen 2:1 dominance ratio to call an effect `phase_dominant`, `population_dominant`, `mixed`, or `negligible`.

## Secondary observable diagnostics

For each example the audit independently reconstructs exact final-state measurement probabilities in the X, Y, and Z bases and reports:

- X/Y/Z full-register total-variation distance;
- per-qubit X/Y/Z expectation RMS change;
- per-qubit maximum absolute expectation change;
- state infidelity.

These quantities help explain *how* the final-state effect manifests. They do not define the primary label semantics.

## Frozen decision policy

The runner returns one of four development decisions:

- `PHENOMENOLOGICALLY_STABLE`: both coarse labels have at least 0.90 expected-sign concordance overall; all adequately populated family/qubit/raw-label/strength strata have at least 0.80 concordance; median directions are correct; negligible effects are at most 10%.
- `CONTEXT_DEPENDENT`: both labels retain at least 0.70 expected-sign concordance overall and the overall median directions remain correct, but one or more context strata fail the stable criterion or reverse direction.
- `SEMANTICALLY_UNSTABLE`: at least one coarse label has less than 0.70 overall expected-sign concordance or its median direction is opposite the claimed phenotype.
- `INSUFFICIENT_EFFECT`: more than 10% of at least one coarse-label cohort has negligible total overlap loss, so the phenotype is not consistently physically resolved.

These thresholds were frozen before running the audit and must not be adjusted after seeing the result to obtain a preferred status.

## Source cohort

The first run intentionally targets the existing 280-example phase/amplitude development pilot because those artifacts preserve full clean and distorted statevectors. The runner refuses any manifest split other than `train` or `validation`.

The 160-example confirmatory cohort is now `SPENT_CONFIRMATORY` and may be used for postmortem work, but it must not be silently treated as untouched evidence. If its full exact-state provenance is later included in this audit, that inclusion must be explicitly documented as previously observed confirmatory data. No regeneration should be silently substituted for the original holdout artifact.

## Observed Step 2 result

Audit `audit_f84fb9da972f2c6e071bf40c` completed with frozen decision `CONTEXT_DEPENDENT`.

- `phase_like`: expected-sign concordance `1.0000`, 95% group-bootstrap CI `[1.0000, 1.0000]`, median dominance log-ratio `+19.5601`, negligible fraction `0.0143`.
- `amplitude_like`: expected-sign concordance `0.8031`, 95% group-bootstrap CI `[0.7338, 0.8672]`, median dominance log-ratio `-3.3807`, negligible fraction `0.0929`.
- six adequately populated context strata failed the frozen stable criterion: RX overrotation, hardware-efficient ansatz, QAOA-like, 6 qubits, 8 qubits, and strength `0.05`.
- the historical v0.1 test was not accessed, no classifier was trained, and no labels were rewritten.

The archived result summary and hash-bound evidence live under:

`docs/evidence/phase_amplitude_label_semantics/audit_f84fb9da972f2c6e071bf40c/`

The development conclusion is that mechanism identity and observed phenomenology must not be conflated. The tested RZ-drift cohort is highly phase-dominant under this generator, whereas RX/RY overrotation is only directionally population-dominant overall and becomes mixed, phase-dominant, or negligible in material circuit/state contexts.

The perfect RZ result is specific to the tested injection protocol until the exact insertion/propagation policy is explicitly checked; it must not be generalized to arbitrary in-circuit RZ errors.

## Run on the pod

After checking out this branch:

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_phase_amplitude_label_semantics.py
```

The default source pointer is:

`/workspace/triqto-data/phase15_6_pilot_v2/data/v0_2_phase_amplitude_identifiability_pilot/current_product.json`

The audit writes a separate content-addressed directory under:

`/workspace/triqto-data/phase15_6_label_semantics_audit/`

It never writes into the source product.

Expected outputs are:

- `example_metrics.csv` — one row per source example;
- `stratified_metrics.csv` — overall and context-stratified semantic statistics;
- `decision.json` — frozen decision plus group-bootstrap intervals;
- `audit_complete.json` — hashes, source identity, and completion boundary.

## Interpretation discipline

A result that the coarse phenotype labels are unstable does **not** imply that the injected mechanisms were invalid. It means the project should stop conflating mechanism with final-state effect.

Possible next targets after this audit are therefore:

- retain mechanism labels such as RZ drift / RX overrotation / RY overrotation;
- use continuous population-versus-phase effect coordinates;
- use a mixed/uncertain phenomenology target;
- keep coarse labels only if the audit shows they are genuinely stable.

No architecture work should be selected on the basis of this audit until the result is inspected and Step 2 is closed.
