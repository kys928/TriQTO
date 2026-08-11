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

## Observed Step 2 result

Audit `audit_f84fb9da972f2c6e071bf40c` returned `CONTEXT_DEPENDENT` under the frozen decision policy. The exact archived result is documented under:

`docs/evidence/phase_amplitude_label_semantics/audit_f84fb9da972f2c6e071bf40c/`

The development conclusion is that mechanism identity and observed phenomenology must be represented separately. The RZ-drift cohort was exceptionally phase-dominant in this source product, while RX/RY overrotation was only conditionally population-dominant and depended materially on circuit/state context.

## Step 3 v1 insertion-policy finding

The first matched-B-delta Step 3 runner intentionally required the archived distortion to be terminal before using a final-state rotation shortcut. Its preflight inspected all 280 development examples and refused the scientific audit before matched counterfactual generation:

- `280 / 280` source distortions were nonterminal;
- `140 / 140` RZ-drift examples were nonterminal;
- `70 / 70` RX-overrotation examples were nonterminal;
- `70 / 70` RY-overrotation examples were nonterminal;
- inspected distortion axes matched the raw mechanism labels;
- no classifier was trained;
- historical v0.1 test was not accessed;
- spent confirmatory cohort was not accessed.

This means the final-state shortcut is invalid for this source product. The refusal is a valid audit outcome and must not be weakened by removing the terminality guard. Step 3 v2 therefore requires exact ordered circuit replay at the original intervention index.

An additional implication is that Step 2's 100% expected-sign concordance for the RZ cohort was not produced by a trivial terminal-RZ construction: those RZ perturbations were also mid-circuit and were propagated through later gates. The result remains development- and generator-specific and is not a universal statement about arbitrary RZ errors.

## Interpretation discipline

A result that the coarse phenotype labels are unstable does **not** imply that the injected mechanisms were invalid. It means the project should stop conflating mechanism with final-state effect.

Possible next targets after this audit are therefore:

- retain mechanism labels such as RZ drift / RX overrotation / RY overrotation;
- use continuous population-versus-phase effect coordinates;
- use a mixed/uncertain phenomenology target;
- keep coarse labels only if the audit shows they are genuinely stable.

No architecture work should be selected on the basis of this audit until the result is inspected and Step 2 is closed.
