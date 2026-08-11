# Matched B_delta identifiability audit

## Status

This is the frozen **Step 3 development audit** after the Phase 15.6 confirmatory closeout and the Step 2 label-semantics result.

Step 2 found that mechanism and phenomenology must be represented separately: RZ drift was strongly phase-dominant under the tested generator, while RX/RY overrotation produced population-dominant, mixed, phase-dominant, and negligible final-state effects depending on context. Step 3 therefore does not ask whether a binary `phase_like` / `amplitude_like` classifier works.

It asks two narrower questions:

1. **Mechanism identifiability:** for a fixed physical context, do hardware-facing relational Z/X/Y observables distinguish matched RZ, RX, and RY perturbations?
2. **Phenomenology identifiability:** when matched mechanisms produce different state-derived phenomenologies, does the same observable evidence remain distinguishable?

No classifier is trained in this audit.

## Why the design is matched

The earlier phase/amplitude probes compared examples produced from different circuits and states. That is useful for generalization testing, but it mixes two effects:

- the distortion mechanism;
- the quantum state/circuit context on which that mechanism acts.

The Step 3 design holds context fixed. For every eligible clean context it generates:

```text
same clean state
same affected qubit
same distortion strength
        |
        +-- RZ drift
        +-- RX overrotation
        +-- RY overrotation
```

The three resulting evidence bundles can then be compared directly.

## Mandatory insertion-policy preflight

The current development product stores the clean final state and identifies the distortion gate that was removed during clean-circuit reconstruction.

The audit first requires, for **every source example**:

- exactly one removed distortion gate;
- the removed gate axis matches the raw mechanism label;
- the removed distortion gate is the final stored gate.

Only if all three conditions hold may the audit generate matched counterfactuals directly from the clean final state.

For a terminal one-qubit rotation,

```text
|psi_distorted> = R_axis(delta) |psi_clean>
```

is exactly the intervention being tested. If even one source example violates the terminal-insertion condition, the audit returns an operational refusal instead of silently treating a mid-circuit rotation as a terminal rotation. A later implementation would then need full circuit replay at the original insertion location.

This preflight also resolves the Step 2 caveat: the perfect RZ phase-dominance result must not be generalized before the actual injection position is verified.

## Matched context identity

For the terminal protocol, the final clean state is sufficient to determine the effect of the appended one-qubit rotation. Contexts are content-addressed from:

- global-phase-invariant clean final-state hash;
- circuit family;
- qubit count;
- affected qubit;
- strength;
- phase-sensitive-family flag.

Duplicate source examples with the same physical context are collapsed before matched counterfactual generation so repeated source rows do not silently overweight the audit.

## B_delta evidence contract

The evidence is deliberately hardware-facing in form. For each matched counterfactual the audit reconstructs exact simulator probabilities and local expectations in the Z, X, and Y bases and compares them with the same clean reference:

- `delta_p_Z`, `delta_p_X`, `delta_p_Y`;
- `delta_<Z_i>`, `delta_<X_i>`, `delta_<Y_i>`.

The privileged statevector is **not** part of the evidence vector. It is used only to:

- generate the matched simulator counterfactual when the terminal-insertion preflight passes;
- define the state-derived phenomenology ground truth.

Step 4 will separately ask whether an equivalent relational evidence contract is actually obtainable with finite shots and on hardware.

## Pairwise mechanism separation

For each context the audit evaluates three mechanism pairs:

- RZ vs RX;
- RZ vs RY;
- RX vs RY.

For each measurement basis it computes the total-variation distance between the two distorted outcome distributions. It also computes the RMS difference between their local Pauli expectations.

The frozen pair separation score is:

```text
mean(
    TV_X,
    TV_Y,
    TV_Z,
    0.5 * RMS_expectation_X,
    0.5 * RMS_expectation_Y,
    0.5 * RMS_expectation_Z,
)
```

The factor `0.5` maps the Pauli-expectation difference scale, whose maximum is 2, onto the same nominal 0-to-1 scale as total variation.

Each mechanism also has a signal score against the common clean reference using the same six blocks. Relative separation is:

```text
2 * pair_separation / (signal_left + signal_right + epsilon)
```

A pair is pre-frozen as **strongly separated** only if:

- raw pair separation is at least `1e-6`; and
- relative separation is at least `0.25`.

A pair is a numerical collision if its separation score is at most `1e-10`.

These thresholds describe exact simulator evidence geometry. They are not finite-shot detection thresholds.

## Phenomenology ground truth

The same global-phase-invariant Step 2 decomposition is retained. For normalized clean and distorted pure states, define

```text
BC = sum_i sqrt(p_i q_i)
Q  = |<psi_clean|psi_distorted>|

1 - Q = (1 - BC) + (BC - Q)
```

where:

- `population_component = 1 - BC`;
- `phase_component = BC - Q`;
- `dominance_log_ratio = log((phase + eps)/(population + eps))`.

The descriptive categories remain:

- `phase_dominant`;
- `mixed`;
- `population_dominant`;
- `negligible`.

For mechanism pairs whose state-derived phenomenology differs, Step 3 reports whether B_delta still strongly separates the pair. This is the direct test of whether the observable evidence can distinguish physically different effects without using the hidden state as an input.

## Frozen decision policy

The audit returns one scientific-development status after a successful preflight:

- `IDENTIFIABLE`
- `CONTEXT_DEPENDENT`
- `NON_IDENTIFIABLE_REGIMES`
- `INSUFFICIENT_EFFECT`

The decision thresholds are stored in:

`configs/v0_2/matched_bdelta_identifiability_audit.json`

In summary:

### IDENTIFIABLE

Requires all of the following:

- at least 0.90 overall strong mechanism-pair separation;
- at least 0.80 in every adequately populated context stratum;
- at least 0.90 strong separation among pairs with different phenomenology;
- at most 0.10 negligible counterfactuals;
- at most 0.01 numerical collisions.

### CONTEXT_DEPENDENT

Requires at least:

- 0.70 overall strong mechanism-pair separation; and
- 0.70 strong separation among different-phenomenology pairs;

while failing the stricter stable criteria.

### NON_IDENTIFIABLE_REGIMES

Triggered if any of the following occurs:

- overall strong pair separation falls below 0.70;
- an adequately populated context stratum falls below 0.50;
- numerical collisions exceed 0.05.

### INSUFFICIENT_EFFECT

Triggered if more than 0.15 of the generated counterfactuals have negligible overlap loss.

All thresholds are frozen before the audit outcome is inspected.

## Stratification

Pairwise results are reported overall and by:

- mechanism pair;
- circuit family;
- qubit count;
- strength;
- phase-sensitive-family flag;
- affected-qubit signature.

Group bootstrap intervals resample matched physical contexts, not individual pair rows, so the three comparisons from one context remain statistically linked.

## Source boundary

The first Step 3 run uses only the existing 280-example train/validation development pilot because it preserves the clean full state required for matched simulator counterfactuals.

The runner refuses any source manifest split other than `train` or `validation`.

It does not access:

- the historical v0.1 test;
- the 160-example spent confirmatory cohort.

Those boundaries are recorded in the completion marker.

## Outputs

A successful run writes a new content-addressed directory under:

`/workspace/triqto-data/phase15_6_matched_bdelta_identifiability/`

containing:

- `source_preflight.csv`;
- `preflight.json`;
- `counterfactual_metrics.csv`;
- `pairwise_metrics.csv`;
- `stratified_metrics.csv`;
- `decision.json`;
- `audit_complete.json`.

If the terminal-insertion preflight fails, the runner archives the refusal and stops before generating the matched counterfactual result.

## Run on the pod

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_matched_bdelta_identifiability.py
```

## Interpretation discipline

A successful exact-state result means only that the selected relational observables contain separable information under matched ideal-simulator conditions.

It does **not** establish:

- finite-shot identifiability;
- robustness to hardware noise and drift;
- availability of the clean reference on real hardware;
- correctness of a learned TriQTO diagnosis head;
- correction efficacy.

Those questions remain downstream. Step 4 is specifically responsible for auditing whether the relational evidence contract is hardware-deployable.
