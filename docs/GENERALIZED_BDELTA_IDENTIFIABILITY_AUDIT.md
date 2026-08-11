# Step 3.5 — generalized B-delta identifiability audit

## Purpose

Step 3 established an `IDENTIFIABLE` development result for exact clean-relative Z/X/Y `B_delta` evidence, but the matched cohort had two important shortcuts:

1. every audited perturbation affected qubit `q0`;
2. every audited perturbation was the final unitary operation before terminal measurement.

Those conditions are too narrow to approve a TriQTO diagnostic training dataset. Step 3.5 asks the next question before any model training:

> Does `B_delta` remain identifiable when the affected qubit and insertion depth are generalized?

This is still a simulator development audit. It is not a hardware experiment and it does not train TriQTO.

## Source circuits

The audit reuses only the 280 train/validation source examples from the Phase 15.6 identifiability pilot. It never accesses the historical v0.1 test or the spent Phase 15.6 confirmatory cohort.

Before generalized generation, all 280 source examples must pass the merged Step 3 v3 exact-replay gate. The original audited RZ/RX/RY distortion is then removed and the resulting clean physical unitary circuit is reconstructed.

Clean circuits are deduplicated by their ordered unitary sequence. Terminal measurement markers and barriers do not count toward physical unitary depth.

## Generalized factorial design

For each deduplicated clean circuit, Step 3.5 varies:

- **affected qubit:** every qubit `q0 ... q(n-1)`;
- **insertion depth:** distinct physical boundaries nearest 25%, 50%, 75%, and 100% of clean unitary depth;
- **strength:** `0.05` and `0.15` radians;
- **mechanism:** matched RZ drift, RX overrotation, and RY overrotation.

Testing every available qubit is intentionally stronger than random affected-qubit sampling: the audit cannot accidentally draw only favorable qubits.

A boundary rank `k` means the injected rotation is applied after `k` clean unitary events and before the remaining suffix. The suffix is then propagated exactly. Thus nonterminal perturbations genuinely pass through downstream circuit evolution.

Short circuits may map multiple target fractions onto the same integer boundary. Those duplicate physical boundaries are generated once only; they are never replicated merely to fill four nominal depth categories.

Depth bins are descriptive:

- `early`: nonterminal depth fraction <= 0.34;
- `middle`: nonterminal depth fraction <= 0.67;
- `late`: remaining nonterminal positions;
- `terminal`: final unitary boundary.

## Matched comparison

For one generalized context, the following are fixed:

- underlying clean circuit;
- affected qubit;
- exact insertion boundary;
- strength.

Only the mechanism changes:

```text
clean circuit C
      |
      +-- q_i, boundary k, strength s -- RZ
      +-- q_i, boundary k, strength s -- RX
      +-- q_i, boundary k, strength s -- RY
```

This produces three mechanism pairs per context:

- RZ vs RX;
- RZ vs RY;
- RX vs RY.

## Evidence

The evidence contract is unchanged from Step 3:

- clean-relative full-register Z probability change;
- clean-relative full-register X probability change;
- clean-relative full-register Y probability change;
- clean-relative local Z/X/Y Pauli-expectation change.

Exact statevectors are privileged audit-only quantities used to generate counterfactuals and derive phenomenology. They are not part of `B_delta`.

## Frozen separation thresholds

The Step 3 thresholds are retained:

- minimum raw pair separation: `1e-6`;
- minimum relative separation: `0.25`;
- numerical collision threshold: `1e-10`;
- `2,000` grouped bootstrap repeats;
- bootstrap seed `20260811`.

Possible base decisions remain:

- `IDENTIFIABLE`;
- `CONTEXT_DEPENDENT`;
- `NON_IDENTIFIABLE_REGIMES`;
- `INSUFFICIENT_EFFECT`.

## Shortcut-removal gate

Because this study exists specifically to remove two shortcuts, a base `IDENTIFIABLE` result is not sufficient by itself.

The following subsets are frozen before outcome inspection:

- all **nonterminal** pairs;
- all **non-q0** pairs;
- all pairs that are both **nonterminal and non-q0**.

Each eligible subset must contain at least 100 mechanism pairs and must have strong-pair fraction >= `0.90`.

If the ordinary Step 3 decision would be `IDENTIFIABLE` but any of these shortcut-removal subsets fails, the final Step 3.5 decision is downgraded to `CONTEXT_DEPENDENT`.

## Independence and bootstrap grouping

The factorial expansion can produce many derived contexts from one clean circuit. Those derivatives are not treated as independent samples.

For bootstrap uncertainty, all qubit/depth/strength derivatives from the same underlying clean circuit share the same `context_id`. Resampling therefore occurs at the clean-circuit level.

## Stratification

The audit reports separation by:

- pair type;
- circuit family;
- qubit count;
- strength;
- phase-sensitive-family flag;
- exact affected qubit;
- qubit position class (`q0`, interior, final qubit);
- insertion-depth bin;
- terminal vs nonterminal insertion;
- pair type x depth.

These strata are intended to reveal where identifiability breaks rather than hiding a failure behind a strong aggregate average.

## Relationship to the training roadmap

A strong Step 3.5 result does **not** itself become the final TriQTO training dataset. It validates the data-generation recipe.

The intended sequence remains:

```text
Step 3      matched terminal-q0 identifiability        COMPLETE
    |
Step 3.5    generalized qubit/depth identifiability    CURRENT
    |
Step 4      hardware-feasibility contract audit
    |
Step 5      matched training dataset: 500 -> 1k -> 2k -> 5k
    |
Step 6      cheap baselines
    |
Step 7      Graph + Diagnostic B_delta TriQTO training
    |
Step 8      ablations
```

If Step 3.5 exposes non-identifiable regimes, those regimes should inform uncertainty/abstention targets and the Step 5 sampling design rather than being hidden by relabeling.

## Run

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_generalized_bdelta_identifiability_audit.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_generalized_bdelta_identifiability.py
```

Outputs are written under:

`/workspace/triqto-data/step3_5_generalized_bdelta_identifiability/audit_*`

A successful audit contains:

- `source_replay_preflight.csv`;
- `source_replay_preflight.json`;
- `clean_circuit_manifest.csv`;
- `generalized_contexts.csv`;
- `counterfactual_metrics.csv`;
- `pairwise_metrics.csv`;
- `stratified_metrics.csv`;
- `decision.json`;
- `audit_complete.json`.
