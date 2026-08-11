# Matched B-delta exact-replay identifiability audit

## Status

This is **Step 3 v2**, a frozen development audit.

Step 3 v1 (`triqto.v0_2.matched_bdelta_identifiability_audit.v1`) correctly returned `REFUSED_INSERTION_POLICY` before producing a scientific identifiability result: all 280 source examples used axis-matched but nonterminal RX/RY/RZ distortion gates. That refusal remains valid and is not overwritten.

v2 keeps the scientific question and replaces only the invalid state-only counterfactual shortcut with exact ordered circuit replay at the archived intervention index.

## Why replay is required

For a nonterminal error, applying RX/RY/RZ to the final clean state is generally not equivalent to inserting that error into the circuit and propagating through the remaining gates. Later gates can rotate, interfere with, entangle, or otherwise transform the perturbation. A phase-axis perturbation can therefore manifest as population change at the final state, and vice versa.

The matched counterfactual must hold fixed:

- the clean circuit sequence;
- the exact intervention index;
- the affected qubit;
- the archived rotation angle;
- every gate before and after the intervention;

while changing only the replacement mechanism among RZ, RX and RY.

## Archived replay representation

The 280-example pilot artifacts retain the ordered graph gate event representation used by the previous frozen probe:

- `a__x_graph_gate_names`;
- `a__x_graph_gate_qubit_ptr` and `a__x_graph_gate_qubit_indices`;
- `a__x_graph_gate_parameter_ptr`;
- `a__x_graph_gate_parameter_sin` and `a__x_graph_gate_parameter_cos`;
- `audit__removed_distortion_gate_indices`.

Gate parameters are reconstructed with `atan2(sin, cos)`. This representation is not trusted merely because it exists.

## Mandatory replay-validation gate

Before any matched counterfactual is admitted, every source example must pass both checks:

1. replay the entire archived sequence and reproduce the archived distorted statevector;
2. remove the single audited distortion gate, replay the remaining ordered sequence, and reproduce the archived clean statevector.

Validation is invariant to global phase. Frozen limits are:

- overlap loss `1 - |<stored|replayed>| <= 1e-9`;
- maximum amplitude error after global-phase alignment `<= 1e-8`.

The audit requires **all 280 source examples** to pass. Unsupported gates, malformed pointer arrays, axis/qubit disagreement, or state mismatch cause `REFUSED_REPLAY_VALIDATION`. The runner must not discard failing examples and continue with a convenient subset.

## Matched evidence

After replay validation, each deduplicated physical context produces three exact simulator counterfactuals:

- RZ drift;
- RX overrotation;
- RY overrotation.

The context identity binds the clean circuit sequence, intervention index, affected qubit, exact archived angle and qubit count.

`B_delta_matched_ZXY_exact_replay` contains only clean-relative hardware-facing observable quantities:

- full-register Z/X/Y probability changes;
- local Z/X/Y Pauli expectation changes.

Privileged statevectors are used only for replay validation and phenomenology ground truth. They are not part of B_delta evidence.

## Mechanism and phenomenology remain separate

Mechanism comparisons are:

- RZ vs RX;
- RZ vs RY;
- RX vs RY.

The privileged phenomenology audit separately computes:

- population/amplitude-magnitude component;
- relative-phase/interference component;
- dominance log ratio;
- total overlap loss;
- descriptive phenotype: phase-dominant, mixed, population-dominant or negligible.

Thus the study can distinguish “mechanisms are observationally separable” from “their final-state phenomenology differs.”

## Frozen decision policy

The pairwise separation score and thresholds are inherited unchanged from Step 3 v1. The result is one of:

- `IDENTIFIABLE`;
- `CONTEXT_DEPENDENT`;
- `NON_IDENTIFIABLE_REGIMES`;
- `INSUFFICIENT_EFFECT`.

The study is additionally stratified by intervention depth quartile so we can see whether identifiability degrades as a disturbance propagates through a longer suffix.

## Scientific boundaries

- development evidence only;
- no classifier is trained;
- no model architecture is changed;
- historical v0.1 test is not accessed;
- spent confirmatory cohort is not accessed;
- v1 refusal remains part of the audit record;
- exact noiseless simulator identifiability does not establish finite-shot or hardware deployability; that remains Step 4.

## Run

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_matched_bdelta_replay_identifiability_audit.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_matched_bdelta_replay_identifiability.py
```

Outputs are written under:

`/workspace/triqto-data/phase15_6_matched_bdelta_replay_identifiability/audit_*`

A successful run contains:

- `replay_preflight.csv`;
- `replay_preflight.json`;
- `counterfactual_metrics.csv`;
- `pairwise_metrics.csv`;
- `stratified_metrics.csv`;
- `decision.json`;
- `audit_complete.json`.
