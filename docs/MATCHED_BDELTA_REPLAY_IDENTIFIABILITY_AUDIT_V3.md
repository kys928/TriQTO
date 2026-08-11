# Matched B-delta replay audit v3: measurement-tail semantics

## Status

This is a narrow execution correction to Step 3. It does not change the scientific thresholds, evidence definition, bootstrap policy, or possible decisions.

Step 3 v1 refused all 280 source examples because the distortion gate was not the final **graph event**. Step 3 v2 then showed why: all 280 examples failed before numerical replay with `unsupported replay gate 'measure'`.

The observed index pattern is consistent with one terminal measurement event per qubit after the distortion. Examples include:

- 8 qubits: distortion index 46 in 55 graph events, leaving 8 events;
- 4 qubits: distortion index 16 in 21 graph events, leaving 4 events;
- 3 qubits: distortion index 3 in 7 graph events, leaving 3 events;
- 2 qubits: distortion index 14 in 17 graph events, leaving 2 events.

Therefore `terminal` must be interpreted physically as terminal with respect to **unitary evolution**, not literally the final graph event.

## Correction

v3 treats `measure` as a nonunitary readout marker for the archived pre-measurement statevector replay.

The runner:

- requires every measurement event to reference exactly one qubit and have no unitary parameters;
- requires that once measurement starts, no later unitary gate occurs;
- excludes measurement markers from `Statevector` propagation;
- excludes measurement markers and barriers from intervention-depth stratification;
- preserves the exact distortion index, affected qubit, rotation angle, unitary prefix, and unitary suffix;
- retains the mandatory all-280 replay-validation gate.

If any unitary gate follows the measurement tail, or if archived clean/distorted statevectors do not replay within the frozen tolerances, the audit still refuses.

## Scientific correction to earlier interpretation

The v1 statement "all distortions are nonterminal" was correct only in the graph-event sense. It must not be interpreted as evidence that later unitary gates propagated the distortion.

Until v3 replay reports otherwise, the evidence instead suggests the distortion may be the final unitary operation followed only by measurement markers. Consequently, the Step 2 result that RZ drift was phase-dominant in all resolved cases must not be strengthened by claiming that those RZ distortions survived a later unitary suffix. That earlier interpretation is withdrawn.

If v3 confirms that the distortion is terminal with respect to unitary evolution for all source examples, then the original final-clean-state counterfactual equivalence is physically valid for this pilot, even though the graph contains later measurement events.

## Frozen science retained

The following remain unchanged from Step 3 v1/v2:

- development-only status;
- no classifier training;
- no architecture modification;
- no historical v0.1 test access;
- no spent confirmatory cohort access;
- exact Z/X/Y `B_delta` evidence geometry;
- phenomenology decomposition;
- pairwise RZ-vs-RX, RZ-vs-RY, RX-vs-RY comparisons;
- minimum raw separation `1e-6`;
- minimum relative separation `0.25`;
- 2,000 context-bootstrap repeats with seed `20260811`;
- `IDENTIFIABLE`, `CONTEXT_DEPENDENT`, `NON_IDENTIFIABLE_REGIMES`, and `INSUFFICIENT_EFFECT` decision policy.

## Run

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_matched_bdelta_replay_identifiability_v3.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_matched_bdelta_replay_identifiability_v3.py
```

The v3 config hash changes the audit identity, so the previous refused v2 output remains immutable and a new `audit_*` directory is produced.
