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

The completed v3 replay confirms that the distortion is terminal with respect to unitary evolution for the audited source product and is followed only by terminal measurement markers. Consequently, the original final-clean-state counterfactual equivalence was physically valid for this pilot. The earlier interpretation that the RZ perturbations necessarily propagated through a later unitary suffix is withdrawn.

## Completed audit result

Audit `audit_bde0644f24fd2473e650d661` completed with frozen decision:

`IDENTIFIABLE`

Replay validation passed for all `280 / 280` source examples. Maximum replay errors were at floating-point precision:

- maximum clean overlap loss: `2.220446049250313e-16`;
- maximum clean aligned amplitude error: `3.3393655763633256e-16`;
- maximum distorted overlap loss: `2.220446049250313e-16`;
- maximum distorted aligned amplitude error: `3.3706997905829204e-16`.

The 280 source examples deduplicated to 178 matched physical contexts, producing 534 RZ/RX/RY counterfactuals and 534 mechanism-pair comparisons.

Frozen decision metrics:

- overall strong-pair fraction: `1.0000`, 95% context-bootstrap CI `[1.0000, 1.0000]`;
- different-phenotype strong-pair fraction: `1.0000`, 95% CI `[1.0000, 1.0000]` over 392 different-phenotype pairs;
- numerical-collision fraction: `0.0000`;
- negligible-counterfactual fraction: `0.0187266`, 95% CI `[0.0074906, 0.0318352]`;
- bad eligible strata: `0`;
- severe non-identifiable strata: `0`.

Every pair type had strong-pair fraction `1.0`:

- RX vs RY: 178/178;
- RZ vs RX: 178/178;
- RZ vs RY: 178/178.

The weakest observed pair separation was an RX-vs-RY GHZ 3-qubit context at strength `0.05`: raw pair separation `0.0001041449670861` and relative separation `0.6666666645328881`. This remains far above the frozen minimum raw separation `1e-6` and relative-separation threshold `0.25`.

All tested families, qubit counts (2, 3, 4, 6, 8), strengths (`0.05`, `0.15`), and phase-sensitivity strata had strong-pair fraction `1.0`.

### Phenomenology remains distinct from mechanism

The matched counterfactual phenotype counts reinforce Step 2's separation between injected mechanism and observed effect:

- RZ drift: 174 phase-dominant, 4 negligible, 0 mixed, 0 population-dominant;
- RX overrotation: 92 population-dominant, 47 mixed, 33 phase-dominant, 6 negligible;
- RY overrotation: 111 population-dominant, 39 mixed, 28 phase-dominant, 0 negligible.

Thus the mechanism is strongly identifiable in exact Z/X/Y relational evidence even when RX/RY final-state phenomenology is mixed or phase-dominant.

### Scope limit

All 178 matched contexts fall in the `late_75_100` insertion-depth bin because the audited distortion is the last unitary operation before terminal measurement. Therefore Step 3 establishes exact-simulator identifiability for this **terminal-unitary perturbation regime**. It does not establish identifiability for perturbations inserted earlier in a circuit and propagated through a nontrivial unitary suffix.

It also does not establish finite-shot detectability, noisy-backend robustness, or hardware deployability. Those are explicitly deferred to Step 4.

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

## Evidence archive

The repository archive preserves the exact completion marker, frozen decision, replay preflight, stratified metrics, and a compact result summary under:

`docs/evidence/matched_bdelta_identifiability/audit_bde0644f24fd2473e650d661/`

The large raw counterfactual, pairwise, and replay-preflight CSVs remain hash-bound by `audit_complete.json`.

## Run

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_matched_bdelta_replay_identifiability_v3.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_matched_bdelta_replay_identifiability_v3.py
```

The v3 config hash changes the audit identity, so the previous refused v2 output remains immutable and a new `audit_*` directory is produced.
