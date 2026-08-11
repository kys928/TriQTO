# Step 4 — B_delta hardware-feasibility contract

## Purpose

Step 3.5 established that exact clean-relative Z/X/Y `B_delta` remains highly mechanism-separable after removing the fixed-q0 and terminal-only shortcuts. Step 4 asks a different question before Step 5 training data are generated:

> Which parts of `B_delta` are legitimate hardware-facing inputs, what reference semantics are required, and what bounded measurement summaries retain enough mechanism information without relying on exponential full-register distributions?

This stage performs no QPU execution and no classifier training.

## Hardware acquisition contract

For one observed/current circuit and one paired reference circuit:

- Z evidence uses ordinary computational-basis measurement;
- X evidence uses H before measurement;
- Y evidence uses Sdg then H before measurement.

The paired acquisition therefore requires six basis-program variants per logical observed/reference pair before shot replication:

- observed Z, X, Y;
- reference Z, X, Y.

The same sampled bitstrings may be reused to derive local expectations, same-basis pairwise correlations, global parity, sparse histogram summaries, or full empirical histograms.

Exact probabilities are never treated as hardware observations; hardware evidence is finite-shot empirical evidence.

## Reference contract

Every relational feature must carry explicit `reference_kind` metadata. The primary Step 5 reference is `paired_hardware_compatible_reference`:

- execute intended reference and observed/current circuits on the same backend;
- use the same physical layout and diagnostic basis policy;
- keep the executions within a bounded calibration/time relationship;
- record backend identity, layout identity, time/window, basis code, and shot counts.

This does **not** claim the reference execution is noise-free. It only makes the relational meaning explicit and hardware-realizable.

Simulator ideal references remain simulation ablations only. Stale fixed historical references are disallowed as the primary Step 5 contract.

## Privileged versus deployable quantities

Hardware-facing inputs may include only quantities derived from finite-shot Z/X/Y observations plus explicit metadata. Simulator statevectors, population/phase decomposition, and privileged effect/negligible labels may supervise or audit Step 5 but may not enter deployable model inputs.

Signed relational diagnostic evidence remains a future `DiagnosticTensorBatch -> DiagnosticEncoder` surface. It must not be forced into the existing Born-probability tensor contract.

## Step 4 v1 — one-local core

The first frozen scalable candidate used only signed local X/Y/Z expectation deltas, width O(n).

Audit `audit_066fbee3308c7e13c3308f17` returned:

`DEPLOYABLE_CONTRACT_CORE_IDENTIFIABILITY_UNPROVEN`

The acquisition/reference contract passed, but the one-local core failed a severe GHZ stratum:

- effectful pair strong fraction: `0.959890`;
- effectful nonterminal+non-q0 strong fraction: `0.964672`;
- GHZ strong fraction: `0.082873` across 362 effectful pairs;
- GHZ numerical-collision fraction: `0.917127`.

The corresponding Step 3.5 full basis distributions separated the same GHZ mechanisms strongly, localizing the missing information to joint bitstring correlations rather than invalidating `B_delta` acquisition.

## Step 4.1 — bounded same-shot correlation recovery

Before seeing the Step 4.1 outcome, the following feature ladder and pass gates were frozen:

1. local one-body X/Y/Z deltas;
2. local + three global basis-parity deltas;
3. local + all same-basis two-body correlation deltas;
4. local + pairwise + global parity.

All additions use the same Z/X/Y bitstrings and therefore add zero basis programs.

A variant passes only if:

- overall effectful strong fraction >= 0.90;
- effectful nonterminal+non-q0 strong fraction >= 0.90;
- every eligible >=100-pair major stratum >= 0.80;
- GHZ strong fraction >= 0.90.

Audit `audit_26e54b3ac6fbf8c196b2cd3d` returned:

`CORRELATION_CORE_RECOVERED`

The smallest passing variant is:

`local_plus_pairwise_plus_global_parity`

Observed ladder:

| Variant | Overall effectful strong | Nonterminal + non-q0 | GHZ | Pass |
|---|---:|---:|---:|---|
| local only | 0.959890 | 0.964672 | 0.082873 | no |
| local + global parity | 0.986755 | 0.984499 | 0.895028 | no |
| local + same-basis pairwise | 0.991378 | 0.991468 | 0.685083 | no |
| local + pairwise + global parity | 0.996439 | 0.994833 | 0.961326 | yes |

The selected variant's overall 95% clean-circuit bootstrap interval is `[0.993036, 0.998794]`.

The result demonstrates complementarity: global parity nearly recovers GHZ by itself but stops just below the frozen 0.90 gate; same-basis pairwise correlations recover a different part of the signal; using both clears all frozen gates.

## Frozen Step 5 diagnostic evidence contract

The default Step 5 hardware-facing `B_delta` diagnostic representation should contain, for each basis X/Y/Z:

- signed local expectation deltas;
- signed same-basis pairwise correlation deltas;
- signed global parity delta;
- masks for valid local and pairwise entries under variable qubit count;
- explicit basis identity;
- observed and reference shot counts;
- reference kind and reference-availability mask;
- backend/layout/time-window metadata required by the paired-reference contract.

Dense full-register probability deltas remain optional small-n audit/ablation evidence rather than mandatory training input.

The representation scales as O(n^2), not O(2^n), and requires zero new basis programs beyond the existing paired Z/X/Y acquisition.

## Step 5 supervision contract carried forward from Step 3.5

Step 5 should separately preserve:

- clean/no-distortion controls;
- observable-effect / negligible supervision from privileged simulator ground truth;
- mechanism targets with mechanism loss masked when the intervention is privileged-ground-truth negligible;
- phenomenology targets distinct from mechanism identity;
- grouping by independent clean circuit so factorial derivatives never cross leakage-sensitive splits.

Step 5 must increase independent clean-circuit diversity rather than treating perturbation derivatives as independent data.

## Scientific boundaries

Step 4 establishes a hardware-valid acquisition/reference contract and a noiseless-simulator correlation-core sufficiency result. It does not establish:

- finite-shot robustness;
- robustness to realistic hardware noise or calibration drift;
- adequacy of any particular shot budget;
- performance on IBM hardware;
- utility of the final learned TriQTO model.

Those remain later empirical gates.
