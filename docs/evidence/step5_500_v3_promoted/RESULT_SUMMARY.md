# Step 5 v3 — accepted 500-root dataset gate

## Decision

`PROMOTION_READY`

The 500-root v3 product passed generation-time scheduling gates and the independent full-artifact EDA over all 6,500 NPZ artifacts. This unlocks the next staged dataset size (1,000 independent clean-circuit roots) but does not make a model-quality or hardware-robustness claim.

## Product identity

- product id: `product_85a26bf36d9ed1769b6a5e23`
- schema: `triqto.v0_2.step5_matched_diagnostic_training_dataset.v3`
- clean roots: 500
- examples: 6,500
- train/validation roots: 400 / 100
- train/validation examples: 5,200 / 1,300
- clean controls: 500
- RZ/RX/RY examples: 2,000 each
- mechanism-supervised examples: 5,357
- injected-but-negligible examples: 643
- selected diagnostic core: `local_plus_pairwise_plus_global_parity`
- primary inputs: empirical finite-shot paired-reference diagnostics
- statevectors persisted in model artifacts: NO
- raw semantic reference-window id persisted as model input: NO
- historical v0.1 test accessed: NO
- spent confirmatory cohort accessed: NO
- classifier trained: NO

Uploaded product ZIP SHA-256:

`sha256:b018c8247a95d29e652ed4762feca1a504cfae3b8abea274ffe7b84ed8717f7a`

## Full-artifact EDA identity

- audit id: `audit_3ac00c9b31d1be6c54d7b852`
- decision: `PROMOTION_READY`
- artifacts verified: 6,500 / 6,500
- artifact issues: 0
- distinct NPZ schemas: 1
- family/split Cramer's V: 0.000000
- depth/strength Cramer's V: 0.012000
- family split total variation: 0.000000
- phenomenology train/validation total variation: 0.015625
- 3q roots: 72 train / 16 validation

Uploaded audit ZIP SHA-256:

`sha256:c60e604772dd60bb689e6530bba717383f2a4dcf94c34b30d5837a89eb730828`

`eda_complete.json` SHA-256:

`sha256:c60a018ad078371fbed5a92cb58ac7789dc8d0234eb40eb1d179eada6cb362e5`

## Scheduling-alias verification

Global acquisition/context associations:

- shot/strength V = 0.060992
- shot/depth V = 0.034176
- shot/family V = 0.000000
- shot/split V = 0.000000
- clean-shot/family V = 0.009784
- clean-shot/split V = 0.037905

Per-qubit-count context associations:

| qubits | shot/affected V | strength/affected V | depth/affected V |
|---:|---:|---:|---:|
| 2 | 0.052058 | 0.024390 | 0.018179 |
| 3 | 0.087697 | 0.195883 | 0.070402 |
| 4 | 0.097717 | 0.048858 | 0.153093 |
| 6 | 0.096958 | 0.096084 | 0.097252 |
| 8 | 0.142347 | 0.201369 | 0.127353 |

All frozen v3 scheduling-alias gates passed.

An additional cross-factor sweep found no new synthetic shortcut of concern: shot/mechanism V=0, shot/effect-present V≈0.011, shot/phenomenology V≈0.017, strength/family V=0, strength/split V=0, strength/mechanism V=0, depth/family V=0, depth/split V=0, affected-qubit/mechanism V=0, split/effect-present V≈0.002, and split/phenomenology V≈0.015.

## Full artifact integrity

Independent inspection of the uploaded product confirmed:

- all manifest hashes matched `dataset_complete.json`;
- `stage_validation.json` matched its recorded SHA-256;
- all 6,500 artifact SHA-256 values matched the example manifest;
- all artifacts loaded with `allow_pickle=False`;
- one common NPZ schema;
- zero privileged/target-derived `x__` keys;
- zero example-id or clean-group-id mismatches;
- all primary diagnostic arrays finite;
- maximum absolute empirical diagnostic deltas: local 0.292969, pairwise 0.265625, global parity 0.261719.

## Mechanism/effect structure

- RX: 1,620 effectful / 380 negligible (81.0% effectful)
- RY: 2,000 effectful / 0 negligible (100.0% effectful)
- RZ: 1,737 effectful / 263 negligible (86.85% effectful)

Phenomenology remains distinct from mechanism:

- RZ: 72.65% phase-dominant, 6.25% population-dominant, 7.95% mixed, 13.15% negligible
- RX: 31.20% phase-dominant, 26.25% population-dominant, 23.55% mixed, 19.00% negligible
- RY: 15.00% phase-dominant, 59.65% population-dominant, 25.35% mixed, 0% negligible

## Finite-shot difficulty (report only)

Finite-shot noise is intentionally not a promotion blocker. Exact simulator diagnostics remain audit-only.

For strength 0.05, median exact core RMS is roughly 0.0056–0.0068 while median shot error falls from 0.0557 at 512 shots to 0.0197 at 4096 shots. The empirical error exceeds the exact signal in 100.0%, 99.6%, 97.9%, and 91.3% of effectful examples at 512, 1024, 2048, and 4096 shots respectively.

For strength 0.15, median exact core RMS is roughly 0.0174–0.0195 while median shot error falls from 0.0550 at 512 shots to 0.0200 at 4096 shots. Noise exceeds exact signal in 89.0%, 79.6%, 70.5%, and 56.6% of effectful examples respectively.

This remains a Step 6 baseline question rather than a reason to simplify the dataset.

## Promotion boundary

The 500-root v3 stage has passed its dataset-quality gate. The 1,000-root v3 stage is now unlocked. The 1,000-root product must still pass the same full-artifact EDA before 2,000 roots are generated.
