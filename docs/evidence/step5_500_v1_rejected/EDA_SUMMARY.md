# Step 5 500-root v1 full-artifact EDA — rejected candidate

Product: `product_bb5cbdcd33a591f2f8c97b7c`

Decision: **REJECT_FOR_TRAINING_REGENERATE_V2**.

The product is structurally and cryptographically intact, but two deterministic scheduling bugs make the train/validation dataset scientifically unsuitable for Step 6/7 model evaluation.

## Evidence inspected

The EDA independently scanned every uploaded NPZ artifact, not only manifests:

- train artifacts: 5,200;
- validation artifacts: 1,300;
- total artifacts: 6,500;
- train ZIP SHA-256: `63884c58bfd99fd3ec2ded2727044eec45393c03411825c3220efd6eb5fa7e38`;
- validation ZIP SHA-256: `1acab861770ea0118565ffe09592aeea55b8a36f14aee6f7709c465ee09c1609`;
- manifests ZIP SHA-256: `115ca9e1bc7894d281880a91e4d15a4d7ba19aa5835b448873d4199d03ad4e52`;
- `dataset_complete.json` SHA-256: `ac3d50e3fdbd06f74c1f94c91a26f09e708d1ccc2a04b7eb2c38c545de89d20b`;
- `stage_validation.json` SHA-256: `826f64e3e16e28f259aa856d3b5e2f2d5ad80815689a594c52826527e52724ba`.

## Integrity results

All 6,500 artifact byte hashes match `example_manifest.csv`. All NPZ files load with `allow_pickle=False`. A single common array schema is used across the cohort. There were zero mismatches for:

- `meta__example_id`;
- `meta__clean_circuit_group_id`;
- clean-control target;
- effect-present target;
- mechanism target;
- mechanism-loss mask;
- phenomenology target;
- affected-qubit audit truth;
- insertion-boundary audit truth;
- strength audit truth;
- shot counts;
- basis codes;
- reference-availability mask;
- identity logical-to-physical layout.

All deployable diagnostic arrays are finite. Maximum absolute empirical values were:

- local expectation delta: `0.30859375`;
- same-basis pairwise delta: `0.296875`;
- global parity delta: `0.296875`.

Maximum absolute exact audit values were below `0.15` for all three blocks.

No persisted `x__` key contains mechanism, affected-qubit, insertion-depth, strength, phenomenology, clean-control, or effect-present truth.

## Blocker 1 — family/split aliasing

The original global split rule `validation if root_index % 5 == 0` aliases with the 20-entry deterministic family cycle.

| family | train roots | validation roots |
|---|---:|---:|
| bell_like | 0 | 25 |
| ghz | 75 | 0 |
| hardware_efficient_ansatz | 50 | 25 |
| phase_interference | 75 | 0 |
| qaoa_like | 50 | 25 |
| qft_like | 50 | 25 |
| random_shallow | 100 | 0 |

Family-vs-split Cramer's V is approximately `0.61237`.

This is not an intentional out-of-family test. It makes validation composition materially different from training and confounds downstream target distributions.

### Frozen v2 repair

Split by zero-based **within-family occurrence index** instead:

`validation iff family_occurrence_index % 5 == 0`.

At 500 roots this produces the exact nested 80/20 family split:

- bell_like `20 / 5`;
- ghz `60 / 15`;
- hardware_efficient_ansatz `60 / 15`;
- phase_interference `60 / 15`;
- qaoa_like `60 / 15`;
- qft_like `60 / 15`;
- random_shallow `80 / 20`.

## Blocker 2 — insertion depth perfectly confounded with strength

The v1 fixed strength schedule `[0.05, 0.15, 0.05, 0.15]` was zipped with depth targets `[0.25, 0.50, 0.75, 1.00]`.

Therefore:

| depth | 0.05 examples | 0.15 examples |
|---|---:|---:|
| early | 1,500 | 0 |
| middle | 0 | 1,500 |
| late | 1,500 | 0 |
| terminal | 0 | 1,500 |

Depth-vs-strength Cramer's V is `1.0`.

The missing factorial cells make depth effects inseparable from strength effects.

### Frozen v2 repair

Alternate the two schedules by within-family occurrence parity:

- even occurrence: `[0.05, 0.15, 0.05, 0.15]`;
- odd occurrence: `[0.15, 0.05, 0.15, 0.05]`.

This keeps four contexts/root and exact mechanism matching while supplying both strengths at every depth within every family and both splits.

## 3-qubit addition

The v1 clean-root universe used only `2, 4, 6, 8` qubits. Step 5 v2 adds `3` to the non-Bell qubit choices: `[2, 3, 4, 6, 8]`. Bell-like remains forced to 2 qubits.

This avoids teaching the first variable-size graph training cohort an unnecessary even-size-only prior. Three-qubit circuits have already appeared in earlier Step 3/3.5 audit regimes, so this is not an entirely novel un-audited size class.

## Target structure in v1

Injected examples: `6,000`.

- mechanism supervised/effectful: `5,374`;
- injected negligible/masked: `626` (`10.43%`);
- clean controls: `500`.

Negligible rates by mechanism:

- RX: `381/2000 = 19.05%`;
- RY: `1/2000 = 0.05%`;
- RZ: `244/2000 = 12.20%`.

Negligible rates by family:

- HEA: `0.89%`;
- GHZ: `4.78%`;
- QAOA-like: `6.11%`;
- Bell-like: `8.33%`;
- phase-interference: `8.33%`;
- random-shallow: `10.00%`;
- QFT-like: `33.33%`.

Mechanism and phenomenology remain appropriately distinct:

- RZ: 1,422 phase-dominant, 159 population-dominant, 175 mixed, 244 negligible;
- RX: 610 phase-dominant, 552 population-dominant, 457 mixed, 381 negligible;
- RY: 325 phase-dominant, 1,140 population-dominant, 534 mixed, 1 negligible.

## Finite-shot numerical EDA

The selected hardware-facing correlation core is correctly noisy rather than idealized. Median empirical-vs-exact core error falls with shots approximately as expected:

| shots | median core shot error | fraction where shot error > exact signal |
|---:|---:|---:|
| 512 | 0.05555 | 0.9668 |
| 1024 | 0.03945 | 0.9126 |
| 2048 | 0.02756 | 0.8948 |
| 4096 | 0.01960 | 0.7766 |

For clean controls the median empirical core norm is:

- 512 shots: `0.05669`;
- 1024 shots: `0.03984`;
- 2048 shots: `0.02698`;
- 4096 shots: `0.01918`.

This confirms realistic `~1/sqrt(shots)` behavior, but also shows that many weak `0.05` perturbations are shot-noise dominated. This is **report-only at Step 5**, not a reason to leak exact simulator evidence into inputs. Step 6 baselines must determine learnability under this finite-shot regime.

## Reference-window metadata boundary

The manifest raw identifier distinguishes strings such as `rootN:clean` from `rootN:ctxK`. It is not persisted in the primary `x__` arrays. Step 5 v2 freezes raw reference-window IDs as identity/audit metadata only; any future model-facing timing feature must be neutral numerical metadata (for example age/delta/calibration-window relation), not a semantic identifier that reveals control/context status.

## Promotion decision

`500 -> 1000` remains **LOCKED** for v1.

The v2 500-root cohort must be regenerated and pass:

1. exact per-family train/validation coverage;
2. no missing depth x strength cells in either split or any family;
3. 3-qubit coverage in both train and validation;
4. full NPZ hash/schema/label/numerical EDA;
5. the original root/group/graph leakage gates.
