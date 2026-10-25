# TriQTO 2026-10-25 — narrower-lineage remedy and blind replacement TEST preparation

**Status:** Substantial development completed; **NO FINAL REMEDY SELECTED; REAL REPLACEMENT TEST NOT ACQUIRED**. This report is FIT exploratory and cryptographic construction QA, **not** confirmatory inference.

## 1. Actual frozen-model data lineage

[Source lineage record](FROZEN_MODEL_DATA_LINEAGE_20261025.md) distinguishes the actual candidate mechanism networks (Step-14 FIT-only equivalence-aware, seeds 6201, 6202, 6203, SHA256 bound) from the older original Step-14 cross-motif ensemble (1701–1703) and the downstream Step-15 FIT-only rank adapter. Source product: `development_b087edfd6629ac250299391d`, 600 historical FIT and 150 historical selection families. Known earlier Step-10 training/model-selection sources conservatively excluded, along with spent 150-family holdout and selected 36 diagnostics. No explicit permission to inspect protected reserves was exercised.

The previous read-only S3 lineage scan [run 40216188215](https://github.com/kys928/TriQTO/actions/runs/40216188215) compared 11 eligible manifests (15,400 rows) with the current cohort and replacement candidate identities; no available/comparable overlaps across five structural signature fields. **Exhaustive historical/source or protected-reserve certification NOT claimed.** Frozen Step-14 source expressly says future-hardware reserve was not materialized in Step-14; post-Step-14 absence has not been proved. No protected reserve source content was accessed.

## 2. FIT-derived uncertainty estimator and mechanics

[Successful FIT coefficients + R1/R2 feature QA run 40219331737](https://github.com/kys928/TriQTO/actions/runs/40219331737): 7/7 initial mechanics tests, 120 FIT families, 3 shot budgets; covariance corrections are from one observed-count calibration acquisition and historical **FIT-trained** two-block factors, not future simulator oracle probabilities or extra quantum shots.

| Shot budget | Coordinate multiplier | Matched multiplier |
|---|---:|---:|
| 4096 | 1.9043089474 | 1.3909188307 |
| 8192 | 2.3276603323 | 1.2883596651 |
| 16384 | 2.2822104796 | 1.2223094136 |

R0: exact canonical feature identity. R1: smooth observed-uncertainty attenuation of matched signs, with absolute matched slots recomputed; R1 now strictly tends to original features for **zero sampling covariance**. R2: full joint 6x6 observed-count covariance applied to coordinate plus matched feature vector with derived `abs` and unit-direction slots recomputed. Frozen C geometry and frame/position/gate pooling inputs unchanged. This is specifically a **candidate-network score-path intervention**, not a proof of full geometry/pooling repair. R2 caused much larger FIT feature moves (median 15D RMS feature shift at N ~0.409) vs R1 (~0.035 in the original uncorrected R1 pilot); serious frozen-network input distribution warning.

[Selection-only rule implementation](../../scripts/cohort/select_stabilization_remedy.py) enforces exactly 60 families, four roots each, 12 injected observations per root, 16 draws per mechanism case; 2000 family-block bootstrap repeats, 2pp absolute balanced-accuracy noninferiority and Bonferroni-adjusted disagreement benefit. It returns R0 unless a frozen R1/R2 beats R0 and passes both statistical gates. The separate synthetic [decision/geometry tests run 40220806497](https://github.com/kys928/TriQTO/actions/runs/40220806497) passed **12/12** including stable-constant-classifier accuracy rejection. **No actual SELECTION evidence has been scored with this selector.**

## 3. Bounded real frozen-network FIT score-path comparison

The 3 frozen candidate network checkpoints were loaded **read-only from the non-protected Step-14 equivalence-aware S3 method folder**, and each file's SHA256 was checked against the public committed freeze. Frozen pooling functions were extracted from source whose SHA256 is pinned in `scripts/cohort/core.py`; no training weights or pooling hyperparameters changed.

After correcting the zero-uncertainty R1 identity limit, the successful [FIT-only network replay run 40220555456](https://github.com/kys928/TriQTO/actions/runs/40220555456) reports **8 FIT families, first root/family, first 3 non-clean FIT examples, 16 independent count acquisitions at N=4096**. These 24 examples are **not representative confirmatory selection**; their labels were used only for descriptive FIT classification checks.

| Candidate | Pairwise disagreement | Per-draw accuracy (not balanced accuracy) |
|---|---:|---:|
| R0 | 22.9167% | 70.3125% |
| R1 (corrected) | 23.7847% | 70.5729% |
| R2 | 23.8194% | 70.5729% |

Neither remedy improves this small FIT sample's *stability*. Small apparent mean accuracy gains are statistically uninformative. There is **no validated intervention winner** and no current basis to claim a remedy improves performance or is safe to use in TEST. R0 remains the default/fallback pending a complete eligible one-time SELECTION study. All performance figures are ideal finite-shot simulation, not QPU evidence.

## 4. Cryptographically blind TEST: preparation and hard gate

- [S3-independent TEST identity generation #40213970222](https://github.com/kys928/TriQTO/actions/runs/40213970222): 60 new Step-14 families / 240 roots, five comparable known-history structural overlap categories zero; identity commitment `sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1`.
- [Full identity-only reconstruction #40220087083](https://github.com/kys928/TriQTO/actions/runs/40220087083): **all 60 / 240 exact structural identities reproduced**, no TEST measurements.
- [Sacrificial sealed-test pilot #40219704691](https://github.com/kys928/TriQTO/actions/runs/40219704691): 6/6 secrecy/gating tests, four roots / 52 observations generated with unpredictable identifiers and private keyed order. AES-256-GCM truth encryption and RSA-OAEP-SHA256 key wrapping verified in a disposable fixture; its private test key, fixture counts and labels **were never published**, and the fixture was destroyed.
- [Real TEST builder](../../scripts/cohort/build_blind_replacement_test_v2.py) requires a separate final method freeze `triqto.stabilization_final_method_freeze.v1`, a recipient-controlled RSA public key, all 60 exact identities and absence of existing output; it refuses to overwrite. Public circuit/shot/count files can be released while secret permutation/truth/locations remain sealed. **Only the mechanism-label ordering is cryptographically blinded:** preserved first-RY circuit grammar exposes intended affected-qubit structural proxy; strict location blindness is not claimed.

**The real 60-family replacement TEST has not been sampled or acquired.** This is deliberate: user-required stage 3, independent SELECTION accuracy safety and method freeze are not finished. Releasing TEST observations before that point would risk contaminating confirmation. Original public-ID reconstructable TEST is still invalid and never reused for confirmation.

## Verdict and remaining practical engineering work

- Steps 1 and 2: lineage and read-only provenance scope written; protected reserves sealed; model used FIT-only per frozen source declarations. Stronger historical completeness is separately unresolved, not falsely marked certified.
- Step 3: R1/R2 implemented, 7+12 mechanistic tests, FIT uncertainty factors and *actual frozen-model score* exploratory validation **done**. **Complete 60-family SELECTION scoring, acceptable accuracy/stability margins and final method decision still PENDING**.
- Step 4: replacement 60-family circuit and public-label-blind acquisition/sealing builder **implemented and pilot-validated**, **actual TEST not yet materialized**, to preserve one-shot evaluation integrity.
- Step 5: untouched final single-use TEST **NOT RUN**.

Research conclusion: do not promote R1/R2 merely because they have correct covariance estimates; actual preliminary score disagreement got slightly worse. Either improve methods on FIT and freeze a fresh candidate before a full one-time SELECTION evaluation, or let the predefined selector retain R0 if no tested remedy passes. Then lock model/protocol/evidence hashes and acquire/test once with a separately held recipient private key. **Do not re-open the old TEST or protected reserves.**
