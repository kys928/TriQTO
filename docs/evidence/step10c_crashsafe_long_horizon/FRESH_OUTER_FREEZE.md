# Step 10C fresh outer cohort — frozen outcome and independent audit

Status: **ACCEPTED / FROZEN BEFORE ANY STEP-10C MODEL OUTCOME**

Product: `product_57ee407d62ea794bfc9ff169`

This document freezes the fresh outer-validation data-quality outcome before any Step-10C model training or evaluation result exists. The cohort is validation-only and may not select epochs, thresholds, architecture, or hyperparameters.

## Immutable product identity

- Product schema: `triqto.v0_2.step10c_fresh_outer_cohort.v1`
- Product status: `COMPLETE_FROZEN_OUTER_VALIDATION`
- `dataset_complete.json` SHA256: `sha256:858ddeadb5b4380b71325b8294727b6a9e3431293e0cfd150c4f18d05f0129ce`
- `eda.json` SHA256: `sha256:f5a97c66792b23b8ba7a7a382cb4dbd44081320db058f5294938697c1d0b0964`
- `EDA_SUMMARY.md` SHA256: `sha256:9a740812bb2c403adfe6a2b6b4fe40b849c77b5fae18bcf62a315cfa8485f591`
- Original root manifest SHA256: `sha256:2d7aa22c9ab662eeab7d4d12806cd14ae1e8ee1304c8a4cc2053f36082c4fb3e`
- Original example manifest SHA256: `sha256:10380939508cb753d5e4fc99e4ebf14a9c37affdbdf4ff6ada1add8157229949`
- Bridge root manifest SHA256: `sha256:eb20b48e7ca820ef4dd7e90128463ee7c2f3361f3c1f55e8b474aa85f8756b85`
- Bridge example manifest SHA256: `sha256:a073fc16959df1020e758454b03886fe88fba3b4e90745266bcc6515b1a7656a`
- Independently uploaded full-cohort ZIP SHA256: `c85b9e4f5a9a5e476b3eaae137779e4abaf7df76a3809ba10556aae13ba7d33f`
- Uploaded ZIP size: 60,122,491 bytes.

The ZIP passed CRC validation and contains exactly 19,250 unique entries: 19,240 NPZ artifacts, four manifests, three JSON files, one EDA markdown summary, one generation log, and one non-scientific PID file. No duplicate ZIP member names were found.

## Frozen source identity

The generated product records the following immutable source identities:

- Step-10 mixture: `product_0f7112597501f7ea5fbe123b`
- Step-10 `dataset_complete.json` SHA256: `sha256:fc4b529677c5165c96ac191aee33622efdbba4fade3867795f9f84778e1ca3c4`
- Original Step-5-v3 product: `product_b2d78ad2309b71a55f9bb54f`
- Original product `dataset_complete.json` SHA256: `sha256:063b8f6db05bca754092bf1b4ff524cdfc39e762f3f5217fd8be1cd0e9a171a1`
- Fresh bridge base seed: `2026082101`
- Fresh original global generator range: 5000–5999 inclusive.
- Fresh-outer generator SHA256 recorded by the product: `sha256:6fa1c035d38f5312e0b79c505f4c3cf0042bcf70f34dc47c1214b33c894da4d2`.

The source Step-10 dataset hash is the previously frozen Step-10A product hash; the fresh generator checked overlap against the corresponding historical original and bridge manifests from those immutable source products.

## Population and independence structure

Fresh original domain:

- 1,000 independent clean roots.
- Global root IDs exactly 5000–5999 with no gaps.
- 13 examples per root = 13,000 examples.
- Every root has exactly one clean control and 12 injected examples.
- Every root has exactly four RZ, four RX, and four RY interventions.
- Every root uses all four injected shot levels: 512, 1024, 2048, 4096.
- Family root counts: bell-like 50; GHZ 150; hardware-efficient ansatz 150; phase-interference 150; QAOA-like 150; QFT-like 150; random-shallow 200.
- Qubit root counts: 2q 259; 3q 181; 4q 194; 6q 189; 8q 177.

Fresh bridge domain:

- 60 independent parent groups.
- Exactly eight variants per parent = 480 clean roots.
- Exactly 13 examples per root = 6,240 examples.
- Each parent contains variant indices 0–7 exactly once.
- Each parent contains two variants of each frozen motif.
- Parent qubit counts are exactly balanced: 20 parents at 2q, 20 at 3q, 20 at 4q; therefore 160 roots at each qubit count.
- Motif root counts are exactly balanced: 120 each for pilot-core-variant, spectator-pre, spectator-mid, spectator-tail.
- Every bridge root has exactly one clean control, four RZ, four RX, and four RY interventions and all four shot levels.
- Bootstrap dependence is preserved at the 60-parent-group level.

Across the fresh cohort, all 1,480 clean graph hashes are unique and original-vs-bridge clean-graph overlap is zero.

## Full independent artifact audit

The uploaded archive was independently scanned, not merely accepted from `eda.json`.

All 19,240 NPZ files were opened with `allow_pickle=False`. For every artifact the independent audit verified:

- archive member exists and manifest artifact SHA256 matches the NPZ bytes;
- a single identical NPZ key schema is used across all artifacts;
- no persisted key contains `statevector`;
- no deployable `x__` key contains mechanism, phenomenology, effect, overlap, affected-qubit, insertion, strength, or statevector target/privileged semantics;
- `meta__example_id` and `meta__clean_circuit_group_id` match the manifest;
- clean/effect/mechanism/mask/phenomenology targets match the manifest;
- affected qubit, insertion boundary, and strength audit values match the manifest;
- the graph hash independently recomputed from gate names, qubit pointers/indices, parameter pointers, and rounded per-gate sine/cosine values matches the manifest;
- logical-to-physical layout is identity and has the expected qubit count;
- diagnostic basis codes are exactly `[0,1,2]`;
- observed/reference shot arrays agree with the manifest and with each other;
- reference masks are available and the frozen reference-kind code is correct;
- local, pairwise, parity and pair-index tensor shapes agree with qubit count;
- pair indices are exactly all unordered qubit pairs;
- graph qubit/parameter pointer arrays are monotonic, correctly terminated, and in range;
- stored parameter sine/cosine phasors satisfy `sin²+cos²=1` within numerical tolerance;
- primary and exact diagnostic arrays are finite;
- all primary diagnostics satisfy the frozen absolute bound 2.0000001;
- continuous supervision values are finite and physically consistent.

Independent counts for all of the above violation classes are zero.

The largest absolute primary diagnostic value anywhere in the 19,240 artifacts is `0.45703125`, far below the frozen `2.0000001` bound.

## Independent supervision-consistency audit

For every injected artifact, the persisted privileged continuous targets satisfy:

- `population_component + phase_component = total_overlap_loss` to a worst observed numerical error of approximately `1.11e-15`;
- `effect_present` is exactly `total_overlap_loss >= 1e-8`;
- phenomenology exactly follows the frozen negligible / phase-dominant / population-dominant / mixed rule with dominance ratio 2.0;
- `dominance_log_ratio = log((phase+1e-12)/(population+1e-12))` exactly to the tested precision.

For all clean controls, privileged continuous targets are zero and all exact diagnostic deltas are zero. Empirical clean-control deltas remain nonzero as expected from independent finite-shot sampling.

## Mechanism and effect supports

Nominal intervention counts remain exactly balanced.

Original effect-positive mechanism supports:

- RZ: 3,478 / 4,000 = 0.8695
- RX: 3,243 / 4,000 = 0.81075
- RY: 4,000 / 4,000 = 1.0

Bridge effect-positive mechanism supports:

- RZ: 1,279 / 1,920 = 0.6661458333
- RX: 1,440 / 1,920 = 0.75
- RY: 1,920 / 1,920 = 1.0

These masks reflect simulator-defined effect detectability; they do not change the nominal matched intervention schedule. Balanced accuracy remains the correct primary mechanism metric because effect-positive support differs by mechanism.

The fresh bridge support is very close to the spent Step-10B bridge truth support (RZ 1,280; RX 1,437; RY 1,919), which is consistent with a fresh draw from the same frozen bridge family rather than a materially shifted validation distribution.

## Finite-shot sanity check

The independent audit recomputed empirical-vs-exact diagnostic RMS error. Mean error decreases monotonically with shot count in both domains:

Original: 512=`0.05520818`, 1024=`0.03881777`, 2048=`0.02733041`, 4096=`0.01942533`.

Bridge: 512=`0.05414453`, 1024=`0.03810160`, 2048=`0.02688066`, 4096=`0.01910256`.

A log-log fit of error versus shots has slopes `-0.50270` (original) and `-0.50124` (bridge), essentially the expected finite-sampling `1/sqrt(N)` behavior. This is a strong acquisition sanity check and shows no anomalous shot-noise scaling.

## Diagnostic magnitude

Mean diagnostic RMS:

- original: `0.03966818` (median `0.03709846`);
- bridge: `0.05038482` (median `0.04587261`).

Bridge evidence is somewhat larger in magnitude, which is expected because its frozen strengths span 0.08–0.30 rather than only 0.05/0.15. This is a distribution property, not a reason to alter Step-10C training or gates after seeing the outer cohort.

## Design associations and an important qualification

Independent Cramér's-V recomputation exactly reproduces the frozen EDA.

Original domain maximum reported association is small: shot/affected-qubit `0.04693`; shot/depth `0.04205`; shot/strength `0.02341`; depth/strength and depth/family are exactly zero.

Bridge domain has small acquisition associations: shot/affected-qubit `0.03374`, shot/depth `0.02768`, shot/strength `0.03632`, shot/family `0.0`, strength/family `0.0`. It also has larger **structural** associations inherited from the frozen Step-10 bridge context schedule: strength/depth `0.45734` and depth/family `0.34314`.

These two larger associations do **not** constitute mechanism-label leakage: mechanism has Cramér's V exactly `0.0` with strength, depth, family, shots, and affected qubit in both domains, and every matched context contains exactly one RZ, one RX, and one RY example. Strength, hidden insertion depth, affected qubit, and family labels are also excluded from deployable model inputs under the frozen contract.

Nevertheless this is a real claim-boundary qualification: Step-10C bridge conclusions are conditional on this structured deployment-domain family and must not be generalized to arbitrary independent strength/depth/motif distributions. A later broader-domain study would need to break or explicitly randomize those structural associations if that stronger claim is desired.

## Freshness against historical data

The generated EDA reports:

- overlap with all Step-10 original clean graphs: 0;
- overlap with all Step-10 bridge clean graphs: 0;
- duplicate fresh clean graphs: 0;
- exact Step-9D pilot graph present: false;
- minimum wrapped phi distance from pilot phi=0.7: `0.0152595658`, outside the frozen exclusion half-width 0.01.

The full uploaded archive does not itself contain the historical Step-10 graph manifests, so an entirely archive-local recomputation of historical overlap is impossible. The historical-overlap result is therefore grounded in the frozen generator's comparison against the immutable source products identified above, not independently reconstructed from this ZIP alone. This limitation is explicit rather than hidden.

## Decision

**ACCEPT `product_57ee407d62ea794bfc9ff169` AS THE FROZEN STEP-10C FRESH OUTER COHORT.**

No model performance was inspected while accepting the cohort. The acceptance decision is based only on data integrity, independence/freshness, acquisition behavior, supervision consistency, and frozen design-contract checks.

Step-10C training may proceed only after the crash-safe runner and authoritative fresh-outer access boundary are frozen and tested. The frozen numerical performance gates remain unchanged.