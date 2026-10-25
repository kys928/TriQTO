# TriQTO frozen Step-15 model: research-data lineage and exclusion scope

**Date: 2026-10-25. Status: repository-declared provenance verified for consistency; historical metadata completeness remains bounded.** This is based on the frozen tracked protocols and manifests, not an attestation of every storage path.

## The actual frozen inference model

- The **three candidate mechanism networks** are the Step-14 equivalence-aware **FIT-only latent candidate networks** with seeds **6201, 6202, 6203**, not the preceding Step-14 cross-motif training seeds 1701–1703.
- Primary checkpoint hashes:
  - seed 6201: `sha256:861f9cd1d20b2b3d53d5a04d969e06bb0d539e59084bc3e47ff34c9a0ed8d416`
  - seed 6202: `sha256:2469148e71d8fa7b200edac65adc4c0d0c745dd0ab9ffc37cf915216985c56c8`
  - seed 6203: `sha256:4bd586b015b0a9aaa863d856108da2552eee3e62259574e1adb6f60ee1e3c40f`
- Method: `equiv_fit_4df8aea718950bdc2650a03b`, frozen `tau=0.02`, frame temperature `2.0`, method bundle hash `sha256:b074e9278ee9d77465cc88b655ec214c0f477ae8aaa4bb157c3c600fc7d38f0b`. Frozen file: `configs/v0_2/step14_equivalence_aware_latent_frame_final_method_freeze.json`.
- The **Step-15 FIT-only linear frame rank adapter** method `step15_rank_462fb0f8b4531b25f9a29345`, adapter bytes hash `sha256:4e98bac63955b7709008a0fe9d83a70c7134c429d05e1f364ac7edcdf71f41aa`, developed around the above three frozen networks. Step-15 adapter fit source is the same Step-14 FIT partition; the mechanism network is not retrained by Step 15. See `configs/v0_2/step15_frame_ranking_development.json` and `runpod/step15-verify/frozen_462fb0f8_20261016.json`.

## Influence classification — which data can matter

| Data lineage | Influence on this frozen mechanism/adapter evaluation | Exclusion treatment |
|---|---|---|
| Step-14 `development_b087edfd6629ac250299391d`, 600 FIT families / 2400 roots | **Direct candidate-network training and fit-fold hyperparameter choice**, subsequent Step-15 rank-adapter training | **MUST EXCLUDE** at family, context-free skeleton, wire and graph identity levels |
| Same Step-14 product, 150 selection families / 600 roots | Used in earlier Step-14 development and model-selection provenance for predecessor checkpoints; **not** permitted in equivalence-aware FIT-only or Step-15 FIT-only ranker training | **EXCLUDE conservatively** |
| Legacy Step-10 original/bridge FIT and SELECTION, method product `product_0f7112597501f7ea5fbe123b` and predecessor checkpoint development | Influenced the **earlier Step-14 cross-motif training stage (seeds 1701–1703)**; **not established as direct training data for seed 6201–6203**, which were trained anew on Step-14 FIT | **EXCLUDE conservatively** because model-family research/architecture decisions may carry forward |
| Earlier Step-5 and Step-12 diagnostics/source motifs | Influenced background representation and hypothesis selection; Step-12 known-failure diagnostic is report-only under frozen Step-14 rule | **Exclude known circuit identities where available; not proof of direct training** |
| Spent 150-family Step-14 equivalence fresh holdout; published 36 selected calibration diagnostics | **Post-fit scientific inspection**, not reported as Step-14/Step-15 FIT hyperparameter selection | **Exclude** from subsequent exploratory/confirmatory claims because publicly examined |
| Simulator outer; future-hardware reserve | Declared forbidden from FIT-only candidate-network and rank-adapter training by both frozen protocols. Step-14 config specifically says future-hardware reserve **not materialized in Step 14** | Keep sealed; lack of access does **not** establish that the reserve never existed later. Not a condition for narrower training-lineage disjointness; do not claim full protected-source independence |
| 2026-10-25 finite-shot 240-family archive (FIT/SELECTION/compromised original TEST) | Generated *after* the frozen model was trained, so cannot be prior checkpoint training examples. FIT has been used to explore uncertainty estimator; SELECTION not used as of this memo | FIT for estimator training, SELECTION only for *one* method selection, original TEST not confirmatory |

## Existing checked structural overlaps

S3 identity-only read-only [run 40216188215](https://github.com/kys928/TriQTO/actions/runs/40216188215) compared two candidate commitments against *11 permitted* historical identity manifests (15,400 total source rows, duplicated source families possible), and found **zero** available/comparable identity intersections in five structural fields for both original 240-family and prospective 60-family replacement TEST candidate commitments. [Source coverage digest 40216354888](https://github.com/kys928/TriQTO/actions/runs/40216354888): 4 protected namespaces and 14 unclassified top-level namespaces not inspected, some early products lack full operation/family/wire signatures. **This is stronger evidence for observed graph disjointness, not a proof of exhaustive equivalence-closed exclusion.**

## Revised scientific claim (narrow, not overstated)

A forthcoming replacement TEST may support *post-freeze generalization to newly proposed 60-family Step-14 simulator circuit families, checked against enumerated used/spent history to the available structural identity scope*. It **cannot** support “all historical, protected, topologically equivalent or future-hardware circuits are independently certified unseen,” nor a QPU-noise transfer claim. The public mechanism ordering failure invalidates the originally released TEST irrespective of graph disjointness.

**No need to open protected reserves to proceed with FIT development or with preparation of a genuinely blinded future TEST.** Keep source incompleteness and external certification as separate explicit qualifications. Do not spend original TEST or unlock protected evaluation, and do not retrofit the old public labels.
