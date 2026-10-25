# FIT-only inference-marginalization pilot and family-screen result (2026-10-25)

**Scientific status:** exploratory development result, NOT a completed method freeze, not independently confirmed, and not a replacement TEST evaluation. User approved keeping protected reserves and the real blind 60-family TEST untouched; the historical SELECTION cohort has already been spent for R0/R1/R2.

## Reproducibility

- Prospective [protocol](../../experiments/FIT_INFERENCE_POSTERIOR_MARGINALIZATION_PREDECLARATION_20261025.md).
- [Full experimental run 40227975547](https://github.com/kys928/TriQTO/actions/runs/40227975547): success, 7 numerical tests passed, four FIT-only inference shards, 32 distinct FIT circuit families, first root per family, two injected observations of each of three mechanisms (6/root), 16 independent archived acquisitions, shot budget N=4096, 64 conditional Monte Carlo replicates per method.
- [Aggregate FIT screen artifact](https://github.com/kys928/TriQTO/actions/runs/40227975547): `fit-posterior-marginalization-screen` ID **12612391370**. Source candidate network weights 6201/6202/6203 were SHA-256 verified against original frozen Step-14 FIT-only equivalence-aware checkpoint signatures; frozen head/pooling never retrained.
- [Post-hoc descriptive whole-family uncertainty run 40228272651](https://github.com/kys928/TriQTO/actions/runs/40228272651) success, artifact `posterior-inference-fit-paired-diagnostics` ID **12612411638**. No method reselection from the screening group.

## Hypotheses

I0 is exact R0 inference baseline. I1 draws 64 latent probabilities from Dirichlet(observed calibration outcome counts + 1/2) and integrates softmax mechanism probabilities from the frozen network. I2 draws such latent probabilities AND resamples a simulated N-shot acquisition before integrating frozen output probabilities. Original measured 24D features, frozen evidence profiles and equivalent-frame distance/gate geometry are never edited, and no extra QPU shots are used. This only integrates **candidate-mechanism score-path uncertainty**, not a replacement for uncertainty in the full frame-geometry posterior.

## Exact FIT observations

| Cohort | Method | Within-example 16-acquisition disagreement | Per-draw mechanism accuracy |
|---|---|---:|---:|
| Families 40:56 (training) | I0 | 21.4583% | 74.8047% |
| | I1 | 18.7413% | 75.1302% |
| | I2 | **17.9688%** | 74.6745% |
| Families 56:72 (family-disjoint screen) | I0 | 16.7014% | 75.4557% |
| | I1 | **16.0503%** | 75.1953% |
| | I2 | 16.2326% | 75.1302% |

I2 was the **prospectively prechosen** method from training because it had the greatest observed disagreement reduction with at most 1 percentage point observed accuracy loss. I2's 16-family FIT screening disagreement reduction was **0.46875 percentage points** and accuracy reduction **0.32552 points**. I1 had better screening numbers but was **not substituted post-hoc**. Neither is yet an accepted R0 replacement.

## Family-level uncertainty and prediction-flip warning

FIT screening's descriptive family-cluster 4,000-rep bootstrap for I2−I0 gives:
- disagreement delta −0.46875 percentage points; percentile 95% interval **[−2.8212, +1.9967] points**;
- accuracy delta −0.32552 percentage points; percentile 95% interval **[−1.6927, +0.9766] points**.

Both contain zero. The accuracy lower bound is near the prior 2pp noninferiority safeguard; the stability upper bound is **positive**. This is not sufficiently precise to claim a reliable benefit.

Of 1,536 screening draw-level mechanism predictions, I2 altered **82**: **34 wrong→correct**, **39 correct→wrong**, and 9 wrong→different-wrong. Effects were not class-symmetric: truth class 0 gained 22 correct and lost 7, but classes 1 and 2 each gained 6 and lost 16. This **could indicate a systematic class-specific score shift** requiring verification before treating reduced disagreement as superior inference.

## Scientific interpretation / hard preservation rules

Averaging frozen-network output across uncertain calibration evidence can lower finite-shot decision disagreement **without attenuating quantum-coordinate features**, supporting it as a plausible new research direction. However neither I1 nor I2 is independently validated, nor is posterior Dirichlet+multinomial a certified description of hardware noise; I2 may overcount different noise components. A conservative one-method (I2) repeat has been prospectively specified for the last unused FIT 48 families, indices 72:120, with full family bootstrap and no retuning: [locked I2 replication protocol](../../experiments/LOCKED_I2_FIT_INTERNAL_REPLICATION_20261025.md). That replication is a further FIT screening gate, not the spent SELECTION or the blind TEST.

**No final method-selected claim, no new SELECTION data acquisition, no genuine blind replacement TEST acquisition, no S3 writes, QPU usage or protected reserve access.** Original 2026-10-25 frozen R0 remains the currently valid model method.
