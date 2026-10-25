# Locked I2 posterior inference: final unused FIT-family replication

**Date:** 25 October 2026. **Scientific verdict:** internal FIT replication passed the preregistered exploratory go/no-go criteria for *designing a fresh SELECTION*. This is not external confirmation, not independent TEST, not a certified noise model, and does not supersede the previous R0 final method freeze.

## Protocol, code and execution

- [Predeclared 48-family internal FIT replication](../../experiments/LOCKED_I2_FIT_INTERNAL_REPLICATION_20261025.md) before the 72:120 FIT cohort was loaded.
- [GitHub Actions run 40228543373](https://github.com/kys928/TriQTO/actions/runs/40228543373) succeeded; four 12-family inference shards and one strict aggregate audit completed; frozen Step-14 mechanism checkpoints were SHA-256 verified. Result artifact `locked-i2-fit-replication-result` ID **12612432132**, retention 60 days.
- Exact fixed **I2**, no tuning or I1 substitution: one observed finite-shot calibration acquisition; 64 posterior predictive replicates using Dirichlet(counts+0.5) and a hypothetical multinomial N-shot draw; recompute candidate hypothetical 24D evidence; score through the three frozen mechanism heads; original observed frame profiles and pair geometry stay fixed; average softmax of pooled logits. *No additional real QPU shots*. R0 is unchanged baseline.
- Use 48 previously unused FIT families (manifest indices 72:120), first root per family, six class-balanced injected observations per root, 16 independent original acquisitions, N=4096; **288 mechanism examples/4,608 paired decisions per method**, 4,000 whole-family deterministic bootstrap resamples. No spent SELECTION, original leaked TEST, new blind TEST or protected reserve accessed.

## Exact results

| Metric | Frozen R0 | Locked I2 |
|---|---:|---:|
| Mean within-example 16-draw mechanism disagreement | 18.6372% | **17.1325%** |
| Mean draw-level mechanism accuracy | 75.6727% | **75.7161%** |

Paired I2−R0:
- Disagreement **−1.50463 percentage points**, exploratory whole-family 95% percentile interval **[−2.70544, −0.39352] points**. The full interval is below zero.
- Accuracy **+0.04340 percentage points**, exploratory whole-family 95% percentile interval **[−1.15017, +1.17188] points**. Lower bound exceeds the −2 percentage point internal noninferiority limit.
- Prospective FIT internal signal `go_to_fresh_independent_selection_design = true`. **This authorizes planning a new developmental cohort, not claiming validated superiority.**

## Mechanism-specific effects still matter

Of 4,608 draw-level decisions, I2 changed **305** predictions: **136** wrong→correct, **134** correct→wrong, **35** wrong→different wrong. Truth class 0: 48 repaired and 48 harmed, truth class 1: 43 repaired and 71 harmed, truth class 2: 45 repaired and 15 harmed. Overall stability improvement **does not imply each class improved**. The next study must report all classes separately and impose a prespecified accuracy safeguard, not rely solely on pooled agreement.

## Boundaries

- The Dirichlet+multinomial posterior predictive is an assumed statistical model; source covariance and joint geometrical calibration from earlier FIT studies are imperfect. No QPU/hardware calibration or robustness claimed.
- Feature tensor used by *original* observed R0 remains unchanged. I2 changes the downstream **mechanism-prediction computation**, integrating hypothetical noisy evidence and averaging class probabilities. Frozen candidate neural heads and original pair/profile kernels are untouched; Step-15 rank adapter **not tested**.
- The 60-family old SELECTION cannot be reused for a new method decision; it is spent by the earlier R0/R1/R2 comparison.
- The newly proposed original 60-family replacement TEST identity `sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1` has not been acquired or opened.
- Full independence from all historical/protected reserve sources remains unverified. Qualification: no comparable collisions with enumerated permitted training-lineage identities; do not claim source-exhaustive certification.

**Next valid milestone:** prepare an independently generated fresh development-SELECTION family cohort excluding *both* all known training/spent identities and the 60 reserved TEST families, lock I0/I2 code/model and the primary accuracy/stability criteria *before acquisition*, then complete exactly one new SELECTION study. Retain R0 if I2 fails that independent development gate. Keep replacement TEST sealed.
