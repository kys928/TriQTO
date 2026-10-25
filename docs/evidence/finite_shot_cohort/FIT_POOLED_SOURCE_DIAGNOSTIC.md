# TriQTO — pooled-calibration source-centering control (FIT only)

**Date:** 2026-10-25. **Status:** Completed exploratory acquisition-source diagnostic; NOT a deployable R2 estimator validated at equal shot cost.

**Fixed protocol:** [FIT_POOLED_COUNT_CENTER_DIAGNOSTIC.md](../../experiments/FIT_POOLED_COUNT_CENTER_DIAGNOSTIC.md), committed before its run. **Source:** successful [GitHub Actions run 40207829396](https://github.com/kys928/TriQTO/actions/runs/40207829396). Evidence artifact `pooled-source-fit-analysis`, ID 12604531837 (retention-limited). Original SHA256-verified release, 120 FIT families, one first root/candidate/observation each. **No SELECTION, TEST, oracle source probabilities, model scores, labels or protected reserve data were opened.**

## Procedure

For each N=4096, 2N=8192, 4N=16384, first K independent original calibration realization count tables were pooled for K in {1,2,4,8}. Their empirical pooled distribution estimated the generating probability for each of 18 plus/minus × axes × bases candidate calibration calls. Exactly 512 future **N-shot** count realizations were simulated *from that estimated distribution*, transformed to the frozen 3 inferred coordinates and 3 matched responses and evaluated with a 6D covariance.

The evaluation target was the **same independent last eight acquisitions (draw indices 8–15)** for all K. No target draws were pooled into any estimated distribution. This eight-draw reference has substantial sampling noise. K=8 requires **eight times** the original per-call acquisition shots just to estimate the source probabilities; repeated simulation of future measurements costs no additional physical shots in this CPU experiment but a deployment would require its K observed acquisitions. All comparisons are deliberately **not shot-budget matched**.

Report median ratio of empirical coordinate/matched covariance trace to estimated trace; ideal is one.

| Budget | K=1 coordinates | K=2 coordinates | K=4 coordinates | K=8 coordinates |
|---|---:|---:|---:|---:|
| N | 1.6072 | 1.3411 | 1.0933 | 0.9951 |
| 2N | 2.1845 | 1.6546 | 1.2801 | 1.1164 |
| 4N | 1.9314 | 1.3837 | 1.1572 | 1.0057 |

Matched-response ratios for K=1,2,4,8: N 1.3081, 1.1272, 1.0502, 1.0050; 2N 1.2038, 1.0338, 0.9712, 0.9745; 4N 1.1089, 1.0093, 0.9491, 0.8853.

**Scientific inference:** replacing the noisy single acquisition's empirical source distribution with a better-observed pooled source produces nonlinear feature covariance that more closely tracks the independent 8-draw sample. This is substantial evidence that *source-centering error* contributes to the old 1.96× coordinate-bootstrap underdispersion. The improvement is not universal at 2N, and the matched features may overdisperse at larger K and 4N.

## Crucial budget and estimator boundary

The (K)-pooled source distribution is more accurate because **more measurement information was acquired**, not merely because shots were grouped into K draws. Dividing a *fixed* shot budget across subdraws and summing their counts does not increase the information relative to one aggregated multinomial draw with the same total count. Thus the K=8 result must **not** be advertised as a free same-shot solution.

Further work should test a data-derived, physically permissible regularizer of the single-shot source probability or Jacobian geometry, without ideal probabilities, and measure accurate full 6x6 covariance and feature coverage on separate development families. It must also compare extra-shot options against an appropriately increased-shot R0 baseline for fair cost. This report neither fits nor selects an R2 hyperparameter.

## Final gate

`FIT_EXPLORATORY_MECHANISM_SUPPORTED_BUT_NO_DEPLOYABLE_METHOD_APPROVED`. Existing historical/reserve identity attestation remains unsigned/unverified, and the original TEST mechanism blinding failed independently. **No replacement TEST may be generated** by this result, and no R0/R1/R2 comparison or SELECTION evaluation was conducted.
