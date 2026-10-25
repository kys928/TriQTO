# TriQTO — one-time N-shot stabilization SELECTION and R0 method freeze

**Date:** 25 October 2026. **Scientific status:** Completed one-time development-method selection; **R0 retained** under preregistered accuracy-and-stability gates. **Not independent TEST confirmation, not a successful stabilization remedy, not an all-historical-source certification.**

## Provenance and methodology

- FIT-only alpha screen: [completed FIT reducer run 40222393103](https://github.com/kys928/TriQTO/actions/runs/40222393103), four FIT scoring shards under [run 40222061717](https://github.com/kys928/TriQTO/actions/runs/40222061717). The shards succeeded; the original reducer job failed only due an optional Qiskit import and was fixed by using the **same original FIT outputs**, without rerunning scoring or opening SELECTION. Predeclared grid selected candidate R1 alpha **0.5**, R2 alpha **0.01**. On disjoint FIT screening R1 reduced disagreement by 0.2431 percentage points with zero observed accuracy loss; R2 increased disagreement by 0.0521 percentage points. Not statistically validated.
- Pre-SELECTION lock: [STABILIZATION_PRE_SELECTION_LOCK_20261025.json](STABILIZATION_PRE_SELECTION_LOCK_20261025.json). Hash commits the alpha settings, the immutable frozen network checkpoint identities, exact scoring and statistical test source blobs, fixed N-shot uncertainty factors, original-cohort identity and explicit narrower lineage scope. All 60 SELECTION families were untouched until this lock was committed.
- One-time method-selection run: [GitHub Actions 40222647546](https://github.com/kys928/TriQTO/actions/runs/40222647546), **success**, all fifteen disjoint 4-family shards succeeded, then exactly one aggregate comparison. Hash-verified immutable source archive; **SELECTION partition only**, no original TEST, replacement TEST, original sealed reserve, model retraining or QPU.
- Full design: **60 independent selection families × four circuit variants × 12 injected mechanism observations × 16 finite-shot calibration draws at N=4096** = 2,880 mechanism observations and 46,080 per-remedy decisions. 16 draws from one root are not independent circuit families.
- Same SHA-256 verified seed 6201/6202/6203 frozen mechanism heads and fixed equivalence-aware pooling for R0/R1/R2. R1 alters matched-response signed and absolute feature slots; R2 jointly attenuates coordinate/matched slots and recomputes derived abs/unit directions. Other geometry, frame-profile/pair/routing and network weights frozen. The claim is **candidate mechanism score-path only**, not the full Step-15 ranking adapter.
- Predeclared decision: 2,000 seeded family-cluster bootstrap resamples, paired mechanism balanced accuracy (BA) noninferiority margin **−2.00 absolute percentage points**, require upper simultaneous Bonferroni 97.5% confidence bound for remedy−R0 disagreement to be below **zero**. R0 default if either gate fails.

## Exact one-time SELECTION result

| Quantity | R0 | R1 alpha 0.5 | R2 alpha 0.01 |
|---|---:|---:|---:|
| Mechanism balanced accuracy | 74.2318% | 74.2057% | 74.2405% |
| Pairwise within-example 16-draw label disagreement | 19.7338% | 19.6172% | 19.7260% |
| Change in BA vs R0 (percentage points) | reference | −0.02604 | +0.00868 |
| Change in disagreement vs R0 (percentage points) | reference | −0.11661 | −0.00781 |
| Lower simultaneous BA difference confidence bound (percentage points) | — | −0.17798 | −0.02170 |
| Upper simultaneous disagreement difference confidence bound (percentage points) | — | **+0.01597** | **+0.04804** |
| Accuracy noninferiority | — | **PASS** | **PASS** |
| Positive stability-effect inference | — | **FAIL** | **FAIL** |
| Joint decision pass | — | **NO** | **NO** |

Even R1's positive numerical improvement is compatible with *no improvement* under the prespecified uncertainty bound. Do not claim statistically established remedy benefit or a degradation of R2 beyond the measured non-significant contrast. The effect of R2 was particularly small. Both preserve balanced accuracy according to the noninferiority safeguard.

**Decision: R0 — frozen original mechanism score path.** This is a predefined *no-demonstrated-improvement fallback*, not evidence that R0 is better than every possible stabilizer.

## Immutable method freeze and post-run verification

- Source aggregate verdict preserved verbatim in [ONE_TIME_STABILIZATION_SELECTION_VERDICT_20261025.json](ONE_TIME_STABILIZATION_SELECTION_VERDICT_20261025.json).
- [FROZEN_STABILIZATION_METHOD_R0_20261025.json](FROZEN_STABILIZATION_METHOD_R0_20261025.json) binds the exact selected R0 implementation hash, calibration factor commitment, model checkpoint identities, original comparison protocol, narrower source lineage record, preselection alpha lock, and the SHA-256 of the complete SELECTION verdict. Generated only **after** the one-time run completed, **before any replacement TEST acquisition**.
- Reproducibility: [GitHub Actions 40223770478](https://github.com/kys928/TriQTO/actions/runs/40223770478) built the freeze from the single completed verdict. Additional verification workflow re-derives it from committed byte-identical evidence and applies fail-closed tests.

## Scientific limitations and next decision

1. Original finite-shot TEST labels were leaked by public deterministic ID-seeded mechanism ordering; it was never used for confirmation.
2. Source completeness remains bounded to documented, actually checked training and earlier development sources; four protected source namespace identities remain unaudited. No exhaustive independent protected-source certification is claimed.
3. R1/R2 intervention was applied to candidate **network input features only**; original frame-profile/pair kernels and Step-15 rank adapter remain untouched. These results cannot prove stabilization of all upstream geometry or downstream decision instability, and do not establish QPU noise behavior.
4. At this point, the original 60-family **SELECTION is spent** for method choice. Do **not** inspect it to invent a new R1/R2 method, tune a new alpha, redo a method selection using these same families, or claim independent validation of new methods on it. Future remediation requires new FIT development then a **fresh family-independent selection cohort**.
5. The prospective 60-family/240-root replacement TEST construction and cryptographic pilot are validated, but the **real TEST has not been acquired, unsealed or evaluated**. Because R0 was retained due to failure to demonstrate remedy improvement, it is defensible to hold the replacement TEST for a future method that passes fresh selection, rather than spend it on an unsupported candidate.

**Verdict:** Finite-shot covariance calibration improved numerically on FIT, but **neither proposed score-path attenuation remedy proved a stability gain while preserving accuracy** on full preregistered SELECTION. The correct research outcome is a published negative result, frozen R0 baseline and a still-unused replacement TEST.
