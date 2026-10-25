# Independent follow-up: public-metadata label recovery and nonlinear covariance

**Status:** Prospective, data-informed adversarial audit, defined after the quarantined cohort was constructed. **Not a confirmatory experiment** and not authorization to tune R0/R1/R2.

## A. Public metadata leakage: permitted-development demonstration

Use only `corpus/roots.json`, `corpus/families.json` (both contain structural/public metadata), and `corpus/development_truth/FIT|SELECTION/*.json` for an *audit comparison*. The label recovery code must be executable without importing the original generator, reading TEST records, knowing a TEST decryption key, or opening any `sealed/` files.

For every one of 720 development roots:
- Reconstruct the deterministic `PCG64` observation permutation using the public namespace, the literal `observation-order` stream label, and the public root ID.
- Map ordinal to the generator's original 13-row schedule (one clean then four groups of RZ/RX/RY). Recover each observation's mechanism, clean flag, and strength-bin label.
- From the clean circuit's **first** RY gate recover the intended affected qubit; from public family `injection_context_class` and entangling-event indices recover the injection boundary by the frozen generator contract.
- Compare all reconstructed labels with permitted development truth and fail if any mismatch. Report aggregate numerator/denominator; do not output recovered label rows or IDs.

**No test truth / raw arrays / ciphertext are permitted.** This audit proves the original TEST blinding design is unsound as a function of public input even if tested only on development. An existing released TEST cohort cannot be made untouched again.

## B. Nonlinear uncertainty calibration: independent diagnostic

Use development-only first roots per family, first ordered candidate, first diagnostic observation, matched across N,2N,4N as specified in cohort EDA. Reproduce the original empirical 16-draw covariance for 3 coordinates and 3 matched-response components, then recompute the observed-count bootstrap covariance using **three distinct observed reference draws** (indices 0, 7, 15), **128 plug-in multinomial replicates per reference**, fixed SHA256-derived audit seeds. Do not choose reference indices or sample based on the observed ratios.

Primary descriptive outcomes per budget: median and [p10,p90] empirical/bootstrap trace ratios for coordinates and matched responses, fraction of roots above 1.5 and 2, between-reference bootstrap variability, and empirical changes across N,2N,4N. Preserve by-family clustering. Compare estimated ratios in strata of **analysis-only exact central-difference minimum Jacobian-axis norm** (thresholds 0.1,0.3 frozen here), and realized rank/condition features. Do not feed exact probabilities to an uncertainty estimator; they may be used only for investigation into why it fails.

Potential (not assumed) mechanism: normalizing a near-zero/noise-dominated response column produces a random direction. A plug-in bootstrap centered on a noisy estimate may underrepresent variation across fully independent acquisitions. Distinguish this from finite 16-draw empirical variance and 128-replicate Monte Carlo error. Report diagnostics as exploratory mechanism evidence, never causal certainty.

**Do not change old archives, source freeze, inferred network normalization, Step-15 weights, selection choices, or any protected reserve.** If the audit requires an unanticipated scientific code change, record a new amendment before rerun.

## C. Replacement untouched TEST: design, not generation

Publish a separate prospective, scientifically comparable TEST protocol retaining the Step-14 physical task and circuit grammar, with a new family namespace and identity-only disjointness attestation. Unlike the publicly leaked TEST:
- Keep the observation-to-label permutation **secret-keyed**, generated inside a trusted acquisition/custodian process, and do not publish the permutation/seed/key or a deterministic public means of deriving it.
- Public root IDs and observation IDs must be fresh, opaque, unrelated to secret labels or salts; no slot-order, diagnostic-shot, strength-bin, or timing side channel may encode mechanism. Pair clean/reference and distorted evidence with the frozen acquisition contract; shuffled order stays private.
- Seal TEST truth and scoring before any predictions. Provide a blind inference interface and publish only attested aggregate metrics after commitment, not blind row labels or reproducible secret recipes.
- Explicitly recognize that preserving the original *first-RY* circuit grammar leaks the affected qubit and that publishing the injection-context class leaks the boundary. Blind the *metadata interface* (at minimum do not provide the true context class or label-correlated indices) and avoid claiming location blindness where the reference circuit itself reveals structural proxies; testing new location-neutral grammar would be a separate experiment.
- Identify exactly which public/protected artifacts are exposed and threat-model untrusted investigators with public repository/release access. Add blind-only adversarial recovery tests before archive release.
- Do not use previously released public TEST cases or the 36 diagnostic examples for confirmation.

The existing FIT/SELECTION sample remains **quarantined for R0/R1/R2 method selection** until independent historical/reserve exclusion is attested. Existing publicly compromised TEST remains suitable for labeled engineering diagnostics only, never untouched performance confirmation.
