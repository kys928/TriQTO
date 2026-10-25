# TriQTO — stabilization research continuation: independent-source gates and nonlinear uncertainty (pre-execution)

**Recorded 2026-10-25.** Branch `research/stabilization-gates-20261025`, based on PR #108. This is a prospective extension to an *already explored* cohort; exploratory method feasibility, **not** a fresh preregistered confirmatory hypothesis. The original corpus and source hashes remain immutable.

## Scope and staged execution

1. **Historical/reserve identity attestation (must precede remedy selection and TEST construction):** existing development/historical sources were checked at [the prior EDA/audit](../evidence/finite_shot_cohort/SUPPLEMENTAL_AUDIT.md), including 8,171 unique extra graph hashes and the spent holdout; all checked overlaps were zero. Three unresolved gates remain: protected reserve identities not accessible under previously frozen no-inspection policy, incomplete historical registry, and missing early-product operation/family signatures. This new protocol does **not** authorize searching sealed truth, silently opening inaccessible reserve manifests, or asserting complete source coverage from a partial directory list.

2. **Independent custodian attestation workflow:** export only public candidate family/graph/ordered-operation/wire-relabel identity tokens; an independently authorized custodian can compare them with *its own* verified protected identity-only manifests and sign an aggregate coverage/no-overlap statement. The custodian chooses and explicitly declares a comprehensive source registry. A missing source, missing required identity type, positive overlap, unsworn exhaustive coverage, or missing valid cryptographic signature is **BLOCKED**. Raw protected identities never leave the custodian, and no model outcomes, oracle labels or reserve predictions enter the comparison. Signature and authoritative source coverage must be checked by the research reviewer; unsigned trial runs are merely engineering tests. No protected path will be read by ordinary CI.

3. **Replacement TEST:** preserved Step-14 grammar, new independent families, 60 families×4 variants×13 observations prospectively; fully secret-keyed observation-to-mechanism schedule and opaque IDs controlled by an external custodian, with blinded aggregated scoring. Never use the publicly released old TEST as intact independent confirmation. **Do not generate or acquire replacement TEST until the complete identity gate is accepted and final R0/R1/R2 methodology is frozen.** First prepare and test the secrecy contract using only synthetic fixtures.

4. **Nonlinear uncertainty FIT feasibility:** examine two explicitly named **same-observed-shot** estimator families, without using true simulator probabilities inside the estimator:
   - `plugin_multinomial`: original multinomial bootstrap from observed normalized counts.
   - `jeffreys_posterior_predictive`: for each calibration call and outcome, draw (p'\sim\mathrm{Dirichlet}(n+1/2)), then future counts from (\mathrm{Multinomial}(N,p')). This posterior-predictive quantity combines uncertainty about (p) with the next measurement's noise; it is **not** an unbiased plug-in estimator of frequentist fixed-probability acquisition variance.
   
   Use FIT family data only. One first root/candidate/observation per family, all N/2N/4N, 16 independent acquisition draws to estimate descriptive reference covariance; compute 192 observed-only resamples from three fixed count references (indices 0,7,15) per method, average estimated covariance matrices across the 3 references, report coordinate/matched traces, trace-ratio distribution, median absolute log ratio, over/underestimation and numerical failures. No hyperparameter tuning or method selection in this stage. Frozen coefficient 0.5 is the Jeffreys prior and is **not** fitted to the previous 1.96 ratio. Record posterior-predictive comparison as a *feasibility method*, never as validated R2 reliability.

5. **After FIT only:** prospectively freeze one estimator or reject both based on prespecified accuracy and calibration requirements and assess it once on 60 SELECTION families, then only if independent-source gate is independently attested. Evaluate J-level inflation, 6x6 coordinate/matched covariance, covariance PSD, coverage of independent repeated-acquisition features and robust tail behavior. Any candidate that improves coordinate calibration by massively inflating linear/Jacobian uncertainty or degrading matched reliability is not automatically acceptable. Do not use oracle probabilities to train/fit/correct uncertainty estimates.

6. **The frozen mechanism network (Step-15) remains untouched.** No QPU access, no pod, no selection on spent selected 36 diagnostics, no decryption of original TEST truth. No claim of real hardware transfer.

## Truth custody and subtlety

Synthetic finite-shot acquisition is physically informative but does not validate QPU noise. Ideal simulator probabilities may be used solely in explicitly labeled post-hoc mechanism controls, not in estimator inputs or replacement test selection. The first-RY qubit proxy remains a documented structural limitation of the preserved grammar. A truly independent *mechanism-label* TEST is possible only if mechanism order is secret-keyed; strict location-agnostic TEST is not possible under the current first-RY grammar, and is not claimed.

## Required evidence before proceeding

- Source registry names, completeness declaration, protected identity-only manifest checksums, independently validated Ed25519 custodian signature, zero overlaps across all **required** structural identity types, explicit equivalence-scope limits.
- Public candidate identity-only export checksum, fixture-negative and positive overlap unit tests.
- FIT estimator distribution and methods compared at *equal single-draw N* (not extra calibration shots). Evidence of any numerical instability or overconfidence is retained.
- Revised acceptance record stays quarantined until independently reviewed; original TEST never promoted.

**STOP CONDITION:** if no custodian can authorize protected reserve identity-only attestation, do **not** create a replacement TEST or select R0/R1/R2. Publish the precise missing permission/provenance instead.
