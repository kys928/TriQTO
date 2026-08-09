# Phase 15.6 confirmatory closeout

**Status:** CLOSED
**Study:** `confirm_d4d5b26d35fc64eff27c`
**Confirmatory holdout:** `SPENT_CONFIRMATORY`
**Future confirmatory reuse:** FORBIDDEN
**Future development/postmortem use:** ALLOWED, but only with explicit disclosure that these examples were previously observed in a confirmatory study.

This document permanently closes the one-shot Phase 15.6 phase/amplitude identifiability confirmation. It records the frozen outcome without rewriting the preregistered claims after seeing the holdout labels, preserves the study provenance, and separates confirmatory conclusions from post-hoc development observations.

## 1. Scientific boundary after closeout

The 160 entities in this holdout are no longer untouched evidence. They must never again be described as a fresh or independent confirmatory set.

They may be reused for development, diagnosis of failure modes, representation design, calibration research, or other postmortem work, provided that every future report clearly marks them as previously observed confirmatory data.

Any future untouched confirmatory holdout must exclude:

- these exact confirmatory entities;
- their split-group identities;
- direct derivatives that preserve the same source identity.

The prior 200-example development cohort and these 160 spent confirmatory examples may therefore form a 360-example known-development evidence pool for subsequent postmortem work, but none of the 160 retain confirmatory status.

## 2. Frozen provenance

| Field | Frozen value |
|---|---|
| Study ID | `confirm_d4d5b26d35fc64eff27c` |
| Holdout count | `160` |
| Completion time | `2026-08-08T13:14:32.869668+00:00` |
| Protocol SHA-256 | `sha256:d4d5b26d35fc64eff27c5d910440c54ea7881c0ce643411531c2d44f604a0809` |
| Holdout product ID | `sha256:996376802e121c568a2a3c8792d72a576dde4f8d0e2d3807ccb6e5aaeb9322b2` |
| Results SHA-256 | `sha256:7e7a5cc656e08939bb11316e6b90793a275ab6afb3b3344dd9aee0599eccc5b4` |
| Predictions SHA-256 | `sha256:6b1e8bdd604a4d1ce757acc7b7b02b8c980257d8c22f55d5ba36a39ac3e73f84` |
| Metrics SHA-256 | `sha256:96a0ea142c3fc61906411820dcadea9e5f55a01d04405ba8374f5f0dbea04c18` |
| Stratified metrics SHA-256 | `sha256:5c6f7f3ab1f45fd0814b834d7f0f3d422ac304527c0a568c5d49db913c643ede` |
| Decision SHA-256 | `sha256:0256f211ef8aaee5f3943b322f93befc8438f456ba283ea0849dfd4621b1ac02` |
| Blinded predictions SHA-256 | `sha256:faae1280bdfd6ceaab85b8c6c773650718c43cf56d91f8a009d00b348690f27e` |
| Blinded fit SHA-256 | `sha256:dbc552cfc84ecbcf163188d341c17232e5f0786573dcdf7b9e4eeb512af87d23` |

The one-shot evaluator reported `confirmatory_metrics_accessed_once=true`. The historical v0.1 test was not accessed, and development-validation feature artifacts were not accessed during the confirmatory evaluation.

## 3. Frozen confirmatory outcome

### Primary claim — NOT_SUPPORTED

Frozen claim:

> Relational Z/X/Y observables provide reproducible phase/amplitude identifiability and materially outperform absolute distorted observables on a new untouched holdout.

The frozen decision was `NOT_SUPPORTED` because:

- linear `B_delta` macro-F1 was `0.687695`, below the required `0.70`;
- linear `B_delta` amplitude recall was `0.462500`, below the required `0.70`;
- nonlinear `B_delta` amplitude recall was `0.562500`, below the required `0.70`.

This negative decision must not be retrospectively changed.

### Secondary claim — NOT_SUPPORTED

Frozen claim:

> Compressed Hilbert summaries add reproducible incremental information beyond relational observables.

The frozen decision was `NOT_SUPPORTED` because:

- linear `C_summary - B_delta` balanced-accuracy point difference was `-0.056250`;
- linear `C_summary` exceeded the frozen degradation limit;
- neither model had a `C_summary - B_delta` balanced-accuracy confidence-interval lower bound above zero.

This negative decision must not be retrospectively changed.

## 4. Confirmatory metrics worth preserving

### Absolute-observable control

`B_absolute` was effectively chance-level in balanced accuracy:

| Model | BA | Macro-F1 | Phase recall | Amplitude recall | AUROC | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.5000 | 0.3730 | 0.0500 | 0.9500 | 0.5678 | 0.0240 |
| Nonlinear | 0.5000 | 0.3333 | 0.0000 | 1.0000 | 0.5116 | 0.0067 |

### Relational observables

`B_delta` did not satisfy the full preregistered claim, but it did preserve a non-chance signal under the frozen controls:

| Model | BA | BA 95% CI | Macro-F1 | Phase recall | Amplitude recall | AUROC | ECE | Shuffle p |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.7063 | [0.6296, 0.7787] | 0.6877 | 0.9500 | 0.4625 | 0.8527 | 0.2005 | 0.047619 |
| Nonlinear | 0.7125 | [0.6136, 0.7896] | 0.7059 | 0.8625 | 0.5625 | 0.8433 | 0.1492 | 0.047619 |

Relative to `B_absolute`, the paired balanced-accuracy improvements were:

- linear: `+0.20625`, 95% CI `[0.11510, 0.28847]`;
- nonlinear: `+0.21250`, 95% CI `[0.11873, 0.29135]`.

Therefore the correct confirmatory interpretation is not that `B_delta` solved phase/amplitude diagnosis. It did not. The defensible result is narrower: relational evidence materially outperformed the absolute-observable control and contained non-chance information, while the frozen balanced class-performance requirements were not met, especially for amplitude recall.

### Hilbert-summary extension

`C_summary` did not demonstrate reproducible incremental value beyond `B_delta`:

- linear BA changed from `0.70625` to `0.65000`, difference `-0.05625`, 95% CI `[-0.09880, -0.01695]`;
- nonlinear BA changed from `0.71250` to `0.73125`, difference `+0.01875`, 95% CI `[-0.02444, 0.06564]`.

The linear degradation was reliably negative, while the nonlinear improvement interval crossed zero. The main v0.2 diagnostic design therefore receives no confirmatory justification from this study to include `C_summary`.

## 5. Confirmatory findings versus post-hoc observations

The following are **confirmatory findings**, because they come directly from the frozen evaluation and its predefined decision procedure:

1. The primary claim is `NOT_SUPPORTED`.
2. The secondary claim is `NOT_SUPPORTED`.
3. `B_absolute` did not provide useful balanced phase/amplitude discrimination under the tested setup.
4. `B_delta` materially outperformed `B_absolute` in balanced accuracy for both frozen model families, with paired bootstrap intervals above zero and empirical shuffle p-value `0.047619` for both models.
5. `B_delta` nevertheless failed the frozen balanced class-performance requirements, most clearly through inadequate amplitude recall.
6. `C_summary` did not establish reproducible incremental value beyond `B_delta`.
7. The historical v0.1 test remained untouched.

The following are **POST-HOC / EXPLORATORY ONLY** and must never be rewritten as confirmatory findings from this study:

- alternative decision thresholds inspected after the holdout labels became visible;
- the observation that AUROC remained substantially stronger than hard-threshold class recall, suggesting a calibration/decision-boundary problem;
- circuit-family-dependent performance differences;
- qubit-count-dependent performance differences and bias reversal in some strata;
- strength-dependent or RX-versus-RY interaction hypotheses;
- any claim that circuit-conditioned graph context will repair the weakness;
- any conclusion that a future `DiagnosticEncoder` architecture is validated by this holdout.

Those observations are now legitimate development hypotheses because the holdout is spent. They can motivate the next experiments, but they require new evidence before becoming confirmatory claims.

## 6. Consequence for the roadmap

Phase 15.6 is now closed. No additional threshold tuning, classifier tuning, or architecture selection performed on these 160 examples may be represented as confirmation.

The next gate is **label-semantics audit**, before any architecture change:

- test whether injected RZ drift is consistently meaningful as a `phase-like` target across circuit states;
- test whether injected RX/RY overrotation is consistently meaningful as an `amplitude-like` target across circuit states;
- separate error mechanism from observable effect;
- determine whether the future target should remain categorical, become effect-based, or become continuous/mixed.

Only after that label-semantics audit should the project proceed to relational-evidence identifiability, hardware deployability, new paired dataset construction, baselines, and diagnostic-adapter work.

## 7. Permanent closeout statement

`confirm_d4d5b26d35fc64eff27c` is permanently classified as `SPENT_CONFIRMATORY`.

Its original confirmatory outcome is preserved as negative for both frozen claims. Its data may be used only as previously observed development/postmortem evidence from this point forward. Any later positive result produced by tuning on these examples must be described as development evidence and must be confirmed on a new untouched holdout before supporting a scientific claim.
