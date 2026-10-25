# Post-release adversarial audit — completed findings and prospective oracle-only explanation check

**Date:** 2026-10-25. Original immutable cohort: `sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d`. **Data-informed exploratory audit, not a new blinded test.**  
Source jobs: [40203936394](https://github.com/kys928/TriQTO/actions/runs/40203936394) (**success**, hash-verified original release, development-only extraction). Scripts:
`scripts/cohort/adversarial_public_metadata_recovery.py`,
`scripts/cohort/review_nonlinear_covariance.py`.

## Verified public-metadata attack (development truth only)

The audit reconstructed all 13 observation mechanisms, strength-bin identities and clean/target indicators from publicly recorded root IDs and ordered observation records by reproducing the published PCG64 permutation. It reconstructed affected qubit from the first RY gate and injection boundary from the public injection context plus entangling-event positions. Independent FIT/SELECTION truth was used **solely to verify the attack**:

| Recovery measure | Correct / total |
|---|---:|
| Mechanism | 9,360 / 9,360 |
| Strength bin | 9,360 / 9,360 |
| Clean control | 9,360 / 9,360 |
| Affected qubit | 9,360 / 9,360 |
| Injection boundary | 9,360 / 9,360 |

**720/720 FIT/SELECTION roots; zero TEST truth accessed, zero TEST raw arrays extracted or inspected, no decryption key or sealed archive used.** A publicly available, deterministic input-to-label computation is enough to establish that the released TEST is not an untouched **mechanism-blind** evaluation. It may remain useful for engineering and exploratory diagnostics, but cannot be resealed by renaming/reordering after release.

Proof uses the known frozen generator structure and exposes no reconstructed individual TEST label. The attack's exact-code equality to the development archive is stronger evidence than a theoretical review alone; a new secret-keyed permutation must be independently audited **before** any replacement TEST is acquired.

## Observed-count nonlinear variance check

Replicated original EDA selection: first root of each of 180 FIT/SELECTION families, first candidate, first diagnostic observation, calibration draws (16), same frozen geometry algebra. For each of N=4096, 2N, 4N, independently drew 128 plug-in multinomial bootstraps from 3 predeclared reference draws (0, 7, 15). Compared trace of the empirical 16-draw coordinate/matched covariance to the **average** bootstrap covariance trace across these three references; not the average of ratios. Independent coordinate/matched reconstruction was checked against the retained frozen feature bytes for draw 0.

| Budget | Median coordinate empirical / pooled-plugin trace | Median matched empirical / pooled-plugin trace |
|---|---:|---:|
| N | 1.9310 | 1.3056 |
| 2N | 2.3266 | 1.2831 |
| 4N | 2.2966 | 1.1227 |

Median first-reference coordinate ratios: 1.8580, 2.2939, 2.1911 at N,2N,4N. An elevated coordinate ratio is **not explained solely by using only 64 bootstrap draws around an unfortunate first observed reference**, though the analysis cannot rule out finite empirical-draw or bootstrap Monte Carlo effects altogether.

### Stratification by true central-difference minimum axis norm

Analysis-only exact reference geometry (never estimator input) produced 44 near-null roots (min axis norm <0.1), 39 weak roots [0.1,0.3), and 97 roots >=0.3. N pooled bootstrap coordinate ratios: **1.3632 near-null; 1.8300 weak; 2.2910 >=0.3**. At 2N: 1.4776, 2.1552, 2.8951. At 4N: 1.6583, 2.1789, 2.8481.

Consequently, **the simple story that low true column norm alone explains underdispersion is contradicted by this stratification**. Threshold 0.3 was prespecified but is not a shot-normalized signal-to-noise criterion. A stronger reference geometry may have large coordinates/response sensitivity or distorted geometry in other axes; per-axis SNR and true-centered finite-shot sampling require a separate follow-up before mechanistic claims.

## Prospectively defined *second* diagnostic: true-centered acquisition control

After observing the above but **before running the next job**, compare the following on the **same permitted 180 roots**: (1) empirical 16-draw feature covariance, (2) a plug-in 128-draw covariance drawn from 3 observed count references, and (3) an independent **256-draw oracle-centered Monte Carlo** generated from the originally archived simulator probabilities of the **first candidate only**. The latter is **analysis-only simulator truth**, forbidden from deployable estimators and model inputs.

Across N/2N/4N, report trace-ratio distributions empirical/oracle, plug-in/oracle, and empirical/plug-in, plus axis SNR=`||true_j[:,k]|| / sqrt(tr(true count covariance for axis k))` in strata `[0,1),[1,3),[3,∞)`. Compare true-centered and plug-in distributions for the coordinates and matched responses; show whether reference centering or the nonlinear estimator accounts for elevated ratios. Fixed protocol seed `sha256(["oracle-centered-nonlinear-control-20261025",root_id,budget])`; 256 draws each, not selected by results.

**Interpretation guard:** even if true-centered simulation supports a specific mechanism, this remains exploratory finite-shot simulator analysis and cannot establish calibration quality on real hardware. Do not select R2 weights/hyperparameters on these audit results or promote the cohort to independent development.

## Integrity and research boundary

Original TEST, sealed labels, private key, frozen Step-15 checkpoints, original corpus archive, published manifest, and external reserve remain untouched. The original `COHORT_ACCEPTANCE.json` must not be changed to VERIFIED. Replacement plan: `docs/experiments/REPLACEMENT_BLINDED_STABILIZATION_TEST_PROTOCOL.md`. The public TEST label blindness is **invalid regardless** of whether historical external identity exclusion becomes complete.

## Completed true-centered control (prospective second audit)

After pre-recording the separate oracle-centered control in this document, the independent [GitHub Actions run 40204202193](https://github.com/kys928/TriQTO/actions/runs/40204202193) **succeeded**. It used the same first root/candidate/observation of 180 development families and all three shot budgets; each has 16 original independent acquisitions, 3×128 observed-count plug-in resamples and 256 independent **analysis-only** counts drawn from the simulator's known acquisition probabilities. No TEST files were extracted and no model inference was executed. The source run has the full-precision aggregate JSON artifact `truecentered-nonlinearity-aggregates` (ID `12603805429`, retention-limited). The permanent [machine-readable summary](POST_RELEASE_ADVERSARIAL_SUMMARY.json) preserves the source-log medians.

| Calibration budget | Median empirical-coordinate / true-centered variance | Median observed-bootstrap-coordinate / true-centered variance | Median empirical-matched / true-centered variance | Median observed-bootstrap-matched / true-centered variance |
|---|---:|---:|---:|---:|
| N=4096 | 0.97198 | **0.52109** | 1.00438 | 0.72573 |
| 2N=8192 | 1.03807 | **0.44218** | 0.96463 | 0.79217 |
| 4N=16384 | 0.97593 | **0.40513** | 0.99468 | 0.82830 |

Empirical coordinate variation closely reproduces the true-centered simulator-control distribution at all three shot budgets, whereas the *plug-in distribution centered on a single noisy observed count table* captures only about 41–52% of true-centered coordinate variance in the median root. Even when pooling three reference covariance estimates, this underdispersion persists. Matched responses also exhibit underdispersion, though smaller.

**Supported mechanism-level inference:** the failure is consistent with the local plug-in multinomial distribution insufficiently representing unconditional independent-acquisition variation after the nonlinear coordinate map. The original count-covariance Jacobian calculation can remain accurate while downstream coordinate uncertainty fails. The exact role of angular normalization, sign/null regions, statistical finite-sample effects, and source-centering cannot be uniquely isolated by these three covariance trace comparisons. In particular, the N true-norm stratification above refutes the overly narrow claim that only near-zero true axis norms matter.

**Implication:** do not use uncorrected plug-in covariance as a calibrated reliability weight for R2. A subsequent independent-development calibration study must predefine candidate remedies for uncertainty estimation and test their coverage/calibration before comparing stabilization performance. Do not tune a scale factor (e.g. 2×) on this diagnostic or infer hardware validity. No method R0/R1/R2 has been scored or selected.

**Global research decision:** public TEST label-blinding integrity **FAIL**; new independent TEST needed. Development-family source-exclusion attestation remains pending. Both gates block confirmatory remedy selection. The previous original cohort archive is retained unchanged and remains valuable for descriptive finite-shot diagnostics.
