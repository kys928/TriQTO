# Independent finite-shot stabilization cohort

Date: 2026-10-25. Parent: PR #107, commit `1a2d25defa7cb21fc495c9a634431f5ac88c1f73`.
Status: preregistered construction and feasibility; no remedy comparison permitted.

## Scientific question and boundary

Build an independent sample of the **existing** Step-14 task, suitable for later testing whether uncertainty-aware geometry preserves mechanism information while reducing measurement instability. This construction makes no model-performance, generalization, hardware, or remedy-efficacy claim. Step-15 weights, normalizer, architecture, candidates, profiles, and pooling are frozen. No network inference or accuracy evaluation is performed here; archived raw evidence must support those computations later.

Frozen source: `configs/v0_2/step14_cross_motif_generalization_training.json`, its generator, `analyze_step14_latent_frame_inference.py`, `analyze_step14_local_frame_canonicalization.py`, and `finite_shot_uncertainty.py`. The broad hypothesis paper is background, not permission to change this task.

## Proposed design and feasibility gate

240 accepted families, four angle variants each, 13 observations each (one clean, four strengths crossed with three mechanisms): 960 roots, 12,480 observations. FIT/SELECTION/TEST = 120/60/60 families; representative/stress = 192/48, with 96/24, 48/12, 48/12 in the three splits. All descendants and measurement draws stay with their family. Split allocation is fixed before measurement, with a namespace distinct from historical generation. Engineering pilot: four separate families, excluded from the scientific cohort.

Preserve qubits 2–5, the six named grammar topologies, 4–9 reference layers, H/RX/RY/RZ/CX/CZ probabilities, three injection contexts, four strength intervals [0.05,0.12), [0.12,0.20), [0.20,0.28), [0.28,0.36], and root-specific permutation of diagnostic shots 512/1024/2048/4096 shared across mechanisms. Record both nominal topology and realized operation graph: the six grammar names need not denote six distinct graphs, particularly at two qubits.

Calibration: Z/X/Y axes, independent plus/minus probes and Z/X/Y measurement bases, original epsilon and N=4096 shots **per call**. Proposed 16 independent calibration draws at each of N/2N/4N, independent across budgets. Pair all eventual remedies on identical acquired counts. Diagnostic reference and observed draws are independent, sampled once per observation and held fixed across calibration draws to isolate calibration variation. Their randomness uses a separate stream. This does not estimate joint repeated-diagnostic instability; new diagnostic draws are a later, separately preregistered question.

Before full materialization, report exact candidate count, number of 18-call calibrations, total simulated shots, pilot wall time, projected storage and runtime. Multinomial CPU simulation incurs no QPU charges; no pod is authorized by this protocol. If impractical, document a prospective amendment before full data generation, never quietly reduce a completed sample.

## Admission, representative sampling, stress, and failures

Retain the original model-blind admission: each variant's true mechanism response norm at strength 0.12 >=0.02 and pairwise mechanism separation >=0.05, together with the original legal-location rules. All attempts and rejection reasons are logged. Failed admission families are a separately inventoried non-identifiable screening population, not erased or presented as accepted observations. This is an admission-conditioned population, not uniform arbitrary circuits.

Representative families are the first eligible independent families from the original grammar; do not exclude naturally difficult representatives. Stress families are independently generated under the same grammar and admission, additionally satisfying a frozen geometric rule: at least one admitted variant at the intended injection has (a) minimum response-column norm <0.30, (b) minimum singular value of unit-normalized response frame <0.15, or (c) maximum absolute cosine between two nonzero columns >0.98. Use the central-difference response at the original calibration probe for this rule. Structural zero columns and discrete mechanism identifiability are reported separately: a low-rank local derivative does not automatically imply indistinguishable finite distortions. Maximum 20,000 attempts; exhaustion fails the planned cohort. These rules use no model output. True-location/ideal screening quantities remain analysis-only and never enter observed-input extraction or uncertainty estimation.

## Independence and TEST sealing

Compare family skeleton signatures (original context-sensitive and context-free), ordered operation signatures, original rounded graph hashes, and exact parameterized circuit identities against accessible historical products and within/across new partitions. Preserve source checksums and coverage inventory. Original family IDs/seed changes alone do not establish independence. Parameter-free graph/skeleton matches are rejected even when angles differ. Graph isomorphism up to qubit relabeling is an additional audit; topology-class overlap is intentional.

Never read inaccessible sealed reserves to manufacture a complete exclusion claim. A missing historical exclusion inventory remains **UNVERIFIED** and prevents the global VERIFIED_AND_FROZEN_FOR_STABILIZATION_DEVELOPMENT status. If accessible products are disjoint, construction may continue as a quarantined candidate corpus, but no remedy selection is admitted until all independence gates are resolved.

Store TEST truth and generation recipes separately from blind evidence; encrypt before external archival and never publish the key or plaintext. Mechanism, affected qubit, boundary, strengths, ideal probabilities/states, admission scores, seeds that reconstruct TEST labels, and label-bearing IDs must not enter estimator/model inputs. Public TEST EDA includes only counts, integrity, and structural metadata; TEST measured difficulty, feature geometry, outcomes, and labels are excluded from development EDA. Integrity audit may check TEST array shapes/count sums/finiteness without releasing distribution summaries.

## Raw evidence and reconstruction

Retain original clean circuit operation order/angles, opaque observation IDs, basis/outcome convention, ordered plausible candidates, calibration counts [budget,draw,candidate,axis,sign,basis,outcome], per-observation paired diagnostic counts and actual shot totals, and separate RNG provenance. Keep simulator probabilities in a restricted acquisition-audit area; they are never estimator inputs. Persist derived Jacobians, 24-wide canonical features and profile/pair geometry or a hash-bound deterministic reconstruction recipe if storage demands. Candidate logits/final decisions are deliberately deferred: computing them is model inference outside this construction task.

Uncertainty: analytic unbiased count covariance includes within-basis correlations. For A/B joint uncertainty, use an observed-count multinomial bootstrap with 64 replicates on a deterministic development audit sample (first root per selected family, first ordered candidate, opaque first observation, N draw zero). Report bootstrap failure/boundary behavior. Empirical repeated-calibration covariance over 16 draws is a distinct repeated-acquisition diagnostic, not a deployable per-draw estimator. It must not be mislabeled as the observed-count estimator. Conditional bootstrap holds diagnostic evidence fixed. Calibration variance/bias against simulator acquisition targets is evaluation-only; no ideal quantity feeds extraction. Nonlinearity means covariance agreement is descriptive, with no automatic validity assertion near null directions.

## EDA, audit, and acceptance

Pre-materialization: grammar/joint coverage, identity overlap, legality, split and stratum accounting, feasibility. Post-materialization: completeness, immutable hashes, independent count-to-Jacobian and canonical reconstruction, seed replay, nonnegative integer counts and exact totals, basis ordering, independent RNG provenance, correlated coordinate/matched uncertainty, singular spectra/norms/condition numbers, stress vs representative comparisons, diagnostic shot/strength balance and nuisance associations on development truth only, outliers, nonfinite values, and numerical edge cases.

Report family-cluster uncertainty for descriptive rates, joint coverage gaps and sampling limitations; do not interpret the number of candidate draws as independent circuit sample size. No hypothesis/performance tuning or EDA-driven removal. Unexpected issues are either integrity failures (fail closed) or scientific properties (retain and disclose). A separate audit implementation must not merely call the generator's validation functions. Meaningful regression tests must catch wrong axes/bases/signs, count corruption, label leakage, split leakage, covariance dependence, and seed reuse.

All gates must pass before the requested VERIFIED status: complete cohort + raw provenance + source-wide content independence + raw/derived reconstruction + uncertainty checks + sealed TEST + independent audit + durable versioned archive + scientific report. Otherwise use a specific BLOCKED/INCOMPLETE status and enumerate every failed/unverified gate. Preregister changes before their run; commit protocol/code before full generation. R0/R1/R2 comparison, training, parameter selection and TEST outcome inspection are forbidden.
