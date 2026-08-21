# Step 10C — crash-safe longer-horizon dual-initialization benchmark

Status: **FROZEN BEFORE FRESH-OUTER GENERATION OUTCOME OR STEP-10C MODEL OUTCOME**

## Motivation and frozen predecessor

Step 10B completed as `benchmark_0971e5bb2ec77c2a1bb550d8` with the official decision `NO_INITIALIZATION_PASSES_FULL_DUAL_DOMAIN_GATE`. The completed result and posthoc audit were frozen before this protocol was written. Warm-start and scratch both passed confidence-lower-bound, minimum-recall, original-retention, and selected-checkpoint-eligibility subgates but missed only the bridge mechanism balanced-accuracy threshold of 0.80. Scratch selected epoch 20 for all three seeds, while warm-start selected epochs 19/17/18. This motivates a separately frozen optimization-horizon follow-up, not a reinterpretation of Step 10B.

The Step-10B outer-development cohorts are spent for this follow-up. Step 10C must use a new untouched outer cohort.

## Scientific question

With the Step-7 `late_concat` architecture, Step-7 graph-information contract, Step-10 leakage-safe fit/selection mixture, optimizer, learning rate, loss weights, gradient clipping, domain balance, retention rule, and final gates held fixed, does increasing only the maximum optimization horizon from 20 to 40 epochs permit either initialization to pass the dual-domain gate on a new untouched outer cohort?

## Frozen architecture and inputs

- Architecture: `late_concat`.
- Trainable parameter count: exactly 453,829.
- Graph information contract: unchanged from Step 10B / Step 7.
- No input-shape change.
- No per-gate phasor or candidate-location feature addition.
- No QPU access.
- No spent confirmatory cohort access.

Any architecture or input-contract mismatch is a hard failure before training.

## Frozen training and selection data

Step 10C reuses only the Step-10 product `product_0f7112597501f7ea5fbe123b` fit and selection partitions.

Original domain:
- fit roots: 3,000;
- selection roots: 1,000.

Bridge domain:
- fit roots: 1,440;
- selection roots: 480;
- all neighboring variants remain parent-group leakage-safe.

The Step-10B original and bridge outer **artifacts/predictions are never materialized or evaluated** by the Step-10C runner. Source manifests may be read only for immutable product-integrity verification and partition filtering; spent-outer rows never enter a model batch, selection metric, threshold, or final Step-10C metric.

## Frozen fresh outer cohort

The fresh outer cohort is generated and EDA-audited before any Step-10C model training. Data-quality EDA may inspect cohort labels, supports, finite-shot diagnostics, simulator-only audit arrays, and design associations, but no Step-10C model is evaluated on the cohort before the training/selection procedure is complete.

### Fresh original domain

- Same frozen Step-5 v3 circuit-family and acquisition contract.
- Previously unused global generator root indices 5000–5999 inclusive.
- 1,000 independent clean roots.
- 13 examples per root = 13,000 examples.
- New empirical reference and observed finite-shot samples generated deterministically from those fresh root/context identities.
- Zero clean-graph overlap with all Step-10 original or bridge roots is required.

### Fresh bridge domain

- Independent frozen base seed: `2026082101`.
- 60 independent parent groups.
- 8 neighboring variants per parent = 480 clean roots.
- 13 examples per root = 6,240 examples.
- Same four bridge motif families, 2/3/4-qubit coverage, phi/beta ranges, variant offsets, context strengths, shot levels, and RZ/RX/RY intervention semantics as Step 10A.
- Exact Step-9D pilot graph remains forbidden.
- Zero clean-graph overlap with all Step-10 original or bridge roots is required.
- Final bridge bootstrap unit is parent group, not individual variant/root.

The fresh outer generator must pass full-artifact EDA before Step-10C training is unlocked: artifact hashes, pickle-free NPZ loading, finite/bounded primary diagnostics, root derivative counts, mechanism schedule, shot coverage, graph uniqueness/overlap, effect-positive rates, phenomenology, diagnostic norms, empirical-versus-exact finite-shot error, and design association reports.

## Frozen initializations

Both are retained because all scratch seeds were visibly horizon-limited at epoch 20 in Step 10B.

- `warm_start`: strict matching-seed load of the frozen Step-9A `state_dict` for seeds 1701/1702/1703; new AdamW optimizer on a fresh run.
- `scratch`: ordinary deterministic seeded initialization for the same architecture; new AdamW optimizer.

No optimizer state from Step 9A or Step 10B is reused at a fresh Step-10C start.

## Frozen optimization contract

Unchanged from Step 10B except `max_epochs`:

- seeds: 1701, 1702, 1703;
- root batch size: 32;
- AdamW;
- learning rate: 0.0003;
- weight decay: 0.0001;
- effect loss weight: 1.0;
- mechanism loss weight: 1.0;
- gradient clip norm: 1.0;
- domain schedule: equal optimizer-block count from original and bridge per epoch, with deterministic cycling of the shorter domain;
- early-stopping patience: 4 eligible non-improving epochs;
- early-stopping min delta: 0.0005;
- **maximum epochs: 40**.

No scheduler, EMA, SWA, loss reweighting, batch-size change, learning-rate change, clipping change, or domain-weight change is permitted in Step 10C.

## Frozen best-checkpoint rule

Step 10C retains the exact Step-10B selection rule. A checkpoint is eligible only if original-domain selection mechanism BA and effect BA are each at least the matching warm-start epoch-0 value minus 0.02.

Eligible checkpoints are ordered by:

1. bridge selection mechanism balanced accuracy;
2. minimum(original mechanism BA, bridge mechanism BA);
3. bridge selection effect BA;
4. earlier epoch.

The fresh outer cohort selects nothing. Human monitoring may observe progress but may not override this rule or manually pick a checkpoint.

## Crash-safe persistence contract

Engineering hardening does not alter the model-selection rule.

For every initialization/seed run:

- `best.pt` is atomically replaced whenever the frozen eligible-checkpoint key improves;
- `resume.pt` is atomically replaced after every completed epoch;
- `progress.json` is atomically replaced after every epoch and is informational only.

Atomic write protocol: write a same-directory temporary file, flush, `fsync` the file, `os.replace`, then `fsync` the parent directory.

A resume checkpoint contains model state, AdamW state, completed epoch, stale counter, best eligible and unconstrained keys/epochs/states, history, and Python/NumPy/Torch RNG states. Resume is allowed only when the full frozen identity matches exactly: training config hash, runner hash, mixture product, fresh outer product, Step-9A bundle, initialization, seed, architecture, parameter count, execution device, and runtime fingerprint (Python/PyTorch/NumPy versions, Torch thread count, and CUDA runtime/device identity when applicable). Any mismatch is a hard refusal.

If a run resumes, provenance must record optimizer-state/RNG continuation truthfully. This is exact crash recovery, not a fresh optimizer run.

## Additional telemetry

The optimizer is unchanged. Step 10C adds report-only observability:

- wall time per epoch;
- mean and maximum pre-clip gradient norm;
- fraction of optimizer steps clipped;
- mean post-clip gradient norm.

These metrics may motivate a later separately frozen optimization study but may not alter Step 10C while it is running.

## Frozen evaluation

After per-seed checkpoint selection is complete:

- aggregate three seeds by mean logits;
- choose one effect threshold per initialization from concatenated original+bridge **selection** logits only;
- freeze that threshold;
- evaluate the fresh original and bridge outer cohorts independently;
- original bootstrap unit: clean root;
- bridge bootstrap unit: parent group;
- 2,000 bootstrap replicates;
- bootstrap seed: 2026082102;
- 95% confidence interval.

The untouched Step-9A ensemble is evaluated on the same fresh original outer cohort using its already archived deployment effect threshold, solely to define original-domain retention.

## Frozen gates

Bridge mechanism diagnosis must satisfy all three:

- balanced accuracy >= 0.80;
- 95% bootstrap CI lower >= 0.75;
- minimum class recall across RZ/RX/RY >= 0.70.

Original retention must satisfy both:

- effect BA drop versus untouched Step-9A on fresh original outer <= 0.02;
- mechanism BA drop <= 0.02.

All three selected seed checkpoints must be retention-eligible.

Warm-start is preferred if it passes the full gate and scratch does not demonstrate the predeclared clear-superiority condition. Scratch clear superiority requires both initializations to pass, the paired bootstrap CI lower for scratch-minus-warm bridge mechanism BA to exceed +0.01, and scratch not to worsen original mechanism retention by more than 0.01 relative to warm. Otherwise the official decision is `NO_INITIALIZATION_PASSES_FULL_DUAL_DOMAIN_GATE`.

## Claim boundary

Step 10C remains a simulator development benchmark. Passing would support reuse of the current architecture/checkpoint under this expanded simulator coverage and optimization horizon. It would not establish universal circuit optimization, hardware robustness, or QPU advantage. No IBM/QPU job is authorized by this protocol.
