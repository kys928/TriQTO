# Step 7 full development benchmark — result summary

Status: **STEP7_DEVELOPMENT_BENCHMARK_COMPLETE**

Evidence status: **DEVELOPMENT_ARCHITECTURE_SELECTION_NOT_CONFIRMATORY**

This document records the frozen Step 7 neural development benchmark on the accepted Step 5 v3 cohort. The outer development validation set had already been inspected in Step 6 and is not an untouched test set. No confirmatory or generalization claim is made from Step 7.

## Source identity

- Step 5 product: `product_b2d78ad2309b71a55f9bb54f`
- accepted Step 7 smoke: `smoke_185f69415ea6bb082dd93ef7`
- Step 7 benchmark: `benchmark_9989fbf9ef9feeaf283fe23f`
- uploaded benchmark ZIP SHA-256: `sha256:96e5c894165d19b35b776c90b40394856123c417ff5f85de4826c593f03a6e3d`
- experiment-config SHA-256: `sha256:2715c93c0d22e4aff6caf71a5e4c8ae872f5f17c236ab3445808bcefd1895a77`
- execution-config SHA-256: `sha256:0a73fdba5fa40440780bd3187829f29217b18b60b838687a29a76f1675b9b10f`
- runner SHA-256: `sha256:078db079d37bebb02e51900a3dafc4f4c49b15b11a4440e39d95e6d29215b712`
- all nine recorded result-file hashes were independently re-hashed after upload and matched exactly.
- fit / selection / outer-development roots: `3000 / 1000 / 1000`
- model runs: `15`
- hardware executed: **NO**
- historical v0.1 test accessed: **NO**
- spent Phase 15.6 confirmatory cohort accessed: **NO**
- new confirmatory cohort accessed: **NO**
- outer validation used for model/epoch/threshold selection: **NO**

## Primary aggregate results

| Variant | Effect BA | Mechanism BA | Integrated BA |
|---|---:|---:|---:|
| `diagnostic_only` | 0.6814 | 0.4774 | 0.4200 |
| `graph_only` | 0.6100 | 0.3759 | 0.3230 |
| `late_concat` | **0.7128** | **0.5118** | **0.4614** |
| `structured_interaction` | 0.7135 | 0.5052 | 0.4540 |

95% clean-root bootstrap intervals for mechanism BA:

- `diagnostic_only`: `[0.4666, 0.4870]`
- `graph_only`: `[0.3731, 0.3787]`
- `late_concat`: `[0.5011, 0.5229]`
- `structured_interaction`: `[0.4938, 0.5157]`

## Frozen architecture gate

The predeclared Step 7 architecture-specific claim required

`mechanism BA(structured_interaction) - mechanism BA(late_concat)`

with paired clean-root 95% CI lower bound > 0.

Observed paired result:

- BA difference: `-0.0067`
- 95% CI: **`[-0.0120, -0.00165]`**
- macro-F1 difference: `-0.0058`
- 95% CI: `[-0.0110, -0.00064]`

Therefore the architecture-specific gate **FAILS**. On this development cohort the structured interaction is slightly but significantly worse than the matched late-concatenation neural control. No claim that explicit node/pair graph-diagnostic interaction is superior is supported.

The integrated diagnosis comparison is consistent:

- structured minus late-concat integrated BA: `-0.00746`
- 95% CI: `[-0.01198, -0.00296]`.

Effect detection is effectively tied:

- structured minus late-concat effect BA: `+0.00067`
- 95% CI: `[-0.00413, +0.00531]`.

## What *is* supported

The negative architecture result does not erase the broader neural signal.

### Learned graph + diagnostics strongly outperform the cheap Step 6 mechanism baseline

`structured_interaction` vs Step 6A `diag_full_context_graph`:

- mechanism BA gain: **`+0.1027`**
- 95% CI: **`[+0.0926, +0.1126]`**
- mechanism macro-F1 gain: `+0.0989`
- 95% CI: `[+0.0889, +0.1089]`.

`late_concat` is numerically stronger still, reaching mechanism BA `0.5118`.

This supports the development conclusion that a learned nonlinear joint representation of finite-shot diagnostics and circuit information is substantially more useful than the frozen cheap linear/QDA baselines. It does **not** establish that the bespoke structured interaction is necessary.

### Diagnostics carry the dominant mechanism signal; graph-only remains weak

Structured vs diagnostic-only mechanism BA:

- gain: `+0.0278`
- 95% CI: `[+0.0199, +0.0353]`.

Structured vs graph-only mechanism BA:

- gain: `+0.1291`
- 95% CI: `[+0.1186, +0.1394]`.

Graph-only mechanism BA is only `0.3759`, while diagnostic-only reaches `0.4774`. Circuit structure helps when combined with diagnostics, but does not diagnose the injected mechanism well by itself.

### Learned effect pathway improves over the Step 6 SNR rule

Structured vs Step 6B `diag_snr_proxy_threshold`:

- effect BA gain: **`+0.0822`**
- 95% CI: **`[+0.0721, +0.0926]`**
- macro-F1 gain: `+0.0778`
- 95% CI: `[+0.0686, +0.0876]`.

The frozen effect-path noninferiority gate therefore passes comfortably.

## Seed stability

Primary three-seed outer-development mechanism BA:

- diagnostic-only: `0.4701 / 0.4750 / 0.4679`
- graph-only: `0.3759 / 0.3749 / 0.3765`
- late-concat: `0.5042 / 0.4999 / 0.5039`
- structured: `0.4955 / 0.4971 / 0.4982`

The late-concat vs structured ordering is not caused by one pathological seed.

## Predeclared ablations

Ablations were run only at seed `1701`, so they are development diagnostics rather than multi-seed architecture claims.

### Magnitude pathway

Full structured minus `structured_no_magnitude`:

- effect BA: `+0.0104`, CI `[+0.00195, +0.01829]`
- mechanism BA: `-0.00932`, CI `[-0.01585, -0.00270]`.

The explicit magnitude pathway helps the task it was designed for — effect/null — while slightly hurting mechanism at this one seed. This supports keeping effect and mechanism inductive biases conceptually separate rather than forcing magnitude features into the mechanism representation.

### Pairwise diagnostics

Full structured minus `structured_no_pairwise`:

- effect BA: `-0.00983`, CI `[-0.01619, -0.00421]`
- mechanism BA: `-0.00469`, CI `[-0.01128, +0.00218]`.

No positive mechanism contribution from the current pairwise interaction is demonstrated at this seed. Removing pairwise even improves effect BA slightly.

### Global parity

Full structured minus `structured_no_parity`:

- mechanism BA: **`+0.03199`**, CI **`[+0.02350, +0.04038]`**
- integrated BA: `+0.00770`, CI `[+0.00033, +0.01502]`.

This one-seed ablation gives meaningful development evidence that global parity is useful to the structured model for mechanism diagnosis. It is consistent with the earlier Step 4.1 correlation-recovery motivation, but it is not a multi-seed confirmation.

## Important strata

Mechanism diagnosis remains strongly context/noise dependent.

### Shot count

Late-concat mechanism BA:

- 512: `0.4563`
- 1024: `0.5078`
- 2048: `0.5185`
- 4096: `0.5644`

Structured interaction:

- 512: `0.4574`
- 1024: `0.4894`
- 2048: `0.5228`
- 4096: `0.5508`

Higher shots continue to improve mechanism learnability, consistent with Step 5/6 finite-shot-noise findings.

### Perturbation strength

Late concat:

- strength 0.05: `0.4368`
- strength 0.15: `0.5867`

Structured:

- strength 0.05: `0.4332`
- strength 0.15: `0.5771`

Weak perturbations remain substantially harder.

### Circuit family

Late-concat mechanism BA ranges from about `0.351` on hardware-efficient ansatz and `0.393` on phase-interference circuits to `0.703` on QAOA-like and `0.629` on QFT-like circuits. The problem remains strongly family/context dependent.

### Qubit count

Late-concat mechanism BA declines from roughly `0.545` at 2q to `0.470` at 6q and `0.498` at 8q. Structured has a similar pattern. Scaling remains a development concern.

## Step 7 interpretation

Step 7 supports the following development conclusions:

1. the specific bespoke structured graph-diagnostic interaction **did not earn its complexity** over an equally sized late-concat neural control;
2. the matched late-concat model is the strongest tested deployable Step 7 mechanism architecture on this development cohort;
3. learned joint graph + diagnostic representation substantially outperforms the cheap Step 6 mechanism baselines;
4. the learned magnitude/shot-aware effect pathway substantially outperforms the frozen Step 6 SNR threshold;
5. diagnostic evidence is the dominant mechanism source; graph-only is weak, but combining graph/circuit information with diagnostics helps;
6. current pairwise interaction is not supported as beneficial in the one-seed ablation;
7. global parity appears useful for mechanism diagnosis in the one-seed structured ablation;
8. finite-shot noise, weak distortions, circuit family, and qubit count remain major sources of difficulty;
9. Step 7 is development evidence only and cannot support a confirmation/generalization claim.

## Consequence for the next step

Do **not** automatically promote the current `structured_interaction` architecture into a confirmatory test. The frozen interpretation gate explicitly disallows that.

The next development decision should start from the simpler `late_concat` result and ask whether a narrowly motivated architecture revision can beat it without post-hoc metric chasing. Plausible targets from the predeclared evidence are:

- retain the separate magnitude/shot-aware effect pathway;
- preserve global parity evidence;
- reconsider or simplify the current pairwise gated interaction;
- test more targeted geometry-conditioning only if it is frozen against the late-concat control before outcome;
- keep a future new confirmatory cohort untouched until the final architecture is frozen.

`Step 8 automatically unlocked`: **NO**.
