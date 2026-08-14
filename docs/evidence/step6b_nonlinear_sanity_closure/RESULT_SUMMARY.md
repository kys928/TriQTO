# Step 6B nonlinear sanity closure — result summary

Status: **NONLINEAR_SANITY_CLOSURE_COMPLETE**

This document records the controlled Step 6B run on the frozen Step 5 v3 development cohort. Step 6B was chosen **after inspecting Step 6A validation results**, so it is adaptive development evidence, not pre-registered or confirmatory evidence.

## Source identity

- Step 5 product: `product_b2d78ad2309b71a55f9bb54f`
- Step 6A benchmark: `benchmark_383d4c3070350f0bef6fdb23`
- Step 6B closure: `closure_e7de7cdd47142287352f8de8`
- Uploaded result ZIP SHA-256: `sha256:7bbaef238293799b48fb29ff1a23ebff6cca3d14b2fb2fa0092e7175fe8eb985`
- All six result files recorded in `closure_complete.json` were independently re-hashed after upload and matched exactly.
- Historical v0.1 test accessed: **NO**
- Spent Phase 15.6 confirmatory cohort accessed: **NO**
- Hardware executed: **NO**
- TriQTO neural architecture changed: **NO**

## Primary validation results

### Effect detection

| Baseline | BA | 95% CI | macro-F1 |
|---|---:|---:|---:|
| finite-shot diagnostic RMS threshold | 0.5643 | [0.5539, 0.5745] | 0.4976 |
| finite-shot shot-normalized SNR proxy | **0.6313** | **[0.6201, 0.6419]** | **0.5537** |
| finite-shot full diagnostic diagonal-QDA | 0.6042 | [0.5937, 0.6146] | 0.5298 |
| finite-shot full diagnostic + context + graph diagonal-QDA | 0.6047 | [0.5948, 0.6139] | 0.5323 |
| privileged exact diagnostic RMS threshold | **0.9991** | [0.9985, 0.9997] | 0.9980 |
| privileged exact diagnostic diagonal-QDA | 0.9866 | [0.9834, 0.9896] | 0.9633 |

Paired against Step 6A linear `diag_full`:

- RMS threshold BA gain: `+0.0501`, 95% CI `[+0.0348,+0.0657]`;
- shot-normalized SNR proxy BA gain: **`+0.1172`**, CI **`[+0.1027,+0.1320]`**;
- diagonal-QDA BA gain: `+0.0901`, CI `[+0.0753,+0.1048]`.

The exact diagnostic RMS ceiling being almost perfect while the finite-shot SNR proxy is only ~0.63 BA shows that effect detection is strongly limited by finite-shot acquisition noise. It also shows that effect presence is fundamentally a magnitude/uncertainty-aware problem that signed linear coordinates handle poorly.

The SNR proxy improves with shot count: BA is ~0.583 at 512 shots, ~0.610 at 1024, ~0.636 at 2048, and ~0.696 at 4096. It is also substantially easier at strength 0.15 (~0.710 BA) than 0.05 (~0.590 BA).

### Mechanism diagnosis

| Baseline | BA | 95% CI | macro-F1 |
|---|---:|---:|---:|
| graph statistics diagonal-QDA | 0.3633 | [0.3601, 0.3667] | 0.3544 |
| full finite-shot diagnostic diagonal-QDA | 0.3911 | [0.3837, 0.3986] | 0.3784 |
| full finite-shot diagnostic + context + graph diagonal-QDA | 0.3901 | [0.3833, 0.3973] | 0.3822 |
| privileged exact diagnostic diagonal-QDA | **0.4876** | **[0.4808, 0.4949]** | 0.4539 |

Crucially, cheap quadratic nonlinearity does **not** improve mechanism diagnosis over Step 6A linear models:

- finite diagnostic QDA minus linear `diag_full`: BA `-0.0024`, 95% CI `[-0.0118,+0.0069]`;
- context+graph diagnostic QDA minus linear `diag_full_context_graph`: BA `-0.0124`, CI `[-0.0216,-0.0036]`;
- exact diagnostic QDA minus exact linear diagnostic: BA `+0.0022`, CI `[-0.0112,+0.0157]`.

Therefore generic quadratic nonlinearity is not the missing mechanism ingredient. The Step 6A finding that simple graph context adds reproducible mechanism value over finite-shot diagnostics remains the more relevant architectural signal.

Mechanism diagnosis remains easier at strength 0.15 (~0.418 BA for context+graph QDA) than 0.05 (~0.362). It is highly context-dependent across circuit families and qubit counts, reinforcing the need to interpret diagnostic evidence conditionally on circuit structure rather than through one global classifier.

## Integrated diagnosis

- finite-shot `diag_full` diagonal-QDA integrated BA: `0.3227`;
- finite-shot `diag_full_context_graph` diagonal-QDA integrated BA: `0.3199`;
- privileged exact diagnostic diagonal-QDA integrated BA: `0.6048`.

The exact-vs-finite integrated gap is large, again indicating strong acquisition-noise headroom.

## Stratified-report caveat

Effect-detection strata such as `strength=0` / `clean_control` contain only the `no_effect` class. The current generic stratified helper reports a nominal balanced accuracy by assigning zero recall to the absent class. Those single-class stratum BA values are therefore **undefined for scientific interpretation** and must not be used as evidence. This reporting limitation does not affect any overall validation metric, paired root-bootstrap comparison, model-selection decision, or the conclusions in this document.

## Step 6 interpretation

Step 6A + 6B jointly support the following development conclusions:

1. finite-shot `B_delta` contains real learnable mechanism information above chance;
2. simple circuit geometry adds reproducible mechanism value beyond finite-shot diagnostics under the linear benchmark;
3. pairwise/parity information is not cleanly exploited by a global linear classifier, despite Step 4.1 identifiability evidence;
4. effect detection should use an explicit magnitude/shot-aware path rather than relying on signed diagnostic coordinates alone;
5. generic diagonal quadratic nonlinearity does not solve mechanism diagnosis;
6. exact diagnostic ceilings remain substantially better than finite-shot mechanism models, so shot noise is a major bottleneck;
7. the next architecture should test **structured graph-diagnostic interaction**, not merely a larger generic MLP/QDA.

## Step 7 consequence

Step 7 should preserve separate roles:

- an effect/null/uncertainty pathway with explicit magnitude and shot metadata;
- a mechanism pathway trained only where `mechanism_loss_mask=true`;
- a dedicated diagnostic encoder for signed local/pairwise/parity deltas;
- circuit-graph conditioning/fusion that can interpret those diagnostics relative to circuit geometry;
- ablations that distinguish `diagnostic only`, `graph only`, `diagnostic + graph`, and structured interaction from generic nonlinear capacity.

No Step 6 metric is a confirmation of TriQTO architecture quality. Step 6 is development evidence used to define what Step 7 must beat.