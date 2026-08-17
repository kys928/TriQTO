# Step 7.1 final parity-residual revision — result summary

Status: **STEP7_1_FINAL_REVISION_COMPLETE**

Evidence status: **ADAPTIVE_DEVELOPMENT_FINAL_ARCHITECTURE_REVISION_NOT_CONFIRMATORY**

This document records the one-shot Step 7.1 final architecture revision on the already-inspected Step 5 development cohort. Step 7.1 was frozen before outcome and explicitly forbids any Step 7.2 architecture search on this cohort.

## Source identity and integrity

- Step 5 product: `product_b2d78ad2309b71a55f9bb54f`
- source Step 7 benchmark: `benchmark_9989fbf9ef9feeaf283fe23f`
- Step 7.1 benchmark: `benchmark_acdea8dffa258412b7566d83`
- uploaded benchmark ZIP SHA-256: `sha256:7a36cb3499c025ad7923c5473608df680db73f01a82ddb64e69916e01de03b3a`
- all nine result-file hashes recorded by `benchmark_complete.json` were independently re-hashed after upload and matched exactly.
- fit / selection / outer-development roots: `3000 / 1000 / 1000`
- primary seeds: `1701, 1702, 1703`
- model runs: `7`
- hardware executed: **NO**
- historical v0.1 test accessed: **NO**
- spent Phase 15.6 confirmatory cohort accessed: **NO**
- new confirmatory cohort accessed: **NO**
- outer validation used for selection: **NO**

## Primary aggregate results

| Variant | Effect BA | Mechanism BA | Integrated BA |
|---|---:|---:|---:|
| `late_concat` | 0.7085 | **0.5059** | 0.4569 |
| `late_concat_parity_residual` | **0.7129** | 0.5041 | 0.4569 |
| frozen Step-7 `late_concat` reference | 0.7128 | **0.5118** | 0.4614 |

## Frozen decision gates

The candidate could replace the champion only if all three pre-frozen conditions passed.

### 1. Champion reproducibility — PASS

Re-run `late_concat` mechanism BA: `0.50594`.

Frozen Step-7 `late_concat` mechanism BA: `0.51176`.

Absolute difference: about `0.00582`, inside the frozen `0.01` tolerance.

The paired re-run-minus-frozen mechanism BA bootstrap difference is about `-0.00587`, 95% CI `[-0.01049,-0.00111]`. This is a small reproducibility shift but remains inside the predeclared absolute point-estimate tolerance.

### 2. Parity-residual mechanism architecture signal — FAIL

Candidate minus re-run champion mechanism BA:

- mean paired bootstrap difference: **`-0.00170`**
- 95% CI: **`[-0.00622,+0.00310]`**
- macro-F1 difference: approximately `+0.00003`, CI `[-0.00459,+0.00484]`.

The CI lower bound is not greater than zero. The parity residual does not earn an architecture-specific mechanism improvement over late concat.

### 3. Effect non-inferiority — PASS

Candidate minus champion effect BA:

- mean paired bootstrap difference: `+0.00442`
- 95% CI: `[-0.00037,+0.00909]`.

The lower bound is comfortably above the frozen `-0.005` non-inferiority margin.

## Final architecture decision

Because the mechanism-superiority gate failed, the candidate does not win.

**Selected final development architecture: `late_concat`.**

Interpretation flags from the completion marker:

- champion rerun reproducible: **YES**
- candidate mechanism architecture signal: **NO**
- candidate effect non-inferior: **YES**
- candidate wins final revision: **NO**
- architecture search stops after Step 7.1: **YES**
- new untouched confirmatory cohort required: **YES**.

There is no Step 7.2 architecture search on this development validation cohort.

## Parity ablation

The predeclared seed-1701 diagnostic ablation compares the champion against `late_concat_no_parity`.

Champion minus no-parity mechanism BA:

- mean paired bootstrap difference: **`+0.02581`**
- 95% CI: **`[+0.01705,+0.03414]`**
- macro-F1 difference: `+0.02247`, CI `[+0.01391,+0.03063]`.

Integrated BA also improves with parity by about `+0.01255`, CI `[+0.00551,+0.01920]`.

Effect BA difference is small and inconclusive: about `+0.00324`, CI `[-0.00368,+0.01038]`.

This supports the development conclusion that global parity is useful evidence inside late concat, while the extra graph-conditioned parity residual tested in Step 7.1 is unnecessary.

## Seed-level mechanism BA

`late_concat`:

- seed 1701: `0.49650`
- seed 1702: `0.50424`
- seed 1703: `0.49270`

`late_concat_parity_residual`:

- seed 1701: `0.49983`
- seed 1702: `0.50342`
- seed 1703: `0.49443`

The aggregate failure of the candidate is not caused by one catastrophic seed; the two models remain close across all three seeds.

## Scientific consequence

Step 7.1 closes architecture selection on this development cohort.

The final development architecture is the simpler `late_concat` model with:

- dedicated signed `DiagnosticTensorBatch -> DiagnosticEncoder` input boundary;
- generic learned graph + diagnostic late fusion;
- global parity retained inside the diagnostic evidence;
- pairwise diagnostics retained in the generic diagnostic encoder but no special pair-endpoint gating;
- explicit magnitude / shot-aware effect pathway;
- mechanism supervision only under `mechanism_loss_mask`.

The next scientific stage is **not** another architecture iteration. It is to generate and freeze a genuinely new untouched confirmatory cohort, then evaluate the already-selected final architecture under a protocol frozen before any confirmatory outcome is inspected.
