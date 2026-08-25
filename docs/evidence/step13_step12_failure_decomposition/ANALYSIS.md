# Step 13 — Step-12 failure decomposition

Status: **POSTHOC DIAGNOSTIC ANALYSIS; NO NEW QPU ACCESS; NO TRAINING**

## Question

Step 12 failed the predeclared cross-backend mechanism-generalization gate even though Step-10C detected an effect in all 18 distorted cases. This decomposition asks which explanation is better supported by the already-acquired Step-12 data:

1. the hardware-facing RZ/RX/RY evidence collapsed or became non-identifiable on `ibm_marrakesh`; or
2. the evidence remained mechanism-specific, but the learned representation/classifier did not generalize its evidence-to-label mapping to the new circuit contexts.

This is a posthoc failure analysis. It does not alter the Step-12 pass/fail outcome.

## Source integrity

The source execution bundle is `step12plan_24b2631b5cd94c58af951492.zip`, SHA256 `9c9a4a726e0403a466960669917a394ef7df73c2f6fb12239dbdf799f818d7ff`. ZIP CRC validation passes. The frozen Step-12 result remains 8/18 Step-10C mechanism correctness versus 9/18 for the report-only Step-9A baseline, with the predeclared gate failed.

## 1. The ideal signatures were identifiable before QPU access

The frozen model-blind statevector audit passed before Step-12 planning. At the low strength 0.13, the weakest ideal local-Bloch distortion norm was `0.0736908542`, above the frozen `0.04` floor, and the weakest pairwise mechanism distance was `0.1181629573`, above the frozen `0.10` floor.

This excludes the trivial explanation that the test was built from ideal mechanisms that were already indistinguishable.

## 2. The physical QPU evidence preserved the ideal mechanism geometry

For each of the 18 distorted cases, the analysis reconstructs the exact deployable local Pauli-diagnostic vector from the frozen X/Y/Z reference and observed counts. It compares that QPU vector with the corresponding ideal statevector local-Bloch delta for the same frozen motif, mechanism, and strength.

Across all 18 cases:

- mean cosine between the QPU vector and its true ideal mechanism template: **0.9534**;
- median cosine: **0.9707**;
- minimum cosine: **0.7909**.

The geometry is also preserved across mechanism pairs. Across the 18 within-motif/within-strength pairwise mechanism distances:

- ideal-vs-QPU distance Pearson correlation: **0.9664**;
- Spearman correlation: **0.9009**;
- median QPU/ideal pairwise-distance ratio: **1.0628**;
- observed ratio range: **0.8715–1.2942**.

Therefore the mechanism signatures did not simply collapse into one another on the physical device.

## 3. A posthoc physics-template diagnostic recovers all 18 labels

As a diagnostic only, each QPU local Pauli-delta vector was compared with the three ideal mechanism templates for the already-known motif and strength. Nearest-template Euclidean assignment recovered **18/18** injected mechanism labels.

A parametric multinomial shot-noise bootstrap was then performed from the acquired 4096-shot empirical distributions, with 1000 replicates per case and frozen RNG seed 20260825. The mean probability that the nearest-template diagnostic retained the correct mechanism was **0.9982**; the weakest individual case was **0.983**.

This oracle is **not** a deployable model and **not** a confirmatory metric. It uses known motif, strength, and ideal injection templates after the outcome. Its purpose is narrower: to test whether the acquired physical evidence contains enough structured information to distinguish the three controlled mechanisms. It does.

## 4. The learned model fails where the evidence remains structured

Step-10C mechanism correctness was strongly motif-dependent:

| motif | Step-10C |
| --- | ---: |
| `cz_echo_ramsey` | **0/6** |
| `dual_arm_recombination` | **3/6** |
| `three_qubit_phase_fanout` | **5/6** |

By mechanism:

| mechanism | Step-10C |
| --- | ---: |
| `rz_drift` | 3/6 |
| `rx_overrotation` | **1/6** |
| `ry_overrotation` | 4/6 |

Yet the QPU evidence for those same failed cases remains close to the corresponding ideal mechanism direction. In particular, the six `cz_echo_ramsey` cases are all misclassified by Step-10C even though the posthoc physics-template diagnostic assigns all six correctly.

Step-10C and Step-9A also produce the same mechanism class on **16/18** Step-12 distorted cases. This shows that the Step-10 bridge intervention did not materially move the decision behavior across most of these new motifs.

## 5. Decomposition conclusion

The evidence does **not** support hardware-evidence collapse as the primary explanation for the Step-12 failure. The acquired Pauli evidence remains strongly mechanism-specific and retains the ideal controlled-perturbation geometry.

The evidence instead supports **out-of-distribution learned mapping/generalization failure as the primary explanation**. The model is detecting the perturbations, but its mapping from graph-conditioned diagnostic evidence to RZ/RX/RY identity does not generalize robustly to the new Step-12 motif context.

This is encouraging for a training/generalization repair, but it does not yet identify the exact architectural subcomponent at fault. The current decomposition cannot distinguish among:

- graph-context over-conditioning;
- insufficient diversity in the diagnostic encoder's training support;
- graph/diagnostic fusion failure;
- mechanism-head decision-boundary failure;
- interactions among these components.

## 6. Required next no-QPU test before retraining

Before changing weights, run an **ideal-diagnostic counterfactual replay** of the frozen Step-10C and Step-9A models on the exact Step-12 graphs/layouts. Replace only the hardware diagnostic tensors with exact statevector-derived diagnostic tensors while leaving model weights, graph inputs, layouts, shot metadata, and thresholds frozen.

Interpretation:

- if Step-10C still misclassifies the Step-12 motifs under ideal diagnostics, the failure is decisively a learned graph/context/generalization problem rather than hardware noise;
- if ideal diagnostics largely restore mechanism correctness, hardware-domain distortion is materially involved even though the raw evidence remains separable.

No QPU run is required for this replay, and no training should happen before its outcome is frozen.
