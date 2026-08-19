# Step 9D post-hoc raw training-coverage analysis

## Scientific boundary

This is a post-hoc analysis of the already-frozen Step-9D exploratory IBM-QPU pilot. It performs no QPU submission, retraining, weight update, threshold change, or confirmatory interpretation.

The analysis compares the Step-9D pilot diagnostic vectors against the exact frozen Step-5 v3 product `product_b2d78ad2309b71a55f9bb54f`.

Distance is Euclidean distance after per-subset feature standardization, excluding dimensions whose Step-5 training standard deviation is <= 1e-10. OOD percentile is the query nearest-neighbor distance relative to Step-5 leave-one-out nearest-neighbor distances in the same standardized subset.

The exact full `training_coverage_v1.json` generated in the analysis environment has SHA-256:

`58a331b351f4bd9c9a1eceb411d408789e65ef34dcd875b69ef6cf8ea8acca09`

`TRAINING_COVERAGE_SUMMARY.json` records the compact result needed for repository review.

## Main result

The Step-9D phase-family mechanism inversion is already visible in raw diagnostic space before any neural encoder, fusion layer, or mechanism head is applied.

For the exact ideal Step-9D phase diagnostics compared with the exact Step-5 audit diagnostics:

| True Step-9D mechanism | Nearest Step-5 label | Top-25 majority | Correct-label nearest rank | OOD percentile |
|---|---|---:|---:|---:|
| RZ drift | RY overrotation | RY 20/25 | 101 | 92.58% |
| RX overrotation | RZ drift | RZ 21/25 | 8 | 60.43% |
| RY overrotation | RZ drift | RZ 25/25 | 39 | 70.34% |

This is exactly the same label mapping observed in the frozen Step-9D deployment ensemble:

- true RZ -> predicted RY;
- true RX -> predicted RZ;
- true RY -> predicted RZ.

The finite-shot hardware diagnostics compared with finite-shot Step-5 inputs preserve the same nearest-neighbor mapping for all three phase cases.

Therefore a neural-network failure is not required to create the observed inversion. The frozen model is largely consistent with the local mechanism geometry already present in the Step-5 diagnostic distribution for these Step-9D phase signatures.

## What this does and does not establish

This substantially strengthens the training-distribution/context-shift explanation. It does not prove that the Step-5 phase generator mismatch is the unique causal source, and it does not prove that standardized Euclidean distance is identical to the metric learned by the deployed network.

OOD distance alone is also insufficient as an explanation. For example, Bell RY has a high ideal/exact OOD percentile (~86.39%) while its nearest neighbor and all 25 nearest examples still carry the correct RY label. What matters in the phase failure is the conjunction of distribution displacement and systematic wrong-class neighborhood alignment.

The controls are not perfectly separable in raw space either. Bell RX and GHZ RY contain local class ambiguity. This is consistent with earlier evidence that mechanism identification is intrinsically harder than effect detection. The phase family is distinguished by the fact that all three Step-9D mechanism cases show a coherent wrong mapping that reproduces the frozen model's hardware errors.

## Working diagnosis after this analysis

The strongest current diagnosis is:

1. Step-9D hardware acquisition preserved the expected physical diagnostic signatures.
2. Exact ideal Step-9D phase diagnostics produce the same frozen-model misclassifications as hardware diagnostics.
3. Zeroing the graph embedding does not recover the labels.
4. The raw Step-5 diagnostic neighborhood itself reproduces the same RZ->RY, RX->RZ, RY->RZ mapping.
5. Therefore the dominant failure is upstream of the deployed classifier decision: the Step-9D phase circuit occupies a diagnostic/context regime whose mechanism-label geometry differs from the Step-5 training regime.

The Step-5 `phase_interference` generator is structurally different from the Step-9D pilot phase circuit, so circuit/context distribution shift remains the leading explanation, but direct causal attribution still requires a context-controlled simulator replay.

## Recommended next test

Before retraining or using additional QPU time, run an offline context-controlled simulator replay.

The test should hold intervention mechanism and strength fixed while varying only the clean circuit context between:

- the exact Step-9D phase circuit;
- representative Step-5 phase circuits/contexts near the nearest-neighbor regions.

It should trace exact signed diagnostic vectors across RZ/RX/RY and intervention positions. The goal is to determine whether changing circuit context alone transports the mechanism signatures into the observed wrong-class neighborhoods.

If that replay reproduces the mapping, the causal failure mode is a context-dependent mechanism-label geometry not sufficiently covered by Step-5. That would justify redesigning the simulator training distribution before any new hardware confirmation. If it does not, the next target is the learned representation geometry.
