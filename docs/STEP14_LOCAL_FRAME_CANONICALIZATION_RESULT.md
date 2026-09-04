# Step 14 local-frame canonicalization — frozen post-hoc result

Status: **FROZEN POST-HOC RESULT**  
Date: 2026-09-04

This document records the completed Step-14 local-Pauli-frame canonicalization diagnostic. It freezes the result and its claim boundary before the subsequent oracle-free latent-location experiment.

## Immutable scientific identity

- Diagnostic ID: `local_frame_540d5f72552356577c252c2e`
- Diagnostic status: `COMPLETE_FROZEN_LOCAL_FRAME_CANONICALIZATION`
- Result SHA-256: `sha256:42b03cd293d4d623b1458f558154921778824a70564e52b16a0bb3bf58e570f5`
- Completion SHA-256: `sha256:7495decc59e1b6a6cb6eb88db0738f18f93c51faca949c4d3e6524f6847decd2`
- Frozen Step-14 training run: `training_18e0b4ed6e685af30b6c4a35`
- Selection-freeze SHA-256: `sha256:af7ffaece77d16134c33b1c898afb7165c7ace5dcfccfea23f713ae242e4ed6f`
- Development product: `development_b087edfd6629ac250299391d`
- Development dataset SHA-256: `sha256:c471919d2c2a8ca2c206228ac59774f5708fc5182c17bbfb48eca6ecf9491280`
- Protocol config SHA-256: `sha256:fedb862d1baff9ecc9269530e38e6f965c37427bcb2182dcadccabfb101b75ab`
- Network-volume pointer: `triqto-data/step14_local_frame_canonicalization/current_local_frame_canonicalization.json`
- Result-read Actions run: `33851494683`

Machine-readable freeze: `data/manifests/step14_local_frame_canonicalization_result_20260904.json`.

## Frozen protocol boundary

The diagnostic used only the already-materialized Step-14 development cohort: 600 FIT families / 28,800 injected FIT examples and 150 selection families / 7,200 injected selection examples. Selection was evaluation-only. The main TriQTO checkpoint was never updated. Simulator outer, future-hardware reserve, and QPU data were not accessed.

The canonical-frame calculation was intentionally privileged analysis. It used the generator's true affected qubit and true injection boundary and computed an exact first-order Born-response Jacobian with simulator statevector access. Those privileged fields were never written into the deployable model input.

Therefore this result proves a representation/invariance phenomenon under an analysis oracle. It **does not prove hardware-facing deployability**.

## Frozen outcome

Selection mechanism balanced accuracy:

| representation | BA |
|---|---:|
| raw diagnostics | 0.494028 |
| raw + affected-qubit oracle | 0.462917 |
| raw + affected-qubit + raw local insertion context | 0.484444 |
| deterministic canonical-coordinate rule | 0.755417 |
| canonical local frame + small fixed probe | **0.910556** |
| canonical local frame + high-capacity probe | 0.910139 |

Canonical small-probe mechanism recalls were `[0.901667, 0.907083, 0.922917]`; minimum recall was 0.901667. The deterministic canonical rule, with no learned nonlinear classifier, reached recalls `[0.703750, 0.705417, 0.857083]`.

Paired family-bootstrap effects, 1,000 replicates over the 150 selection families:

- canonical small probe minus raw-neighborhood oracle: **+0.426111**, 95% CI `[+0.403611, +0.450306]`;
- canonical small probe minus raw diagnostics: **+0.416528**, 95% CI `[+0.394028, +0.436399]`;
- deterministic canonical rule minus raw diagnostics: **+0.261389**, 95% CI `[+0.238333, +0.287639]`;
- high-capacity canonical probe minus small canonical probe: `-0.000417`, 95% CI `[-0.003611, +0.003194]`.

The predeclared local-frame support gate was therefore passed by a very large margin. The formal verdict is:

`SUPPORTED_LOCAL_FRAME_CANONICALIZATION_SIGNAL`

The raw-evidence limitation update remains:

`EVIDENCE_LIMIT_NOT_RESOLVED`

## What this establishes

The same finite-shot Born diagnostic bundle that gives about 0.49 mechanism BA in raw coordinates gives about 0.91 BA across unseen circuit families after exact local Pauli-frame canonicalization. The deterministic frame rule alone reaches about 0.76. A much larger neural probe provides essentially no improvement once the evidence is canonicalized.

The supported interpretation is therefore that a dominant Step-14 mechanism-generalization bottleneck is **circuit-dependent quantum-frame / invariance handling**, rather than a simple final-head, late-fusion, generic-capacity, or raw-neighborhood problem.

This result must not be rewritten as a claim that the current deployable TriQTO model already performs this transformation. It does not. The analysis supplies the correct location/frame.

## Remaining identifiability structure

The exact frame geometry is not uniformly well-conditioned:

- 25.5% of roots have a minimum pairwise response-axis angle below 15 degrees;
- 44.03% have minimum singular value below 0.1;
- median minimum axis angle is 42.02 degrees;
- median minimum singular value is 0.2604;
- median condition number is 5.31;
- median canonicalization residual fraction is 0.5792.

Thus some circuit contexts remain locally ambiguous or poorly conditioned under the finite-shot measurement bundle. The 0.91 result shows that this does not impose a universal ~0.5 mechanism ceiling, but it does not eliminate context-specific identifiability limits.

## Frozen next question

The next experiment removes the location oracle without changing the main model. For each example it will construct a candidate set of plausible `(qubit, insertion-boundary)` frames using only the known circuit and model-visible finite-shot diagnostics, score or marginalize over those candidates, and infer mechanism while treating location as latent.

The true affected qubit and true injection boundary are forbidden from candidate generation, scoring, fitting, hyperparameter selection, and mechanism prediction. They may be read only after prediction to audit localization accuracy.

A positive result would show that the analysis oracle can be removed at the **latent-location inference** level. It still would not by itself prove scalable hardware deployment if the candidate-frame calculator uses exact statevector simulation; that remaining distinction must be preserved explicitly.
