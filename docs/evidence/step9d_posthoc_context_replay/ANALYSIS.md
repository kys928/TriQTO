# Step 9D post-hoc matched-context replay

Scientific boundary: post-hoc only. No QPU access, retraining, weight change, threshold change, or confirmatory interpretation.

## Frozen artifact identity

The exact replay artifact supplied after running `scripts/v0_2/analyze_step9d_posthoc_context_replay.py` is `context_replay_v1.json`.

SHA-256:

`16e6a53fb31538edf11ea991e9bbb41de94a04fe4cabf2e2cec68fc3ead81915`

The artifact contains 116 fully supervised matched Step-5 phase-interference contexts with:

- 2 logical qubits;
- affected qubit 0;
- intervention strength 0.15;
- identical clean root and insertion boundary within each RZ/RX/RY triplet.

Depth coverage is late=52, middle=30, terminal=34.

## Result

The Step-9D ideal phase mechanism signature is not invariant across the matched Step-5 contexts.

- Query RZ: correct-context fraction 26/116 = 0.2241. Winner counts: RX 50, RY 40, RZ 26. However, the median winner margin is approximately 9.28e-16 and all median candidate cosines are approximately zero. Therefore the RZ vote counts are treated as numerically fragile and are not used as evidence of a stable remapping.
- Query RX: correct-context fraction 32/116 = 0.2759. Winner counts: RY 43, RZ 41, RX 32. Median winner margin = 0.377706. Late contexts favor RY (26/52); middle contexts favor RZ (17/30).
- Query RY: correct-context fraction 37/116 = 0.3190. Winner counts: RZ 51, RY 37, RX 28. Median winner margin = 0.397200. Late contexts favor RZ (28/52).

The replay does **not** establish a deterministic global permutation such as RZ->RY and RX/RY->RZ. It does establish strong context dependence: the same physical intervention axis can occupy substantially different observable directions depending on the clean circuit and insertion location, before any learned representation or classifier is applied.

This strengthens the interpretation that global mechanism labels are too brittle in the present Born-diagnostic coordinate system. The next test is therefore a context-conditioned identifiability audit using a local RZ/RX/RY response frame for each circuit/insertion context.

## Erratum

Earlier post-hoc discussion referenced `src/triqto/circuits/phase_interference.py` as if it generated the Step-5 training data and therefore described the Step-5 phase family as having no entangler. That statement was incorrect.

Step-5 actually used the dedicated `build_clean_circuit()` implementation in `scripts/v0_2/generate_step5_matched_diagnostic_training_dataset.py`. Its `phase_interference` branch includes a CZ chain.

The correction changes the specific structural description, but it does not remove the observed Step-5-vs-Step-9D context distribution shift or the matched-context replay result above.
