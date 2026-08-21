# Step 9D post-hoc context-conditioned identifiability audit

Scientific boundary: post-hoc only. No QPU submission, retraining, weight change, threshold change, or confirmatory interpretation.

## Frozen artifact identity

The exact audit artifact supplied after running `scripts/v0_2/analyze_step9d_posthoc_context_identifiability.py` is `context_identifiability_v1.json`.

SHA-256:

`2e1f314eb14217c736785aede3c418a8303109299ab5b1178e9f5c1b5e4d245d`

The audit uses 116 fully supervised matched Step-5 `phase_interference` contexts with 2 qubits, affected qubit 0, intervention strength 0.15, and complete RZ/RX/RY triplets.

## Local-frame geometry

The operational local response frame is defined by exact same-context RZ/RX/RY response vectors at strength 0.15. It is an operational finite-strength frame, not an infinitesimal mathematical tangent.

Across the 116 contexts:

- median minimum pairwise angle = 66.72 degrees;
- p10 minimum pairwise angle = 6.18 degrees;
- minimum observed pairwise angle = 1.80 degrees;
- fraction of contexts with minimum pairwise angle below 15 degrees = 0.1293;
- fraction below 30 degrees = 0.1466;
- exact self-classification is 116/116 for each mechanism.

This shows that most contexts have well-separated local mechanism directions, while a minority contain intrinsically close local directions, especially RX/RY in some contexts.

## Step-5 finite-shot local-oracle result

Predeclared exploratory gate:

- balanced accuracy >= 0.80;
- cluster-bootstrap 95% lower bound >= 0.75;
- minimum mechanism recall >= 0.70.

Cosine decoder:

- balanced accuracy = 0.8736;
- cluster-bootstrap 95% CI = [0.8362, 0.9080];
- recalls: RZ=0.9655, RX=0.7759, RY=0.8793;
- median classification margin = 0.5140;
- 4096-shot subset balanced accuracy = 0.9615;
- gate: PASS.

Euclidean decoder:

- balanced accuracy = 0.9310;
- cluster-bootstrap 95% CI = [0.9023, 0.9598];
- recalls: RZ=0.9569, RX=0.8793, RY=0.9569;
- median classification margin = 0.0947;
- 4096-shot subset balanced accuracy = 0.9872;
- gate: PASS.

## Frozen Step-9D phase QPU cases

The same simulator-privileged exact local frame was applied to the frozen Step-9D phase QPU diagnostics.

Both passing decoders classify all three cases correctly:

- RZ -> RZ;
- RX -> RX;
- RY -> RY.

Cosine own-direction alignments are 0.9889 (RZ), 0.9412 (RX), and 0.9429 (RY), with margins 0.9565, 0.8596, and 1.1267 respectively.

The frozen deployed model had instead predicted RZ->RY, RX->RZ, RY->RZ for these same three cases.

## Decision gate

Status:

`LOCAL_RESPONSE_FRAME_SUFFICIENT_IN_TARGETED_AUDIT__SIMULATOR_PRIVILEGED`

Interpretation:

1. In this targeted audit, the six-program Born diagnostics contain enough information for strong mechanism identification when interpreted relative to the correct same-context local response frame.
2. The earlier phase failure is therefore not explained by information destruction in the measured diagnostics alone.
3. The current deployed global mechanism mapping is the wrong coordinate system for this context-sensitive problem.
4. This does **not** yet demonstrate a hardware-deployable solution because the local response frame is generated from simulator-privileged exact mechanism responses.

## Hard next action

Do not retrain yet.

The next research step is to design and test a hardware-valid approximation to the local response frame using only deployable circuit/context information and already-available diagnostic evidence. A hardware-deployable frame has not yet been demonstrated.
