# Step 9F hard next action

Frozen after receipt of `frame_failure_decomposition_v1.json`.

Do not retrain TriQTO and do not submit a new QPU job.

Step 9F established two separate facts:

1. The hardware-valid known-axis probe route is physically/geometrically viable in simulation once the SNR budget is sufficient; Euclidean decoding first passes the inherited full gate at probe strength 0.15 and 2048 shots per basis per axis.
2. The frozen Step-9D phase QPU circuit is extreme out-of-domain relative to the Step-5 circuit-only feature support, and alternative neighbor transfer does not rescue that missing support.

Therefore the next audit is a zero-QPU **deployment-domain simulator coverage bridge**. It must add simulator response-frame examples around the Step-9D phase circuit motif without including the exact frozen pilot circuit, fit the same circuit-only estimator, and evaluate on held-out bridge circuits plus the untouched frozen Step-9D phase QPU query.

If deployment-domain coverage rescues the circuit-only frame, the next model-design decision should favor dataset/domain augmentation and reuse of compatible checkpoint weights rather than assuming a new architecture is required.

If deployment-domain coverage does not rescue the circuit-only frame, the known-axis probe route remains the evidence-backed hardware-valid fallback, subject to a separately frozen bounded-QPU protocol.
