# Step 9C hardware-path dry-run result

Status: **PASS — plumbing / semantic-equivalence gate only**

Run: `dryrun_e2af1f01746a772332ae2255`

Deployment bundle: `deploy_ac536a74b2f8dd571d353a12`

Uploaded dry-run ZIP SHA-256:

`sha256:95d63333fb49f5c4dea12f27c07cd89b1fa115577dec63e416a9f0d0072c7c86`

`dry_run_complete.json` SHA-256:

`sha256:3cbfebde28efc9be67c7bae2d6bc9544c37ab38346c5c981f2e84e3e01a2f1de`

## Frozen identities

- schema: `triqto.v0_2.step9c_hardware_path_dry_run.v1`;
- backend class: `GenericBackendV2`;
- sampler class: `SamplerV2` (Aer);
- shots per program: 1024;
- checkpoint hashes:
  - seed 1701: `sha256:549a64c0be5a6612a2a67b9c60fe20a011a81cff2f3c00d0759e8d0a91e8e42d`;
  - seed 1702: `sha256:53339d86313a73358fee0ecbae7874cf8f9cac67dc5f1e34618f7f8f32ed4de7`;
  - seed 1703: `sha256:b5f4a7f876816449da8c38948e81b439d9144589b28101c553566c126708dabc`.

The independently audited ZIP contained four files. The three hashes recorded by `dry_run_complete.json` were independently recomputed and all matched:

- `case_results.jsonl`: `sha256:508e9bbbd2474475c5f505e5059889745e1387b08a4a7a415db6396a32baa65b`;
- `dry_run_summary.json`: `sha256:68b238a9499fe34ea66cb7f2006205403bbe25043b1613576820274d5da5a8a2`;
- `inference_predictions.csv`: `sha256:bf11ce9d85fc4bdd8c2612ea81af6e7b6cc1f9d17bea216f0e48935278dc5f6a`.

## Gate outcome

Every frozen Step-9C gate passed:

- Step-9B identity verified;
- deployment checkpoint hashes verified;
- all six programs executed for every case;
- `meas` survived transpilation;
- realized shots equaled requested shots;
- hardware-path tensors exactly matched tensors reconstructed through the original Step-7 training adapter;
- all three frozen checkpoints loaded with the expected parameter count;
- all model outputs were finite;
- prediction correctness was not used as a gate.

No physical QPU was executed. No training, model change, or deployment-threshold change occurred.

## Report-only predictions

The three predictions were deliberately outside the pass/fail contract:

| case | injected mechanism | effect prediction | mechanism prediction | effect probability |
|---|---|---:|---|---:|
| `bell_rz_q0` | RZ | false | RX | 0.3953 |
| `ghz_rx_q1` | RX | true | RX | 0.5372 |
| `phase_ry_q0` | RY | true | RZ | 0.5593 |

Thus only one of the three report-only mechanism guesses matched the injected mechanism. This does **not** invalidate Step 9C because Step 9C was explicitly a plumbing / semantic-equivalence gate, not a performance experiment. It is, however, a warning against interpreting a tiny first hardware sample as model confirmation.

## Interpretation

Step 9C establishes that the actual hardware-facing software path preserves the model-input semantics tested during development:

`ISA transpilation → Sampler counts → Qiskit bit-order correction → observed-minus-reference local/pair/parity diagnostics → DiagnosticTensorBatch → frozen late_concat ensemble`

It does not establish physical-QPU transfer performance.

The next allowed stage is one explicitly exploratory physical-QPU pilot. Any physical-QPU result must be recorded separately and must not be relabeled as confirmatory evidence.
