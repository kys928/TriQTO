# Phase 15 scope

Phase 15 evaluates one completed Phase 14 checkpoint on untouched Phase 12 `test` rows and publishes immutable per-item evidence, aggregates, uncertainty diagnostics, inference-time stream ablations, and optional Phase 10 comparisons.

The boundary is strict:

- `iid_test` means an untouched partition, not out-of-distribution generalization;
- `ood_axis_holdout` requires exact, audited family, qubit-count, distortion-type, or later backend disjointness across development and test;
- unidentifiable diagnosis/action labels are excluded and reported;
- each basis-conditioned distribution is scored separately;
- the uncertainty head is evaluated directly, while softmax confidence remains a separate descriptive signal;
- baseline identities are task-qualified and their information privileges remain visible;
- no optimizer, gradient, scheduler, checkpoint mutation, hardware execution, topology loss, universal-correction claim, or quantum-advantage claim is permitted.

The repository contains no committed trained checkpoint or empirical Phase 15 result. The engine and contracts are executable; scientific performance claims require a real run with sufficient coverage.
