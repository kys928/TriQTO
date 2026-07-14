# Phase 15 scope

Phase 15 evaluates one completed Phase 14 checkpoint on the untouched Phase 12 `test` split. It produces immutable per-item evidence, aggregate generalization tables, calibration summaries, inference-time stream ablations, and optional Phase 10 baseline comparisons.

The scientific boundary is strict:

- only `test` records are evaluated;
- no optimizer, gradient, scheduler, or checkpoint mutation is permitted;
- validation data is not reused for Phase 15 reporting;
- topology remains an audit/input stream with `topology_loss_weight = 0.0`;
- hardware-masked simulation remains simulator-derived evidence;
- no real hardware execution occurs;
- no universal-correction or quantum-advantage claim is introduced.

Phase 15 reports measured held-out performance. It does not decide that TriQTO is superior merely because a metric improves; baseline privileges, sample coverage, support size, and subgroup counts remain explicit.
