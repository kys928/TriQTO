# Step 14 implementation boundary

This implementation realizes the already-frozen Step-14 protocol without changing its scientific design.

The implementation adds four runners: development/simulator-outer cross-motif generation, warm-start fit/selection training with pre-outer checkpoint/threshold freeze, fresh legacy-retention outer generation, and one-shot outer evaluation. It also adds implementation contract tests.

Before training, only cross-motif fit and selection families may be materialized. Simulator outer requires the frozen selection marker. The future-hardware reserve cannot be materialized by these Step-14 runners. Step 14 contains no QPU execution path.

No architecture, loss, mechanism classes, training seeds, optimizer, learning rate, gradient-clipping threshold, family split, outer gate, or bootstrap policy is changed by this implementation.
