# Phase 14 scope

Implement the training engine on top of Phase 12 views and the Phase 13 model while preserving strict masking, deterministic identities, immutable artifacts, and `topology_loss_weight = 0.0`.

Born-distribution losses are complete per-graph distances, gradient accumulation is item-normalized for every full or partial window, and the output root must not overlap the Phase 12 or optional Phase 7 source trees.
