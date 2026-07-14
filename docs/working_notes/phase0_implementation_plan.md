# Phase 0 implementation plan

1. Establish CPU-safe direct dependency pins and a non-overlapping optional GPU profile without claiming a complete transitive lock.
2. Add CI dependency import verification before test collection.
3. Validate repository YAML capability claims, derive implemented distortions from the real registry, and fail closed on planning-only configs.
4. Quarantine broad future data-generation configs instead of implying executable hardware/noisy/fake-backend support.
5. Document supported versions, claim boundaries, and migration notes.

Later phases remain explicitly unsupported unless corresponding code and tests are added.
