# Phase 0 implementation plan

1. Establish a CPU-safe pinned dependency path and separate optional GPU constraints.
2. Add CI dependency import verification before test collection.
3. Validate configs for executable-vs-unsupported truthfulness and reject unknown active modes.
4. Quarantine broad future data-generation configs instead of implying executable hardware/noisy/fake-backend support.
5. Document supported versions, claim boundaries, and migration notes.

Later phases remain explicitly unsupported unless corresponding code and tests are added.
