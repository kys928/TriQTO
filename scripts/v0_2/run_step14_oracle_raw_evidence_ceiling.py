#!/usr/bin/env python3
"""Compatibility entrypoint for the frozen Step-14 oracle/raw-evidence diagnostic.

PR #101's v1 diagnostic accidentally bounded the frozen Step-14 reference
circuit at 16 operations. The frozen generator can legally create 17 operations:
for nine reference layers it emits the initial RY, nine per-layer single-qubit
operations, six entanglers (layers 0 and 1 satisfy the minimum-entangler rule,
then even layers 2/4/6/8), and the final H.

This entrypoint changes only that implementation support bound. It delegates all
feature definitions, probe hyperparameters, splits, frozen identifiers, output
schema, and scientific gates to the original frozen diagnostic module. No model
checkpoint is updated and no outer/reserve/QPU data is enabled here.
"""
from __future__ import annotations

import analyze_step14_oracle_raw_evidence_ceiling as diagnostic

FROZEN_STEP14_MAX_REFERENCE_OPERATIONS = 17


def apply_frozen_support_bound() -> None:
    """Apply the one bug-fix bound while failing closed on unexpected drift."""
    if diagnostic.MAX_GATES not in {16, FROZEN_STEP14_MAX_REFERENCE_OPERATIONS}:
        raise RuntimeError(
            "unexpected underlying oracle diagnostic MAX_GATES; review before changing frozen support"
        )
    diagnostic.MAX_GATES = FROZEN_STEP14_MAX_REFERENCE_OPERATIONS


def main() -> None:
    apply_frozen_support_bound()
    diagnostic.main()


if __name__ == "__main__":
    main()
