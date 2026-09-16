#!/usr/bin/env python3
"""Step-15 FIT entrypoint with the literal-candidate positive-label contract.

The frozen response-equivalence criterion is unchanged for alternative
candidates. The injected FIT candidate itself is always a positive when it is
present in the plausible-candidate set, matching the Step-14 privileged audit
contract. This matters when the true local frame is internally axis-ambiguous:
a candidate is necessarily response-equivalent to itself even if the
identity-vs-permutation assignment margin is below the threshold used to admit
*additional* equivalent candidates.
"""
from __future__ import annotations

import numpy as np

import fit_step15_frame_ranker as base

_BASE_RESPONSE_EQUIVALENT = base.response_equivalent


def response_equivalent_with_literal_identity(
    candidate_exact: np.ndarray,
    true_exact: np.ndarray,
    spec,
) -> bool:
    candidate = np.asarray(candidate_exact)
    truth = np.asarray(true_exact)
    if candidate.shape == truth.shape and np.array_equal(candidate, truth):
        return True
    return bool(_BASE_RESPONSE_EQUIVALENT(candidate, truth, spec))


def main() -> None:
    base.response_equivalent = response_equivalent_with_literal_identity
    base.main()


if __name__ == "__main__":
    main()
