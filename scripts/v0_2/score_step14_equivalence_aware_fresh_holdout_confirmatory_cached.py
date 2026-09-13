#!/usr/bin/env python3
"""Run the frozen fresh-holdout scorer with exact-frame Jacobians cached per root.

This wrapper changes no statistic, feature definition, seed, threshold, or model.
It only avoids recomputing the same privileged exact response Jacobian for the
12 target examples that share a root.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

import analyze_step14_local_frame_canonicalization as frame
import score_step14_equivalence_aware_fresh_holdout_confirmatory as scorer

_JAC_CACHE: dict[tuple[object, ...], np.ndarray] = {}


def _cached_exact_feature(
    loaded: Mapping[str, np.ndarray],
    root: Mapping[str, str],
) -> np.ndarray:
    delta, weights, pairs = frame.measured_delta_and_weights(loaded)
    key = (
        str(root.get("step14_partition", "fresh_equivalence_holdout")),
        str(root.get("family_id", "")),
        str(root.get("variant_id", "")),
        str(root.get("root_index", "")),
        int(root["injection_boundary_rank"]),
        int(root["affected_qubit"]),
        tuple(int(v) for v in np.asarray(pairs).reshape(-1).tolist()),
    )
    jac = _JAC_CACHE.get(key)
    if jac is None:
        clean = frame.circuit_from_serialized(loaded)
        jac = frame.frame_response_jacobian(
            clean,
            int(root["injection_boundary_rank"]),
            int(root["affected_qubit"]),
            pairs,
        )
        _JAC_CACHE[key] = jac
    feature = frame.canonicalize_evidence(delta, jac, weights)[0]
    if feature.shape != (24,) or not np.all(np.isfinite(feature)):
        raise RuntimeError("exact canonical frame feature contract drift")
    return feature.astype(np.float32)


scorer.exact_feature = _cached_exact_feature

if __name__ == "__main__":
    scorer.main()
