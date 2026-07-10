"""Deterministic exact-objective SPSA baseline for Phase 10."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .config import BaselineSuiteConfig
from .models import EvaluationSnapshot
from .optimizer_common import ExactObjectiveEvaluator


def _sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(
        f"triqto-phase10-spsa:{base_seed}:{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def run_spsa(
    *,
    sample_id: str,
    evaluator: ExactObjectiveEvaluator,
    config: BaselineSuiteConfig,
) -> tuple[EvaluationSnapshot, int, dict[str, Any]]:
    """Run bounded deterministic SPSA and return its best evaluated point."""
    dimensions = len(evaluator.axes)
    if dimensions <= 0:
        raise ValueError("SPSA requires at least one optimizer coordinate")
    if dimensions > config.max_optimizer_dimensions:
        raise RuntimeError(
            f"SPSA dimension {dimensions} exceeds max_optimizer_dimensions"
        )
    required = 1 + 3 * config.spsa_iterations
    if required > config.max_objective_evaluations:
        raise RuntimeError(
            "Configured SPSA iteration count exceeds max_objective_evaluations"
        )

    seed = _sample_seed(config.random_seed, sample_id)
    rng = np.random.default_rng(seed)
    vector = np.zeros(dimensions, dtype=np.float64)
    evaluator.evaluate(vector)
    for iteration in range(config.spsa_iterations):
        k = iteration + 1
        ak = config.spsa_a / ((k + 10.0) ** config.spsa_alpha)
        ck = config.spsa_c / (k**config.spsa_gamma)
        delta = rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float64),
            size=dimensions,
            replace=True,
        )
        plus = evaluator.evaluate(vector + ck * delta)
        minus = evaluator.evaluate(vector - ck * delta)
        gradient = ((plus.objective - minus.objective) / (2.0 * ck)) * delta
        vector = np.clip(
            vector - ak * gradient,
            -config.max_abs_angle,
            config.max_abs_angle,
        ).astype(np.float64, copy=False)
        evaluator.evaluate(vector)

    if evaluator.best is None:  # pragma: no cover - evaluate always sets best
        raise RuntimeError("SPSA produced no objective evaluation")
    return evaluator.best, config.spsa_iterations, {
        "algorithm": "simultaneous perturbation stochastic approximation",
        "deterministic_seed": seed,
        "dimensions": dimensions,
        "iterations": config.spsa_iterations,
        "a": config.spsa_a,
        "c": config.spsa_c,
        "alpha": config.spsa_alpha,
        "gamma": config.spsa_gamma,
        "best_point_returned": True,
        "clean_target_used_for_objective": True,
        "learned_model_used": False,
    }


__all__ = ["run_spsa"]
