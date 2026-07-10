"""Deterministic exact-objective COBYLA baseline for Phase 10."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from .config import BaselineSuiteConfig
from .models import EvaluationSnapshot
from .optimizer_common import ExactObjectiveEvaluator


def run_cobyla(
    *,
    evaluator: ExactObjectiveEvaluator,
    config: BaselineSuiteConfig,
) -> tuple[EvaluationSnapshot, int, dict[str, Any]]:
    """Run bounded SciPy COBYLA and return the best point actually evaluated."""
    dimensions = len(evaluator.axes)
    if dimensions <= 0:
        raise ValueError("COBYLA requires at least one optimizer coordinate")
    if dimensions > config.max_optimizer_dimensions:
        raise RuntimeError(
            f"COBYLA dimension {dimensions} exceeds max_optimizer_dimensions"
        )
    if config.cobyla_maxiter > config.max_objective_evaluations:
        raise RuntimeError(
            "cobyla_maxiter exceeds max_objective_evaluations guardrail"
        )

    lower = -config.max_abs_angle
    upper = config.max_abs_angle
    constraints = []
    for index in range(dimensions):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, i=index: float(x[i] - lower),
            }
        )
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, i=index: float(upper - x[i]),
            }
        )

    def objective(vector: np.ndarray) -> float:
        return evaluator.evaluate(vector).objective

    initial = np.zeros(dimensions, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="COBYLA",
        constraints=constraints,
        options={
            "maxiter": config.cobyla_maxiter,
            "rhobeg": config.cobyla_initial_step,
            "tol": config.cobyla_tolerance,
            "catol": config.cobyla_tolerance,
        },
    )
    if evaluator.best is None:
        raise RuntimeError("COBYLA produced no objective evaluation")
    iterations_raw = getattr(result, "nit", 0)
    iterations = (
        int(iterations_raw)
        if isinstance(iterations_raw, (int, np.integer))
        else 0
    )
    return evaluator.best, iterations, {
        "algorithm": "scipy.optimize.minimize(method='COBYLA')",
        "dimensions": dimensions,
        "maxiter": config.cobyla_maxiter,
        "initial_step": config.cobyla_initial_step,
        "tolerance": config.cobyla_tolerance,
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "reported_function_evaluations": int(getattr(result, "nfev", evaluator.evaluations)),
        "best_point_returned": True,
        "clean_target_used_for_objective": True,
        "learned_model_used": False,
    }


__all__ = ["run_cobyla"]
