#!/usr/bin/env python3
"""Stable v2 runner for the TriQTO phase/amplitude identifiability probes.

This runner preserves the frozen evidence regimes, train-only preprocessing,
group-safe cross-validation, OOF Platt calibration, bootstrap intervals, and
retuned label-shuffle controls from the v1 evaluator. It replaces the unstable
LBFGS MLP nonlinear probe with a deterministic RBF-kernel SVM and turns any
scikit-learn convergence warning into a hard failure.

The historical v0.1 test split remains inaccessible to the underlying protocol.
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "train_phase_amplitude_identifiability_probes.py"
MODULE_NAME = "triqto_v0_2_phase_amplitude_probe_v1"


def _load_v1_module():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the v1 evaluator from {V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _nonlinear_estimator(parameters: Mapping[str, Any], seed: int) -> Pipeline:
    gamma_value = parameters["gamma"]
    gamma: str | float = (
        str(gamma_value)
        if isinstance(gamma_value, str)
        else float(gamma_value)
    )
    classifier = SVC(
        C=float(parameters["C"]),
        kernel="rbf",
        gamma=gamma,
        shrinking=True,
        tol=1.0e-5,
        cache_size=512.0,
        max_iter=-1,
        random_state=seed,
    )
    return Pipeline(
        [
            ("variance", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=0.995, svd_solver="full")),
            ("model", classifier),
        ]
    )


def main() -> None:
    module = _load_v1_module()
    original_estimator = module.estimator

    module.SCHEMA = "triqto.v0_2.phase_amplitude_probe_evaluation.v2"
    module.NONLINEAR_GRID = tuple(
        {"C": c_value, "gamma": gamma}
        for c_value in (0.1, 1.0, 10.0)
        for gamma in ("scale", 0.01, 0.1)
    )

    def stable_estimator(
        model: str,
        parameters: Mapping[str, Any],
        seed: int,
    ) -> Pipeline:
        if model == "nonlinear":
            return _nonlinear_estimator(parameters, seed)
        return original_estimator(model, parameters, seed)

    module.estimator = stable_estimator

    # The protocol must bind to this runner, not silently retain the v1 script hash.
    module.__file__ = str(Path(__file__).resolve())

    # A published scientific report must never silently include optimizer-capped fits.
    warnings.filterwarnings("error", category=ConvergenceWarning)
    module.main()


if __name__ == "__main__":
    main()
