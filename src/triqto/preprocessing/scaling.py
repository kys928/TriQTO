"""Training-only semantic-family scaler fitting and application metadata."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np


def _fit_robust(values: np.ndarray) -> dict[str, Any]:
    median = float(np.median(values))
    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    scale = q3 - q1
    return {
        "center": median,
        "scale": scale if scale > 0.0 else 1.0,
        "q1": q1,
        "q3": q3,
    }


def fit_semantic_scalers(
    rows: Iterable[Mapping[str, Any]],
    *,
    feature_specs: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    materialized = list(rows)
    scalers: dict[str, Any] = {}
    for feature_name, (semantic_family, method) in sorted(feature_specs.items()):
        values: list[float] = []
        for row in materialized:
            raw = row.get(feature_name)
            if raw is None:
                continue
            numeric = float(raw)
            if math.isfinite(numeric):
                values.append(numeric)
        if not values:
            scalers[feature_name] = {
                "semantic_family": semantic_family,
                "method": method,
                "status": "unavailable",
            }
            continue
        array = np.asarray(values, dtype=float)
        if method == "identity":
            parameters: dict[str, Any] = {}
        elif method == "robust":
            parameters = _fit_robust(array)
        elif method == "log_robust":
            if np.any(array <= 0.0):
                raise ValueError(f"feature {feature_name} requires positive values for log scaling")
            parameters = {**_fit_robust(np.log(array)), "pre_transform": "log"}
        elif method == "log1p_robust":
            if np.any(array < 0.0):
                raise ValueError(f"feature {feature_name} requires nonnegative values for log1p scaling")
            parameters = {**_fit_robust(np.log1p(array)), "pre_transform": "log1p"}
        elif method == "sincos":
            parameters = {"period": 2.0 * math.pi, "outputs": ["sin", "cos"]}
        else:
            raise ValueError(f"unsupported scaling method {method!r}")
        scalers[feature_name] = {
            "semantic_family": semantic_family,
            "method": method,
            "status": "fit_on_training_only",
            "count": len(values),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
            "parameters": parameters,
        }
    return scalers


def apply_scaler(value: float, scaler: Mapping[str, Any]) -> Any:
    method = scaler["method"]
    if method == "identity":
        return float(value)
    if method == "sincos":
        return [math.sin(float(value)), math.cos(float(value))]
    parameters = scaler.get("parameters", {})
    transformed = float(value)
    if parameters.get("pre_transform") == "log":
        transformed = math.log(transformed)
    elif parameters.get("pre_transform") == "log1p":
        transformed = math.log1p(transformed)
    return (transformed - float(parameters["center"])) / float(parameters["scale"])
