"""Born-probability metrics for comparing clean and distorted circuit outputs."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .probability import align_probability_distributions, probabilities_from_input
from .results import BornMetricBundle, BornMetricResult


def _validate_log_base(base: float) -> float:
    value = float(base)
    if value <= 0.0 or value == 1.0 or not math.isfinite(value):
        raise ValueError("log base must be finite, positive, and not equal to 1.")
    return value


def total_variation_distance(p: Mapping[str, float] | Any, q: Mapping[str, float] | Any) -> float:
    """Return total variation distance between two Born distributions."""
    p_aligned, q_aligned = align_probability_distributions(probabilities_from_input(p), probabilities_from_input(q))
    value = 0.5 * sum(abs(p_aligned[key] - q_aligned[key]) for key in p_aligned)
    return min(1.0, max(0.0, value))


def hellinger_distance(p: Mapping[str, float] | Any, q: Mapping[str, float] | Any) -> float:
    """Return Hellinger distance between two Born distributions."""
    p_aligned, q_aligned = align_probability_distributions(probabilities_from_input(p), probabilities_from_input(q))
    affinity = sum(math.sqrt(p_aligned[key] * q_aligned[key]) for key in p_aligned)
    return math.sqrt(max(0.0, 1.0 - min(1.0, affinity)))


def kl_divergence(p: Mapping[str, float] | Any, q: Mapping[str, float] | Any, *, base: float = 2.0) -> float:
    """Return directional KL(P || Q), with infinity for impossible support under Q."""
    log_base = _validate_log_base(base)
    p_aligned, q_aligned = align_probability_distributions(probabilities_from_input(p), probabilities_from_input(q))
    total = 0.0
    for key, p_value in p_aligned.items():
        if p_value == 0.0:
            continue
        q_value = q_aligned[key]
        if q_value == 0.0:
            return math.inf
        total += p_value * (math.log(p_value / q_value) / math.log(log_base))
    return max(0.0, total)


def jensen_shannon_divergence(p: Mapping[str, float] | Any, q: Mapping[str, float] | Any, *, base: float = 2.0) -> float:
    """Return symmetric Jensen-Shannon divergence between two Born distributions."""
    _validate_log_base(base)
    p_aligned, q_aligned = align_probability_distributions(probabilities_from_input(p), probabilities_from_input(q))
    midpoint = {key: 0.5 * (p_aligned[key] + q_aligned[key]) for key in p_aligned}
    value = 0.5 * kl_divergence(p_aligned, midpoint, base=base) + 0.5 * kl_divergence(q_aligned, midpoint, base=base)
    if base == 2.0:
        return min(1.0, max(0.0, value))
    return max(0.0, value)


def jensen_shannon_distance(p: Mapping[str, float] | Any, q: Mapping[str, float] | Any, *, base: float = 2.0) -> float:
    """Return the square-root Jensen-Shannon distance."""
    return math.sqrt(jensen_shannon_divergence(p, q, base=base))


def _result(
    name: str,
    value: float,
    *,
    symmetric: bool,
    bounded: bool,
    value_range: tuple[float | None, float | None],
    metadata: Mapping[str, Any] | None = None,
) -> BornMetricResult:
    return BornMetricResult(
        metric_name=name,
        metric_family="born",
        value=value,
        lower_is_better=True,
        symmetric=symmetric,
        bounded=bounded,
        value_range=value_range,
        metadata=dict(metadata or {}),
    )


def _metadata_for_inputs(clean: Any, distorted: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "clean_input_type": type(clean).__name__,
        "distorted_input_type": type(distorted).__name__,
        "probability_source": "probabilities attribute or mapping; no simulation, distortion, counts inference, or manifest writes performed",
    }
    for label, obj in (("clean", clean), ("distorted", distorted)):
        obj_metadata = getattr(obj, "metadata", None)
        if isinstance(obj_metadata, Mapping):
            simulation_mode = obj_metadata.get("simulation_mode") or getattr(obj, "simulation_mode", None)
            if simulation_mode is not None:
                metadata[f"{label}_simulation_mode"] = simulation_mode
            if "source_simulation_mode" in obj_metadata:
                metadata[f"{label}_source_simulation_mode"] = obj_metadata["source_simulation_mode"]
        elif hasattr(obj, "simulation_mode"):
            metadata[f"{label}_simulation_mode"] = getattr(obj, "simulation_mode")
    return metadata


def compare_born_distributions(
    clean: Any,
    distorted: Any,
    *,
    context_metadata: Mapping[str, Any] | None = None,
    include_kl: bool = True,
    include_js_distance: bool = True,
) -> BornMetricBundle:
    """Compare two supplied Born distributions and return structured metric results.

    This function is deliberately measurement-only: it never simulates circuits,
    applies distortions, writes manifests, or fabricates readout/layout effects.
    """
    clean_probabilities = probabilities_from_input(clean)
    distorted_probabilities = probabilities_from_input(distorted)
    clean_aligned, distorted_aligned = align_probability_distributions(clean_probabilities, distorted_probabilities)
    support = list(clean_aligned)

    metrics: dict[str, BornMetricResult] = {
        "total_variation": _result(
            "total_variation",
            total_variation_distance(clean_aligned, distorted_aligned),
            symmetric=True,
            bounded=True,
            value_range=(0.0, 1.0),
        ),
        "hellinger": _result(
            "hellinger",
            hellinger_distance(clean_aligned, distorted_aligned),
            symmetric=True,
            bounded=True,
            value_range=(0.0, 1.0),
        ),
        "jensen_shannon_divergence": _result(
            "jensen_shannon_divergence",
            jensen_shannon_divergence(clean_aligned, distorted_aligned),
            symmetric=True,
            bounded=True,
            value_range=(0.0, 1.0),
            metadata={"log_base": 2.0},
        ),
    }
    if include_js_distance:
        metrics["jensen_shannon_distance"] = _result(
            "jensen_shannon_distance",
            jensen_shannon_distance(clean_aligned, distorted_aligned),
            symmetric=True,
            bounded=True,
            value_range=(0.0, 1.0),
            metadata={"log_base": 2.0, "definition": "sqrt(jensen_shannon_divergence)"},
        )
    if include_kl:
        metrics["kl_clean_to_distorted"] = _result(
            "kl_clean_to_distorted",
            kl_divergence(clean_aligned, distorted_aligned),
            symmetric=False,
            bounded=False,
            value_range=(0.0, None),
            metadata={"direction": "clean||distorted", "log_base": 2.0, "impossible_support_returns": "inf"},
        )
        metrics["kl_distorted_to_clean"] = _result(
            "kl_distorted_to_clean",
            kl_divergence(distorted_aligned, clean_aligned),
            symmetric=False,
            bounded=False,
            value_range=(0.0, None),
            metadata={"direction": "distorted||clean", "log_base": 2.0, "impossible_support_returns": "inf"},
        )

    context = dict(context_metadata or {})
    metadata = {
        "metric_family": "born",
        "support_size": len(support),
        "clean_support_size": len(clean_probabilities),
        "distorted_support_size": len(distorted_probabilities),
        "aligned_support_size": len(support),
        "zero_safe_handling": "missing bitstrings are aligned with probability 0.0; KL returns infinity for positive mass against zero support",
        "input_metadata": _metadata_for_inputs(clean, distorted),
    }
    if context:
        metadata["context_metadata"] = context
        if context.get("marker_only") is True or context.get("distortion_family") in {"readout", "layout"}:
            metadata["applicability_warning"] = (
                "marker-only distortion context; Born metrics only compare supplied distributions and do not simulate readout/layout effects"
            )

    return BornMetricBundle(metric_family="born", metrics=metrics, support=support, metadata=metadata)
