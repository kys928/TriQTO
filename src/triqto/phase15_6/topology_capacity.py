"""Resolve Phase 11 topology capacity from completed campaign manifests.

The configured ``topology_max_points_per_group`` is treated as the requested
minimum capacity for Phase 15.6.  The effective Phase 11 capacity is expanded
when the completed Phase 7/9 data contains a larger legitimate group.  This
preserves every scientific point while retaining a hard operational ceiling.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TOPOLOGY_HARD_POINT_LIMIT = 4096


def _parquet_batches(path: Path, columns: list[str]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency environment
        raise RuntimeError("pyarrow is required to resolve topology capacity") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest does not exist: {path}")
    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(columns=columns, batch_size=65536)


def _distortion_types(phase7_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = phase7_root / "manifests" / "distortion_manifest.parquet"
    for batch in _parquet_batches(path, ["distortion_id", "distortion_type"]):
        rows = batch.to_pydict()
        for distortion_id, distortion_type in zip(
            rows["distortion_id"], rows["distortion_type"], strict=True
        ):
            if not isinstance(distortion_id, str) or not distortion_id:
                raise ValueError("distortion_manifest contains an invalid distortion_id")
            if not isinstance(distortion_type, str) or not distortion_type:
                raise ValueError(
                    f"Distortion {distortion_id} has an invalid distortion_type"
                )
            previous = result.setdefault(distortion_id, distortion_type)
            if previous != distortion_type:
                raise ValueError(
                    f"Distortion {distortion_id} has inconsistent distortion_type values"
                )
    return result


def resolve_topology_group_capacity(
    phase7_root: str | Path,
    phase9_root: str | Path,
    requested_capacity: int,
    *,
    hard_limit: int = DEFAULT_TOPOLOGY_HARD_POINT_LIMIT,
) -> tuple[int, dict[str, Any]]:
    """Return a lossless effective Phase 11 point capacity and audit metadata.

    Capacity is derived from every topology group kind used by Phase 15.6:
    action neighborhoods, family/qubit cohorts, and
    family/qubit/distortion-type cohorts. No points are sampled or truncated.
    """
    if isinstance(requested_capacity, bool) or not isinstance(requested_capacity, int):
        raise TypeError("requested_capacity must be an integer and not bool")
    if requested_capacity <= 0:
        raise ValueError("requested_capacity must be positive")
    if isinstance(hard_limit, bool) or not isinstance(hard_limit, int):
        raise TypeError("hard_limit must be an integer and not bool")
    if hard_limit < requested_capacity:
        raise ValueError("hard_limit must be at least requested_capacity")

    phase7 = Path(phase7_root)
    phase9 = Path(phase9_root)
    distortion_types = _distortion_types(phase7)

    family_qubit: Counter[tuple[str, int]] = Counter()
    family_qubit_distortion: Counter[tuple[str, int, str]] = Counter()
    sample_path = phase7 / "manifests" / "sample_manifest.parquet"
    for batch in _parquet_batches(
        sample_path,
        ["family", "n_qubits", "distortion_id"],
    ):
        rows = batch.to_pydict()
        for family, n_qubits, distortion_id in zip(
            rows["family"],
            rows["n_qubits"],
            rows["distortion_id"],
            strict=True,
        ):
            if not isinstance(family, str) or not family:
                raise ValueError("sample_manifest contains an invalid family")
            if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
                raise TypeError("sample_manifest n_qubits must be an integer")
            distortion_type = distortion_types.get(distortion_id)
            if distortion_type is None:
                raise ValueError(
                    f"Sample references missing distortion {distortion_id!r}"
                )
            family_qubit[(family, n_qubits)] += 1
            family_qubit_distortion[(family, n_qubits, distortion_type)] += 1

    action_neighborhood: Counter[str] = Counter()
    candidate_path = phase9 / "manifests" / "action_candidate_manifest.parquet"
    for batch in _parquet_batches(candidate_path, ["sample_id"]):
        for sample_id in batch.column(0).to_pylist():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    "action_candidate_manifest contains an invalid sample_id"
                )
            action_neighborhood[sample_id] += 1

    maxima = {
        "action_neighborhood": max(action_neighborhood.values(), default=0),
        "family_qubit_cohort": max(family_qubit.values(), default=0),
        "family_qubit_distortion_cohort": max(
            family_qubit_distortion.values(), default=0
        ),
    }
    required = max(maxima.values(), default=0)
    effective = max(requested_capacity, required)
    if effective > hard_limit:
        raise RuntimeError(
            "Phase 11 requires a topology group capacity of "
            f"{effective}, exceeding the hard operational limit {hard_limit}. "
            "Increase the hard limit only after reviewing memory and persistent-"
            "homology complexity; no points were truncated."
        )
    metadata = {
        "requested_capacity": requested_capacity,
        "required_capacity": required,
        "effective_capacity": effective,
        "hard_limit": hard_limit,
        "auto_expanded": effective > requested_capacity,
        "maximum_group_sizes": maxima,
        "point_policy": "lossless_auto_expand_no_sampling_no_truncation",
    }
    return effective, metadata


__all__ = [
    "DEFAULT_TOPOLOGY_HARD_POINT_LIMIT",
    "resolve_topology_group_capacity",
]
