"""Losslessly resolve the Phase 11 topology-group count before computation.

The campaign config carries an operational requested group ceiling. Large completed
campaigns can legitimately contain more eligible topology groups, especially because
there is one action-neighborhood group per Phase 7 sample. This module derives the
exact group count from immutable Phase 7/9 manifests, expands the operational capacity
without dropping groups, and retains a separate hard safety ceiling.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import time
from typing import Any, Iterable

from triqto.topology.constants import GROUP_KINDS


DEFAULT_TOPOLOGY_HARD_GROUP_LIMIT = 65_536
_BATCH_SIZE = 65_536
_SCHEMA = "triqto.phase15_6.topology_group_count_resolution.v1"


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer and not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_group_kinds(value: Iterable[str]) -> tuple[str, ...]:
    kinds = tuple(value)
    if not kinds:
        raise ValueError("group_kinds must not be empty")
    if len(set(kinds)) != len(kinds):
        raise ValueError("group_kinds must be unique")
    unknown = set(kinds) - set(GROUP_KINDS)
    if unknown:
        raise ValueError(f"unknown topology group kinds: {sorted(unknown)}")
    expected = tuple(kind for kind in GROUP_KINDS if kind in kinds)
    if kinds != expected:
        raise ValueError(
            "group_kinds must follow the fixed Phase 11 order: "
            f"{list(GROUP_KINDS)}"
        )
    return kinds


def _parquet(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to resolve topology group counts") from exc
    if not path.is_file():
        raise FileNotFoundError(f"required manifest does not exist: {path}")
    return pq.ParquetFile(path)


def _log(message: str) -> None:
    print(f"[Phase 11][group-count] {message}", flush=True)


def _distortion_types(phase7_root: Path) -> dict[str, str]:
    path = phase7_root / "manifests" / "distortion_manifest.parquet"
    parquet = _parquet(path)
    result: dict[str, str] = {}
    for batch in parquet.iter_batches(
        columns=["distortion_id", "distortion_type"],
        batch_size=_BATCH_SIZE,
    ):
        values = batch.to_pydict()
        for distortion_id, distortion_type in zip(
            values["distortion_id"], values["distortion_type"], strict=True
        ):
            if not isinstance(distortion_id, str) or not distortion_id:
                raise ValueError("distortion manifest contains an invalid distortion_id")
            if not isinstance(distortion_type, str) or not distortion_type:
                raise ValueError(
                    f"distortion {distortion_id!r} has an invalid distortion_type"
                )
            previous = result.setdefault(distortion_id, distortion_type)
            if previous != distortion_type:
                raise ValueError(
                    f"distortion {distortion_id!r} has inconsistent types"
                )
    return result


def _phase7_group_sizes(
    phase7_root: Path,
    *,
    include_family_qubit: bool,
    include_family_qubit_distortion: bool,
) -> tuple[set[str], Counter[tuple[str, int]], Counter[tuple[str, int, str]]]:
    distortion_types = (
        _distortion_types(phase7_root)
        if include_family_qubit_distortion
        else {}
    )
    samples: set[str] = set()
    family_qubit: Counter[tuple[str, int]] = Counter()
    family_qubit_distortion: Counter[tuple[str, int, str]] = Counter()
    path = phase7_root / "manifests" / "sample_manifest.parquet"
    parquet = _parquet(path)
    columns = ["sample_id", "family", "n_qubits", "distortion_id"]
    started = time.monotonic()
    total_rows = int(parquet.metadata.num_rows) if parquet.metadata is not None else 0
    completed = 0
    _log(f"scanning Phase 7 sample manifest | rows={total_rows:,}")
    for batch_index, batch in enumerate(
        parquet.iter_batches(columns=columns, batch_size=_BATCH_SIZE),
        start=1,
    ):
        values = batch.to_pydict()
        for sample_id, family, n_qubits, distortion_id in zip(
            values["sample_id"],
            values["family"],
            values["n_qubits"],
            values["distortion_id"],
            strict=True,
        ):
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("sample manifest contains an invalid sample_id")
            if sample_id in samples:
                raise ValueError(f"duplicate Phase 7 sample_id {sample_id!r}")
            if not isinstance(family, str) or not family:
                raise ValueError(f"sample {sample_id!r} has an invalid family")
            if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
                raise TypeError(f"sample {sample_id!r} n_qubits must be integer")
            samples.add(sample_id)
            if include_family_qubit:
                family_qubit[(family, n_qubits)] += 1
            if include_family_qubit_distortion:
                distortion_type = distortion_types.get(distortion_id)
                if distortion_type is None:
                    raise ValueError(
                        f"sample {sample_id!r} references missing distortion "
                        f"{distortion_id!r}"
                    )
                family_qubit_distortion[(family, n_qubits, distortion_type)] += 1
        completed += int(batch.num_rows)
        elapsed = max(time.monotonic() - started, 1e-9)
        _log(
            f"Phase 7 samples batch={batch_index} | "
            f"rows={completed:,}/{total_rows:,} | rate={completed/elapsed:,.0f}/s"
        )
    if total_rows and completed != total_rows:
        raise ValueError("Phase 7 sample manifest row count changed during scan")
    return samples, family_qubit, family_qubit_distortion


def _action_neighborhood_sizes(
    phase9_root: Path,
    expected_sample_ids: set[str],
) -> Counter[str]:
    path = phase9_root / "manifests" / "action_candidate_manifest.parquet"
    parquet = _parquet(path)
    total_rows = int(parquet.metadata.num_rows) if parquet.metadata is not None else 0
    total_batches = max(1, math.ceil(total_rows / _BATCH_SIZE))
    counts: Counter[str] = Counter()
    started = time.monotonic()
    completed = 0
    _log(
        "scanning Phase 9 candidate manifest for action-neighborhood sizes | "
        f"rows={total_rows:,} | batches={total_batches}"
    )
    for batch_index, batch in enumerate(
        parquet.iter_batches(columns=["sample_id"], batch_size=_BATCH_SIZE),
        start=1,
    ):
        for sample_id in batch.column(0).to_pylist():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("candidate manifest contains an invalid sample_id")
            counts[sample_id] += 1
        completed += int(batch.num_rows)
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed / elapsed
        remaining = max(total_rows - completed, 0)
        eta = remaining / rate if rate > 0.0 else 0.0
        _log(
            f"candidate manifest batch={batch_index}/{total_batches} | "
            f"rows={completed:,}/{total_rows:,} | rate={rate:,.0f}/s | "
            f"ETA≈{eta/60.0:.1f}m"
        )
    if total_rows and completed != total_rows:
        raise ValueError("Phase 9 candidate manifest row count changed during scan")
    if set(counts) != expected_sample_ids:
        missing = sorted(expected_sample_ids - set(counts))[:10]
        unexpected = sorted(set(counts) - expected_sample_ids)[:10]
        raise ValueError(
            "Phase 9 action-neighborhood sample coverage differs from Phase 7; "
            f"missing_examples={missing}, unexpected_examples={unexpected}"
        )
    return counts


def _eligible_summary(
    sizes: Iterable[int],
    min_points: int,
) -> tuple[int, int, int, int]:
    eligible = [int(size) for size in sizes if int(size) >= min_points]
    return (
        len(eligible),
        sum(eligible),
        sum(size * size for size in eligible),
        max(eligible, default=0),
    )


def resolve_topology_group_count(
    phase7_root: str | Path,
    phase9_root: str | Path,
    requested_max_groups: int,
    *,
    min_points: int,
    group_kinds: Iterable[str] = GROUP_KINDS,
    hard_limit: int = DEFAULT_TOPOLOGY_HARD_GROUP_LIMIT,
) -> tuple[int, dict[str, Any]]:
    """Return a lossless effective Phase 11 group capacity and audit metadata."""
    requested = _require_positive_int(requested_max_groups, "requested_max_groups")
    minimum = _require_positive_int(min_points, "min_points")
    hard = _require_positive_int(hard_limit, "hard_limit")
    if hard < requested:
        raise ValueError("hard_limit must be at least requested_max_groups")
    kinds = _require_group_kinds(group_kinds)
    phase7 = Path(phase7_root)
    phase9 = Path(phase9_root)

    sample_ids, family_qubit, family_qubit_distortion = _phase7_group_sizes(
        phase7,
        include_family_qubit="family_qubit_cohort" in kinds,
        include_family_qubit_distortion=(
            "family_qubit_distortion_cohort" in kinds
        ),
    )
    action_neighborhood = (
        _action_neighborhood_sizes(phase9, sample_ids)
        if "action_neighborhood" in kinds
        else Counter()
    )

    raw_sizes: dict[str, Iterable[int]] = {
        "action_neighborhood": action_neighborhood.values(),
        "family_qubit_cohort": family_qubit.values(),
        "family_qubit_distortion_cohort": family_qubit_distortion.values(),
    }
    kind_counts: dict[str, int] = {}
    kind_points: dict[str, int] = {}
    kind_squared_points: dict[str, int] = {}
    maximum_group_sizes: dict[str, int] = {}
    for kind in kinds:
        count, points, squared, maximum = _eligible_summary(
            raw_sizes[kind], minimum
        )
        kind_counts[kind] = count
        kind_points[kind] = points
        kind_squared_points[kind] = squared
        maximum_group_sizes[kind] = maximum

    required = sum(kind_counts.values())
    effective = max(requested, required)
    if effective > hard:
        raise RuntimeError(
            "Phase 11 requires "
            f"{required:,} topology groups, exceeding the hard operational limit "
            f"{hard:,}. Increase the hard limit only after reviewing projected "
            "persistent-homology cost; no groups were sampled or truncated."
        )
    metadata = {
        "schema": _SCHEMA,
        "requested_max_groups": requested,
        "required_group_count": required,
        "effective_max_groups": effective,
        "hard_group_limit": hard,
        "auto_expanded": effective > requested,
        "min_points": minimum,
        "group_kinds": list(kinds),
        "group_kind_counts": kind_counts,
        "group_kind_point_counts": kind_points,
        "group_kind_squared_point_counts": kind_squared_points,
        "maximum_group_sizes": maximum_group_sizes,
        "phase7_sample_count": len(sample_ids),
        "total_group_point_count": sum(kind_points.values()),
        "sum_squared_group_sizes": sum(kind_squared_points.values()),
        "group_policy": "lossless_auto_expand_no_sampling_no_truncation",
    }
    _log(
        "resolved | "
        f"requested={requested:,} | required={required:,} | effective={effective:,} | "
        f"total_points={metadata['total_group_point_count']:,} | "
        f"sum_squared_sizes={metadata['sum_squared_group_sizes']:,}"
    )
    for kind in kinds:
        _log(
            f"{kind}: groups={kind_counts[kind]:,} | "
            f"points={kind_points[kind]:,} | max_points={maximum_group_sizes[kind]:,}"
        )
    return effective, metadata


__all__ = [
    "DEFAULT_TOPOLOGY_HARD_GROUP_LIMIT",
    "resolve_topology_group_count",
]
