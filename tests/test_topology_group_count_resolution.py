from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from triqto.phase15_6.topology_group_count import resolve_topology_group_count
from triqto.topology import TopologyAuditConfig
from triqto.topology.identities import topology_operational_config_id
from triqto.topology.topology_groups import build_topology_group_specs


def _write_manifests(root: Path) -> tuple[Path, Path]:
    phase7 = root / "phase7"
    phase9 = root / "phase9"
    (phase7 / "manifests").mkdir(parents=True)
    (phase9 / "manifests").mkdir(parents=True)

    pq.write_table(
        pa.table(
            {
                "distortion_id": ["d1", "d2"],
                "distortion_type": ["phase", "amplitude"],
            }
        ),
        phase7 / "manifests" / "distortion_manifest.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "family": ["bell", "bell", "bell", "ghz"],
                "n_qubits": [2, 2, 2, 3],
                "distortion_id": ["d1", "d1", "d2", "d2"],
            }
        ),
        phase7 / "manifests" / "sample_manifest.parquet",
    )
    sample_ids = ["s1"] * 3 + ["s2"] * 2 + ["s3"] * 4 + ["s4"]
    pq.write_table(
        pa.table({"sample_id": sample_ids}),
        phase9 / "manifests" / "action_candidate_manifest.parquet",
    )
    return phase7, phase9


def test_group_count_auto_expands_without_dropping_groups(tmp_path: Path) -> None:
    phase7, phase9 = _write_manifests(tmp_path)
    effective, metadata = resolve_topology_group_count(
        phase7,
        phase9,
        2,
        min_points=3,
    )

    assert effective == 3
    assert metadata["requested_max_groups"] == 2
    assert metadata["required_group_count"] == 3
    assert metadata["effective_max_groups"] == 3
    assert metadata["auto_expanded"] is True
    assert metadata["group_kind_counts"] == {
        "action_neighborhood": 2,
        "family_qubit_cohort": 1,
        "family_qubit_distortion_cohort": 0,
    }
    assert metadata["total_group_point_count"] == 10
    assert metadata["sum_squared_group_sizes"] == 34
    assert metadata["group_policy"] == (
        "lossless_auto_expand_no_sampling_no_truncation"
    )


def test_group_count_fails_closed_above_hard_limit(tmp_path: Path) -> None:
    phase7, phase9 = _write_manifests(tmp_path)
    with pytest.raises(RuntimeError, match="no groups were sampled or truncated"):
        resolve_topology_group_count(
            phase7,
            phase9,
            2,
            min_points=3,
            hard_limit=2,
        )


class _LazyAction:
    is_lazy = True

    def __init__(self, sizes: dict[str, int]) -> None:
        self.sizes = sizes

    def action_ids_for_sample(self, sample_id: str) -> tuple[str, ...]:
        return tuple(
            f"{sample_id}-action-{index:03d}"
            for index in range(self.sizes[sample_id])
        )


def test_group_plan_preserves_all_groups_and_orders_smallest_first() -> None:
    samples = [
        SimpleNamespace(
            sample_id="sample-large",
            family="bell",
            n_qubits=2,
            distortion_id="d1",
        ),
        SimpleNamespace(
            sample_id="sample-small",
            family="bell",
            n_qubits=2,
            distortion_id="d1",
        ),
    ]
    sources = SimpleNamespace(
        phase7=SimpleNamespace(samples=samples, distortions=[]),
        action=_LazyAction({"sample-large": 5, "sample-small": 3}),
    )
    config = TopologyAuditConfig(
        group_kinds=("action_neighborhood",),
        min_points=3,
        max_points_per_group=8,
        max_groups=2,
    )
    specs, skipped = build_topology_group_specs(sources, config)

    assert skipped == {}
    assert len(specs) == 2
    assert [len(spec.point_ids) for spec in specs] == [3, 5]
    assert {spec.metadata["sample_id"] for spec in specs} == {
        "sample-small",
        "sample-large",
    }


def test_effective_group_capacity_is_bound_into_topology_identity() -> None:
    requested = TopologyAuditConfig(max_groups=2)
    effective = TopologyAuditConfig(max_groups=13_566)
    assert topology_operational_config_id(requested) != topology_operational_config_id(
        effective
    )
