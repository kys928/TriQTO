from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triqto.phase15_6.topology_capacity import resolve_topology_group_capacity


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_topology_capacity_expands_to_largest_completed_group(tmp_path: Path) -> None:
    phase7 = tmp_path / "phase7"
    phase9 = tmp_path / "phase9"
    _write_parquet(
        phase7 / "manifests" / "distortion_manifest.parquet",
        [
            {"distortion_id": "d1", "distortion_type": "rx"},
            {"distortion_id": "d2", "distortion_type": "rz"},
        ],
    )
    samples = []
    for index in range(640):
        samples.append(
            {
                "sample_id": f"s{index}",
                "family": "bell",
                "n_qubits": 2,
                "distortion_id": "d1" if index < 320 else "d2",
            }
        )
    _write_parquet(phase7 / "manifests" / "sample_manifest.parquet", samples)
    candidate_rows = [
        {"sample_id": "s0"} for _ in range(512)
    ] + [{"sample_id": "s1"} for _ in range(17)]
    _write_parquet(
        phase9 / "manifests" / "action_candidate_manifest.parquet",
        candidate_rows,
    )

    effective, metadata = resolve_topology_group_capacity(
        phase7,
        phase9,
        512,
    )

    assert effective == 640
    assert metadata["auto_expanded"] is True
    assert metadata["maximum_group_sizes"] == {
        "action_neighborhood": 512,
        "family_qubit_cohort": 640,
        "family_qubit_distortion_cohort": 320,
    }
    assert metadata["point_policy"] == (
        "lossless_auto_expand_no_sampling_no_truncation"
    )


def test_topology_capacity_fails_closed_above_hard_limit(tmp_path: Path) -> None:
    phase7 = tmp_path / "phase7"
    phase9 = tmp_path / "phase9"
    _write_parquet(
        phase7 / "manifests" / "distortion_manifest.parquet",
        [{"distortion_id": "d1", "distortion_type": "rx"}],
    )
    _write_parquet(
        phase7 / "manifests" / "sample_manifest.parquet",
        [
            {
                "sample_id": f"s{index}",
                "family": "bell",
                "n_qubits": 2,
                "distortion_id": "d1",
            }
            for index in range(11)
        ],
    )
    _write_parquet(
        phase9 / "manifests" / "action_candidate_manifest.parquet",
        [{"sample_id": "s0"}],
    )

    with pytest.raises(RuntimeError, match="hard operational limit 10"):
        resolve_topology_group_capacity(
            phase7,
            phase9,
            5,
            hard_limit=10,
        )
