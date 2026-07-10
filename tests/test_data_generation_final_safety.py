"""Regression tests for the final Phase 7 publication-safety patch."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    generate_dataset,
    validate_dataset_joins,
    write_dataset,
)


def _config(*, base_seed: int = 101) -> DatasetGenerationConfig:
    return DatasetGenerationConfig(
        dataset_name="phase7-final-safety",
        base_seed=base_seed,
        circuit_specs=[
            CircuitGenerationSpec(
                family="hardware_efficient_ansatz",
                n_qubits=2,
                generator_kwargs={
                    "layers": 1,
                    "entanglement": "none",
                    "measure": True,
                },
                repetitions=1,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            )
        ],
        ideal_shots=4,
        max_samples=4,
    )


def _file_bytes(root: Path, paths: list[Path]) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in paths
        if path.is_file()
    }


def test_backup_phase_failure_restores_every_old_managed_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    import triqto.data_generation.artifacts as artifacts

    root = tmp_path / "dataset"
    old_write = write_dataset(generate_dataset(_config(base_seed=11)), root)
    before = _file_bytes(root, old_write.written_paths)
    unrelated = root / "artifacts" / "private" / "note.txt"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("preserve")

    original_backup = artifacts._backup_existing_file
    calls = {"count": 0}

    def fail_during_backup(
        output_root: Path,
        backup_root: Path,
        relative: str,
    ) -> bool:
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("forced backup-phase failure")
        return original_backup(output_root, backup_root, relative)

    monkeypatch.setattr(
        artifacts,
        "_backup_existing_file",
        fail_during_backup,
    )

    with pytest.raises(OSError, match="backup-phase failure"):
        write_dataset(
            generate_dataset(_config(base_seed=12)),
            root,
            overwrite=True,
        )

    assert _file_bytes(root, [root / relative for relative in before]) == before
    assert unrelated.read_text() == "preserve"
    assert not any(tmp_path.glob(".dataset.triqto-staging-*"))
    assert not any(tmp_path.glob(".dataset.triqto-backup-*"))
    assert not list(root.glob(".dataset_complete.json.*.tmp"))


def test_completion_marker_replace_failure_removes_temporary_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    import triqto.data_generation.artifacts as artifacts

    root = tmp_path / "dataset"
    old_write = write_dataset(generate_dataset(_config(base_seed=21)), root)
    before = _file_bytes(root, old_write.written_paths)
    original_replace = artifacts._atomic_replace

    def fail_marker_replace(source: Path, destination: Path) -> None:
        if destination.name == "dataset_complete.json":
            raise OSError("forced marker replacement failure")
        original_replace(source, destination)

    monkeypatch.setattr(artifacts, "_atomic_replace", fail_marker_replace)

    with pytest.raises(OSError, match="marker replacement failure"):
        write_dataset(
            generate_dataset(_config(base_seed=22)),
            root,
            overwrite=True,
        )

    assert _file_bytes(root, [root / relative for relative in before]) == before
    assert not list(root.glob(".dataset_complete.json.*.tmp"))
    assert not any(tmp_path.glob(".dataset.triqto-staging-*"))
    assert not any(tmp_path.glob(".dataset.triqto-backup-*"))


def test_join_validation_rejects_malformed_record_shapes_cleanly() -> None:
    result = generate_dataset(_config())

    bad_circuit = copy.deepcopy(result.circuit_records[0])
    bad_circuit.metadata = []  # type: ignore[assignment]
    circuits = [bad_circuit, *result.circuit_records[1:]]
    with pytest.raises(ValueError, match="CircuitRecord.*metadata.*mapping"):
        validate_dataset_joins(
            result.sample_records,
            circuits,
            result.simulation_records,
            result.distortion_records,
            result.metric_records,
        )

    bad_simulation = copy.deepcopy(result.simulation_records[0])
    bad_simulation.metadata = []  # type: ignore[assignment]
    simulations = [bad_simulation, *result.simulation_records[1:]]
    with pytest.raises(ValueError, match="SimulationRecord.*metadata.*mapping"):
        validate_dataset_joins(
            result.sample_records,
            result.circuit_records,
            simulations,
            result.distortion_records,
            result.metric_records,
        )

    bad_distortion = copy.deepcopy(result.distortion_records[0])
    bad_distortion.metadata = []  # type: ignore[assignment]
    distortions = [bad_distortion, *result.distortion_records[1:]]
    with pytest.raises(ValueError, match="DistortionRecord.*metadata.*mapping"):
        validate_dataset_joins(
            result.sample_records,
            result.circuit_records,
            result.simulation_records,
            distortions,
            result.metric_records,
        )

    bad_sample = replace(result.sample_records[0], schema_version=7)  # type: ignore[arg-type]
    samples = [bad_sample, *result.sample_records[1:]]
    with pytest.raises(ValueError, match="DatasetSampleRecord.*schema_version"):
        validate_dataset_joins(
            samples,
            result.circuit_records,
            result.simulation_records,
            result.distortion_records,
            result.metric_records,
        )


def test_legacy_marker_derives_and_removes_only_explicit_artifacts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")

    root = tmp_path / "dataset"
    write_dataset(generate_dataset(_config(base_seed=31)), root)
    marker_path = root / "dataset_complete.json"
    marker = json.loads(marker_path.read_text())
    old_managed = set(marker.pop("managed_files"))
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")

    unrelated = root / "artifacts" / "private" / "keep.txt"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text("keep")

    write_dataset(
        generate_dataset(_config(base_seed=32)),
        root,
        overwrite=True,
    )
    new_managed = set(json.loads(marker_path.read_text())["managed_files"])

    old_artifacts = {
        path for path in old_managed if path.startswith("artifacts/")
    }
    for obsolete in old_artifacts - new_managed:
        assert not (root / obsolete).exists()
    assert unrelated.read_text() == "keep"
