from __future__ import annotations

import json
from pathlib import Path

from triqto.actions import ActionEngineConfig, build_action_engine_result
from triqto.actions.parallel_pipeline import build_action_engine_result_parallel
from triqto.actions.sharded_artifacts import write_sharded_action_dataset
from triqto.actions.streaming_pipeline import (
    build_and_write_action_dataset_streaming,
)
from triqto.baselines import load_baseline_sources
from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    generate_dataset,
    write_dataset,
)
from triqto.graph import (
    GraphConversionConfig,
    convert_completed_dataset_to_graphs,
    write_graph_dataset,
)


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    phase7 = tmp_path / "phase7"
    phase8 = tmp_path / "phase8"
    config = DatasetGenerationConfig(
        dataset_name="optimized-phase9-source",
        base_seed=31,
        circuit_specs=[
            CircuitGenerationSpec(
                family="bell",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=2,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            )
        ],
        store_statevectors=False,
        max_samples=4,
    )
    write_dataset(generate_dataset(config), phase7)
    graph = convert_completed_dataset_to_graphs(
        phase7,
        GraphConversionConfig(include_supplemental_counts=False),
    )
    write_graph_dataset(graph, phase8)
    return phase7, phase8


def _action_config() -> ActionEngineConfig:
    return ActionEngineConfig(
        candidate_magnitudes=(0.2,),
        max_candidates_per_sample=64,
        max_edits_per_action=16,
        improvement_atol=1e-7,
    )


def test_parallel_phase9_is_logically_identical(tmp_path: Path) -> None:
    phase7, phase8 = _sources(tmp_path)
    serial = build_action_engine_result(phase7, phase8, _action_config())
    parallel = build_action_engine_result_parallel(
        phase7,
        phase8,
        _action_config(),
        workers=2,
        progress_every=1,
    )
    assert [item.action_id for item in parallel.candidates] == [
        item.action_id for item in serial.candidates
    ]
    assert [item.content_hash for item in parallel.candidates] == [
        item.content_hash for item in serial.candidates
    ]
    assert [item.rollout_id for item in parallel.rollouts] == [
        item.rollout_id for item in serial.rollouts
    ]
    assert [item.content_hash for item in parallel.rollouts] == [
        item.content_hash for item in serial.rollouts
    ]
    assert parallel.action_engine_id == serial.action_engine_id


def test_sharded_phase9_roundtrip_uses_bounded_file_count(tmp_path: Path) -> None:
    phase7, phase8 = _sources(tmp_path)
    result = build_action_engine_result_parallel(
        phase7,
        phase8,
        _action_config(),
        workers=2,
    )
    phase9 = tmp_path / "phase9"
    write = write_sharded_action_dataset(result, phase9, shard_count=4)
    marker = json.loads((phase9 / "action_complete.json").read_text())
    shard_files = sorted((phase9 / "artifacts" / "shards").glob("*.zip"))
    assert marker["complete"] is True
    assert shard_files
    assert len(shard_files) <= 4
    assert len(write.artifact_paths) == len(shard_files)
    assert len(write.artifact_paths) < len(result.candidates)
    loaded = load_baseline_sources(phase7, phase8, phase9).action
    assert all(
        ".zip#actions/" in record.action_ref
        for record in loaded.candidate_records
    )
    assert all(
        ".zip#circuits/" in record.circuit_ref
        for record in loaded.candidate_records
    )
    assert all(
        ".zip#rollouts/" in record.rollout_ref
        for record in loaded.rollout_records
    )


def test_streaming_phase9_matches_serial_without_full_result_accumulation(
    tmp_path: Path,
) -> None:
    phase7, phase8 = _sources(tmp_path)
    serial = build_action_engine_result(phase7, phase8, _action_config())
    phase9 = tmp_path / "phase9-streaming"
    progress: list[dict] = []
    write = build_and_write_action_dataset_streaming(
        phase7,
        phase8,
        phase9,
        _action_config(),
        workers=2,
        shard_count=4,
        progress_callback=progress.append,
        progress_every=1,
    )
    loaded = load_baseline_sources(phase7, phase8, phase9).action

    assert sorted(loaded.candidates_by_id) == [
        item.action_id for item in serial.candidates
    ]
    assert [
        loaded.candidates_by_id[item.action_id].content_hash
        for item in serial.candidates
    ] == [item.content_hash for item in serial.candidates]
    assert sorted(loaded.rollouts_by_id) == sorted(
        item.rollout_id for item in serial.rollouts
    )
    assert {
        rollout_id: rollout.content_hash
        for rollout_id, rollout in loaded.rollouts_by_id.items()
    } == {
        rollout.rollout_id: rollout.content_hash
        for rollout in serial.rollouts
    }
    assert loaded.summary["action_engine_id"] == serial.action_engine_id
    assert loaded.summary["streaming_bounded_memory"] is True
    assert write.candidate_count == len(serial.candidates)
    assert write.rollout_count == len(serial.rollouts)
    assert len(write.artifact_paths) <= 4
    assert progress
    assert progress[-1]["completed_samples"] == len(
        loaded.rollouts_by_sample_id
    )
    assert not phase9.with_name(f".{phase9.name}.streaming").exists()
