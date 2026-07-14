from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triqto.actions import (
    ActionEngineConfig,
    build_action_engine_result,
    write_action_dataset,
)
from triqto.baselines import (
    BASELINE_NAMES,
    BaselineSuiteConfig,
    baseline_config_from_dict,
    baseline_config_to_dict,
    load_baseline_result_artifact,
    run_baseline_suite,
    validate_baseline_result,
    write_baseline_dataset,
)
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
    snapshot_managed_files,
    write_graph_dataset,
)
from triqto.storage import BaselineResultRecord, ManifestReader


def build_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    phase7_root = tmp_path / "phase7"
    graph_root = tmp_path / "phase8"
    action_root = tmp_path / "phase9"
    generation_config = DatasetGenerationConfig(
        dataset_name="phase10-source",
        base_seed=71,
        circuit_specs=[
            CircuitGenerationSpec(
                family="bell",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=1,
            )
        ],
        distortion_specs=[
            DistortionSpec(
                name="rx_overrotation",
                kwargs={"strength": 0.2, "qubits": [0]},
            ),
            DistortionSpec(
                name="readout_bitflip",
                kwargs={"probability": 0.1, "qubits": [0]},
            ),
        ],
        store_statevectors=False,
        max_samples=4,
    )
    write_dataset(generate_dataset(generation_config), phase7_root)
    graph_result = convert_completed_dataset_to_graphs(
        phase7_root,
        GraphConversionConfig(include_supplemental_counts=False),
    )
    write_graph_dataset(graph_result, graph_root)
    action_config = ActionEngineConfig(
        candidate_magnitudes=(0.2,),
        max_candidates_per_sample=64,
        max_edits_per_action=16,
    )
    action_result = build_action_engine_result(
        phase7_root,
        graph_root,
        action_config,
    )
    write_action_dataset(action_result, action_root)
    return phase7_root, graph_root, action_root


def managed_snapshot(root: Path, marker_name: str):
    marker = json.loads((root / marker_name).read_text())
    return snapshot_managed_files(root, tuple(marker["managed_files"]))


def small_config(**overrides) -> BaselineSuiteConfig:
    values = {
        "enabled_baselines": BASELINE_NAMES,
        "random_seed": 17,
        "spsa_iterations": 2,
        "cobyla_maxiter": 12,
        "cobyla_initial_step": 0.2,
        "max_optimizer_dimensions": 32,
        "max_objective_evaluations": 100,
    }
    values.update(overrides)
    return BaselineSuiteConfig(**values)


def test_baseline_config_is_strict_and_roundtrips() -> None:
    config = small_config()
    assert baseline_config_from_dict(baseline_config_to_dict(config)) == config
    with pytest.raises(ValueError, match="Unknown baseline config fields"):
        baseline_config_from_dict({**baseline_config_to_dict(config), "extra": 1})
    with pytest.raises(TypeError, match="random_seed"):
        BaselineSuiteConfig(random_seed=True)
    with pytest.raises(TypeError, match="max_abs_angle"):
        BaselineSuiteConfig(max_abs_angle="3.14")
    with pytest.raises(ValueError, match="fixed Phase 10 baseline order"):
        BaselineSuiteConfig(enabled_baselines=("cobyla", "spsa"))
    with pytest.raises(ValueError, match="transpiler_optimization_level"):
        BaselineSuiteConfig(transpiler_optimization_level=4)


def test_full_baseline_suite_is_deterministic_and_truthful(tmp_path: Path) -> None:
    phase7_root, graph_root, action_root = build_sources(tmp_path)
    before = (
        managed_snapshot(phase7_root, "dataset_complete.json"),
        managed_snapshot(graph_root, "graph_complete.json"),
        managed_snapshot(action_root, "action_complete.json"),
    )
    config = small_config()
    first = run_baseline_suite(phase7_root, graph_root, action_root, config)
    second = run_baseline_suite(phase7_root, graph_root, action_root, config)

    assert len(first.results) == 2 * len(BASELINE_NAMES)
    assert first.summary["result_count"] == first.summary["expected_result_count"]
    assert first.summary["triqto_model_compared"] is False
    assert first.summary["hardware_aware_baseline_present"] is False
    assert first.summary["source_immutability_verified"] is True
    assert [
        (item.sample_id, item.baseline_name, item.selected_action_id)
        for item in first.results
    ] == [
        (item.sample_id, item.baseline_name, item.selected_action_id)
        for item in second.results
    ]

    by_sample: dict[str, list] = {}
    for result in first.results:
        validate_baseline_result(result, config)
        by_sample.setdefault(result.sample_id, []).append(result)
        assert result.metadata["learned_model_used"] is False
        assert result.metadata["evaluation_mode"] == "ideal_statevector"
        assert np.isclose(
            result.objective_improvement,
            result.objective_before - result.objective_after,
            atol=1e-15,
            rtol=0.0,
        )
    for results in by_sample.values():
        assert [item.baseline_name for item in results] == list(BASELINE_NAMES)

    rule_results = [
        item for item in first.results if item.baseline_name == "rule_only"
    ]
    fallback_values = {
        item.metadata["selection"]["fallback_to_no_op"] for item in rule_results
    }
    assert fallback_values == {False, True}
    assert any(item.success for item in rule_results)

    loss_results = [
        item for item in first.results if item.baseline_name == "loss_only"
    ]
    assert all(
        item.objective_after <= item.objective_before + config.improvement_atol
        for item in loss_results
    )
    transpiler_results = [
        item for item in first.results if item.baseline_name == "transpiler_only"
    ]
    assert all(
        np.isclose(
            item.objective_after,
            item.objective_before,
            atol=1e-10,
            rtol=0.0,
        )
        for item in transpiler_results
    )
    for name in ("spsa", "cobyla"):
        optimizer_results = [
            item for item in first.results if item.baseline_name == name
        ]
        assert all(item.parameter_vector.size > 0 for item in optimizer_results)
        assert all(item.evaluations > 0 for item in optimizer_results)

    after = (
        managed_snapshot(phase7_root, "dataset_complete.json"),
        managed_snapshot(graph_root, "graph_complete.json"),
        managed_snapshot(action_root, "action_complete.json"),
    )
    assert after == before


def test_baseline_dataset_roundtrip_and_immutable_publication(tmp_path: Path) -> None:
    phase7_root, graph_root, action_root = build_sources(tmp_path)
    config = small_config(
        enabled_baselines=(
            "random_correction",
            "rule_only",
            "loss_only",
            "transpiler_only",
        )
    )
    result = run_baseline_suite(phase7_root, graph_root, action_root, config)
    output = tmp_path / "phase10"
    written = write_baseline_dataset(result, output)

    marker = json.loads((output / "baseline_complete.json").read_text())
    assert marker["complete"] is True
    assert marker["result_count"] == len(result.results)
    assert marker["sample_count"] == 2
    assert marker["managed_files"] == sorted(marker["managed_files"])
    assert set(marker["managed_files"]) == {
        path.relative_to(output).as_posix() for path in written.written_paths
    }

    reader = ManifestReader(output / "manifests")
    records = reader.read_typed_records(
        "baseline_result_manifest", BaselineResultRecord
    )
    assert len(records) == len(result.results)
    loaded = []
    for record in records:
        record.validate()
        item = load_baseline_result_artifact(
            output / record.artifact_ref,
            config,
            record.content_hash,
        )
        loaded.append(item)
    assert {item.baseline_result_id for item in loaded} == {
        item.baseline_result_id for item in result.results
    }
    assert any(item.selected_action_id is None for item in loaded)
    assert any(item.selected_action_id is not None for item in loaded)

    with pytest.raises(FileExistsError):
        write_baseline_dataset(result, output)


def test_optimizer_guardrail_fails_instead_of_truncating(tmp_path: Path) -> None:
    phase7_root, graph_root, action_root = build_sources(tmp_path)
    config = small_config(
        enabled_baselines=("spsa",),
        max_optimizer_dimensions=1,
    )
    with pytest.raises(RuntimeError, match="max_optimizer_dimensions"):
        run_baseline_suite(phase7_root, graph_root, action_root, config)
