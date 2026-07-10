from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from qiskit import qpy

from triqto.circuits.families import generate_circuit_family as real_generate_circuit_family
from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    config_from_dict,
    config_to_dict,
    derive_child_seed,
    generate_dataset,
    verify_dataset_references,
    write_dataset,
)
from triqto.metrics import BornMetricBundle, compare_born_distributions
from triqto.storage import CircuitRecord, ManifestReader
from triqto.storage.schema import DatasetSampleRecord


def base_config(**overrides: Any) -> DatasetGenerationConfig:
    values = {
        "dataset_name": "tiny",
        "base_seed": 11,
        "circuit_specs": [
            CircuitGenerationSpec(
                family="hardware_efficient_ansatz",
                n_qubits=2,
                generator_kwargs={"layers": 1, "entanglement": "none", "measure": True},
                repetitions=1,
            )
        ],
        "distortion_specs": [
            DistortionSpec(name="rx_overrotation", kwargs={"strength": 0.3, "qubits": [0]}),
            DistortionSpec(name="readout_bitflip_marker", kwargs={"probability": 0.1, "qubits": [0]}),
        ],
        "max_samples": 10,
    }
    values.update(overrides)
    return DatasetGenerationConfig(**values)


def sample_signature(result):
    return [
        {
            "sample_id": sample.sample_id,
            "clean_circuit_id": sample.clean_circuit_id,
            "distorted_circuit_id": sample.distorted_circuit_id,
            "clean_run_id": sample.clean_run_id,
            "distorted_run_id": sample.distorted_run_id,
            "distortion_id": sample.distortion_id,
            "metric_id": sample.metric_id,
            "parameter_bindings": sample.parameter_bindings,
            "clean_probabilities": sample.clean_result.probabilities,
            "distorted_probabilities": sample.distorted_result.probabilities,
        }
        for sample in sorted(result.samples, key=lambda item: item.metadata["distortion_name"])
    ]


def normalize_parameter(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Unsupported nonfinite circuit parameter in test: {value!r}")
        return numeric
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"Unsupported nonfinite complex circuit parameter in test: {value!r}")
        return [float(value.real), float(value.imag)]
    if value is None or isinstance(value, str):
        return value
    return str(value)


def circuit_structure(circuit):
    instructions = []
    for instruction in circuit.data:
        operation = instruction.operation
        instructions.append(
            {
                "name": operation.name,
                "params": [normalize_parameter(param) for param in operation.params],
                "qubits": [circuit.find_bit(qubit).index for qubit in instruction.qubits],
                "clbits": [circuit.find_bit(clbit).index for clbit in instruction.clbits],
                "condition": normalize_parameter(getattr(operation, "condition", None)),
            }
        )
    return {
        "n_qubits": circuit.num_qubits,
        "n_clbits": circuit.num_clbits,
        "global_phase": normalize_parameter(circuit.global_phase),
        "parameters": sorted(parameter.name for parameter in circuit.parameters),
        "instructions": instructions,
    }


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        if isinstance(value, dict):
            return {key: normalize(val) for key, val in sorted(value.items())}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return sorted((normalize(row) for row in rows), key=lambda row: json.dumps(row, sort_keys=True, default=str))


def test_config_validation_and_roundtrip() -> None:
    with pytest.raises(ValueError):
        base_config(dataset_name="  ")
    with pytest.raises(ValueError):
        base_config(circuit_specs=[])
    with pytest.raises(ValueError):
        base_config(distortion_specs=[])
    with pytest.raises(ValueError):
        base_config(max_samples=1)
    with pytest.raises(ValueError):
        CircuitGenerationSpec("bell", 2, {}, 0)
    config = base_config()
    assert config_from_dict(config_to_dict(config)) == config
    with pytest.raises(ValueError):
        config_from_dict({**config_to_dict(config), "split": "nope"})
    with pytest.raises(ValueError):
        config_from_dict({**config_to_dict(config), "circuit_specs": [{"family": "bell", "n_qubits": 2, "bad": True}]})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": ""},
        {"parameter_low": float("nan")},
        {"parameter_high": float("inf")},
        {"parameter_low": float("-inf")},
        {"base_seed": True},
        {"ideal_shots": True},
        {"max_samples": True},
        {"born_zero_atol": -1.0},
    ],
)
def test_dataset_config_rejects_malformed_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        base_config(**kwargs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CircuitGenerationSpec(" ", 2, {}),
        lambda: CircuitGenerationSpec("bell", True, {}),
        lambda: CircuitGenerationSpec("bell", 2, [], 1),
        lambda: CircuitGenerationSpec("bell", 2, {}, True),
        lambda: DistortionSpec(" ", {}),
        lambda: DistortionSpec("rx_overrotation", []),
    ],
)
def test_nested_specs_reject_malformed_values(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_seed_derivation_deterministic_namespaced_and_validated() -> None:
    seed = derive_child_seed(1, "parameter_binding", {"x": 1})
    assert seed == derive_child_seed(1, "parameter_binding", {"x": 1})
    assert seed != derive_child_seed(1, "clean_shots", {"x": 1})
    with pytest.raises(TypeError):
        derive_child_seed(True, "x", {})
    with pytest.raises(ValueError):
        derive_child_seed(1, " ", {})


def test_generation_ids_parameters_counts_metric_metadata_and_marker_honesty() -> None:
    first = generate_dataset(base_config())
    second = generate_dataset(base_config())
    different_seed = generate_dataset(base_config(base_seed=12))
    assert sample_signature(first) == sample_signature(second)
    assert [sample.parameter_bindings for sample in first.samples] != [sample.parameter_bindings for sample in different_seed.samples]
    assert len(first.samples) == 2
    assert first.summary["unique_clean_circuit_count"] == 1
    for sample in first.samples:
        assert not sample.clean_circuit.parameters
        assert not sample.distorted_circuit.parameters
        assert sample.clean_result.probabilities
        assert sample.distorted_result.probabilities
        assert isinstance(sample.born_metrics, BornMetricBundle)
    visible = next(sample for sample in first.samples if sample.metadata["distortion_name"] == "rx_overrotation")
    assert visible.born_metrics.metrics["total_variation"].value > 0
    marker = next(sample for sample in first.samples if sample.metadata["marker_only"])
    assert marker.clean_result.probabilities == marker.distorted_result.probabilities
    assert marker.born_metrics.metadata.get("applicability_warning")
    assert marker.metadata["born_zero_shift"] is True
    for record in first.metric_records:
        assert record.born_metrics
        assert record.hilbert_metrics == {}
        assert record.parameter_metrics == {}
        assert record.topology_metrics == {}
        assert record.hilbert_available_mask is False
        assert record.metadata["computed_metric_families"] == ["born"]


def test_simulation_backend_metadata_for_exact_and_shot_records() -> None:
    result = generate_dataset(base_config(ideal_shots=8))
    exact = [record for record in result.simulation_records if record.simulation_mode == "ideal_statevector"]
    shots = [record for record in result.simulation_records if record.simulation_mode == "ideal_shot"]
    assert exact and shots
    assert {record.backend_name for record in exact} == {"qiskit.quantum_info.Statevector"}
    assert {record.metadata["sampling_source"] for record in exact} == {"exact_statevector"}
    assert {record.backend_name for record in shots} == {"triqto.ideal_probability_sampler"}
    assert {record.metadata["sampling_source"] for record in shots} == {"sampled_from_exact_born_probabilities"}
    assert all(record.metadata.get("source_run_id") for record in shots)
    assert all(record.metadata.get("seed") is not None for record in shots)


def test_born_zero_tolerance_labels_without_changing_identities() -> None:
    exact_zero = generate_dataset(
        base_config(distortion_specs=[DistortionSpec("phase_rz_drift", {"strength": 0.4, "qubits": [0]})])
    )
    assert exact_zero.samples[0].metadata["born_zero_shift"] is True
    tiny_config = base_config(
        distortion_specs=[DistortionSpec("rx_overrotation", {"strength": 1e-8, "qubits": [0]})],
        born_zero_atol=1.0,
    )
    visible_config = base_config(
        distortion_specs=[DistortionSpec("rx_overrotation", {"strength": 1e-8, "qubits": [0]})],
        born_zero_atol=0.0,
    )
    tiny = generate_dataset(tiny_config)
    visible = generate_dataset(visible_config)
    assert tiny.samples[0].metadata["born_zero_shift"] is True
    assert visible.samples[0].metadata["born_zero_shift"] is False
    assert tiny.samples[0].clean_circuit_id == visible.samples[0].clean_circuit_id
    assert tiny.samples[0].distorted_circuit_id == visible.samples[0].distorted_circuit_id
    assert tiny.samples[0].born_metrics.metrics["total_variation"].value == visible.samples[0].born_metrics.metrics["total_variation"].value


def test_storage_execution_invariance_and_distortion_extension() -> None:
    baseline = generate_dataset(base_config(dataset_name="dataset_a", store_statevectors=True, ideal_shots=None, max_samples=10))
    variants = [
        base_config(dataset_name="dataset_a", store_statevectors=False, ideal_shots=None, max_samples=10),
        base_config(dataset_name="dataset_a", store_statevectors=True, ideal_shots=32, max_samples=10),
        base_config(dataset_name="dataset_a", store_statevectors=True, ideal_shots=None, max_samples=1000),
        base_config(dataset_name="dataset_b", store_statevectors=True, ideal_shots=None, max_samples=10),
    ]
    for variant_config in variants:
        variant = generate_dataset(variant_config)
        assert sample_signature(variant) == sample_signature(baseline)
        assert variant.scientific_generation_id == baseline.scientific_generation_id
    assert len({generate_dataset(config).config_id for config in variants}) > 1

    extended = generate_dataset(
        base_config(
            distortion_specs=base_config().distortion_specs
            + [DistortionSpec("phase_rz_drift", {"strength": 0.2, "qubits": [0]})],
            max_samples=10,
        )
    )
    baseline_by_distortion = {sample.metadata["distortion_name"]: sample for sample in baseline.samples}
    extended_by_distortion = {sample.metadata["distortion_name"]: sample for sample in extended.samples}
    for name in baseline_by_distortion:
        assert baseline_by_distortion[name].clean_circuit_id == extended_by_distortion[name].clean_circuit_id
        assert baseline_by_distortion[name].parameter_bindings == extended_by_distortion[name].parameter_bindings
        assert baseline_by_distortion[name].sample_id == extended_by_distortion[name].sample_id
    assert len(extended.samples) == len(baseline.samples) + 1
    assert extended.scientific_generation_id != baseline.scientific_generation_id


def test_original_generated_circuit_is_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    import triqto.data_generation.pipeline as pipeline

    captured = {}

    def capturing_generator(family: str, n_qubits: int, **kwargs: Any):
        generated = real_generate_circuit_family(family, n_qubits, **kwargs)
        captured["generated"] = generated
        captured["circuit_object"] = generated.circuit
        captured["parameters"] = sorted(parameter.name for parameter in generated.circuit.parameters)
        captured["structure"] = circuit_structure(generated.circuit)
        return generated

    monkeypatch.setattr(pipeline, "generate_circuit_family", capturing_generator)
    result = generate_dataset(base_config())
    generated = captured["generated"]
    assert generated.circuit is captured["circuit_object"]
    assert sorted(parameter.name for parameter in generated.circuit.parameters) == captured["parameters"]
    assert circuit_structure(generated.circuit) == captured["structure"]
    assert result.samples[0].clean_circuit is not generated.circuit


def test_dataset_sample_record_roundtrip() -> None:
    record = DatasetSampleRecord("s", "d", "v", "c", "dc", "r", "dr", "dist", "m", "bell", 2, 0, {"x": 1.0}, 5, {})
    record.validate()
    assert DatasetSampleRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError):
        DatasetSampleRecord("", "d", "v", "c", "dc", "r", "dr", "dist", "m", "bell", 0, 0, {}, 0, {}).validate()


def test_duplicate_conflicting_ids_detected() -> None:
    import triqto.data_generation.pipeline as pipeline

    first = CircuitRecord("same", "bell", 2, 2, 1, 0, 0, {"role": "clean"})
    second = CircuitRecord("same", "ghz", 2, 2, 1, 0, 0, {"role": "clean"})
    records = {}
    pipeline._add_unique_record(records, first, "same")
    with pytest.raises(ValueError):
        pipeline._add_unique_record(records, second, "same")


def test_write_dataset_artifact_paths_contract_and_readback(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    result = generate_dataset(base_config(ideal_shots=8))
    write_result = write_dataset(result, tmp_path / "dataset")
    assert set(write_result.artifact_paths) == {"circuits", "probabilities", "statevectors", "counts"}
    assert len(write_result.artifact_paths["circuits"]) == len(result.circuit_records)
    assert len(write_result.artifact_paths["probabilities"]) == len({sample.clean_run_id for sample in result.samples} | {sample.distorted_run_id for sample in result.samples})
    assert len(write_result.artifact_paths["statevectors"]) == len(write_result.artifact_paths["probabilities"])
    assert len(write_result.artifact_paths["counts"]) == len([record for record in result.simulation_records if record.simulation_mode == "ideal_shot"])
    assert len(write_result.written_paths) == len(set(write_result.written_paths))
    assert all(path.exists() for path in write_result.written_paths)
    _assert_artifact_readback(result, write_result.output_root)
    with pytest.raises(FileExistsError):
        write_dataset(result, tmp_path / "dataset")


def test_write_dataset_no_optional_artifact_categories(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    result = generate_dataset(base_config(ideal_shots=None, store_statevectors=False))
    write_result = write_dataset(result, tmp_path / "dataset")
    assert write_result.artifact_paths["statevectors"] == []
    assert write_result.artifact_paths["counts"] == []
    assert all(path.exists() for path in write_result.written_paths)


def _assert_artifact_readback(result, root: Path) -> None:
    reader = ManifestReader(root / "manifests")
    sample_rows = normalize_rows(reader.read_records("sample_manifest"))
    circuit_rows = normalize_rows(reader.read_records("circuit_manifest"))
    simulation_rows = normalize_rows(reader.read_records("simulation_manifest"))
    distortion_rows = normalize_rows(reader.read_records("distortion_manifest"))
    metric_rows = normalize_rows(reader.read_records("metric_manifest"))
    circuit_ids = {row["circuit_id"] for row in circuit_rows}
    run_ids = {row["run_id"] for row in simulation_rows}
    distortion_ids = {row["distortion_id"] for row in distortion_rows}
    metric_ids = {row["metric_id"] for row in metric_rows}
    for row in sample_rows:
        assert row["clean_circuit_id"] in circuit_ids
        assert row["distorted_circuit_id"] in circuit_ids
        assert row["clean_run_id"] in run_ids
        assert row["distorted_run_id"] in run_ids
        assert row["distortion_id"] in distortion_ids
        assert row["metric_id"] in metric_ids
    samples_by_run = {}
    expected_counts_by_run = {}
    for sample in result.samples:
        samples_by_run[sample.clean_run_id] = (sample.clean_result.probabilities, sample.clean_result.statevector.data)
        samples_by_run[sample.distorted_run_id] = (sample.distorted_result.probabilities, sample.distorted_result.statevector.data)
        if sample.clean_shot_run_id and sample.clean_shot_result is not None:
            expected_counts_by_run[sample.clean_shot_run_id] = sample.clean_shot_result.counts
        if sample.distorted_shot_run_id and sample.distorted_shot_result is not None:
            expected_counts_by_run[sample.distorted_shot_run_id] = sample.distorted_shot_result.counts
    for row in simulation_rows:
        for key in ("probabilities_ref", "statevector_ref", "counts_ref"):
            if row.get(key) is not None:
                assert not Path(row[key]).is_absolute()
                assert (root / row[key]).exists()
        if row["simulation_mode"] == "ideal_statevector":
            probabilities = json.loads((root / row["probabilities_ref"]).read_text())
            assert probabilities == samples_by_run[row["run_id"]][0]
            if row.get("statevector_ref") is not None:
                np.testing.assert_allclose(np.load(root / row["statevector_ref"]), samples_by_run[row["run_id"]][1])
        elif row["simulation_mode"] == "ideal_shot":
            counts = json.loads((root / row["counts_ref"]).read_text())
            assert counts == expected_counts_by_run[row["run_id"]]
            assert all(isinstance(key, str) for key in counts)
            assert all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts.values())
            assert sum(counts.values()) == result.config.ideal_shots
    expected_structures = {sample.clean_circuit_id: circuit_structure(sample.clean_circuit) for sample in result.samples}
    expected_structures.update({sample.distorted_circuit_id: circuit_structure(sample.distorted_circuit) for sample in result.samples})
    for row in circuit_rows:
        with (root / row["metadata"]["artifact_ref"]).open("rb") as handle:
            loaded_circuits = qpy.load(handle)
        assert len(loaded_circuits) == 1
        loaded = loaded_circuits[0]
        assert circuit_structure(loaded) == expected_structures[row["circuit_id"]]


def test_missing_and_absolute_references_raise_explicit_errors(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    result = generate_dataset(base_config())
    write_result = write_dataset(result, tmp_path / "dataset")
    reader = ManifestReader(write_result.output_root / "manifests")
    circuit_records = reader.read_typed_records("circuit_manifest", CircuitRecord)
    simulation_records = reader.read_typed_records("simulation_manifest", type(result.simulation_records[0]))
    missing_record = copy.deepcopy(circuit_records[0])
    missing_record.metadata["artifact_ref"] = "artifacts/circuits/missing.qpy"
    with pytest.raises(FileNotFoundError, match=missing_record.circuit_id):
        verify_dataset_references(write_result.output_root, [missing_record], simulation_records, require_statevectors=result.config.store_statevectors)
    absolute_record = copy.deepcopy(circuit_records[0])
    absolute_record.metadata["artifact_ref"] = str((tmp_path / "absolute.qpy").absolute())
    with pytest.raises(ValueError, match=absolute_record.circuit_id):
        verify_dataset_references(write_result.output_root, [absolute_record], simulation_records, require_statevectors=result.config.store_statevectors)


def test_qpy_lazy_failure_keeps_in_memory_generation_usable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import triqto.data_generation.artifacts as artifacts

    result = generate_dataset(base_config())
    assert result.samples
    monkeypatch.setattr(
        artifacts,
        "_load_qpy_module",
        lambda: (_ for _ in ()).throw(RuntimeError("Qiskit QPY support is required to persist circuit artifacts.")),
    )
    with pytest.raises(RuntimeError, match="Qiskit QPY support is required"):
        write_dataset(result, tmp_path / "dataset")


def test_logical_reproducibility_across_output_roots(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    result = generate_dataset(base_config(ideal_shots=8))
    first = write_dataset(result, tmp_path / "a")
    second = write_dataset(result, tmp_path / "b")
    assert json.loads(first.config_path.read_text()) == json.loads(second.config_path.read_text())
    assert json.loads(first.summary_path.read_text()) == json.loads(second.summary_path.read_text())
    for manifest_name in first.manifest_paths:
        left = normalize_rows(ManifestReader(first.output_root / "manifests").read_records(manifest_name))
        right = normalize_rows(ManifestReader(second.output_root / "manifests").read_records(manifest_name))
        assert left == right
    _compare_artifact_trees(first.output_root, second.output_root)


def _compare_artifact_trees(left_root: Path, right_root: Path) -> None:
    for left_path in sorted((left_root / "artifacts" / "probabilities").glob("*.json")):
        assert json.loads(left_path.read_text()) == json.loads((right_root / left_path.relative_to(left_root)).read_text())
    for left_path in sorted((left_root / "artifacts" / "counts").glob("*.json")):
        assert json.loads(left_path.read_text()) == json.loads((right_root / left_path.relative_to(left_root)).read_text())
    for left_path in sorted((left_root / "artifacts" / "statevectors").glob("*.npy")):
        np.testing.assert_allclose(np.load(left_path), np.load(right_root / left_path.relative_to(left_root)))
    for left_path in sorted((left_root / "artifacts" / "circuits").glob("*.qpy")):
        with left_path.open("rb") as left_handle, (right_root / left_path.relative_to(left_root)).open("rb") as right_handle:
            assert circuit_structure(qpy.load(left_handle)[0]) == circuit_structure(qpy.load(right_handle)[0])


def test_kl_infinity_encoded_and_no_aer_import() -> None:
    import triqto.data_generation.pipeline as pipeline

    values = pipeline._metric_values(compare_born_distributions({"0": 1.0}, {"1": 1.0}))
    assert values["kl_clean_to_distorted"] is None
    assert values["kl_clean_to_distorted__nonfinite"] == "positive_infinity"
    json.dumps(values, allow_nan=False)
    assert "qiskit_aer" not in sys.modules


def test_metric_manifest_typed_readback_preserves_empty_metric_maps(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    result = generate_dataset(base_config())
    write_result = write_dataset(result, tmp_path / "dataset")
    raw_rows = ManifestReader(write_result.output_root / "manifests").read_records("metric_manifest")
    assert raw_rows
    typed = ManifestReader(write_result.output_root / "manifests").read_typed_records("metric_manifest", type(result.metric_records[0]))
    original_by_id = {record.metric_id: record for record in result.metric_records}
    for record in typed:
        original = original_by_id[record.metric_id]
        assert record.born_metrics == original.born_metrics
        assert record.hilbert_metrics == {}
        assert record.parameter_metrics == {}
        assert record.topology_metrics == {}
        assert record.hilbert_available_mask is False
        assert record.hilbert_metrics.get("missing") is None


def test_reference_verification_required_fields_and_path_safety(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    result = generate_dataset(base_config(ideal_shots=4))
    write_result = write_dataset(result, tmp_path / "dataset")
    reader = ManifestReader(write_result.output_root / "manifests")
    circuits = reader.read_typed_records("circuit_manifest", CircuitRecord)
    simulations = reader.read_typed_records("simulation_manifest", type(result.simulation_records[0]))
    verify_dataset_references(write_result.output_root, circuits, simulations, require_statevectors=True)

    missing_artifact_ref = copy.deepcopy(circuits[0])
    missing_artifact_ref.metadata.pop("artifact_ref")
    with pytest.raises(ValueError, match="artifact_ref"):
        verify_dataset_references(write_result.output_root, [missing_artifact_ref], simulations, require_statevectors=True)

    directory_ref = copy.deepcopy(circuits[0])
    directory_ref.metadata["artifact_ref"] = "artifacts/circuits"
    with pytest.raises(ValueError, match="artifact_ref"):
        verify_dataset_references(write_result.output_root, [directory_ref], simulations, require_statevectors=True)

    for bad_ref in ["../outside.json", "artifacts/../../outside.json", str((tmp_path / "outside.json").absolute()), "", "."]:
        bad_record = copy.deepcopy(circuits[0])
        bad_record.metadata["artifact_ref"] = bad_ref
        with pytest.raises(ValueError, match=bad_record.circuit_id):
            verify_dataset_references(write_result.output_root, [bad_record], simulations, require_statevectors=True)

    statevector = next(record for record in simulations if record.simulation_mode == "ideal_statevector")
    no_prob = copy.deepcopy(statevector)
    no_prob.probabilities_ref = None
    with pytest.raises(ValueError, match="probabilities_ref"):
        verify_dataset_references(write_result.output_root, circuits, [no_prob], require_statevectors=True)
    missing_prob = copy.deepcopy(statevector)
    missing_prob.probabilities_ref = "artifacts/probabilities/missing.json"
    with pytest.raises(FileNotFoundError, match="probabilities_ref"):
        verify_dataset_references(write_result.output_root, circuits, [missing_prob], require_statevectors=True)
    no_statevector = copy.deepcopy(statevector)
    no_statevector.statevector_ref = None
    with pytest.raises(ValueError, match="statevector_ref"):
        verify_dataset_references(write_result.output_root, circuits, [no_statevector], require_statevectors=True)
    missing_statevector = copy.deepcopy(statevector)
    missing_statevector.statevector_ref = "artifacts/statevectors/missing.npy"
    with pytest.raises(FileNotFoundError, match="statevector_ref"):
        verify_dataset_references(write_result.output_root, circuits, [missing_statevector], require_statevectors=True)
    with_statevector_when_disabled = copy.deepcopy(statevector)
    with pytest.raises(ValueError, match="statevector_ref"):
        verify_dataset_references(write_result.output_root, circuits, [with_statevector_when_disabled], require_statevectors=False)

    shot = next(record for record in simulations if record.simulation_mode == "ideal_shot")
    no_counts = copy.deepcopy(shot)
    no_counts.counts_ref = None
    with pytest.raises(ValueError, match="counts_ref"):
        verify_dataset_references(write_result.output_root, circuits, [no_counts], require_statevectors=True)
    missing_counts = copy.deepcopy(shot)
    missing_counts.counts_ref = "artifacts/counts/missing.json"
    with pytest.raises(FileNotFoundError, match="counts_ref"):
        verify_dataset_references(write_result.output_root, circuits, [missing_counts], require_statevectors=True)
    no_source = copy.deepcopy(shot)
    no_source.metadata.pop("source_run_id")
    with pytest.raises(ValueError, match="source_run_id"):
        verify_dataset_references(write_result.output_root, circuits, [no_source], require_statevectors=True)
    unknown = copy.deepcopy(shot)
    unknown.simulation_mode = "hardware"
    with pytest.raises(ValueError, match="simulation_mode"):
        verify_dataset_references(write_result.output_root, circuits, [unknown], require_statevectors=True)


def test_nonfinite_metric_encoding_and_decoding_strictness() -> None:
    import triqto.data_generation.pipeline as pipeline
    from triqto.metrics.results import BornMetricBundle, BornMetricResult

    finite = {"total_variation": 0.25}
    assert pipeline._decode_metric_values(finite) == {"total_variation": 0.25}
    encoded = pipeline._metric_values(compare_born_distributions({"0": 1.0}, {"1": 1.0}))
    decoded = pipeline._decode_metric_values(encoded)
    assert math.isinf(decoded["kl_clean_to_distorted"])
    json.dumps(encoded, allow_nan=False)

    def bundle(value: Any) -> BornMetricBundle:
        return BornMetricBundle(
            metric_family="born",
            support=["0"],
            metrics={
                "bad_metric": BornMetricResult(
                    metric_name="bad_metric",
                    metric_family="born",
                    value=value,
                    lower_is_better=True,
                    symmetric=False,
                    bounded=False,
                    value_range=(0.0, None),
                )
            },
        )

    with pytest.raises(ValueError, match="bad_metric"):
        pipeline._metric_values(bundle(float("nan")))
    with pytest.raises(ValueError, match="bad_metric"):
        pipeline._metric_values(bundle(float("-inf")))
    with pytest.raises(TypeError, match="bad_metric"):
        pipeline._metric_values(bundle("1.0"))
    with pytest.raises(ValueError, match="null"):
        pipeline._decode_metric_values({"x": None})
    with pytest.raises(ValueError, match="unknown"):
        pipeline._decode_metric_values({"x": None, "x__nonfinite": "nan"})
    with pytest.raises(ValueError, match="Orphan"):
        pipeline._decode_metric_values({"x__nonfinite": "positive_infinity"})


def test_strict_config_validation_and_json_loading(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        base_config(store_statevectors=1)
    with pytest.raises(TypeError):
        base_config(parameter_low="0.25")
    with pytest.raises(TypeError):
        base_config(born_zero_atol=True)
    with pytest.raises(TypeError):
        base_config(base_seed=2.0)
    with pytest.raises(TypeError):
        base_config(ideal_shots=4.0)
    with pytest.raises(TypeError):
        CircuitGenerationSpec(" bell ", 2.0, {})
    config = base_config(dataset_name=" tiny ", schema_version=" triqto.phase7.v1 ")
    assert config.dataset_name == "tiny"
    assert config.schema_version == "triqto.phase7.v1"
    original_circuit_specs = [CircuitGenerationSpec("bell", 2, {}, 1)]
    original_distortion_specs = [DistortionSpec("rx_overrotation", {"strength": 0.1})]
    config = DatasetGenerationConfig("copy", 1, original_circuit_specs, original_distortion_specs)
    original_circuit_specs.clear()
    original_distortion_specs.clear()
    assert config.circuit_specs and config.distortion_specs
    with pytest.raises(TypeError):
        config_from_dict([])  # type: ignore[arg-type]
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"dataset_name":"x","base_seed":NaN}')
    from triqto.data_generation import load_generation_config
    with pytest.raises(ValueError, match="Invalid non-finite JSON constant"):
        load_generation_config(bad_json)


def test_typed_manifest_roundtrips_and_semantic_joins(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    from triqto.storage import DistortionRecord, MetricRecord, SimulationRecord
    from triqto.data_generation import validate_dataset_joins

    result = generate_dataset(base_config(ideal_shots=8))
    write_result = write_dataset(result, tmp_path / "dataset")
    reader = ManifestReader(write_result.output_root / "manifests")
    sample_records = reader.read_typed_records("sample_manifest", DatasetSampleRecord)
    circuit_records = reader.read_typed_records("circuit_manifest", CircuitRecord)
    simulation_records = reader.read_typed_records("simulation_manifest", SimulationRecord)
    distortion_records = reader.read_typed_records("distortion_manifest", DistortionRecord)
    metric_records = reader.read_typed_records("metric_manifest", MetricRecord)
    for records in [sample_records, circuit_records, simulation_records, distortion_records, metric_records]:
        for record in records:
            record.validate()
    validate_dataset_joins(sample_records, circuit_records, simulation_records, distortion_records, metric_records)
    original_samples = {record.sample_id: normalize_rows([record.to_dict()])[0] for record in result.sample_records}
    for record in sample_records:
        typed_dict = normalize_rows([record.to_dict()])[0]
        original = original_samples[record.sample_id]
        assert typed_dict["sample_id"] == original["sample_id"]
        assert typed_dict["clean_circuit_id"] == original["clean_circuit_id"]
        assert typed_dict["distorted_circuit_id"] == original["distorted_circuit_id"]
        assert typed_dict["parameter_bindings"] == original["parameter_bindings"]
        assert typed_dict["metadata"]["born_zero_atol"] == result.config.born_zero_atol
    for record in simulation_records:
        if record.simulation_mode == "ideal_statevector":
            assert record.probabilities_ref
            assert record.metadata["sampling_source"] == "exact_statevector"
        if record.simulation_mode == "ideal_shot":
            assert record.counts_ref
            assert record.metadata["sampling_source"] == "sampled_from_exact_born_probabilities"
    for record in metric_records:
        assert record.hilbert_metrics == {}
        assert record.parameter_metrics == {}
        assert record.topology_metrics == {}
        assert record.born_metrics


def test_write_dataset_failure_cleanup_preserves_unrelated_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import triqto.data_generation.artifacts as artifacts

    result = generate_dataset(base_config())
    target = tmp_path / "dataset"
    target.mkdir()
    unrelated = target / "unrelated.txt"
    unrelated.write_text("keep me")
    monkeypatch.setattr(
        artifacts,
        "_load_qpy_module",
        lambda: (_ for _ in ()).throw(RuntimeError("QPY forced failure")),
    )
    with pytest.raises(RuntimeError, match="QPY forced failure"):
        write_dataset(result, target, overwrite=True)
    assert unrelated.read_text() == "keep me"
    assert not (target / "dataset_complete.json").exists()


def test_write_dataset_manifest_failure_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import triqto.data_generation.artifacts as artifacts

    result = generate_dataset(base_config())

    def fail_write_records(self, *args, **kwargs):
        raise RuntimeError("manifest forced failure")

    monkeypatch.setattr(artifacts.ManifestWriter, "write_records", fail_write_records)
    with pytest.raises(RuntimeError, match="manifest forced failure"):
        write_dataset(result, tmp_path / "dataset")
    assert not (tmp_path / "dataset" / "dataset_complete.json").exists()
    assert not any(tmp_path.glob(".dataset.triqto-staging-*"))
