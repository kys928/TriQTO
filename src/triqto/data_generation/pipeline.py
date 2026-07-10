"""Phase 7 deterministic raw data generation pipeline."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import inspect
import json
import math
import random
from typing import Any

from qiskit import QuantumCircuit

from triqto.circuits.circuit_metadata import GeneratedCircuit, count_two_qubit_gates
from triqto.circuits.families import generate_circuit_family, get_circuit_family
from triqto.core.ids import make_circuit_id, make_deterministic_id, make_run_id, make_sample_id
from triqto.distortions import apply_distortion
from triqto.metrics import compare_born_distributions
from triqto.metrics.results import BornMetricBundle
from triqto.simulation import simulate_ideal_shots, simulate_ideal_statevector
from triqto.storage import CircuitRecord, DistortionRecord, MetricRecord, SimulationRecord
from triqto.storage.schema import DatasetSampleRecord

from .records import DatasetGenerationResult, GeneratedDatasetSample
from .seeding import derive_child_seed
from .specs import (
    DatasetGenerationConfig,
    PHASE7_METRIC_SCHEMA_VERSION,
    config_id,
    scientific_generation_id,
)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _circuit_scientific_payload(config: DatasetGenerationConfig, spec_index: int, repetition_index: int) -> dict[str, Any]:
    spec = config.circuit_specs[spec_index]
    return {
        "schema_version": config.schema_version,
        "base_seed": config.base_seed,
        "family": spec.family,
        "n_qubits": spec.n_qubits,
        "generator_kwargs": spec.generator_kwargs,
        "explicit_generator_seed": spec.generator_kwargs.get("seed"),
        "repetition_index": repetition_index,
        "parameter_low": config.parameter_low,
        "parameter_high": config.parameter_high,
    }


def _generator_accepts_seed(family: str) -> bool:
    return "seed" in inspect.signature(get_circuit_family(family)).parameters


def _generator_kwargs(family: str, kwargs: dict[str, Any], generation_seed: int) -> dict[str, Any]:
    prepared_kwargs = _json_copy(kwargs)
    if _generator_accepts_seed(family) and "seed" not in prepared_kwargs:
        prepared_kwargs["seed"] = generation_seed
    return prepared_kwargs


def _bind_parameters(
    generated: GeneratedCircuit,
    seed: int,
    low: float,
    high: float,
) -> tuple[QuantumCircuit, dict[str, float], dict[str, list[float]]]:
    copied_circuit = generated.circuit.copy()
    parameters = sorted(copied_circuit.parameters, key=lambda parameter: parameter.name)
    rng = random.Random(seed)
    bindings = {parameter.name: float(rng.uniform(low, high)) for parameter in parameters}
    assignment = {parameter: bindings[parameter.name] for parameter in parameters}
    bound_circuit = copied_circuit.assign_parameters(assignment, inplace=False) if assignment else copied_circuit
    encodings = {name: [math.sin(value), math.cos(value)] for name, value in bindings.items()}
    return bound_circuit, bindings, encodings


def _make_circuit_record(
    circuit_id: str,
    circuit: QuantumCircuit,
    family: str,
    metadata: dict[str, Any],
) -> CircuitRecord:
    return CircuitRecord(
        circuit_id=circuit_id,
        family=family,
        n_qubits=circuit.num_qubits,
        n_clbits=circuit.num_clbits,
        depth=circuit.depth(),
        two_qubit_gate_count=count_two_qubit_gates(circuit),
        parameter_count=len(circuit.parameters),
        metadata=_json_copy(metadata),
    )


def _make_simulation_record(
    run_id: str,
    circuit_id: str,
    simulation_mode: str,
    shots: int | None,
    metadata: dict[str, Any],
) -> SimulationRecord:
    if simulation_mode == "ideal_statevector":
        backend_name = "qiskit.quantum_info.Statevector"
        metadata = {"sampling_source": "exact_statevector", **metadata}
    elif simulation_mode == "ideal_shot":
        backend_name = "triqto.ideal_probability_sampler"
        metadata = {"sampling_source": "sampled_from_exact_born_probabilities", **metadata}
    else:
        backend_name = None
    return SimulationRecord(
        run_id=run_id,
        circuit_id=circuit_id,
        simulation_mode=simulation_mode,
        backend_name=backend_name,
        shots=shots,
        metadata=_json_copy(metadata),
    )


def _metric_values(bundle: BornMetricBundle) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, result in sorted(bundle.metrics.items()):
        metric_value = float(result.value)
        if math.isfinite(metric_value):
            values[name] = metric_value
        elif metric_value > 0:
            values[name] = None
            values[f"{name}__nonfinite"] = "positive_infinity"
        else:
            values[name] = None
            values[f"{name}__nonfinite"] = "negative_infinity"
    return values


def _add_unique_record(records_by_id: dict[str, Any], record: Any, record_id: str) -> None:
    existing = records_by_id.get(record_id)
    if existing is None:
        records_by_id[record_id] = record
        return
    if existing.to_dict() != record.to_dict():
        raise ValueError(f"Conflicting deterministic ID detected: {record_id}")


def _sorted_records(records_by_id: dict[str, Any]) -> list[Any]:
    return [records_by_id[key] for key in sorted(records_by_id)]


def generate_dataset(config: DatasetGenerationConfig) -> DatasetGenerationResult:
    """Generate deterministic in-memory Phase 7 raw clean/distorted Born samples."""
    operational_config_id = config_id(config)
    generation_id = scientific_generation_id(config)
    circuit_records: dict[str, CircuitRecord] = {}
    simulation_records: dict[str, SimulationRecord] = {}
    distortion_records: dict[str, DistortionRecord] = {}
    metric_records: dict[str, MetricRecord] = {}
    sample_ids: set[str] = set()
    sample_records: list[DatasetSampleRecord] = []
    samples: list[GeneratedDatasetSample] = []

    for spec_index, spec in enumerate(config.circuit_specs):
        for repetition_index in range(spec.repetitions):
            clean_payload = _circuit_scientific_payload(config, spec_index, repetition_index)
            generation_seed = derive_child_seed(config.base_seed, "circuit_generation", clean_payload)
            parameter_seed = derive_child_seed(config.base_seed, "parameter_binding", clean_payload)
            generator_kwargs = _generator_kwargs(spec.family, spec.generator_kwargs, generation_seed)
            generated = generate_circuit_family(spec.family, spec.n_qubits, **generator_kwargs)
            bound_circuit, parameter_bindings, parameter_sin_cos = _bind_parameters(
                generated,
                parameter_seed,
                config.parameter_low,
                config.parameter_high,
            )

            clean_circuit_id = make_circuit_id(
                {
                    **clean_payload,
                    "generation_seed": generation_seed,
                    "generator_kwargs_used": generator_kwargs,
                    "parameter_bindings": parameter_bindings,
                }
            )
            clean_run_id = make_run_id(
                {
                    "circuit_id": clean_circuit_id,
                    "simulation_mode": "ideal_statevector",
                    "schema_version": config.schema_version,
                    "metric_source": "exact_born_probabilities",
                }
            )
            clean_result = simulate_ideal_statevector(bound_circuit)
            _add_unique_record(
                circuit_records,
                _make_circuit_record(
                    clean_circuit_id,
                    bound_circuit,
                    spec.family,
                    {
                        "role": "clean",
                        "family": spec.family,
                        "parameter_bindings": parameter_bindings,
                        "parameter_sin_cos": parameter_sin_cos,
                        "generator_kwargs": generator_kwargs,
                    },
                ),
                clean_circuit_id,
            )
            _add_unique_record(
                simulation_records,
                _make_simulation_record(
                    clean_run_id,
                    clean_circuit_id,
                    "ideal_statevector",
                    None,
                    {
                        "role": "clean",
                        "probabilities_ref": f"artifacts/probabilities/{clean_run_id}.json",
                        "statevector_ref": f"artifacts/statevectors/{clean_run_id}.npy" if config.store_statevectors else None,
                    },
                ),
                clean_run_id,
            )

            clean_shot_result = None
            clean_shot_run_id = None
            if config.ideal_shots is not None:
                clean_shot_seed = derive_child_seed(config.base_seed, "clean_shots", clean_payload)
                clean_shot_result = simulate_ideal_shots(bound_circuit, shots=config.ideal_shots, seed=clean_shot_seed)
                clean_shot_run_id = make_run_id(
                    {
                        "circuit_id": clean_circuit_id,
                        "simulation_mode": "ideal_shot",
                        "shots": config.ideal_shots,
                        "seed": clean_shot_seed,
                        "schema_version": config.schema_version,
                    }
                )
                _add_unique_record(
                    simulation_records,
                    _make_simulation_record(
                        clean_shot_run_id,
                        clean_circuit_id,
                        "ideal_shot",
                        config.ideal_shots,
                        {
                            "source_run_id": clean_run_id,
                            "counts_ref": f"artifacts/counts/{clean_shot_run_id}.json",
                            "seed": clean_shot_seed,
                        },
                    ),
                    clean_shot_run_id,
                )

            for distortion_index, distortion_spec in enumerate(config.distortion_specs):
                distortion = apply_distortion(distortion_spec.name, bound_circuit, **_json_copy(distortion_spec.kwargs))
                distortion_id = make_deterministic_id(
                    "distortion",
                    {
                        "clean_circuit_id": clean_circuit_id,
                        "name": distortion_spec.name,
                        "kwargs": distortion_spec.kwargs,
                        "metadata": distortion.metadata,
                        "schema_version": config.schema_version,
                    },
                )
                distorted_circuit_id = make_circuit_id(
                    {
                        "clean_circuit_id": clean_circuit_id,
                        "distortion_id": distortion_id,
                        "schema_version": config.schema_version,
                    }
                )
                distorted_run_id = make_run_id(
                    {
                        "distorted_circuit_id": distorted_circuit_id,
                        "simulation_mode": "ideal_statevector",
                        "distortion_id": distortion_id,
                        "schema_version": config.schema_version,
                        "metric_source": "exact_born_probabilities",
                    }
                )
                distorted_result = simulate_ideal_statevector(distortion.distorted_circuit)
                context_metadata = dict(distortion.metadata)
                context_metadata["distortion_family"] = distortion.distortion_family
                born_metrics = compare_born_distributions(
                    clean_result,
                    distorted_result,
                    context_metadata=context_metadata,
                )
                metric_id = make_deterministic_id(
                    "metric",
                    {
                        "clean_run_id": clean_run_id,
                        "distorted_run_id": distorted_run_id,
                        "metric_family": "born",
                        "metric_names": sorted(born_metrics.metrics),
                        "metric_schema_version": PHASE7_METRIC_SCHEMA_VERSION,
                        "schema_version": config.schema_version,
                    },
                )
                sample_id = make_sample_id(
                    {
                        "clean_circuit_id": clean_circuit_id,
                        "distortion_id": distortion_id,
                        "metric_id": metric_id,
                        "schema_version": config.schema_version,
                    }
                )
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id detected: {sample_id}")
                sample_ids.add(sample_id)

                total_variation = abs(float(born_metrics.metrics["total_variation"].value))
                born_zero_shift = total_variation <= config.born_zero_atol
                sample_metadata = {
                    "distortion_name": distortion_spec.name,
                    "distortion_kwargs": distortion_spec.kwargs,
                    "parameter_sin_cos": parameter_sin_cos,
                    "marker_only": bool(distortion.metadata.get("marker_only")),
                    "born_zero_shift": born_zero_shift,
                    "born_observable_shift_absent": born_zero_shift,
                    "born_zero_atol": config.born_zero_atol,
                    "total_variation_exact": total_variation,
                }

                _add_unique_record(
                    circuit_records,
                    _make_circuit_record(
                        distorted_circuit_id,
                        distortion.distorted_circuit,
                        spec.family,
                        {
                            "role": "distorted",
                            "source_clean_circuit_id": clean_circuit_id,
                            "distortion_id": distortion_id,
                            "parameter_bindings": parameter_bindings,
                        },
                    ),
                    distorted_circuit_id,
                )
                _add_unique_record(
                    simulation_records,
                    _make_simulation_record(
                        distorted_run_id,
                        distorted_circuit_id,
                        "ideal_statevector",
                        None,
                        {
                            "role": "distorted",
                            "distortion_id": distortion_id,
                            "probabilities_ref": f"artifacts/probabilities/{distorted_run_id}.json",
                            "statevector_ref": f"artifacts/statevectors/{distorted_run_id}.npy" if config.store_statevectors else None,
                        },
                    ),
                    distorted_run_id,
                )

                distorted_shot_result = None
                distorted_shot_run_id = None
                if config.ideal_shots is not None:
                    distorted_shot_seed = derive_child_seed(
                        config.base_seed,
                        "distorted_shots",
                        {"clean_payload": clean_payload, "distortion_index": distortion_index, "distortion_id": distortion_id},
                    )
                    distorted_shot_result = simulate_ideal_shots(
                        distortion.distorted_circuit,
                        shots=config.ideal_shots,
                        seed=distorted_shot_seed,
                    )
                    distorted_shot_run_id = make_run_id(
                        {
                            "circuit_id": distorted_circuit_id,
                            "simulation_mode": "ideal_shot",
                            "shots": config.ideal_shots,
                            "seed": distorted_shot_seed,
                            "distortion_id": distortion_id,
                            "schema_version": config.schema_version,
                        }
                    )
                    _add_unique_record(
                        simulation_records,
                        _make_simulation_record(
                            distorted_shot_run_id,
                            distorted_circuit_id,
                            "ideal_shot",
                            config.ideal_shots,
                            {
                                "source_run_id": distorted_run_id,
                                "counts_ref": f"artifacts/counts/{distorted_shot_run_id}.json",
                                "seed": distorted_shot_seed,
                            },
                        ),
                        distorted_shot_run_id,
                    )

                _add_unique_record(
                    distortion_records,
                    DistortionRecord(
                        distortion_id=distortion_id,
                        circuit_id=clean_circuit_id,
                        distortion_type=distortion.distortion_type,
                        strength=distortion.strength,
                        affected_qubits=distortion.affected_qubits,
                        affected_gates=distortion.affected_gates,
                        metadata={**_json_copy(distortion.metadata), "distorted_circuit_id": distorted_circuit_id},
                    ),
                    distortion_id,
                )
                metric_record = MetricRecord(
                    metric_id=metric_id,
                    run_id=distorted_run_id,
                    circuit_id=distorted_circuit_id,
                    distortion_id=distortion_id,
                    born_metrics=_metric_values(born_metrics),
                    hilbert_metrics={},
                    parameter_metrics={},
                    topology_metrics={},
                    hilbert_available_mask=False,
                    metadata={
                        "clean_run_id": clean_run_id,
                        "distorted_run_id": distorted_run_id,
                        "sample_id": sample_id,
                        "metric_family": "born",
                        "metric_schema_version": PHASE7_METRIC_SCHEMA_VERSION,
                        "support_size": len(born_metrics.support),
                        "nonfinite_encoding": "positive infinity encoded as null plus metric__nonfinite",
                        "applicability_warning": born_metrics.metadata.get("applicability_warning"),
                        "computed_metric_families": ["born"],
                        "deferred_metric_families": ["hilbert", "parameter", "topology"],
                    },
                )
                _add_unique_record(metric_records, metric_record, metric_id)

                sample_record = DatasetSampleRecord(
                    sample_id=sample_id,
                    dataset_name=config.dataset_name,
                    schema_version=config.schema_version,
                    clean_circuit_id=clean_circuit_id,
                    distorted_circuit_id=distorted_circuit_id,
                    clean_run_id=clean_run_id,
                    distorted_run_id=distorted_run_id,
                    distortion_id=distortion_id,
                    metric_id=metric_id,
                    family=spec.family,
                    n_qubits=spec.n_qubits,
                    repetition_index=repetition_index,
                    parameter_bindings=_json_copy(parameter_bindings),
                    base_seed=config.base_seed,
                    metadata=sample_metadata,
                )
                sample_record.validate()
                sample_records.append(sample_record)
                samples.append(
                    GeneratedDatasetSample(
                        sample_id=sample_id,
                        clean_circuit_id=clean_circuit_id,
                        distorted_circuit_id=distorted_circuit_id,
                        clean_run_id=clean_run_id,
                        distorted_run_id=distorted_run_id,
                        distortion_id=distortion_id,
                        metric_id=metric_id,
                        family=spec.family,
                        n_qubits=spec.n_qubits,
                        repetition_index=repetition_index,
                        parameter_bindings=parameter_bindings,
                        generation_seed=generation_seed,
                        parameter_seed=parameter_seed,
                        clean_circuit=bound_circuit,
                        distorted_circuit=distortion.distorted_circuit,
                        clean_result=clean_result,
                        distorted_result=distorted_result,
                        distortion_result=distortion,
                        born_metrics=born_metrics,
                        clean_shot_result=clean_shot_result,
                        distorted_shot_result=distorted_shot_result,
                        clean_shot_run_id=clean_shot_run_id,
                        distorted_shot_run_id=distorted_shot_run_id,
                        metadata=sample_metadata,
                    )
                )

    family_counts = Counter(sample.family for sample in samples)
    distortion_counts = Counter(sample.metadata["distortion_name"] for sample in samples)
    summary = {
        "sample_count": len(samples),
        "unique_clean_circuit_count": len({sample.clean_circuit_id for sample in samples}),
        "unique_distorted_circuit_count": len({sample.distorted_circuit_id for sample in samples}),
        "simulation_record_count": len(simulation_records),
        "distortion_record_count": len(distortion_records),
        "metric_record_count": len(metric_records),
        "family_counts": dict(sorted(family_counts.items())),
        "distortion_counts": dict(sorted(distortion_counts.items())),
        "marker_only_sample_count": sum(sample.metadata["marker_only"] for sample in samples),
        "born_visible_sample_count": sum(not sample.metadata["born_zero_shift"] for sample in samples),
        "born_zero_shift_sample_count": sum(sample.metadata["born_zero_shift"] for sample in samples),
        "born_zero_atol": config.born_zero_atol,
        "base_seed": config.base_seed,
        "schema_version": config.schema_version,
        "config_id": operational_config_id,
        "scientific_generation_id": generation_id,
        "scientific_scope": "synthetic simulator-derived raw data; no hardware, training, correction actions, topology, or quantum-advantage claim",
    }
    return DatasetGenerationResult(
        dataset_name=config.dataset_name,
        schema_version=config.schema_version,
        config_id=operational_config_id,
        scientific_generation_id=generation_id,
        config=config,
        samples=samples,
        circuit_records=_sorted_records(circuit_records),
        simulation_records=_sorted_records(simulation_records),
        distortion_records=_sorted_records(distortion_records),
        metric_records=_sorted_records(metric_records),
        sample_records=sorted(sample_records, key=lambda record: record.sample_id),
        summary=summary,
    )
