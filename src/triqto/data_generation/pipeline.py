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
from triqto.simulation import (
    MEASUREMENT_SCHEMA_VERSION,
    measurement_setting,
    sample_measurement_counts,
    simulate_ideal_statevector,
    simulate_measurement_probabilities,
)
from triqto.storage import (
    CircuitRecord,
    DistortionRecord,
    MeasurementSettingRecord,
    MetricRecord,
    SimulationRecord,
)
from triqto.storage.schema import DatasetSampleRecord

from .records import DatasetGenerationResult, GeneratedDatasetSample
from .identifiability import (
    assess_identifiability,
    observable_evidence_fingerprint,
    reject_conflicting_identifiable_labels,
)
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
    measurement_setting_id: str | None = None,
) -> SimulationRecord:
    if simulation_mode == "ideal_statevector":
        backend_name = "qiskit.quantum_info.Statevector"
        metadata = {"sampling_source": "exact_statevector", **metadata}
    elif simulation_mode == "ideal_shot":
        backend_name = "triqto.ideal_probability_sampler"
        metadata = {"sampling_source": "sampled_from_exact_born_probabilities", **metadata}
    elif simulation_mode == "ideal_measurement_probabilities":
        backend_name = "qiskit.quantum_info.Statevector"
        metadata = {
            "sampling_source": "basis_conditioned_exact_probabilities",
            **metadata,
        }
    else:
        backend_name = None
    return SimulationRecord(
        run_id=run_id,
        circuit_id=circuit_id,
        simulation_mode=simulation_mode,
        backend_name=backend_name,
        shots=shots,
        metadata=_json_copy(metadata),
        measurement_setting_id=measurement_setting_id,
    )


def _metric_values(bundle: BornMetricBundle) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, result in sorted(bundle.metrics.items()):
        raw_value = result.value
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise TypeError(f"Metric {name} must be numeric")
        metric_value = float(raw_value)
        if math.isfinite(metric_value):
            values[name] = metric_value
        elif math.isnan(metric_value):
            raise ValueError(f"Metric {name} is NaN and cannot be serialized")
        elif metric_value == math.inf:
            values[name] = None
            values[f"{name}__nonfinite"] = "positive_infinity"
        else:
            raise ValueError(f"Metric {name} is negative infinity and cannot be serialized")
    return values


def _decode_metric_values(payload: dict[str, Any]) -> dict[str, float]:
    decoded: dict[str, float] = {}
    markers = {key[:-11]: value for key, value in payload.items() if key.endswith("__nonfinite")}
    for marker_base in markers:
        if marker_base not in payload:
            raise ValueError(f"Orphan nonfinite marker for metric {marker_base}")
    for name, value in payload.items():
        if name.endswith("__nonfinite"):
            continue
        marker = markers.get(name)
        if value is None:
            if marker is None:
                raise ValueError(f"Metric {name} is null without a nonfinite marker")
            if marker != "positive_infinity":
                raise ValueError(f"Metric {name} has unknown nonfinite marker {marker!r}")
            decoded[name] = math.inf
            continue
        if marker is not None:
            raise ValueError(f"Metric {name} has finite value and nonfinite marker")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"Metric {name} must be numeric")
        metric_value = float(value)
        if not math.isfinite(metric_value):
            raise ValueError(f"Metric {name} contains unencoded nonfinite value")
        decoded[name] = metric_value
    return decoded


def _add_unique_record(records_by_id: dict[str, Any], record: Any, record_id: str) -> None:
    existing = records_by_id.get(record_id)
    if existing is None:
        records_by_id[record_id] = record
        return
    if existing.to_dict() != record.to_dict():
        raise ValueError(f"Conflicting deterministic ID detected: {record_id}")


def _sorted_records(records_by_id: dict[str, Any]) -> list[Any]:
    return [records_by_id[key] for key in sorted(records_by_id)]


def _measurement_channel_kwargs(distortion: Any) -> dict[str, Any]:
    metadata = distortion.metadata
    channel = metadata.get("measurement_channel")
    if channel is None:
        return {}
    if channel != "independent_symmetric_readout_bitflip":
        raise ValueError(f"Unsupported observable measurement channel {channel!r}")
    return {
        "readout_bitflip_probability": metadata.get("probability"),
        "readout_qubits": metadata.get("selected_qubits"),
    }


def _measurement_metric_values(
    metrics_by_setting: dict[str, BornMetricBundle],
    settings_by_id: dict[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for setting_id in sorted(metrics_by_setting):
        label = settings_by_id[setting_id].label
        for name, value in _metric_values(metrics_by_setting[setting_id]).items():
            values[f"{label}:{name}"] = value
    return values


def generate_dataset(config: DatasetGenerationConfig) -> DatasetGenerationResult:
    """Generate deterministic measurement-conditioned clean/distorted samples."""
    operational_config_id = config_id(config)
    generation_id = scientific_generation_id(config)
    circuit_records: dict[str, CircuitRecord] = {}
    simulation_records: dict[str, SimulationRecord] = {}
    measurement_records: dict[str, MeasurementSettingRecord] = {}
    distortion_records: dict[str, DistortionRecord] = {}
    metric_records: dict[str, MetricRecord] = {}
    sample_ids: set[str] = set()
    sample_records: list[DatasetSampleRecord] = []
    samples: list[GeneratedDatasetSample] = []
    conflict_rows: list[dict[str, str]] = []

    for spec_index, spec in enumerate(config.circuit_specs):
        for repetition_index in range(spec.repetitions):
            clean_payload = _circuit_scientific_payload(config, spec_index, repetition_index)
            generation_seed = derive_child_seed(config.base_seed, "circuit_generation", clean_payload)
            parameter_seed = derive_child_seed(config.base_seed, "parameter_binding", clean_payload)
            generator_kwargs = _generator_kwargs(spec.family, spec.generator_kwargs, generation_seed)
            generated = generate_circuit_family(spec.family, spec.n_qubits, **generator_kwargs)
            bound_circuit, parameter_bindings, parameter_sin_cos = _bind_parameters(
                generated, parameter_seed, config.parameter_low, config.parameter_high
            )
            settings = [measurement_setting(value, spec.n_qubits) for value in config.measurement_settings]
            settings_by_id = {setting.setting_id: setting for setting in settings}
            if len(settings_by_id) != len(settings):
                raise ValueError("measurement settings collapse to duplicate canonical settings")
            for setting in settings:
                _add_unique_record(
                    measurement_records,
                    MeasurementSettingRecord(
                        measurement_setting_id=setting.setting_id,
                        schema_version=MEASUREMENT_SCHEMA_VERSION,
                        n_qubits=setting.n_qubits,
                        bases=list(setting.bases),
                        metadata={"label": setting.label, "observable_context": True},
                    ),
                    setting.setting_id,
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

            clean_measurement_results: dict[str, Any] = {}
            clean_measurement_run_ids: dict[str, str] = {}
            clean_measurement_shot_results: dict[str, Any] = {}
            clean_measurement_shot_run_ids: dict[str, str] = {}
            for setting in settings:
                setting_id = setting.setting_id
                result = simulate_measurement_probabilities(bound_circuit, setting)
                run_id = make_run_id(
                    {
                        "circuit_id": clean_circuit_id,
                        "simulation_mode": result.simulation_mode,
                        "measurement_setting_id": setting_id,
                        "schema_version": config.schema_version,
                    }
                )
                clean_measurement_results[setting_id] = result
                clean_measurement_run_ids[setting_id] = run_id
                _add_unique_record(
                    simulation_records,
                    _make_simulation_record(
                        run_id,
                        clean_circuit_id,
                        result.simulation_mode,
                        None,
                        {
                            **result.metadata,
                            "role": "clean",
                            "probabilities_ref": f"artifacts/probabilities/{run_id}.json",
                        },
                        measurement_setting_id=setting_id,
                    ),
                    run_id,
                )
                if config.ideal_shots is not None:
                    seed = derive_child_seed(
                        config.base_seed,
                        "clean_measurement_shots",
                        {**clean_payload, "measurement_setting_id": setting_id},
                    )
                    shot = sample_measurement_counts(
                        result,
                        shots=config.ideal_shots,
                        seed=seed,
                    )
                    shot_run_id = make_run_id(
                        {
                            "circuit_id": clean_circuit_id,
                            "simulation_mode": "ideal_shot",
                            "measurement_setting_id": setting_id,
                            "source_run_id": run_id,
                            "shots": config.ideal_shots,
                            "seed": seed,
                            "schema_version": config.schema_version,
                        }
                    )
                    clean_measurement_shot_results[setting_id] = shot
                    clean_measurement_shot_run_ids[setting_id] = shot_run_id
                    _add_unique_record(
                        simulation_records,
                        _make_simulation_record(
                            shot_run_id,
                            clean_circuit_id,
                            "ideal_shot",
                            config.ideal_shots,
                            {
                                "source_run_id": run_id,
                                "counts_ref": f"artifacts/counts/{shot_run_id}.json",
                                "seed": seed,
                            },
                            measurement_setting_id=setting_id,
                        ),
                        shot_run_id,
                    )

            for distortion_index, distortion_spec in enumerate(config.distortion_specs):
                distortion = apply_distortion(
                    distortion_spec.name,
                    bound_circuit,
                    **_json_copy(distortion_spec.kwargs),
                )
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
                    }
                )
                distorted_result = simulate_ideal_statevector(distortion.distorted_circuit)
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

                channel_kwargs = _measurement_channel_kwargs(distortion)
                distorted_measurement_results: dict[str, Any] = {}
                distorted_measurement_run_ids: dict[str, str] = {}
                distorted_measurement_shot_results: dict[str, Any] = {}
                distorted_measurement_shot_run_ids: dict[str, str] = {}
                metrics_by_setting: dict[str, BornMetricBundle] = {}
                context_metadata = {
                    **dict(distortion.metadata),
                    "distortion_family": distortion.distortion_family,
                }
                for setting in settings:
                    setting_id = setting.setting_id
                    result = simulate_measurement_probabilities(
                        distortion.distorted_circuit,
                        setting,
                        **channel_kwargs,
                    )
                    run_id = make_run_id(
                        {
                            "circuit_id": distorted_circuit_id,
                            "simulation_mode": result.simulation_mode,
                            "measurement_setting_id": setting_id,
                            "distortion_id": distortion_id,
                            "measurement_channel": result.metadata.get("readout_channel"),
                            "schema_version": config.schema_version,
                        }
                    )
                    distorted_measurement_results[setting_id] = result
                    distorted_measurement_run_ids[setting_id] = run_id
                    _add_unique_record(
                        simulation_records,
                        _make_simulation_record(
                            run_id,
                            distorted_circuit_id,
                            result.simulation_mode,
                            None,
                            {
                                **result.metadata,
                                "role": "distorted",
                                "distortion_id": distortion_id,
                                "probabilities_ref": f"artifacts/probabilities/{run_id}.json",
                            },
                            measurement_setting_id=setting_id,
                        ),
                        run_id,
                    )
                    metrics_by_setting[setting_id] = compare_born_distributions(
                        clean_measurement_results[setting_id],
                        result,
                        context_metadata={
                            **context_metadata,
                            "measurement_setting_id": setting_id,
                            "measurement_bases": list(setting.bases),
                        },
                    )
                    if config.ideal_shots is not None:
                        seed = derive_child_seed(
                            config.base_seed,
                            "distorted_measurement_shots",
                            {
                                "clean_payload": clean_payload,
                                "distortion_index": distortion_index,
                                "distortion_id": distortion_id,
                                "measurement_setting_id": setting_id,
                            },
                        )
                        shot = sample_measurement_counts(
                            result,
                            shots=config.ideal_shots,
                            seed=seed,
                        )
                        shot_run_id = make_run_id(
                            {
                                "circuit_id": distorted_circuit_id,
                                "simulation_mode": "ideal_shot",
                                "measurement_setting_id": setting_id,
                                "source_run_id": run_id,
                                "shots": config.ideal_shots,
                                "seed": seed,
                                "distortion_id": distortion_id,
                                "schema_version": config.schema_version,
                            }
                        )
                        distorted_measurement_shot_results[setting_id] = shot
                        distorted_measurement_shot_run_ids[setting_id] = shot_run_id
                        _add_unique_record(
                            simulation_records,
                            _make_simulation_record(
                                shot_run_id,
                                distorted_circuit_id,
                                "ideal_shot",
                                config.ideal_shots,
                                {
                                    "source_run_id": run_id,
                                    "counts_ref": f"artifacts/counts/{shot_run_id}.json",
                                    "seed": seed,
                                    "distortion_id": distortion_id,
                                },
                                measurement_setting_id=setting_id,
                            ),
                            shot_run_id,
                        )

                assessment = assess_identifiability(
                    distortion_type=distortion.distortion_type,
                    marker_only=bool(distortion.metadata.get("marker_only")),
                    measurement_settings=settings_by_id,
                    clean_probabilities={
                        key: value.probabilities
                        for key, value in clean_measurement_results.items()
                    },
                    distorted_probabilities={
                        key: value.probabilities
                        for key, value in distorted_measurement_results.items()
                    },
                    atol=config.born_zero_atol,
                )
                if assessment.status == "unidentifiable" and config.unidentifiable_policy == "error":
                    raise ValueError(
                        "Unidentifiable diagnosis target rejected by strict policy: "
                        f"{distortion.distortion_type} ({assessment.reason})"
                    )
                supervision = assessment.status != "unidentifiable" or config.unidentifiable_policy == "allow"
                fingerprint = observable_evidence_fingerprint(
                    bound_circuit,
                    measurement_settings=settings_by_id,
                    probabilities={
                        key: value.probabilities
                        for key, value in distorted_measurement_results.items()
                    },
                )
                metric_id = make_deterministic_id(
                    "metric",
                    {
                        "clean_measurement_run_ids": clean_measurement_run_ids,
                        "distorted_measurement_run_ids": distorted_measurement_run_ids,
                        "metric_family": "measurement_conditioned_born",
                        "metric_schema_version": PHASE7_METRIC_SCHEMA_VERSION,
                        "schema_version": config.schema_version,
                    },
                )
                sample_id = make_sample_id(
                    {
                        "clean_circuit_id": clean_circuit_id,
                        "distortion_id": distortion_id,
                        "metric_id": metric_id,
                        "measurement_setting_ids": [setting.setting_id for setting in settings],
                        "schema_version": config.schema_version,
                    }
                )
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample_id detected: {sample_id}")
                sample_ids.add(sample_id)
                total_variations = {
                    setting_id: abs(float(bundle.metrics["total_variation"].value))
                    for setting_id, bundle in metrics_by_setting.items()
                }
                born_zero_shift = max(total_variations.values(), default=0.0) <= config.born_zero_atol
                sample_metadata = {
                    "distortion_name": distortion_spec.name,
                    "distortion_kwargs": distortion_spec.kwargs,
                    "parameter_sin_cos": parameter_sin_cos,
                    "marker_only": bool(distortion.metadata.get("marker_only")),
                    "born_zero_shift": born_zero_shift,
                    "born_observable_shift_absent": born_zero_shift,
                    "born_zero_atol": config.born_zero_atol,
                    "total_variation_by_measurement_setting": total_variations,
                    "identifiability_status": assessment.status,
                    "identifiability_reason": assessment.reason,
                    "visible_measurement_setting_ids": list(assessment.visible_measurement_setting_ids),
                    "blind_measurement_setting_ids": list(assessment.blind_measurement_setting_ids),
                    "diagnosis_supervision_mask": supervision,
                    "unidentifiable_supervision_override": (
                        True if assessment.status == "unidentifiable" and supervision else False
                    ),
                }
                _add_unique_record(
                    distortion_records,
                    DistortionRecord(
                        distortion_id=distortion_id,
                        circuit_id=clean_circuit_id,
                        distortion_type=distortion.distortion_type,
                        strength=distortion.strength,
                        affected_qubits=distortion.affected_qubits,
                        affected_gates=distortion.affected_gates,
                        metadata={
                            **_json_copy(distortion.metadata),
                            "distorted_circuit_id": distorted_circuit_id,
                        },
                    ),
                    distortion_id,
                )
                first_setting_id = settings[0].setting_id
                first_metrics = metrics_by_setting[first_setting_id]
                metric_record = MetricRecord(
                    metric_id=metric_id,
                    run_id=distorted_run_id,
                    circuit_id=distorted_circuit_id,
                    distortion_id=distortion_id,
                    born_metrics=_measurement_metric_values(metrics_by_setting, settings_by_id),
                    hilbert_metrics={},
                    parameter_metrics={},
                    topology_metrics={},
                    hilbert_available_mask=False,
                    metadata={
                        "clean_run_id": clean_run_id,
                        "distorted_run_id": distorted_run_id,
                        "clean_measurement_run_ids": clean_measurement_run_ids,
                        "distorted_measurement_run_ids": distorted_measurement_run_ids,
                        "sample_id": sample_id,
                        "metric_family": "measurement_conditioned_born",
                        "metric_schema_version": PHASE7_METRIC_SCHEMA_VERSION,
                        "support_size": sum(len(bundle.support) for bundle in metrics_by_setting.values()),
                        "nonfinite_encoding": "positive infinity encoded as null plus metric__nonfinite",
                        "applicability_warning": first_metrics.metadata.get("applicability_warning"),
                        "computed_metric_families": ["born"],
                        "deferred_metric_families": ["hilbert", "parameter", "topology"],
                        "identifiability_status": assessment.status,
                        "identifiability_reason": assessment.reason,
                    },
                )
                _add_unique_record(metric_records, metric_record, metric_id)
                setting_ids = [setting.setting_id for setting in settings]
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
                    measurement_setting_ids=setting_ids,
                    clean_measurement_run_ids=[clean_measurement_run_ids[key] for key in setting_ids],
                    distorted_measurement_run_ids=[distorted_measurement_run_ids[key] for key in setting_ids],
                    identifiability_status=assessment.status,
                    identifiability_reason=assessment.reason,
                    diagnosis_supervision_mask=supervision,
                    observable_evidence_fingerprint=fingerprint,
                )
                sample_record.validate()
                sample_records.append(sample_record)
                conflict_rows.append(
                    {
                        "observable_evidence_fingerprint": fingerprint,
                        "distortion_type": distortion.distortion_type,
                        "identifiability_status": assessment.status,
                    }
                )
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
                        born_metrics=first_metrics,
                        clean_shot_result=clean_measurement_shot_results.get(first_setting_id),
                        distorted_shot_result=distorted_measurement_shot_results.get(first_setting_id),
                        clean_shot_run_id=clean_measurement_shot_run_ids.get(first_setting_id),
                        distorted_shot_run_id=distorted_measurement_shot_run_ids.get(first_setting_id),
                        measurement_settings=dict(settings_by_id),
                        clean_measurement_results=dict(clean_measurement_results),
                        distorted_measurement_results=dict(distorted_measurement_results),
                        clean_measurement_run_ids=dict(clean_measurement_run_ids),
                        distorted_measurement_run_ids=dict(distorted_measurement_run_ids),
                        clean_measurement_shot_results=dict(clean_measurement_shot_results),
                        distorted_measurement_shot_results=dict(distorted_measurement_shot_results),
                        clean_measurement_shot_run_ids=dict(clean_measurement_shot_run_ids),
                        distorted_measurement_shot_run_ids=dict(distorted_measurement_shot_run_ids),
                        measurement_born_metrics=dict(metrics_by_setting),
                        metadata=sample_metadata,
                    )
                )

    reject_conflicting_identifiable_labels(conflict_rows)
    family_counts = Counter(sample.family for sample in samples)
    distortion_counts = Counter(sample.metadata["distortion_name"] for sample in samples)
    identifiability_counts = Counter(
        sample.metadata["identifiability_status"] for sample in samples
    )
    reason_counts = Counter(
        sample.metadata["identifiability_reason"]
        for sample in samples
        if sample.metadata["identifiability_reason"] is not None
    )
    summary = {
        "sample_count": len(samples),
        "unique_clean_circuit_count": len({sample.clean_circuit_id for sample in samples}),
        "unique_distorted_circuit_count": len({sample.distorted_circuit_id for sample in samples}),
        "simulation_record_count": len(simulation_records),
        "measurement_setting_record_count": len(measurement_records),
        "distortion_record_count": len(distortion_records),
        "metric_record_count": len(metric_records),
        "family_counts": dict(sorted(family_counts.items())),
        "distortion_counts": dict(sorted(distortion_counts.items())),
        "identifiability_counts": dict(sorted(identifiability_counts.items())),
        "identifiability_reason_counts": dict(sorted(reason_counts.items())),
        "identifiable_diagnosis_coverage": (
            sum(
                sample.metadata["identifiability_status"] != "unidentifiable"
                for sample in samples
            )
            / len(samples)
            if samples
            else 0.0
        ),
        "diagnosis_supervision_coverage": (
            sum(sample.metadata["diagnosis_supervision_mask"] for sample in samples)
            / len(samples)
            if samples
            else 0.0
        ),
        "unidentifiable_supervision_override_count": sum(
            sample.metadata["unidentifiable_supervision_override"] for sample in samples
        ),
        "marker_only_sample_count": sum(sample.metadata["marker_only"] for sample in samples),
        "born_visible_sample_count": sum(not sample.metadata["born_zero_shift"] for sample in samples),
        "born_zero_shift_sample_count": sum(sample.metadata["born_zero_shift"] for sample in samples),
        "measurement_settings": list(config.measurement_settings),
        "unidentifiable_policy": config.unidentifiable_policy,
        "born_zero_atol": config.born_zero_atol,
        "base_seed": config.base_seed,
        "schema_version": config.schema_version,
        "config_id": operational_config_id,
        "scientific_generation_id": generation_id,
        "scientific_scope": "synthetic simulator-derived basis-conditioned data; no hardware, trained correction, calibrated uncertainty, topology causality, or quantum-advantage claim",
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
        measurement_setting_records=_sorted_records(measurement_records),
        distortion_records=_sorted_records(distortion_records),
        metric_records=_sorted_records(metric_records),
        sample_records=sorted(sample_records, key=lambda record: record.sample_id),
        summary=summary,
    )
