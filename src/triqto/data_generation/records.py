"""In-memory records for Phase 7 raw dataset generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

from triqto.distortions import DistortedCircuit
from triqto.metrics import BornMetricBundle
from triqto.simulation import MeasurementProbabilityResult, MeasurementShotResult
from triqto.simulation.results import IdealShotResult, IdealStatevectorResult
from triqto.storage import (
    CircuitRecord,
    DistortionRecord,
    MeasurementSettingRecord,
    MetricRecord,
    SimulationRecord,
)
from triqto.storage.schema import DatasetSampleRecord


@dataclass(slots=True)
class GeneratedDatasetSample:
    """One in-memory clean/distorted exact Born comparison sample."""

    sample_id: str
    clean_circuit_id: str
    distorted_circuit_id: str
    clean_run_id: str
    distorted_run_id: str
    distortion_id: str
    metric_id: str
    family: str
    n_qubits: int
    repetition_index: int
    parameter_bindings: dict[str, float]
    generation_seed: int
    parameter_seed: int
    clean_circuit: QuantumCircuit
    distorted_circuit: QuantumCircuit
    clean_result: IdealStatevectorResult
    distorted_result: IdealStatevectorResult
    distortion_result: DistortedCircuit
    born_metrics: BornMetricBundle
    clean_shot_result: IdealShotResult | MeasurementShotResult | None = None
    distorted_shot_result: IdealShotResult | MeasurementShotResult | None = None
    clean_shot_run_id: str | None = None
    distorted_shot_run_id: str | None = None
    measurement_settings: dict[str, Any] = field(default_factory=dict)
    clean_measurement_results: dict[str, MeasurementProbabilityResult] = field(default_factory=dict)
    distorted_measurement_results: dict[str, MeasurementProbabilityResult] = field(default_factory=dict)
    clean_measurement_run_ids: dict[str, str] = field(default_factory=dict)
    distorted_measurement_run_ids: dict[str, str] = field(default_factory=dict)
    clean_measurement_shot_results: dict[str, MeasurementShotResult] = field(default_factory=dict)
    distorted_measurement_shot_results: dict[str, MeasurementShotResult] = field(default_factory=dict)
    clean_measurement_shot_run_ids: dict[str, str] = field(default_factory=dict)
    distorted_measurement_shot_run_ids: dict[str, str] = field(default_factory=dict)
    measurement_born_metrics: dict[str, BornMetricBundle] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetGenerationResult:
    """In-memory result returned by :func:`generate_dataset`."""

    dataset_name: str
    schema_version: str
    config_id: str
    scientific_generation_id: str
    config: Any
    samples: list[GeneratedDatasetSample]
    circuit_records: list[CircuitRecord]
    simulation_records: list[SimulationRecord]
    measurement_setting_records: list[MeasurementSettingRecord]
    distortion_records: list[DistortionRecord]
    metric_records: list[MetricRecord]
    sample_records: list[DatasetSampleRecord]
    summary: dict[str, Any]


@dataclass(slots=True)
class DatasetWriteResult:
    """Local filesystem result returned by :func:`write_dataset`."""

    output_root: Path
    written_paths: list[Path]
    manifest_paths: dict[str, Path]
    artifact_paths: dict[str, list[Path]]
    summary_path: Path
    config_path: Path
