"""Seeded Aer noisy-shot execution with explicit noise provenance."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from triqto.core.ids import make_deterministic_id

from .measurement import MeasurementSetting, measurement_setting_for
from .result_normalization import (
    bind_parameter_values,
    copy_without_final_measurements,
    counts_to_probabilities,
    extract_quantum_circuit,
    normalize_counts,
)
from .results import IdealShotResult

SUPPORTED_NOISE_CHANNELS = {"depolarizing", "amplitude_damping", "phase_damping", "thermal_relaxation", "readout_error"}


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    channels: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = []
        for raw in self.channels:
            if not isinstance(raw, Mapping):
                raise TypeError("noise channels must be mappings")
            channel = dict(raw)
            kind = str(channel.get("type", ""))
            if kind not in SUPPORTED_NOISE_CHANNELS:
                raise ValueError(f"unsupported noise channel: {kind}")
            probability = channel.get("probability", channel.get("error", None))
            if probability is not None:
                p = float(probability)
                if not 0.0 <= p <= 1.0:
                    raise ValueError("noise probabilities must be in [0,1]")
            normalized.append(dict(sorted(channel.items())))
        object.__setattr__(self, "channels", tuple(normalized))

    @property
    def noise_model_id(self) -> str:
        return make_deterministic_id("noisemodel", {"channels": list(self.channels), "schema": "triqto.noise.v1"})


def _build_noise_model(spec: NoiseSpec):
    from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error, depolarizing_error, phase_damping_error, thermal_relaxation_error

    model = NoiseModel()
    for channel in spec.channels:
        kind = channel["type"]
        gates = list(channel.get("gates", ["x", "sx", "h", "rx", "ry", "rz"]))
        if kind == "depolarizing":
            p = float(channel["probability"])
            qubits = int(channel.get("qubits", 1))
            model.add_all_qubit_quantum_error(depolarizing_error(p, qubits), gates)
        elif kind == "amplitude_damping":
            model.add_all_qubit_quantum_error(amplitude_damping_error(float(channel["probability"])), gates)
        elif kind == "phase_damping":
            model.add_all_qubit_quantum_error(phase_damping_error(float(channel["probability"])), gates)
        elif kind == "thermal_relaxation":
            model.add_all_qubit_quantum_error(
                thermal_relaxation_error(float(channel["t1"]), float(channel["t2"]), float(channel["time"])), gates
            )
        elif kind == "readout_error":
            p = float(channel["probability"])
            model.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return model


def _apply_basis_rotation(circuit: Any, setting: MeasurementSetting) -> None:
    for qubit, basis in enumerate(setting.bases):
        if basis == "X":
            circuit.h(qubit)
        elif basis == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)


def simulate_noisy_aer_shots(
    circuit_or_generated: Any,
    *,
    noise: NoiseSpec,
    shots: int = 1024,
    seed: int = 0,
    parameter_values: Mapping[str, float] | Mapping[Any, float] | None = None,
    measurement_basis: str | tuple[str, ...] | MeasurementSetting | None = None,
) -> IdealShotResult:
    """Run seeded noisy Aer shots for an explicit basis-conditioned measurement."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    from qiskit_aer import AerSimulator

    original = extract_quantum_circuit(circuit_or_generated)
    bound = bind_parameter_values(original, parameter_values)
    measured = copy_without_final_measurements(bound)
    setting = measurement_basis if isinstance(measurement_basis, MeasurementSetting) else measurement_setting_for(measured.num_qubits, measurement_basis)
    if setting.n_qubits != measured.num_qubits:
        raise ValueError("measurement setting qubit count must match circuit")
    _apply_basis_rotation(measured, setting)
    measured.measure_all()
    simulator = AerSimulator(noise_model=_build_noise_model(noise), seed_simulator=seed)
    result = simulator.run(measured, shots=shots, seed_simulator=seed).result()
    counts = normalize_counts(result.get_counts())
    return IdealShotResult(
        simulation_mode="noisy_aer_shot",
        n_qubits=measured.num_qubits,
        shots=shots,
        counts=counts,
        probabilities=counts_to_probabilities(counts),
        source_probabilities={},
        metadata={
            "seed": seed,
            "shots_requested": shots,
            "shots_realized": sum(counts.values()),
            "noise_model_id": noise.noise_model_id,
            "noise_spec": list(noise.channels),
            "measurement_setting": setting.to_metadata(),
            "probability_domain": "p(y|M)",
            "evidence_tier": "noisy_simulator",
            "physical_hardware": False,
        },
    )


__all__ = ["NoiseSpec", "SUPPORTED_NOISE_CHANNELS", "simulate_noisy_aer_shots"]
