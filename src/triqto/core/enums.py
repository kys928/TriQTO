"""Enum contracts for TriQTO data generation, storage, and training."""
from __future__ import annotations
from enum import Enum

class CircuitFamily(str, Enum):
    BELL="bell"; GHZ="ghz"; PHASE_INTERFERENCE="phase_interference"; QFT_LIKE="qft_like"; HARDWARE_EFFICIENT_ANSATZ="hardware_efficient_ansatz"; RANDOM_SHALLOW="random_shallow"; LATTICE_ENTANGLED="lattice_entangled"; QAOA_LIKE="qaoa_like"
class SimulationMode(str, Enum):
    IDEAL_STATEVECTOR="ideal_statevector"; IDEAL_SHOT="ideal_shot"; NOISY_SHOT="noisy_shot"; FAKE_BACKEND="fake_backend"; HARDWARE="hardware"
class DistortionType(str, Enum):
    PHASE_RZ_DRIFT="phase_rz_drift"; RX_OVERROTATION="rx_overrotation"; RY_OVERROTATION="ry_overrotation"; ENTANGLING_OVERROTATION="entangling_overrotation"; READOUT_NOISE="readout_noise"; DEPOLARIZING_NOISE="depolarizing_noise"; AMPLITUDE_DAMPING="amplitude_damping"; PHASE_DAMPING="phase_damping"; THERMAL_RELAXATION="thermal_relaxation"; MIXED_NOISE="mixed_noise"; TRANSPILATION_LAYOUT_DISTORTION="transpilation_layout_distortion"
class ActionType(str, Enum):
    RZ_PHASE_SHIFT="rz_phase_shift"; RX_AMPLITUDE_SHIFT="rx_amplitude_shift"; RY_AMPLITUDE_SHIFT="ry_amplitude_shift"; ENTANGLER_ADJUSTMENT="entangler_adjustment"; GATE_REMOVAL="gate_removal"; LAYOUT_CHANGE="layout_change"; TRANSPILER_CHANGE="transpiler_change"; MEASUREMENT_BASIS_PROBE="measurement_basis_probe"; DEPTH_REDUCTION="depth_reduction"
class BackendMode(str, Enum):
    LOCAL_SIMULATOR="local_simulator"; FAKE_BACKEND="fake_backend"; HARDWARE="hardware"
class TrainingTask(str, Enum):
    DIAGNOSIS="diagnosis"; ACTION_RANKING="action_ranking"; BORN_PREDICTION="born_prediction"; HILBERT_TO_BORN="hilbert_to_born"; TOPOLOGY_AUDIT="topology_audit"; JOINT_MULTITASK="joint_multitask"; HARDWARE_MASKED="hardware_masked"
class ManifoldType(str, Enum):
    PARAMETER="parameter"; HILBERT="hilbert"; BORN="born"
