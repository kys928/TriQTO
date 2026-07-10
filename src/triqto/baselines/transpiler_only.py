"""Backend-free transpiler-only semantic control for Phase 10."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile

from triqto.metrics import compare_born_distributions
from triqto.simulation import simulate_ideal_statevector

from .config import BaselineSuiteConfig
from .models import EvaluationSnapshot
from .optimizer_common import metric_array, probability_arrays, weighted_objective


def _transpiler_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(
        f"triqto-phase10-transpiler:{base_seed}:{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def run_transpiler_only(
    *,
    sample_id: str,
    source_circuit: QuantumCircuit,
    clean_probabilities: dict[str, float],
    config: BaselineSuiteConfig,
) -> tuple[EvaluationSnapshot, dict[str, Any]]:
    """Transpile without a backend or correction and verify ideal Born semantics."""
    if not isinstance(source_circuit, QuantumCircuit):
        raise TypeError("source_circuit must be QuantumCircuit")
    seed = _transpiler_seed(config.random_seed, sample_id)
    transpiled = transpile(
        source_circuit,
        optimization_level=config.transpiler_optimization_level,
        seed_transpiler=seed,
    )
    simulation = simulate_ideal_statevector(transpiled)
    bundle = compare_born_distributions(
        clean_probabilities,
        simulation.probabilities,
        include_kl=False,
        include_js_distance=False,
    )
    values = metric_array(bundle)
    objective = weighted_objective(values, config)
    bitstrings, probabilities = probability_arrays(simulation.probabilities)
    snapshot = EvaluationSnapshot(
        vector=np.asarray([], dtype=np.float64),
        metric_values=values,
        objective=objective,
        outcome_bitstrings=bitstrings,
        exact_probabilities=probabilities,
        metadata={
            "source_depth": source_circuit.depth(),
            "transpiled_depth": transpiled.depth(),
            "source_gate_count": len(source_circuit.data),
            "transpiled_gate_count": len(transpiled.data),
        },
    )
    return snapshot, {
        "algorithm": "qiskit.transpile without backend or coupling map",
        "optimization_level": config.transpiler_optimization_level,
        "seed_transpiler": seed,
        "clean_target_used_for_transpilation": False,
        "clean_target_used_only_for_evaluation": True,
        "hardware_aware": False,
        "physical_layout_claimed": False,
        "semantic_control_only": True,
        "learned_model_used": False,
    }


__all__ = ["run_transpiler_only"]
