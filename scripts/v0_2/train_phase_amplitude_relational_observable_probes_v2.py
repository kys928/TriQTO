#!/usr/bin/env python3
"""Relational-observable probe runner with dtype-aware audit tolerances.

The frozen model-facing Z Born probabilities were intentionally stored as
float32, while the privileged statevectors and reconstructed X/Y observables
are float64. This runner therefore applies the original pilot's 1e-6 absolute
tolerance only to the stored-Z comparison and retains a strict 1e-10 absolute
tolerance for all float64 observable comparisons.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
V1_PATH = HERE / "train_phase_amplitude_relational_observable_probes.py"
MODULE_NAME = "triqto_v0_2_relational_observable_probe_v1"
STORED_Z_FLOAT32_ATOL = 1.0e-6
FLOAT64_OBSERVABLE_ATOL = 1.0e-10


def load_patched_runner():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load relational runner from {V1_PATH}")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = runner
    spec.loader.exec_module(runner)

    runner.SCHEMA = "triqto.v0_2.phase_amplitude_relational_observable_probe.v2"
    runner.__file__ = str(Path(__file__).resolve())
    runner.AUDIT["stored_z_float32_atol"] = STORED_Z_FLOAT32_ATOL
    runner.AUDIT["float64_observable_atol"] = FLOAT64_OBSERVABLE_ATOL

    @lru_cache(maxsize=None)
    def derived_observables(
        artifact_path: str,
        n_qubits: int,
    ) -> dict[str, np.ndarray | float]:
        with np.load(artifact_path, allow_pickle=False) as archive:
            value = {name: archive[name] for name in archive.files}

        clean_state = runner.state_from_arrays(value, "clean")
        distorted_state = runner.state_from_arrays(value, "distorted")
        dimension = 1 << n_qubits

        result: dict[str, np.ndarray | float] = {}
        for basis in ("Z", "X", "Y"):
            clean_probability = runner.basis_probabilities(
                clean_state,
                n_qubits,
                basis,
            )
            distorted_probability = runner.basis_probabilities(
                distorted_state,
                n_qubits,
                basis,
            )
            clean_expectation = runner.pauli_expectations(
                clean_probability,
                n_qubits,
            )
            distorted_expectation = runner.pauli_expectations(
                distorted_probability,
                n_qubits,
            )

            result[f"clean_{basis}_probability"] = clean_probability
            result[f"distorted_{basis}_probability"] = distorted_probability
            result[f"delta_{basis}_probability"] = (
                distorted_probability - clean_probability
            )
            result[f"clean_{basis}_expectation"] = clean_expectation
            result[f"distorted_{basis}_expectation"] = distorted_expectation
            result[f"delta_{basis}_expectation"] = (
                distorted_expectation - clean_expectation
            )
            result[f"{basis}_tv"] = runner.total_variation(
                clean_probability,
                distorted_probability,
            )
            result[f"{basis}_hellinger"] = runner.hellinger_distance(
                clean_probability,
                distorted_probability,
            )
            result[f"{basis}_js_distance"] = runner.jensen_shannon_distance(
                clean_probability,
                distorted_probability,
            )

        stored_z = np.zeros(dimension, dtype=np.float64)
        bitstrings = np.asarray(
            value["a__x_born_input_outcome_bitstrings"]
        ).astype(str)
        stored_probabilities = np.asarray(
            value["a__x_born_input_probabilities"],
            dtype=np.float64,
        )
        for bits, probability in zip(
            bitstrings,
            stored_probabilities,
            strict=True,
        ):
            stored_z[int(bits, 2)] = probability

        comparisons: dict[str, tuple[float, float]] = {
            "max_distorted_z_probability_abs_error": (
                runner.max_abs_error(
                    result["distorted_Z_probability"],
                    stored_z,
                ),
                STORED_Z_FLOAT32_ATOL,
            ),
            "max_distorted_x_probability_abs_error": (
                runner.max_abs_error(
                    result["distorted_X_probability"],
                    value["b__distorted_x_probabilities"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
            "max_distorted_y_probability_abs_error": (
                runner.max_abs_error(
                    result["distorted_Y_probability"],
                    value["b__distorted_y_probabilities"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
            "max_clean_x_expectation_abs_error": (
                runner.max_abs_error(
                    result["clean_X_expectation"],
                    value["c__clean_x_expectations"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
            "max_clean_y_expectation_abs_error": (
                runner.max_abs_error(
                    result["clean_Y_expectation"],
                    value["c__clean_y_expectations"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
            "max_clean_z_expectation_abs_error": (
                runner.max_abs_error(
                    result["clean_Z_expectation"],
                    value["c__clean_z_expectations"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
            "max_distorted_x_expectation_abs_error": (
                runner.max_abs_error(
                    result["distorted_X_expectation"],
                    value["b__distorted_x_expectations"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
            "max_distorted_y_expectation_abs_error": (
                runner.max_abs_error(
                    result["distorted_Y_expectation"],
                    value["b__distorted_y_expectations"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
            "max_distorted_z_expectation_abs_error": (
                runner.max_abs_error(
                    result["distorted_Z_expectation"],
                    value["c__distorted_z_expectations"],
                ),
                FLOAT64_OBSERVABLE_ATOL,
            ),
        }

        for name, (error, tolerance) in comparisons.items():
            runner.AUDIT[name] = max(float(runner.AUDIT[name]), error)
            if error > tolerance:
                raise ValueError(
                    f"Observable derivation mismatch for {artifact_path}: "
                    f"{name}={error:.3e} > tolerance={tolerance:.3e}"
                )

        entity_id = str(np.asarray(value["entity_id"]).reshape(-1)[0])
        runner.AUDIT["verified_entities"].add(entity_id)
        return result

    runner.derived_observables = derived_observables
    return runner


def main() -> None:
    runner = load_patched_runner()
    runner.main()


if __name__ == "__main__":
    main()
