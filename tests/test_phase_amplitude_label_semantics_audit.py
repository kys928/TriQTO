from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/audit_phase_amplitude_label_semantics.py"
)
SPEC = importlib.util.spec_from_file_location("label_semantics_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_rz_on_plus_is_phase_only_in_overlap_decomposition() -> None:
    delta = 0.2
    clean = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    distorted = np.asarray(
        [np.exp(-0.5j * delta), np.exp(0.5j * delta)], dtype=np.complex128
    ) / np.sqrt(2.0)

    metrics = AUDIT.overlap_decomposition(clean, distorted, epsilon=1e-12)

    assert metrics["population_component"] < 1e-12
    assert metrics["phase_component"] > 0.0
    assert metrics["dominance_log_ratio"] > 0.0
    assert AUDIT.strong_phenotype(
        metrics["population_component"],
        metrics["phase_component"],
        metrics["total_overlap_loss"],
        negligible_floor=1e-10,
        dominance_ratio=2.0,
    ) == "phase_dominant"


def test_ry_on_zero_is_population_only_in_overlap_decomposition() -> None:
    delta = 0.2
    clean = np.asarray([1.0, 0.0], dtype=np.complex128)
    distorted = np.asarray(
        [np.cos(delta / 2.0), np.sin(delta / 2.0)], dtype=np.complex128
    )

    metrics = AUDIT.overlap_decomposition(clean, distorted, epsilon=1e-12)

    assert metrics["population_component"] > 0.0
    assert metrics["phase_component"] < 1e-12
    assert metrics["dominance_log_ratio"] < 0.0
    assert AUDIT.strong_phenotype(
        metrics["population_component"],
        metrics["phase_component"],
        metrics["total_overlap_loss"],
        negligible_floor=1e-10,
        dominance_ratio=2.0,
    ) == "population_dominant"


def test_global_phase_is_negligible() -> None:
    clean = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    distorted = np.exp(0.73j) * clean

    metrics = AUDIT.overlap_decomposition(clean, distorted, epsilon=1e-12)

    assert metrics["total_overlap_loss"] < 1e-12
    assert AUDIT.strong_phenotype(
        metrics["population_component"],
        metrics["phase_component"],
        metrics["total_overlap_loss"],
        negligible_floor=1e-10,
        dominance_ratio=2.0,
    ) == "negligible"


def test_overlap_decomposition_closes_for_random_states() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(20):
        clean = rng.normal(size=8) + 1j * rng.normal(size=8)
        distorted = rng.normal(size=8) + 1j * rng.normal(size=8)
        clean = clean / np.linalg.norm(clean)
        distorted = distorted / np.linalg.norm(distorted)

        metrics = AUDIT.overlap_decomposition(clean, distorted, epsilon=1e-12)

        assert metrics["population_component"] >= 0.0
        assert metrics["phase_component"] >= 0.0
        assert metrics["decomposition_closure_error"] < 1e-12


def test_basis_measurement_probabilities_match_known_states() -> None:
    plus = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    plus_i = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)

    assert np.allclose(AUDIT.measurement_probabilities(plus, "Z"), [0.5, 0.5])
    assert np.allclose(AUDIT.measurement_probabilities(plus, "X"), [1.0, 0.0])
    assert np.allclose(AUDIT.measurement_probabilities(plus_i, "Y"), [1.0, 0.0])


def test_expected_sign_semantics_are_explicit() -> None:
    assert AUDIT.expected_sign_match("phase_like", 0.1)
    assert not AUDIT.expected_sign_match("phase_like", -0.1)
    assert AUDIT.expected_sign_match("amplitude_like", -0.1)
    assert not AUDIT.expected_sign_match("amplitude_like", 0.1)
