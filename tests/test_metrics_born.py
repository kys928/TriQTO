"""Tests for Phase 6 Born-probability metrics."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict

import pytest
from qiskit import QuantumCircuit

from triqto.circuits.bell import make_bell_circuit
from triqto.distortions import apply_phase_rz_drift, apply_rx_overrotation
from triqto.metrics import (
    BornMetricBundle,
    align_probability_distributions,
    compare_born_distributions,
    hellinger_distance,
    jensen_shannon_distance,
    jensen_shannon_divergence,
    kl_divergence,
    normalize_probability_distribution,
    total_variation_distance,
)
from triqto.simulation import simulate_ideal_shots, simulate_ideal_statevector


def test_normalize_probability_distribution_normalizes_valid_non_normalized_input() -> None:
    assert normalize_probability_distribution({"1": 3.0, "0": 1.0}) == {"0": 0.25, "1": 0.75}


def test_normalize_probability_distribution_preserves_tiny_positive_probability() -> None:
    normalized = normalize_probability_distribution({"0": 1.0, "1": 1e-13})
    assert set(normalized) == {"0", "1"}
    assert normalized["1"] > 0.0
    assert sum(normalized.values()) == pytest.approx(1.0)


def test_normalize_probability_distribution_clips_tiny_negative_noise() -> None:
    assert normalize_probability_distribution({"0": 1.0, "1": -1e-13}) == {"0": 1.0}


def test_normalize_probability_distribution_rejects_real_negative_probabilities() -> None:
    with pytest.raises(ValueError, match="negative"):
        normalize_probability_distribution({"0": 1.0, "1": -0.01})


def test_normalize_probability_distribution_rejects_empty_distributions() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_probability_distribution({})


def test_align_probability_distributions_fills_missing_keys_over_union_support() -> None:
    p, q = align_probability_distributions({"0": 1.0}, {"1": 1.0})
    assert p == {"0": 1.0, "1": 0.0}
    assert q == {"0": 0.0, "1": 1.0}


def test_total_variation_distance_known_values() -> None:
    assert total_variation_distance({"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) == pytest.approx(0.0)
    assert total_variation_distance({"0": 1.0}, {"1": 1.0}) == pytest.approx(1.0)


def test_hellinger_distance_known_values() -> None:
    assert hellinger_distance({"0": 0.5, "1": 0.5}, {"0": 0.5, "1": 0.5}) == pytest.approx(0.0)
    assert hellinger_distance({"0": 1.0}, {"1": 1.0}) == pytest.approx(1.0)


def test_kl_divergence_identical_zero_infinite_support_and_directional() -> None:
    assert kl_divergence({"0": 0.25, "1": 0.75}, {"0": 0.25, "1": 0.75}) == pytest.approx(0.0)
    assert math.isinf(kl_divergence({"0": 1.0}, {"1": 1.0}))
    assert math.isfinite(kl_divergence({"1": 1.0}, {"0": 1.0, "1": 1e-13}))
    forward = kl_divergence({"0": 0.9, "1": 0.1}, {"0": 0.5, "1": 0.5})
    reverse = kl_divergence({"0": 0.5, "1": 0.5}, {"0": 0.9, "1": 0.1})
    assert forward != pytest.approx(reverse)


@pytest.mark.parametrize("base", [0.0, -2.0, 0.5, 1.0, math.nan, math.inf])
def test_invalid_log_base_raises_value_error(base: float) -> None:
    with pytest.raises(ValueError, match="log base"):
        kl_divergence({"0": 1.0}, {"0": 1.0}, base=base)
    with pytest.raises(ValueError, match="log base"):
        jensen_shannon_divergence({"0": 1.0}, {"0": 1.0}, base=base)


def test_jensen_shannon_divergence_known_values_and_symmetry() -> None:
    p = {"0": 0.8, "1": 0.2}
    q = {"0": 0.1, "1": 0.9}
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0)
    assert jensen_shannon_divergence({"0": 1.0}, {"1": 1.0}) == pytest.approx(1.0)
    assert jensen_shannon_divergence(p, q) == pytest.approx(jensen_shannon_divergence(q, p))


def test_jensen_shannon_distance_is_square_root_of_divergence() -> None:
    p = {"0": 0.8, "1": 0.2}
    q = {"0": 0.1, "1": 0.9}
    assert jensen_shannon_distance(p, q) == pytest.approx(math.sqrt(jensen_shannon_divergence(p, q)))


def test_compare_born_distributions_returns_expected_metric_bundle_and_names() -> None:
    bundle = compare_born_distributions({"0": 1.0}, {"1": 1.0})
    assert isinstance(bundle, BornMetricBundle)
    assert bundle.metric_family == "born"
    assert set(bundle.metrics) == {
        "total_variation",
        "hellinger",
        "jensen_shannon_divergence",
        "jensen_shannon_distance",
        "kl_clean_to_distorted",
        "kl_distorted_to_clean",
    }
    assert bundle.support == ["0", "1"]


def test_compare_born_distributions_has_json_friendly_metadata() -> None:
    bundle = compare_born_distributions({"0": 1.0}, {"0": 0.5, "1": 0.5}, context_metadata={"case": "json"})
    json.dumps(asdict(bundle))
    assert bundle.metadata["metric_family"] == "born"
    assert bundle.metadata["aligned_support_size"] == 2


def test_compare_born_distributions_accepts_simulation_result_objects_without_running_simulation() -> None:
    circuit = make_bell_circuit(2, measure=True)
    clean = simulate_ideal_statevector(circuit)
    sampled = simulate_ideal_shots(circuit, shots=64, seed=123)
    bundle = compare_born_distributions(clean, sampled)
    assert bundle.metadata["input_metadata"]["clean_simulation_mode"] == "ideal_statevector"
    assert bundle.metadata["input_metadata"]["distorted_simulation_mode"] == "ideal_shot"


def test_compare_born_distributions_does_not_mutate_input_dictionaries() -> None:
    clean = {"1": 3.0, "0": 1.0}
    distorted = {"2": 4.0}
    clean_before = dict(clean)
    distorted_before = dict(distorted)
    compare_born_distributions(clean, distorted)
    assert clean == clean_before
    assert distorted == distorted_before


def test_marker_only_context_metadata_adds_applicability_warning() -> None:
    bundle = compare_born_distributions({"0": 1.0}, {"0": 1.0}, context_metadata={"marker_only": True, "distortion_family": "readout"})
    assert "marker-only distortion context" in bundle.metadata["applicability_warning"]


def test_distortion_family_readout_without_marker_context_does_not_warn() -> None:
    bundle = compare_born_distributions({"0": 1.0}, {"0": 1.0}, context_metadata={"distortion_family": "readout"})
    assert "applicability_warning" not in bundle.metadata


def test_not_a_noisy_simulator_context_adds_applicability_warning() -> None:
    bundle = compare_born_distributions({"0": 1.0}, {"0": 1.0}, context_metadata={"not_a_noisy_simulator": True})
    assert "marker-only distortion context" in bundle.metadata["applicability_warning"]


def test_not_transpiled_context_adds_applicability_warning() -> None:
    bundle = compare_born_distributions({"0": 1.0}, {"0": 1.0}, context_metadata={"not_transpiled": True})
    assert "marker-only distortion context" in bundle.metadata["applicability_warning"]


def test_no_qiskit_aer_import_is_required() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import triqto.metrics; import sys; "
            "assert 'qiskit_aer' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_born_visible_rx_distortion_produces_nonzero_metric_shift() -> None:
    generated = make_bell_circuit(2, measure=True)
    clean = simulate_ideal_statevector(generated)
    distorted_circuit = apply_rx_overrotation(generated, strength=0.3).distorted_circuit
    distorted = simulate_ideal_statevector(distorted_circuit)
    bundle = compare_born_distributions(clean, distorted)
    assert bundle.metrics["total_variation"].value > 0.0
    assert bundle.metrics["hellinger"].value > 0.0


def test_phase_only_rz_drift_can_be_born_blind_for_relative_phase_in_computational_basis() -> None:
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    clean = simulate_ideal_statevector(circuit)
    distorted_circuit = apply_phase_rz_drift(circuit, strength=0.75).distorted_circuit
    distorted = simulate_ideal_statevector(distorted_circuit)
    bundle = compare_born_distributions(clean, distorted)
    # RZ changes the relative phase of |+>, so the Hilbert state changes.
    # Computational-basis Born probabilities still remain 50/50, demonstrating
    # why later Hilbert/phase metrics are needed for phase-sensitive damage.
    assert clean.probabilities == {"0": pytest.approx(0.5), "1": pytest.approx(0.5)}
    assert distorted.probabilities == {"0": pytest.approx(0.5), "1": pytest.approx(0.5)}
    assert bundle.metrics["total_variation"].value == pytest.approx(0.0)
    assert bundle.metrics["hellinger"].value == pytest.approx(0.0)
