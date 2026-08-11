from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v0_2/audit_matched_bdelta_identifiability.py"
SPEC = importlib.util.spec_from_file_location("matched_bdelta_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_rotation_matrices_are_unitary() -> None:
    for axis in ("rx", "ry", "rz"):
        matrix = AUDIT.rotation_matrix(axis, 0.37)
        np.testing.assert_allclose(
            matrix.conj().T @ matrix,
            np.eye(2),
            atol=1e-12,
            rtol=0.0,
        )


def test_rz_on_plus_is_phase_only_under_overlap_decomposition() -> None:
    clean = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    distorted = AUDIT.apply_single_qubit_matrix(
        clean, 0, AUDIT.rotation_matrix("rz", 0.2)
    )
    result = AUDIT.overlap_decomposition(clean, distorted, epsilon=1e-12)
    assert result["population_component"] < 1e-12
    assert result["phase_component"] > 0.0
    assert result["dominance_log_ratio"] > 0.0


def test_rx_on_zero_changes_population_magnitude() -> None:
    clean = np.asarray([1.0, 0.0], dtype=np.complex128)
    distorted = AUDIT.apply_single_qubit_matrix(
        clean, 0, AUDIT.rotation_matrix("rx", 0.4)
    )
    result = AUDIT.overlap_decomposition(clean, distorted, epsilon=1e-12)
    assert result["population_component"] > 0.0
    assert result["phase_component"] < 1e-12
    assert result["dominance_log_ratio"] < 0.0


def test_xyz_evidence_distinguishes_rz_from_rx_on_plus() -> None:
    clean = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rz = AUDIT.apply_single_qubit_matrix(
        clean, 0, AUDIT.rotation_matrix("rz", 0.2)
    )
    rx = AUDIT.apply_single_qubit_matrix(
        clean, 0, AUDIT.rotation_matrix("rx", 0.2)
    )
    rz_evidence = AUDIT.evidence_for_state(clean, rz, n_qubits=1)
    rx_evidence = AUDIT.evidence_for_state(clean, rx, n_qubits=1)
    separation = AUDIT.pair_separation(
        rz_evidence,
        rx_evidence,
        epsilon=1e-12,
        raw_minimum=1e-6,
        relative_minimum=0.25,
        collision_maximum=1e-10,
    )
    assert separation["pair_separation_score"] > 1e-6
    assert separation["strong_pair"] is True
    assert separation["numerical_collision"] is False


def test_qubit_and_strength_parsing() -> None:
    assert AUDIT.parse_affected_qubit("q[3]", 5) == 3
    assert AUDIT.parse_affected_qubit("q0", 2) == 0
    assert math.isclose(AUDIT.parse_strength("0.15"), 0.15)
    assert math.isclose(AUDIT.parse_strength("strength=5e-2"), 0.05)


def test_source_preflight_accepts_terminal_matching_rotation(tmp_path: Path) -> None:
    artifact = tmp_path / "item.npz"
    clean = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    np.savez_compressed(
        artifact,
        entity_id=np.asarray("e1"),
        c__clean_statevector_real=clean.real,
        c__clean_statevector_imag=clean.imag,
        a__x_graph_gate_names=np.asarray(["h", "rz"]),
        audit__removed_distortion_gate_indices=np.asarray([1], dtype=np.int64),
    )
    row = {
        "entity_id": "e1",
        "artifact_ref": "item.npz",
        "raw_label": "phase_rz_drift",
        "split_group_id": "g1",
        "family": "bell",
        "phase_sensitive_family": True,
        "n_qubits": 1,
        "strength_key": "0.05",
        "affected_qubit_signature": "q0",
    }
    info = AUDIT.inspect_source_example(tmp_path.resolve(), row)
    assert info["valid"] is True
    assert info["terminal"] is True
    assert info["axis_match"] is True


def test_source_preflight_rejects_nonterminal_rotation(tmp_path: Path) -> None:
    artifact = tmp_path / "item.npz"
    clean = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    np.savez_compressed(
        artifact,
        entity_id=np.asarray("e1"),
        c__clean_statevector_real=clean.real,
        c__clean_statevector_imag=clean.imag,
        a__x_graph_gate_names=np.asarray(["rz", "h"]),
        audit__removed_distortion_gate_indices=np.asarray([0], dtype=np.int64),
    )
    row = {
        "entity_id": "e1",
        "artifact_ref": "item.npz",
        "raw_label": "phase_rz_drift",
        "split_group_id": "g1",
        "family": "bell",
        "phase_sensitive_family": True,
        "n_qubits": 1,
        "strength_key": "0.05",
        "affected_qubit_signature": "q0",
    }
    info = AUDIT.inspect_source_example(tmp_path.resolve(), row)
    assert info["valid"] is False
    assert info["reason"] == "nonterminal_distortion"


def test_identifiable_decision_on_uniformly_strong_synthetic_pairs() -> None:
    config = json.loads(
        (ROOT / "configs/v0_2/matched_bdelta_identifiability_audit.json").read_text()
    )
    pairs = []
    counterfactuals = []
    for index in range(20):
        context = f"c{index}"
        for pair_type in ("rz_vs_rx", "rz_vs_ry", "rx_vs_ry"):
            pairs.append(
                {
                    "context_id": context,
                    "pair_type": pair_type,
                    "family": "synthetic",
                    "n_qubits": 2,
                    "strength_key": "0.1",
                    "phase_sensitive_family": False,
                    "affected_qubit_signature": "q0",
                    "strong_pair": True,
                    "numerical_collision": False,
                    "phenotype_differs": True,
                    "pair_separation_score": 0.2,
                    "relative_separation": 1.0,
                }
            )
        for mechanism in AUDIT.MECHANISMS:
            counterfactuals.append(
                {
                    "context_id": context,
                    "mechanism": mechanism,
                    "phenotype": "phase_dominant" if mechanism == "rz_drift" else "population_dominant",
                }
            )
    strata = AUDIT.stratified_pair_summaries(pairs)
    decision = AUDIT.decide(pairs, counterfactuals, strata, config)
    assert decision["status"] == "IDENTIFIABLE"
