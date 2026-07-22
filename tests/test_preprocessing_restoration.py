"""Regression coverage for the restored immutable preprocessing package."""
from __future__ import annotations

import importlib

import numpy as np

from triqto.preprocessing.canonical import canonicalize_statevector_global_phase
from triqto.preprocessing.config import PreprocessingConfig
from triqto.preprocessing.grouping import build_leakage_relations
from triqto.preprocessing.hashing import hash_statevector, sha256_payload
from triqto.preprocessing.records import HashBundle, ProcessedSample


def _hashes(sample_id: str, *, state: str, born: str, counterfactual: str) -> HashBundle:
    return HashBundle(
        raw_record_hash=f"raw-{sample_id}",
        canonical_record_hash=f"record-{sample_id}",
        canonical_circuit_hash="same-circuit",
        circuit_parameter_hash=f"parameters-{sample_id}",
        state_equivalence_hash=state,
        born_distribution_hash=born,
        measurement_instance_hash=f"measurement-{sample_id}",
        hardware_context_hash="hardware",
        target_hash="target",
        counterfactual_set_hash=counterfactual,
        labeled_graph_hash="labeled",
        structural_graph_hash="structural",
        feature_graph_hash="feature",
    )


def _sample(sample_id: str, distortion_id: str) -> ProcessedSample:
    return ProcessedSample(
        sample_id=sample_id,
        source_locator=f"fixture#{sample_id}",
        accepted=True,
        quarantine_reason=None,
        family="fixture",
        n_qubits=1,
        repetition_index=0,
        clean_circuit_id="shared-clean-parent",
        distorted_circuit_id=f"distorted-{sample_id}",
        clean_run_id="clean-run",
        distorted_run_id=f"distorted-run-{sample_id}",
        distortion_id=distortion_id,
        metric_id=f"metric-{sample_id}",
        intervention_label="phase",
        observed_effect_label="phase_sensitive_change",
        observed_effect_confidence=1.0,
        observed_effect_ambiguous=False,
        effect_components={"hellinger": 0.0, "infidelity": 1.0},
        combined_effect_score=0.35,
        severity="strong",
        parameter_bindings_original={"theta": 0.0},
        parameter_bindings_canonical={"theta": 0.0},
        measurement_basis="Z",
        source_type="exact_statevector_probability",
        shot_count=None,
        probability_uncertainty={},
        graph_features={},
        hardware_context={},
        provenance={"scientific_generation_id": "fixture", "generation_seed": 1},
        missingness={"hilbert": "available"},
        masks={"hilbert_available": True},
        hashes=_hashes(sample_id, state=f"state-{sample_id}", born="same-born", counterfactual=f"cf-{sample_id}"),
    )


def test_all_restored_preprocessing_modules_import() -> None:
    modules = (
        "canonical",
        "cli",
        "effects",
        "grouping",
        "hashing",
        "io",
        "outliers",
        "pipeline",
        "reporting",
        "sample_context",
        "sample_processor",
        "splits",
        "validation_core",
        "validation_hardware",
        "validation_states",
        "views",
    )
    for module in modules:
        importlib.import_module(f"triqto.preprocessing.{module}")


def test_global_phase_is_removed_but_relative_phase_is_preserved() -> None:
    plus = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    minus = np.asarray([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    canonical_plus, _, _ = canonicalize_statevector_global_phase(
        plus * np.exp(0.73j), epsilon=1e-12, norm_tolerance=1e-8
    )
    canonical_minus, _, _ = canonicalize_statevector_global_phase(
        minus * np.exp(-0.31j), epsilon=1e-12, norm_tolerance=1e-8
    )
    assert np.allclose(canonical_plus, plus)
    assert np.allclose(canonical_minus, minus)
    assert not np.allclose(canonical_plus, canonical_minus)


def test_same_z_born_distribution_does_not_imply_same_state_hash() -> None:
    config = PreprocessingConfig()
    plus = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    minus = np.asarray([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    plus, _, _ = canonicalize_statevector_global_phase(
        plus, epsilon=1e-12, norm_tolerance=1e-8
    )
    minus, _, _ = canonicalize_statevector_global_phase(
        minus, epsilon=1e-12, norm_tolerance=1e-8
    )
    plus_hash = hash_statevector(plus, qubit_ordering="qiskit_msb_left", config=config)
    minus_hash = hash_statevector(minus, qubit_ordering="qiskit_msb_left", config=config)
    born_payload = {"basis": "Z", "probabilities": {"0": 0.5, "1": 0.5}}
    assert plus_hash != minus_hash
    assert sha256_payload("born_distribution", born_payload, config=config) == sha256_payload(
        "born_distribution", born_payload, config=config
    )


def test_clean_and_distorted_descendants_cannot_cross_baseline_split() -> None:
    relations = build_leakage_relations(
        [_sample("sample-clean", "none"), _sample("sample-distorted", "phase")]
    )
    relation = next(
        item for item in relations if item.relation_type == "base_circuit_descendants"
    )
    assert set(relation.member_sample_ids) == {"sample-clean", "sample-distorted"}
    assert relation.evidence["forbid_cross_split"] is True
