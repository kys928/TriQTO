from __future__ import annotations

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit

from triqto.data_generation import (
    CircuitGenerationSpec,
    DatasetGenerationConfig,
    DistortionSpec,
    assess_identifiability,
    generate_dataset,
    reject_conflicting_identifiable_labels,
)
from triqto.evaluation import (
    build_identifiability_evaluation_report,
    filter_diagnosis_evaluation_rows,
)
from triqto.model.encoders.born_encoder import MeasurementConditionedBasisEncoder
from triqto.simulation import (
    apply_independent_readout_bitflips,
    measurement_setting,
    simulate_measurement_probabilities,
)


def phase_config(
    *,
    settings: tuple[str, ...],
    policy: str = "mask",
    distortion: DistortionSpec | None = None,
) -> DatasetGenerationConfig:
    return DatasetGenerationConfig(
        dataset_name="measurement-identifiability",
        base_seed=17,
        circuit_specs=[
            CircuitGenerationSpec(
                family="phase_interference",
                n_qubits=2,
                generator_kwargs={"measure": True},
                repetitions=1,
            )
        ],
        distortion_specs=[
            distortion
            or DistortionSpec(
                "phase_rz_drift",
                {"strength": 0.4, "qubits": [0]},
            )
        ],
        measurement_settings=settings,
        unidentifiable_policy=policy,
        max_samples=4,
    )


def test_pauli_measurement_settings_are_physical_and_explicit() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    z = simulate_measurement_probabilities(circuit, "Z")
    x = simulate_measurement_probabilities(circuit, "X")
    y = simulate_measurement_probabilities(circuit, "Y")
    assert z.probabilities == pytest.approx({"0": 0.5, "1": 0.5})
    assert x.probabilities == pytest.approx({"0": 1.0})
    assert y.probabilities == pytest.approx({"0": 0.5, "1": 0.5})
    assert z.measurement_setting.bases == ("Z",)
    assert len({z.measurement_setting.setting_id, x.measurement_setting.setting_id, y.measurement_setting.setting_id}) == 3
    with pytest.raises(ValueError, match="one of"):
        measurement_setting("A", 1)
    with pytest.raises(ValueError, match="exactly 2"):
        measurement_setting("XYZ", 2)


def test_generation_rejects_product_basis_that_cannot_fit_all_circuits() -> None:
    with pytest.raises(ValueError, match="must match every circuit width"):
        DatasetGenerationConfig(
            dataset_name="bad-product-setting",
            base_seed=3,
            circuit_specs=[
                CircuitGenerationSpec("bell", 2, {"measure": True}),
                CircuitGenerationSpec("ghz", 3, {"measure": True}),
            ],
            distortion_specs=[DistortionSpec("rx_overrotation", {"strength": 0.1})],
            measurement_settings=("ZX",),
        )


def test_exact_readout_channel_changes_observable_distribution() -> None:
    changed = apply_independent_readout_bitflips(
        {"0": 1.0},
        n_qubits=1,
        probability=0.25,
        qubits=[0],
    )
    assert changed == pytest.approx({"0": 0.75, "1": 0.25})
    circuit = QuantumCircuit(1)
    result = simulate_measurement_probabilities(
        circuit,
        "Z",
        readout_bitflip_probability=0.25,
        readout_qubits=[0],
    )
    assert result.probabilities == pytest.approx(changed)
    assert result.metadata["readout_channel"]["channel"] == "independent_symmetric_readout_bitflip"


def test_rz_is_unidentifiable_under_z_only_and_visible_with_x_y() -> None:
    z_only = generate_dataset(phase_config(settings=("Z",)))
    z_sample = z_only.samples[0]
    assert z_sample.metadata["identifiability_status"] == "unidentifiable"
    assert z_sample.metadata["identifiability_reason"] == "computational_basis_phase_blindness"
    assert z_sample.metadata["diagnosis_supervision_mask"] is False
    assert z_only.sample_records[0].diagnosis_supervision_mask is False
    assert z_only.summary["identifiable_diagnosis_coverage"] == 0.0

    multi_basis = generate_dataset(phase_config(settings=("Z", "X", "Y")))
    sample = multi_basis.samples[0]
    assert sample.metadata["identifiability_status"] == "conditionally_identifiable"
    assert sample.metadata["identifiability_reason"] == "requires_selected_measurement_settings"
    assert sample.metadata["diagnosis_supervision_mask"] is True
    assert sample.metadata["visible_measurement_setting_ids"]
    assert sample.metadata["blind_measurement_setting_ids"]
    assert multi_basis.summary["identifiable_diagnosis_coverage"] == 1.0


def test_marker_only_layout_is_audit_only_and_strict_mode_rejects_it() -> None:
    distortion = DistortionSpec(
        "layout_permutation_marker",
        {"permutation": [1, 0]},
    )
    masked = generate_dataset(
        phase_config(settings=("Z", "X", "Y"), distortion=distortion)
    )
    sample = masked.sample_records[0]
    assert sample.identifiability_status == "unidentifiable"
    assert sample.identifiability_reason == "backend_feature_unavailable"
    assert sample.diagnosis_supervision_mask is False
    with pytest.raises(ValueError, match="strict policy"):
        generate_dataset(
            phase_config(
                settings=("Z", "X", "Y"),
                policy="error",
                distortion=distortion,
            )
        )


def test_explicit_unidentifiable_override_is_machine_recorded() -> None:
    result = generate_dataset(
        phase_config(settings=("Z",), policy="allow")
    )
    sample = result.sample_records[0]
    assert sample.identifiability_status == "unidentifiable"
    assert sample.diagnosis_supervision_mask is True
    assert sample.metadata["unidentifiable_supervision_override"] is True


def test_conflicting_labels_with_identical_allowed_evidence_are_rejected() -> None:
    rows = [
        {
            "observable_evidence_fingerprint": "same",
            "distortion_type": "phase_rz_drift",
            "identifiability_status": "identifiable",
        },
        {
            "observable_evidence_fingerprint": "same",
            "distortion_type": "layout_permutation_marker",
            "identifiability_status": "unidentifiable",
        },
    ]
    with pytest.raises(ValueError, match="Conflicting supervised diagnosis labels"):
        reject_conflicting_identifiable_labels(rows)
    for row in rows:
        row["identifiability_status"] = "unidentifiable"
    reject_conflicting_identifiable_labels(rows)


def test_assessment_rejects_mismatched_measurement_provenance() -> None:
    setting = measurement_setting("Z", 1)
    with pytest.raises(ValueError, match="clean probability settings"):
        assess_identifiability(
            distortion_type="phase_rz_drift",
            marker_only=False,
            measurement_settings={setting.setting_id: setting},
            clean_probabilities={},
            distorted_probabilities={setting.setting_id: {"0": 1.0}},
            atol=1e-12,
        )


def test_measurement_basis_is_part_of_the_learned_born_representation() -> None:
    torch.manual_seed(3)
    encoder = MeasurementConditionedBasisEncoder(16).eval()
    bits = torch.tensor([[0.0], [1.0]])
    mask = torch.ones((2, 1), dtype=torch.bool)
    z = encoder(bits, mask, torch.zeros((2, 1), dtype=torch.long))
    x = encoder(bits, mask, torch.ones((2, 1), dtype=torch.long))
    assert not torch.allclose(z, x)


def test_measurement_setting_rows_have_per_setting_normalization() -> None:
    result = generate_dataset(phase_config(settings=("Z", "X", "Y")))
    sample = result.samples[0]
    for measurement in sample.distorted_measurement_results.values():
        assert np.isclose(sum(measurement.probabilities.values()), 1.0)


def test_headline_evaluation_excludes_unidentifiable_overrides() -> None:
    identifiable = generate_dataset(phase_config(settings=("Z", "X", "Y")))
    overridden = generate_dataset(phase_config(settings=("Z",), policy="allow"))
    rows = [identifiable.sample_records[0], overridden.sample_records[0]]
    report = build_identifiability_evaluation_report(rows)
    assert report.to_dict() == {
        "total_count": 2,
        "default_scored_count": 1,
        "default_excluded_count": 1,
        "explicit_override_count": 1,
        "default_scored_coverage": 0.5,
        "status_counts": {
            "conditionally_identifiable": 1,
            "unidentifiable": 1,
        },
        "reason_counts": {
            "computational_basis_phase_blindness": 1,
            "requires_selected_measurement_settings": 1,
        },
        "unidentifiable_rows_in_headline_metrics": False,
    }
    assert filter_diagnosis_evaluation_rows(rows) == [rows[0]]
    assert filter_diagnosis_evaluation_rows(
        rows,
        include_explicit_unidentifiable_overrides=True,
    ) == rows
