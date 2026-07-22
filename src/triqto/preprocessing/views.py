"""Task-specific training-view rows built after leakage-safe splitting."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

from .records import ProcessedSample
from .scaling import apply_scaler


def _flat_training_row(sample: ProcessedSample) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "family": sample.family,
        "n_qubits": sample.n_qubits,
        "intervention_label": sample.intervention_label,
        "observed_effect_label": sample.observed_effect_label,
        "observed_effect_confidence": sample.observed_effect_confidence,
        "severity": sample.severity,
        "measurement_basis": sample.measurement_basis,
        "source_type": sample.source_type,
        "shot_count": sample.shot_count,
        "combined_effect_score": sample.combined_effect_score,
        "hilbert_available": sample.masks.get("hilbert_available", False),
        "hardware_available": sample.masks.get("hardware_available", False),
        "layout_available": sample.masks.get("layout_available", False),
        "canonical_circuit_hash": sample.hashes.canonical_circuit_hash,
        "structural_graph_hash": sample.hashes.structural_graph_hash,
        "hardware_context_hash": sample.hashes.hardware_context_hash,
        "clean_circuit_id": sample.clean_circuit_id,
        "counterfactual_set_hash": sample.hashes.counterfactual_set_hash,
        "depth": len(sample.canonical_payload.get("distorted_circuit", {}).get("operations", [])),
        "one_qubit_event_count": sample.graph_features.get("one_qubit_event_count"),
        "two_qubit_event_count": sample.graph_features.get("two_qubit_event_count"),
        "measurement_event_count": sample.graph_features.get("measurement_event_count"),
    }
    for name, value in sample.effect_components.items():
        row[f"effect_{name}"] = value
    for name, value in sample.parameter_bindings_canonical.items():
        row[f"angle_{name}"] = value
    return row


def _scaled_feature_payload(
    row: Mapping[str, Any],
    scalers: Mapping[str, Any],
) -> dict[str, Any]:
    scaled: dict[str, Any] = {}
    for feature_name, scaler in sorted(scalers.items()):
        if not isinstance(scaler, Mapping) or scaler.get("status") != "fit_on_training_only":
            continue
        raw = row.get(feature_name)
        if raw is None:
            continue
        try:
            scaled[feature_name] = apply_scaler(float(raw), scaler)
        except (TypeError, ValueError, KeyError, OverflowError):
            scaled[feature_name] = None
    return scaled


def _task_view_rows(
    samples: list[ProcessedSample],
    assignments: list[Any],
    split_name: str,
    weights: Mapping[str, float],
    scalers: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    partition = {
        item.sample_id: item.partition
        for item in assignments
        if item.split_name == split_name
    }
    views: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if not sample.accepted or sample.sample_id not in partition:
            continue
        base = _flat_training_row(sample)
        base.update(
            {
                "split_name": split_name,
                "partition": partition[sample.sample_id],
                "sample_weight": weights.get(sample.sample_id, 1.0),
                "scaler_manifest_ref": "../../scalers/grouped_baseline_scalers.json",
                "parent_sample_id": sample.sample_id,
            }
        )
        base["scaled_features"] = _scaled_feature_payload(base, scalers)
        views["distortion_diagnosis"].append(
            {
                **base,
                "target": sample.intervention_label,
                "observed_effect_target": sample.observed_effect_label,
            }
        )
        views["observed_effect_classification"].append(
            {
                **base,
                "target": sample.observed_effect_label,
                "target_confidence": sample.observed_effect_confidence,
            }
        )
        views["no_action_abstention"].append(
            {
                **base,
                "target_no_action": sample.severity in {"negligible", "weak"},
                "target_abstain": sample.observed_effect_ambiguous,
            }
        )
        views["simulation_full_hilbert"].append(
            {
                **base,
                "hilbert_mask": sample.masks.get("hilbert_available", False),
                "statevector_ref": sample.provenance.get("distorted_statevector_ref")
                or sample.canonical_payload.get("statevector_storage", {}).get(
                    "distorted_statevector_ref"
                ),
            }
        )
        views["simulation_hilbert_masked"].append(
            {
                **base,
                "hilbert_mask": False,
                "hilbert_missingness": "intentionally_masked",
                "statevector_ref": None,
            }
        )
        if sample.shot_count is not None:
            views["finite_shot"].append(
                {
                    **base,
                    "uncertainty": sample.probability_uncertainty,
                }
            )
        if sample.hardware_context.get("backend_name") is not None:
            views["hardware_context"].append(
                {
                    **base,
                    "hardware_context": sample.hardware_context,
                    "hilbert_mask": False,
                }
            )
        if split_name in {"held_out_layout_identity", "held_out_layout_structure"}:
            views["layout_generalization"].append(
                {
                    **base,
                    "layout_identity": sample.hashes.labeled_graph_hash,
                    "layout_structure": sample.hashes.structural_graph_hash,
                }
            )
    return {
        name: sorted(rows, key=lambda row: (row["partition"], row["sample_id"]))
        for name, rows in sorted(views.items())
    }
