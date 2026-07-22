"""Per-sample preprocessing context and quarantine construction."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping

from .canonical import canonical_basis_label
from .config import PreprocessingConfig
from .hashing import sha256_payload
from .io import Phase7Source
from .records import HashBundle, ProcessedSample
from .validation import ValidationCollector, quarantine_reason


def _extract_measurement_basis(sample_metadata: Mapping[str, Any], config: PreprocessingConfig) -> str:
    setting = sample_metadata.get("measurement_setting")
    candidate: Any = None
    if isinstance(setting, Mapping):
        for key in ("basis", "label", "name"):
            if setting.get(key) not in (None, ""):
                candidate = setting[key]
                break
        if candidate is None:
            bases = setting.get("bases")
            if isinstance(bases, (list, tuple)) and bases:
                normalized = [str(item).upper() for item in bases]
                candidate = normalized[0] if len(set(normalized)) == 1 else "".join(normalized)
    if candidate is None:
        candidate = sample_metadata.get("measurement_basis", "Z")
    return canonical_basis_label(candidate, config.canonicalization)


def _calibration_window_id(timestamp: Any, seconds: int) -> str | None:
    if timestamp in (None, ""):
        return None
    if isinstance(timestamp, (int, float)) and math.isfinite(float(timestamp)):
        numeric = float(timestamp)
    elif isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        numeric = parsed.timestamp()
    else:
        return None
    return f"window_{int(math.floor(numeric / seconds))}"


def _hardware_context(
    sample: Any,
    clean_circuit_record: Any,
    distorted_circuit_record: Any,
    clean_run: Any,
    distorted_run: Any,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in (
        sample.metadata,
        clean_circuit_record.metadata,
        distorted_circuit_record.metadata,
        clean_run.metadata,
        distorted_run.metadata,
    ):
        if isinstance(source, Mapping):
            metadata.update(source)
    timestamp = metadata.get("backend_calibration_timestamp") or metadata.get(
        "calibration_timestamp"
    )
    return {
        "backend_id": metadata.get("backend_id"),
        "backend_name": metadata.get("backend_name") or distorted_run.backend_name,
        "backend_source": metadata.get("backend_source"),
        "backend_class": metadata.get("backend_class"),
        "backend_n_qubits": metadata.get("backend_n_qubits"),
        "backend_basis_gates": metadata.get("backend_basis_gates"),
        "coupling_map": metadata.get("backend_coupling_map")
        or metadata.get("coupling_map"),
        "calibration_timestamp": timestamp,
        "calibration_snapshot_id": metadata.get("calibration_snapshot_id")
        or metadata.get("backend_id"),
        "calibration_window_id": _calibration_window_id(
            timestamp, config.grouping.calibration_window_seconds
        ),
        "backend_run_id": metadata.get("backend_run_id"),
        "feature_values": metadata.get("backend_feature_values"),
        "feature_available": metadata.get("backend_feature_available"),
        "missing_reasons": metadata.get("backend_missing_reasons"),
        "layout": metadata.get("layout") or metadata.get("initial_layout"),
        "routing": metadata.get("routing") or metadata.get("routing_method"),
        "optimization_level": metadata.get("optimization_level"),
        "noise_model_id": metadata.get("noise_model_id"),
        "hardware_quality_bin": metadata.get("hardware_quality_bin"),
    }


def _source_type(distorted_run: Any, shot_run: Any | None) -> str:
    if distorted_run.simulation_mode == "ideal_statevector" and shot_run is None:
        return "exact_statevector_probability"
    if shot_run is not None and shot_run.simulation_mode == "ideal_shot":
        return "finite_shot_ideal_simulation"
    mode = str(distorted_run.simulation_mode).lower()
    if "noisy" in mode:
        return "finite_shot_noisy_simulation"
    if "hardware" in mode:
        return "real_hardware"
    return "hardware_mode_simulation"


def _placeholder_hash_bundle(sample_id: str, config: PreprocessingConfig) -> HashBundle:
    digest = sha256_payload("quarantine_placeholder", {"sample_id": sample_id}, config=config)
    return HashBundle(
        raw_record_hash=digest,
        canonical_record_hash=digest,
        canonical_circuit_hash=digest,
        circuit_parameter_hash=digest,
        state_equivalence_hash=None,
        born_distribution_hash=digest,
        measurement_instance_hash=None,
        hardware_context_hash=digest,
        target_hash=None,
        counterfactual_set_hash=digest,
        labeled_graph_hash=digest,
        structural_graph_hash=digest,
        feature_graph_hash=digest,
    )


def _quarantined_sample(
    sample: Any,
    *,
    source_locator: str,
    collector: ValidationCollector,
    config: PreprocessingConfig,
    reason: str | None = None,
) -> ProcessedSample:
    return ProcessedSample(
        sample_id=str(getattr(sample, "sample_id", "unknown")),
        source_locator=source_locator,
        accepted=False,
        quarantine_reason=reason or quarantine_reason(collector.findings) or "unexpected_error",
        family=str(getattr(sample, "family", "unknown")),
        n_qubits=int(getattr(sample, "n_qubits", 0) or 0),
        repetition_index=int(getattr(sample, "repetition_index", 0) or 0),
        clean_circuit_id=str(getattr(sample, "clean_circuit_id", "")),
        distorted_circuit_id=str(getattr(sample, "distorted_circuit_id", "")),
        clean_run_id=str(getattr(sample, "clean_run_id", "")),
        distorted_run_id=str(getattr(sample, "distorted_run_id", "")),
        distortion_id=str(getattr(sample, "distortion_id", "")),
        metric_id=str(getattr(sample, "metric_id", "")),
        intervention_label="invalid_or_unknown",
        observed_effect_label="invalid_or_corrupted",
        observed_effect_confidence=1.0,
        observed_effect_ambiguous=False,
        effect_components={},
        combined_effect_score=None,
        severity="catastrophic",
        parameter_bindings_original=dict(getattr(sample, "parameter_bindings", {}) or {}),
        parameter_bindings_canonical={},
        measurement_basis="unknown",
        source_type="exact_statevector_probability",
        shot_count=None,
        probability_uncertainty={},
        graph_features={},
        hardware_context={},
        provenance={},
        missingness={
            "hilbert": "corrupted",
            "born": "corrupted",
            "hardware": "unknown_legacy_record",
        },
        masks={"hilbert_available": False, "born_available": False},
        hashes=_placeholder_hash_bundle(str(getattr(sample, "sample_id", "unknown")), config),
        findings=list(collector.findings),
        audit_flags=["quarantined"],
        canonical_payload={},
    )
