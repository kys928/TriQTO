"""Canonicalize, validate, audit, and hash one immutable Phase 7 sample."""
from __future__ import annotations

from typing import Any

from .canonical import (
    canonicalize_circuit,
    canonicalize_counts,
    canonicalize_parameter_bindings,
    canonicalize_probability_map,
    canonicalize_statevector_global_phase,
    circuit_graph_payload,
    circuit_parameter_signature,
)
from .config import PreprocessingConfig
from .effects import audit_observed_effect, compute_effects, hellinger_distance, severity_from_score
from .hashing import build_hash_bundle
from .io import (
    Phase7Source,
    load_counts,
    load_probabilities,
    load_qpy_circuit,
    load_statevector,
)
from .records import ProcessedSample
from .sample_context import (
    _extract_measurement_basis,
    _hardware_context,
    _quarantined_sample,
    _source_type,
)
from .shot_noise import (
    bootstrap_distance_interval,
    dirichlet_jeffreys_summary,
    repeated_batch_summary,
)
from .validation import (
    ValidationCollector,
    validate_counts,
    validate_layout_context,
    validate_manifest_record,
    validate_metric_ranges,
    validate_probability_distribution,
    validate_statevector,
)


def _process_sample(
    source: Phase7Source,
    sample: Any,
    config: PreprocessingConfig,
) -> ProcessedSample:
    locator = f"manifests/sample_manifest.parquet#sample_id={sample.sample_id}"
    collector = ValidationCollector()
    try:
        clean_circuit_record = source.circuits[sample.clean_circuit_id]
        distorted_circuit_record = source.circuits[sample.distorted_circuit_id]
        clean_run = source.simulations[sample.clean_run_id]
        distorted_run = source.simulations[sample.distorted_run_id]
        distortion = source.distortions[sample.distortion_id]
        metric = source.metrics[sample.metric_id]
    except KeyError as exc:
        collector.add("schema.join_missing", "error", locator, str(exc),
                      "all Phase 7 joins resolve", "quarantine")
        return _quarantined_sample(
            sample, source_locator=locator, collector=collector, config=config
        )

    for name, record in (
        ("sample", sample),
        ("clean_circuit", clean_circuit_record),
        ("distorted_circuit", distorted_circuit_record),
        ("clean_run", clean_run),
        ("distorted_run", distorted_run),
        ("distortion", distortion),
        ("metric", metric),
    ):
        validate_manifest_record(record, record_name=name, collector=collector)

    if clean_run.circuit_id != sample.clean_circuit_id:
        collector.add("schema.clean_run_circuit_join", "error", "clean_run.circuit_id",
                      clean_run.circuit_id, sample.clean_circuit_id, "quarantine")
    if distorted_run.circuit_id != sample.distorted_circuit_id:
        collector.add("schema.distorted_run_circuit_join", "error", "distorted_run.circuit_id",
                      distorted_run.circuit_id, sample.distorted_circuit_id, "quarantine")
    if distortion.circuit_id != sample.distorted_circuit_id:
        collector.add("schema.distortion_circuit_join", "error", "distortion.circuit_id",
                      distortion.circuit_id, sample.distorted_circuit_id, "quarantine")
    if metric.run_id != sample.distorted_run_id or metric.circuit_id != sample.distorted_circuit_id:
        collector.add("schema.metric_join", "error", "metric", metric.to_dict(),
                      "metric references distorted run and circuit", "quarantine")
    if clean_circuit_record.n_qubits != sample.n_qubits or distorted_circuit_record.n_qubits != sample.n_qubits:
        collector.add("schema.qubit_dimension_match", "error", locator,
                      [clean_circuit_record.n_qubits, distorted_circuit_record.n_qubits, sample.n_qubits],
                      "sample and circuit qubit dimensions match", "quarantine")
    if len(sample.parameter_bindings) != clean_circuit_record.parameter_count:
        collector.add("schema.parameter_dimension_match", "error", "sample.parameter_bindings",
                      len(sample.parameter_bindings),
                      f"parameter count {clean_circuit_record.parameter_count}", "quarantine")

    try:
        basis = _extract_measurement_basis(sample.metadata, config)
    except Exception as exc:
        collector.add("schema.measurement_basis", "error", "sample.metadata.measurement_setting",
                      str(exc), "recognized basis or Pauli string", "quarantine")
        basis = "unknown"

    try:
        clean_probabilities_raw = load_probabilities(source.root, clean_run)
        distorted_probabilities_raw = load_probabilities(source.root, distorted_run)
    except Exception as exc:
        collector.add("artifact.probability_load", "error", locator, str(exc),
                      "readable probability artifacts", "quarantine")
        return _quarantined_sample(
            sample, source_locator=locator, collector=collector, config=config
        )

    clean_probabilities = validate_probability_distribution(
        clean_probabilities_raw, width=sample.n_qubits, config=config,
        collector=collector, field_path="clean_probabilities"
    )
    distorted_probabilities = validate_probability_distribution(
        distorted_probabilities_raw, width=sample.n_qubits, config=config,
        collector=collector, field_path="distorted_probabilities"
    )
    for group_name, values in (
        ("born_metrics", metric.born_metrics),
        ("hilbert_metrics", metric.hilbert_metrics),
        ("parameter_metrics", metric.parameter_metrics),
        ("topology_metrics", metric.topology_metrics),
    ):
        validate_metric_ranges(values, collector=collector, field_path=f"metric.{group_name}")

    try:
        clean_state_raw = load_statevector(source.root, clean_run)
        distorted_state_raw = load_statevector(source.root, distorted_run)
    except Exception as exc:
        collector.add("artifact.statevector_load", "error", locator, str(exc),
                      "readable statevector when referenced", "quarantine")
        clean_state_raw = None
        distorted_state_raw = None

    clean_state = None if clean_state_raw is None else validate_statevector(
        clean_state_raw, n_qubits=sample.n_qubits, config=config,
        collector=collector, field_path="clean_statevector"
    )
    distorted_state = None if distorted_state_raw is None else validate_statevector(
        distorted_state_raw, n_qubits=sample.n_qubits, config=config,
        collector=collector, field_path="distorted_statevector"
    )

    hardware_context = _hardware_context(
        sample, clean_circuit_record, distorted_circuit_record,
        clean_run, distorted_run, config
    )
    validate_layout_context(
        hardware_context, n_qubits=sample.n_qubits, config=config,
        collector=collector, field_path="hardware_context"
    )
    if collector.quarantined:
        return _quarantined_sample(
            sample, source_locator=locator, collector=collector, config=config
        )

    try:
        clean_circuit = load_qpy_circuit(source.root, clean_circuit_record)
        distorted_circuit = load_qpy_circuit(source.root, distorted_circuit_record)
        canonical_clean_circuit = canonicalize_circuit(
            clean_circuit, config.canonicalization, config.numerical_tolerances
        )
        canonical_distorted_circuit = canonicalize_circuit(
            distorted_circuit, config.canonicalization, config.numerical_tolerances
        )
        canonical_parameters, wraps = canonicalize_parameter_bindings(
            sample.parameter_bindings, config.canonicalization
        )
        clean_probabilities_canonical = canonicalize_probability_map(
            clean_probabilities, width=sample.n_qubits,
            decimals=config.numerical_tolerances.hash_rounding_decimals
        )
        distorted_probabilities_canonical = canonicalize_probability_map(
            distorted_probabilities, width=sample.n_qubits,
            decimals=config.numerical_tolerances.hash_rounding_decimals
        )
        clean_state_canonical = None
        distorted_state_canonical = None
        clean_phase = distorted_phase = None
        clean_pivot = distorted_pivot = None
        if clean_state is not None:
            clean_state_canonical, clean_phase, clean_pivot = canonicalize_statevector_global_phase(
                clean_state,
                epsilon=config.canonicalization.state_global_phase_epsilon,
                norm_tolerance=config.numerical_tolerances.state_norm_repair,
            )
        if distorted_state is not None:
            distorted_state_canonical, distorted_phase, distorted_pivot = canonicalize_statevector_global_phase(
                distorted_state,
                epsilon=config.canonicalization.state_global_phase_epsilon,
                norm_tolerance=config.numerical_tolerances.state_norm_repair,
            )
    except Exception as exc:
        collector.add("canonicalization.failure", "error", locator, str(exc),
                      "deterministic canonicalization completes", "quarantine")
        return _quarantined_sample(
            sample, source_locator=locator, collector=collector, config=config
        )

    clean_graph = circuit_graph_payload(canonical_clean_circuit)
    distorted_graph = circuit_graph_payload(canonical_distorted_circuit)

    shot_runs = source.shot_runs_by_source_run.get(sample.distorted_run_id, [])
    shot_run = shot_runs[0] if shot_runs else None
    counts: dict[str, int] | None = None
    shot_count: int | None = None
    uncertainty: dict[str, Any] = {}
    repeated_counts: list[dict[str, int]] = []
    for index, repeated_run in enumerate(shot_runs):
        try:
            raw_counts = load_counts(source.root, repeated_run)
            checked = validate_counts(
                raw_counts, width=sample.n_qubits, declared_shots=repeated_run.shots,
                collector=collector, field_path=f"shot_runs[{index}].counts"
            )
            canonical_counts = canonicalize_counts(checked, width=sample.n_qubits)
            repeated_counts.append(canonical_counts)
            if index == 0:
                counts = canonical_counts
                shot_count = sum(canonical_counts.values())
        except Exception as exc:
            collector.add("artifact.shot_count_load", "error", f"shot_runs[{index}]", str(exc),
                          "valid finite-shot artifact", "quarantine")
    if counts and not collector.quarantined:
        uncertainty = dirichlet_jeffreys_summary(counts)
        uncertainty["repeated_batch_statistics"] = repeated_batch_summary(repeated_counts)
        clean_shot_runs = source.shot_runs_by_source_run.get(sample.clean_run_id, [])
        if clean_shot_runs:
            try:
                clean_counts = canonicalize_counts(
                    validate_counts(
                        load_counts(source.root, clean_shot_runs[0]), width=sample.n_qubits,
                        declared_shots=clean_shot_runs[0].shots, collector=collector,
                        field_path="clean_shot_counts"
                    ),
                    width=sample.n_qubits,
                )
                uncertainty["clean_vs_distorted_hellinger_bootstrap"] = bootstrap_distance_interval(
                    clean_counts, counts, distance_fn=hellinger_distance,
                    seed=config.random_seed, draws=256
                )
            except Exception as exc:
                collector.add("shot_noise.uncertainty_summary", "warning", locator, str(exc),
                              "uncertainty summary optional and non-destructive", "pass_with_warning")
    if collector.quarantined:
        return _quarantined_sample(
            sample, source_locator=locator, collector=collector, config=config
        )

    effects, combined_score, contributions = compute_effects(
        clean_probabilities=clean_probabilities_canonical,
        distorted_probabilities=distorted_probabilities_canonical,
        clean_statevector=clean_state_canonical,
        distorted_statevector=distorted_state_canonical,
        clean_parameters=circuit_parameter_signature(canonical_clean_circuit),
        distorted_parameters=circuit_parameter_signature(canonical_distorted_circuit),
        clean_graph=clean_graph,
        distorted_graph=distorted_graph,
        config=config.effects,
    )
    intervention_label = str(distortion.distortion_type)
    graph_changed = clean_graph.get("wl_feature_hash") != distorted_graph.get("wl_feature_hash")
    observed_label, observed_confidence, ambiguous, audit_flags = audit_observed_effect(
        intervention_label=intervention_label,
        measurement_basis=basis,
        effect_components=effects,
        combined_score=combined_score,
        graph_changed=graph_changed,
        readout_only="readout" in intervention_label.lower(),
        entanglement_evidence=None,
        config=config.effects,
    )
    severity = severity_from_score(combined_score, config.effects)

    provenance = {
        "dataset_name": sample.dataset_name,
        "raw_schema_version": sample.schema_version,
        "scientific_generation_id": source.completion_marker.get("scientific_generation_id"),
        "generation_config_id": source.completion_marker.get("config_id"),
        "source_circuit_id": sample.clean_circuit_id,
        "base_circuit_id": sample.clean_circuit_id,
        "target_id": sample.clean_run_id,
        "trajectory_id": sample.metadata.get("trajectory_id"),
        "candidate_set_id": sample.metadata.get("candidate_set_id"),
        "parameter_neighbourhood_id": sample.metadata.get("parameter_neighbourhood_id"),
        "generator_version": sample.metadata.get("generator_version") or sample.schema_version,
        "generation_seed": sample.base_seed,
        "parameter_seed": sample.metadata.get("parameter_seed"),
        "distortion_name": sample.metadata.get("distortion_name"),
        "distortion_strength": distortion.strength,
        "distortion_components": sample.metadata.get("distortion_components") or [intervention_label],
        "distortion_seed": distortion.metadata.get("seed"),
        "affected_qubits": list(distortion.affected_qubits),
        "affected_gates": list(distortion.affected_gates),
        "optimization_level": hardware_context.get("optimization_level"),
        "basis_gate_set": hardware_context.get("backend_basis_gates"),
        "noise_model_version": distortion.noise_model_id,
        "parameter_wrap_counts": wraps,
        "combined_effect_contributions": contributions,
        "clean_statevector_ref": clean_run.statevector_ref,
        "distorted_statevector_ref": distorted_run.statevector_ref,
    }
    missingness = {
        "hilbert": "available" if distorted_state_canonical is not None else "unavailable_by_design",
        "born": "available",
        "hardware": "available" if hardware_context.get("backend_name") else "not_computed",
        "shot_counts": "available" if counts is not None else "not_computed",
        "density_matrix": "not_computed",
        "layout": "available" if hardware_context.get("layout") is not None else "not_computed",
    }
    masks = {
        "hilbert_available": distorted_state_canonical is not None,
        "born_available": True,
        "hardware_available": hardware_context.get("backend_name") is not None,
        "shot_counts_available": counts is not None,
        "layout_available": hardware_context.get("layout") is not None,
    }

    raw_manifest_hashes = {
        record.relative_path: record.sha256
        for record in source.inventory
        if record.relative_path.startswith("manifests/")
    }
    raw_record_payload = {
        "source_locator": locator,
        "source_manifest_file_hashes": raw_manifest_hashes,
        "sample": source.raw_rows["samples"][sample.sample_id],
        "clean_circuit": source.raw_rows["circuits"][sample.clean_circuit_id],
        "distorted_circuit": source.raw_rows["circuits"][sample.distorted_circuit_id],
        "clean_run": source.raw_rows["simulations"][sample.clean_run_id],
        "distorted_run": source.raw_rows["simulations"][sample.distorted_run_id],
        "distortion": source.raw_rows["distortions"][sample.distortion_id],
        "metric": source.raw_rows["metrics"][sample.metric_id],
    }
    canonical_record_payload = {
        "sample_id": sample.sample_id,
        "clean_circuit": canonical_clean_circuit,
        "distorted_circuit": canonical_distorted_circuit,
        "parameters": canonical_parameters,
        "measurement_basis": basis,
        "clean_probabilities": clean_probabilities_canonical,
        "distorted_probabilities": distorted_probabilities_canonical,
        "hardware_context": hardware_context,
        "intervention_label": intervention_label,
    }
    target_payload = (
        {"kind": "clean_global_phase_canonical_state", "statevector": clean_state_canonical}
        if clean_state_canonical is not None
        else {"kind": "clean_basis_conditioned_born_distribution", "basis": basis,
              "probabilities": clean_probabilities_canonical}
    )
    counterfactual_payload = {
        "clean_circuit_id": sample.clean_circuit_id,
        "distorted_circuit_id": sample.distorted_circuit_id,
        "distortion_id": sample.distortion_id,
        "candidate_ids": [],
        "candidate_status": "not_available_in_phase7_input",
    }
    hashes = build_hash_bundle(
        raw_record_payload=raw_record_payload,
        canonical_record_payload=canonical_record_payload,
        canonical_circuit_payload=canonical_distorted_circuit,
        canonical_parameters=canonical_parameters,
        statevector=distorted_state_canonical,
        born_distribution=distorted_probabilities_canonical,
        measurement_basis=basis,
        counts=counts,
        shot_count=shot_count,
        hardware_context=hardware_context,
        target_payload=target_payload,
        counterfactual_payload=counterfactual_payload,
        graph_payload=distorted_graph,
        config=config,
    )
    canonical_payload = {
        **canonical_record_payload,
        "clean_graph": clean_graph,
        "distorted_graph": distorted_graph,
        "counts": counts,
        "shot_count": shot_count,
        "statevector_storage": {
            "clean_statevector_ref": clean_run.statevector_ref,
            "distorted_statevector_ref": distorted_run.statevector_ref,
            "canonicalization": "global_phase_only; relative_phase_preserved",
        },
    }
    audit_flags.extend(
        finding.rule_id for finding in collector.findings if finding.repair_applied
    )
    if clean_phase is not None:
        provenance["clean_global_phase_factor"] = [float(clean_phase.real), float(clean_phase.imag)]
        provenance["clean_global_phase_pivot"] = clean_pivot
    if distorted_phase is not None:
        provenance["distorted_global_phase_factor"] = [float(distorted_phase.real), float(distorted_phase.imag)]
        provenance["distorted_global_phase_pivot"] = distorted_pivot

    return ProcessedSample(
        sample_id=sample.sample_id,
        source_locator=locator,
        accepted=True,
        quarantine_reason=None,
        family=sample.family,
        n_qubits=sample.n_qubits,
        repetition_index=sample.repetition_index,
        clean_circuit_id=sample.clean_circuit_id,
        distorted_circuit_id=sample.distorted_circuit_id,
        clean_run_id=sample.clean_run_id,
        distorted_run_id=sample.distorted_run_id,
        distortion_id=sample.distortion_id,
        metric_id=sample.metric_id,
        intervention_label=intervention_label,
        observed_effect_label=observed_label,
        observed_effect_confidence=observed_confidence,
        observed_effect_ambiguous=ambiguous,
        effect_components=effects,
        combined_effect_score=combined_score,
        severity=severity,
        parameter_bindings_original={str(k): float(v) for k, v in sample.parameter_bindings.items()},
        parameter_bindings_canonical=canonical_parameters,
        measurement_basis=basis,
        source_type=_source_type(distorted_run, shot_run),
        shot_count=shot_count,
        probability_uncertainty=uncertainty,
        graph_features=distorted_graph,
        hardware_context=hardware_context,
        provenance=provenance,
        missingness=missingness,
        masks=masks,
        hashes=hashes,
        findings=list(collector.findings),
        audit_flags=sorted(set(audit_flags)),
        canonical_payload=canonical_payload,
    )
