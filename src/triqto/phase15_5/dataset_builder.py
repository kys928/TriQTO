"""Matched operational-supervision dataset construction for Phase 15.5."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
from qiskit import transpile
from triqto.actions import layout_selection_action, routing_transpilation_action, semantics_verified_depth_reduction
from triqto.backends import local_line_backend
from triqto.core.ids import make_deterministic_id
from triqto.data_generation import derive_child_seed
from triqto.metrics.hilbert import purity
from triqto.simulation import NoiseSpec, simulate_density_matrix
from .candidate_utils import _backend_summary, _candidate_features, _cost_utility, _distribution_summary, _evidence_for_basis, _no_op_row, _noise_strength
from .config import Phase155Config
from .constants import CANDIDATE_FEATURE_NAMES, CONTEXT_SUMMARY_NAMES
from .policy import FAMILY_NAMES, PolicyDataset

def _build_rows(
    *,
    phase7: Any,
    examples_by_split: Mapping[str, Sequence[Any]],
    latents: Mapping[str, np.ndarray],
    config: Phase155Config,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], PolicyDataset, tuple[str, ...], tuple[str, ...]]:
    samples_by_id = {value.sample_id: value for value in phase7.samples}
    evidence_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    latent_dim = len(next(iter(latents.values())))
    context_names = tuple([f"diagnosis_latent_{index}" for index in range(latent_dim)] + list(CONTEXT_SUMMARY_NAMES))
    for split in ("train", "validation", "test"):
        for example in examples_by_split[split]:
            sample = samples_by_id.get(example.entity_id)
            if sample is None:
                raise ValueError(f"Phase 15.5 diagnosis entity {example.entity_id} has no Phase 7 sample")
            latent = latents.get(sample.sample_id)
            if latent is None:
                raise ValueError(f"Phase 15.5 sample {sample.sample_id} has no checkpoint latent")
            clean = phase7.circuits_by_id[sample.clean_circuit_id]
            distorted = phase7.circuits_by_id[sample.distorted_circuit_id]
            backend = local_line_backend(max(2, sample.n_qubits), name=str(sample.metadata.get("backend_name", "triqto_phase155_fake")))
            degree_mean, degree_max = _backend_summary(sample)
            for profile in config.noise_profiles:
                profile_payload = {"name": profile.name, "channels": list(profile.channels), "shots": profile.shots}
                profile_id = make_deterministic_id("phase155_noise_profile", profile_payload)
                noise = NoiseSpec(profile.channels)
                base_seed = derive_child_seed(config.seed, "phase155_sample_noise", {"sample_id": sample.sample_id, "noise_profile_id": profile_id})
                basis_evidence = {
                    basis: _evidence_for_basis(clean, distorted, noise=noise, shots=profile.shots, seed=base_seed, basis=basis)
                    for basis in config.measurement_bases
                }
                z_basis = basis_evidence["Z"] if "Z" in basis_evidence else basis_evidence[config.measurement_bases[0]]
                distribution_summary = _distribution_summary(z_basis["distorted_probabilities"], sample.n_qubits)
                context = np.concatenate(
                    (
                        latent,
                        np.asarray(
                            [
                                sample.n_qubits / 8.0,
                                _noise_strength(profile),
                                len(profile.channels) / 8.0,
                                *distribution_summary,
                                degree_mean,
                                degree_max,
                            ],
                            dtype=np.float64,
                        ),
                    )
                )
                if context.shape != (len(context_names),) or not np.isfinite(context).all():
                    raise ValueError("Phase 15.5 context feature shape/value mismatch")
                density_summary: dict[str, Any] | None = None
                if config.include_density_matrix:
                    density = simulate_density_matrix(distorted, noise=noise, seed=base_seed)
                    density_summary = {
                        "purity": float(purity(density.density_matrix)),
                        "probabilities": dict(sorted(density.probabilities.items())),
                        "noise_model_id": density.metadata.get("noise_model_id"),
                    }
                evidence_rows.append(
                    {
                        "evidence_id": make_deterministic_id("phase155_noisy_evidence", {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "basis_evidence": basis_evidence, "density_summary": density_summary}),
                        "sample_id": sample.sample_id,
                        "split": split,
                        "split_group_id": example.split_group_id,
                        "noise_profile_id": profile_id,
                        "noise_profile": profile_payload,
                        "basis_evidence": basis_evidence,
                        "density_summary": density_summary,
                        "physical_hardware": False,
                        "evidence_tier": "noisy_simulator",
                        "privileged_clean_pair_used_for_supervision_only": True,
                    }
                )
                for family_id in range(len(FAMILY_NAMES)):
                    rows.append(_no_op_row(sample, profile_id, split, example.split_group_id, family_id, context))
                for basis, evidence in sorted(basis_evidence.items()):
                    utility = max(0.0, min(1.0, float(evidence["jensen_shannon_divergence"]) - config.probe_cost))
                    family_id = 0
                    payload = {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id], "kind": "probe", "basis": basis, "shots": profile.shots}
                    rows.append(
                        {
                            "candidate_id": make_deterministic_id("phase155_candidate", payload),
                            "group_id": make_deterministic_id("phase155_group", {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id]}),
                            "split_group_id": example.split_group_id,
                            "split": split,
                            "family_id": family_id,
                            "context": context,
                            "candidate": np.asarray(_candidate_features(kind="probe", basis=basis, acquires_evidence=True), dtype=np.float64),
                            "utility": utility,
                            "available": True,
                            "metadata": {**payload, "available": True, "target_js_divergence": evidence["jensen_shannon_divergence"], "probe_cost": config.probe_cost, "target_uses_clean_pair": True, "input_excludes_clean_pair": True},
                        }
                    )
                for seed in config.layout_seeds:
                    _, result = layout_selection_action(distorted, backend, seed=seed)
                    family_id = 1
                    objective = dict(result.objective_comparison)
                    payload = {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id], "kind": "layout", "seed": seed, "backend_id": backend.backend_id}
                    rows.append(
                        {
                            "candidate_id": make_deterministic_id("phase155_candidate", payload),
                            "group_id": make_deterministic_id("phase155_group", {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id]}),
                            "split_group_id": example.split_group_id,
                            "split": split,
                            "family_id": family_id,
                            "context": context,
                            "candidate": np.asarray(_candidate_features(kind="layout", objective=objective, backend_evidence_available=True), dtype=np.float64),
                            "utility": _cost_utility(objective, accepted=result.available),
                            "available": bool(result.available),
                            "metadata": {**payload, "available": bool(result.available), "status": result.status, "rejection_reason": result.rejection_reason, "objective": objective, "target_uses_semantics_preserving_compile_evidence": True},
                        }
                    )
                for level in config.routing_optimization_levels:
                    _, result = routing_transpilation_action(distorted, backend, seed=base_seed, optimization_level=level)
                    family_id = 2
                    objective = dict(result.objective_comparison)
                    payload = {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id], "kind": "routing", "optimization_level": level, "backend_id": backend.backend_id}
                    rows.append(
                        {
                            "candidate_id": make_deterministic_id("phase155_candidate", payload),
                            "group_id": make_deterministic_id("phase155_group", {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id]}),
                            "split_group_id": example.split_group_id,
                            "split": split,
                            "family_id": family_id,
                            "context": context,
                            "candidate": np.asarray(_candidate_features(kind="routing", optimization_level=level, objective=objective, backend_evidence_available=True), dtype=np.float64),
                            "utility": _cost_utility(objective, accepted=result.available),
                            "available": bool(result.available),
                            "metadata": {**payload, "available": bool(result.available), "status": result.status, "rejection_reason": result.rejection_reason, "objective": objective, "target_uses_semantics_preserving_compile_evidence": True},
                        }
                    )
                for level in config.depth_optimization_levels:
                    candidate_circuit = transpile(distorted, optimization_level=level, seed_transpiler=base_seed)
                    result = semantics_verified_depth_reduction(distorted, candidate_circuit)
                    family_id = 3
                    objective = dict(result.objective_comparison)
                    payload = {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id], "kind": "depth", "optimization_level": level}
                    rows.append(
                        {
                            "candidate_id": make_deterministic_id("phase155_candidate", payload),
                            "group_id": make_deterministic_id("phase155_group", {"sample_id": sample.sample_id, "noise_profile_id": profile_id, "family": FAMILY_NAMES[family_id]}),
                            "split_group_id": example.split_group_id,
                            "split": split,
                            "family_id": family_id,
                            "context": context,
                            "candidate": np.asarray(_candidate_features(kind="depth", optimization_level=level, objective=objective, semantic_validation_available=bool(result.evidence)), dtype=np.float64),
                            "utility": _cost_utility(objective, accepted=result.available),
                            "available": bool(result.available),
                            "metadata": {**payload, "available": bool(result.available), "status": result.status, "rejection_reason": result.rejection_reason, "objective": objective, "semantic_validation_method": result.semantic_validation_method, "target_uses_simulator_semantic_validation": True},
                        }
                    )
    rows.sort(key=lambda value: value["candidate_id"])
    candidate_ids = tuple(value["candidate_id"] for value in rows)
    dataset = PolicyDataset(
        candidate_ids=candidate_ids,
        group_ids=tuple(value["group_id"] for value in rows),
        split_group_ids=tuple(value["split_group_id"] for value in rows),
        splits=tuple(value["split"] for value in rows),
        family_ids=np.asarray([value["family_id"] for value in rows], dtype=np.int64),
        context_features=np.ascontiguousarray(np.stack([value["context"] for value in rows]), dtype=np.float64),
        candidate_features=np.ascontiguousarray(np.stack([value["candidate"] for value in rows]), dtype=np.float64),
        utilities=np.asarray([value["utility"] for value in rows], dtype=np.float64),
        available_mask=np.asarray([value["available"] for value in rows], dtype=np.bool_),
    )
    dataset.validate()
    metadata_rows = [
        {
            "candidate_id": value["candidate_id"],
            "group_id": value["group_id"],
            "split_group_id": value["split_group_id"],
            "split": value["split"],
            "family": FAMILY_NAMES[value["family_id"]],
            "utility": value["utility"],
            "available": value["available"],
            "metadata": value["metadata"],
        }
        for value in rows
    ]
    return evidence_rows, metadata_rows, dataset, context_names, CANDIDATE_FEATURE_NAMES
