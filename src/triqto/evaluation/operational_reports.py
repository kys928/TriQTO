"""Family-specific Phase 15 reporting for operational-action artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from triqto.actions.operational_artifacts import load_operational_action_dataset
from triqto.topology.checkpoint_latent import load_checkpoint_bound_latent_topology


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_operational_actions(root: str | Path) -> dict[str, Any]:
    dataset = load_operational_action_dataset(root)
    results = dataset["results"]
    if any(result.physical_hardware for result in results):
        raise ValueError("offline operational report rejects physical-hardware rows")
    probes = [result for result in results if result.action_type == "basis_probe"]
    compilation = [result for result in results if result.action_type in {"layout_selection", "routing_transpilation"}]
    reductions = [result for result in results if result.action_type == "depth_reduction"]
    probe_success = [result for result in probes if result.status == "accepted"]
    settings = sorted({"".join(result.after_metadata.get("measurement_setting", {}).get("measurement_bases", [])) for result in probe_success} - {""})
    compilation_pass = [result for result in compilation if result.status == "accepted" and result.semantic_validation_method]
    reduction_accepted = [result for result in reductions if result.status == "accepted"]
    report = {
        "operational_action_dataset_id": dataset["manifest"]["operational_action_dataset_id"],
        "evidence_tier": dataset["manifest"]["evidence_tier"],
        "physical_hardware": False,
        "claim_scope": "family-specific operational engineering metrics; not logical correction success",
        "basis_probe": {
            "available_count": sum(result.available for result in probes),
            "selected_or_executed_count": len(probe_success),
            "selection_rate": len(probe_success) / len(probes) if probes else None,
            "measurement_settings": settings,
            "new_evidence_acquired_count": sum(bool(result.evidence.get("probe_evidence_id")) for result in probe_success),
            "failed_or_invalid_count": len(probes) - len(probe_success),
            "downstream_diagnosis_change": None,
            "not_correction_success": True,
        },
        "layout_and_routing": {
            "available_count": sum(result.available for result in compilation),
            "semantic_validation_pass_count": len(compilation_pass),
            "semantic_validation_pass_rate": len(compilation_pass) / len(compilation) if compilation else None,
            "mean_depth_delta": _mean([float(result.objective_comparison.get("depth_delta", 0.0)) for result in compilation_pass]),
            "mean_two_qubit_gate_delta": _mean([float(result.objective_comparison.get("two_qubit_gate_delta", 0.0)) for result in compilation_pass]),
            "mean_swap_count": _mean([float(result.objective_comparison["swap_count"]) for result in compilation_pass if "swap_count" in result.objective_comparison]),
            "rejected_count": sum(result.status != "accepted" for result in compilation),
            "rejection_reasons": sorted({result.rejection_reason for result in compilation if result.rejection_reason}),
            "not_logical_fidelity_correction": True,
        },
        "depth_reduction": {
            "accepted_count": len(reduction_accepted),
            "rejected_count": sum(result.status == "rejected" for result in reductions),
            "no_op_count": sum(result.status == "no_op" for result in reductions),
            "mean_state_fidelity": _mean([float(result.evidence["state_fidelity"]) for result in reduction_accepted if "state_fidelity" in result.evidence]),
            "mean_depth_delta": _mean([float(result.objective_comparison.get("depth_delta", 0.0)) for result in reduction_accepted]),
            "mean_size_delta": _mean([float(result.objective_comparison.get("size_delta", 0.0)) for result in reduction_accepted]),
            "mean_two_qubit_gate_delta": _mean([float(result.objective_comparison.get("two_qubit_gate_delta", 0.0)) for result in reduction_accepted]),
            "rejection_reasons": sorted({result.rejection_reason for result in reductions if result.rejection_reason}),
        },
    }
    return report


def summarize_checkpoint_latent_topology(root: str | Path) -> dict[str, Any]:
    artifact = load_checkpoint_bound_latent_topology(root)
    result, metadata = artifact["result"], artifact["result"]["metadata"]
    if result.get("physical_hardware") is not False or result.get("diagnostic_only") is not True or result.get("topology_loss_weight") != 0.0:
        raise ValueError("latent topology report violates diagnostic-only boundary")
    return {
        "latent_topology_id": result["latent_topology_id"],
        "latent_extraction_id": result["coordinate_source_identity"],
        "checkpoint_id": metadata["checkpoint_id"],
        "checkpoint_content_hash": metadata["checkpoint_content_hash"],
        "model_architecture_id": metadata["model_architecture_id"],
        "model_config_id": metadata["model_config_id"],
        "training_view_dataset_id": metadata["training_view_dataset_id"],
        "split": metadata["split"],
        "head": metadata["head"],
        "representation": metadata["representation"],
        "point_count": len(metadata["point_ids"]),
        "coordinate_dim": metadata["coordinate_dim"],
        "normalization_mode": metadata["normalization_mode"],
        "persistence_summary": result["persistence_summary"],
        "diagnostic_only": True,
        "topology_loss_weight": 0.0,
        "no_topology_benefit_claim": True,
    }


__all__ = ["summarize_checkpoint_latent_topology", "summarize_operational_actions"]
