"""Deterministic planning and conservative resource estimates for Phase 15.6."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from triqto.core.ids import make_deterministic_id
from triqto.data_generation import load_generation_config, predicted_sample_count
from triqto.model import load_model_config
from triqto.phase15_5 import load_phase155_config
from triqto.training import load_training_config
from triqto.training_views import load_training_view_config

from .config import Phase156CampaignConfig, phase156_config_to_dict

PHASE156_PLAN_SCHEMA = "triqto.phase15_6.plan.v1"


def resolve_config_path(repo_root: str | Path, value: str) -> Path:
    """Resolve an absolute path or a repository-relative path without requiring outputs in Git."""
    root = Path(repo_root).resolve()
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = root / target
    return target.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_record(repo_root: Path, path_value: str) -> dict[str, Any]:
    path = resolve_config_path(repo_root, path_value)
    if not path.is_file():
        raise FileNotFoundError(f"Phase 15.6 source config does not exist: {path}")
    try:
        reference = path.relative_to(repo_root).as_posix()
    except ValueError:
        reference = str(path)
    return {
        "reference": reference,
        "absolute_path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _resource_estimate(
    *,
    sample_count: int,
    max_qubits: int,
    measurement_basis_count: int,
    stores_statevectors: bool,
    training_seed_count: int,
) -> dict[str, Any]:
    """Return intentionally conservative planning estimates, not billing promises."""
    outcomes = 2**max_qubits
    statevector_bytes = outcomes * 16
    probability_bytes = outcomes * 8
    per_sample_payload = (
        2 * statevector_bytes + 2 * measurement_basis_count * probability_bytes + 24_000
        if stores_statevectors
        else 2 * measurement_basis_count * probability_bytes + 24_000
    )
    estimated_dataset_gb = sample_count * per_sample_payload / (1024**3)
    estimated_campaign_disk_gb = max(
        50.0,
        estimated_dataset_gb * 3.0 + 10.0 * training_seed_count,
    )
    max_simulator_working_set_gb = max(
        0.25,
        statevector_bytes * 12 / (1024**3),
    )
    if sample_count <= 20_000 and max_qubits <= 6:
        cpu_cores, memory_gb, gpu_vram_gb = 8, 32, 12
        tier = "pilot"
    elif sample_count <= 150_000 and max_qubits <= 10:
        cpu_cores, memory_gb, gpu_vram_gb = 24, 96, 24
        tier = "recommended"
    else:
        cpu_cores, memory_gb, gpu_vram_gb = 48, 192, 48
        tier = "large"
    memory_gb = max(memory_gb, int(math.ceil(max_simulator_working_set_gb * 8)))
    return {
        "estimate_kind": "conservative_heuristic_not_benchmark",
        "tier": tier,
        "sample_count": sample_count,
        "max_qubits": max_qubits,
        "measurement_basis_count": measurement_basis_count,
        "stores_statevectors": stores_statevectors,
        "training_seed_count": training_seed_count,
        "estimated_dataset_payload_gb": round(estimated_dataset_gb, 3),
        "recommended_persistent_disk_gb": int(math.ceil(estimated_campaign_disk_gb)),
        "estimated_peak_single_simulation_working_set_gb": round(max_simulator_working_set_gb, 3),
        "recommended_cpu_cores": cpu_cores,
        "recommended_system_memory_gb": memory_gb,
        "recommended_gpu_vram_gb": gpu_vram_gb,
        "gpu_role": "Phase 14 neural training; Phase 7/Qiskit Aer generation remains CPU-first by default",
        "parallelism_boundary": (
            "The v1 runner keeps deterministic Phase 7 generation single-process and runs seeds explicitly. "
            "Do not launch two writers into the same campaign workspace."
        ),
    }


def build_campaign_plan(
    *,
    repo_root: str | Path,
    config: Phase156CampaignConfig,
) -> dict[str, Any]:
    if not isinstance(config, Phase156CampaignConfig):
        raise TypeError("config must be Phase156CampaignConfig")
    root = Path(repo_root).resolve()
    sources = {
        "generation": _source_record(root, config.generation_config),
        "training_view": _source_record(root, config.training_view_config),
        "model": _source_record(root, config.model_config),
        "training": _source_record(root, config.training_config),
        "phase15_5": _source_record(root, config.phase15_5_config),
    }
    generation = load_generation_config(sources["generation"]["absolute_path"])
    training_view = load_training_view_config(sources["training_view"]["absolute_path"])
    model = load_model_config(sources["model"]["absolute_path"])
    training = load_training_config(sources["training"]["absolute_path"])
    phase155 = load_phase155_config(sources["phase15_5"]["absolute_path"])
    if training.topology_loss_weight != 0.0 or model.topology_loss_weight != 0.0:
        raise ValueError("Phase 15.6 source configs must keep topology loss exactly zero")
    if phase155.physical_hardware is not False or phase155.topology_loss_weight != 0.0:
        raise ValueError("Phase 15.6 Phase 15.5 config must remain offline with zero topology loss")
    sample_count = predicted_sample_count(generation)
    max_qubits = max(spec.n_qubits for spec in generation.circuit_specs)
    estimate = _resource_estimate(
        sample_count=sample_count,
        max_qubits=max_qubits,
        measurement_basis_count=len(generation.measurement_bases),
        stores_statevectors=generation.store_statevectors,
        training_seed_count=len(config.training_seeds),
    )
    payload = {
        "schema": PHASE156_PLAN_SCHEMA,
        "campaign_config": phase156_config_to_dict(config),
        "source_configs": sources,
        "dataset": {
            "scientific_sample_count": sample_count,
            "circuit_spec_count": len(generation.circuit_specs),
            "distortion_spec_count": len(generation.distortion_specs),
            "max_qubits": max_qubits,
            "measurement_bases": list(generation.measurement_bases),
            "backend_names": list(generation.backend_names),
            "stores_statevectors": generation.store_statevectors,
        },
        "training": {
            "training_seeds": list(config.training_seeds),
            "configured_tasks": list(training.configured_tasks),
            "epochs_per_seed": training.total_epochs,
            "requested_device": config.execution_device,
            "model_hidden_dim": model.hidden_dim,
            "model_message_passing_layers": model.graph_message_passing_layers,
            "phase15_5_epochs_per_seed": phase155.epochs,
        },
        "split_contract": {
            "split_grouping": training_view.split_grouping,
            "train_fraction": training_view.train_fraction,
            "validation_fraction": training_view.validation_fraction,
            "test_fraction": training_view.test_fraction,
            "test_used_for_optimization": False,
            "audit_only_used_for_optimization": False,
        },
        "stages": [
            {"name": "prepare", "writes": ["campaign_plan.json", "source_config_snapshots"]},
            {"name": "data", "writes": ["phase7", "phase8", "phase9", "phase11", "phase12"]},
            {
                "name": "train",
                "fanout": [{"training_seed": seed} for seed in config.training_seeds],
                "writes": ["phase14 checkpoint and training artifacts per seed"],
            },
            {
                "name": "evaluate",
                "enabled": config.run_phase15_5,
                "fanout": [{"training_seed": seed} for seed in config.training_seeds],
                "writes": ["Phase 15.5 policy checkpoint and grouped test report per seed"],
            },
            {"name": "aggregate", "writes": ["cross_seed_summary.json"]},
        ],
        "resource_estimate": estimate,
        "claim_boundaries": {
            "physical_hardware": False,
            "research_quality_claim_before_results": False,
            "broad_ood_claim": False,
            "calibrated_uncertainty_claim": False,
            "topology_benefit_claim": False,
            "topology_loss_weight": 0.0,
        },
    }
    identity_payload = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    for record in identity_payload["source_configs"].values():
        record.pop("absolute_path", None)
    payload["campaign_id"] = make_deterministic_id("phase156_campaign", identity_payload)
    return payload


def plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + "\n"


__all__ = [
    "PHASE156_PLAN_SCHEMA",
    "build_campaign_plan",
    "plan_json",
    "resolve_config_path",
]
