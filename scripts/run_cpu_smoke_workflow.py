#!/usr/bin/env python3
"""Run a deterministic CPU end-to-end TriQTO smoke workflow into a chosen output directory."""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json
from qiskit import QuantumCircuit

from triqto.actions import ActionEngineConfig, build_action_engine_result, write_action_dataset
from triqto.actions.operational import basis_probe_action, layout_selection_action, routing_transpilation_action, semantics_verified_depth_reduction
from triqto.actions.operational_adapter import OperationalViewAdapterConfig, build_operational_action_tensor_batch, load_operational_view_adapter_config
from triqto.actions.operational_artifacts import write_operational_action_dataset
from triqto.actions.operational_config import OperationalActionSmokeConfig, load_operational_action_smoke_config
from triqto.backends import local_line_backend
from triqto.data_generation import generate_dataset, load_generation_config, write_dataset
from triqto.evaluation import load_phase15_config, run_phase15_evaluation
from triqto.evaluation.integrated import load_integrated_phase15_config, run_integrated_phase15_evaluation
from triqto.graph import GraphConversionConfig, convert_completed_dataset_to_graphs, write_graph_dataset
from triqto.model import TriQTOModelConfig
from triqto.topology import TopologyAuditConfig, build_topology_audit_result, write_topology_dataset
from triqto.topology.checkpoint_latent import run_checkpoint_bound_latent_topology
from triqto.topology.latent import load_latent_topology_config
from triqto.training import CurriculumStageConfig, LossConfig, OptimizerConfig, SchedulerConfig, TrainingConfig, run_training
from triqto.training.latent_extraction import extract_checkpoint_latents, load_latent_extraction_config
from triqto.training_views import build_training_view_result, load_training_view_config, write_training_view_dataset


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        run_name="cpu_smoke_phase14", seed=2026,
        stages=(CurriculumStageConfig(name="diagnosis_smoke", epochs=1, tasks=("diagnosis",)),),
        batch_size=2,
        optimizer=OptimizerConfig(name="adamw", learning_rate=1e-3, weight_decay=0.0),
        scheduler=SchedulerConfig(name="constant", warmup_steps=0, minimum_learning_rate_ratio=1.0),
        loss=LossConfig(geometry_weight=0.0, uncertainty_weighting=False),
        deterministic_algorithms=True, device="cpu", checkpoint_every_epochs=1,
        early_stopping_patience=0, topology_loss_weight=0.0,
    )


def _model_config() -> TriQTOModelConfig:
    return TriQTOModelConfig(
        hidden_dim=32, graph_message_passing_layers=1, residual_mlp_layers=1,
        backend_input_dim=16, topology_input_dim=8, hilbert_deformation_dim=8,
        topology_prediction_dim=8, dropout=0.0, initialization_seed=2026,
    )


def _operational_results(
    config: OperationalActionSmokeConfig,
    adapter_config: OperationalViewAdapterConfig,
) -> list[object]:
    backend = local_line_backend(config.backend_n_qubits, name=config.backend_name)
    source = QuantumCircuit(config.backend_n_qubits)
    source.h(0)
    source.cx(0, 1)
    probe = basis_probe_action(
        config.backend_n_qubits,
        config.probe_bases,
        circuit=source,
        shots=config.probe_shots,
        seed=config.seed,
    )
    _, layout = layout_selection_action(source, backend, seed=config.seed)
    _, routing = routing_transpilation_action(
        source,
        backend,
        seed=config.seed,
        optimization_level=config.transpilation_optimization_level,
    )
    reducible = QuantumCircuit(1)
    reducible.h(0)
    reducible.h(0)
    reducible.x(0)
    reduced = QuantumCircuit(1)
    reduced.x(0)
    depth = semantics_verified_depth_reduction(
        reducible,
        reduced,
        tolerance=config.semantic_tolerance,
    )
    results = [probe, layout, routing, depth]
    adapter = build_operational_action_tensor_batch(results, config=adapter_config)
    if bool(adapter.candidate_target_mask.any()) or bool(adapter.privileged_information_mask.any()):
        raise RuntimeError("operational adapter exposed invalid supervision or privilege")
    return results


def run(output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Smoke output already exists: {output}")
    output.mkdir(parents=True)
    phase7, phase8, phase9 = output / "phase7", output / "phase8", output / "phase9"
    operational, phase11 = output / "operational_actions", output / "phase11"
    phase12, phase14 = output / "phase12", output / "phase14"
    latent, latent_topology = output / "latent_extraction", output / "latent_topology"
    phase15, integrated_root = output / "phase15", output / "phase15_integrated"

    operational_config = load_operational_action_smoke_config(ROOT / "configs/actions/operational_smoke.yaml")
    adapter_config = load_operational_view_adapter_config(ROOT / "configs/training_views/operational_actions_smoke.yaml")
    latent_config = load_latent_extraction_config(ROOT / "configs/eval/latent_extraction_smoke.yaml")

    generation = generate_dataset(load_generation_config(ROOT / "configs/data/backend_holdout_generation.json"))
    write_dataset(generation, phase7)
    graph = convert_completed_dataset_to_graphs(phase7, GraphConversionConfig(include_supplemental_counts=False))
    write_graph_dataset(graph, phase8)
    actions = build_action_engine_result(phase7, phase8, ActionEngineConfig(candidate_magnitudes=(0.1, 0.2), max_candidates_per_sample=64, max_edits_per_action=16))
    write_action_dataset(actions, phase9)
    operational_result = write_operational_action_dataset(
        operational,
        _operational_results(operational_config, adapter_config),
        source_dataset_id=generation.scientific_generation_id,
        evidence_tier=operational_config.evidence_tier,
    )
    topology = build_topology_audit_result(phase7, phase8, phase9, TopologyAuditConfig(min_points=3, betti_grid_size=8, top_k_lifetimes=2, max_points_per_group=128, max_groups=64, max_statevector_amplitudes=64, include_hilbert=False))
    write_topology_dataset(topology, phase11)
    views = build_training_view_result(phase7, phase8, phase9, phase11, load_training_view_config(ROOT / "configs/training_views/backend_holdout.yaml"))
    write_training_view_dataset(views, phase12)
    training = run_training(training_view_root=phase12, output_root=phase14, training_config=_training_config(), model_config=_model_config(), phase7_root=phase7)
    checkpoint = phase14 / "artifacts" / "checkpoints" / f"final-epoch-{training.epoch_metrics[-1].epoch:04d}.npz"
    latent_result = extract_checkpoint_latents(
        training_view_root=phase12,
        training_root=phase14,
        checkpoint=checkpoint,
        output_root=latent,
        config=latent_config,
        phase7_root=phase7,
    )
    latent_topology_result = run_checkpoint_bound_latent_topology(
        latent_extraction_root=latent, output_root=latent_topology,
        config=load_latent_topology_config(ROOT / "configs/eval/latent_topology_smoke.yaml"),
    )
    core_config = load_phase15_config(ROOT / "configs/eval/phase15_smoke.yaml")
    phase15_result = run_phase15_evaluation(training_view_root=phase12, training_root=phase14, checkpoint=checkpoint, output_root=phase15, config=core_config, phase7_root=phase7)
    integrated_result = run_integrated_phase15_evaluation(
        training_view_root=phase12, training_root=phase14, checkpoint=checkpoint,
        output_root=integrated_root, core_config=core_config,
        integration_config=load_integrated_phase15_config(ROOT / "configs/eval/phase15_operational_topology_smoke.yaml"),
        operational_action_root=operational, latent_topology_root=latent_topology, phase7_root=phase7,
    )
    manifest = {
        "label": "smoke engineering validation",
        "commands": ["scripts/run_cpu_smoke_workflow.py --output <dir>"],
        "seed": operational_config.seed,
        "dependency_profile": "CPU-safe pinned repository environment",
        "evidence_tier": "fake_backend_fixture",
        "operational_evidence_tier": operational_config.evidence_tier,
        "artifact_ids": {
            "phase7_generation_id": generation.scientific_generation_id,
            "phase8_graph_conversion_id": graph.graph_conversion_id,
            "phase9_action_engine_id": actions.action_engine_id,
            "operational_action_dataset_id": operational_result["manifest"]["operational_action_dataset_id"],
            "phase12_training_view_dataset_id": views.training_view_dataset_id,
            "phase14_training_run_id": training.training_run_id,
            "checkpoint_id": phase15_result["summary"]["checkpoint_id"],
            "latent_extraction_id": latent_result["metadata"]["latent_extraction_id"],
            "latent_topology_id": latent_topology_result["result"]["latent_topology_id"],
            "phase15_run_id": phase15_result["summary"]["phase15_run_id"],
            "integrated_phase15_run_id": integrated_result["summary"]["integrated_phase15_run_id"],
        },
        "test_split": phase15_result["summary"]["split_semantics"],
        "latent_split": latent_result["metadata"]["split"],
        "metrics": phase15_result["summary"]["metrics"],
        "operational_action_families_pooled": False,
        "topology_loss_weight": 0.0,
        "limitations": [
            "not research-quality evidence", "no physical hardware",
            "fake backend is offline fixture evidence",
            "basis probe is evidence acquisition, not correction",
            "no calibration or physical-hardware generalization claim",
            "no topology-benefit or causal-topology claim",
            "topology loss remains exactly zero",
        ],
    }
    (output / "smoke_workflow_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New output directory outside committed artifacts.")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.output))["artifact_ids"], sort_keys=True))


if __name__ == "__main__":
    main()
