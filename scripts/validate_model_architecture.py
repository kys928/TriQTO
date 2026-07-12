#!/usr/bin/env python3
"""Instantiate and forward-check the untrained Phase 13 TriQTO architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from triqto.model import (
    GraphTensorBatch,
    TriQTOBatch,
    TriQTOModel,
    architecture_manifest,
    load_model_config,
)


def _minimal_batch(model: TriQTOModel) -> TriQTOBatch:
    config = model.config
    graph = GraphTensorBatch(
        node_features=torch.zeros((3, config.node_input_dim), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_features=torch.zeros((2, config.edge_input_dim), dtype=torch.float32),
        edge_event_index=torch.tensor([0, 0], dtype=torch.long),
        gate_features=torch.zeros((1, config.gate_input_dim), dtype=torch.float32),
        gate_qubit_ptr=torch.tensor([0, 2], dtype=torch.long),
        gate_qubit_indices=torch.tensor([0, 1], dtype=torch.long),
        node_batch=torch.tensor([0, 0, 0], dtype=torch.long),
        gate_batch=torch.tensor([0], dtype=torch.long),
        graph_count=1,
    )
    return TriQTOBatch(graph=graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/model/triqto_small_debug.yaml",
        help="Strict Phase 13 YAML/JSON model config.",
    )
    parser.add_argument(
        "--manifest-output",
        help="Optional fresh JSON path for the untrained architecture manifest.",
    )
    args = parser.parse_args()

    config = load_model_config(args.config)
    model = TriQTOModel(config).eval()
    with torch.no_grad():
        output = model(_minimal_batch(model))
    manifest = architecture_manifest(model)
    summary = {
        "model_schema_id": manifest["model_schema_id"],
        "model_architecture_id": manifest["model_architecture_id"],
        "model_config_id": manifest["model_config_id"],
        "parameter_count": manifest["parameter_count"],
        "initialized_state_signature": manifest["initialized_state_signature"],
        "graph_embedding_shape": list(output.graph_embedding.shape),
        "node_embedding_shape": list(output.node_embeddings.shape),
        "head_latent_shape": list(output.head_latents.shape),
        "trained": manifest["trained"],
        "optimizer_state_present": manifest["optimizer_state_present"],
        "training_checkpoint": manifest["training_checkpoint"],
        "topology_loss_weight": manifest["topology_loss_weight"],
    }
    print(json.dumps(summary, sort_keys=True, indent=2))

    if args.manifest_output:
        target = Path(args.manifest_output)
        if target.exists():
            raise FileExistsError(f"Manifest output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
