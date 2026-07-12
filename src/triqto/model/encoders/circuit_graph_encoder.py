"""Variable-size circuit graph encoder for Phase 13 TriQTO."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.contracts import GraphTensorBatch
from triqto.model.layers import GraphPooling, LatticeInteractionStack, ProjectionMLP
from triqto.model.tensor_ops import segment_mean


@dataclass(slots=True)
class GraphEncoderOutput:
    node_embeddings: Tensor
    gate_embeddings: Tensor
    edge_embeddings: Tensor
    graph_embedding: Tensor


class CircuitGraphEncoder(nn.Module):
    """Encode logical circuit structure without fixed qubit-count padding."""

    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.node_projection = ProjectionMLP(
            config.node_input_dim,
            hidden,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.edge_projection = ProjectionMLP(
            config.edge_input_dim,
            hidden,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.gate_projection = ProjectionMLP(
            config.gate_input_dim,
            hidden,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.node_gate_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.interaction = LatticeInteractionStack(
            hidden,
            config.graph_message_passing_layers,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.pooling = GraphPooling(
            hidden,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )

    def forward(self, batch: GraphTensorBatch) -> GraphEncoderOutput:
        node = self.node_projection(batch.node_features)
        edge = self.edge_projection(batch.edge_features)
        gate = self.gate_projection(batch.gate_features)
        gate_count = gate.shape[0]
        if gate_count:
            incidence_counts = batch.gate_qubit_ptr[1:] - batch.gate_qubit_ptr[:-1]
            incidence_gate_index = torch.arange(
                gate_count,
                device=gate.device,
                dtype=torch.long,
            ).repeat_interleave(incidence_counts)
            gate_on_incidence = gate.index_select(0, incidence_gate_index)
            gate_context = segment_mean(
                gate_on_incidence,
                batch.gate_qubit_indices,
                node.shape[0],
            )
        else:
            gate_context = torch.zeros_like(node)
        node = self.node_gate_fusion(torch.cat((node, gate_context), dim=1))
        edge_gate = (
            gate.index_select(0, batch.edge_event_index)
            if batch.edge_event_index.numel()
            else edge.new_zeros((0, edge.shape[1]))
        )
        node = self.interaction(node, batch.edge_index, edge, edge_gate)
        graph = self.pooling(node, batch.node_batch, batch.graph_count)
        return GraphEncoderOutput(
            node_embeddings=node,
            gate_embeddings=gate,
            edge_embeddings=edge,
            graph_embedding=graph,
        )


__all__ = ["CircuitGraphEncoder", "GraphEncoderOutput"]
