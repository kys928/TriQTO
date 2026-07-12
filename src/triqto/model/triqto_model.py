"""Top-level mask-safe Phase 13 TriQTO neural architecture."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import TriQTOModelConfig
from .constants import HEAD_ORDER, HEAD_STREAM_POLICY, STREAM_ORDER
from .contracts import TriQTOBatch
from .encoders import (
    BackendEncoder,
    BornEncoder,
    CircuitGraphEncoder,
    HilbertEncoder,
    ParameterEncoder,
    PhasorEncoder,
    TopologyEncoder,
)
from .fusion import DualModeFusion
from .heads import (
    ActionRankingHead,
    BornPredictionHead,
    DistortionHead,
    HilbertDeformationHead,
    TopologyHead,
    UncertaintyHead,
)
from .identities import model_architecture_id
from .layers import ResidualMLP
from .outputs import TriQTOModelOutput


class TriQTOModel(nn.Module):
    """Variable-size, dual-mode, phase-coupled graph architecture.

    Phase 13 implements forward computation only. No optimizer, training loop,
    checkpoint schedule, learned correction claim, or hardware execution is included.
    """

    def __init__(self, config: TriQTOModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or TriQTOModelConfig()
        if not isinstance(self.config, TriQTOModelConfig):
            raise TypeError("config must be TriQTOModelConfig or None")
        self.architecture_id = model_architecture_id(self.config)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.initialization_seed)
            self._build_modules()
            self.apply(self._initialize_module)

    def _build_modules(self) -> None:
        config = self.config
        hidden = config.hidden_dim
        self.graph_encoder = CircuitGraphEncoder(config)
        self.parameter_encoder = ParameterEncoder(config)
        self.phasor_encoder = PhasorEncoder(config)
        self.hilbert_encoder = HilbertEncoder(config) if config.use_hilbert else None
        self.born_encoder = BornEncoder(config)
        self.backend_encoder = BackendEncoder(config) if config.use_backend else None
        self.topology_encoder = TopologyEncoder(config) if config.use_topology else None
        self.fusion = DualModeFusion(
            hidden,
            len(STREAM_ORDER),
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.head_embeddings = nn.Embedding(len(HEAD_ORDER), hidden)
        self.head_refinement = ResidualMLP(
            hidden,
            layers=config.residual_mlp_layers,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.distortion_head = DistortionHead(config)
        self.action_ranking_head = ActionRankingHead(config)
        self.born_prediction_head = BornPredictionHead(config)
        self.hilbert_deformation_head = HilbertDeformationHead(config)
        self.uncertainty_head = UncertaintyHead(config)
        self.topology_head = TopologyHead(config)
        policy = torch.tensor(
            [HEAD_STREAM_POLICY[head] for head in HEAD_ORDER],
            dtype=torch.bool,
        )
        self.register_buffer("hard_head_stream_policy", policy, persistent=True)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _optional_stream(
        self,
        encoder: nn.Module | None,
        batch: object | None,
        graph_count: int,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if encoder is None:
            return (
                reference.new_zeros((graph_count, self.config.hidden_dim)),
                torch.zeros(graph_count, dtype=torch.bool, device=reference.device),
            )
        return encoder(batch, graph_count, reference)  # type: ignore[misc]

    def forward(self, batch: TriQTOBatch) -> TriQTOModelOutput:
        if not isinstance(batch, TriQTOBatch):
            raise TypeError("batch must be TriQTOBatch")
        batch.validate(self.config)
        graph_output = self.graph_encoder(batch.graph)
        graph_count = batch.graph.graph_count
        graph_embedding = graph_output.graph_embedding
        graph_available = torch.ones(
            graph_count,
            dtype=torch.bool,
            device=graph_embedding.device,
        )
        parameter, parameter_available = self.parameter_encoder(
            batch.parameter,
            graph_count,
            graph_embedding,
        )
        phasor, phasor_available = self.phasor_encoder(
            batch.parameter,
            graph_count,
            graph_embedding,
        )
        hilbert, hilbert_available = self._optional_stream(
            self.hilbert_encoder,
            batch.hilbert,
            graph_count,
            graph_embedding,
        )
        born, born_available = self.born_encoder(
            batch.born,
            graph_count,
            graph_embedding,
        )
        backend, backend_available = self._optional_stream(
            self.backend_encoder,
            batch.backend,
            graph_count,
            graph_embedding,
        )
        topology, topology_available = self._optional_stream(
            self.topology_encoder,
            batch.topology,
            graph_count,
            graph_embedding,
        )
        hardware_mode = batch.resolved_hardware_mask()
        topology_dependency = batch.resolved_topology_hilbert_dependency()
        hilbert_available = hilbert_available & ~hardware_mode
        topology_available = topology_available & ~(hardware_mode & topology_dependency)
        hilbert = hilbert * hilbert_available.to(hilbert.dtype).unsqueeze(1)
        topology = topology * topology_available.to(topology.dtype).unsqueeze(1)
        stream_embeddings = torch.stack(
            (graph_embedding, parameter, phasor, hilbert, born, backend, topology),
            dim=1,
        )
        stream_available = torch.stack(
            (
                graph_available,
                parameter_available,
                phasor_available,
                hilbert_available,
                born_available,
                backend_available,
                topology_available,
            ),
            dim=1,
        )
        effective = (
            stream_available.unsqueeze(1)
            & self.hard_head_stream_policy.unsqueeze(0)
        )
        if batch.head_stream_mask is not None:
            effective = effective & batch.head_stream_mask
        head_active = batch.resolved_head_active_mask()
        empty_active = head_active & ~effective.any(dim=2)
        if bool(empty_active.any()):
            missing = torch.nonzero(empty_active, as_tuple=False)[0]
            raise ValueError(
                "An active head has no permitted available stream after masking: "
                f"batch={int(missing[0])}, head={HEAD_ORDER[int(missing[1])]}"
            )
        head_latents: list[Tensor] = []
        fusion_weights: list[Tensor] = []
        for head_index, _head_name in enumerate(HEAD_ORDER):
            active = head_active[:, head_index]
            fusion_mask = effective[:, head_index, :].clone()
            # MaskAwareFusion requires one stream per row. Inactive heads receive a
            # temporary graph anchor only for shape-safe computation; outputs are
            # zeroed immediately and the anchor is never reported as effective.
            fusion_mask[~active, 0] = True
            latent, weights = self.fusion(
                stream_embeddings,
                fusion_mask,
                hardware_mode,
            )
            head_index_tensor = torch.full(
                (graph_count,),
                head_index,
                dtype=torch.long,
                device=graph_embedding.device,
            )
            latent = self.head_refinement(
                latent + self.head_embeddings(head_index_tensor)
            )
            typed_active = active.to(latent.dtype).unsqueeze(1)
            head_latents.append(latent * typed_active)
            fusion_weights.append(weights * typed_active)
        latent_tensor = torch.stack(head_latents, dim=1)
        weight_tensor = torch.stack(fusion_weights, dim=1)
        head_available = head_active & effective.any(dim=2)
        head_lookup = {name: index for index, name in enumerate(HEAD_ORDER)}
        distortion_latent = latent_tensor[:, head_lookup["diagnosis"], :]
        action_latent = latent_tensor[:, head_lookup["action_ranking"], :]
        born_latent = latent_tensor[:, head_lookup["born_prediction"], :]
        hilbert_latent = latent_tensor[:, head_lookup["hilbert_deformation"], :]
        uncertainty_latent = latent_tensor[:, head_lookup["uncertainty"], :]
        topology_latent = latent_tensor[:, head_lookup["topology"], :]
        return TriQTOModelOutput(
            model_architecture_id=self.architecture_id,
            graph_embedding=graph_embedding,
            node_embeddings=graph_output.node_embeddings,
            stream_embeddings=stream_embeddings,
            stream_available_mask=stream_available,
            effective_head_stream_mask=effective,
            fusion_weights=weight_tensor,
            head_latents=latent_tensor,
            distortion=self.distortion_head(
                distortion_latent,
                graph_output.node_embeddings,
                batch.graph.node_batch,
                head_available[:, head_lookup["diagnosis"]],
            ),
            action_ranking=self.action_ranking_head(
                action_latent,
                batch.actions,
                head_available[:, head_lookup["action_ranking"]],
            ),
            born_prediction=self.born_prediction_head(
                born_latent,
                batch.born_queries,
                head_available[:, head_lookup["born_prediction"]],
            ),
            hilbert_deformation=self.hilbert_deformation_head(
                hilbert_latent,
                head_available[:, head_lookup["hilbert_deformation"]],
            ),
            uncertainty=self.uncertainty_head(
                uncertainty_latent,
                head_available[:, head_lookup["uncertainty"]],
            ),
            topology=self.topology_head(
                topology_latent,
                head_available[:, head_lookup["topology"]],
            ),
        )


__all__ = ["TriQTOModel"]
