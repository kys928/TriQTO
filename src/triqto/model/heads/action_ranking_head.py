"""Two-stage variable-candidate action head."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from triqto.model.config import TriQTOModelConfig
from triqto.model.contracts import ActionCandidateTensorBatch
from triqto.model.outputs import ActionRankingHeadOutput
from triqto.model.tensor_ops import segment_mean, segment_softmax


class ActionRankingHead(nn.Module):
    """Predict whether to act, then score the available deployable actions."""

    def __init__(self, config: TriQTOModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        edit_dim = hidden // 4
        self.should_act = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.edit_type = nn.Embedding(config.action_edit_type_count, edit_dim)
        self.edit = nn.Sequential(
            nn.Linear(edit_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.candidate = nn.Sequential(
            nn.Linear(config.action_candidate_feature_dim + hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.pair = nn.Sequential(
            nn.Linear(hidden * 2, hidden * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )
        self.score = nn.Linear(hidden, 1)
        self.reward = nn.Linear(hidden, 1)
        self.reset_output_baselines()

    def reset_reward_baseline(self) -> None:
        """Start reward regression at the empirical zero-reward baseline."""
        nn.init.zeros_(self.reward.weight)
        nn.init.zeros_(self.reward.bias)

    def reset_output_baselines(self) -> None:
        """Start gate, ranking, and reward outputs at neutral baselines.

        A zero gate logit means probability 0.5, zero candidate scores produce a
        uniform distribution over eligible candidates, and zero reward is the real
        data's strong baseline. The top-level model calls this method after its generic
        Xavier pass so these policies survive full-model construction.
        """
        gate_output = self.should_act[-1]
        if not isinstance(gate_output, nn.Linear):
            raise TypeError("should-act network must end with a Linear layer")
        nn.init.zeros_(gate_output.weight)
        nn.init.zeros_(gate_output.bias)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.reset_reward_baseline()

    def forward(
        self,
        graph_latent: Tensor,
        actions: ActionCandidateTensorBatch | None,
        graph_active_mask: Tensor,
    ) -> ActionRankingHeadOutput:
        graph_count = graph_latent.shape[0]
        device = graph_latent.device
        if graph_active_mask.dtype != torch.bool or graph_active_mask.shape != (graph_count,):
            raise ValueError("graph_active_mask must be bool with shape [graph_count]")

        should_act_logit = self.should_act(graph_latent).squeeze(1)
        should_act_logit = should_act_logit * graph_active_mask.to(should_act_logit.dtype)
        should_act_probability = torch.sigmoid(should_act_logit)
        should_act_probability = (
            should_act_probability * graph_active_mask.to(should_act_probability.dtype)
        )

        if actions is None or actions.candidate_features.shape[0] == 0:
            empty_float = graph_latent.new_zeros((0,))
            empty_long = torch.zeros(0, dtype=torch.long, device=device)
            empty_bool = torch.zeros(0, dtype=torch.bool, device=device)
            return ActionRankingHeadOutput(
                should_act_logit=should_act_logit,
                should_act_probability=should_act_probability,
                should_act_available_mask=graph_active_mask,
                candidate_scores=empty_float,
                candidate_probabilities=empty_float,
                predicted_rewards=empty_float,
                candidate_batch=empty_long,
                candidate_available_mask=empty_bool,
                graph_available_mask=torch.zeros(
                    graph_count, dtype=torch.bool, device=device
                ),
            )

        candidate_count = actions.candidate_features.shape[0]
        if actions.edit_type_ids.numel():
            edit = self.edit(
                torch.cat(
                    (
                        self.edit_type(actions.edit_type_ids),
                        actions.edit_magnitudes.unsqueeze(1),
                        actions.edit_qubit_positions.unsqueeze(1),
                    ),
                    dim=1,
                )
            )
            edit_context = segment_mean(
                edit,
                actions.edit_candidate_index,
                candidate_count,
            )
        else:
            edit_context = graph_latent.new_zeros(
                (candidate_count, graph_latent.shape[1])
            )
        candidate = self.candidate(
            torch.cat((actions.candidate_features, edit_context), dim=1)
        )
        graph_context = graph_latent.index_select(0, actions.candidate_batch)
        paired = self.pair(torch.cat((candidate, graph_context), dim=1))
        score = self.score(paired).squeeze(1)
        reward = self.reward(paired).squeeze(1)
        candidate_mask = (
            actions.candidate_available_mask
            & graph_active_mask.index_select(0, actions.candidate_batch)
        )
        probabilities = segment_softmax(
            score,
            actions.candidate_batch,
            graph_count,
            candidate_mask,
        )
        typed_mask = candidate_mask.to(score.dtype)
        score = score * typed_mask
        reward = reward * typed_mask
        graph_available = torch.zeros(graph_count, dtype=torch.bool, device=device)
        active_graphs = actions.candidate_batch[candidate_mask].unique()
        if active_graphs.numel():
            graph_available.scatter_(0, active_graphs, True)
        return ActionRankingHeadOutput(
            should_act_logit=should_act_logit,
            should_act_probability=should_act_probability,
            should_act_available_mask=graph_active_mask,
            candidate_scores=score,
            candidate_probabilities=probabilities,
            predicted_rewards=reward,
            candidate_batch=actions.candidate_batch,
            candidate_available_mask=candidate_mask,
            graph_available_mask=graph_available,
        )


__all__ = ["ActionRankingHead"]
