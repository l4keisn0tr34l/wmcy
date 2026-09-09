"""Action-conditioned extension of the permutation-equivariant GraphRSSM."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.cyberwm.graph_rssm import GraphRSSM


class ActionGraphRSSM(GraphRSSM):
    """Condition future rollout on a chosen action type and directed pair.

    Action-pair features use shared edge operations, outgoing/incoming node
    aggregation, and invariant graph pooling. Host relabeling therefore commutes
    with action encoding when the directed pair vector is relabeled consistently.
    """

    def __init__(self, *args: Any, action_type_count: int = 2,
                 action_embedding: int = 32, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_type_count = action_type_count
        self.action_embedding = action_embedding
        self.action_edge_encoder = nn.Sequential(
            nn.Linear(1 + action_type_count, action_embedding), nn.SiLU(),
            nn.Linear(action_embedding, self.edge_hidden, bias=False),
        )
        self.action_node_shift = nn.Sequential(
            nn.Linear(2 * self.edge_hidden, 64), nn.SiLU(), nn.Linear(64, self.node_hidden)
        )
        self.action_global_shift = nn.Sequential(
            nn.Linear(action_type_count + self.edge_hidden, 64), nn.SiLU(),
            nn.Linear(64, self.global_hidden + self.stochastic_size),
        )

    def validate_action(self, action_type: torch.Tensor, action_pair: torch.Tensor,
                        batch: int) -> None:
        if action_type.shape != (batch, self.action_type_count):
            raise ValueError(f"action_type must be {(batch, self.action_type_count)}, got {action_type.shape}")
        if action_pair.shape != (batch, self.pair_count):
            raise ValueError(f"action_pair must be {(batch, self.pair_count)}, got {action_pair.shape}")

    def apply_action(self, global_hidden: torch.Tensor, node_hidden: torch.Tensor,
                     edge_hidden: torch.Tensor, z: torch.Tensor,
                     action_type: torch.Tensor, action_pair: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(global_hidden); self.validate_action(action_type, action_pair, batch)
        selected = action_pair[..., None]
        expanded_type = action_type[:, None, :].expand(-1, self.pair_count, -1)
        action_edge = self.action_edge_encoder(torch.cat([selected, selected * expanded_type], dim=-1))
        # Ensure unselected edges receive exactly zero even if an upstream layer
        # gains a bias in a future refactor.
        action_edge = action_edge * selected
        outgoing, incoming = self.aggregate_edges(action_edge)
        node_shift = self.action_node_shift(torch.cat([outgoing, incoming], dim=-1))
        pooled = action_edge.sum(dim=1)
        global_shift, z_shift = self.action_global_shift(
            torch.cat([action_type, pooled], dim=-1)
        ).split([self.global_hidden, self.stochastic_size], dim=-1)
        return (global_hidden + global_shift, node_hidden + node_shift,
                edge_hidden + action_edge, z + z_shift)

    def forecast_action(self, context: torch.Tensor, action_type: torch.Tensor,
                        action_pair: torch.Tensor, sample: bool = False
                        ) -> dict[str, torch.Tensor]:
        posterior = self.observe(context, sample=sample)
        state = self.apply_action(
            posterior["h"][:, -1], posterior["node_h"][:, -1],
            posterior["edge_h"][:, -1], posterior["z"][:, -1],
            action_type, action_pair,
        )
        return self.imagine(*state, steps=self.horizon, sample=sample)

    def forward_action(self, observations: torch.Tensor, context_steps: int,
                       action_type: torch.Tensor, action_pair: torch.Tensor,
                       sample: bool = True
                       ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        posterior = self.observe(observations, sample=sample)
        index = context_steps - 1
        state = self.apply_action(
            posterior["h"][:, index], posterior["node_h"][:, index],
            posterior["edge_h"][:, index], posterior["z"][:, index],
            action_type, action_pair,
        )
        future = self.imagine(*state, steps=observations.shape[1] - context_steps, sample=sample)
        return posterior, future

    def action_config(self) -> dict[str, Any]:
        return {**self.config(), "action_type_count": self.action_type_count,
                "action_embedding": self.action_embedding}
