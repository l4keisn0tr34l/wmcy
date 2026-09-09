"""Explicit finite-mixture extension of the permutation-equivariant GraphRSSM."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.cyberwm.graph_rssm import GraphRSSM


class BranchingGraphRSSM(GraphRSSM):
    """Context-weighted candidate futures with shared equivariant dynamics.

    Branch identity is global/invariant. Each branch applies the same learned
    shift to every node and every edge slot, so host relabeling commutes with
    branch rollout.
    """

    def __init__(self, *args: Any, branch_count: int = 2,
                 branch_embedding_size: int = 32, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if branch_count < 2:
            raise ValueError("branch_count must be at least two")
        self.branch_count = branch_count
        self.branch_embedding_size = branch_embedding_size
        graph_feature = self.global_hidden + self.stochastic_size + self.node_hidden + self.edge_hidden
        self.branch_gate = nn.Sequential(
            nn.Linear(graph_feature, 64), nn.SiLU(), nn.Linear(64, branch_count)
        )
        self.branch_embedding = nn.Embedding(branch_count, branch_embedding_size)
        self.branch_global_shift = nn.Linear(branch_embedding_size, self.global_hidden, bias=False)
        self.branch_node_shift = nn.Linear(branch_embedding_size, self.node_hidden, bias=False)
        self.branch_edge_shift = nn.Linear(branch_embedding_size, self.edge_hidden, bias=False)
        self.branch_z_shift = nn.Linear(branch_embedding_size, self.stochastic_size, bias=False)

    def _branch_initial_states(
        self, global_hidden: torch.Tensor, node_hidden: torch.Tensor,
        edge_hidden: torch.Tensor, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = len(global_hidden); embedding = self.branch_embedding.weight
        global_value = global_hidden[:, None, :] + self.branch_global_shift(embedding)[None, :, :]
        node_value = node_hidden[:, None, :, :] + self.branch_node_shift(embedding)[None, :, None, :]
        edge_value = edge_hidden[:, None, :, :] + self.branch_edge_shift(embedding)[None, :, None, :]
        z_value = z[:, None, :] + self.branch_z_shift(embedding)[None, :, :]
        return (
            global_value.reshape(batch * self.branch_count, self.global_hidden),
            node_value.reshape(batch * self.branch_count, self.node_count, self.node_hidden),
            edge_value.reshape(batch * self.branch_count, self.pair_count, self.edge_hidden),
            z_value.reshape(batch * self.branch_count, self.stochastic_size),
        )

    def _reshape_branches(self, value: torch.Tensor, batch: int) -> torch.Tensor:
        return value.reshape(batch, self.branch_count, *value.shape[1:])

    def branch_from_posterior(self, posterior: dict[str, torch.Tensor], context_index: int,
                              sample: bool) -> dict[str, torch.Tensor]:
        batch = len(posterior["h"])
        branch_logits = self.branch_gate(posterior["feature"][:, context_index])
        initial = self._branch_initial_states(
            posterior["h"][:, context_index], posterior["node_h"][:, context_index],
            posterior["edge_h"][:, context_index], posterior["z"][:, context_index],
        )
        imagined = self.imagine(*initial, steps=self.horizon, sample=sample)
        output = {name: self._reshape_branches(value, batch) for name, value in imagined.items()}
        output["branch_logits"] = branch_logits
        output["branch_weights"] = torch.softmax(branch_logits, dim=-1)
        return output

    def forward_branches(self, observations: torch.Tensor, context_steps: int,
                         sample: bool = True
                         ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if observations.shape[1] < context_steps:
            raise ValueError("observations shorter than context")
        posterior = self.observe(observations, sample=sample)
        return posterior, self.branch_from_posterior(posterior, context_steps - 1, sample)

    def forecast_branches(self, context: torch.Tensor, sample: bool = False
                          ) -> dict[str, torch.Tensor]:
        posterior = self.observe(context, sample=sample)
        return self.branch_from_posterior(posterior, context.shape[1] - 1, sample)

    def branching_config(self) -> dict[str, Any]:
        return {**self.config(), "branch_count": self.branch_count,
                "branch_embedding_size": self.branch_embedding_size}
