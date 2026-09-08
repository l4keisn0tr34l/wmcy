"""Permutation-equivariant graph RSSM for fixed-size directed host graphs."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


class GraphFeatureScaler(StandardScaler):
    """Z-score globals normally and share statistics across node/edge slots."""


def fit_graph_feature_scaler(context: np.ndarray, global_size: int, node_size: int,
                             node_count: int, edge_size: int, pair_count: int
                             ) -> GraphFeatureScaler:
    if context.ndim != 3:
        raise ValueError(f"expected [sample,time,feature], got {context.shape}")
    expected = global_size + node_count * node_size + pair_count * edge_size
    if context.shape[-1] != expected:
        raise ValueError(f"feature width {context.shape[-1]} != graph layout {expected}")
    flat = context.reshape(-1, expected)
    node_start = global_size
    edge_start = node_start + node_count * node_size
    global_values = flat[:, :global_size]
    node_values = flat[:, node_start:edge_start].reshape(-1, node_size)
    edge_values = flat[:, edge_start:].reshape(-1, edge_size)

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        return mean, scale

    global_mean, global_scale = moments(global_values)
    node_mean, node_scale = moments(node_values)
    edge_mean, edge_scale = moments(edge_values)
    scaler = GraphFeatureScaler()
    scaler.mean_ = np.concatenate([
        global_mean, np.tile(node_mean, node_count), np.tile(edge_mean, pair_count)
    ])
    scaler.scale_ = np.concatenate([
        global_scale, np.tile(node_scale, node_count), np.tile(edge_scale, pair_count)
    ])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = expected
    scaler.n_samples_seen_ = len(flat)
    return scaler


def _stats(parameters: torch.Tensor, stochastic_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean, raw_std = parameters.split(stochastic_size, dim=-1)
    return mean, torch.nn.functional.softplus(raw_std) + 0.1


def _sample(mean: torch.Tensor, std: torch.Tensor, sample: bool) -> torch.Tensor:
    return mean + std * torch.randn_like(std) if sample else mean


class GraphRSSM(nn.Module):
    """Structured recurrent stochastic state with shared node/edge operations."""

    def __init__(self, global_size: int = 15, node_size: int = 18, node_count: int = 3,
                 edge_size: int = 12, pair_count: int = 6, horizon: int = 6,
                 global_hidden: int = 64, node_hidden: int = 32, edge_hidden: int = 32,
                 stochastic_size: int = 16, local_embedding: int = 32,
                 technique_count: int = 3, semantic_from_decoded: bool = False) -> None:
        super().__init__()
        if pair_count != node_count * (node_count - 1):
            raise ValueError("pair_count must represent every directed non-self node pair")
        self.global_size = global_size; self.node_size = node_size; self.node_count = node_count
        self.edge_size = edge_size; self.pair_count = pair_count; self.horizon = horizon
        self.global_hidden = global_hidden; self.node_hidden = node_hidden
        self.edge_hidden = edge_hidden; self.stochastic_size = stochastic_size
        self.semantic_from_decoded = semantic_from_decoded
        self.observation_size = global_size + node_count * node_size + pair_count * edge_size

        sources, destinations = [], []
        for source in range(node_count):
            for destination in range(node_count):
                if source != destination:
                    sources.append(source); destinations.append(destination)
        self.register_buffer("pair_sources", torch.tensor(sources), persistent=False)
        self.register_buffer("pair_destinations", torch.tensor(destinations), persistent=False)

        self.global_encoder = nn.Sequential(nn.Linear(global_size, local_embedding), nn.SiLU(),
                                            nn.Linear(local_embedding, local_embedding), nn.SiLU())
        self.node_encoder = nn.Sequential(nn.Linear(node_size, local_embedding), nn.SiLU(),
                                          nn.Linear(local_embedding, local_embedding), nn.SiLU())
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_size + 2 * local_embedding, 64), nn.SiLU(),
            nn.Linear(64, local_embedding), nn.SiLU(),
        )
        self.node_message = nn.Sequential(
            nn.Linear(4 * local_embedding, 64), nn.SiLU(),
            nn.Linear(64, local_embedding), nn.SiLU(),
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(3 * local_embedding, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )

        transition_width = stochastic_size + node_hidden + edge_hidden
        self.global_transition = nn.GRUCell(transition_width, global_hidden)
        self.global_observation = nn.GRUCell(64, global_hidden)
        self.prior = nn.Sequential(nn.Linear(global_hidden, 64), nn.SiLU(),
                                   nn.Linear(64, 2 * stochastic_size))
        self.posterior = nn.Sequential(nn.Linear(global_hidden + 64, 64), nn.SiLU(),
                                       nn.Linear(64, 2 * stochastic_size))
        self.node_transition = nn.GRUCell(
            global_hidden + stochastic_size + 2 * edge_hidden, node_hidden
        )
        self.edge_transition = nn.GRUCell(
            global_hidden + stochastic_size + 2 * node_hidden, edge_hidden
        )
        self.node_observation = nn.GRUCell(
            local_embedding + global_hidden + stochastic_size, node_hidden
        )
        self.edge_observation = nn.GRUCell(
            local_embedding + 2 * node_hidden + global_hidden + stochastic_size, edge_hidden
        )

        graph_feature = global_hidden + stochastic_size + node_hidden + edge_hidden
        node_decode = node_hidden + global_hidden + stochastic_size + 2 * edge_hidden
        edge_decode = edge_hidden + 2 * node_hidden + global_hidden + stochastic_size
        self.global_decoder = nn.Sequential(nn.Linear(graph_feature, 96), nn.SiLU(),
                                            nn.Linear(96, global_size))
        self.node_decoder = nn.Sequential(nn.Linear(node_decode, 96), nn.SiLU(),
                                          nn.Linear(96, node_size))
        self.edge_decoder = nn.Sequential(nn.Linear(edge_decode, 96), nn.SiLU(),
                                          nn.Linear(96, edge_size))
        self.edge_presence_head = nn.Linear(edge_decode, 1)
        self.pair_embedding = nn.Sequential(nn.Linear(edge_decode, 64), nn.SiLU(),
                                            nn.Linear(64, 32), nn.SiLU())
        self.pair_head = nn.Sequential(nn.Linear(horizon * 32, 64), nn.SiLU(), nn.Linear(64, 1))
        semantic_per_step = (
            global_size + 2 * node_size + 2 * edge_size
            if semantic_from_decoded else graph_feature
        )
        horizon_semantic = horizon * semantic_per_step
        self.lm_head = nn.Sequential(
            nn.Linear(horizon_semantic, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.technique_head = nn.Sequential(
            nn.Linear(horizon_semantic, 64), nn.SiLU(), nn.Linear(64, technique_count)
        )

    def split_observation(self, observation: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_start = self.global_size + self.node_count * self.node_size
        global_value = observation[..., :self.global_size]
        node_value = observation[..., self.global_size:edge_start].reshape(
            *observation.shape[:-1], self.node_count, self.node_size
        )
        edge_value = observation[..., edge_start:].reshape(
            *observation.shape[:-1], self.pair_count, self.edge_size
        )
        return global_value, node_value, edge_value

    def aggregate_edges(self, edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*edges.shape[:-2], self.node_count, edges.shape[-1])
        outgoing = edges.new_zeros(shape); incoming = edges.new_zeros(shape)
        outgoing.index_add_(-2, self.pair_sources, edges)
        incoming.index_add_(-2, self.pair_destinations, edges)
        divisor = float(self.node_count - 1)
        return outgoing / divisor, incoming / divisor

    def encode_graph(self, observation: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_value, node_value, edge_value = self.split_observation(observation)
        global_encoded = self.global_encoder(global_value)
        node_encoded = self.node_encoder(node_value)
        edge_encoded = self.edge_encoder(torch.cat([
            edge_value, node_encoded[:, self.pair_sources], node_encoded[:, self.pair_destinations]
        ], dim=-1))
        outgoing, incoming = self.aggregate_edges(edge_encoded)
        global_per_node = global_encoded[:, None, :].expand(-1, self.node_count, -1)
        node_message = self.node_message(torch.cat(
            [node_encoded, outgoing, incoming, global_per_node], dim=-1
        ))
        graph = self.graph_encoder(torch.cat([
            global_encoded, node_message.mean(dim=1), edge_encoded.mean(dim=1)
        ], dim=-1))
        return graph, node_message, edge_encoded

    def initial(self, batch_size: int, device: torch.device
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.global_hidden, device=device),
            torch.zeros(batch_size, self.node_count, self.node_hidden, device=device),
            torch.zeros(batch_size, self.pair_count, self.edge_hidden, device=device),
            torch.zeros(batch_size, self.stochastic_size, device=device),
        )

    def transition(self, global_hidden: torch.Tensor, node_hidden: torch.Tensor,
                   edge_hidden: torch.Tensor, z: torch.Tensor, sample: bool
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor]:
        global_input = torch.cat(
            [z, node_hidden.mean(dim=1), edge_hidden.mean(dim=1)], dim=-1
        )
        global_hidden = self.global_transition(global_input, global_hidden)
        prior_mean, prior_std = _stats(self.prior(global_hidden), self.stochastic_size)
        z = _sample(prior_mean, prior_std, sample)
        outgoing, incoming = self.aggregate_edges(edge_hidden)
        expanded_global = global_hidden[:, None, :].expand(-1, self.node_count, -1)
        expanded_z = z[:, None, :].expand(-1, self.node_count, -1)
        node_input = torch.cat([expanded_global, expanded_z, outgoing, incoming], dim=-1)
        node_hidden = self.node_transition(
            node_input.reshape(-1, node_input.shape[-1]),
            node_hidden.reshape(-1, self.node_hidden),
        ).reshape(-1, self.node_count, self.node_hidden)
        edge_global = global_hidden[:, None, :].expand(-1, self.pair_count, -1)
        edge_z = z[:, None, :].expand(-1, self.pair_count, -1)
        edge_input = torch.cat([
            edge_global, edge_z, node_hidden[:, self.pair_sources],
            node_hidden[:, self.pair_destinations],
        ], dim=-1)
        edge_hidden = self.edge_transition(
            edge_input.reshape(-1, edge_input.shape[-1]),
            edge_hidden.reshape(-1, self.edge_hidden),
        ).reshape(-1, self.pair_count, self.edge_hidden)
        return global_hidden, node_hidden, edge_hidden, z, prior_mean, prior_std

    def incorporate_observation(self, global_hidden: torch.Tensor, node_hidden: torch.Tensor,
                                edge_hidden: torch.Tensor, z: torch.Tensor,
                                graph: torch.Tensor, node_message: torch.Tensor,
                                edge_encoded: torch.Tensor
                                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_hidden = self.global_observation(graph, global_hidden)
        node_global = global_hidden[:, None, :].expand(-1, self.node_count, -1)
        node_z = z[:, None, :].expand(-1, self.node_count, -1)
        node_input = torch.cat([node_message, node_global, node_z], dim=-1)
        node_hidden = self.node_observation(
            node_input.reshape(-1, node_input.shape[-1]),
            node_hidden.reshape(-1, self.node_hidden),
        ).reshape(-1, self.node_count, self.node_hidden)
        edge_global = global_hidden[:, None, :].expand(-1, self.pair_count, -1)
        edge_z = z[:, None, :].expand(-1, self.pair_count, -1)
        edge_input = torch.cat([
            edge_encoded, node_hidden[:, self.pair_sources], node_hidden[:, self.pair_destinations],
            edge_global, edge_z,
        ], dim=-1)
        edge_hidden = self.edge_observation(
            edge_input.reshape(-1, edge_input.shape[-1]),
            edge_hidden.reshape(-1, self.edge_hidden),
        ).reshape(-1, self.pair_count, self.edge_hidden)
        return global_hidden, node_hidden, edge_hidden

    def structured_features(self, global_hidden: torch.Tensor, node_hidden: torch.Tensor,
                            edge_hidden: torch.Tensor, z: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outgoing, incoming = self.aggregate_edges(edge_hidden)
        graph = torch.cat([
            global_hidden, z, node_hidden.mean(dim=1), edge_hidden.mean(dim=1)
        ], dim=-1)
        node_global = global_hidden[:, None, :].expand(-1, self.node_count, -1)
        node_z = z[:, None, :].expand(-1, self.node_count, -1)
        node = torch.cat([node_hidden, node_global, node_z, outgoing, incoming], dim=-1)
        edge_global = global_hidden[:, None, :].expand(-1, self.pair_count, -1)
        edge_z = z[:, None, :].expand(-1, self.pair_count, -1)
        edge = torch.cat([
            edge_hidden, node_hidden[:, self.pair_sources], node_hidden[:, self.pair_destinations],
            edge_global, edge_z,
        ], dim=-1)
        return graph, node, edge

    def decode(self, global_hidden: torch.Tensor, node_hidden: torch.Tensor,
               edge_hidden: torch.Tensor, z: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graph, node, edge = self.structured_features(global_hidden, node_hidden, edge_hidden, z)
        decoded = torch.cat([
            self.global_decoder(graph), self.node_decoder(node).reshape(len(graph), -1),
            self.edge_decoder(edge).reshape(len(graph), -1),
        ], dim=-1)
        return decoded, graph, edge, self.edge_presence_head(edge).squeeze(-1)

    def observe(self, observations: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        global_hidden, node_hidden, edge_hidden, z = self.initial(len(observations), observations.device)
        records: dict[str, list[torch.Tensor]] = {name: [] for name in [
            "h", "node_h", "edge_h", "z", "prior_mean", "prior_std", "post_mean",
            "post_std", "feature", "decoded"
        ]}
        for time_index in range(observations.shape[1]):
            global_hidden, node_hidden, edge_hidden, _, prior_mean, prior_std = self.transition(
                global_hidden, node_hidden, edge_hidden, z, sample=False
            )
            graph_observation, node_observation, edge_observation = self.encode_graph(
                observations[:, time_index]
            )
            post_mean, post_std = _stats(
                self.posterior(torch.cat([global_hidden, graph_observation], dim=-1)),
                self.stochastic_size,
            )
            z = _sample(post_mean, post_std, sample)
            global_hidden, node_hidden, edge_hidden = self.incorporate_observation(
                global_hidden, node_hidden, edge_hidden, z, graph_observation,
                node_observation, edge_observation,
            )
            decoded, graph_feature, _, _ = self.decode(global_hidden, node_hidden, edge_hidden, z)
            for name, value in [
                ("h", global_hidden), ("node_h", node_hidden), ("edge_h", edge_hidden),
                ("z", z), ("prior_mean", prior_mean), ("prior_std", prior_std),
                ("post_mean", post_mean), ("post_std", post_std),
                ("feature", graph_feature), ("decoded", decoded),
            ]:
                records[name].append(value)
        return {name: torch.stack(values, dim=1) for name, values in records.items()}

    def imagine(self, global_hidden: torch.Tensor, node_hidden: torch.Tensor,
                edge_hidden: torch.Tensor, z: torch.Tensor, steps: int,
                sample: bool = True) -> dict[str, torch.Tensor]:
        records: dict[str, list[torch.Tensor]] = {name: [] for name in [
            "h", "node_h", "edge_h", "z", "mean", "std", "feature", "edge_feature",
            "decoded", "edge_logits", "pair_embedding"
        ]}
        for _ in range(steps):
            global_hidden, node_hidden, edge_hidden, z, mean, std = self.transition(
                global_hidden, node_hidden, edge_hidden, z, sample=sample
            )
            decoded, graph_feature, edge_feature, edge_logits = self.decode(
                global_hidden, node_hidden, edge_hidden, z
            )
            pair_embedding = self.pair_embedding(edge_feature)
            for name, value in [
                ("h", global_hidden), ("node_h", node_hidden), ("edge_h", edge_hidden),
                ("z", z), ("mean", mean), ("std", std), ("feature", graph_feature),
                ("edge_feature", edge_feature), ("decoded", decoded),
                ("edge_logits", edge_logits), ("pair_embedding", pair_embedding),
            ]:
                records[name].append(value)
        result = {name: torch.stack(values, dim=1) for name, values in records.items()}
        if self.semantic_from_decoded:
            global_value, node_value, edge_value = self.split_observation(result["decoded"])
            semantic = torch.cat([
                global_value, node_value.mean(dim=2), node_value.amax(dim=2),
                edge_value.mean(dim=2), edge_value.amax(dim=2),
            ], dim=-1).reshape(len(global_hidden), -1)
        else:
            semantic = result["feature"].reshape(len(global_hidden), -1)
        pair_flat = result["pair_embedding"].permute(0, 2, 1, 3).reshape(
            len(global_hidden), self.pair_count, -1
        )
        result["lm_logits"] = self.lm_head(semantic).squeeze(-1)
        result["technique_logits"] = self.technique_head(semantic)
        result["pair_logits"] = self.pair_head(pair_flat).squeeze(-1)
        return result

    def forward(self, observations: torch.Tensor, context_steps: int,
                sample: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        posterior = self.observe(observations, sample=sample)
        index = context_steps - 1
        future = self.imagine(
            posterior["h"][:, index], posterior["node_h"][:, index],
            posterior["edge_h"][:, index], posterior["z"][:, index],
            observations.shape[1] - context_steps, sample=sample,
        )
        return posterior, future

    def forecast(self, context: torch.Tensor, sample: bool = False) -> dict[str, torch.Tensor]:
        posterior = self.observe(context, sample=sample)
        return self.imagine(
            posterior["h"][:, -1], posterior["node_h"][:, -1],
            posterior["edge_h"][:, -1], posterior["z"][:, -1],
            self.horizon, sample=sample,
        )

    def config(self) -> dict[str, Any]:
        return {
            "global_size": self.global_size, "node_size": self.node_size,
            "node_count": self.node_count, "edge_size": self.edge_size,
            "pair_count": self.pair_count, "horizon": self.horizon,
            "global_hidden": self.global_hidden, "node_hidden": self.node_hidden,
            "edge_hidden": self.edge_hidden, "stochastic_size": self.stochastic_size,
            "semantic_from_decoded": self.semantic_from_decoded,
        }
