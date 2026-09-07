#!/usr/bin/env python3
"""Train and evaluate a compact passive RSSM on chronological graph states.

The model observes three past/current states, infers a deterministic plus
stochastic latent state, and rolls its prior forward for six future states.
Security heads consume only the imagined future latent trajectory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import permutations
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(directory: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(directory / f"{split}.npz") as loaded:
        return {name: loaded[name].astype(np.float32, copy=True) for name in loaded.files}


def state_and_edge_permutation_indices(metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return all six consistent host relabelings as new-slot-to-old-slot indices."""
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"])
    edge_width = len(metadata["edge_feature_names"])
    node_count = len(metadata["node_slots"])
    edge_start = global_width + node_count * node_width
    slot_ips = [row["ip"] for row in metadata["node_slots"]]
    ip_to_slot = {ip: slot for slot, ip in enumerate(slot_ips)}
    old_pair_to_slot = {
        (ip_to_slot[row["source_ip"]], ip_to_slot[row["destination_ip"]]): row["slot"]
        for row in metadata["directed_edge_slots"]
    }
    state_indices: list[np.ndarray] = []
    edge_indices: list[np.ndarray] = []
    width = metadata["state_feature_count"]
    for order in permutations(range(node_count)):
        indices = np.arange(width)
        for new_slot, old_slot in enumerate(order):
            new_start = global_width + new_slot * node_width
            old_start = global_width + old_slot * node_width
            indices[new_start:new_start + node_width] = np.arange(old_start, old_start + node_width)
        pair_order: list[int] = []
        new_edge_slot = 0
        for new_source in range(node_count):
            for new_destination in range(node_count):
                if new_source == new_destination:
                    continue
                old_edge_slot = old_pair_to_slot[(order[new_source], order[new_destination])]
                new_start = edge_start + new_edge_slot * edge_width
                old_start = edge_start + old_edge_slot * edge_width
                indices[new_start:new_start + edge_width] = np.arange(old_start, old_start + edge_width)
                pair_order.append(old_edge_slot)
                new_edge_slot += 1
        state_indices.append(indices)
        edge_indices.append(np.asarray(pair_order))
    return np.stack(state_indices), np.stack(edge_indices)


def fit_context_scaler(train_context: np.ndarray) -> StandardScaler:
    """Match the Ridge contract: fit only on original observable train contexts."""
    return StandardScaler().fit(train_context.reshape(-1, train_context.shape[-1]))


def normalize(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    return scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape).astype(np.float32)


def stats(parameters: torch.Tensor, stochastic_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean, raw_std = parameters.split(stochastic_size, dim=-1)
    std = F.softplus(raw_std) + 0.1
    return mean, std


def sample_normal(mean: torch.Tensor, std: torch.Tensor, sample: bool) -> torch.Tensor:
    return mean + std * torch.randn_like(std) if sample else mean


class CompactRSSM(nn.Module):
    def __init__(self, observation_size: int = 141, embedding_size: int = 64,
                 deterministic_size: int = 64, stochastic_size: int = 16,
                 horizon: int = 6, technique_count: int = 3, pair_count: int = 6) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size
        self.horizon = horizon
        feature_size = deterministic_size + stochastic_size
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, embedding_size), nn.SiLU(),
            nn.Linear(embedding_size, embedding_size), nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(stochastic_size, deterministic_size)
        self.prior = nn.Sequential(nn.Linear(deterministic_size, 64), nn.SiLU(),
                                   nn.Linear(64, 2 * stochastic_size))
        self.posterior = nn.Sequential(
            nn.Linear(deterministic_size + embedding_size, 64), nn.SiLU(),
            nn.Linear(64, 2 * stochastic_size),
        )
        self.decoder = nn.Sequential(nn.Linear(feature_size, 128), nn.SiLU(),
                                     nn.Linear(128, observation_size))
        self.edge_head = nn.Linear(feature_size, pair_count)
        horizon_width = horizon * feature_size
        self.lm_head = nn.Linear(horizon_width, 1)
        self.technique_head = nn.Linear(horizon_width, technique_count)
        self.pair_head = nn.Linear(horizon_width, pair_count)

    def initial(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deterministic_size, device=device)
        z = torch.zeros(batch_size, self.stochastic_size, device=device)
        return h, z

    def observe(self, observations: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        h, z = self.initial(len(observations), observations.device)
        hs, zs, prior_means, prior_stds, post_means, post_stds, features, decoded = ([] for _ in range(8))
        for time_index in range(observations.shape[1]):
            h = self.recurrent(z, h)
            prior_mean, prior_std = stats(self.prior(h), self.stochastic_size)
            embedding = self.encoder(observations[:, time_index])
            post_mean, post_std = stats(
                self.posterior(torch.cat([h, embedding], dim=-1)), self.stochastic_size
            )
            z = sample_normal(post_mean, post_std, sample)
            feature = torch.cat([h, z], dim=-1)
            hs.append(h); zs.append(z)
            prior_means.append(prior_mean); prior_stds.append(prior_std)
            post_means.append(post_mean); post_stds.append(post_std)
            features.append(feature); decoded.append(self.decoder(feature))
        return {
            "h": torch.stack(hs, dim=1), "z": torch.stack(zs, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_std": torch.stack(prior_stds, dim=1),
            "post_mean": torch.stack(post_means, dim=1),
            "post_std": torch.stack(post_stds, dim=1),
            "feature": torch.stack(features, dim=1),
            "decoded": torch.stack(decoded, dim=1),
        }

    def imagine(self, h: torch.Tensor, z: torch.Tensor, steps: int,
                sample: bool = True) -> dict[str, torch.Tensor]:
        hs, zs, means, stds, features, decoded, edge_logits = ([] for _ in range(7))
        for _ in range(steps):
            h = self.recurrent(z, h)
            mean, std = stats(self.prior(h), self.stochastic_size)
            z = sample_normal(mean, std, sample)
            feature = torch.cat([h, z], dim=-1)
            hs.append(h); zs.append(z); means.append(mean); stds.append(std)
            features.append(feature); decoded.append(self.decoder(feature))
            edge_logits.append(self.edge_head(feature))
        feature_sequence = torch.stack(features, dim=1)
        flat = feature_sequence.reshape(len(feature_sequence), -1)
        return {
            "h": torch.stack(hs, dim=1), "z": torch.stack(zs, dim=1),
            "mean": torch.stack(means, dim=1), "std": torch.stack(stds, dim=1),
            "feature": feature_sequence, "decoded": torch.stack(decoded, dim=1),
            "edge_logits": torch.stack(edge_logits, dim=1),
            "lm_logits": self.lm_head(flat).squeeze(-1),
            "technique_logits": self.technique_head(flat),
            "pair_logits": self.pair_head(flat),
        }

    def forward(self, observations: torch.Tensor, context_steps: int,
                sample: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        posterior = self.observe(observations, sample=sample)
        future = self.imagine(
            posterior["h"][:, context_steps - 1], posterior["z"][:, context_steps - 1],
            observations.shape[1] - context_steps, sample=sample,
        )
        return posterior, future

    def forecast(self, context: torch.Tensor, sample: bool = False) -> dict[str, torch.Tensor]:
        posterior = self.observe(context, sample=sample)
        return self.imagine(posterior["h"][:, -1], posterior["z"][:, -1],
                            self.horizon, sample=sample)


def grouped_mse(predicted: torch.Tensor, target: torch.Tensor,
                group_slices: list[slice]) -> torch.Tensor:
    return torch.stack([F.mse_loss(predicted[..., group], target[..., group])
                        for group in group_slices]).mean()


def gaussian_kl(post_mean: torch.Tensor, post_std: torch.Tensor,
                prior_mean: torch.Tensor, prior_std: torch.Tensor) -> torch.Tensor:
    variance_ratio = (post_std.square() + (post_mean - prior_mean).square()) / prior_std.square()
    return (torch.log(prior_std / post_std) + 0.5 * variance_ratio - 0.5).sum(dim=-1)


def positive_weight(target: np.ndarray, maximum: float = 20.0) -> torch.Tensor:
    flattened = target.reshape(-1, target.shape[-1]) if target.ndim > 1 else target.reshape(-1, 1)
    positives = flattened.sum(axis=0)
    negatives = len(flattened) - positives
    weight = np.divide(negatives, positives, out=np.ones_like(negatives), where=positives > 0)
    return torch.tensor(np.clip(weight, 1.0, maximum), dtype=torch.float32)


def augment_batch(batch: tuple[torch.Tensor, ...], state_indices: torch.Tensor,
                  edge_indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    context, future, future_edges, lm, techniques, lm_pairs = batch
    choices = torch.randint(len(state_indices), (len(context),), device=context.device)
    context_out, future_out = context.clone(), future.clone()
    edge_out, pair_out = future_edges.clone(), lm_pairs.clone()
    for choice in choices.unique():
        mask = choices == choice
        state_order = state_indices[choice]
        pair_order = edge_indices[choice]
        context_out[mask] = context[mask][..., state_order]
        future_out[mask] = future[mask][..., state_order]
        edge_out[mask] = future_edges[mask][..., pair_order]
        pair_out[mask] = lm_pairs[mask][..., pair_order]
    return context_out, future_out, edge_out, lm, techniques, pair_out


def make_loader(data: dict[str, np.ndarray], scaler: StandardScaler,
                batch_size: int, shuffle: bool) -> DataLoader:
    context = normalize(scaler, data["context_states"])
    future = normalize(scaler, data["future_states"])
    tensors = [
        torch.from_numpy(context), torch.from_numpy(future),
        torch.from_numpy(data["future_edge_presence"]),
        torch.from_numpy(data["lateral_movement_within_horizon"]),
        torch.from_numpy(data["future_techniques"]),
        torch.from_numpy(data["future_lateral_edges"]),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def loss_terms(model: CompactRSSM, batch: tuple[torch.Tensor, ...], context_steps: int,
               group_slices: list[slice], weights: dict[str, float],
               pos_weights: dict[str, torch.Tensor], sample: bool) -> dict[str, torch.Tensor]:
    context, future, future_edges, lm, techniques, lm_pairs = batch
    full = torch.cat([context, future], dim=1)
    posterior, imagined = model(full, context_steps=context_steps, sample=sample)
    technique_target = techniques.max(dim=1).values
    pair_target = lm_pairs.max(dim=1).values
    terms = {
        "future_state": grouped_mse(imagined["decoded"], future, group_slices),
        "reconstruction": grouped_mse(posterior["decoded"], full, group_slices),
        "edge": F.binary_cross_entropy_with_logits(
            imagined["edge_logits"], future_edges, pos_weight=pos_weights["edge"]
        ),
        "kl": gaussian_kl(
            posterior["post_mean"], posterior["post_std"],
            posterior["prior_mean"], posterior["prior_std"],
        ).clamp_min(weights["free_nats"]).mean(),
        "lm": F.binary_cross_entropy_with_logits(
            imagined["lm_logits"], lm, pos_weight=pos_weights["lm"]
        ),
        "technique": F.binary_cross_entropy_with_logits(
            imagined["technique_logits"], technique_target,
            pos_weight=pos_weights["technique"],
        ),
        "pair": F.binary_cross_entropy_with_logits(
            imagined["pair_logits"], pair_target, pos_weight=pos_weights["pair"]
        ),
    }
    terms["total"] = (
        weights["future_state"] * terms["future_state"]
        + weights["reconstruction"] * terms["reconstruction"]
        + weights["edge"] * terms["edge"]
        + weights["kl"] * terms["kl"]
        + weights["lm"] * terms["lm"]
        + weights["technique"] * terms["technique"]
        + weights["pair"] * terms["pair"]
    )
    return terms


def move(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in batch)


def validate_objective(model: CompactRSSM, loader: DataLoader, context_steps: int,
                       group_slices: list[slice], weights: dict[str, float],
                       pos_weights: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move(raw_batch, device)
            terms = loss_terms(model, batch, context_steps, group_slices, weights, pos_weights, sample=False)
            size = len(batch[0]); count += size
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.0) + float(value) * size
    averaged = {name: value / count for name, value in totals.items()}
    averaged["selection"] = averaged["future_state"] + weights["edge"] * averaged["edge"]
    return averaged


def smoke_tests(model: CompactRSSM, loader: DataLoader, context_steps: int,
                group_slices: list[slice], weights: dict[str, float],
                pos_weights: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    raw = next(iter(loader))
    batch = move(tuple(tensor[:16] for tensor in raw), device)
    full = torch.cat([batch[0], batch[1]], dim=1)
    model.eval()
    with torch.no_grad():
        posterior_a, forecast_a = model(full, context_steps, sample=False)
        changed = full.clone(); changed[:, context_steps:] += 123.0
        posterior_b, _ = model(changed, context_steps, sample=False)
        causal_delta = float((posterior_a["feature"][:, :context_steps]
                              - posterior_b["feature"][:, :context_steps]).abs().max())
        if causal_delta != 0.0:
            raise AssertionError(f"future observations changed context posterior: {causal_delta}")
        if forecast_a["decoded"].shape != (len(full), full.shape[1] - context_steps, model.observation_size):
            raise AssertionError("unexpected RSSM forecast shape")

    smoke_model = deepcopy(model)
    optimizer = torch.optim.Adam(smoke_model.parameters(), lr=1e-3)
    losses = []
    smoke_model.train()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(smoke_model, batch, context_steps, group_slices, weights, pos_weights, sample=True)
        if not torch.isfinite(terms["total"]):
            raise AssertionError("non-finite smoke loss")
        terms["total"].backward()
        nn.utils.clip_grad_norm_(smoke_model.parameters(), 10.0)
        optimizer.step(); losses.append(float(terms["total"].detach()))
    first = float(np.mean(losses[:5])); last = float(np.mean(losses[-5:]))
    if last >= first:
        raise AssertionError(f"tiny-overfit smoke loss did not decrease: {first} -> {last}")
    return {"causal_context_max_delta": causal_delta, "overfit_first_loss": first,
            "overfit_last_loss": last}


def train_seed(seed: int, train_loader: DataLoader, validation_loader: DataLoader,
               model_config: dict[str, int], context_steps: int, group_slices: list[slice],
               weights: dict[str, float], pos_weights: dict[str, torch.Tensor],
               state_indices: torch.Tensor, edge_indices: torch.Tensor,
               device: torch.device, max_epochs: int, patience: int, learning_rate: float
               ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    seed_everything(seed)
    model = CompactRSSM(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train(); train_total = 0.0; count = 0
        for raw_batch in train_loader:
            batch = augment_batch(move(raw_batch, device), state_indices, edge_indices)
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, batch, context_steps, group_slices, weights, pos_weights, sample=True)
            terms["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_total += float(terms["total"].detach()) * len(batch[0]); count += len(batch[0])
        validation = validate_objective(
            model, validation_loader, context_steps, group_slices, weights, pos_weights, device
        )
        history.append({"epoch": epoch, "train_total": train_total / count, **validation})
        if validation["selection"] < best_validation - 1e-5:
            best_validation = validation["selection"]
            best_epoch = epoch; best_state = deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"seed={seed} epoch={epoch} train={train_total/count:.4f} val_selection={validation['selection']:.4f}")
        if stale >= patience:
            break
    return best_state, {"seed": seed, "best_epoch": best_epoch,
                        "best_validation_selection": best_validation,
                        "epochs_run": len(history), "history": history}


def mc_predictions(model: CompactRSSM, context: np.ndarray, scaler: StandardScaler,
                   device: torch.device, samples: int, batch_size: int,
                   seed: int) -> dict[str, np.ndarray]:
    normalized = normalize(scaler, context)
    loader = DataLoader(torch.from_numpy(normalized), batch_size=batch_size, shuffle=False)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in ["state", "edge", "lm", "technique", "pair"]}
    model.eval(); seed_everything(seed)
    with torch.no_grad():
        for _ in range(samples):
            per_draw: dict[str, list[np.ndarray]] = {name: [] for name in collected}
            for batch in loader:
                imagined = model.forecast(batch.to(device), sample=True)
                per_draw["state"].append(imagined["decoded"].cpu().numpy())
                per_draw["edge"].append(torch.sigmoid(imagined["edge_logits"]).cpu().numpy())
                per_draw["lm"].append(torch.sigmoid(imagined["lm_logits"]).cpu().numpy())
                per_draw["technique"].append(torch.sigmoid(imagined["technique_logits"]).cpu().numpy())
                per_draw["pair"].append(torch.sigmoid(imagined["pair_logits"]).cpu().numpy())
            for name in collected:
                collected[name].append(np.concatenate(per_draw[name], axis=0))
    return {name: np.stack(draws, axis=0) for name, draws in collected.items()}


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0, 1, 201), scores]))
    f1 = np.asarray([f1_score(labels, scores >= candidate, zero_division=0) for candidate in candidates])
    return float(candidates[np.flatnonzero(f1 == f1.max())[-1]])


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = labels.astype(int); prediction = scores >= threshold
    result: dict[str, Any] = {
        "count": len(labels), "positive_count": int(labels.sum()), "threshold": threshold,
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
        "brier": float(np.mean((scores - labels) ** 2)),
    }
    if len(np.unique(labels)) == 2:
        result["average_precision"] = float(average_precision_score(labels, scores))
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    return result


def state_metrics(predicted: np.ndarray, target: np.ndarray, context: np.ndarray,
                  raw_future: np.ndarray, group_slices: dict[str, slice]) -> dict[str, Any]:
    persistence = np.repeat(context[:, -1:, :], target.shape[1], axis=1)
    absolute = np.abs(predicted - target); persistence_absolute = np.abs(persistence - target)
    result: dict[str, Any] = {
        "normalized_mae": float(absolute.mean()),
        "persistence_normalized_mae": float(persistence_absolute.mean()),
        "normalized_mae_by_horizon": [float(absolute[:, h].mean()) for h in range(target.shape[1])],
        "persistence_mae_by_horizon": [float(persistence_absolute[:, h].mean()) for h in range(target.shape[1])],
    }
    active = raw_future[:, :, 0] > 0
    for name, mask in [("active", active), ("quiet", ~active)]:
        result[f"{name}_future_state_count"] = int(mask.sum())
        result[f"{name}_normalized_mae"] = float(absolute[mask].mean()) if mask.any() else None
        result[f"{name}_persistence_normalized_mae"] = (
            float(persistence_absolute[mask].mean()) if mask.any() else None
        )
        result[f"{name}_normalized_mae_by_horizon"] = [
            float(absolute[:, h][mask[:, h]].mean()) if mask[:, h].any() else None
            for h in range(target.shape[1])
        ]
        result[f"{name}_persistence_normalized_mae_by_horizon"] = [
            float(persistence_absolute[:, h][mask[:, h]].mean()) if mask[:, h].any() else None
            for h in range(target.shape[1])
        ]
    for name, group in group_slices.items():
        result[f"{name}_normalized_mae"] = float(absolute[..., group].mean())
        result[f"{name}_persistence_normalized_mae"] = float(persistence_absolute[..., group].mean())
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--model-out", default=str(ROOT / "models/mvp_v2_rssm.pt"))
    ap.add_argument("--metrics-out", default=str(ROOT / "outputs/mvp_v2/rssm/metrics.json"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--mc-samples", type=int, default=20)
    args = ap.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cpu")
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    if metadata["state_feature_names"][0] != "flow_count":
        raise ValueError("active-state metric requires flow_count at feature index zero")
    data = {split: load_split(sequence_dir, split) for split in ["train", "validation", "test"]}
    context_steps = int(metadata["context_states"]); horizon = int(metadata["future_horizon_states"])
    state_permutations, edge_permutations = state_and_edge_permutation_indices(metadata)
    scaler = fit_context_scaler(data["train"]["context_states"])

    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    total_width = metadata["state_feature_count"]
    group_slices_list = [slice(0, global_width), slice(global_width, global_width + node_width),
                         slice(global_width + node_width, total_width)]
    group_slices = {"global": group_slices_list[0], "node": group_slices_list[1],
                    "edge": group_slices_list[2]}
    model_config = {
        "observation_size": total_width, "embedding_size": 64,
        "deterministic_size": 64, "stochastic_size": 16, "horizon": horizon,
        "technique_count": len(metadata["technique_targets"]),
        "pair_count": len(metadata["directed_edge_slots"]),
    }
    weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25,
               "kl": 0.1, "lm": 0.2, "technique": 0.1, "pair": 0.1,
               "free_nats": 1.0}
    train_techniques = data["train"]["future_techniques"].max(axis=1)
    train_pairs = data["train"]["future_lateral_edges"].max(axis=1)
    pos_weights = {
        "edge": positive_weight(data["train"]["future_edge_presence"]),
        "lm": positive_weight(data["train"]["lateral_movement_within_horizon"]),
        "technique": positive_weight(train_techniques), "pair": positive_weight(train_pairs),
    }
    pos_weights = {name: value.to(device) for name, value in pos_weights.items()}
    state_indices_t = torch.from_numpy(state_permutations).long().to(device)
    edge_indices_t = torch.from_numpy(edge_permutations).long().to(device)
    train_loader = make_loader(data["train"], scaler, args.batch_size, shuffle=True)
    validation_loader = make_loader(data["validation"], scaler, args.batch_size, shuffle=False)

    seed_everything(123)
    smoke_model = CompactRSSM(**model_config).to(device)
    smoke = smoke_tests(smoke_model, train_loader, context_steps, group_slices_list,
                        weights, pos_weights, device)
    print("smoke tests:", smoke)

    candidate_summaries = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        state, summary = train_seed(
            seed, train_loader, validation_loader, model_config, context_steps,
            group_slices_list, weights, pos_weights, state_indices_t, edge_indices_t,
            device, args.epochs, args.patience, args.learning_rate,
        )
        states[seed] = state; candidate_summaries.append(summary)
    selected = min(candidate_summaries, key=lambda row: row["best_validation_selection"])
    selected_seed = int(selected["seed"])
    model = CompactRSSM(**model_config).to(device)
    model.load_state_dict(states[selected_seed]); model.eval()

    mc = {
        split: mc_predictions(model, data[split]["context_states"], scaler, device,
                              args.mc_samples, args.batch_size, selected_seed + offset)
        for split, offset in [("validation", 1000), ("test", 2000)]
    }
    predictions = {split: {name: values.mean(axis=0) for name, values in outputs.items()}
                   for split, outputs in mc.items()}
    normalized = {
        split: {
            "context": normalize(scaler, data[split]["context_states"]),
            "future": normalize(scaler, data[split]["future_states"]),
        } for split in ["validation", "test"]
    }
    state_results = {
        split: state_metrics(predictions[split]["state"], normalized[split]["future"],
                             normalized[split]["context"], data[split]["future_states"], group_slices)
        for split in ["validation", "test"]
    }

    val_lm = data["validation"]["lateral_movement_within_horizon"].astype(int)
    test_lm = data["test"]["lateral_movement_within_horizon"].astype(int)
    lm_threshold = best_f1_threshold(val_lm, predictions["validation"]["lm"])
    lm_results: dict[str, Any] = {
        "threshold_selected_on_validation": lm_threshold,
        "validation": binary_metrics(val_lm, predictions["validation"]["lm"], lm_threshold),
        "test": binary_metrics(test_lm, predictions["test"]["lm"], lm_threshold),
    }
    sample_manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    for split, labels in [("validation", val_lm), ("test", test_lm)]:
        audit = sample_manifest[sample_manifest.split.eq(split)].reset_index(drop=True)
        mask = audit.lateral_movement_already_observed.eq(0).to_numpy()
        lm_results[f"{split}_before_any_observed_lateral"] = binary_metrics(
            labels[mask], predictions[split]["lm"][mask], lm_threshold
        )

    technique_results = {}
    for index, technique in enumerate(metadata["technique_targets"]):
        technique_results[technique] = {}
        for split in ["validation", "test"]:
            target = data[split]["future_techniques"].max(axis=1)[:, index].astype(int)
            technique_results[technique][split] = binary_metrics(
                target, predictions[split]["technique"][:, index], 0.5
            )
    pair_results = {}
    for split in ["validation", "test"]:
        target = data[split]["future_lateral_edges"].max(axis=1)
        positive = target.max(axis=1) > 0
        top = predictions[split]["pair"][positive].argmax(axis=1)
        correct = target[positive][np.arange(positive.sum()), top] > 0
        pair_results[split] = {"positive_samples": int(positive.sum()),
                               "top1_accuracy_any_true_lm_pair": float(correct.mean())}
    edge_results = {}
    for split in ["validation", "test"]:
        target = data[split]["future_edge_presence"]
        probability = predictions[split]["edge"]
        edge_results[split] = {
            "average_precision": float(average_precision_score(target.ravel(), probability.ravel())),
            "positive_rate": float(target.mean()),
        }

    test_state_std = mc["test"]["state"].std(axis=0)
    test_absolute_error = np.abs(predictions["test"]["state"] - normalized["test"]["future"])
    uncertainty = {
        "mc_samples": args.mc_samples,
        "mean_normalized_state_std": float(test_state_std.mean()),
        "active_mean_normalized_state_std": float(
            test_state_std[data["test"]["future_states"][:, :, 0] > 0].mean()
        ),
        "quiet_mean_normalized_state_std": float(
            test_state_std[data["test"]["future_states"][:, :, 0] == 0].mean()
        ),
        "state_std_absolute_error_correlation": float(np.corrcoef(
            test_state_std.mean(axis=2).ravel(), test_absolute_error.mean(axis=2).ravel()
        )[0, 1]),
        "mean_lm_probability_std": float(mc["test"]["lm"].std(axis=0).mean()),
    }

    permutation_rows = []
    permuted_scores = []
    for index, order in enumerate(state_permutations):
        permuted_context = data["test"]["context_states"][..., order]
        outputs = mc_predictions(model, permuted_context, scaler, device, 5,
                                 args.batch_size, selected_seed + 3000 + index)
        state_mean = outputs["state"].mean(axis=0)
        lm_mean = outputs["lm"].mean(axis=0)
        permuted_future = data["test"]["future_states"][..., order]
        target_normalized = normalize(scaler, permuted_future)
        permutation_rows.append({
            "permutation_index": index,
            "identity": index == 0,
            "lm_average_precision": float(average_precision_score(test_lm, lm_mean)),
            "lm_f1_fixed_threshold": float(f1_score(test_lm, lm_mean >= lm_threshold, zero_division=0)),
            "state_normalized_mae": float(np.abs(state_mean - target_normalized).mean()),
        })
        permuted_scores.append(lm_mean)
    score_stack = np.stack(permuted_scores)

    ridge_metrics_path = ROOT / "outputs/mvp_v2/model/baseline_metrics.json"
    ridge_summary = None
    if ridge_metrics_path.exists():
        ridge = json.loads(ridge_metrics_path.read_text())
        ridge_summary = {
            "state_prediction_test": ridge["state_prediction"]["test"],
            "future_lateral_movement_test": ridge["future_lateral_movement"]["test"],
            "future_edge_presence_test": ridge["future_edge_presence"]["test"],
            "future_lateral_pair_test": ridge["future_lateral_pair_ranking"]["test"],
        }

    metrics = {
        "scope": "compact passive RSSM on equal-duration V2; overlapping windows remain correlated",
        "device": str(device), "torch_version": torch.__version__, "model_config": model_config,
        "training_config": {"weights": weights, "batch_size": args.batch_size,
                            "learning_rate": args.learning_rate, "max_epochs": args.epochs,
                            "patience": args.patience, "candidate_seeds": [row["seed"] for row in candidate_summaries],
                            "selected_seed": selected_seed, "selection_rule": "validation future group MSE + 0.25 edge BCE"},
        "smoke_tests": smoke,
        "seed_candidates": [{key: value for key, value in row.items() if key != "history"}
                            for row in candidate_summaries],
        "selected_training_history": selected["history"],
        "state_prediction": state_results,
        "future_lateral_movement": lm_results,
        "future_techniques": technique_results,
        "future_lateral_pair_ranking": pair_results,
        "future_edge_presence": edge_results,
        "uncertainty": uncertainty,
        "host_permutation_sensitivity": {
            "rows": permutation_rows,
            "mean_per_sample_lm_probability_range": float(
                np.mean(score_stack.max(axis=0) - score_stack.min(axis=0))
            ),
            "max_per_sample_lm_probability_range": float(
                np.max(score_stack.max(axis=0) - score_stack.min(axis=0))
            ),
        },
        "ridge_reference": ridge_summary,
        "limitations": [
            "Only 24 controlled episodes and 12 LM events.",
            "Overlapping sequence windows are correlated within episodes.",
            "The flattened fixed-slot encoder is not a graph message-passing network.",
            "Monte Carlo latent spread is not guaranteed calibrated uncertainty.",
            "No defensive action variable or intervention data is used.",
        ],
    }

    model_out = Path(args.model_out); metrics_out = Path(args.metrics_out)
    model_out.parent.mkdir(parents=True, exist_ok=True); metrics_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(), "model_config": model_config,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "feature_metadata": metadata, "selected_seed": selected_seed,
        "lm_threshold": lm_threshold,
    }
    torch.save(checkpoint, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    test_state = state_results["test"]
    print(f"selected seed={selected_seed} epoch={selected['best_epoch']} val={selected['best_validation_selection']:.4f}")
    print(f"test state MAE={test_state['normalized_mae']:.4f} persistence={test_state['persistence_normalized_mae']:.4f}")
    print(f"test active MAE={test_state['active_normalized_mae']:.4f} persistence={test_state['active_persistence_normalized_mae']:.4f}")
    print("test future LM:", lm_results["test"])
    print("test pre-first LM:", lm_results["test_before_any_observed_lateral"])
    print("test edge AP:", edge_results["test"]["average_precision"])
    print("test pair top-1:", pair_results["test"]["top1_accuracy_any_true_lm_pair"])
    print("uncertainty:", uncertainty)
    print(f"model -> {model_out}\nmetrics -> {metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
