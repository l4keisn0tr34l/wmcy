"""Metrics for probabilistic multi-branch future graph forecasts."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


def normalized_weights(weights: np.ndarray, samples: int, branches: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (samples, branches):
        raise ValueError(f"weights must be [sample,branch]={samples,branches}, got {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("branch weights must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("every sample must have positive total branch weight")
    return values / totals


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray,
                        bins: int = 10) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("binary labels/probabilities differ or labels are not binary")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite in [0,1]")
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    brier = float(np.mean((probabilities - labels) ** 2))
    log_loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, probabilities, side="right") - 1, bins - 1)
    rows = []; ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if not mask.any(): continue
        confidence = float(probabilities[mask].mean()); frequency = float(labels[mask].mean())
        gap = abs(confidence - frequency); ece += mask.mean() * gap
        rows.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(mask.sum()), "mean_probability": confidence,
            "observed_frequency": frequency, "absolute_gap": gap,
        })
    ap = (float(average_precision_score(labels, probabilities))
          if labels.sum() and labels.sum() < len(labels) else None)
    return {"count": len(labels), "positive_count": int(labels.sum()),
            "brier": brier, "log_loss": log_loss,
            "expected_calibration_error": float(ece), "average_precision": ap,
            "reliability_bins": rows}


def _effective_count(proportions: np.ndarray) -> float:
    positive = proportions[proportions > 0]
    return float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.0


def branch_forecast_metrics(
    state_branches: np.ndarray,
    target_states: np.ndarray,
    weights: np.ndarray,
    group_slices: dict[str, slice],
    edge_probability_branches: np.ndarray | None = None,
    edge_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate candidate futures.

    Args:
        state_branches: [branch, sample, horizon, feature], normalized coordinates.
        target_states: [sample, horizon, feature], same coordinates.
        weights: [sample, branch], inference-time branch probabilities.
    """
    predicted = np.asarray(state_branches, dtype=np.float64)
    target = np.asarray(target_states, dtype=np.float64)
    if predicted.ndim != 4 or target.ndim != 3 or predicted.shape[1:] != target.shape:
        raise ValueError(f"state branch/target shape mismatch: {predicted.shape} vs {target.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("non-finite state branches/targets")
    branches, samples = predicted.shape[:2]
    probability = normalized_weights(weights, samples, branches)
    expected = np.einsum("nb,bnhf->nhf", probability, predicted)
    absolute = np.abs(predicted - target[None])
    trajectory_error = absolute.mean(axis=(2, 3))  # [branch,sample]
    best = trajectory_error.argmin(axis=0)
    utilization = np.bincount(best, minlength=branches).astype(float) / samples
    weight_mean = probability.mean(axis=0)
    weighted_variance = np.einsum(
        "nb,bnhf->nhf", probability, (predicted - expected[None]) ** 2
    )
    if branches > 1:
        pair_distances = [np.abs(predicted[a] - predicted[b]).mean()
                          for a in range(branches) for b in range(a + 1, branches)]
        diversity = float(np.mean(pair_distances))
    else:
        diversity = 0.0
    result: dict[str, Any] = {
        "branches": branches, "samples": samples,
        "expected_forecast_mae": float(np.abs(expected - target).mean()),
        "oracle_best_branch_mae": float(trajectory_error.min(axis=0).mean()),
        "oracle_gain_over_expected_mae": float(
            np.abs(expected - target).mean() - trajectory_error.min(axis=0).mean()
        ),
        "mean_pairwise_branch_mae": diversity,
        "mean_weighted_branch_std": float(np.sqrt(weighted_variance).mean()),
        "best_branch_utilization": utilization.tolist(),
        "best_branch_effective_count": _effective_count(utilization),
        "mean_inference_weights": weight_mean.tolist(),
        "weight_effective_count": _effective_count(weight_mean),
        "groups": {},
    }
    for name, group in group_slices.items():
        group_abs = absolute[..., group]
        group_branch_error = group_abs.mean(axis=(2, 3))
        result["groups"][name] = {
            "expected_forecast_mae": float(np.abs(expected[..., group] - target[..., group]).mean()),
            "oracle_best_branch_mae": float(group_branch_error.min(axis=0).mean()),
        }
    if (edge_probability_branches is None) != (edge_targets is None):
        raise ValueError("edge probabilities and targets must be provided together")
    if edge_probability_branches is not None:
        edge = np.asarray(edge_probability_branches, dtype=np.float64)
        edge_target = np.asarray(edge_targets, dtype=np.int64)
        if edge.shape[0] != branches or edge.shape[1:] != edge_target.shape:
            raise ValueError("edge branch/target shape mismatch")
        expected_edge = np.einsum("nb,bnhp->nhp", probability, edge)
        result["future_edge"] = {
            "average_precision": float(average_precision_score(edge_target.ravel(), expected_edge.ravel())),
            "positive_rate": float(edge_target.mean()),
            "mean_probability": float(expected_edge.mean()),
        }
    return result
