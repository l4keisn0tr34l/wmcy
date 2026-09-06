#!/usr/bin/env python3
"""Train and evaluate an interpretable latent-dynamics MVP baseline.

The primary model predicts six future observable network states from three context
states through a PCA latent space. Security classifiers interpret predicted future
latents; they do not receive labels or scenario metadata as model inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def load_split(directory: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(directory / f"{split}.npz") as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def transform_states(scaler: StandardScaler, states: np.ndarray) -> np.ndarray:
    shape = states.shape
    return scaler.transform(states.reshape(-1, shape[-1])).reshape(shape)


def encode_states(scaler: StandardScaler, pca: PCA, states: np.ndarray) -> np.ndarray:
    shape = states.shape
    normalized = scaler.transform(states.reshape(-1, shape[-1]))
    return pca.transform(normalized).reshape(*shape[:-1], pca.n_components_)


def decode_latents(scaler: StandardScaler, pca: PCA, latents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = latents.shape
    normalized = pca.inverse_transform(latents.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)
    original = scaler.inverse_transform(normalized.reshape(-1, normalized.shape[-1])).reshape(normalized.shape)
    return normalized, original


def fit_dynamics(
    scaler: StandardScaler,
    pca: PCA,
    train: dict[str, np.ndarray],
    alpha: float,
) -> Ridge:
    context = encode_states(scaler, pca, train["context_states"])
    future = encode_states(scaler, pca, train["future_states"])
    model = Ridge(alpha=alpha)
    model.fit(context.reshape(len(context), -1), future.reshape(len(future), -1))
    return model


def forecast_latents(
    scaler: StandardScaler,
    pca: PCA,
    dynamics: Ridge,
    data: dict[str, np.ndarray],
) -> np.ndarray:
    context = encode_states(scaler, pca, data["context_states"])
    predicted = dynamics.predict(context.reshape(len(context), -1))
    horizon = data["future_states"].shape[1]
    return predicted.reshape(len(context), horizon, pca.n_components_)


def state_errors(
    scaler: StandardScaler,
    predicted_normalized: np.ndarray,
    data: dict[str, np.ndarray],
    group_slices: dict[str, slice],
) -> dict[str, object]:
    target_normalized = transform_states(scaler, data["future_states"])
    last_normalized = transform_states(scaler, data["context_states"][:, -1:, :])
    persistence = np.repeat(last_normalized, target_normalized.shape[1], axis=1)

    output: dict[str, object] = {
        "normalized_mae": float(np.mean(np.abs(predicted_normalized - target_normalized))),
        "persistence_normalized_mae": float(np.mean(np.abs(persistence - target_normalized))),
        "normalized_mae_by_horizon": [
            float(np.mean(np.abs(predicted_normalized[:, h] - target_normalized[:, h])))
            for h in range(target_normalized.shape[1])
        ],
        "persistence_mae_by_horizon": [
            float(np.mean(np.abs(persistence[:, h] - target_normalized[:, h])))
            for h in range(target_normalized.shape[1])
        ],
    }
    for name, feature_slice in group_slices.items():
        output[f"{name}_normalized_mae"] = float(
            np.mean(np.abs(predicted_normalized[:, :, feature_slice] - target_normalized[:, :, feature_slice]))
        )
        output[f"{name}_persistence_normalized_mae"] = float(
            np.mean(np.abs(persistence[:, :, feature_slice] - target_normalized[:, :, feature_slice]))
        )
    return output


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, object]:
    y_true = y_true.astype(int)
    prediction = (probability >= threshold).astype(int)
    result: dict[str, object] = {
        "count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        result["average_precision"] = float(average_precision_score(y_true, probability))
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
    return result


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), probability]))
    scores = [f1_score(y_true, probability >= threshold, zero_division=0) for threshold in candidates]
    best = np.flatnonzero(np.asarray(scores) == max(scores))
    # Prefer the highest equally good threshold to reduce false positives.
    return float(candidates[best[-1]])


def constant_or_logistic(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> object:
    values = np.unique(y.astype(int))
    if len(values) == 1:
        return float(values[0])
    model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=42)
    model.fit(x, y.astype(int))
    return model


def probabilities(model: object, x: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(x), model, dtype=float)
    return model.predict_proba(x)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp/sequences"))
    ap.add_argument("--model-out", default=str(ROOT / "models/mvp_baseline.joblib"))
    ap.add_argument("--metrics-out", default=str(ROOT / "outputs/mvp/model/baseline_metrics.json"))
    args = ap.parse_args()

    sequences_dir = Path(args.sequences_dir)
    train = load_split(sequences_dir, "train")
    validation = load_split(sequences_dir, "validation")
    test = load_split(sequences_dir, "test")
    feature_metadata = json.loads((sequences_dir / "feature_metadata.json").read_text())
    sample_manifest = pd.read_csv(sequences_dir / "sample_manifest.csv")

    train_state_pool = np.concatenate(
        [train["context_states"].reshape(-1, train["context_states"].shape[-1]),
         train["future_states"].reshape(-1, train["future_states"].shape[-1])],
        axis=0,
    )
    scaler = StandardScaler().fit(train_state_pool)

    global_width = len(feature_metadata["global_feature_names"])
    node_width = len(feature_metadata["node_feature_names"]) * len(feature_metadata["node_slots"])
    total_width = feature_metadata["state_feature_count"]
    group_slices = {
        "global": slice(0, global_width),
        "node": slice(global_width, global_width + node_width),
        "edge": slice(global_width + node_width, total_width),
    }

    candidates: list[dict[str, float]] = []
    fitted: dict[tuple[int, float], tuple[PCA, Ridge]] = {}
    for components in [8, 16, 24, 32]:
        pca = PCA(n_components=components, random_state=42).fit(scaler.transform(train_state_pool))
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            dynamics = fit_dynamics(scaler, pca, train, alpha)
            predicted_latent = forecast_latents(scaler, pca, dynamics, validation)
            predicted_normalized, _ = decode_latents(scaler, pca, predicted_latent)
            errors = state_errors(scaler, predicted_normalized, validation, group_slices)
            candidates.append({
                "components": components,
                "alpha": alpha,
                "validation_normalized_mae": errors["normalized_mae"],
                "validation_persistence_mae": errors["persistence_normalized_mae"],
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
            })
            fitted[(components, alpha)] = (pca, dynamics)

    selected = min(candidates, key=lambda row: row["validation_normalized_mae"])
    pca, dynamics = fitted[(int(selected["components"]), float(selected["alpha"]))]

    predicted_latents: dict[str, np.ndarray] = {}
    predicted_normalized: dict[str, np.ndarray] = {}
    predicted_states: dict[str, np.ndarray] = {}
    state_metric_sets: dict[str, dict[str, object]] = {}
    for name, data in [("validation", validation), ("test", test)]:
        predicted_latents[name] = forecast_latents(scaler, pca, dynamics, data)
        predicted_normalized[name], predicted_states[name] = decode_latents(
            scaler, pca, predicted_latents[name]
        )
        state_metric_sets[name] = state_errors(
            scaler, predicted_normalized[name], data, group_slices
        )

    # Train semantic heads on true train future latents; at inference they receive
    # only dynamics-predicted future latents.
    train_future_latent = encode_states(scaler, pca, train["future_states"]).reshape(len(train["future_states"]), -1)
    validation_predicted_flat = predicted_latents["validation"].reshape(len(validation["future_states"]), -1)
    test_predicted_flat = predicted_latents["test"].reshape(len(test["future_states"]), -1)
    train_lm = train["lateral_movement_within_horizon"]
    validation_lm = validation["lateral_movement_within_horizon"]
    test_lm = test["lateral_movement_within_horizon"]

    lm_candidates: list[tuple[float, object, np.ndarray, float]] = []
    for c in [0.01, 0.1, 1.0, 10.0]:
        model = constant_or_logistic(train_future_latent, train_lm, c)
        val_probability = probabilities(model, validation_predicted_flat)
        score = average_precision_score(validation_lm, val_probability)
        lm_candidates.append((score, model, val_probability, c))
    _, lm_model, validation_lm_probability, selected_c = max(lm_candidates, key=lambda item: item[0])
    threshold = best_f1_threshold(validation_lm, validation_lm_probability)
    test_lm_probability = probabilities(lm_model, test_predicted_flat)

    semantic_metrics: dict[str, object] = {
        "selected_C": selected_c,
        "validation": binary_metrics(validation_lm, validation_lm_probability, threshold),
        "test": binary_metrics(test_lm, test_lm_probability, threshold),
    }
    for split_name, y, probability in [
        ("validation", validation_lm, validation_lm_probability),
        ("test", test_lm, test_lm_probability),
    ]:
        audit = sample_manifest[sample_manifest.split.eq(split_name)].reset_index(drop=True)
        first_warning = audit.lateral_movement_already_observed.eq(0).to_numpy()
        semantic_metrics[f"{split_name}_before_any_observed_lateral"] = binary_metrics(
            y[first_warning], probability[first_warning], threshold
        )

    technique_models: dict[str, object] = {}
    technique_metrics: dict[str, object] = {}
    train_techniques = train["future_techniques"].max(axis=1)
    validation_techniques = validation["future_techniques"].max(axis=1)
    test_techniques = test["future_techniques"].max(axis=1)
    for index, technique in enumerate(feature_metadata["technique_targets"]):
        model = constant_or_logistic(train_future_latent, train_techniques[:, index])
        technique_models[technique] = model
        val_probability = probabilities(model, validation_predicted_flat)
        test_probability = probabilities(model, test_predicted_flat)
        technique_metrics[technique] = {
            "validation": binary_metrics(validation_techniques[:, index], val_probability, 0.5),
            "test": binary_metrics(test_techniques[:, index], test_probability, 0.5),
        }

    lm_edge_models: list[object] = []
    validation_lm_edge_probability = []
    test_lm_edge_probability = []
    train_lm_edges = train["future_lateral_edges"].max(axis=1)
    validation_lm_edges = validation["future_lateral_edges"].max(axis=1)
    test_lm_edges = test["future_lateral_edges"].max(axis=1)
    for pair_index in range(train_lm_edges.shape[1]):
        model = constant_or_logistic(train_future_latent, train_lm_edges[:, pair_index])
        lm_edge_models.append(model)
        validation_lm_edge_probability.append(probabilities(model, validation_predicted_flat))
        test_lm_edge_probability.append(probabilities(model, test_predicted_flat))
    validation_lm_edge_probability = np.stack(validation_lm_edge_probability, axis=1)
    test_lm_edge_probability = np.stack(test_lm_edge_probability, axis=1)

    pair_metrics: dict[str, object] = {}
    for name, truth, probability in [
        ("validation", validation_lm_edges, validation_lm_edge_probability),
        ("test", test_lm_edges, test_lm_edge_probability),
    ]:
        positive_samples = truth.max(axis=1) > 0
        top1 = np.argmax(probability[positive_samples], axis=1)
        top1_correct = truth[positive_samples][np.arange(positive_samples.sum()), top1] > 0
        pair_metrics[name] = {
            "positive_samples": int(positive_samples.sum()),
            "top1_accuracy_any_true_lm_pair": float(top1_correct.mean()) if len(top1_correct) else None,
        }

    edge_presence_indices = [
        index for index, feature in enumerate(feature_metadata["state_feature_names"])
        if feature.startswith("edge_") and feature.endswith("__presence_mask")
    ]
    edge_presence_metrics: dict[str, object] = {}
    for name, data in [("validation", validation), ("test", test)]:
        probability = np.clip(predicted_states[name][:, :, edge_presence_indices], 0.0, 1.0)
        truth = data["future_edge_presence"]
        edge_presence_metrics[name] = {
            "average_precision": float(average_precision_score(truth.ravel(), probability.ravel())),
            "positive_rate": float(truth.mean()),
        }

    metrics = {
        "scope": "controlled-lab MVP; overlapping samples remain correlated within each held-out episode",
        "selected_dynamics": selected,
        "all_dynamics_candidates": candidates,
        "state_prediction": state_metric_sets,
        "future_lateral_movement": semantic_metrics,
        "future_techniques": technique_metrics,
        "future_lateral_pair_ranking": pair_metrics,
        "future_edge_presence": edge_presence_metrics,
        "split_sequence_counts": {"train": len(train["context_states"]), "validation": len(validation["context_states"]), "test": len(test["context_states"])},
    }

    model_bundle = {
        "scaler": scaler,
        "pca": pca,
        "dynamics": dynamics,
        "future_lateral_model": lm_model,
        "future_lateral_threshold": threshold,
        "technique_models": technique_models,
        "lateral_edge_models": lm_edge_models,
        "feature_metadata": feature_metadata,
        "metrics_scope": metrics["scope"],
    }
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_out)
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"selected PCA={selected['components']} alpha={selected['alpha']} explained_variance={selected['explained_variance']:.3f}")
    print(
        f"test normalized state MAE={state_metric_sets['test']['normalized_mae']:.4f} "
        f"persistence={state_metric_sets['test']['persistence_normalized_mae']:.4f}"
    )
    print("test future LM:", semantic_metrics["test"])
    print("test pre-first-LM:", semantic_metrics["test_before_any_observed_lateral"])
    print("test future edge AP:", edge_presence_metrics["test"]["average_precision"])
    print("test LM pair top-1:", pair_metrics["test"]["top1_accuracy_any_true_lm_pair"])
    print(f"model -> {model_out}")
    print(f"metrics -> {metrics_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
