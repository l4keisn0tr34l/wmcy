#!/usr/bin/env python3
"""Audit fixed-identity, timing, and metadata shortcuts in the MVP corpus/model."""
from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]


def load_split(path: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(path / f"{split}.npz") as loaded:
        return {key: loaded[key].copy() for key in loaded.files}


def probabilities(model: object, values: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(values), model, dtype=float)
    return model.predict_proba(values)[:, 1]


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0, 1, 201), scores]))
    f1 = np.asarray([f1_score(labels, scores >= candidate, zero_division=0) for candidate in candidates])
    return float(candidates[np.flatnonzero(f1 == f1.max())[-1]])


def fit_shortcut(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray,
                 val_y: np.ndarray) -> tuple[object, np.ndarray, float, float]:
    if len(np.unique(train_y)) == 1:
        model: object = float(train_y[0])
        val_scores = probabilities(model, val_x)
        return model, val_scores, 1.0, best_threshold(val_y, val_scores)
    candidates = []
    for c_value in [0.01, 0.1, 1.0, 10.0]:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=42),
        )
        model.fit(train_x, train_y)
        scores = probabilities(model, val_x)
        candidates.append((average_precision_score(val_y, scores), c_value, model, scores))
    _, selected_c, model, val_scores = max(candidates, key=lambda item: item[0])
    return model, val_scores, selected_c, best_threshold(val_y, val_scores)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = scores >= threshold
    return {
        "count": int(len(labels)),
        "positives": int(labels.sum()),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
    }


def permute_hosts(values: np.ndarray, order: tuple[int, ...], metadata: dict[str, object]) -> np.ndarray:
    """Consistently relabel node and directed-edge slots; leave global features unchanged."""
    result = values.copy()
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"])
    edge_width = len(metadata["edge_feature_names"])
    node_count = len(metadata["node_slots"])
    edge_start = global_width + node_count * node_width
    old_pair_to_slot: dict[tuple[int, int], int] = {}
    slot_ips = [row["ip"] for row in metadata["node_slots"]]
    ip_to_slot = {ip: slot for slot, ip in enumerate(slot_ips)}
    for edge in metadata["directed_edge_slots"]:
        old_pair_to_slot[(ip_to_slot[edge["source_ip"]], ip_to_slot[edge["destination_ip"]])] = edge["slot"]

    for new_slot, old_slot in enumerate(order):
        new_slice = slice(global_width + new_slot * node_width, global_width + (new_slot + 1) * node_width)
        old_slice = slice(global_width + old_slot * node_width, global_width + (old_slot + 1) * node_width)
        result[..., new_slice] = values[..., old_slice]
    new_edge_slot = 0
    for new_source in range(node_count):
        for new_destination in range(node_count):
            if new_source == new_destination:
                continue
            old_edge_slot = old_pair_to_slot[(order[new_source], order[new_destination])]
            new_slice = slice(edge_start + new_edge_slot * edge_width, edge_start + (new_edge_slot + 1) * edge_width)
            old_slice = slice(edge_start + old_edge_slot * edge_width, edge_start + (old_edge_slot + 1) * edge_width)
            result[..., new_slice] = values[..., old_slice]
            new_edge_slot += 1
    return result


def forecast(model: dict[str, object], context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = context.shape
    normalized = model["scaler"].transform(context.reshape(-1, shape[-1]))
    latent = model["pca"].transform(normalized).reshape(shape[0], shape[1], -1)
    future_flat = model["dynamics"].predict(latent.reshape(len(latent), -1))
    scores = probabilities(model["future_lateral_model"], future_flat)
    horizon = model["feature_metadata"]["future_horizon_states"]
    future_latent = future_flat.reshape(len(context), horizon, -1)
    predicted_normalized = model["pca"].inverse_transform(
        future_latent.reshape(-1, future_latent.shape[-1])
    ).reshape(len(context), horizon, -1)
    return scores, predicted_normalized


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp/sequences"))
    ap.add_argument("--episode-manifest", default=str(ROOT / "outputs/mvp/episode_manifest.csv"))
    ap.add_argument("--model", default=str(ROOT / "models/mvp_baseline.joblib"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp/model/shortcut_audit.json"))
    args = ap.parse_args()

    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    sample_manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    episode_manifest = pd.read_csv(args.episode_manifest)
    data = {split: load_split(sequence_dir, split) for split in ["train", "validation", "test"]}
    labels = {split: data[split]["lateral_movement_within_horizon"].astype(int)
              for split in data}
    names = metadata["state_feature_names"]

    forbidden_terms = [
        "timestamp", "time", "state_id", "episode", "scenario", "seed", "actor", "target",
        "technique", "tactic", "lateral", "label", "source_ip", "destination_ip",
    ]
    forbidden_matches = {
        term: [name for name in names if term in name.lower()] for term in forbidden_terms
    }

    node_activity = [index for index, name in enumerate(names) if name.endswith("__activity_mask")]
    edge_presence = [index for index, name in enumerate(names) if name.endswith("__presence_mask")]
    global_width = len(metadata["global_feature_names"])
    actor_categories = ["ws1", "srv1", "srv2"]

    split_audits: dict[str, pd.DataFrame] = {}
    feature_sets: dict[str, dict[str, np.ndarray]] = {
        "full_observable_context": {},
        "last_observable_state_only": {},
        "global_context_only": {},
        "slot_specific_activity_and_edge_masks": {},
        "permutation_invariant_activity_counts": {},
        "forbidden_actor_metadata_only": {},
        "forbidden_context_state_index_only": {},
    }
    for split in data:
        context = data[split]["context_states"]
        audit = sample_manifest[sample_manifest.split.eq(split)].reset_index(drop=True)
        audit = audit.merge(
            episode_manifest[["episode_id", "actor"]], on="episode_id", how="left", validate="many_to_one"
        )
        if len(audit) != len(context):
            raise ValueError(f"sample manifest order mismatch for {split}")
        split_audits[split] = audit
        feature_sets["full_observable_context"][split] = context.reshape(len(context), -1)
        feature_sets["last_observable_state_only"][split] = context[:, -1, :]
        feature_sets["global_context_only"][split] = context[:, :, :global_width].reshape(len(context), -1)
        mask_indices = node_activity + edge_presence
        feature_sets["slot_specific_activity_and_edge_masks"][split] = context[:, :, mask_indices].reshape(len(context), -1)
        invariant = np.stack([
            context[:, :, node_activity].sum(axis=2),
            context[:, :, edge_presence].sum(axis=2),
        ], axis=2)
        feature_sets["permutation_invariant_activity_counts"][split] = invariant.reshape(len(context), -1)
        feature_sets["forbidden_actor_metadata_only"][split] = np.asarray([
            [float(actor == category) for category in actor_categories] for actor in audit.actor
        ])
        feature_sets["forbidden_context_state_index_only"][split] = audit[["context_last_state"]].to_numpy(float)

    shortcut_results: dict[str, object] = {}
    for name, split_features in feature_sets.items():
        fitted, val_scores, selected_c, threshold = fit_shortcut(
            split_features["train"], labels["train"],
            split_features["validation"], labels["validation"],
        )
        test_scores = probabilities(fitted, split_features["test"])
        val_pre_first = split_audits["validation"].lateral_movement_already_observed.eq(0).to_numpy()
        test_pre_first = split_audits["test"].lateral_movement_already_observed.eq(0).to_numpy()
        shortcut_results[name] = {
            "selected_C": selected_c,
            "threshold_selected_on_validation": threshold,
            "validation": metrics(labels["validation"], val_scores, threshold),
            "validation_pre_first_lateral_movement": metrics(
                labels["validation"][val_pre_first], val_scores[val_pre_first], threshold
            ),
            "test": metrics(labels["test"], test_scores, threshold),
            "test_pre_first_lateral_movement": metrics(
                labels["test"][test_pre_first], test_scores[test_pre_first], threshold
            ),
        }

    fitted_model = joblib.load(args.model)
    model_threshold = float(fitted_model["future_lateral_threshold"])
    permutation_rows = []
    per_sample_scores = []
    for order in permutations(range(len(metadata["node_slots"]))):
        context = permute_hosts(data["test"]["context_states"], order, metadata)
        future = permute_hosts(data["test"]["future_states"], order, metadata)
        scores, predicted_normalized = forecast(fitted_model, context)
        target_shape = future.shape
        target_normalized = fitted_model["scaler"].transform(
            future.reshape(-1, target_shape[-1])
        ).reshape(target_shape)
        permutation_rows.append({
            "new_slots_receive_old_slots": list(order),
            "identity_permutation": bool(order == tuple(range(len(order)))),
            "future_lm_average_precision": float(average_precision_score(labels["test"], scores)),
            "future_lm_f1_at_fixed_threshold": float(
                f1_score(labels["test"], scores >= model_threshold, zero_division=0)
            ),
            "normalized_future_state_mae": float(np.mean(np.abs(predicted_normalized - target_normalized))),
        })
        per_sample_scores.append(scores)
    per_sample_scores_array = np.asarray(per_sample_scores)

    role_tables: dict[str, object] = {}
    for split in ["train", "validation", "test"]:
        rows = episode_manifest[episode_manifest.split.eq(split)]
        role_tables[split] = {
            scenario: {
                "actors": group.actor.value_counts().to_dict(),
                "pivots": group["pivot"].value_counts().to_dict(),
                "targets": group.target.value_counts().to_dict(),
            }
            for scenario, group in rows.groupby("scenario")
        }

    pair_rows = []
    for episode in episode_manifest.itertuples(index=False):
        truth = pd.read_csv(ROOT / "lab/episodes" / episode.episode_id / "ground_truth.csv")
        lateral = truth[truth.tactic.astype(str).eq("Lateral Movement")]
        for event in lateral.itertuples(index=False):
            pair_rows.append({
                "split": episode.split,
                "episode_id": episode.episode_id,
                "source": event.actor,
                "target": event.target,
            })
    pairs = pd.DataFrame(pair_rows)
    pair_counts = (
        pairs.groupby(["split", "source", "target"]).size().rename("count").reset_index().to_dict("records")
    )

    report = {
        "scope": "MVP fixed-identity, metadata, timing-proxy, and host-permutation audit",
        "observed_input_integrity": {
            "model_input_npz_key": "context_states",
            "forbidden_feature_name_matches": forbidden_matches,
            "note": "IP strings are absent as values, but node/edge positions are fixed IP-specific slots.",
        },
        "role_distribution_by_split_and_scenario": role_tables,
        "lateral_pair_counts": pair_counts,
        "shortcut_baselines": shortcut_results,
        "host_permutation_sensitivity": {
            "equivalent_relabelings": permutation_rows,
            "mean_per_sample_lm_probability_range": float(
                np.mean(per_sample_scores_array.max(axis=0) - per_sample_scores_array.min(axis=0))
            ),
            "max_per_sample_lm_probability_range": float(
                np.max(per_sample_scores_array.max(axis=0) - per_sample_scores_array.min(axis=0))
            ),
            "interpretation": "A host-identity-invariant model would be stable under consistent relabeling; large changes indicate fixed-slot dependence.",
        },
        "known_design_risks": [
            "Only 20 episodes and four held-out episodes; overlapping windows are correlated.",
            "All progression follows the same scan -> password guessing -> SSH movement order.",
            "Scenario durations and action offsets occupy narrow scenario-specific ranges.",
            "Training LM pairs cover only four of six directed host pairs and no LM originates at srv2.",
            "Containers persist across episodes, so ARP caches, SSH known-host state, and other environment state can drift with capture order.",
            "Fixed SSH credentials, port, commands, and topology can create packet/byte templates.",
            "The selected lab_023 replay uses a train-seen srv1->srv2 LM pair and was selected after test inspection.",
        ],
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Shortcut baseline test results:")
    for name, result in shortcut_results.items():
        test = result["test"]
        pre_first = result["test_pre_first_lateral_movement"]
        print(
            f"  {name:43s} AP={test['average_precision']:.3f} F1={test['f1']:.3f} "
            f"pre-first-AP={pre_first['average_precision']:.3f} pre-first-F1={pre_first['f1']:.3f}"
        )
    print("Host permutation results:")
    for row in permutation_rows:
        print(
            f"  {row['new_slots_receive_old_slots']} AP={row['future_lm_average_precision']:.3f} "
            f"F1={row['future_lm_f1_at_fixed_threshold']:.3f} state_MAE={row['normalized_future_state_mae']:.3f}"
        )
    print(f"report -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
