#!/usr/bin/env python3
"""Audit RSSM host-relabeling equivariance for state, edge, LM, and pair outputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_training_module() -> Any:
    path = ROOT / "scripts/15_train_rssm.py"
    spec = importlib.util.spec_from_file_location("cyberwm_rssm_training", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def checkpoint_scaler(checkpoint: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_); scaler.n_samples_seen_ = 1
    return scaler


def reorder_normalized_state(module: Any, values: np.ndarray, scaler: StandardScaler,
                             state_order: np.ndarray) -> np.ndarray:
    raw = scaler.inverse_transform(values.reshape(-1, values.shape[-1])).reshape(values.shape)
    return module.normalize(scaler, raw[..., state_order])


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])


def pair_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    positive = target.max(axis=1) > 0
    top = probability[positive].argmax(axis=1)
    positive_target = target[positive]
    correct = positive_target[np.arange(positive.sum()), top] > 0
    per_slot = []
    for slot in range(target.shape[1]):
        slot_mask = target[:, slot] > 0
        per_slot.append({
            "slot": slot,
            "positive_samples": int(slot_mask.sum()),
            "top1_correct": int((probability[slot_mask].argmax(axis=1) == slot).sum()),
        })
    return {
        "positive_samples": int(positive.sum()),
        "micro_average_precision": float(average_precision_score(target.ravel(), probability.ravel())),
        "top1_accuracy_any_true_pair": float(correct.mean()),
        "top1_correct": int(correct.sum()),
        "top_choice_slots": np.bincount(probability.argmax(axis=1), minlength=target.shape[1]).tolist(),
        "per_pair": per_slot,
    }


def evaluate_split(module: Any, model: torch.nn.Module, scaler: StandardScaler,
                   split: dict[str, np.ndarray], state_permutations: np.ndarray,
                   pair_permutations: np.ndarray, mc_samples: int, batch_size: int,
                   seed: int, threshold: float | None) -> tuple[dict[str, Any], float]:
    width = split["context_states"].shape[-1]
    assert all(np.array_equal(np.sort(row), np.arange(width)) for row in state_permutations)
    pair_count = split["future_lateral_edges"].shape[-1]
    assert all(np.array_equal(np.sort(row), np.arange(pair_count)) for row in pair_permutations)
    assert np.array_equal(state_permutations[0], np.arange(width))
    assert np.array_equal(pair_permutations[0], np.arange(pair_count))

    pair_target_identity = split["future_lateral_edges"].max(axis=1).astype(int)
    edge_target_identity = split["future_edge_presence"].astype(int)
    lm_target = split["lateral_movement_within_horizon"].astype(int)
    outputs = []
    identity = None
    for permutation_index, (state_order, pair_order) in enumerate(
        zip(state_permutations, pair_permutations)
    ):
        mc = module.mc_predictions(
            model, split["context_states"][..., state_order], scaler, torch.device("cpu"),
            mc_samples, batch_size, seed,
        )
        prediction = {name: values.mean(axis=0) for name, values in mc.items()}
        if identity is None:
            identity = prediction
            if threshold is None:
                threshold = module.best_f1_threshold(lm_target, prediction["lm"])
        assert threshold is not None and identity is not None
        pair_target = pair_target_identity[:, pair_order]
        edge_target = edge_target_identity[..., pair_order]
        if int(pair_target.sum()) != int(pair_target_identity.sum()):
            raise AssertionError("pair target support changed under relabeling")
        expected_pair = identity["pair"][:, pair_order]
        expected_edge = identity["edge"][..., pair_order]
        expected_state = reorder_normalized_state(module, identity["state"], scaler, state_order)
        selected_new = prediction["pair"].argmax(axis=1)
        selected_old = pair_order[selected_new]
        identity_selected = identity["pair"].argmax(axis=1)
        outputs.append({
            "permutation_index": permutation_index,
            "state_order_new_to_old": state_order.tolist(),
            "pair_order_new_to_old": pair_order.tolist(),
            "lm_average_precision": float(average_precision_score(lm_target, prediction["lm"])),
            "lm_f1": float(f1_score(lm_target, prediction["lm"] >= threshold, zero_division=0)),
            "lm_probability_mae_from_identity": float(np.abs(prediction["lm"] - identity["lm"]).mean()),
            "technique_probability_mae_from_identity": float(
                np.abs(prediction["technique"] - identity["technique"]).mean()
            ),
            "edge_average_precision": float(average_precision_score(edge_target.ravel(), prediction["edge"].ravel())),
            "edge_equivariance_mae": float(np.abs(prediction["edge"] - expected_edge).mean()),
            "edge_equivariance_correlation": safe_correlation(prediction["edge"], expected_edge),
            "state_equivariance_mae_normalized": float(np.abs(prediction["state"] - expected_state).mean()),
            "state_equivariance_correlation": safe_correlation(prediction["state"], expected_state),
            "pair": pair_metrics(pair_target, prediction["pair"]),
            "pair_equivariance_mae": float(np.abs(prediction["pair"] - expected_pair).mean()),
            "pair_equivariance_correlation": safe_correlation(prediction["pair"], expected_pair),
            "pair_top_choice_consistency_with_identity": float(
                (selected_old == identity_selected).mean()
            ),
        })
    ranges = {}
    for metric in ["lm_average_precision", "lm_f1", "edge_average_precision",
                   "state_equivariance_mae_normalized", "edge_equivariance_mae",
                   "pair_equivariance_mae", "pair_top_choice_consistency_with_identity"]:
        values = [row[metric] for row in outputs]
        ranges[metric] = {"minimum": min(values), "maximum": max(values),
                          "mean": float(np.mean(values))}
    pair_top1 = [row["pair"]["top1_accuracy_any_true_pair"] for row in outputs]
    pair_ap = [row["pair"]["micro_average_precision"] for row in outputs]
    ranges["pair_top1_accuracy_any_true_pair"] = {
        "minimum": min(pair_top1), "maximum": max(pair_top1), "mean": float(np.mean(pair_top1))
    }
    ranges["pair_micro_average_precision"] = {
        "minimum": min(pair_ap), "maximum": max(pair_ap), "mean": float(np.mean(pair_ap))
    }
    return {"threshold": threshold, "permutations": outputs, "summary": ranges}, threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--published-checkpoint", default=str(ROOT / "models/mvp_v2_rssm.pt"))
    ap.add_argument("--tuned-checkpoint", default=str(ROOT / "models/mvp_v2_rssm_kl_tuned.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v2/rssm/pair_equivariance_audit.json"))
    ap.add_argument("--mc-samples", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=921)
    args = ap.parse_args()

    module = load_training_module()
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    data = {split: module.load_split(sequence_dir, split) for split in ["validation", "test"]}
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    report = {
        "scope": "post-selection host-relabeling diagnostic; no retraining or model selection",
        "mc_samples": args.mc_samples,
        "models": {},
        "limitations": [
            "The fixed validation/test episodes have been inspected repeatedly.",
            "Only 13 validation and 14 test pair-positive windows exist.",
            "Monte Carlo estimates still contain finite-draw noise.",
            "This audit does not replace evaluation on new role-balanced episodes.",
        ],
    }
    for model_index, (name, path) in enumerate([
        ("published_joint", args.published_checkpoint),
        ("kl_tuned_candidate", args.tuned_checkpoint),
    ]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        scaler = checkpoint_scaler(checkpoint)
        model = module.CompactRSSM(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        validation, threshold = evaluate_split(
            module, model, scaler, data["validation"], state_permutations,
            pair_permutations, args.mc_samples, args.batch_size,
            args.seed + model_index * 10_000, None,
        )
        test, _ = evaluate_split(
            module, model, scaler, data["test"], state_permutations,
            pair_permutations, args.mc_samples, args.batch_size,
            args.seed + model_index * 10_000 + 1_000, threshold,
        )
        report["models"][name] = {"checkpoint": path, "validation": validation, "test": test}
        summary = test["summary"]
        print(f"{name}: pair_top1={summary['pair_top1_accuracy_any_true_pair']} "
              f"pair_eq_mae={summary['pair_equivariance_mae']} "
              f"choice_consistency={summary['pair_top_choice_consistency_with_identity']}")

    for model in report["models"].values():
        for split in ["validation", "test"]:
            if len(model[split]["permutations"]) != 6:
                raise AssertionError("expected all six host permutations")
            for row in model[split]["permutations"]:
                for name, value in row.items():
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError(f"non-finite {name}={value}")
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"audit -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
