#!/usr/bin/env python3
"""One-shot sealed V4 action-test evaluation for both frozen initializations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.action_graph_rssm import ActionGraphRSSM  # noqa: E402
from src.cyberwm.branch_metrics import calibration_metrics  # noqa: E402
from src.cyberwm.device import resolve_device  # noqa: E402
from src.cyberwm.graph_rssm import GraphFeatureScaler  # noqa: E402

FROZEN_HASHES = {
    "scratch": "f69ece2d24ac07593d35666a83ea52ef8a768d1f51d4dcab13e7c1dc9f1c876f",
    "friday_initialized": "6b4cf070da873f04cd1fecb17cf351b9f9ba07bcbf45275b80aa69a8b100e351",
}


def load_training() -> Any:
    path = ROOT / "scripts/15_train_rssm.py"
    spec = importlib.util.spec_from_file_location("v4_sealed_metrics", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def scaler_from(checkpoint: dict[str, Any]) -> GraphFeatureScaler:
    scaler = GraphFeatureScaler(); scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"]); scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_); scaler.n_samples_seen_ = 1; return scaler


def binary_metrics(module: Any, labels: np.ndarray, probabilities: np.ndarray,
                   threshold: float) -> dict[str, Any]:
    result = module.binary_metrics(labels.astype(int), probabilities, threshold)
    result["calibration"] = calibration_metrics(labels.astype(int), probabilities)
    return result


def pair_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    positive = target.sum(axis=1) > 0
    prediction = probability.argmax(axis=1)
    return {
        "micro_average_precision": float(average_precision_score(target.ravel(), probability.ravel())),
        "positive_samples": int(positive.sum()),
        "top1_accuracy_positive_samples": float(np.mean(target[positive, prediction[positive]] > 0)),
    }


def equivariance(model: ActionGraphRSSM, context: torch.Tensor, action_type: torch.Tensor,
                 action_pair: torch.Tensor, state_permutations: np.ndarray,
                 pair_permutations: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad(): base = model.forecast_action(context, action_type, action_pair, sample=False)
    delta = {name: 0.0 for name in ["state", "edge", "pair", "lm", "technique"]}
    for state_order, pair_order in zip(state_permutations, pair_permutations):
        state_index = torch.from_numpy(state_order).to(device)
        pair_index = torch.from_numpy(pair_order).to(device)
        with torch.no_grad():
            predicted = model.forecast_action(
                context[..., state_index], action_type, action_pair[..., pair_index], sample=False
            )
        delta["state"] = max(delta["state"], float((predicted["decoded"] - base["decoded"][..., state_index]).abs().max()))
        delta["edge"] = max(delta["edge"], float((predicted["edge_logits"] - base["edge_logits"][..., pair_index]).abs().max()))
        delta["pair"] = max(delta["pair"], float((predicted["pair_logits"] - base["pair_logits"][..., pair_index]).abs().max()))
        delta["lm"] = max(delta["lm"], float((predicted["lm_logits"] - base["lm_logits"]).abs().max()))
        delta["technique"] = max(delta["technique"], float((predicted["technique_logits"] - base["technique_logits"]).abs().max()))
    if max(delta.values()) > 1e-5: raise AssertionError(f"sealed-test equivariance failed: {delta}")
    return {f"{name}_max_delta": value for name, value in delta.items()}


def evaluate(module: Any, model: ActionGraphRSSM, checkpoint: dict[str, Any],
             data: dict[str, np.ndarray], audit: pd.DataFrame,
             state_permutations: np.ndarray, pair_permutations: np.ndarray,
             device: torch.device) -> dict[str, Any]:
    scaler = scaler_from(checkpoint)
    context = torch.from_numpy(module.normalize(scaler, data["context_states"])).to(device)
    action_type = torch.from_numpy(data["action_type"]).to(device)
    action_pair = torch.from_numpy(data["action_pair"]).to(device)
    model.eval()
    with torch.no_grad(): factual = model.forecast_action(context, action_type, action_pair, sample=False)
    predicted_state = factual["decoded"].cpu().numpy(); target_state = module.normalize(scaler, data["future_states"])
    absolute = np.abs(predicted_state - target_state); active = data["future_states"] != 0
    edge_probability = torch.sigmoid(factual["edge_logits"]).cpu().numpy()
    lm_probability = torch.sigmoid(factual["lm_logits"]).cpu().numpy()
    lm_label = data["lateral_movement_within_horizon"].astype(int)
    pair_probability = torch.sigmoid(factual["pair_logits"]).cpu().numpy()
    pair_target = data["future_lateral_edges"].max(axis=1)
    permit_type = torch.tensor([[1.0, 0.0]], device=device).expand(len(context), -1)
    block_type = torch.tensor([[0.0, 1.0]], device=device).expand(len(context), -1)
    with torch.no_grad():
        permit = model.forecast_action(context, permit_type, action_pair, sample=False)
        block = model.forecast_action(context, block_type, action_pair, sample=False)
    permit_state = permit["decoded"].cpu().numpy(); block_state = block["decoded"].cpu().numpy()
    permit_lm = torch.sigmoid(permit["lm_logits"]).cpu().numpy()
    block_lm = torch.sigmoid(block["lm_logits"]).cpu().numpy()
    factual_error = absolute.mean(axis=(1, 2))
    opposite_state = np.where(lm_label[:, None, None].astype(bool), block_state, permit_state)
    opposite_error = np.abs(opposite_state - target_state).mean(axis=(1, 2))
    by_action = {}
    for action, mask in [("permit_ssh", lm_label == 1), ("block_ssh", lm_label == 0)]:
        by_action[action] = {
            "samples": int(mask.sum()), "normalized_state_mae": float(absolute[mask].mean()),
            "active_normalized_state_mae": float(absolute[mask][active[mask]].mean()),
            "future_edge_average_precision": float(average_precision_score(
                data["future_edge_presence"][mask].ravel(), edge_probability[mask].ravel()
            )), "mean_lm_probability": float(lm_probability[mask].mean()),
        }
    pair_rows = []
    for family, rows in audit.groupby("paired_family", sort=False):
        permit_index = int(rows.index[rows.action.eq("permit_ssh")][0])
        block_index = int(rows.index[rows.action.eq("block_ssh")][0])
        pair_rows.append({
            "paired_family": family, "permit_probability": float(lm_probability[permit_index]),
            "block_probability": float(lm_probability[block_index]),
            "permit_minus_block": float(lm_probability[permit_index] - lm_probability[block_index]),
            "factual_state_mae_permit": float(factual_error[permit_index]),
            "factual_state_mae_block": float(factual_error[block_index]),
        })
    train_threshold = float(checkpoint["lm_threshold_from_training"])
    return {
        "state": {"normalized_mae": float(absolute.mean()),
                  "active_normalized_mae": float(absolute[active].mean()),
                  "quiet_normalized_mae": float(absolute[~active].mean())},
        "future_edge": {"average_precision": float(average_precision_score(
            data["future_edge_presence"].ravel(), edge_probability.ravel()
        ))},
        "lm_fixed_threshold_0_5": binary_metrics(module, lm_label, lm_probability, 0.5),
        "lm_frozen_training_threshold": binary_metrics(module, lm_label, lm_probability, train_threshold),
        "pair": pair_metrics(pair_target, pair_probability), "by_factual_action": by_action,
        "same_context_action_effect": {
            "mean_permit_lm_probability": float(permit_lm.mean()),
            "mean_block_lm_probability": float(block_lm.mean()),
            "mean_permit_minus_block_lm_probability": float((permit_lm - block_lm).mean()),
            "contexts_with_permit_probability_above_block": int((permit_lm > block_lm).sum()),
            "contexts": len(context),
            "mean_normalized_state_difference": float(np.abs(permit_state - block_state).mean()),
            "factual_action_lower_state_error_fraction": float((factual_error < opposite_error).mean()),
            "mean_opposite_minus_factual_state_mae": float((opposite_error - factual_error).mean()),
        },
        "matched_capture_pairs": pair_rows,
        "equivariance": equivariance(
            model, context, action_type, action_pair, state_permutations, pair_permutations, device
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v4/action_sequences"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v4/action_model/sealed_test.json"))
    ap.add_argument("--protocol-check-only", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args(); directory = Path(args.sequences_dir)
    paths = {regime: Path(args.models_dir) / f"mvp_v4_action_graph_rssm_{regime}.pt" for regime in FROZEN_HASHES}
    for regime, path in paths.items():
        actual = sha256(path)
        if actual != FROZEN_HASHES[regime]: raise ValueError(f"{regime} checkpoint hash {actual} is not frozen hash")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("v4_test_loaded") is not False: raise ValueError(f"{regime} checkpoint test flag invalid")
    if args.protocol_check_only:
        if (directory / "test.npz").exists(): raise ValueError("test arrays already exist during pre-unlock check")
        print("V4 sealed evaluator/checkpoint protocol PASS; test arrays absent")
        return 0
    if not (directory / "test.npz").is_file(): raise FileNotFoundError("build test once with script 42 --unlock-test")
    with np.load(directory / "test.npz") as loaded: data = {name: loaded[name] for name in loaded.files}
    metadata = json.loads((directory / "test_feature_metadata.json").read_text())
    audit = pd.read_csv(directory / "test_sample_manifest.csv")
    if len(data["context_states"]) != 10 or len(audit) != 10: raise ValueError("sealed test must have 10 samples after timing-only paired-family exclusion")
    if set(audit.episode_id) & {"lab_089", "lab_090"}: raise ValueError("temporally invalid paired family entered test")
    module = load_training(); state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    device = resolve_device(args.device); results = {}
    for regime, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["feature_metadata"]["state_feature_names"] != metadata["state_feature_names"]:
            raise ValueError(f"{regime}: train/test feature contract mismatch")
        model = ActionGraphRSSM(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        results[regime] = evaluate(
            module, model, checkpoint, data, audit,
            state_permutations, pair_permutations, device,
        )
    report = {
        "scope": "one-shot V4 sealed action-test evaluation",
        "checkpoint_hashes": FROZEN_HASHES, "test_samples": 10,
        "test_pairs": 5, "model_selection": "none; both prespecified initializations reported",
        "post_unlock_protocol_deviation": {
            "prediction_before_deviation": False,
            "reason": "lab_090 had only five complete post-action windows for the frozen six-state horizon",
            "resolution": "exclude the complete predefined lab_089/lab_090 permit-block family using timing only",
        },
        "results": results,
        "comparison": {
            "friday_minus_scratch_state_mae": results["friday_initialized"]["state"]["normalized_mae"] - results["scratch"]["state"]["normalized_mae"],
            "friday_minus_scratch_edge_ap": results["friday_initialized"]["future_edge"]["average_precision"] - results["scratch"]["future_edge"]["average_precision"],
            "friday_minus_scratch_lm_brier": results["friday_initialized"]["lm_fixed_threshold_0_5"]["brier"] - results["scratch"]["lm_fixed_threshold_0_5"]["brier"],
        },
        "limitations": ["five correlated permit/block seed pairs after one timing-only family exclusion", "one fixed three-container topology",
                        "action deterministically controls SSH outcome", "matched prefixes are not exact clones",
                        "counterfactual unfactual outcomes follow lab design and are not separately observed"],
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError("sealed test report already exists; refusing repeat evaluation")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({regime: {"state_mae": value["state"]["normalized_mae"],
                               "edge_ap": value["future_edge"]["average_precision"],
                               "lm_f1_0_5": value["lm_fixed_threshold_0_5"]["f1"],
                               "lm_brier": value["lm_fixed_threshold_0_5"]["brier"],
                               "action_lm_delta": value["same_context_action_effect"]["mean_permit_minus_block_lm_probability"],
                               "factual_lower_error": value["same_context_action_effect"]["factual_action_lower_state_error_fraction"]}
                      for regime, value in results.items()}, indent=2))
    print(f"sealed report -> {output}"); return 0


if __name__ == "__main__": sys.exit(main())
