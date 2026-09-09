#!/usr/bin/env python3
"""Define branch-aware metrics using V3 validation only; never load V3 test."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.branch_metrics import branch_forecast_metrics, calibration_metrics  # noqa: E402
from src.cyberwm.device import device_summary, resolve_device  # noqa: E402
from src.cyberwm.graph_rssm import GraphFeatureScaler, GraphRSSM  # noqa: E402


def load_training() -> Any:
    path = ROOT / "scripts/15_train_rssm.py"
    spec = importlib.util.spec_from_file_location("branch_contract_training", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def checkpoint_scaler(checkpoint: dict[str, Any]) -> GraphFeatureScaler:
    scaler = GraphFeatureScaler(); scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"]); scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_); scaler.n_samples_seen_ = 1; return scaler


def self_tests() -> dict[str, bool]:
    target = np.asarray([[[0.0]], [[1.0]]])
    complementary = np.asarray([[[[0.0]], [[0.0]]], [[[1.0]], [[1.0]]]])
    weights = np.full((2, 2), 0.5)
    result = branch_forecast_metrics(complementary, target, weights, {"all": slice(0, 1)})
    assert result["oracle_best_branch_mae"] == 0.0
    assert result["expected_forecast_mae"] == 0.5
    assert result["best_branch_utilization"] == [0.5, 0.5]
    assert result["mean_pairwise_branch_mae"] == 1.0
    identical = branch_forecast_metrics(
        np.repeat(target[None], 2, axis=0), target, weights, {"all": slice(0, 1)}
    )
    assert identical["mean_pairwise_branch_mae"] == 0.0
    collapsed = branch_forecast_metrics(
        complementary, target, np.asarray([[1.0, 0.0], [1.0, 0.0]]), {"all": slice(0, 1)}
    )
    assert abs(collapsed["weight_effective_count"] - 1.0) < 1e-12
    calibrated = calibration_metrics(np.asarray([0, 1]), np.asarray([0.0, 1.0]), bins=2)
    assert calibrated["brier"] == 0.0 and calibrated["expected_calibration_error"] == 0.0
    return {"complementary_oracle": True, "identical_collapse": True,
            "collapsed_weights": True, "calibration_boundaries": True}


def episode_summary(module: Any, sequence_dir: Path, probabilities: np.ndarray,
                    labels: np.ndarray, threshold: float, plan_path: Path) -> dict[str, Any]:
    manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    audit = manifest[manifest.split.eq("validation")].reset_index(drop=True).copy()
    if len(audit) != len(probabilities): raise ValueError("validation manifest/prediction mismatch")
    audit["probability"] = probabilities; audit["label"] = labels; audit["alert"] = probabilities >= threshold
    episode = audit.groupby(["episode_id", "scenario"], as_index=False).agg(
        max_probability=("probability", "max"), alert_windows=("alert", "sum"),
        exact_positive_windows=("label", "sum"),
    )
    plan = pd.read_csv(plan_path, dtype={"episode_id": str})[["episode_id", "seed"]]
    episode = episode.merge(plan, on="episode_id", how="left", validate="one_to_one")
    if episode.seed.isna().any(): raise ValueError("validation episode missing paired seed")
    pairs = []
    for seed, group in episode.groupby("seed"):
        stop = group[group.scenario.eq("scan_guess_then_stop")].iloc[0]
        progress = group[group.scenario.eq("one_hop")].iloc[0]
        pairs.append({
            "seed": int(seed), "stopped_episode": stop.episode_id,
            "progress_episode": progress.episode_id,
            "stopped_max_probability": float(stop.max_probability),
            "progress_max_probability": float(progress.max_probability),
            "progress_minus_stopped": float(progress.max_probability - stop.max_probability),
            "stopped_alerted": bool(stop.alert_windows > 0),
            "progress_alerted": bool(progress.alert_windows > 0),
        })
    difference = np.asarray([row["progress_minus_stopped"] for row in pairs])
    return {
        "threshold_selected_on_same_validation": threshold,
        "episodes": episode.to_dict(orient="records"), "matched_pairs": pairs,
        "matched_summary": {
            "pairs": len(pairs),
            "mean_progress_minus_stopped_max_probability": float(difference.mean()),
            "mean_absolute_probability_gap": float(np.abs(difference).mean()),
            "progress_ranked_above_stopped_fraction": float((difference > 0).mean()),
            "stopped_episode_alert_fraction": float(np.mean([row["stopped_alerted"] for row in pairs])),
            "progress_episode_alert_fraction": float(np.mean([row["progress_alerted"] for row in pairs])),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models/mvp_v3_graph_rssm_scratch.pt"))
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v3/sequences"))
    ap.add_argument("--paired-plan", default=str(ROOT / "configs/mvp_v3_new_episode_plan.csv"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v3/branching/contract_baseline_validation.json"))
    ap.add_argument("--mc-samples", type=int, default=20); ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=33001)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args(); device = resolve_device(args.device); module = load_training()
    tests = self_tests(); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GraphRSSM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    scaler = checkpoint_scaler(checkpoint); sequence_dir = Path(args.sequences_dir)
    # Intentional invariant: load validation only. Test is frozen and absent here.
    validation = module.load_split(sequence_dir, "validation")
    draws = module.mc_predictions(
        model, validation["context_states"], scaler, device,
        args.mc_samples, args.batch_size, args.seed,
    )
    target = module.normalize(scaler, validation["future_states"])
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    groups = {"global": slice(0, global_width), "node": slice(global_width, global_width + node_width),
              "edge": slice(global_width + node_width, metadata["state_feature_count"])}
    weights = np.full((len(target), args.mc_samples), 1.0 / args.mc_samples)
    future = branch_forecast_metrics(
        draws["state"], target, weights, groups,
        edge_probability_branches=draws["edge"], edge_targets=validation["future_edge_presence"],
    )
    lm_probability = draws["lm"].mean(axis=0)
    lm_labels = validation["lateral_movement_within_horizon"].astype(int)
    calibration = calibration_metrics(lm_labels, lm_probability)
    threshold = module.best_f1_threshold(lm_labels, lm_probability)
    outcome = module.binary_metrics(lm_labels, lm_probability, threshold)
    episodes = episode_summary(
        module, sequence_dir, lm_probability, lm_labels, threshold, Path(args.paired_plan)
    )
    report = {
        "scope": "V3 validation-only branching metric contract on existing RSSM stochastic draws",
        "test_access": "V3 test is not loaded by this script",
        "runtime": device_summary(device), "self_tests": tests,
        "forecast_candidates": {"type": "existing stochastic RSSM draws", "count": args.mc_samples,
                                "inference_weights": "uniform; current model has no discrete branch gate"},
        "future_graph": future,
        "exact_lm_within_30_seconds": {"meaning": "realized eventual outcome, not attacker intent",
                                        "threshold": threshold, "classification": outcome,
                                        "calibration": calibration,
                                        "mean_draw_probability_std": float(draws["lm"].std(axis=0).mean())},
        "dangerous_precursor_operational_view": {
            "meaning": "alert burden/detection on matched stopped/progressing precursor episodes",
            **episodes,
        },
        "interpretation_rules": [
            "Oracle best-branch MAE measures coverage only and is not an inference-time point score.",
            "Expected forecast MAE uses inference-time branch probabilities.",
            "High risk on a stopped precursor may be operationally valid while false for exact 30-second LM.",
            "ECE from 90 correlated validation windows is diagnostic, not deployment calibration evidence.",
        ],
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "expected_mae": future["expected_forecast_mae"],
        "oracle_mae": future["oracle_best_branch_mae"],
        "diversity": future["mean_pairwise_branch_mae"],
        "effective_best_draws": future["best_branch_effective_count"],
        "lm_brier": calibration["brier"], "lm_ece": calibration["expected_calibration_error"],
        "matched": episodes["matched_summary"],
    }, indent=2)); print(f"contract -> {output}"); return 0


if __name__ == "__main__": sys.exit(main())
