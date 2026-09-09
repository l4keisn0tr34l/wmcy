#!/usr/bin/env python3
"""Evaluate the already-frozen V3 passive branch model once on V4 cohorts."""
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
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.branch_metrics import calibration_metrics  # noqa: E402
from src.cyberwm.branching_graph_rssm import BranchingGraphRSSM  # noqa: E402
from src.cyberwm.device import resolve_device  # noqa: E402
from src.cyberwm.graph_rssm import GraphFeatureScaler  # noqa: E402

CHECKPOINT_SHA256 = "4f0524b5c25a9bd0e4f591362be948f0df12fa684b5140d55a7abcb147be874b"
FROZEN_LM_THRESHOLD = 0.37077322602272034


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scaler_from(checkpoint: dict[str, Any]) -> GraphFeatureScaler:
    scaler = GraphFeatureScaler(); scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"]); scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_); scaler.n_samples_seen_ = 1; return scaler


def build_passive_data(plan_path: Path, episodes: Path, metadata: dict[str, Any],
                       module: Any) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    plan = pd.read_csv(plan_path, dtype={"episode_id": str, "seed": str})
    plan = plan[plan.cohort.isin(["passive_branching", "direct_credential"])].copy()
    if len(plan) != 12 or not plan.split.eq("test").all(): raise ValueError("expected 12 passive/direct V4 test episodes")
    global_features = metadata["global_feature_names"]
    node_features = metadata["node_feature_names"][:-1]
    edge_features = metadata["edge_feature_names"][:-1]
    stores = {name: [] for name in ["context_states", "future_states", "future_edge_presence",
                                     "future_lateral_movement", "future_techniques", "future_lateral_edges",
                                     "lateral_movement_within_horizon"]}
    rows = []; sample = 0; context_steps = 3; horizon = 6
    for planned in plan.itertuples(index=False):
        state, edge, lm, technique, pair, global_states = module.build_episode_states(
            episodes / planned.episode_id, global_features, node_features, edge_features
        )
        times = pd.to_datetime(global_states.window_start, utc=True)
        for context_end in range(context_steps - 1, len(state) - horizon):
            future = slice(context_end + 1, context_end + 1 + horizon)
            stores["context_states"].append(state[context_end - context_steps + 1:context_end + 1])
            stores["future_states"].append(state[future]); stores["future_edge_presence"].append(edge[future])
            stores["future_lateral_movement"].append(lm[future]); stores["future_techniques"].append(technique[future])
            stores["future_lateral_edges"].append(pair[future])
            label = int(lm[future].max() > 0); stores["lateral_movement_within_horizon"].append(label)
            first_lm = np.flatnonzero(lm[future] > 0)
            rows.append({"sample_id": sample, "episode_id": planned.episode_id, "cohort": planned.cohort,
                         "scenario": planned.scenario, "paired_family": planned.paired_family,
                         "context_last_state": context_end, "future_first_state": context_end + 1,
                         "prediction_available_time": times.iloc[context_end] + pd.Timedelta(seconds=5),
                         "lateral_movement_within_horizon": label,
                         "seconds_to_first_lm": int(first_lm[0] * 5) if len(first_lm) else np.nan})
            sample += 1
    arrays = {name: np.asarray(values, dtype=np.float32) for name, values in stores.items()}
    if arrays["context_states"].shape != (180, 3, 141): raise ValueError(arrays["context_states"].shape)
    return arrays, pd.DataFrame(rows)


def predict(model: BranchingGraphRSSM, context: torch.Tensor) -> dict[str, np.ndarray]:
    outputs = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(context), 128): outputs.append(model.forecast_branches(context[start:start + 128], sample=False))
    return {name: torch.cat([part[name].cpu() for part in outputs]).numpy() for name in outputs[0]}


def metrics(module: Any, data: dict[str, np.ndarray], audit: pd.DataFrame,
            output: dict[str, np.ndarray], scaler: GraphFeatureScaler) -> dict[str, Any]:
    target = module.normalize(scaler, data["future_states"])
    weights = output["branch_weights"]; decoded = output["decoded"]
    expected = (weights[..., None, None] * decoded).sum(axis=1)
    branch_error = np.abs(decoded - target[:, None]).mean(axis=(2, 3))
    labels = data["lateral_movement_within_horizon"].astype(int)
    probability = weights[:, 1]
    expected_edge = (weights[..., None, None] * torch.sigmoid(torch.from_numpy(output["edge_logits"])).numpy()).sum(axis=1)
    classification = module.binary_metrics(labels, probability, FROZEN_LM_THRESHOLD)
    episodes = []
    for episode, rows in audit.groupby("episode_id", sort=False):
        indices = rows.index.to_numpy(); positive_rows = indices[labels[indices] == 1]
        pre_lm = indices if not len(positive_rows) else indices[indices <= positive_rows[0]]
        alert = pre_lm[probability[pre_lm] >= FROZEN_LM_THRESHOLD]
        episodes.append({"episode_id": episode, "cohort": rows.cohort.iloc[0], "scenario": rows.scenario.iloc[0],
                         "positive_windows": int(labels[indices].sum()), "max_pre_lm_probability": float(probability[pre_lm].max()),
                         "alert_before_or_at_first_positive": bool(len(alert)),
                         "first_alert_sample": int(alert[0]) if len(alert) else None})
    groups = {}
    for cohort, rows in audit.groupby("cohort", sort=False):
        idx = rows.index.to_numpy(); y = labels[idx]; p = probability[idx]
        groups[cohort] = {"samples": len(idx), "episodes": rows.episode_id.nunique(), "positive_windows": int(y.sum()),
                          "expected_state_mae": float(np.abs(expected[idx] - target[idx]).mean()),
                          "oracle_state_mae": float(branch_error[idx].min(axis=1).mean()),
                          "lm_brier": float(np.mean((p-y)**2)),
                          "lm_average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
                          "alerts_by_episode": int(sum(e["alert_before_or_at_first_positive"] for e in episodes if e["cohort"] == cohort))}
    return {
        "samples": len(labels), "positive_windows": int(labels.sum()),
        "expected_state_mae": float(np.abs(expected-target).mean()),
        "oracle_state_mae": float(branch_error.min(axis=1).mean()),
        "oracle_gain": float(np.abs(expected-target).mean() - branch_error.min(axis=1).mean()),
        "outcome_conditioned_branch_mae": float(branch_error[np.arange(len(labels)), labels].mean()),
        "opposite_outcome_branch_mae": float(branch_error[np.arange(len(labels)), 1-labels].mean()),
        "branch_state_diversity": float(np.abs(decoded[:, 0]-decoded[:, 1]).mean()),
        "future_edge_average_precision_expected": float(average_precision_score(data["future_edge_presence"].ravel(), expected_edge.ravel())),
        "lm_at_frozen_v3_validation_threshold": classification,
        "lm_calibration_diagnostic": calibration_metrics(labels, probability),
        "mean_branch_weights": weights.mean(axis=0).tolist(), "by_cohort": groups,
        "episodes": episodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models/mvp_v3_outcome_branching_graph_rssm_validation.pt"))
    ap.add_argument("--plan", default=str(ROOT / "configs/mvp_v4_episode_plan.csv"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--action-sequences", default=str(ROOT / "outputs/mvp_v4/action_sequences"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v4/passive_branch/sealed_test.json"))
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args(); checkpoint_path = Path(args.checkpoint)
    if sha256(checkpoint_path) != CHECKPOINT_SHA256: raise ValueError("passive branch checkpoint hash changed")
    output_path = Path(args.out)
    if output_path.exists(): raise FileExistsError("passive V4 test output already exists")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint["validation_only"] or checkpoint["v3_test_loaded"]: raise ValueError("checkpoint provenance changed")
    metadata = checkpoint["feature_metadata"]; module = load_script("passive_v4_base", ROOT / "scripts/10_build_mvp_sequences.py")
    data, audit = build_passive_data(Path(args.plan), Path(args.episodes_dir), metadata, module)
    action_dir = Path(args.action_sequences)
    with np.load(action_dir / "test.npz") as loaded: action_data = {name: loaded[name] for name in loaded.files}
    action_audit = pd.read_csv(action_dir / "test_sample_manifest.csv")
    if len(action_audit) != 10: raise ValueError("action-aligned passive comparison requires 10 eligible samples")
    scaler = scaler_from(checkpoint); device = resolve_device(args.device)
    model = BranchingGraphRSSM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    passive_context = torch.from_numpy(module.normalize(scaler, data["context_states"])).to(device)
    action_context = torch.from_numpy(module.normalize(scaler, action_data["context_states"])).to(device)
    passive_result = metrics(module, data, audit, predict(model, passive_context), scaler)
    action_result = metrics(module, action_data, action_audit.assign(cohort="action_aligned"),
                            predict(model, action_context), scaler)
    paired = []
    action_probability = predict(model, action_context)["branch_weights"][:, 1]
    for family, rows in action_audit.groupby("paired_family", sort=False):
        idx = rows.index.to_numpy(); permit = idx[rows.action.to_numpy() == "permit_ssh"][0]
        block = idx[rows.action.to_numpy() == "block_ssh"][0]
        paired.append({"paired_family": family, "permit_probability": float(action_probability[permit]),
                       "block_probability": float(action_probability[block]),
                       "absolute_probability_difference": float(abs(action_probability[permit]-action_probability[block]))})
    report = {
        "scope": "one-shot frozen passive outcome-branch evaluation on fresh V4 cohorts",
        "checkpoint_sha256": CHECKPOINT_SHA256, "training_or_tuning": "none",
        "frozen_lm_threshold_from_v3_validation": FROZEN_LM_THRESHOLD,
        "passive_and_direct_sliding_windows": passive_result,
        "pre_action_aligned_contexts_without_action_input": action_result,
        "pre_action_matched_pairs": paired,
        "interpretation": "branch-1 weight is supervised LM-outcome probability; oracle branch error measures coverage only",
        "limitations": ["overlapping windows are correlated", "12 passive/direct episodes on one topology",
                        "10 action contexts after timing-only family exclusion", "ECE is diagnostic, not deployment calibration",
                        "passive telemetry cannot identify future permit/block choice"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passive_direct": {k: passive_result[k] for k in ["samples","expected_state_mae","oracle_state_mae","future_edge_average_precision_expected"]},
                      "passive_direct_lm": {k: passive_result["lm_at_frozen_v3_validation_threshold"][k] for k in ["f1","average_precision","brier"]},
                      "action_aligned_without_action": {k: action_result["lm_at_frozen_v3_validation_threshold"][k] for k in ["f1","average_precision","brier"]}}, indent=2))
    print(f"passive V4 report -> {output_path}"); return 0


if __name__ == "__main__": sys.exit(main())
