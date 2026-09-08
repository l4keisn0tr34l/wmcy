#!/usr/bin/env python3
"""Evaluate episode-level LM alerts for a saved RSSM checkpoint."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
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
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = 1
    return scaler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "models/mvp_v2_rssm_kl_tuned.pt"))
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp_v2/rssm/kl_tuned"))
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=507)
    args = ap.parse_args()

    module = load_training_module()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = module.CompactRSSM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    scaler = checkpoint_scaler(checkpoint)
    sequence_dir = Path(args.sequences_dir)
    data = {split: module.load_split(sequence_dir, split) for split in ["validation", "test"]}
    predictions = {
        split: module.mc_predictions(
            model, data[split]["context_states"], scaler, torch.device("cpu"),
            args.mc_samples, args.batch_size, args.seed + index * 1000,
        )
        for index, split in enumerate(["validation", "test"])
    }
    lm_probability = {split: predictions[split]["lm"].mean(axis=0) for split in predictions}
    lm_labels = {split: data[split]["lateral_movement_within_horizon"].astype(int)
                 for split in data}
    threshold = module.best_f1_threshold(lm_labels["validation"], lm_probability["validation"])
    sample_manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    episode_frame, summary, sample_frame = module.episode_alert_report(
        sample_manifest, lm_probability, lm_labels, threshold, Path(args.episodes_dir)
    )
    summary["checkpoint"] = str(Path(args.checkpoint))
    summary["mc_samples"] = args.mc_samples
    summary["validation_metrics"] = module.binary_metrics(
        lm_labels["validation"], lm_probability["validation"], threshold
    )
    summary["test_metrics"] = module.binary_metrics(
        lm_labels["test"], lm_probability["test"], threshold
    )
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    episode_frame.to_csv(out / "episode_alerts.csv", index=False)
    sample_frame.to_csv(out / "sample_predictions.csv", index=False)
    (out / "episode_alert_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        out / "predictions.npz",
        validation_lm_probability=lm_probability["validation"],
        test_lm_probability=lm_probability["test"],
        validation_lm_draws=predictions["validation"]["lm"],
        test_lm_draws=predictions["test"]["lm"],
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"outputs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
