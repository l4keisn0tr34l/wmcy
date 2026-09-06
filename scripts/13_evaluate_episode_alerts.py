#!/usr/bin/env python3
"""Evaluate horizon alerts and exact lead time at held-out episode level."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def model_probabilities(model: object, values: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(values), model, dtype=float)
    return model.predict_proba(values)[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mvp_baseline.joblib"))
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp/sequences"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp/model"))
    args = ap.parse_args()

    model = joblib.load(args.model)
    sequence_dir = Path(args.sequences_dir)
    sample_manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    threshold = float(model["future_lateral_threshold"])
    horizon_seconds = int(model["feature_metadata"]["future_horizon_seconds"])
    rows: list[dict[str, object]] = []

    for split in ["validation", "test"]:
        audit = sample_manifest[sample_manifest.split.eq(split)].reset_index(drop=True).copy()
        with np.load(sequence_dir / f"{split}.npz") as loaded:
            context = loaded["context_states"]
            actual = loaded["lateral_movement_within_horizon"].astype(int)
        shape = context.shape
        context_latent = model["pca"].transform(
            model["scaler"].transform(context.reshape(-1, shape[-1]))
        ).reshape(shape[0], shape[1], -1)
        predicted_future = model["dynamics"].predict(context_latent.reshape(len(context), -1))
        probability = model_probabilities(model["future_lateral_model"], predicted_future)
        audit["probability"] = probability
        audit["alert"] = probability >= threshold
        audit["actual"] = actual
        audit["prediction_available_time"] = pd.to_datetime(audit.prediction_available_time, utc=True)

        for episode_id, episode_samples in audit.groupby("episode_id", sort=False):
            episode_samples = episode_samples.sort_values("prediction_available_time")
            truth = pd.read_csv(Path(args.episodes_dir) / episode_id / "ground_truth.csv")
            if len(truth):
                truth["start_time"] = pd.to_datetime(truth.start_time, utc=True)
                lm_events = truth[truth.tactic.astype(str).eq("Lateral Movement")]
            else:
                lm_events = truth
            first_lm_time = lm_events.start_time.min() if len(lm_events) else None

            true_horizon_alerts = episode_samples[episode_samples.alert & episode_samples.actual.eq(1)]
            pre_first_alerts = true_horizon_alerts[
                true_horizon_alerts.lateral_movement_already_observed.eq(0)
            ]
            first_pre_lm_alert_time = (
                pre_first_alerts.prediction_available_time.min() if len(pre_first_alerts) else None
            )
            exact_lead = (
                float((first_lm_time - first_pre_lm_alert_time).total_seconds())
                if first_lm_time is not None and first_pre_lm_alert_time is not None else None
            )
            false_windows = episode_samples[episode_samples.alert & episode_samples.actual.eq(0)]
            rows.append({
                "split": split,
                "episode_id": episode_id,
                "scenario": episode_samples.scenario.iloc[0],
                "has_lateral_movement": int(first_lm_time is not None),
                "num_samples": len(episode_samples),
                "num_actual_positive_windows": int(episode_samples.actual.sum()),
                "num_alert_windows": int(episode_samples.alert.sum()),
                "num_false_alert_windows_for_30s_target": len(false_windows),
                "max_probability": float(episode_samples.probability.max()),
                "detected_lm_within_horizon_before_first_lm": int(len(pre_first_alerts) > 0),
                "first_lm_time": first_lm_time,
                "first_pre_lm_alert_time": first_pre_lm_alert_time,
                "exact_warning_lead_seconds": exact_lead,
            })

    report = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "scope": "episode-level summary over correlated horizon windows in controlled validation/test episodes",
        "threshold_selected_on_validation": threshold,
        "forecast_horizon_seconds": horizon_seconds,
        "splits": {},
    }
    for split, group in report.groupby("split"):
        progressing = group[group.has_lateral_movement.eq(1)]
        nonprogressing = group[group.has_lateral_movement.eq(0)]
        leads = progressing.exact_warning_lead_seconds.dropna()
        summary["splits"][split] = {
            "episodes": len(group),
            "progressing_episodes": len(progressing),
            "progressing_detected_before_first_lm": int(
                progressing.detected_lm_within_horizon_before_first_lm.sum()
            ),
            "nonprogressing_episodes": len(nonprogressing),
            "nonprogressing_episodes_with_any_30s_false_alert": int(
                (nonprogressing.num_false_alert_windows_for_30s_target > 0).sum()
            ),
            "exact_warning_lead_seconds": [float(value) for value in leads],
            "mean_exact_warning_lead_seconds": float(leads.mean()) if len(leads) else None,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "episode_alerts.csv"
    json_path = out_dir / "episode_alert_summary.json"
    report.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(report.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"episode rows -> {csv_path}")
    print(f"summary -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
