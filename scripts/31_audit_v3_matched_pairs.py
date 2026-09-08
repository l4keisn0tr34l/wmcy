#!/usr/bin/env python3
"""Audit stopped/progressing V3 episode pairs without changing model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-plan", default=str(ROOT / "configs/mvp_v3_new_episode_plan.csv"))
    ap.add_argument("--splits", default=str(ROOT / "configs/mvp_v3_split_assignments.csv"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--results-dir", default=str(ROOT / "outputs/mvp_v3/graph_rssm"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v3/graph_rssm/paired_prefix_audit.json"))
    args = ap.parse_args()
    plan = pd.read_csv(args.new_plan, dtype={"episode_id": str})
    split_map = pd.read_csv(args.splits, dtype=str).set_index("episode_id").split
    plan["split"] = plan.episode_id.map(split_map)
    pair_rows = []
    for seed, group in plan.groupby("seed"):
        stopped = group[group.scenario.eq("scan_guess_then_stop")].iloc[0]
        progress = group[group.scenario.eq("one_hop")].iloc[0]
        row = {"seed": int(seed), "split": stopped.split,
               "stopped_episode": stopped.episode_id, "progress_episode": progress.episode_id}
        if stopped.split != progress.split: raise ValueError(f"seed {seed} crosses splits")
        relative = {}
        role_values = []
        for name, episode in [("stopped", stopped.episode_id), ("progress", progress.episode_id)]:
            directory = Path(args.episodes_dir) / episode
            metadata = pd.read_csv(directory / "episode_metadata.csv").iloc[0]
            truth = pd.read_csv(directory / "ground_truth.csv")
            role_values.append((metadata.actor, metadata.pivot, metadata.target))
            capture_start = pd.to_datetime(metadata.capture_start, utc=True)
            start = pd.to_datetime(truth.start_time, utc=True)
            relative[name] = (start - capture_start).dt.total_seconds().to_numpy()
            expected = ["T1046", "T1110.001"] + (["T1021.004"] if name == "progress" else [])
            if truth.technique_id.tolist() != expected:
                raise ValueError(f"unexpected truth in {episode}: {truth.technique_id.tolist()}")
        if role_values[0] != role_values[1]: raise ValueError(f"roles differ for seed {seed}")
        row["roles"] = "->".join(role_values[0])
        row["discovery_start_delta_seconds"] = float(abs(relative["stopped"][0] - relative["progress"][0]))
        row["guessing_start_delta_seconds"] = float(abs(relative["stopped"][1] - relative["progress"][1]))
        row["progress_lm_start_seconds"] = float(relative["progress"][2])
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)

    regimes = {}
    for regime in ["scratch", "unsw_pretrained"]:
        alerts = pd.read_csv(Path(args.results_dir) / f"{regime}_episode_alerts.csv")
        rows = []; split_summary = {}
        for pair in pairs.itertuples(index=False):
            stop_rows = alerts[alerts.episode_id.eq(pair.stopped_episode)]
            progress_rows = alerts[alerts.episode_id.eq(pair.progress_episode)]
            # Model evaluation intentionally covers validation/test only.
            if stop_rows.empty and progress_rows.empty:
                continue
            if stop_rows.empty or progress_rows.empty:
                raise ValueError(f"incomplete evaluated pair for seed {pair.seed}")
            stop = stop_rows.iloc[0]
            progress = progress_rows.iloc[0]
            rows.append({
                "seed": pair.seed, "split": pair.split,
                "stopped_episode": pair.stopped_episode, "progress_episode": pair.progress_episode,
                "stopped_max_probability": float(stop.max_probability),
                "progress_max_probability": float(progress.max_probability),
                "progress_minus_stopped_max_probability": float(progress.max_probability - stop.max_probability),
                "stopped_alerted": bool(stop.num_alert_windows > 0),
                "progress_alerted_before_lm": bool(progress.detected_lm_within_horizon_before_first_lm),
                "both_alerted": bool(stop.num_alert_windows > 0 and progress.detected_lm_within_horizon_before_first_lm),
                "stopped_alert_windows": int(stop.num_alert_windows),
                "progress_alert_windows": int(progress.num_alert_windows),
            })
        frame = pd.DataFrame(rows)
        for split in ["train", "validation", "test"]:
            current = frame[frame.split.eq(split)]
            if current.empty: continue
            difference = current.progress_minus_stopped_max_probability.to_numpy()
            split_summary[split] = {
                "pairs": len(current),
                "mean_stopped_max_probability": float(current.stopped_max_probability.mean()),
                "mean_progress_max_probability": float(current.progress_max_probability.mean()),
                "mean_progress_minus_stopped_max_probability": float(difference.mean()),
                "mean_absolute_pair_probability_difference": float(np.abs(difference).mean()),
                "progress_ranked_above_stopped_fraction": float((difference > 0).mean()),
                "stopped_episode_alert_fraction": float(current.stopped_alerted.mean()),
                "progress_detected_before_lm_fraction": float(current.progress_alerted_before_lm.mean()),
                "both_pair_members_alerted_fraction": float(current.both_alerted.mean()),
            }
        regimes[regime] = {"pairs": rows, "summary": split_summary}
    timing = {
        split: {
            "pairs": len(group),
            "mean_discovery_start_delta_seconds": float(group.discovery_start_delta_seconds.mean()),
            "max_discovery_start_delta_seconds": float(group.discovery_start_delta_seconds.max()),
            "mean_guessing_start_delta_seconds": float(group.guessing_start_delta_seconds.mean()),
            "max_guessing_start_delta_seconds": float(group.guessing_start_delta_seconds.max()),
        }
        for split, group in pairs.groupby("split")
    }
    report = {
        "scope": "post-selection diagnostic comparison of same-seed stopped/progressing V3 pairs",
        "selection_effect": "none; no model weights, seeds, epochs, or thresholds are changed",
        "pair_design": "same roles and seeded control flow; real command runtimes make relative event times non-identical",
        "causal_interpretation": "the later scenario-controller decision to execute successful SSH is not present in the shared passive-telemetry prefix",
        "timing_match": timing,
        "regimes": regimes,
        "limitations": [
            "Only three validation and three test pairs.",
            "Episode maximum risk is an operational summary, not an independent per-window metric.",
            "Alerting on stopped dangerous precursors can be valid risk detection even when eventual LM does not occur.",
        ],
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for regime in regimes:
        print(regime, json.dumps(regimes[regime]["summary"]["test"], indent=2))
    print(f"audit -> {output}")
    return 0


if __name__ == "__main__": sys.exit(main())
