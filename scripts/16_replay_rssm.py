#!/usr/bin/env python3
"""Render a self-contained held-out V2 replay from saved RSSM predictions."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
HOST_BY_IP = {"10.77.0.20": "ws1", "10.77.0.30": "srv1", "10.77.0.40": "srv2"}


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value).__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="lab_048")
    ap.add_argument("--context-end-state", type=int, default=6)
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--predictions", default=str(ROOT / "outputs/mvp_v2/rssm/predictions.npz"))
    ap.add_argument("--metrics", default=str(ROOT / "outputs/mvp_v2/rssm/metrics.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp_v2/replays"))
    args = ap.parse_args()

    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    metrics = json.loads(Path(args.metrics).read_text())
    manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    selected = manifest[
        manifest.episode_id.eq(args.episode)
        & manifest.context_last_state.eq(args.context_end_state)
    ]
    if len(selected) != 1:
        available = manifest.loc[manifest.episode_id.eq(args.episode), "context_last_state"].tolist()
        raise ValueError(f"expected one sample; available context states={available}")
    sample = selected.iloc[0]
    split = str(sample.split)
    if split not in {"validation", "test"}:
        raise ValueError("replay requires a held-out validation/test episode")
    split_manifest = manifest[manifest.split.eq(split)].reset_index(drop=True)
    local_index = int(split_manifest.index[split_manifest.sample_id.eq(sample.sample_id)][0])
    with np.load(sequence_dir / f"{split}.npz") as loaded:
        data = {name: loaded[name][local_index].copy() for name in loaded.files}
        split_context = loaded["context_states"].copy()
    with np.load(sequence_dir / "train.npz") as train_loaded:
        train_context = train_loaded["context_states"].copy()
    scaler = StandardScaler().fit(train_context.reshape(-1, train_context.shape[-1]))

    with np.load(args.predictions) as predictions:
        predicted_normalized = predictions[f"{split}_state_mean"][local_index]
        predicted_std = predictions[f"{split}_state_std"][local_index]
        edge_probability = predictions[f"{split}_edge_probability"][local_index]
        lm_probability = float(predictions[f"{split}_lm_probability"][local_index])
        lm_probability_std = float(predictions[f"{split}_lm_probability_std"][local_index])
        technique_values = predictions[f"{split}_technique_probability"][local_index]
        pair_values = predictions[f"{split}_pair_probability"][local_index]
    predicted_state = scaler.inverse_transform(predicted_normalized)
    threshold = float(metrics["future_lateral_movement"]["threshold_selected_on_validation"])

    feature_names = metadata["state_feature_names"]
    feature_index = {name: index for index, name in enumerate(feature_names)}
    pairs = metadata["directed_edge_slots"]
    techniques = metadata["technique_targets"]
    technique_probabilities = dict(zip(techniques, technique_values.astype(float)))
    pair_probabilities = [
        {
            "source": HOST_BY_IP[pair["source_ip"]],
            "target": HOST_BY_IP[pair["destination_ip"]],
            "probability": float(pair_values[index]),
        }
        for index, pair in enumerate(pairs)
    ]
    pair_probabilities.sort(key=lambda row: row["probability"], reverse=True)

    episode_dir = ROOT / "lab/episodes" / args.episode
    global_states = pd.read_csv(episode_dir / "states/global_states.csv")
    state_truth = pd.read_csv(episode_dir / "state_ground_truth.csv")
    events = pd.read_csv(episode_dir / "ground_truth.csv")
    prediction_time = pd.to_datetime(sample.prediction_available_time, utc=True)
    exact_lm_lead = None
    if len(events):
        events["start_time"] = pd.to_datetime(events.start_time, utc=True)
        future_lm = events[
            events.tactic.astype(str).eq("Lateral Movement")
            & events.start_time.ge(prediction_time)
            & events.start_time.lt(
                prediction_time + pd.Timedelta(seconds=metadata["future_horizon_seconds"])
            )
        ]
        if len(future_lm):
            exact_lm_lead = float((future_lm.start_time.min() - prediction_time).total_seconds())

    context_rows = []
    for state_id in range(int(sample.context_first_state), int(sample.context_last_state) + 1):
        row = global_states[global_states.state_id.eq(state_id)].iloc[0]
        context_rows.append({
            "state_id": state_id, "time": row.window_start,
            "flow_count": int(row.flow_count), "new_edges": int(row.new_edges),
            "syn_count": float(row.syn_count), "total_bytes": float(row.total_bytes),
        })

    future_rows = []
    for horizon_index, state_id in enumerate(
        range(int(sample.future_first_state), int(sample.future_last_state) + 1)
    ):
        truth = state_truth[state_truth.state_id.eq(state_id)].iloc[0]
        actual = data["future_states"][horizon_index]
        future_rows.append({
            "lead_start_seconds": horizon_index * metadata["window_seconds"],
            "lead_end_seconds": (horizon_index + 1) * metadata["window_seconds"],
            "state_id": state_id,
            "predicted_flow_count": max(0.0, float(predicted_state[horizon_index, feature_index["flow_count"]])),
            "actual_flow_count": float(actual[feature_index["flow_count"]]),
            "predicted_new_edges": max(0.0, float(predicted_state[horizon_index, feature_index["new_edges"]])),
            "actual_new_edges": float(actual[feature_index["new_edges"]]),
            "predicted_syn_count": max(0.0, float(predicted_state[horizon_index, feature_index["syn_count"]])),
            "actual_syn_count": float(actual[feature_index["syn_count"]]),
            "predicted_internal_pair_count": float(edge_probability[horizon_index].sum()),
            "actual_internal_pair_count": float(data["future_edge_presence"][horizon_index].sum()),
            "mean_normalized_uncertainty": float(predicted_std[horizon_index].mean()),
            "actual_techniques": [] if pd.isna(truth.technique_ids) else str(truth.technique_ids).split(";"),
            "actual_lateral_movement": int(truth.has_lateral_movement),
        })

    actual_pairs = []
    true_pair_mask = data["future_lateral_edges"].max(axis=0) > 0
    for pair, present in zip(pairs, true_pair_mask):
        if present:
            actual_pairs.append(f"{HOST_BY_IP[pair['source_ip']]} -> {HOST_BY_IP[pair['destination_ip']]}")

    result = {
        "scope": "selected held-out equal-duration V2 controlled-lab RSSM replay",
        "selection_disclosure": "sample selected after aggregate test inspection for a coherent replay; use aggregate metrics for model claims",
        "episode_id": args.episode, "split": split,
        "context_end_state": int(sample.context_last_state),
        "context_seconds": metadata["context_seconds"],
        "forecast_horizon_seconds": metadata["future_horizon_seconds"],
        "rssm_architecture": "64-D deterministic GRU + 16-D diagonal-Gaussian stochastic state; six-step prior rollout",
        "future_lateral_movement_probability": lm_probability,
        "future_lateral_movement_probability_std": lm_probability_std,
        "alert_threshold_selected_on_validation": threshold,
        "alert": lm_probability >= threshold,
        "actual_lateral_movement_within_horizon": bool(data["lateral_movement_within_horizon"]),
        "lateral_movement_already_observed_in_context": bool(sample.lateral_movement_already_observed),
        "prediction_available_time": prediction_time.isoformat(),
        "first_actual_lateral_movement_event_lead_seconds": exact_lm_lead,
        "predicted_technique_probabilities": technique_probabilities,
        "predicted_lateral_pair_ranking": pair_probabilities,
        "actual_lateral_pairs_within_horizon": actual_pairs,
        "observed_context": context_rows,
        "predicted_and_actual_future": future_rows,
        "aggregate_test_metrics": {
            "state_mae": metrics["state_prediction"]["test"]["normalized_mae"],
            "future_lm_f1": metrics["future_lateral_movement"]["test"]["f1"],
            "pre_first_lm_f1": metrics["future_lateral_movement"]["test_before_any_observed_lateral"]["f1"],
            "edge_average_precision": metrics["future_edge_presence"]["test"]["average_precision"],
            "lm_pair_top1": metrics["future_lateral_pair_ranking"]["test"]["top1_accuracy_any_true_lm_pair"],
        },
        "limitations": [
            "Only 24 controlled episodes and 12 lateral-movement events exist.",
            "This replay was selected after aggregate test inspection and is not independent model selection.",
            "Overlapping sequence windows are correlated within episodes.",
            "Monte Carlo latent spread is not calibrated operational uncertainty.",
            "Aggregate edge and exact source-target forecasting remain weak.",
            "Ground truth is joined only after saved predictions for evaluation/display.",
        ],
    }

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.episode}_context_{args.context_end_state}_rssm"
    json_path = out_dir / f"{stem}.json"; html_path = out_dir / f"{stem}.html"
    json_path.write_text(json.dumps(result, indent=2, default=json_default) + "\n", encoding="utf-8")

    cards = "".join(
        f"<div><strong>{escape(name)}</strong><br>{value:.1%}</div>"
        for name, value in technique_probabilities.items()
    )
    pair_rows = "".join(
        f"<tr><td>{escape(row['source'])} → {escape(row['target'])}</td><td>{row['probability']:.1%}</td></tr>"
        for row in pair_probabilities
    )
    context_table = "".join(
        f"<tr><td>{row['state_id']}</td><td>{escape(row['time'])}</td><td>{row['flow_count']}</td>"
        f"<td>{row['new_edges']}</td><td>{row['syn_count']:.0f}</td><td>{row['total_bytes']:.0f}</td></tr>"
        for row in context_rows
    )
    future_table = "".join(
        f"<tr class={'lm' if row['actual_lateral_movement'] else ''}>"
        f"<td>{row['lead_start_seconds']}–{row['lead_end_seconds']}s</td>"
        f"<td>{row['predicted_flow_count']:.1f} / {row['actual_flow_count']:.0f}</td>"
        f"<td>{row['predicted_new_edges']:.1f} / {row['actual_new_edges']:.0f}</td>"
        f"<td>{row['predicted_syn_count']:.1f} / {row['actual_syn_count']:.0f}</td>"
        f"<td>{row['predicted_internal_pair_count']:.1f} / {row['actual_internal_pair_count']:.0f}</td>"
        f"<td>{row['mean_normalized_uncertainty']:.3f}</td>"
        f"<td>{escape(', '.join(row['actual_techniques']) or 'none')}</td></tr>"
        for row in future_rows
    )
    limitations = "".join(f"<li>{escape(item)}</li>" for item in result["limitations"])
    lead_text = "none in horizon" if exact_lm_lead is None else f"+{exact_lm_lead:.1f} seconds"
    alert_class = "alert" if result["alert"] else "quiet"
    html_path.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>RSSM Cyber World Model Replay</title><style>
body{{font-family:system-ui,sans-serif;background:#0b1020;color:#e6edf7;margin:0;padding:28px}}main{{max-width:1100px;margin:auto}}
h1,h2{{color:#7dd3fc}}.scope{{color:#fbbf24}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card,.grid>div{{background:#151d32;border:1px solid #334155;border-radius:10px;padding:16px}}.big{{font-size:2.2rem;font-weight:700}}
.alert{{color:#fb7185}}.quiet{{color:#86efac}}table{{width:100%;border-collapse:collapse;background:#111827;margin:10px 0 24px}}
th,td{{padding:9px;border-bottom:1px solid #334155;text-align:left}}th{{color:#93c5fd}}tr.lm{{background:#3f1d2e}}.muted{{color:#94a3b8}}code{{color:#a7f3d0}}</style></head>
<body><main><h1>Stochastic Cyber World Model — Held-out V2 Replay</h1>
<p class="scope">{escape(result['scope'])} · <code>{escape(args.episode)}</code> · {escape(split)} split</p>
<div class="card"><div class="{alert_class}">{'ALERT' if result['alert'] else 'NO ALERT'}</div><div class="big">{lm_probability:.1%} ± {lm_probability_std:.1%}</div>
<p>Predicted lateral movement within 30 seconds · validation threshold {threshold:.1%}</p>
<p>No LM previously observed: {str(not result['lateral_movement_already_observed_in_context']).lower()} · exact first event lead: {escape(lead_text)}</p></div>
<h2>Imagined-future ATT&CK interpretation</h2><div class="grid">{cards}</div>
<h2>Observed 15-second context</h2><table><tr><th>State</th><th>UTC time</th><th>Flows</th><th>New edges</th><th>SYN</th><th>Bytes</th></tr>{context_table}</table>
<h2>Six-step imagined future vs actual</h2><p class="muted">Values are predicted / actual. Uncertainty is Monte Carlo normalized state spread. Red rows contain actual LM.</p>
<table><tr><th>Lead</th><th>Flows</th><th>New edges</th><th>SYN</th><th>Internal pairs</th><th>Uncertainty</th><th>Actual ATT&CK</th></tr>{future_table}</table>
<h2>Predicted lateral pair ranking</h2><table><tr><th>Candidate</th><th>Score</th></tr>{pair_rows}</table>
<p>Actual pair(s): <strong>{escape(', '.join(actual_pairs) or 'none')}</strong></p>
<h2>Disclosure and limitations</h2><p>{escape(result['selection_disclosure'])}</p><ul>{limitations}</ul>
</main></body></html>""", encoding="utf-8")

    print(f"episode={args.episode} split={split} context_end={args.context_end_state}")
    print(f"LM={lm_probability:.3f} ± {lm_probability_std:.3f} threshold={threshold:.3f} alert={result['alert']} exact_lead={lead_text}")
    print("techniques:", ", ".join(f"{key}={value:.3f}" for key, value in technique_probabilities.items()))
    print("top pairs:", ", ".join(f"{row['source']}->{row['target']}={row['probability']:.3f}" for row in pair_probabilities[:3]))
    print("actual pairs:", ", ".join(actual_pairs) or "none")
    print(f"JSON -> {json_path}\nHTML -> {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
