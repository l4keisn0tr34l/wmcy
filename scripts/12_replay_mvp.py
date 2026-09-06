#!/usr/bin/env python3
"""Generate a transparent console/JSON/HTML replay for one held-out MVP sample."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HOST_BY_IP = {"10.77.0.20": "ws1", "10.77.0.30": "srv1", "10.77.0.40": "srv2"}


def probability(model: object, values: np.ndarray) -> float:
    if isinstance(model, float):
        return model
    return float(model.predict_proba(values)[0, 1])


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="lab_023")
    ap.add_argument("--context-end-state", type=int, default=8)
    ap.add_argument("--model", default=str(ROOT / "models/mvp_baseline.joblib"))
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp/sequences"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp/replays"))
    args = ap.parse_args()

    model = joblib.load(args.model)
    sequences_dir = Path(args.sequences_dir)
    sample_manifest = pd.read_csv(sequences_dir / "sample_manifest.csv")
    selected_rows = sample_manifest[
        sample_manifest.episode_id.eq(args.episode)
        & sample_manifest.context_last_state.eq(args.context_end_state)
    ]
    if len(selected_rows) != 1:
        available = sample_manifest.loc[
            sample_manifest.episode_id.eq(args.episode), "context_last_state"
        ].tolist()
        raise ValueError(
            f"expected one sample for {args.episode} state {args.context_end_state}; "
            f"available context-end states={available}"
        )
    sample = selected_rows.iloc[0]
    split = str(sample.split)
    split_manifest = sample_manifest[sample_manifest.split.eq(split)].reset_index(drop=True)
    local_index = int(split_manifest.index[split_manifest.sample_id.eq(sample.sample_id)][0])
    with np.load(sequences_dir / f"{split}.npz") as loaded:
        data = {name: loaded[name][local_index : local_index + 1].copy() for name in loaded.files}

    scaler = model["scaler"]
    pca = model["pca"]
    context = data["context_states"]
    shape = context.shape
    context_latent = pca.transform(scaler.transform(context.reshape(-1, shape[-1]))).reshape(
        shape[0], shape[1], -1
    )
    future_latent = model["dynamics"].predict(context_latent.reshape(1, -1)).reshape(
        1, data["future_states"].shape[1], -1
    )
    predicted_normalized = pca.inverse_transform(future_latent.reshape(-1, future_latent.shape[-1]))
    predicted_states = scaler.inverse_transform(predicted_normalized).reshape(data["future_states"].shape)
    predicted_flat = future_latent.reshape(1, -1)

    lm_probability = probability(model["future_lateral_model"], predicted_flat)
    threshold = float(model["future_lateral_threshold"])
    technique_probabilities = {
        technique: probability(head, predicted_flat)
        for technique, head in model["technique_models"].items()
    }
    pair_probabilities = []
    pairs = model["feature_metadata"]["directed_edge_slots"]
    for pair, head in zip(pairs, model["lateral_edge_models"]):
        pair_probabilities.append({
            "source": HOST_BY_IP[pair["source_ip"]],
            "target": HOST_BY_IP[pair["destination_ip"]],
            "probability": probability(head, predicted_flat),
        })
    pair_probabilities.sort(key=lambda row: row["probability"], reverse=True)

    feature_names = model["feature_metadata"]["state_feature_names"]
    feature_index = {name: index for index, name in enumerate(feature_names)}
    global_names = model["feature_metadata"]["global_feature_names"]
    edge_presence_indexes = [
        index for index, name in enumerate(feature_names)
        if name.startswith("edge_") and name.endswith("__presence_mask")
    ]
    actual = data["future_states"][0]
    predicted = predicted_states[0]
    episode_global = pd.read_csv(
        ROOT / "lab/episodes" / args.episode / "states/global_states.csv"
    )
    episode_truth = pd.read_csv(
        ROOT / "lab/episodes" / args.episode / "state_ground_truth.csv"
    )
    episode_events = pd.read_csv(ROOT / "lab/episodes" / args.episode / "ground_truth.csv")
    prediction_available_time = pd.to_datetime(sample.prediction_available_time, utc=True)
    exact_lm_lead_seconds = None
    if len(episode_events):
        episode_events["start_time"] = pd.to_datetime(episode_events.start_time, utc=True)
        future_lm_events = episode_events[
            episode_events.tactic.astype(str).eq("Lateral Movement")
            & episode_events.start_time.ge(prediction_available_time)
            & episode_events.start_time.lt(
                prediction_available_time
                + pd.Timedelta(seconds=model["feature_metadata"]["future_horizon_seconds"])
            )
        ]
        if len(future_lm_events):
            exact_lm_lead_seconds = float(
                (future_lm_events.start_time.min() - prediction_available_time).total_seconds()
            )

    context_rows = []
    for state_id in range(int(sample.context_first_state), int(sample.context_last_state) + 1):
        row = episode_global.loc[episode_global.state_id.eq(state_id)].iloc[0]
        context_rows.append({
            "state_id": state_id,
            "time": row.window_start,
            "flow_count": int(row.flow_count),
            "new_edges": int(row.new_edges),
            "syn_count": float(row.syn_count),
            "total_bytes": float(row.total_bytes),
        })

    future_rows = []
    for horizon_index, state_id in enumerate(
        range(int(sample.future_first_state), int(sample.future_last_state) + 1)
    ):
        truth_row = episode_truth.loc[episode_truth.state_id.eq(state_id)].iloc[0]
        future_rows.append({
            "lead_start_seconds": horizon_index * 5,
            "lead_end_seconds": (horizon_index + 1) * 5,
            "state_id": state_id,
            "time": episode_global.loc[episode_global.state_id.eq(state_id), "window_start"].iloc[0],
            "predicted_flow_count": max(0.0, float(predicted[horizon_index, feature_index["flow_count"]])),
            "actual_flow_count": float(actual[horizon_index, feature_index["flow_count"]]),
            "predicted_new_edges": max(0.0, float(predicted[horizon_index, feature_index["new_edges"]])),
            "actual_new_edges": float(actual[horizon_index, feature_index["new_edges"]]),
            "predicted_syn_count": max(0.0, float(predicted[horizon_index, feature_index["syn_count"]])),
            "actual_syn_count": float(actual[horizon_index, feature_index["syn_count"]]),
            "predicted_internal_pair_count": float(
                np.clip(predicted[horizon_index, edge_presence_indexes], 0.0, 1.0).sum()
            ),
            "actual_internal_pair_count": float(data["future_edge_presence"][0, horizon_index].sum()),
            "actual_techniques": [] if pd.isna(truth_row.technique_ids) else str(truth_row.technique_ids).split(";"),
            "actual_lateral_movement": int(truth_row.has_lateral_movement),
        })

    true_lm_pairs = []
    true_pair_mask = data["future_lateral_edges"][0].max(axis=0) > 0
    for pair, present in zip(pairs, true_pair_mask):
        if present:
            true_lm_pairs.append(
                f"{HOST_BY_IP[pair['source_ip']]} -> {HOST_BY_IP[pair['destination_ip']]}"
            )

    result = {
        "scope": "held-out controlled Docker lab proof of concept",
        "episode_id": args.episode,
        "split": split,
        "context_end_state": int(sample.context_last_state),
        "context_seconds": model["feature_metadata"]["context_seconds"],
        "forecast_horizon_seconds": model["feature_metadata"]["future_horizon_seconds"],
        "future_lateral_movement_probability": lm_probability,
        "alert_threshold_selected_on_validation": threshold,
        "alert": lm_probability >= threshold,
        "actual_lateral_movement_within_horizon": bool(data["lateral_movement_within_horizon"][0]),
        "lateral_movement_already_observed_in_context": bool(sample.lateral_movement_already_observed),
        "prediction_available_time": prediction_available_time.isoformat(),
        "first_actual_lateral_movement_state_lead_seconds": (
            None if pd.isna(sample.first_lateral_movement_state_lead_seconds)
            else int(sample.first_lateral_movement_state_lead_seconds)
        ),
        "first_actual_lateral_movement_event_lead_seconds": exact_lm_lead_seconds,
        "predicted_technique_probabilities": technique_probabilities,
        "predicted_lateral_pair_ranking": pair_probabilities,
        "actual_lateral_pairs_within_horizon": true_lm_pairs,
        "observed_context": context_rows,
        "predicted_and_actual_future": future_rows,
        "limitations": [
            "Probabilities are not calibrated for operational deployment.",
            "The model was trained on 12 small controlled episodes.",
            "Overlapping sequence metrics are correlated within episodes.",
            "Exact source-target ranking is currently weak on the held-out test split.",
            "Ground truth shown here is used only after prediction for evaluation/display.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.episode}_context_{args.context_end_state}"
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"
    json_path.write_text(json.dumps(result, indent=2, default=json_default) + "\n", encoding="utf-8")

    technique_cards = "".join(
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
        f"<tr class={'lm' if row['actual_lateral_movement'] else ''}><td>{row['lead_start_seconds']}–{row['lead_end_seconds']}s</td>"
        f"<td>{row['predicted_flow_count']:.1f} / {row['actual_flow_count']:.0f}</td>"
        f"<td>{row['predicted_new_edges']:.1f} / {row['actual_new_edges']:.0f}</td>"
        f"<td>{row['predicted_syn_count']:.1f} / {row['actual_syn_count']:.0f}</td>"
        f"<td>{row['predicted_internal_pair_count']:.1f} / {row['actual_internal_pair_count']:.0f}</td>"
        f"<td>{escape(', '.join(row['actual_techniques']) or 'none')}</td></tr>"
        for row in future_rows
    )
    limitation_items = "".join(f"<li>{escape(item)}</li>" for item in result["limitations"])
    event_lead_text = (
        "no lateral movement in horizon"
        if exact_lm_lead_seconds is None
        else f"+{exact_lm_lead_seconds:.1f} seconds from prediction time"
    )
    alert_class = "alert" if result["alert"] else "quiet"
    alert_text = "ALERT" if result["alert"] else "NO ALERT"
    html_path.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cyber World Model MVP Replay</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0b1020;color:#e6edf7;margin:0;padding:28px}}main{{max-width:1100px;margin:auto}}
h1,h2{{color:#7dd3fc}}.scope{{color:#fbbf24}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.card,.grid>div{{background:#151d32;border:1px solid #334155;border-radius:10px;padding:16px}}.big{{font-size:2.2rem;font-weight:700}}
.alert{{color:#fb7185}}.quiet{{color:#86efac}}table{{width:100%;border-collapse:collapse;background:#111827;margin:10px 0 24px}}
th,td{{padding:9px;border-bottom:1px solid #334155;text-align:left}}th{{color:#93c5fd}}tr.lm{{background:#3f1d2e}}
small,.muted{{color:#94a3b8}}code{{color:#a7f3d0}}</style></head>
<body><main><h1>Predictive Cyber World Model — Held-out Replay</h1>
<p class="scope">{escape(result['scope'])} · episode <code>{escape(args.episode)}</code> · {escape(split)} split</p>
<div class="card"><div class="{alert_class}">{alert_text}</div><div class="big">{lm_probability:.1%}</div>
<p>Predicted lateral movement within the next 30 seconds · validation threshold {threshold:.1%}</p>
<p>Exact actual first lateral movement: {escape(event_lead_text)} · previously observed in context: {str(result['lateral_movement_already_observed_in_context']).lower()}</p></div>
<h2>Predicted future ATT&CK interpretation</h2><div class="grid">{technique_cards}</div>
<h2>Observed 15-second context (model input)</h2><table><tr><th>State</th><th>UTC time</th><th>Flows</th><th>New edges</th><th>SYN</th><th>Bytes</th></tr>{context_table}</table>
<h2>Predicted future world vs actual future</h2><p class="muted">Values are predicted / actual. Red rows contain actual lateral movement.</p>
<table><tr><th>Lead</th><th>Flows</th><th>New edges</th><th>SYN</th><th>Internal pairs</th><th>Actual ATT&CK truth</th></tr>{future_table}</table>
<h2>Predicted lateral source-target ranking</h2><table><tr><th>Candidate pair</th><th>Score</th></tr>{pair_rows}</table>
<p>Actual LM pair(s) in horizon: <strong>{escape(', '.join(true_lm_pairs) or 'none')}</strong></p>
<h2>Limitations</h2><ul>{limitation_items}</ul>
</main></body></html>""", encoding="utf-8")

    print(f"episode={args.episode} split={split} context_end={args.context_end_state}")
    print(
        f"future LM probability={lm_probability:.3f} threshold={threshold:.3f} "
        f"alert={result['alert']} actual={result['actual_lateral_movement_within_horizon']} "
        f"exact_event_lead={event_lead_text}"
    )
    print("techniques:", ", ".join(f"{key}={value:.3f}" for key, value in technique_probabilities.items()))
    print("top LM pairs:", ", ".join(f"{row['source']}->{row['target']}={row['probability']:.3f}" for row in pair_probabilities[:3]))
    print("actual LM pairs:", ", ".join(true_lm_pairs) or "none")
    print(f"JSON -> {json_path}")
    print(f"HTML -> {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
