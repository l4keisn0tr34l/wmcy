#!/usr/bin/env python3
"""Build a standalone senior-facing HTML report from verified V2/RSSM outputs."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def metric_bar(label: str, value: float, maximum: float, color: str) -> str:
    width = max(1.0, min(100.0, 100.0 * value / maximum))
    return (
        f'<div class="bar-row"><span>{escape(label)}</span><div class="bar-track">'
        f'<div class="bar" style="width:{width:.2f}%;background:{color}"></div></div>'
        f'<strong>{value:.3f}</strong></div>'
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvp-v2-dir", default=str(ROOT / "outputs/mvp_v2"))
    ap.add_argument("--old-mvp-dir", default=str(ROOT / "outputs/mvp"))
    ap.add_argument("--replay", default=str(ROOT / "outputs/mvp_v2/replays/lab_048_context_6_rssm.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v2/report/rssm_eod_report.html"))
    args = ap.parse_args()

    v2 = Path(args.mvp_v2_dir)
    old = Path(args.old_mvp_dir)
    rssm = json.loads((v2 / "rssm/metrics.json").read_text())
    ridge = json.loads((v2 / "model/baseline_metrics.json").read_text())
    shortcut = json.loads((v2 / "model/shortcut_audit.json").read_text())
    old_shortcut = json.loads((old / "model/shortcut_audit.json").read_text())
    ablations = json.loads((v2 / "rssm/ablations.json").read_text())
    replay = json.loads(Path(args.replay).read_text())
    episodes = pd.read_csv(v2 / "episode_manifest.csv")

    rssm_state = rssm["state_prediction"]["test"]
    ridge_state = ridge["state_prediction"]["test"]
    rssm_lm = rssm["future_lateral_movement"]["test"]
    rssm_pre = rssm["future_lateral_movement"]["test_before_any_observed_lateral"]
    ridge_lm = ridge["future_lateral_movement"]["test"]
    ridge_pre = ridge["future_lateral_movement"]["test_before_any_observed_lateral"]
    episode = rssm["episode_alert_summary"]["splits"]["test"]
    uncertainty = rssm["uncertainty"]
    state_index_v2 = shortcut["shortcut_baselines"]["forbidden_context_state_index_only"]["test"]
    state_index_old = old_shortcut["shortcut_baselines"]["forbidden_context_state_index_only"]["test"]
    actor_v2 = shortcut["shortcut_baselines"]["forbidden_actor_metadata_only"]["test"]
    techniques = rssm["future_techniques"]
    parameter_count = 79_421

    duration_min = episodes.capture_duration_seconds.min()
    duration_max = episodes.capture_duration_seconds.max()
    totals = episodes[["num_states", "num_observations", "num_ground_truth_events",
                       "num_lateral_movement_events"]].sum()
    split_episode_counts = episodes.split.value_counts()
    sequence_counts = rssm["ridge_reference"] and ridge["split_sequence_counts"]

    state_bars = "".join([
        metric_bar("RSSM", rssm_state["normalized_mae"], 0.45, "#22c55e"),
        metric_bar("PCA/Ridge", ridge_state["normalized_mae"], 0.45, "#38bdf8"),
        metric_bar("Persistence", rssm_state["persistence_normalized_mae"], 0.45, "#f59e0b"),
    ])
    lm_bars = "".join([
        metric_bar("RSSM test", rssm_lm["f1"], 1.0, "#22c55e"),
        metric_bar("RSSM validation", rssm["future_lateral_movement"]["validation"]["f1"], 1.0, "#86efac"),
        metric_bar("PCA/Ridge test", ridge_lm["f1"], 1.0, "#38bdf8"),
    ])
    horizon_rows = "".join(
        f"<tr><td>+{5 * (index + 1)} s</td><td>{rssm_value:.3f}</td>"
        f"<td>{ridge_state['normalized_mae_by_horizon'][index]:.3f}</td>"
        f"<td>{rssm_state['persistence_mae_by_horizon'][index]:.3f}</td>"
        f"<td>{rssm_state['active_normalized_mae_by_horizon'][index]:.3f}</td>"
        f"<td>{rssm_state['active_persistence_normalized_mae_by_horizon'][index]:.3f}</td></tr>"
        for index, rssm_value in enumerate(rssm_state["normalized_mae_by_horizon"])
    )
    technique_rows = "".join(
        f"<tr><td>{escape(technique)}</td><td>{values['validation']['f1']:.3f}</td>"
        f"<td>{values['test']['f1']:.3f}</td><td>{values['test']['average_precision']:.3f}</td></tr>"
        for technique, values in techniques.items()
    )
    confusion = rssm_lm["confusion_matrix"]
    candidate_rows = "".join(
        f"<tr><td>{row['seed']}</td><td>{row['best_epoch']}</td>"
        f"<td>{row['best_validation_selection']:.4f}</td><td>{row['epochs_run']}</td></tr>"
        for row in rssm["seed_candidates"]
    )
    ablation_labels = {
        "published_joint_dynamics_selected": "Published joint",
        "frozen_two_stage": "Frozen two-stage",
        "pretrained_then_unfrozen": "Pretrained → unfrozen",
        "joint_from_scratch_matched_selection": "Joint scratch (matched)",
        "joint_zero_kl": "Joint zero-KL",
    }
    ablation_rows = "".join(
        f"<tr><td>{ablation_labels[name]}</td>"
        f"<td>{variant['state_prediction']['test']['normalized_mae']:.3f}</td>"
        f"<td>{variant['state_prediction']['test']['active_normalized_mae']:.3f}</td>"
        f"<td>{variant['future_edge_presence']['test']['average_precision']:.3f}</td>"
        f"<td>{variant['future_lateral_movement']['test']['f1']:.3f}</td>"
        f"<td>{variant['future_lateral_movement']['test']['average_precision']:.3f}</td>"
        f"<td>{variant['future_lateral_movement']['test_before_any_observed_lateral']['f1']:.3f}</td>"
        f"<td>{variant['future_lateral_pair_ranking']['test']['top1_accuracy_any_true_lm_pair']:.3f}</td>"
        f"<td>{variant['uncertainty']['mean_normalized_state_std']:.3f}</td></tr>"
        for name, variant in ablations["variants"].items()
    )
    context_rows = "".join(
        f"<tr><td>{row['state_id']}</td><td>{escape(row['time'])}</td><td>{row['flow_count']}</td>"
        f"<td>{row['new_edges']}</td><td>{row['syn_count']:.0f}</td><td>{row['total_bytes']:.0f}</td></tr>"
        for row in replay["observed_context"]
    )
    future_rows = "".join(
        f"<tr class={'event' if row['actual_lateral_movement'] else ''}>"
        f"<td>{row['lead_start_seconds']}–{row['lead_end_seconds']} s</td>"
        f"<td>{row['predicted_flow_count']:.1f} / {row['actual_flow_count']:.0f}</td>"
        f"<td>{row['predicted_new_edges']:.1f} / {row['actual_new_edges']:.0f}</td>"
        f"<td>{row['predicted_internal_pair_count']:.1f} / {row['actual_internal_pair_count']:.0f}</td>"
        f"<td>{row['mean_normalized_uncertainty']:.3f}</td>"
        f"<td>{escape(', '.join(row['actual_techniques']) or 'none')}</td></tr>"
        for row in replay["predicted_and_actual_future"]
    )
    replay_techniques = "".join(
        f'<div class="metric"><span>{escape(name)}</span><strong>{pct(value)}</strong></div>'
        for name, value in replay["predicted_technique_probabilities"].items()
    )
    replay_pairs = "".join(
        f"<tr><td>{escape(row['source'])} → {escape(row['target'])}</td>"
        f"<td>{pct(row['probability'])}</td></tr>"
        for row in replay["predicted_lateral_pair_ranking"][:3]
    )

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictive Cyber World Model — RSSM EOD Report</title>
<style>
:root{{--bg:#07111f;--panel:#0f1c2e;--panel2:#13243a;--line:#2a4360;--text:#e5edf6;--muted:#9eb0c4;--blue:#38bdf8;--green:#22c55e;--amber:#f59e0b;--red:#fb7185}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#06101d,#0a1728);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,sans-serif;line-height:1.55}}
main{{max-width:1180px;margin:auto;padding:34px}}h1{{font-size:2.45rem;margin:.1em 0}}h2{{margin-top:2em;color:#7dd3fc;border-bottom:1px solid var(--line);padding-bottom:.3em}}h3{{color:#bae6fd}}
.subtitle,.muted{{color:var(--muted)}}.badge{{display:inline-block;padding:.25rem .65rem;border:1px solid #0ea5e9;border-radius:999px;color:#7dd3fc;margin-right:.4rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:18px;box-shadow:0 8px 30px #0004}}
.card .value{{font-size:2rem;font-weight:800;color:#86efac}}.metric{{display:flex;justify-content:space-between;gap:15px;padding:.5em 0;border-bottom:1px solid #263b54}}.metric:last-child{{border:0}}
.callout{{border-left:4px solid var(--amber);background:#2b2112;padding:14px 18px;border-radius:7px}}.good{{border-left-color:var(--green);background:#102719}}.bad{{border-left-color:var(--red);background:#321725}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:10px;overflow:hidden;margin:12px 0}}th,td{{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line)}}th{{color:#93c5fd;background:#102138}}tr.event{{background:#3b1728}}
.diagram{{width:100%;height:auto;background:#0b1829;border:1px solid var(--line);border-radius:12px}}.box{{fill:#132a43;stroke:#38bdf8;stroke-width:2}}.box2{{fill:#153422;stroke:#22c55e;stroke-width:2}}.box3{{fill:#392412;stroke:#f59e0b;stroke-width:2}}.arrow{{stroke:#9eb0c4;stroke-width:2;marker-end:url(#arrow)}}.svgtext{{fill:#e5edf6;font-size:14px;font-family:system-ui,sans-serif;text-anchor:middle}}.smalltext{{fill:#9eb0c4;font-size:11px;font-family:system-ui,sans-serif;text-anchor:middle}}
.bar-row{{display:grid;grid-template-columns:120px 1fr 55px;align-items:center;gap:10px;margin:9px 0}}.bar-track{{height:14px;background:#26384d;border-radius:999px;overflow:hidden}}.bar{{height:100%;border-radius:999px}}
code{{color:#a7f3d0}}.foot{{font-size:.9rem;color:var(--muted);margin-top:3em}}@media(max-width:700px){{main{{padding:18px}}h1{{font-size:1.8rem}}.bar-row{{grid-template-columns:95px 1fr 48px}}}}
</style></head><body><main>
<span class="badge">EOD research report</span><span class="badge">Equal-duration V2</span><span class="badge">Passive RSSM</span>
<h1>Predictive Cyber World Model</h1>
<p class="subtitle">A stochastic recurrent model that observes 15 seconds of graph-structured network telemetry and imagines the next 30 seconds before interpreting likely attacker progression.</p>

<div class="grid">
<div class="card"><div class="muted">RSSM future-state MAE</div><div class="value">{rssm_state['normalized_mae']:.3f}</div><div>Ridge {ridge_state['normalized_mae']:.3f} · persistence {rssm_state['persistence_normalized_mae']:.3f}</div></div>
<div class="card"><div class="muted">Future lateral-movement F1</div><div class="value">{rssm_lm['f1']:.3f}</div><div>Validation {rssm['future_lateral_movement']['validation']['f1']:.3f} · Ridge {ridge_lm['f1']:.3f}</div></div>
<div class="card"><div class="muted">Pre-first-LM F1</div><div class="value">{rssm_pre['f1']:.3f}</div><div>Forecasts before movement is observed</div></div>
<div class="card"><div class="muted">Mean exact warning lead</div><div class="value">{episode['mean_exact_warning_lead_seconds']:.1f}s</div><div>{episode['progressing_detected_before_first_lm']}/{episode['progressing_episodes']} progressing test episodes detected</div></div>
</div>

<h2>1. What this system does</h2>
<p>This is not a current-flow IDS classifier. It first predicts how the network world will evolve, then maps those imagined futures to ATT&amp;CK behavior and lateral-movement risk.</p>
<svg class="diagram" viewBox="0 0 1120 190" role="img" aria-label="Cyber world model pipeline">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#9eb0c4"/></marker></defs>
<rect class="box" x="25" y="55" width="155" height="70" rx="10"/><text class="svgtext" x="102" y="85">Raw packets</text><text class="smalltext" x="102" y="106">PCAP telemetry</text>
<line class="arrow" x1="180" y1="90" x2="220" y2="90"/><rect class="box" x="225" y="55" width="155" height="70" rx="10"/><text class="svgtext" x="302" y="82">5-second graph</text><text class="smalltext" x="302" y="105">global + nodes + edges</text>
<line class="arrow" x1="380" y1="90" x2="420" y2="90"/><rect class="box2" x="425" y="55" width="170" height="70" rx="10"/><text class="svgtext" x="510" y="82">RSSM latent state</text><text class="smalltext" x="510" y="105">memory + uncertainty</text>
<line class="arrow" x1="595" y1="90" x2="635" y2="90"/><rect class="box2" x="640" y="55" width="175" height="70" rx="10"/><text class="svgtext" x="727" y="82">6-step imagination</text><text class="smalltext" x="727" y="105">next 30 seconds</text>
<line class="arrow" x1="815" y1="90" x2="855" y2="90"/><rect class="box3" x="860" y="35" width="230" height="110" rx="10"/><text class="svgtext" x="975" y="67">Future network state</text><text class="smalltext" x="975" y="90">edges · ATT&amp;CK · LM risk</text><text class="smalltext" x="975" y="112">source-target ranking</text>
</svg>

<h2>2. Data correction before modeling</h2>
<p>The first corpus produced an apparently excellent LM F1 of 0.963, but negative episodes ended sooner than attack episodes. A forbidden state-index diagnostic achieved AP 1.000. We rejected that score and generated equal-duration V2.</p>
<div class="grid">
<div class="card"><div class="metric"><span>Episodes</span><strong>{len(episodes)}</strong></div><div class="metric"><span>Capture duration</span><strong>{duration_min:.3f}–{duration_max:.3f}s</strong></div><div class="metric"><span>Five-second states</span><strong>{int(totals.num_states)}</strong></div><div class="metric"><span>Observations</span><strong>{int(totals.num_observations):,}</strong></div></div>
<div class="card"><div class="metric"><span>Train / validation / test episodes</span><strong>{split_episode_counts['train']} / {split_episode_counts['validation']} / {split_episode_counts['test']}</strong></div><div class="metric"><span>Train / validation / test sequences</span><strong>{sequence_counts['train']} / {sequence_counts['validation']} / {sequence_counts['test']}</strong></div><div class="metric"><span>ATT&amp;CK events</span><strong>{int(totals.num_ground_truth_events)}</strong></div><div class="metric"><span>LM events</span><strong>{int(totals.num_lateral_movement_events)}</strong></div></div>
</div>
<table><tr><th>Shortcut diagnostic</th><th>Old corpus AP</th><th>Equal-duration V2 AP</th><th>Interpretation</th></tr>
<tr><td>Forbidden context state index</td><td>{state_index_old['average_precision']:.3f}</td><td>{state_index_v2['average_precision']:.3f}</td><td>Dominant duration shortcut removed</td></tr>
<tr><td>Forbidden actor identity</td><td>0.498</td><td>{actor_v2['average_precision']:.3f}</td><td>One common attack IP is not predictive</td></tr></table>
<div class="callout good"><strong>Leakage controls:</strong> labels, ATT&amp;CK truth, scenario, actor, target, timestamps, state IDs, and future states are excluded from inference input. Scaling is fitted on training contexts only. Splits are whole episodes.</div>

<h2>3. RSSM architecture</h2>
<svg class="diagram" viewBox="0 0 1120 390" role="img" aria-label="RSSM prior posterior architecture">
<defs><marker id="arrow2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#9eb0c4"/></marker></defs>
<rect class="box" x="35" y="40" width="190" height="65" rx="10"/><text class="svgtext" x="130" y="68">Observation oₜ</text><text class="smalltext" x="130" y="88">141 observable features</text>
<line class="arrow" x1="225" y1="72" x2="285" y2="72"/><rect class="box" x="290" y="40" width="180" height="65" rx="10"/><text class="svgtext" x="380" y="68">MLP encoder</text><text class="smalltext" x="380" y="88">embedding 64</text>
<rect class="box2" x="290" y="175" width="180" height="70" rx="10"/><text class="svgtext" x="380" y="202">GRU memory hₜ</text><text class="smalltext" x="380" y="225">deterministic 64</text>
<line class="arrow" x1="380" y1="105" x2="380" y2="165"/><line class="arrow" x1="470" y1="210" x2="550" y2="210"/>
<rect class="box2" x="555" y="135" width="210" height="65" rx="10"/><text class="svgtext" x="660" y="162">Prior p(zₜ | hₜ)</text><text class="smalltext" x="660" y="182">imagines without observation</text>
<rect class="box" x="555" y="230" width="210" height="65" rx="10"/><text class="svgtext" x="660" y="257">Posterior q(zₜ | hₜ,oₜ)</text><text class="smalltext" x="660" y="277">training-time inference</text>
<line class="arrow" x1="470" y1="72" x2="550" y2="250"/><line class="arrow" x1="765" y1="168" x2="825" y2="210"/><line class="arrow" x1="765" y1="262" x2="825" y2="220"/>
<rect class="box3" x="830" y="175" width="215" height="70" rx="10"/><text class="svgtext" x="937" y="202">Gaussian zₜ</text><text class="smalltext" x="937" y="225">stochastic 16 · KL aligned</text>
<line class="arrow" x1="937" y1="245" x2="937" y2="305"/><rect class="box3" x="805" y="310" width="265" height="55" rx="10"/><text class="svgtext" x="937" y="337">Decoder + future heads</text><text class="smalltext" x="937" y="356">state · edge · semantics</text>
<path d="M830 210 C760 355,510 355,380 250" fill="none" stroke="#9eb0c4" stroke-width="2" marker-end="url(#arrow2)"/><text class="smalltext" x="610" y="365">z feeds the next recurrent transition</text>
</svg>
<div class="grid"><div class="card"><div class="metric"><span>Parameters</span><strong>{parameter_count:,}</strong></div><div class="metric"><span>Context</span><strong>3 × 5s</strong></div><div class="metric"><span>Prior rollout</span><strong>6 × 5s</strong></div><div class="metric"><span>Monte Carlo futures</span><strong>{uncertainty['mc_samples']}</strong></div></div>
<div class="card"><div class="metric"><span>Optimizer</span><strong>Adam 3e-4</strong></div><div class="metric"><span>Batch</span><strong>32</strong></div><div class="metric"><span>Candidate seeds</span><strong>7, 17, 27</strong></div><div class="metric"><span>Runtime</span><strong>~76s CPU</strong></div></div></div>
<p class="callout"><strong>Training terminology:</strong> telemetry reconstruction, prior prediction, and KL are self-supervised. LM/ATT&amp;CK/pair losses jointly shape the latent state, so the complete regime is <strong>hybrid</strong>, not wholly self-supervised.</p>
<table><tr><th>Seed</th><th>Best epoch</th><th>Validation selection score</th><th>Epochs run</th></tr>{candidate_rows}</table>

<h2>3a. Frozen, unfrozen, and KL ablations</h2>
<p>We independently tested whether telemetry-only pretraining carries security signal, whether semantic losses should adapt the RSSM, and what KL regularization contributes. Frozen and unfrozen conditions use the same internal linear heads and validation-only selection.</p>
<table><tr><th>Regime</th><th>State MAE ↓</th><th>Active MAE ↓</th><th>Edge AP ↑</th><th>LM F1 ↑</th><th>LM AP ↑</th><th>Pre-LM F1 ↑</th><th>Pair top-1 ↑</th><th>Spread</th></tr>{ablation_rows}</table>
<div class="grid"><div class="card"><h3>Representation result</h3><p>Frozen telemetry-pretrained features reach LM F1 0.757. Unfreezing raises it to 0.800, pre-first-LM F1 from 0.727 to 0.815, and pair top-1 from 0.071 to 0.214. Semantic adaptation helps, but edge AP falls from 0.240 to 0.210.</p></div>
<div class="card"><h3>KL result</h3><p>Zero-KL improves several point metrics, including active MAE 0.984 and edge AP 0.316, but mean rollout spread collapses from 0.076 to 0.025. It remains an ablation—not the selected model—and motivates validation-only KL tuning.</p></div></div>
<p class="callout bad"><strong>Small-test warning:</strong> only 14 LM-positive test windows exist, pair top-1 is based on 14 positive windows, and overlapping windows are correlated. Monte Carlo spread is not calibrated uncertainty.</p>

<h2>4. Primary future-state results</h2>
<p>Lower state MAE is better. All models use the same train-context normalization.</p><div class="card">{state_bars}</div>
<table><tr><th>State subset</th><th>RSSM</th><th>PCA/Ridge</th><th>Persistence</th></tr>
<tr><td>All future states</td><td><strong>{rssm_state['normalized_mae']:.3f}</strong></td><td>{ridge_state['normalized_mae']:.3f}</td><td>{rssm_state['persistence_normalized_mae']:.3f}</td></tr>
<tr><td>Active future states</td><td><strong>{rssm_state['active_normalized_mae']:.3f}</strong></td><td>{ridge_state['active_normalized_mae']:.3f}</td><td>{rssm_state['active_persistence_normalized_mae']:.3f}</td></tr>
<tr><td>Quiet future states</td><td><strong>{rssm_state['quiet_normalized_mae']:.3f}</strong></td><td>{ridge_state['quiet_normalized_mae']:.3f}</td><td>{rssm_state['quiet_persistence_normalized_mae']:.3f}</td></tr></table>
<table><tr><th>Horizon</th><th>RSSM all</th><th>Ridge all</th><th>Persistence all</th><th>RSSM active</th><th>Persistence active</th></tr>{horizon_rows}</table>
<div class="callout">RSSM loses to active-state persistence at +5s and +10s, then wins from +15s through +30s. Quiet states are 450/540 test future state-windows, so active metrics are shown explicitly.</div>

<h2>5. Future security interpretation</h2>
<div class="card">{lm_bars}</div>
<table><tr><th>Metric</th><th>RSSM</th><th>PCA/Ridge</th></tr>
<tr><td>Future LM precision / recall / F1</td><td>{rssm_lm['precision']:.3f} / {rssm_lm['recall']:.3f} / <strong>{rssm_lm['f1']:.3f}</strong></td><td>{ridge_lm['precision']:.3f} / {ridge_lm['recall']:.3f} / {ridge_lm['f1']:.3f}</td></tr>
<tr><td>Pre-first-LM F1</td><td><strong>{rssm_pre['f1']:.3f}</strong></td><td>{ridge_pre['f1']:.3f}</td></tr>
<tr><td>Future edge AP</td><td>{rssm['future_edge_presence']['test']['average_precision']:.3f}</td><td><strong>{ridge['future_edge_presence']['test']['average_precision']:.3f}</strong></td></tr>
<tr><td>LM pair top-1</td><td>{rssm['future_lateral_pair_ranking']['test']['top1_accuracy_any_true_lm_pair']:.3f}</td><td>{ridge['future_lateral_pair_ranking']['test']['top1_accuracy_any_true_lm_pair']:.3f}</td></tr></table>
<h3>RSSM test confusion matrix</h3><table><tr><th></th><th>Predicted no LM</th><th>Predicted LM</th></tr><tr><th>Actual no LM</th><td>{confusion[0][0]} true negatives</td><td>{confusion[0][1]} false positives</td></tr><tr><th>Actual LM</th><td>{confusion[1][0]} false negatives</td><td>{confusion[1][1]} true positives</td></tr></table>
<h3>ATT&amp;CK techniques from imagined futures</h3><table><tr><th>Technique</th><th>Validation F1</th><th>Test F1</th><th>Test AP</th></tr>{technique_rows}</table>

<h2>6. Episode alerts and uncertainty</h2>
<div class="grid"><div class="card"><div class="metric"><span>Progressing test episodes detected</span><strong>{episode['progressing_detected_before_first_lm']}/{episode['progressing_episodes']}</strong></div><div class="metric"><span>Mean exact lead</span><strong>{episode['mean_exact_warning_lead_seconds']:.1f}s</strong></div><div class="metric"><span>Negative episodes alerting</span><strong>{episode['nonprogressing_episodes_with_any_30s_false_alert']}/{episode['nonprogressing_episodes']}</strong></div><div class="metric"><span>Clean negatives</span><strong>ping + legitimate SSH</strong></div></div>
<div class="card"><div class="metric"><span>Mean state spread</span><strong>{uncertainty['mean_normalized_state_std']:.3f}</strong></div><div class="metric"><span>Active / quiet spread</span><strong>{uncertainty['active_mean_normalized_state_std']:.3f} / {uncertainty['quiet_mean_normalized_state_std']:.3f}</strong></div><div class="metric"><span>Spread-error correlation</span><strong>{uncertainty['state_std_absolute_error_correlation']:.3f}</strong></div><div class="metric"><span>Mean LM spread</span><strong>{uncertainty['mean_lm_probability_std']:.3f}</strong></div></div></div>
<p class="callout bad"><strong>Remaining false-alert problem:</strong> scan-only and failed-guessing episodes alert, while benign ping and legitimate SSH do not. The model recognizes dangerous precursors but does not always distinguish stalled activity from imminent movement.</p>

<h2>7. Selected held-out chronological replay</h2>
<p>Episode <code>{escape(replay['episode_id'])}</code>, context state {replay['context_end_state']}, {escape(replay['split'])} split.</p>
<p class="muted">{escape(replay['selection_disclosure'])}</p>
<div class="grid"><div class="card"><div class="muted">LM within 30 seconds</div><div class="value">{pct(replay['future_lateral_movement_probability'])} ± {pct(replay['future_lateral_movement_probability_std'])}</div><div>Threshold {pct(replay['alert_threshold_selected_on_validation'])} · no LM previously observed</div></div>
<div class="card"><div class="muted">Exact first event</div><div class="value">+{replay['first_actual_lateral_movement_event_lead_seconds']:.1f}s</div><div>Top pair {escape(replay['predicted_lateral_pair_ranking'][0]['source'])} → {escape(replay['predicted_lateral_pair_ranking'][0]['target'])} at {pct(replay['predicted_lateral_pair_ranking'][0]['probability'])}</div></div></div>
<h3>Imagined ATT&amp;CK probabilities</h3><div class="card">{replay_techniques}</div>
<h3>Observed 15-second input</h3><table><tr><th>State</th><th>UTC time</th><th>Flows</th><th>New edges</th><th>SYN</th><th>Bytes</th></tr>{context_rows}</table>
<h3>Imagined versus actual future</h3><table><tr><th>Lead</th><th>Flows pred/actual</th><th>New edges pred/actual</th><th>Pairs pred/actual</th><th>Spread</th><th>Actual ATT&amp;CK</th></tr>{future_rows}</table>
<h3>Top pair candidates</h3><table><tr><th>Pair</th><th>Score</th></tr>{replay_pairs}</table><p>Actual pair: <strong>{escape(', '.join(replay['actual_lateral_pairs_within_horizon']))}</strong></p>

<h2>8. Interpretation and limitations</h2>
<div class="grid"><div class="card"><h3>What was demonstrated</h3><ul><li>Chronological graph-state modeling rather than shuffled flow classification.</li><li>Posterior latent inference followed by genuine six-step prior imagination.</li><li>Better state and LM forecasting than Ridge on equal-duration V2.</li><li>Useful, nonzero stochastic spread related to error.</li><li>Future ATT&amp;CK interpretation without labels as input.</li></ul></div>
<div class="card"><h3>What was not demonstrated</h3><ul><li>Enterprise or cross-domain generalization.</li><li>Reliable exact LM source-target ranking.</li><li>Calibrated operational uncertainty.</li><li>Purely self-supervised downstream performance.</li><li>Action-conditioned firewall counterfactuals.</li></ul></div></div>
<p>Only 24 controlled episodes and 12 LM events exist. Overlapping windows are correlated. The flattened fixed-slot encoder is not graph permutation equivariant. The selected replay is illustrative; aggregate metrics—not this frame—must support model claims.</p>

<h2>9. Next experiment</h2>
<ol><li>Train RSSM using only telemetry reconstruction, prediction, and KL.</li><li>Freeze its latent dynamics and train downstream security heads separately.</li><li>Compare against the current joint-loss model to determine whether useful security semantics emerge from self-supervised dynamics.</li><li>Add matched scan/guessing non-progression episodes, then improve edge and pair decoders.</li><li>Collect explicit intervention/no-intervention episodes before adding defensive action conditioning.</li></ol>

<p class="foot">Generated by <code>scripts/17_build_rssm_report.py</code> from local verified metrics. No external assets or runtime network resources are required. Source details: <code>docs/RSSM_RESULTS.md</code> and <code>docs/RSSM_ABLATIONS.md</code>.</p>
</main></body></html>"""

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"report -> {output}")
    print(f"size -> {output.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
