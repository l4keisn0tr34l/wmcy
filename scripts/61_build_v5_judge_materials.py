#!/usr/bin/env python3
"""Build standalone V5 senior report and judge deck from sealed saved metrics only.

This script performs no model loading, inference, fitting, threshold search, or test
export. It reads immutable JSON/CSV summaries and formats two self-contained HTML
files plus compatibility copies.
"""
from __future__ import annotations

import csv
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "outputs/mvp_v5/sealed_test/report.json"
PROVENANCE_JSON = ROOT / "outputs/mvp_v5/sealed_test/provenance.json"
CORPUS_JSON = ROOT / "outputs/mvp_v5/corpus_audit/audit.json"
PLAN_CSV = ROOT / "configs/mvp_v5_episode_plan.csv"
OUT = ROOT / "outputs/mvp_v5/report"
DOWNLOADS = Path.home() / "Downloads"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def n(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pc(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def metric_card(label: str, value: str, note: str, cls: str = "") -> str:
    return (f'<article class="metric-card {cls}"><div class="eyebrow">{escape(label)}</div>'
            f'<div class="big">{escape(value)}</div><p>{escape(note)}</p></article>')


def bar(label: str, value: float, maximum: float, color: str, shown: str | None = None) -> str:
    width = max(1, min(100, 100 * value / maximum))
    return (f'<div class="bar-row"><span>{escape(label)}</span><div class="track">'
            f'<div class="fill" style="width:{width:.2f}%;background:{color}"></div></div>'
            f'<strong>{escape(shown or n(value))}</strong></div>')


def horizon_svg(model: list[float], persistence: list[float]) -> str:
    width, height, pad = 760, 270, 52
    vmax = max(model + persistence) * 1.12
    def points(values: list[float]) -> str:
        return " ".join(f"{pad+i*(width-2*pad)/5:.1f},{height-pad-v/vmax*(height-2*pad):.1f}"
                        for i, v in enumerate(values))
    ticks = "".join(
        f'<line x1="{pad}" y1="{height-pad-y*(height-2*pad)/4}" x2="{width-pad}" y2="{height-pad-y*(height-2*pad)/4}" class="gridline"/>'
        f'<text x="{pad-10}" y="{height-pad-y*(height-2*pad)/4+4}" class="tick" text-anchor="end">{vmax*y/4:.2f}</text>'
        for y in range(5)
    )
    xlabels = "".join(f'<text x="{pad+i*(width-2*pad)/5}" y="{height-18}" class="tick" text-anchor="middle">+{5*(i+1)}s</text>' for i in range(6))
    return f'''<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="State MAE by forecast horizon">
{ticks}{xlabels}<polyline points="{points(persistence)}" class="line persist"/><polyline points="{points(model)}" class="line model"/>
<text x="580" y="28" class="legend model-text">● world model</text><text x="580" y="48" class="legend persist-text">● persistence</text></svg>'''


class _StandaloneAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.external_attributes: list[tuple[str, str, str | None]] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in {"src", "href", "action"}:
                self.external_attributes.append((tag, key, value))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def validate_html(path: Path, expected_slides: int) -> None:
    source = path.read_text()
    parser = _StandaloneAudit()
    parser.feed(source)
    if parser.external_attributes:
        raise ValueError(f"external resource/link in {path}: {parser.external_attributes}")
    text = " ".join(parser.text).lower()
    for required in ("0 / 6", "3 / 6", "7 / 10", "8 / 8", "not a pure",
                     "not establish", "no test threshold tuning", "enterprise generalization"):
        if required not in text:
            raise ValueError(f"missing disclosure {required!r} in {path}")
    if source.count('<html') != 1 or source.count('</html>') != 1:
        raise ValueError(f"invalid HTML document boundary: {path}")
    if source.count('class="slide" data-slide=') != expected_slides:
        raise ValueError(f"unexpected slide count: {path}")


def common_css() -> str:
    return """
:root{--ink:#10243a;--muted:#5a6b7d;--paper:#f7fafc;--panel:#fff;--line:#d7e2ec;--navy:#09233f;--blue:#0877b9;--cyan:#11a9c7;--green:#15876b;--amber:#c67a05;--red:#b63b50;--soft:#eef5f9}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:34px}h1{font-size:clamp(2.2rem,5vw,4.4rem);line-height:1.02;letter-spacing:-.04em;margin:.18em 0}h2{font-size:2rem;letter-spacing:-.025em;border-bottom:2px solid var(--line);padding-bottom:.35em;margin-top:2.2em}h3{color:var(--blue)}p{max-width:82ch}.lead{font-size:1.22rem;color:var(--muted)}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:800;color:var(--blue)}.pill{display:inline-block;padding:.3rem .7rem;margin:.15rem;border-radius:999px;background:#dff4fa;color:#075985;font-weight:700;font-size:.82rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:15px;margin:18px 0}.metric-card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:19px;box-shadow:0 10px 30px #17324d0c}.metric-card .big{font-size:2rem;font-weight:850;line-height:1.1;margin:.22em 0;color:var(--green)}.metric-card.warn .big{color:var(--amber)}.metric-card.fail .big{color:var(--red)}.metric-card p{font-size:.9rem;color:var(--muted);margin:.5em 0 0}.callout{padding:16px 20px;margin:18px 0;background:#fff8e8;border-left:5px solid var(--amber);border-radius:8px}.callout.good{background:#eaf8f2;border-color:var(--green)}.callout.fail{background:#fff0f2;border-color:var(--red)}table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;display:table;margin:14px 0}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--navy);color:#fff;font-size:.86rem}tr:last-child td{border-bottom:0}.primary{font-weight:800;color:#096b55}.bar-row{display:grid;grid-template-columns:150px 1fr 66px;align-items:center;gap:10px;margin:10px 0}.track{height:14px;background:#e1e9ef;border-radius:999px;overflow:hidden}.fill{height:100%;border-radius:999px}.chart{width:100%;max-height:330px;background:#fff;border:1px solid var(--line);border-radius:12px}.gridline{stroke:#dbe5ed;stroke-width:1}.tick{font:12px system-ui;fill:#607286}.line{fill:none;stroke-width:4;stroke-linejoin:round;stroke-linecap:round}.line.model{stroke:var(--green)}.line.persist{stroke:var(--amber)}.legend{font:12px system-ui;font-weight:700}.model-text{fill:var(--green)}.persist-text{fill:var(--amber)}code{background:#e8f0f5;padding:.12em .35em;border-radius:4px}.small{font-size:.88rem;color:var(--muted)}.footer{margin-top:4em;padding-top:1em;border-top:1px solid var(--line);color:var(--muted);font-size:.84rem}.pipeline{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;align-items:stretch}.pipe{background:var(--navy);color:#fff;border-radius:10px;padding:12px;text-align:center;font-weight:750}.pipe span{display:block;font-size:.74rem;font-weight:400;color:#bad4e8;margin-top:4px}.arrow{display:none}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.check li,.cross li{margin:.55em 0}.check li::marker{color:var(--green)}.cross li::marker{color:var(--red)}@media(max-width:760px){main{padding:18px}.pipeline{grid-template-columns:1fr 1fr}.two{grid-template-columns:1fr}table{display:block;overflow-x:auto}.bar-row{grid-template-columns:105px 1fr 56px}}
@media print{body{background:#fff}main{max-width:none;padding:0}.metric-card,.panel{box-shadow:none}a{color:inherit}}
"""


def build_report(r: dict, provenance: dict, corpus: dict, splits: dict[str, int]) -> str:
    action = r["action"]["models"]["scratch"]
    apersist = r["action"]["persistence"]
    passive = r["passive"]["models"]["scratch"]
    ppersist = r["passive"]["persistence"]
    aligned = r["passive_on_action_aligned"]["models"]["scratch"]
    intent = r["test_only_intent_probe"]
    warnings = r["episode_warning_primary"]
    low = warnings["frozen_validation_threshold"]["episodes"]
    positives = [x for x in low if x["has_completed_lm"]]
    negatives = [x for x in low if not x["has_completed_lm"]]
    detected = [x for x in positives if x["within_30s_pre_first_detection"]]
    false_alerts = [x for x in negatives if x["pre_first_or_negative_alert"]]
    action_improve = 1 - action["state"]["overall_mae"] / apersist["state"]["overall_mae"]
    passive_improve = 1 - passive["future_graph"]["expected_forecast_mae"] / ppersist["state"]["overall_mae"]
    oracle_gain = passive["future_graph"]["oracle_gain_over_expected_mae"]
    threshold = passive["lm"]["frozen_validation_threshold"]
    profiles = action["by_background_profile"]
    pprofiles = passive["by_background_profile"]
    attack = passive["attack_techniques"]["techniques"]
    action_rows = "".join(
        f'<tr class="{"primary" if name=="scratch" else ""}"><td>{name.title()}{" — selected" if name=="scratch" else ""}</td>'
        f'<td>{m["state"]["overall_mae"]:.6f}</td><td>{m["state"]["active_mae"]:.6f}</td><td>{m["future_edge_average_precision"]:.6f}</td>'
        f'<td>{m["lm"]["fixed_0_5"]["f1"]:.3f}</td><td>{m["counterfactual"]["factual_lower_state_error_count"]}/8</td></tr>'
        for name, m in r["action"]["models"].items())
    passive_rows = "".join(
        f'<tr class="{"primary" if name=="scratch" else ""}"><td>{name.title()}{" — selected" if name=="scratch" else ""}</td>'
        f'<td>{m["future_graph"]["expected_forecast_mae"]:.6f}</td><td>{m["future_graph"]["oracle_best_branch_mae"]:.6f}</td>'
        f'<td>{m["future_graph"]["future_edge"]["average_precision"]:.6f}</td><td>{m["lm"]["fixed_0_5"]["average_precision"]:.6f}</td>'
        f'<td>{m["lm"]["fixed_0_5"]["f1"]:.3f}</td><td>{m["pair"]["micro_average_precision"]:.6f}</td></tr>'
        for name, m in r["passive"]["models"].items())
    profile_rows = "".join(
        f'<tr><td>{name.title()}</td><td>{profiles[name]["state_mae"]:.6f}</td><td>{pprofiles[name]["state_mae"]:.6f}</td></tr>'
        for name in ("quiet", "web", "mixed", "admin"))
    attack_rows = "".join(
        f'<tr><td>{tech}</td><td>{vals["positive_samples"]}</td><td>{"undefined" if vals["average_precision"] is None else n(vals["average_precision"], 6)}</td></tr>'
        for tech, vals in attack.items())
    lead_rows = "".join(f'<li><code>{x["episode_id"]}</code>: {x["lead_seconds_to_successful_action_start"]:.6f}s</li>' for x in detected)
    horizon = horizon_svg(passive["expected_state"]["mae_by_5s_horizon"], ppersist["state"]["mae_by_5s_horizon"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictive Cyber World Model — V5 Senior Report</title><style>{common_css()}</style></head><body><main>
<div><span class="pill">SEALED V5 TEST</span><span class="pill">WORLD MODEL</span><span class="pill">STANDALONE REPORT</span></div>
<h1>Predictive Cyber<br>World Model</h1>
<p class="lead">Observe 15 seconds of graph telemetry; forecast the next 30 seconds of network state, communication edges, security behavior, and action-conditioned alternatives. This is a dynamics study—not a current-flow IDS scorecard.</p>
<div class="grid">
{metric_card("Action state MAE", n(action['state']['overall_mae']), f"{pc(action_improve)} below persistence ({apersist['state']['overall_mae']:.3f})")}
{metric_card("Passive state MAE", n(passive['future_graph']['expected_forecast_mae']), f"{pc(passive_improve)} below persistence ({ppersist['state']['overall_mae']:.3f})")}
{metric_card("Factual action preferred", "8 / 8", "Lower state error than the opposite action")}
{metric_card("Passive warning @0.5", "0 / 6", "Progressing episodes detected", "fail")}
{metric_card("Low-threshold tradeoff", "3 / 6 vs 7 / 10", "Detected progressions vs false-alerted nonprogressions", "warn")}
</div>
<div class="callout fail"><strong>Executive verdict:</strong> learned dynamics beat persistence in this controlled lab, and an already-chosen action improves the factual future. Passive branches, however, add little candidate coverage and do not provide a useful completed-lateral-movement warning rule.</div>

<h2>1. Research question and system boundary</h2>
<p>The objective is <strong>P(S<sub>t+1:t+6</sub> | S<sub>t-2:t</sub>)</strong>: learn how a network/security state evolves. Security heads interpret predicted futures; they do not replace dynamics prediction.</p>
<div class="pipeline"><div class="pipe">PCAP<span>observable packets</span></div><div class="pipe">5s graph<span>global + nodes + edges</span></div><div class="pipe">Graph encoder<span>shared host/pair operations</span></div><div class="pipe">Latent dynamics<span>six prior steps</span></div><div class="pipe">Future graph<span>345 features +20 edges</span></div><div class="pipe">Security meaning<span>ATT&amp;CK · LM · pair</span></div></div>
<div class="two"><div class="panel"><h3>Observable input</h3><ul class="check"><li>3 dense five-second states</li><li>15 global +5×18 node +20×12 edge features =345</li><li>Defender action only when chosen before consequences</li><li>Anonymous shared host/pair slots</li></ul></div><div class="panel"><h3>Deliberately excluded</h3><ul class="cross"><li>Scenario, seed, roles, label, ATT&amp;CK truth</li><li>Future state/edges and completed-LM truth</li><li>Action result and controller intent</li><li>Test-fitted scaling, thresholds, or candidates</li></ul></div></div>

<h2>2. V5 evidence base and seal</h2>
<div class="grid">{metric_card("Episodes", str(corpus['episodes']), "40 independently captured paired families")}{metric_card("Split", f"{splits['train']} / {splits['validation']} / {splits['test']}", "Train / validation / test episodes")}{metric_card("Capture", "150s each", "29 complete states; 145s covered")}{metric_card("Packet drops", str(corpus['packet_drops_total']), "Across all 80 captures")}</div>
<p>Five hosts, four balanced background profiles, scan-based and direct-credential paths, permit/block interventions, benign controls, and a test-only intent probe were captured on one isolated flat topology. Paired captures share family metadata but are independently generated—not packet-identical counterfactuals.</p>
<table><tr><th>Evaluation scope</th><th>Samples</th><th>Independent unit warning</th></tr><tr><td>Action</td><td>8 contexts /8 episodes /4 families</td><td>Small deterministic intervention set</td></tr><tr><td>Passive</td><td>336 windows /16 episodes /8 families</td><td>Overlapping windows are correlated</td></tr><tr><td>Action-aligned passive</td><td>8 contexts</td><td>Same target arrays, different training regimes</td></tr></table>
<div class="callout good"><strong>Methodological control:</strong> family-disjoint splits; scaler fitted only on440 deduplicated train-context states; models, seeds, budgets, selections, thresholds, runtime, and evaluator hashes frozen before one test access. The permanent attempt was consumed successfully at {escape(provenance['started_utc'])}.</div>

<h2>3. Action-conditioned dynamics</h2>
<p>Action candidates receive the already-chosen permit/block action and directed pair before rolling forward. Scratch was selected on validation and stays primary.</p>
<table><tr><th>Initialization</th><th>State MAE↓</th><th>Active MAE↓</th><th>Edge AP↑</th><th>LM F1 @.5</th><th>Factual wins</th></tr>{action_rows}<tr><td>Persistence</td><td>{apersist['state']['overall_mae']:.6f}</td><td>{apersist['state']['active_mae']:.6f}</td><td>{apersist['future_edge_average_precision']:.6f}</td><td>—</td><td>—</td></tr></table>
<div class="panel">{bar("Scratch state MAE", action['state']['overall_mae'], .60, "#15876b")}{bar("Persistence state MAE", apersist['state']['overall_mae'], .60, "#c67a05")}{bar("Scratch edge AP", action['future_edge_average_precision'], 1, "#0877b9")}</div>
<div class="callout"><strong>Do not overclaim perfect semantics.</strong> LM AP, pair AP, and pair top-1 are1 on four permits/four blocks because action type and pair are explicit inputs and the lab outcome is deterministic. Edge AP is only{action['future_edge_average_precision']:.3f}; this is not perfect graph forecasting or a causal-policy-value experiment.</div>

<h2>4. Passive two-future model</h2>
<p>The passive model emits no-LM and completed-LM graph candidates, with context-only branch weights. It does not receive the action or future outcome.</p>
<table><tr><th>Initialization</th><th>Expected MAE↓</th><th>Oracle MAE↓</th><th>Edge AP↑</th><th>LM AP↑</th><th>F1 @.5</th><th>Pair AP↑</th></tr>{passive_rows}<tr><td>Persistence</td><td>{ppersist['state']['overall_mae']:.6f}</td><td>—</td><td>{ppersist['future_edge_average_precision']:.6f}</td><td>—</td><td>—</td><td>—</td></tr></table>
<div class="grid">{metric_card("Oracle gain", f"{oracle_gain:.6f}", f"Only {pc(oracle_gain/passive['future_graph']['expected_forecast_mae'],2)} of point error")}{metric_card("LM-positive windows", "36 / 336", "F1@.5=0; model predicts no positives", "fail")}{metric_card("Pair top-1", "1 / 36", "Future LM source-target localization", "fail")}</div>
{horizon}
<p class="small">State MAE falls below persistence over the horizon, but this does not rescue passive security warning. Oracle picks the best candidate after seeing truth and is coverage—not deployable accuracy. A convex weighted prediction can even beat either branch.</p>

<h2>5. Warning failure is a headline result</h2>
<table><tr><th>Prespecified threshold</th><th>Progressing episodes detected</th><th>Nonprogressing episodes falsely alerted</th></tr><tr><td>Primary0.5</td><td>0 /6</td><td>0 /10</td></tr><tr><td>Frozen validation {threshold:.10f}</td><td>{len(detected)} /6</td><td>{len(false_alerts)} /10</td></tr></table>
<div class="two"><div class="panel"><h3>Detected secondary-threshold leads</h3><ul>{lead_rows}</ul><p class="small">Lead is to controller-recorded successful SSH action start, not independently timed compromise completion.</p></div><div class="panel"><h3>Window-level tradeoff</h3><p><strong>Precision .209 · recall .250 · F1 .228</strong></p><p>34 false-positive windows. Lowering the threshold did not produce an operationally useful tradeoff.</p></div></div>
<div class="callout fail"><strong>Preserved negative result:</strong> the passive gate recognizes some risky precursor contexts but cannot reliably separate later progression from stopping, benign background, or legitimate SSH. No test threshold tuning or adjustment is allowed.</div>

<h2>6. Intent probe: diagnostic, not hidden-intent recovery</h2>
<p>At one metadata-frozen pre-SSH cutoff in four malicious/legitimate pairs: AP {intent['classification']['fixed_0_5']['average_precision']:.3f}, ROC-AUC {intent['classification']['fixed_0_5']['roc_auc']:.3f}, and mean malicious-minus-legitimate probability {intent['mean_malicious_minus_legitimate_probability']:.6f}. Primary F1 is0; the frozen low-threshold F1 is{intent['classification']['at_frozen_validation_threshold']['f1']:.1f}.</p>
<div class="grid">{metric_card("Primary cutoff", "0 / 4", "Malicious intent episodes detected @0.5", "fail")}{metric_card("Frozen low cutoff", "1 / 4", "Malicious detected;0/4 legitimate alerted at this one cutoff", "warn")}{metric_card("Matched probability gap", ".0080", "Mean across four independent pairs", "warn")}</div>
<p>Two pair differences are about ±.00005. Captures are independent and tiny; ranking metrics and one favorable cutoff do not establish latent attacker-intent inference. The broader episode analysis still falsely alerts7/10 nonprogressing episodes at the low threshold.</p>

<h2>7. Same-context action/passive comparison</h2>
<table><tr><th>Metric on the exact same8 targets</th><th>Action scratch</th><th>Passive scratch</th><th>Interpretation</th></tr><tr><td>Overall state MAE↓</td><td class="primary">{action['state']['overall_mae']:.6f}</td><td>{aligned['future_graph']['expected_forecast_mae']:.6f}</td><td>Small overall action advantage (~1.8%)</td></tr><tr><td>Active-feature MAE↓</td><td>{action['state']['active_mae']:.6f}</td><td class="primary">{aligned['expected_state']['active_mae']:.6f}</td><td>Passive is better on active entries</td></tr><tr><td>Edge AP↑</td><td>{action['future_edge_average_precision']:.6f}</td><td class="primary">{aligned['future_graph']['future_edge']['average_precision']:.6f}</td><td>Nearly tied; passive slightly higher</td></tr><tr><td>LM F1 @.5</td><td class="primary">1</td><td>0</td><td>Explicit deterministic action carries outcome information</td></tr></table>
<div class="callout"><strong>Not a pure ablation:</strong> the two models have different training cohorts and objectives. The shared scaler and exact targets make numbers comparable, but do not isolate the causal contribution of conditioning.</div>

<h2>8. ATT&amp;CK, backgrounds, and transfer</h2>
<div class="two"><div><h3>Passive ATT&amp;CK AP</h3><table><tr><th>Technique</th><th>Positive windows</th><th>AP</th></tr>{attack_rows}</table></div><div><h3>Scratch state MAE by profile</h3><table><tr><th>Profile</th><th>Action</th><th>Passive</th></tr>{profile_rows}</table></div></div>
<p>Action ATT&amp;CK micro AP is1, but all eight horizons contain attempted SSH (`T1021.004`) and no earlier scan/guess truth, so per-technique AP is undefined. Blocked SSH is attempted technique—not completed lateral movement.</p>
<p>Friday-PCAP initialization does not improve scratch action or passive dynamics under the fixed budget. V4 initialization has slightly lower passive point MAE, but scratch remains the model selected by the frozen validation composite. Test results do not reopen selection.</p>

<h2>9. What is and is not supported</h2>
<div class="two"><div class="panel"><h3>Supported observations</h3><ul class="check"><li>Chronological graph forecasting beats persistence on aggregate state MAE.</li><li>Chosen-action scratch predicts its factual future better than the opposite action8/8.</li><li>Five-host shared graph operations preserve the intended structural model design.</li><li>One-shot, family-disjoint, leakage-controlled evaluation completed with preserved provenance.</li></ul></div><div class="panel"><h3>Not demonstrated</h3><ul class="cross"><li>Useful passive LM warning or robust pair localization.</li><li>Calibrated multimodal uncertainty or hidden-intent recovery.</li><li>Enterprise generalization or unseen-topology generalization.</li><li>Causal defense-policy value or general firewall effects.</li><li>A public-pretraining benefit in V5.</li></ul></div></div>
<p>Passive Brier/ECE (.094/.038) uses correlated windows; intent Brier is.420 on eight samples. These diagnostics do not establish deployment calibration. V5 is one synthetic flat five-host topology and one executed SSH hop.</p>

<h2>10. Defensible conclusion and next experiment</h2>
<div class="callout good"><strong>Conclusion:</strong> V5 provides evidence for a controlled-lab predictive graph world model—not an enterprise detector. It predicts aggregate future dynamics better than persistence and responds coherently to a chosen intervention. Its passive alternative futures do not yet deliver reliable advance compromise warning.</div>
<ol><li>Design a new prospective corpus with unseen topologies, variable host counts, multiple lateral paths, richer actions, and host telemetry.</li><li>Increase independent episode/family counts; stop treating overlapping windows as independent evidence.</li><li>Train/evaluate intervention conditioning as a controlled ablation with identical cohorts and objectives.</li><li>Evaluate probabilistic calibration and multimodal coverage only on independently captured validation episodes.</li><li>Keep V5 sealed: use saved artifacts for displays; never retune or rerun inference against this test.</li></ol>

<h2>Audit identity</h2><table><tr><th>Artifact</th><th>SHA-256</th></tr><tr><td>Evaluation freeze</td><td><code>{escape(r['evaluation_freeze_sha256'])}</code></td></tr><tr><td>Sealed report</td><td><code>{escape(provenance['artifact_sha256_before_provenance']['report.json'])}</code></td></tr><tr><td>Saved predictions</td><td><code>{escape(r['predictions_sha256'])}</code></td></tr><tr><td>Shared scaler</td><td><code>{escape(r['scaler_sha256'])}</code></td></tr></table>
<p class="footer">Generated deterministically by <code>scripts/61_build_v5_judge_materials.py</code> from saved sealed metrics and audit metadata. No model or prediction code is invoked. No external assets, fonts, analytics, or network resources. Full methodology: <code>docs/V5_FINAL_REVIEW.md</code>. Full results: <code>docs/V5_TEST_RESULTS.md</code>.</p>
</main></body></html>'''


def slide(num_: int, title: str, body: str, kicker: str = "") -> str:
    return f'<section class="slide" data-slide="{num_}"><div class="slide-inner"><div class="kicker">{escape(kicker)}</div><h2>{title}</h2>{body}</div><div class="slide-no">{num_:02d}</div></section>'


def build_deck(r: dict, provenance: dict, corpus: dict, splits: dict[str, int]) -> str:
    a = r["action"]["models"]["scratch"]; ap = r["action"]["persistence"]
    p = r["passive"]["models"]["scratch"]; pp = r["passive"]["persistence"]
    al = r["passive_on_action_aligned"]["models"]["scratch"]
    intent = r["test_only_intent_probe"]
    content = []
    content.append(slide(1, "Predict the network’s future,<br><em>not just today’s label.</em>", '''<p class="deck-lead">A graph world model observes <strong>15 seconds</strong> and forecasts the next <strong>30 seconds</strong>—including alternative and chosen-action futures.</p><div class="deck-pills"><span>5 hosts</span><span>345 features/state</span><span>6-step rollout</span><span>sealed test</span></div><p class="bottom-note">V5 controlled-lab evidence · one-shot evaluation · honest failures retained</p>''', "CYBER WORLD MODEL · V5"))
    content.append(slide(2, "The research question", '''<div class="formula">P(S<sub>t+1:t+6</sub> | S<sub>t−2:t</sub>)</div><div class="flow"><div>Packets<small>observable only</small></div><b>→</b><div>Dynamic graph<small>global · nodes · edges</small></div><b>→</b><div>Latent dynamics<small>future rollout</small></div><b>→</b><div>Interpretation<small>ATT&amp;CK · LM · pair</small></div></div><div class="statement"><strong>Not:</strong> current flow → attack label.<br><strong>Core evidence:</strong> future-state and future-edge accuracy.</div>''', "01 · OBJECTIVE"))
    content.append(slide(3, "A leakage-controlled evidence base", f'''<div class="deck-grid four"><div class="deck-card"><strong>{corpus['episodes']}</strong><span>episodes</span></div><div class="deck-card"><strong>40</strong><span>paired families</span></div><div class="deck-card"><strong>150s</strong><span>per capture</span></div><div class="deck-card"><strong>0</strong><span>packet drops</span></div></div><ul class="deck-list"><li>Family-disjoint {splits['train']} / {splits['validation']} / {splits['test']} episode split</li><li>Scaler fitted on440 deduplicated training-context states only</li><li>Labels, roles, scenario, future state, intent, and action result excluded from inference</li><li>Models, thresholds, runtime, evaluator and artifact hashes frozen before one test access</li></ul><div class="warning">Pairs are independently captured—not packet-identical counterfactuals.</div>''', "02 · METHOD"))
    content.append(slide(4, "Two forecasting tracks", '''<div class="split"><div class="track-card action"><h3>Chosen-action future</h3><p>Context + action type + directed pair</p><div class="mini-flow">permit / block → 30s graph future</div><p class="fine">Action is supplied only because the defender selected it before consequences.</p></div><div class="track-card passive"><h3>Passive alternatives</h3><p>Context only</p><div class="mini-flow">no-LM branch + LM branch</div><p class="fine">Branch weights see no action, outcome, or future truth.</p></div></div><p class="bottom-note">Scratch selected on validation for each track; Friday and V4 retained as fixed diagnostics.</p>''', "03 · MODEL"))
    content.append(slide(5, "Action-conditioned dynamics beat persistence", f'''<div class="deck-grid three"><div class="deck-card good"><strong>{a['state']['overall_mae']:.3f}</strong><span>state MAE<br>vs {ap['state']['overall_mae']:.3f} persistence</span></div><div class="deck-card"><strong>{a['future_edge_average_precision']:.3f}</strong><span>edge AP<br>modest, not perfect</span></div><div class="deck-card good"><strong>8 / 8</strong><span>factual action futures<br>lower error than opposite</span></div></div><div class="comparison"><div style="width:{100*a['state']['overall_mae']/.60:.1f}%">world model {a['state']['overall_mae']:.3f}</div><div class="base" style="width:{100*ap['state']['overall_mae']/.60:.1f}%">persistence {ap['state']['overall_mae']:.3f}</div></div><div class="warning">LM and pair scores are perfect because permit/block and pair are explicit, with deterministic lab outcomes. This is not general causal-policy evidence.</div>''', "04 · RESULT A"))
    content.append(slide(6, "Passive dynamics improve.<br>Passive warning does not.", f'''<div class="deck-grid four"><div class="deck-card good"><strong>{p['future_graph']['expected_forecast_mae']:.3f}</strong><span>state MAE<br>vs {pp['state']['overall_mae']:.3f} persistence</span></div><div class="deck-card"><strong>{p['future_graph']['future_edge']['average_precision']:.3f}</strong><span>edge AP</span></div><div class="deck-card bad"><strong>0</strong><span>LM F1 @0.5</span></div><div class="deck-card bad"><strong>1 / 36</strong><span>LM pair top-1</span></div></div><div class="oracle"><strong>Oracle branch gain: {p['future_graph']['oracle_gain_over_expected_mae']:.4f}</strong><span>Only ~0.43% of point error. Oracle sees truth after the fact; it is candidate coverage, not deployable accuracy.</span></div>''', "05 · RESULT B"))
    content.append(slide(7, "The warning tradeoff fails", '''<div class="tradeoff"><div><span>Primary threshold 0.5</span><strong>0 / 6</strong><small>progressions detected</small><strong>0 / 10</strong><small>nonprogressions falsely alerted</small></div><div class="bad-panel"><span>Frozen low threshold .1003</span><strong>3 / 6</strong><small>progressions detected</small><strong>7 / 10</strong><small>nonprogressions falsely alerted</small></div></div><p class="deck-lead">Lowering the already-frozen threshold creates many false alerts without recovering half the progressing episodes.</p><div class="warning">Lead is to controller-recorded successful SSH start—not independently measured compromise completion. No test threshold tuning.</div>''', "06 · PRESERVED FAILURE"))
    content.append(slide(8, "The intent probe is not intent recovery", f'''<div class="deck-grid four"><div class="deck-card"><strong>8</strong><span>episodes /4 pairs</span></div><div class="deck-card bad"><strong>0</strong><span>F1 @0.5</span></div><div class="deck-card"><strong>{intent['classification']['fixed_0_5']['average_precision']:.3f}</strong><span>AP on tiny sample</span></div><div class="deck-card"><strong>{intent['mean_malicious_minus_legitimate_probability']:.4f}</strong><span>mean probability gap</span></div></div><ul class="deck-list"><li>Frozen low threshold detects1/4 malicious and alerts0/4 legitimate at this one cutoff.</li><li>Two paired probability differences are approximately±0.00005.</li><li>Independent captures do not control away background variation.</li></ul><div class="warning">A favorable-looking ranking statistic does not establish hidden-intent recovery.</div>''', "07 · BOUNDARY"))
    content.append(slide(9, "Action helps overall—<br>but not every graph metric", f'''<table class="deck-table"><tr><th>Same8 contexts</th><th>Action</th><th>Passive</th></tr><tr><td>State MAE↓</td><td class="win">{a['state']['overall_mae']:.3f}</td><td>{al['future_graph']['expected_forecast_mae']:.3f}</td></tr><tr><td>Active-feature MAE↓</td><td>{a['state']['active_mae']:.3f}</td><td class="win">{al['expected_state']['active_mae']:.3f}</td></tr><tr><td>Edge AP↑</td><td>{a['future_edge_average_precision']:.3f}</td><td class="win">{al['future_graph']['future_edge']['average_precision']:.3f}</td></tr><tr><td>LM F1 @.5</td><td class="win">1</td><td>0</td></tr></table><div class="warning">Comparable targets and scaler; different training cohorts/objectives. Descriptive—not a pure conditioning ablation.</div>''', "08 · ABLATION CAUTION"))
    content.append(slide(10, "What survives critical review", '''<div class="split"><div><h3 class="yes">Supported</h3><ul class="deck-list"><li>Temporal graph world modeling, not shuffled IDS classification</li><li>Aggregate state dynamics beat persistence</li><li>Chosen actions alter predicted factual futures coherently</li><li>One-shot leakage-controlled evaluation with preserved provenance</li></ul></div><div><h3 class="no">Not supported</h3><ul class="deck-list"><li>Useful passive compromise warning</li><li>Calibrated multimodal uncertainty</li><li>Hidden-intent or enterprise generalization</li><li>Unseen topology or causal policy value</li><li>Friday public-pretraining benefit on V5</li></ul></div></div>''', "09 · CLAIM BOUNDARY"))
    content.append(slide(11, "The MVP works as a world model.<br><em>The warning system is not ready.</em>", '''<div class="final"><p><strong>Observed:</strong> future graph-state prediction improves over persistence; chosen-action futures are directionally coherent.</p><p><strong>Observed failure:</strong> passive branches add little coverage and cannot produce a useful warning/false-alarm tradeoff.</p><p><strong>Next:</strong> prospective unseen topologies, more independent episodes, richer actions and host telemetry, then genuine calibration.</p></div><div class="seal">V5 is closed. Report from saved artifacts only—no retuning, no inference rerun.</div>''', "10 · CONCLUSION"))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cyber World Model — V5 Judge Presentation</title><style>
{common_css()}
html,body{{height:100%;overflow:hidden;background:#071a2d;color:#ecf6ff}}.deck{{height:100%}}.slide{{display:none;height:100vh;width:100vw;position:relative;background:radial-gradient(circle at 85% 12%,#11476a 0,#071a2d 34%,#051321 100%)}}.slide.active{{display:flex}}.slide-inner{{width:min(1180px,92vw);margin:auto;padding:5vh 2vw}}.slide h2{{border:0;margin:.15em 0 .55em;font-size:clamp(2.5rem,5.7vw,5.2rem);line-height:1.02;color:#f2f9ff;letter-spacing:-.045em}}.slide h2 em{{color:#52d3c5;font-style:normal}}.kicker{{font-size:.82rem;letter-spacing:.18em;color:#62d6e8;font-weight:850}}.slide-no{{position:absolute;right:3vw;bottom:2vh;color:#7190a8;font-weight:800}}.deck-lead{{font-size:clamp(1.3rem,2.4vw,2rem);max-width:900px;color:#c6dae9}}.deck-pills{{display:flex;flex-wrap:wrap;gap:12px;margin:5vh 0}}.deck-pills span{{background:#113a58;border:1px solid #286485;border-radius:999px;padding:.7em 1em;font-weight:750}}.bottom-note{{position:absolute;bottom:5vh;color:#91aabe}}.formula{{font:clamp(2rem,4vw,4.2rem) Georgia,serif;color:#6de1ce;margin:5vh 0;text-align:center}}.flow{{display:flex;align-items:center;justify-content:center;gap:14px;flex-wrap:wrap}}.flow div{{background:#0e3552;border:1px solid #2b6e92;border-radius:12px;padding:18px 24px;text-align:center;font-size:1.25rem;font-weight:800}}.flow small{{display:block;color:#9fbed2;font-weight:400;font-size:.7em}}.flow b{{color:#6de1ce;font-size:2rem}}.statement,.warning{{margin-top:4vh;background:#382b16;border-left:6px solid #e5a329;border-radius:8px;padding:16px 20px;color:#fae8bd}}.deck-grid{{display:grid;gap:18px;margin:3vh 0}}.deck-grid.four{{grid-template-columns:repeat(4,1fr)}}.deck-grid.three{{grid-template-columns:repeat(3,1fr)}}.deck-card{{background:#0e304b;border:1px solid #245878;border-radius:15px;padding:22px}}.deck-card strong{{font-size:clamp(2rem,4vw,3.8rem);display:block;color:#67d9ca}}.deck-card span{{color:#b7cddd}}.deck-card.good strong{{color:#67d9ca}}.deck-card.bad strong{{color:#fb7185}}.deck-list{{font-size:clamp(1rem,1.7vw,1.42rem);line-height:1.5}}.deck-list li{{margin:.5em 0}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}.track-card{{border-radius:16px;padding:30px;border:1px solid #2f7298;background:#0c2e47}}.track-card h3{{font-size:2rem;color:#75dfcf}}.mini-flow{{background:#173f5a;padding:18px;border-radius:9px;text-align:center;font-weight:850;font-size:1.3rem}}.fine{{color:#a9c2d3}}.comparison div{{background:#36b99e;margin:14px 0;padding:12px;border-radius:7px;min-width:35%;color:#052319;font-weight:850}}.comparison .base{{background:#dfa33a;color:#2c1b00}}.oracle,.final{{background:#0d334e;border:1px solid #2a6b8f;border-radius:15px;padding:24px;margin-top:4vh}}.oracle strong{{font-size:2rem;display:block;color:#68ddce}}.oracle span{{color:#bad0df}}.tradeoff{{display:grid;grid-template-columns:1fr 1fr;gap:25px}}.tradeoff>div{{background:#0c334e;border-radius:15px;padding:25px;display:grid;grid-template-columns:1fr 1fr}}.tradeoff span{{grid-column:1/-1;font-weight:850;color:#7ce0d2}}.tradeoff strong{{font-size:3.2rem;color:#eef8ff}}.tradeoff small{{color:#a9c2d3}}.tradeoff .bad-panel{{border:2px solid #d8596c;background:#3a1c29}}.deck-table{{font-size:clamp(1rem,1.8vw,1.5rem);background:#0d2d46;border-color:#2a5d7b}}.deck-table th{{background:#0b2135}}.deck-table td{{border-color:#2a4b61}}.deck-table .win{{color:#6de1ce;font-weight:850}}.yes{{color:#6de1ce!important;font-size:2rem}}.no{{color:#fb7185!important;font-size:2rem}}.final{{font-size:clamp(1.2rem,2vw,1.65rem)}}.seal{{margin-top:3vh;color:#fae8bd;border-left:5px solid #e5a329;padding-left:16px}}.controls{{position:fixed;z-index:4;right:2vw;top:2vh;display:flex;gap:8px}}.controls button{{background:#103a57;color:#e9f5fc;border:1px solid #377494;border-radius:8px;padding:8px 12px;cursor:pointer}}@media(max-width:800px){{.deck-grid.four,.deck-grid.three,.split,.tradeoff{{grid-template-columns:1fr 1fr}}.slide-inner{{overflow:auto;max-height:94vh}}.slide h2{{font-size:2.35rem}}.bottom-note{{position:static}}}}@media print{{html,body{{overflow:visible;background:#fff}}.controls{{display:none}}.slide{{display:flex!important;page-break-after:always;height:100vh;-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body><div class="controls"><button id="prev" aria-label="Previous slide">←</button><button id="next" aria-label="Next slide">→</button></div><main class="deck">{''.join(content)}</main><script>
(()=>{{const slides=[...document.querySelectorAll('.slide')];let i=Math.max(0,Math.min(slides.length-1,Number(location.hash.slice(1)||1)-1));function show(n){{i=(n+slides.length)%slides.length;slides.forEach((s,j)=>s.classList.toggle('active',j===i));history.replaceState(null,'','#'+(i+1));}}document.getElementById('prev').onclick=()=>show(i-1);document.getElementById('next').onclick=()=>show(i+1);addEventListener('keydown',e=>{{if(['ArrowRight','PageDown',' '].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);}});show(i);}})();
</script></body></html>'''


def main() -> int:
    for path in (REPORT_JSON, PROVENANCE_JSON, CORPUS_JSON, PLAN_CSV):
        if not path.exists():
            raise FileNotFoundError(path)
    report = json.loads(REPORT_JSON.read_text())
    provenance = json.loads(PROVENANCE_JSON.read_text())
    corpus = json.loads(CORPUS_JSON.read_text())
    if report["status"] != "SEALED_TEST_COMPLETE" or provenance["status"] != "SEALED_TEST_COMPLETE" or corpus["status"] != "PASS":
        raise ValueError("input status is not complete/pass")
    if sha256(REPORT_JSON) != provenance["artifact_sha256_before_provenance"]["report.json"]:
        raise ValueError("sealed report hash differs from provenance")
    if report["evaluation_freeze_sha256"] != provenance["evaluation_freeze_sha256"]:
        raise ValueError("freeze lineage mismatch")
    with PLAN_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    splits = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    if splits != {"train": 40, "validation": 16, "test": 24}:
        raise ValueError(f"unexpected split counts: {splits}")

    report_html = build_report(report, provenance, corpus, splits)
    deck_html = build_deck(report, provenance, corpus, splits)
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "rssm_eod_report.html"
    deck_path = OUT / "cyberwm_v5_judge_presentation.html"
    report_path.write_text(report_html)
    deck_path.write_text(deck_html)
    validate_html(report_path, 0)
    validate_html(deck_path, 11)

    copies = {
        report_path: [ROOT / "outputs/mvp_v2/report/rssm_eod_report.html", DOWNLOADS / "rssm_eod_report.html"],
        deck_path: [DOWNLOADS / "cyberwm_v5_judge_presentation.html"],
    }
    for source, destinations in copies.items():
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha256(source) != sha256(destination):
                raise IOError(f"copy mismatch: {destination}")
    manifest = {"status": "V5_JUDGE_MATERIALS_COMPLETE", "transformation":
        "saved JSON/CSV formatting only; no model, prediction, fit, or threshold selection",
        "inputs": {"sealed_report_sha256": sha256(REPORT_JSON),
                   "provenance_sha256": sha256(PROVENANCE_JSON),
                   "corpus_audit_sha256": sha256(CORPUS_JSON), "episode_plan_sha256": sha256(PLAN_CSV)},
        "outputs": {str(report_path.relative_to(ROOT)): sha256(report_path),
                    str(deck_path.relative_to(ROOT)): sha256(deck_path)},
        "standalone_html_audit": "PASS", "slides": 11,
        "compatibility_copies_verified": True}
    manifest_path = OUT / "materials_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
