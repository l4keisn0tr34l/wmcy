#!/usr/bin/env python3
"""Post-capture V5 consistency audit; performs no model inference or selection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_contract import HOSTS
from src.cyberwm.v5_sequences import require


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-dir", type=Path, default=ROOT / "lab/episodes")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mvp_v5/corpus_audit")
    ap.add_argument("--deep-validate", action="store_true", help="rerun general and V5 validators on all episodes")
    args = ap.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out_dir}")
    plan = pd.read_csv(ROOT / "configs/mvp_v5_episode_plan.csv", dtype=str)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for expected in plan.to_dict("records"):
        name = expected["episode_id"]; episode = args.episodes_dir / name
        required = ["network.pcap", "episode_metadata.csv", "cleanup.json", "observations.csv.gz",
                    "state_ground_truth.csv", "states/global_states.csv", "states/node_states.csv.gz",
                    "states/edge_states.csv.gz", "tcpdump.log", "ground_truth.csv", "defender_actions.csv"]
        require(all((episode / p).is_file() for p in required), f"{name}: incomplete raw/derived files")
        if args.deep_validate:
            subprocess.run([sys.executable, str(ROOT / "scripts/07_validate_episode.py"), str(episode),
                            "--window-seconds", "5"], check=True, stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, str(ROOT / "scripts/49_freeze_v5_capture.py"), "--episode",
                            str(episode)], check=True, stdout=subprocess.DEVNULL)
        metadata_frame = pd.read_csv(episode / "episode_metadata.csv", dtype=str)
        require(len(metadata_frame) == 1, f"{name}: metadata rows")
        metadata = metadata_frame.iloc[0]
        for key in ["episode_id", "scenario", "seed", "background_profile", "defender_action",
                    "topology_profile", "node_count"]:
            require(metadata[key] == expected[key], f"{name}: plan mismatch {key}")
        start = pd.Timestamp(metadata.capture_start); end = pd.Timestamp(metadata.capture_end)
        states = pd.read_csv(episode / "states/global_states.csv")
        observations = pd.read_csv(episode / "observations.csv.gz")
        truth = pd.read_csv(episode / "ground_truth.csv")
        actions = pd.read_csv(episode / "defender_actions.csv", dtype=str)
        match = re.search(r"(\d+) packets captured\s+(\d+) packets received by filter\s+"
                          r"(\d+) packets dropped by kernel", (episode / "tcpdump.log").read_text())
        require(match is not None, f"{name}: tcpdump summary absent")
        background_events = 0; first_offsets: list[float] = []; last_margins: list[float] = []
        action = actions.iloc[0] if len(actions) else None
        for host, ip in HOSTS.items():
            records = [json.loads(line) for line in (episode / f"background_{host}.jsonl").read_text().splitlines()]
            require(records, f"{name}: no background {host}")
            background_events += len(records)
            first_offsets.append(min(x["start_epoch"] for x in records) - start.timestamp())
            last_margins.append(end.timestamp() - max(x["start_epoch"] for x in records))
            for record in records:
                if record["returncode"] == 0:
                    continue
                block_effect = bool(
                    action is not None and action.action == "block_ssh" and record["kind"] == "admin"
                    and record["source"] == HOSTS[action.source] and record["destination"] == HOSTS[action.target]
                    and record["start_epoch"] >= pd.Timestamp(action.effective_start_time).timestamp()
                    and record["returncode"] == 255
                )
                tail_timeout = record["returncode"] == 124 and end.timestamp() - record["start_epoch"] <= 3
                classification = "expected_block_consequence" if block_effect else "bounded_tail_timeout" if tail_timeout else "unexpected"
                failures.append({"episode_id": name, "host": host, **record, "classification": classification})
                require(classification != "unexpected", f"{name}: uncontrolled background failure {record}")
        action_margin = np.nan; forecast_state = -1; decision_offset = np.nan
        if action is not None:
            cutoff = pd.Timestamp(action.forecast_time)
            action_margin = (end - cutoff).total_seconds()
            decision_offset = (pd.Timestamp(action.start_time) - start).total_seconds()
            state_times = pd.to_datetime(states.window_start, utc=True)
            indexes = np.flatnonzero(state_times == cutoff)
            require(len(indexes) == 1, f"{name}: action cutoff absent")
            forecast_state = int(indexes[0])
        rows.append({**expected, "duration_seconds_actual": (end-start).total_seconds(),
                     "state_count": len(states), "observation_count": len(observations),
                     "pcap_bytes": (episode / "network.pcap").stat().st_size,
                     "packets_captured": int(match.group(1)), "packets_received": int(match.group(2)),
                     "packets_dropped": int(match.group(3)), "zero_state_count": int(states.flow_count.eq(0).sum()),
                     "truth_event_count": len(truth), "completed_lm_events": int(truth.tactic.eq("Lateral Movement").sum()),
                     "attempted_lm_events": int(truth.tactic.eq("Lateral Movement Attempt").sum()),
                     "action_count": len(actions), "action_future_margin_seconds": action_margin,
                     "decision_offset_seconds": decision_offset, "forecast_state": forecast_state,
                     "background_event_count": background_events, "background_first_offset_max": max(first_offsets),
                     "background_last_margin_max": max(last_margins)})
    frame = pd.DataFrame(rows); failure_frame = pd.DataFrame(failures)
    require(len(frame) == 80 and frame.episode_id.nunique() == 80, "episode count/IDs")
    require(frame.duration_seconds_actual.between(150, 153, inclusive="left").all(), "capture duration")
    require(frame.state_count.eq(29).all(), "dense state count")
    require(frame.packets_dropped.eq(0).all(), "packet drops")
    require(frame.action_count.sum() == 40, "action count")
    require(frame.loc[frame.action_count.eq(1), "action_future_margin_seconds"].min() >= 30, "action future horizon")
    families = frame.groupby("paired_family").agg(
        episodes=("episode_id", "size"), duration_range=("duration_seconds_actual", lambda x: x.max()-x.min()),
        state_range=("state_count", lambda x: x.max()-x.min()), profiles=("background_profile", "nunique"),
        splits=("split", "nunique"), seeds=("seed", "nunique"), forecast_state_range=("forecast_state", lambda x: x.max()-x.min()))
    require((families.episodes == 2).all() and (families.state_range == 0).all(), "paired shape inconsistency")
    require((families[["profiles", "splits", "seeds"]] == 1).all().all(), "paired assignment inconsistency")
    benign = frame.scenario.isin(["background_only", "legitimate_ssh", "matched_legitimate_ssh"])
    means = frame.assign(group=np.where(benign, "benign_or_legitimate", "attack_intent")).groupby("group").duration_seconds_actual.mean()
    report = {
        "status": "PASS", "scope": "dataset consistency QA only; no model inference, preprocessing fit, or selection",
        "episodes": len(frame), "general_and_v5_validated": bool(args.deep_validate),
        "capture_duration_seconds": {"min": frame.duration_seconds_actual.min(), "max": frame.duration_seconds_actual.max(),
                                     "mean": frame.duration_seconds_actual.mean()},
        "attack_vs_benign_duration_mean_absolute_difference_seconds": abs(means.iloc[0]-means.iloc[1]),
        "state_count_unique": sorted(frame.state_count.unique().tolist()), "complete_state_coverage_seconds": 145,
        "paired_max_duration_difference_seconds": families.duration_range.max(),
        "paired_max_state_count_difference": int(families.state_range.max()),
        "paired_max_forecast_state_index_difference": int(families.forecast_state_range.max()),
        "packet_drops_total": int(frame.packets_dropped.sum()), "actions": int(frame.action_count.sum()),
        "minimum_action_future_margin_seconds": frame.action_future_margin_seconds.min(),
        "background_failures": failure_frame.classification.value_counts().to_dict() if len(failure_frame) else {},
        "limitations": ["paired captures are independently generated, not packet-identical",
                        "absolute five-second alignment can move paired forecast cutoffs by one state",
                        "traffic volume varies intentionally by background profile and scenario",
                        "QA inspects all split integrity/truth but performs no model evaluation or tuning"],
    }
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v5-audit-", dir=args.out_dir.parent))
    try:
        frame.to_csv(temporary / "episode_summary.csv", index=False)
        families.reset_index().to_csv(temporary / "paired_family_summary.csv", index=False)
        failure_frame.to_csv(temporary / "background_failures.csv", index=False)
        (temporary / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        os.rename(temporary, args.out_dir)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    print(json.dumps(report, indent=2)); print(f"V5 corpus audit -> {args.out_dir}")
    return 0


if __name__ == "__main__": sys.exit(main())
