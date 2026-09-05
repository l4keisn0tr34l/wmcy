#!/usr/bin/env python3
"""Align ATT&CK event ground truth to network-state windows.

Ground truth comes from scenario documentation / our own emulation controller. This script
never infers MITRE from flow features. It only performs temporal alignment.

A state may overlap multiple techniques/tactics, so labels are emitted as semicolon-separated
multi-label sets rather than a single mutually exclusive class.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("global_states")
    ap.add_argument("events_csv")
    ap.add_argument("--window-seconds", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    states = pd.read_csv(args.global_states)
    states["window_start"] = pd.to_datetime(
    states["window_start"],
    errors="coerce",
    utc=True
    )
    states["window_end"] = states["window_start"] + pd.to_timedelta(args.window_seconds, unit="s")

    events = pd.read_csv(args.events_csv)
    
    events["start_time"] = pd.to_datetime(
    events["start_time"],
    errors="coerce",
    utc=True
    )

    events["end_time"] = pd.to_datetime(
    events["end_time"],
    errors="coerce",
    utc=True
    )

    rows = []
    for s in states.itertuples(index=False):
        overlap = events[(events.start_time < s.window_end) & (events.end_time > s.window_start)]
        rows.append({
            "state_id": s.state_id,
            "window_start": s.window_start,
            "technique_ids": ";".join(sorted(set(overlap.technique_id.dropna().astype(str)))),
            "tactics": ";".join(sorted(set(overlap.tactic.dropna().astype(str)))),
            "actors": ";".join(sorted(set(overlap.actor.dropna().astype(str)))) if "actor" in overlap else "",
            "targets": ";".join(sorted(set(overlap.target.dropna().astype(str)))) if "target" in overlap else "",
            "has_lateral_movement": int((overlap.tactic.astype(str) == "Lateral Movement").any()) if len(overlap) else 0,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"aligned {len(rows):,} state windows -> {out}")


if __name__ == "__main__":
    main()
