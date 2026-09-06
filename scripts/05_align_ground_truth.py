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
    events = events.dropna(subset=["start_time", "end_time"]).copy()

    reversed_events = events["end_time"] < events["start_time"]
    if reversed_events.any():
        raise ValueError(f"{int(reversed_events.sum())} ground-truth event(s) end before they start")

    # Intervals use half-open overlap. Zero-duration events are points and align
    # to the one half-open state window containing their timestamp. This avoids
    # inventing timestamp precision or an artificial event duration.
    point_events = events["end_time"] == events["start_time"]

    rows = []
    covered_event_indexes = set()
    for s in states.itertuples(index=False):
        interval_overlap = (
            ~point_events
            & (events.start_time < s.window_end)
            & (events.end_time > s.window_start)
        )
        point_overlap = (
            point_events
            & (events.start_time >= s.window_start)
            & (events.start_time < s.window_end)
        )
        overlap = events[interval_overlap | point_overlap]
        covered_event_indexes.update(overlap.index.tolist())
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

    unaligned = events.loc[~events.index.isin(covered_event_indexes)]
    if len(unaligned):
        print("WARNING: ground-truth events with no overlapping state window:")
        for e in unaligned.itertuples():
            print(f"  {e.start_time}..{e.end_time} {e.technique_id} actor={getattr(e, 'actor', '')} target={getattr(e, 'target', '')}")


if __name__ == "__main__":
    main()
