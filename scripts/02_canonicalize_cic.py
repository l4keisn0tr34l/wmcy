#!/usr/bin/env python3
"""Convert a CICIDS2017 TrafficLabelling CSV into the project canonical event format.

INPUT
-----
Raw CIC flow rows.

OUTPUT
------
1. observations.csv.gz: only defender-observable fields and CICFlowMeter features.
2. row_ground_truth.csv.gz: raw CIC labels kept separately from model inputs.

WHY
---
A world model must learn from observable telemetry. Attack labels are ground truth, not
observations. Keeping them in a separate table prevents accidental target leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cyberwm.common import snake


def parse_cic_timestamp(series: pd.Series) -> pd.Series:
    """Parse CICIDS2017's ambiguous working-hours timestamps.

    The uploaded CSVs use e.g. '6/7/2017 3:04' for 15:04 and do not retain seconds.
    The CIC collection is a working-day capture (roughly 09:00-17:00), so hours 1..7
    are interpreted as 13:00..19:00. The function does NOT invent seconds.
    """
    raw = series.astype(str).str.strip()
    dt = pd.to_datetime(raw, format="%d/%m/%Y %H:%M", errors="coerce")
    add_pm = dt.dt.hour.between(1, 7)
    dt.loc[add_pm] = dt.loc[add_pm] + pd.Timedelta(hours=12)
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src = Path(args.csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs_path = out / "observations.csv.gz"
    gt_path = out / "row_ground_truth.csv.gz"

    first = True
    event_base = 0
    for chunk in pd.read_csv(src, chunksize=75_000, low_memory=False):
        chunk.columns = [snake(c) for c in chunk.columns]
        if "timestamp" not in chunk or "label" not in chunk:
            raise RuntimeError("Expected Timestamp and Label columns in CIC file")

        n = len(chunk)
        chunk.insert(0, "event_id", np.arange(event_base, event_base+n, dtype=np.int64))
        event_base += n

        parsed = parse_cic_timestamp(chunk["timestamp"])
        raw_ts = chunk["timestamp"].astype(str)
        labels = chunk["label"].astype(str).str.strip()

        # Ground truth stays outside observable input.
        gt = pd.DataFrame({
            "event_id": chunk["event_id"],
            "dataset_id": args.dataset_id,
            "timestamp": parsed,
            "raw_cic_label": labels,
        })

        obs = chunk.drop(columns=["label"]).copy()
        obs["timestamp_raw"] = raw_ts
        obs["timestamp"] = parsed
        obs["dataset_id"] = args.dataset_id
        obs["source_file"] = src.name
        obs["timestamp_resolution_sec"] = 60

        # Replace rate-feature infinities with missing values. We do NOT impute here;
        # imputation must be fitted on the training split later to avoid leakage.
        obs = obs.replace([np.inf, -np.inf], np.nan)

        obs.to_csv(obs_path, index=False, mode="w" if first else "a", header=first, compression="gzip")
        gt.to_csv(gt_path, index=False, mode="w" if first else "a", header=first, compression="gzip")
        first = False

    print(f"wrote {event_base:,} observations -> {obs_path}")
    print(f"wrote {event_base:,} row labels   -> {gt_path}")


if __name__ == "__main__":
    main()
