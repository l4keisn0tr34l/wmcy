#!/usr/bin/env python3
"""Profile raw CICIDS2017 TrafficLabelling CSVs.

WHY THIS EXISTS
---------------
Before preprocessing we verify the actual schema, label counts and timestamp precision.
This prevents us from silently assuming that the dataset has second-level timestamps or
specific feature names that are not present in the user's copy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def profile(path: Path) -> dict:
    head = pd.read_csv(path, nrows=5, low_memory=False)
    columns = [c.strip() for c in head.columns]

    label_col = next((c for c in head.columns if c.strip() == "Label"), None)
    ts_col = next((c for c in head.columns if c.strip() == "Timestamp"), None)

    labels = {}
    timestamps = []
    rows = 0
    usecols = [c for c in [label_col, ts_col] if c is not None]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000, low_memory=False):
        rows += len(chunk)
        if label_col is not None:
            vc = chunk[label_col].astype(str).str.strip().value_counts()
            for k, v in vc.items():
                labels[k] = labels.get(k, 0) + int(v)
        if ts_col is not None and len(timestamps) < 20:
            timestamps.extend(chunk[ts_col].astype(str).str.strip().head(20-len(timestamps)).tolist())

    # CICIDS2017 generated labelled flow CSVs commonly store only HH:MM.
    has_seconds = any(str(x).count(":") >= 2 for x in timestamps)

    return {
        "file": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 3),
        "rows": rows,
        "n_columns": len(columns),
        "columns": columns,
        "label_counts": labels,
        "timestamp_examples": timestamps,
        "timestamp_has_seconds": has_seconds,
        "inferred_timestamp_resolution_sec": 1 if has_seconds else 60,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = [profile(Path(p)) for p in args.files]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
