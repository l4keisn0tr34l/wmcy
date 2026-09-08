#!/usr/bin/env python3
"""Convert original UNSW-NB15 rows into sorted, gap-bounded canonical segments.

Observable flow primitives and attack ground truth are always written separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "service", "Spkts", "Dpkts", "Stime", "Ltime", "attack_cat", "Label",
]


def iso_utc(epoch_seconds: pd.Series) -> pd.Series:
    values = pd.to_datetime(epoch_seconds, unit="s", utc=True)
    return values.map(lambda value: value.isoformat())


def read_source(path: Path, names: list[str], chunksize: int,
                max_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = [names.index(name) for name in REQUIRED]
    pieces = []; rows_read = 0; raw_inversions = 0; previous_time = None
    for chunk in pd.read_csv(
        path, header=None, names=names, usecols=usecols, chunksize=chunksize,
        encoding="latin1", low_memory=False,
    ):
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        chunk.insert(0, "source_row_number", np.arange(rows_read, rows_read + len(chunk), dtype=np.int64))
        start = pd.to_numeric(chunk["Stime"], errors="coerce")
        valid_start = start.dropna().to_numpy()
        if len(valid_start):
            if previous_time is not None and valid_start[0] < previous_time:
                raw_inversions += 1
            raw_inversions += int(np.sum(valid_start[1:] < valid_start[:-1]))
            previous_time = float(valid_start[-1])
        pieces.append(chunk); rows_read += len(chunk)
        if max_rows is not None and rows_read >= max_rows:
            break
    if not pieces:
        raise ValueError(f"no rows read from {path}")
    frame = pd.concat(pieces, ignore_index=True)
    return frame, {"rows_read": rows_read, "raw_timestamp_inversions": raw_inversions}


def clean_and_sort(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.copy()
    for column in ["Stime", "Ltime", "sport", "dsport", "dur", "sbytes", "dbytes", "Spkts", "Dpkts", "Label"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["srcip"] = frame["srcip"].astype(str).str.strip()
    frame["dstip"] = frame["dstip"].astype(str).str.strip()
    invalid_time = frame["Stime"].isna()
    invalid_ip = frame["srcip"].isin(["", "nan"]) | frame["dstip"].isin(["", "nan"])
    reversed_time = frame["Ltime"].notna() & (frame["Ltime"] < frame["Stime"])
    invalid = invalid_time | invalid_ip | reversed_time
    counts = {"invalid_time_rows": int(invalid_time.sum()), "invalid_ip_rows": int(invalid_ip.sum()),
              "reversed_time_rows": int(reversed_time.sum()), "dropped_rows": int(invalid.sum())}
    frame = frame.loc[~invalid].copy()
    frame["Ltime"] = frame["Ltime"].fillna(frame["Stime"])
    frame = frame.sort_values(["Stime", "Ltime", "source_row_number"], kind="stable").reset_index(drop=True)
    return frame, counts


def segment_ranges(frame: pd.DataFrame, gap_seconds: float) -> list[tuple[int, int]]:
    starts = frame["Stime"].to_numpy(dtype=float)
    boundaries = np.flatnonzero(np.diff(starts) > gap_seconds) + 1
    points = np.concatenate([[0], boundaries, [len(frame)]])
    return [(int(points[index]), int(points[index + 1])) for index in range(len(points) - 1)]


def write_segment(segment: pd.DataFrame, source: Path, source_index: int,
                  segment_index: int, out_root: Path, dataset_prefix: str) -> dict[str, Any]:
    dataset_id = f"{dataset_prefix}_{source_index:02d}_{segment_index:02d}"
    directory = out_root / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    event_id = np.arange(len(segment), dtype=np.int64)
    start_epoch = segment["Stime"].round().astype("int64")
    end_epoch = segment["Ltime"].round().astype("int64")
    labels = segment["Label"].fillna(0).astype("int64")
    attack_category = segment["attack_cat"].fillna("").astype(str).str.strip()
    attack_category = attack_category.mask((labels == 0) & attack_category.eq(""), "BENIGN")
    attack_category = attack_category.mask((labels != 0) & attack_category.eq(""), "UNKNOWN")

    observations = pd.DataFrame({
        "event_id": event_id,
        "dataset_id": dataset_id,
        "source_file": source.name,
        "timestamp": iso_utc(start_epoch),
        "timestamp_resolution_sec": 1,
        "source_ip": segment["srcip"].to_numpy(),
        "source_port": segment["sport"].fillna(0).clip(lower=0).astype("int64").to_numpy(),
        "destination_ip": segment["dstip"].to_numpy(),
        "destination_port": segment["dsport"].fillna(0).clip(lower=0).astype("int64").to_numpy(),
        "protocol": segment["proto"].fillna("UNKNOWN").astype(str).str.strip().to_numpy(),
        "service": segment["service"].fillna("UNKNOWN").astype(str).str.strip().to_numpy(),
        "connection_state": segment["state"].fillna("UNKNOWN").astype(str).str.strip().to_numpy(),
        "total_fwd_packets": segment["Spkts"].fillna(0).clip(lower=0).to_numpy(),
        "total_backward_packets": segment["Dpkts"].fillna(0).clip(lower=0).to_numpy(),
        "total_length_of_fwd_packets": segment["sbytes"].fillna(0).clip(lower=0).to_numpy(),
        "total_length_of_bwd_packets": segment["dbytes"].fillna(0).clip(lower=0).to_numpy(),
        "flow_duration": segment["dur"].fillna(0).clip(lower=0).to_numpy(),
    })
    truth = pd.DataFrame({
        "event_id": event_id,
        "dataset_id": dataset_id,
        "timestamp": observations["timestamp"],
        "end_timestamp": iso_utc(end_epoch),
        "raw_attack_category": attack_category.to_numpy(),
        "raw_binary_label": labels.to_numpy(),
        "source_file": source.name,
        "source_row_number": segment["source_row_number"].astype("int64").to_numpy(),
    })
    forbidden = {"label", "attack_cat", "raw_attack_category", "raw_binary_label"}
    leaking = forbidden.intersection(observations.columns)
    if leaking:
        raise AssertionError(f"ground truth leaked into observations: {sorted(leaking)}")
    if not observations.event_id.equals(truth.event_id):
        raise AssertionError("observation/truth event IDs differ")
    parsed = pd.to_datetime(observations.timestamp, utc=True)
    if not parsed.is_monotonic_increasing:
        raise AssertionError("segment timestamps are not monotonic")

    observations.to_csv(directory / "observations.csv.gz", index=False, compression="gzip")
    truth.to_csv(directory / "row_ground_truth.csv.gz", index=False, compression="gzip")
    return {
        "dataset_id": dataset_id, "source_file": source.name, "segment_index": segment_index,
        "rows": len(segment), "start_epoch": int(start_epoch.min()), "end_epoch": int(end_epoch.max()),
        "start_time": parsed.min().isoformat(), "end_time": pd.to_datetime(end_epoch.max(), unit="s", utc=True).isoformat(),
        "unique_source_ips": int(segment.srcip.nunique()),
        "unique_destination_ips": int(segment.dstip.nunique()),
        "attack_rows": int((labels != 0).sum()),
        "attack_categories": {str(key): int(value) for key, value in attack_category.value_counts().items()},
        "observations": str(directory / "observations.csv.gz"),
        "row_ground_truth": str(directory / "row_ground_truth.csv.gz"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Original headerless UNSW-NB15_*.csv files")
    ap.add_argument("--features", required=True, help="NUSW-NB15_features.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset-prefix", default="unsw")
    ap.add_argument("--gap-seconds", type=float, default=3600.0)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--max-rows-per-file", type=int, default=None, help="Smoke-test bound")
    args = ap.parse_args()
    if args.gap_seconds <= 0:
        raise ValueError("--gap-seconds must be positive")
    feature_frame = pd.read_csv(args.features, encoding="latin1")
    names = feature_frame["Name"].astype(str).str.strip().tolist()
    if len(names) != 49 or not set(REQUIRED).issubset(names):
        raise ValueError("unexpected UNSW feature definition")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    source_profiles = []; segments = []
    for source_index, input_name in enumerate(args.inputs, start=1):
        source = Path(input_name)
        frame, read_profile = read_source(
            source, names, args.chunksize, args.max_rows_per_file
        )
        frame, clean_profile = clean_and_sort(frame)
        ranges = segment_ranges(frame, args.gap_seconds)
        source_profile = {"source_file": source.name, **read_profile, **clean_profile,
                          "valid_rows": len(frame), "segments": len(ranges)}
        source_profiles.append(source_profile)
        for segment_index, (start, end) in enumerate(ranges):
            segment = frame.iloc[start:end].copy()
            row = write_segment(segment, source, source_index, segment_index, out, args.dataset_prefix)
            segments.append(row)
            print(f"{row['dataset_id']}: {row['rows']:,} rows {row['start_time']} -> {row['end_time']}")
    # Source CSV boundaries overlap in time. Connected intervals must remain one
    # capture group so future train/validation/test splitting cannot leak adjacent
    # moments across source files.
    capture_groups = []
    current_group: dict[str, Any] | None = None
    for row in sorted(segments, key=lambda item: (item["start_epoch"], item["end_epoch"])):
        if current_group is None or row["start_epoch"] > current_group["end_epoch"] + args.gap_seconds:
            current_group = {
                "capture_group": f"unsw_capture_{len(capture_groups):02d}",
                "start_epoch": row["start_epoch"], "end_epoch": row["end_epoch"],
                "dataset_ids": [],
            }
            capture_groups.append(current_group)
        current_group["end_epoch"] = max(current_group["end_epoch"], row["end_epoch"])
        current_group["dataset_ids"].append(row["dataset_id"])
        row["capture_group"] = current_group["capture_group"]
    for group in capture_groups:
        group["start_time"] = pd.to_datetime(group["start_epoch"], unit="s", utc=True).isoformat()
        group["end_time"] = pd.to_datetime(group["end_epoch"], unit="s", utc=True).isoformat()

    manifest = {
        "dataset": "UNSW-NB15 original CSV files",
        "timestamp_interpretation": "Stime/Ltime are Unix seconds converted to UTC; one-second source precision",
        "segmentation_gap_seconds": args.gap_seconds,
        "observable_fields": [
            "IP/port/protocol/service/state", "packets", "bytes", "duration", "timestamp"
        ],
        "unavailable_not_fabricated": [
            "TCP SYN/ACK/RST/FIN counts", "exact ATT&CK technique", "lateral-movement truth"
        ],
        "ground_truth_policy": "raw attack_cat/Label retained separately; not model input or forced ATT&CK truth",
        "source_profiles": source_profiles,
        "segments": segments,
        "capture_groups": capture_groups,
        "split_rule": "split capture_group, never overlapping/adjacent source segments",
        "total_valid_rows": int(sum(row["rows"] for row in segments)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
