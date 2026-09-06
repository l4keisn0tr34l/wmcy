#!/usr/bin/env python3
"""Validate one episode before it is used for world-model training.

This checks temporal/state invariants, referential integrity, temporal ground-truth
coverage, and separation of observable telemetry from target labels.  It does not
infer attacks from traffic and does not write or repair any dataset file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


FORBIDDEN_OBSERVABLE_COLUMNS = {
    "label",
    "raw_cic_label",
    "technique_id",
    "technique_ids",
    "technique",
    "tactic",
    "tactics",
    "actor",
    "actors",
    "target",
    "targets",
    "has_lateral_movement",
    "start_time",
    "end_time",
}
TIMEZONE_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


def split_labels(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def parse_utc(
    df: pd.DataFrame,
    column: str,
    path: Path,
    errors: list[str],
    *,
    require_explicit_timezone: bool = True,
) -> pd.Series:
    if column not in df.columns:
        errors.append(f"{path}: missing required timestamp column {column!r}")
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    raw = df[column].astype("string")
    if require_explicit_timezone:
        missing_zone = ~raw.fillna("").str.contains(TIMEZONE_SUFFIX)
        if missing_zone.any():
            examples = raw[missing_zone].head(3).tolist()
            errors.append(
                f"{path}: {int(missing_zone.sum())} {column} value(s) lack an explicit "
                f"timezone; examples={examples}"
            )

    parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
    invalid = parsed.isna()
    if invalid.any():
        errors.append(f"{path}: {int(invalid.sum())} invalid {column} value(s)")
    return parsed


def check_observable_columns(df: pd.DataFrame, path: Path, errors: list[str]) -> None:
    leaked = sorted(FORBIDDEN_OBSERVABLE_COLUMNS.intersection(df.columns))
    if leaked:
        errors.append(f"{path}: forbidden ground-truth column(s) in model-facing data: {leaked}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir", help="episode containing observations, states, and truth files")
    ap.add_argument("--window-seconds", type=float, default=5.0)
    args = ap.parse_args()

    episode = Path(args.episode_dir)
    states_dir = episode / "states"
    invalid_marker = episode / "INVALID_EPISODE.txt"
    metadata_path = episode / "episode_metadata.csv"
    paths = {
        "global": states_dir / "global_states.csv",
        "nodes": states_dir / "node_states.csv.gz",
        "edges": states_dir / "edge_states.csv.gz",
        "events": episode / "ground_truth.csv",
        "aligned": episode / "state_ground_truth.csv",
    }
    observation_candidates = [episode / "observations.csv.gz", episode / "observations.csv"]
    paths["observations"] = next((p for p in observation_candidates if p.exists()), observation_candidates[0])

    errors: list[str] = []
    warnings: list[str] = []
    if invalid_marker.exists():
        errors.append(f"episode is explicitly quarantined by {invalid_marker}")
    for name, path in paths.items():
        if not path.is_file():
            errors.append(f"missing {name} file: {path}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    observations = pd.read_csv(paths["observations"], low_memory=False)
    global_states = pd.read_csv(paths["global"])
    nodes = pd.read_csv(paths["nodes"], low_memory=False)
    edges = pd.read_csv(paths["edges"], low_memory=False)
    events = pd.read_csv(paths["events"])
    aligned = pd.read_csv(paths["aligned"])
    metadata = pd.read_csv(metadata_path) if metadata_path.is_file() else None

    for df, key in [
        (observations, "observations"),
        (global_states, "global"),
        (nodes, "nodes"),
        (edges, "edges"),
    ]:
        check_observable_columns(df, paths[key], errors)

    observation_times = parse_utc(observations, "timestamp", paths["observations"], errors)
    global_times = parse_utc(global_states, "window_start", paths["global"], errors)
    node_times = parse_utc(nodes, "window_start", paths["nodes"], errors)
    edge_times = parse_utc(edges, "window_start", paths["edges"], errors)
    aligned_times = parse_utc(aligned, "window_start", paths["aligned"], errors)
    event_starts = parse_utc(events, "start_time", paths["events"], errors)
    event_ends = parse_utc(events, "end_time", paths["events"], errors)

    capture_start = capture_end = None
    if metadata is not None:
        if len(metadata) != 1:
            errors.append(f"{metadata_path}: expected exactly one metadata row, found {len(metadata)}")
        required_metadata = {
            "episode_id", "scenario", "seed", "capture_start", "capture_end",
            "actor", "pivot", "target", "window_seconds",
        }
        missing_metadata = sorted(required_metadata.difference(metadata.columns))
        if missing_metadata:
            errors.append(f"{metadata_path}: missing columns {missing_metadata}")
        else:
            capture_starts = parse_utc(metadata, "capture_start", metadata_path, errors)
            capture_ends = parse_utc(metadata, "capture_end", metadata_path, errors)
            if len(metadata) == 1:
                capture_start = capture_starts.iloc[0]
                capture_end = capture_ends.iloc[0]
                metadata_window = pd.to_numeric(metadata["window_seconds"], errors="coerce").iloc[0]
                if pd.isna(metadata_window) or float(metadata_window) != args.window_seconds:
                    errors.append(
                        f"{metadata_path}: window_seconds={metadata_window!r} does not match "
                        f"validator value {args.window_seconds}"
                    )
                if capture_end <= capture_start:
                    errors.append(f"{metadata_path}: capture_end must be after capture_start")

    if global_states.empty:
        errors.append(f"{paths['global']}: no states")
    else:
        state_ids = pd.to_numeric(global_states.get("state_id"), errors="coerce")
        expected_ids = np.arange(len(global_states), dtype=np.int64)
        if state_ids.isna().any() or not np.array_equal(state_ids.to_numpy(), expected_ids):
            errors.append(f"{paths['global']}: state_id must be unique and contiguous from 0")
        if not global_times.is_monotonic_increasing or global_times.duplicated().any():
            errors.append(f"{paths['global']}: window_start must be strictly increasing and unique")
        expected_step = pd.to_timedelta(args.window_seconds, unit="s")
        bad_steps = global_times.diff().iloc[1:] != expected_step
        if bad_steps.any():
            bad_indexes = bad_steps[bad_steps].index[:5].tolist()
            errors.append(
                f"{paths['global']}: {int(bad_steps.sum())} adjacent state step(s) are not "
                f"exactly {expected_step}; row indexes={bad_indexes}"
            )
        if capture_start is not None and capture_end is not None and capture_end > capture_start:
            expected_windows = pd.date_range(
                start=capture_start.ceil(expected_step),
                end=capture_end.floor(expected_step),
                freq=expected_step,
                inclusive="left",
            )
            if not pd.DatetimeIndex(global_times).equals(expected_windows):
                errors.append(
                    f"{paths['global']}: state windows do not match complete windows inside "
                    f"the metadata capture interval (expected {len(expected_windows)})"
                )

    valid_ids = set(pd.to_numeric(global_states.get("state_id"), errors="coerce").dropna().astype(int))
    state_time_by_id = dict(zip(global_states.get("state_id", []), global_times))
    for table, table_times, key in [(nodes, node_times, "nodes"), (edges, edge_times, "edges")]:
        if "state_id" not in table:
            errors.append(f"{paths[key]}: missing state_id")
            continue
        refs = pd.to_numeric(table["state_id"], errors="coerce")
        invalid_refs = refs.isna() | ~refs.fillna(-1).astype(int).isin(valid_ids)
        if invalid_refs.any():
            errors.append(f"{paths[key]}: {int(invalid_refs.sum())} row(s) reference invalid states")
        comparable = ~invalid_refs & table_times.notna()
        expected_times = refs[comparable].astype(int).map(state_time_by_id)
        mismatched_times = table_times[comparable].reset_index(drop=True) != expected_times.reset_index(drop=True)
        if mismatched_times.any():
            errors.append(f"{paths[key]}: {int(mismatched_times.sum())} row(s) disagree with global state time")

    if {"state_id", "host"}.issubset(nodes.columns):
        duplicates = nodes.duplicated(["state_id", "host"])
        if duplicates.any():
            errors.append(f"{paths['nodes']}: {int(duplicates.sum())} duplicate state/host row(s)")
    if {"state_id", "source_ip", "destination_ip"}.issubset(edges.columns):
        duplicates = edges.duplicated(["state_id", "source_ip", "destination_ip"])
        if duplicates.any():
            errors.append(f"{paths['edges']}: {int(duplicates.sum())} duplicate directed edge row(s)")

    numeric_global = global_states.drop(columns=["state_id", "window_start"], errors="ignore").apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric_global.isna().any().any() or not np.isfinite(numeric_global.to_numpy()).all():
        errors.append(f"{paths['global']}: global observable features contain missing/non-finite values")

    if "flow_count" in global_states:
        empty_ids = set(global_states.loc[global_states["flow_count"] == 0, "state_id"].astype(int))
        empty_features = numeric_global.loc[global_states["flow_count"] == 0]
        if len(empty_features) and not np.allclose(empty_features.to_numpy(), 0.0):
            errors.append(f"{paths['global']}: zero-flow state(s) contain non-zero global features")
        if "state_id" in nodes and nodes["state_id"].isin(empty_ids).any():
            errors.append(f"{paths['nodes']}: node rows exist for zero-flow global state(s)")
        if "state_id" in edges and edges["state_id"].isin(empty_ids).any():
            errors.append(f"{paths['edges']}: edge rows exist for zero-flow global state(s)")
        warnings.append(f"{len(empty_ids)} explicit zero-traffic state(s)")

    if "event_id" in observations and observations["event_id"].duplicated().any():
        errors.append(f"{paths['observations']}: duplicate event_id values")
    if not observation_times.is_monotonic_increasing:
        errors.append(f"{paths['observations']}: observations are not chronological")

    reversed_events = event_ends < event_starts
    if reversed_events.any():
        errors.append(f"{paths['events']}: {int(reversed_events.sum())} event(s) end before they start")
    if capture_start is not None and capture_end is not None:
        outside_capture = (event_starts < capture_start) | (event_ends > capture_end)
        if outside_capture.any():
            errors.append(f"{paths['events']}: {int(outside_capture.sum())} event(s) fall outside capture bounds")

    if len(aligned) != len(global_states):
        errors.append(
            f"{paths['aligned']}: {len(aligned)} rows but global state table has {len(global_states)}"
        )
    elif "state_id" not in aligned or not np.array_equal(
        pd.to_numeric(aligned["state_id"], errors="coerce").to_numpy(),
        pd.to_numeric(global_states["state_id"], errors="coerce").to_numpy(),
    ):
        errors.append(f"{paths['aligned']}: state IDs do not exactly match global states")
    if len(aligned_times) == len(global_times) and not aligned_times.equals(global_times):
        errors.append(f"{paths['aligned']}: window timestamps do not exactly match global states")

    state_ends = global_times + pd.to_timedelta(args.window_seconds, unit="s")
    covered_events = 0
    for event_index in events.index:
        start = event_starts.loc[event_index]
        end = event_ends.loc[event_index]
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        if end == start:
            overlap_mask = (start >= global_times) & (start < state_ends)
        else:
            overlap_mask = (start < state_ends) & (end > global_times)
        overlap_positions = np.flatnonzero(overlap_mask.to_numpy())
        descriptor = (
            f"event row {event_index} technique={events.at[event_index, 'technique_id']} "
            f"actor={events.at[event_index, 'actor']} target={events.at[event_index, 'target']}"
        )
        if not len(overlap_positions):
            errors.append(f"{paths['events']}: {descriptor} has no overlapping state window")
            continue
        covered_events += 1

        if len(aligned) != len(global_states):
            continue
        technique = str(events.at[event_index, "technique_id"])
        actor = str(events.at[event_index, "actor"])
        target = str(events.at[event_index, "target"])
        matching_output = False
        for position in overlap_positions:
            row = aligned.iloc[position]
            if (
                technique in split_labels(row.get("technique_ids"))
                and actor in split_labels(row.get("actors"))
                and target in split_labels(row.get("targets"))
            ):
                matching_output = True
                if str(events.at[event_index, "tactic"]) == "Lateral Movement" and int(
                    row.get("has_lateral_movement", 0)
                ) != 1:
                    errors.append(f"{paths['aligned']}: {descriptor} lacks lateral-movement flag")
                break
        if not matching_output:
            errors.append(f"{paths['aligned']}: {descriptor} is missing from its overlapping state row(s)")

    if not observations.empty and not global_states.empty:
        first_state = global_times.iloc[0]
        final_state_end = state_ends.iloc[-1]
        if observation_times.min() < first_state or observation_times.max() >= final_state_end:
            errors.append("observation timestamps fall outside the complete global state coverage")
        if metadata is None:
            warnings.append(
                f"state coverage {first_state.isoformat()} through {final_state_end.isoformat()} "
                "is derived from observed traffic, not an explicit capture manifest"
            )
        else:
            warnings.append(
                f"state coverage uses complete windows inside metadata capture bounds: "
                f"{first_state.isoformat()} through {final_state_end.isoformat()}"
            )

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAIL: {episode} ({len(errors)} error(s))")
        return 1

    lateral_events = int((events.get("tactic", pd.Series(dtype=str)).astype(str) == "Lateral Movement").sum())
    print(
        f"PASS: {episode} | states={len(global_states)} observations={len(observations)} "
        f"nodes={len(nodes)} edges={len(edges)} events={covered_events}/{len(events)} "
        f"lateral_events={lateral_events}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
