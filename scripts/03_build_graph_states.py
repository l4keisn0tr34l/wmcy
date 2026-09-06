#!/usr/bin/env python3
"""Turn canonical flow events into chronological graph-structured network states.

A state is NOT a flow label. For each time window we produce:
  - global_states.csv : network-wide features
  - node_states.csv.gz: per-host behavioral state
  - edge_states.csv.gz: per-source/destination communication state

These are the observable S_t used by the future world model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import ipaddress
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cyberwm.common import shannon_entropy, safe_numeric


def is_internal(ip: object, cidr: ipaddress._BaseNetwork) -> bool:
    try:
        return ipaddress.ip_address(str(ip)) in cidr
    except ValueError:
        return False


def pick(df: pd.DataFrame, names: list[str], default=0.0) -> pd.Series:
    for n in names:
        if n in df.columns:
            return safe_numeric(df[n]).fillna(0)
    return pd.Series(default, index=df.index, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("observations")
    ap.add_argument("--window", default="60s")
    ap.add_argument("--internal-cidr", default="192.168.10.0/24")
    ap.add_argument("--capture-start", help="inclusive capture start; requires --capture-end")
    ap.add_argument("--capture-end", help="exclusive capture end; requires --capture-start")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.observations, low_memory=False)
    df["timestamp"] = pd.to_datetime(
    	df["timestamp"],
    	errors="coerce",
    	utc=True
    )
    df = df.dropna(subset=["timestamp", "source_ip", "destination_ip"]).copy()
    df = df.sort_values("timestamp", kind="stable")
    df["window_start"] = df["timestamp"].dt.floor(args.window)

    if df.empty:
        raise ValueError("no valid observations after timestamp/source/destination filtering")

    window_delta = pd.to_timedelta(args.window)
    if window_delta <= pd.Timedelta(0):
        raise ValueError("--window must be a positive duration")
    if bool(args.capture_start) != bool(args.capture_end):
        raise ValueError("--capture-start and --capture-end must be provided together")

    # Dense fixed-time state grid. Empty traffic windows are still states, because
    # a world model needs state_id + 1 to mean exactly one configured time step.
    # With capture bounds, retain only fully observed windows: ceil(start) through
    # floor(end), excluding the end. This avoids treating partial boundary windows
    # as if they represented a complete interval.
    if args.capture_start:
        capture_start = pd.to_datetime(args.capture_start, errors="raise", utc=True)
        capture_end = pd.to_datetime(args.capture_end, errors="raise", utc=True)
        if capture_end <= capture_start:
            raise ValueError("--capture-end must be after --capture-start")
        first_window = capture_start.ceil(args.window)
        grid_end = capture_end.floor(args.window)
        if grid_end <= first_window:
            raise ValueError("capture interval contains no complete state window")
        outside_complete_grid = (df["timestamp"] < first_window) | (df["timestamp"] >= grid_end)
        if outside_complete_grid.any():
            examples = df.loc[outside_complete_grid, "timestamp"].head(3).tolist()
            raise ValueError(
                f"{int(outside_complete_grid.sum())} observation(s) fall in partial capture-boundary "
                f"windows; examples={examples}"
            )
    else:
        first_window = df["window_start"].min()
        grid_end = df["window_start"].max() + window_delta
    all_windows = pd.date_range(start=first_window, end=grid_end, freq=window_delta, inclusive="left")

    # Normalize useful primitive flow quantities while retaining the canonical events separately.
    df["fwd_packets_x"] = pick(df, ["total_fwd_packets"])
    df["bwd_packets_x"] = pick(df, ["total_backward_packets"])
    df["fwd_bytes_x"] = pick(df, ["total_length_of_fwd_packets"])
    df["bwd_bytes_x"] = pick(df, ["total_length_of_bwd_packets"])
    df["syn_x"] = pick(df, ["syn_flag_count"])
    df["ack_x"] = pick(df, ["ack_flag_count"])
    df["rst_x"] = pick(df, ["rst_flag_count"])
    df["fin_x"] = pick(df, ["fin_flag_count"])
    df["flow_duration_x"] = pick(df, ["flow_duration"])
    df["src_port_x"] = pick(df, ["source_port"])
    df["dst_port_x"] = pick(df, ["destination_port"])
    df["bytes_x"] = df["fwd_bytes_x"] + df["bwd_bytes_x"]
    df["packets_x"] = df["fwd_packets_x"] + df["bwd_packets_x"]

    cidr = ipaddress.ip_network(args.internal_cidr)
    df["src_internal"] = df["source_ip"].map(lambda x: is_internal(x, cidr))
    df["dst_internal"] = df["destination_ip"].map(lambda x: is_internal(x, cidr))
    df["internal_edge"] = df["src_internal"] & df["dst_internal"]

    # -------------------- EDGE STATE --------------------
    edge_group = df.groupby(["window_start", "source_ip", "destination_ip"], sort=True)
    edges = edge_group.agg(
        flow_count=("event_id", "count"),
        bytes_total=("bytes_x", "sum"),
        packets_total=("packets_x", "sum"),
        syn_count=("syn_x", "sum"),
        ack_count=("ack_x", "sum"),
        rst_count=("rst_x", "sum"),
        fin_count=("fin_x", "sum"),
        mean_flow_duration=("flow_duration_x", "mean"),
        unique_dst_ports=("dst_port_x", "nunique"),
        internal_edge=("internal_edge", "max"),
    ).reset_index()

    # New edge = this ordered host pair has never appeared in an earlier state.
    edges = edges.sort_values(["window_start", "source_ip", "destination_ip"])
    first_seen = edges.groupby(["source_ip", "destination_ip"])["window_start"].transform("min")
    edges["is_new_edge"] = (edges["window_start"] == first_seen).astype(int)

    # -------------------- NODE STATE --------------------
    outg = df.groupby(["window_start", "source_ip"]).agg(
        outgoing_flows=("event_id", "count"),
        outgoing_bytes=("bytes_x", "sum"),
        outgoing_packets=("packets_x", "sum"),
        out_neighbors=("destination_ip", "nunique"),
        unique_dst_ports=("dst_port_x", "nunique"),
        out_syn=("syn_x", "sum"),
        out_rst=("rst_x", "sum"),
    ).reset_index().rename(columns={"source_ip": "host"})

    inc = df.groupby(["window_start", "destination_ip"]).agg(
        incoming_flows=("event_id", "count"),
        incoming_bytes=("bytes_x", "sum"),
        incoming_packets=("packets_x", "sum"),
        in_neighbors=("source_ip", "nunique"),
        unique_src_ports=("src_port_x", "nunique"),
        in_syn=("syn_x", "sum"),
        in_rst=("rst_x", "sum"),
    ).reset_index().rename(columns={"destination_ip": "host"})

    nodes = outg.merge(inc, on=["window_start", "host"], how="outer").fillna(0)
    nodes["is_internal"] = nodes["host"].map(lambda x: is_internal(x, cidr)).astype(int)

    # New peer count is calculated from first-ever edge appearances in each direction.
    new_out = edges[edges["is_new_edge"] == 1].groupby(["window_start", "source_ip"]).size().rename("new_out_neighbors").reset_index().rename(columns={"source_ip":"host"})
    new_in = edges[edges["is_new_edge"] == 1].groupby(["window_start", "destination_ip"]).size().rename("new_in_neighbors").reset_index().rename(columns={"destination_ip":"host"})
    nodes = nodes.merge(new_out, on=["window_start","host"], how="left").merge(new_in, on=["window_start","host"], how="left")
    nodes[["new_out_neighbors","new_in_neighbors"]] = nodes[["new_out_neighbors","new_in_neighbors"]].fillna(0)

    # -------------------- GLOBAL STATE --------------------
    global_rows = []
    grouped = {w: g for w, g in df.groupby("window_start", sort=True)}
    for w in all_windows:
        g = grouped.get(w)
        if g is None:
            global_rows.append({
                "window_start": w,
                "flow_count": 0,
                "unique_hosts": 0,
                "unique_src_hosts": 0,
                "unique_dst_hosts": 0,
                "unique_edges": 0,
                "new_edges": 0,
                "internal_edges": 0,
                "total_bytes": 0.0,
                "total_packets": 0.0,
                "syn_count": 0.0,
                "rst_count": 0.0,
                "fin_count": 0.0,
                "max_out_fanout": 0.0,
                "mean_out_fanout": 0.0,
                "dst_port_entropy": 0.0,
            })
            continue

        hosts = pd.unique(pd.concat([g["source_ip"], g["destination_ip"]], ignore_index=True))
        ew = edges[edges["window_start"] == w]
        nw = nodes[nodes["window_start"] == w]
        global_rows.append({
            "window_start": w,
            "flow_count": len(g),
            "unique_hosts": len(hosts),
            "unique_src_hosts": g["source_ip"].nunique(),
            "unique_dst_hosts": g["destination_ip"].nunique(),
            "unique_edges": len(ew),
            "new_edges": int(ew["is_new_edge"].sum()),
            "internal_edges": int(ew["internal_edge"].sum()),
            "total_bytes": float(g["bytes_x"].sum()),
            "total_packets": float(g["packets_x"].sum()),
            "syn_count": float(g["syn_x"].sum()),
            "rst_count": float(g["rst_x"].sum()),
            "fin_count": float(g["fin_x"].sum()),
            "max_out_fanout": float(nw["out_neighbors"].max()) if len(nw) else 0.0,
            "mean_out_fanout": float(nw["out_neighbors"].mean()) if len(nw) else 0.0,
            "dst_port_entropy": shannon_entropy(g["dst_port_x"]),
        })
    global_df = pd.DataFrame(global_rows).sort_values("window_start")
    global_df.insert(0, "state_id", np.arange(len(global_df), dtype=np.int64))

    # Add state_id to node/edge tables for easy graph loading later.
    state_map = global_df.set_index("window_start")["state_id"]
    nodes.insert(0, "state_id", nodes["window_start"].map(state_map).astype("int64"))
    edges.insert(0, "state_id", edges["window_start"].map(state_map).astype("int64"))

    global_df.to_csv(out / "global_states.csv", index=False)
    nodes.to_csv(out / "node_states.csv.gz", index=False, compression="gzip")
    edges.to_csv(out / "edge_states.csv.gz", index=False, compression="gzip")

    print(f"states: {len(global_df):,} -> {out/'global_states.csv'}")
    print(f"node-state rows: {len(nodes):,} -> {out/'node_states.csv.gz'}")
    print(f"edge-state rows: {len(edges):,} -> {out/'edge_states.csv.gz'}")


if __name__ == "__main__":
    main()
