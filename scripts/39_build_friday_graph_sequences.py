#!/usr/bin/env python3
"""Build causal three-host graph-dynamics sequences from Friday PCAP observables."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGE_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
FORBIDDEN = ("label", "attack", "technique", "tactic", "lateral", "actor", "target")
OBS_COLUMNS = [
    "timestamp", "source_ip", "source_port", "destination_ip", "destination_port",
    "total_fwd_packets", "total_backward_packets", "total_length_of_fwd_packets",
    "total_length_of_bwd_packets", "syn_flag_count", "ack_flag_count", "rst_flag_count",
    "fin_flag_count", "flow_duration",
]


def entropy(values: pd.Series) -> float:
    counts = values.value_counts(dropna=True).to_numpy(dtype=float)
    if not len(counts): return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def internal(host: str, network: ipaddress._BaseNetwork) -> float:
    try: return float(ipaddress.ip_address(host) in network)
    except ValueError: return 0.0


def load_observations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=OBS_COLUMNS, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    if frame.timestamp.isna().any() or frame[["source_ip", "destination_ip"]].isna().any().any():
        raise ValueError("canonical Friday observations contain invalid key fields")
    if not frame.timestamp.is_monotonic_increasing: raise ValueError("canonical observations are not chronological")
    numeric = [name for name in OBS_COLUMNS if name not in {"timestamp", "source_ip", "destination_ip"}]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[numeric].isna().any().any() or not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("non-finite Friday observable")
    if (frame[numeric] < 0).any().any(): raise ValueError("negative Friday observable")
    frame["source_ip"] = frame.source_ip.astype(str); frame["destination_ip"] = frame.destination_ip.astype(str)
    frame["epoch"] = frame.timestamp.map(lambda value: int(value.timestamp())).astype("int64")
    frame["window_epoch"] = (frame.epoch // 5) * 5
    frame["bytes"] = frame.total_length_of_fwd_packets + frame.total_length_of_bwd_packets
    frame["packets"] = frame.total_fwd_packets + frame.total_backward_packets
    return frame


def choose_roster(context: pd.DataFrame, dataset_id: str, start: int) -> tuple[str, ...] | None:
    degree = Counter(context.source_ip); degree.update(context.destination_ip)
    if len(degree) < 3: return None
    pair_counts = context.groupby(["source_ip", "destination_ip"], sort=True).size().sort_values(ascending=False)
    first_pair = next(((str(source), str(destination)) for source, destination in pair_counts.index
                       if source != destination), None)
    if first_pair is None: return None
    hosts = list(first_pair)
    hosts.append(next(host for host, _ in sorted(degree.items(), key=lambda item: (-item[1], item[0]))
                      if host not in hosts))
    seed = int.from_bytes(hashlib.sha256(f"{dataset_id}:{start}".encode()).digest()[:8], "little")
    np.random.default_rng(seed).shuffle(hosts)
    return tuple(hosts)


def state_vector(frame: pd.DataFrame, roster: tuple[str, ...], first_seen: dict[tuple[str, str], int],
                 window_epoch: int, network: ipaddress._BaseNetwork) -> tuple[np.ndarray, np.ndarray]:
    host_set = set(roster)
    graph = frame[frame.source_ip.isin(host_set) & frame.destination_ip.isin(host_set)]
    pairs = set(zip(graph.source_ip, graph.destination_ip))
    new_pairs = {pair for pair in pairs if first_seen[pair] == window_epoch}
    internal_map = {host: internal(host, network) for host in roster}
    fanout = graph.groupby("source_ip").destination_ip.nunique() if len(graph) else pd.Series(dtype=float)
    global_features = [
        len(graph), len(set(graph.source_ip).union(graph.destination_ip)) if len(graph) else 0,
        graph.source_ip.nunique(), graph.destination_ip.nunique(), len(pairs), len(new_pairs),
        sum(internal_map.get(source, 0) * internal_map.get(destination, 0) for source, destination in pairs),
        graph.bytes.sum(), graph.packets.sum(), graph.syn_flag_count.sum(), graph.rst_flag_count.sum(),
        graph.fin_flag_count.sum(), fanout.max() if len(fanout) else 0,
        fanout.mean() if len(fanout) else 0, entropy(graph.destination_port) if len(graph) else 0,
    ]
    node_features = []
    for host in roster:
        outgoing = graph[graph.source_ip.eq(host)]; incoming = graph[graph.destination_ip.eq(host)]
        node_features.extend([
            len(outgoing), outgoing.bytes.sum(), outgoing.packets.sum(), outgoing.destination_ip.nunique(),
            outgoing.destination_port.nunique(), outgoing.syn_flag_count.sum(), outgoing.rst_flag_count.sum(),
            len(incoming), incoming.bytes.sum(), incoming.packets.sum(), incoming.source_ip.nunique(),
            incoming.source_port.nunique(), incoming.syn_flag_count.sum(), incoming.rst_flag_count.sum(),
            internal_map[host], sum(pair[0] == host for pair in new_pairs),
            sum(pair[1] == host for pair in new_pairs), float(len(outgoing) + len(incoming) > 0),
        ])
    edge_features = []; presence = []
    for source_slot, destination_slot in EDGE_PAIRS:
        source, destination = roster[source_slot], roster[destination_slot]
        edge = graph[graph.source_ip.eq(source) & graph.destination_ip.eq(destination)]
        active = float(len(edge) > 0); presence.append(active)
        edge_features.extend([
            len(edge), edge.bytes.sum(), edge.packets.sum(), edge.syn_flag_count.sum(),
            edge.ack_flag_count.sum(), edge.rst_flag_count.sum(), edge.fin_flag_count.sum(),
            edge.flow_duration.mean() if len(edge) else 0, edge.destination_port.nunique(),
            internal_map[source] * internal_map[destination], float((source, destination) in new_pairs), active,
        ])
    vector = np.asarray(global_features + node_features + edge_features, dtype=np.float32)
    if vector.shape != (141,) or not np.isfinite(vector).all(): raise ValueError(f"invalid state {vector.shape}")
    return vector, np.asarray(presence, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", default=str(ROOT / "outputs/cic2017_friday_pcap/canonical/observations.csv.gz"))
    ap.add_argument("--canonical-manifest", default=str(ROOT / "outputs/cic2017_friday_pcap/canonical/manifest.json"))
    ap.add_argument("--lab-feature-metadata", default=str(ROOT / "outputs/mvp_v2/sequences/feature_metadata.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/cic2017_friday_pcap/graph_sequences"))
    ap.add_argument("--context", type=int, default=3); ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--stride-seconds", type=int, default=5); ap.add_argument("--internal-cidr", default="192.168.10.0/24")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); expected_outputs = ["train.npz", "sample_manifest.csv", "feature_metadata.json"]
    if not args.force and any((out / name).exists() for name in expected_outputs):
        raise FileExistsError("refusing to replace Friday sequences without --force")
    if args.stride_seconds <= 0 or args.stride_seconds % 5: raise ValueError("stride must be a positive multiple of 5")
    observations = Path(args.observations); canonical = json.loads(Path(args.canonical_manifest).read_text())
    metadata = json.loads(Path(args.lab_feature_metadata).read_text())
    contaminated = [name for name in metadata["state_feature_names"] if any(x in name.lower() for x in FORBIDDEN)]
    if contaminated: raise ValueError(f"contaminated feature metadata: {contaminated}")
    frame = load_observations(observations)
    if len(frame) != canonical["canonical_rows"]: raise ValueError("canonical row count disagrees with manifest")
    first_seen = frame.groupby(["source_ip", "destination_ip"], sort=False).window_epoch.min().to_dict()
    grouped = {int(epoch): frame.iloc[index] for epoch, index in frame.groupby("window_epoch", sort=False).indices.items()}
    empty = frame.iloc[0:0]; network = ipaddress.ip_network(args.internal_cidr)
    capture_min = pd.Timestamp(canonical["capture_timestamp_min"]).timestamp()
    capture_max = pd.Timestamp(canonical["capture_timestamp_max"]).timestamp()
    first_complete = int(math.ceil(capture_min / 5) * 5)
    last_complete = int(math.floor(capture_max / 5) * 5 - 5)
    total_states = args.context + args.horizon
    last_sample_start = last_complete - (total_states - 1) * 5
    contexts = []; futures = []; future_edges = []; audits = []; skipped = 0
    for start in range(first_complete, last_sample_start + 1, args.stride_seconds):
        windows = [grouped.get(start + index * 5, empty) for index in range(total_states)]
        context_rows = pd.concat(windows[:args.context], ignore_index=True)
        roster = choose_roster(context_rows, canonical["dataset_id"], start)
        if roster is None:
            skipped += 1; continue
        states = []; edges = []
        for index, window in enumerate(windows):
            state, edge = state_vector(window, roster, first_seen, start + index * 5, network)
            states.append(state); edges.append(edge)
        state_array = np.stack(states); edge_array = np.stack(edges)
        if state_array[:args.context, 0].sum() <= 0: raise AssertionError("chosen context roster is inactive")
        contexts.append(state_array[:args.context]); futures.append(state_array[args.context:])
        future_edges.append(edge_array[args.context:])
        hashes = [hashlib.sha256(host.encode()).hexdigest()[:12] for host in roster]
        audits.append({
            "split": "train", "capture_group": "friday_working_hours_single_capture",
            "sample_start": pd.to_datetime(start, unit="s", utc=True).isoformat(),
            "context_end": pd.to_datetime(start + args.context * 5, unit="s", utc=True).isoformat(),
            "future_end": pd.to_datetime(start + total_states * 5, unit="s", utc=True).isoformat(),
            "roster_hashes": ";".join(hashes),
            "context_induced_flows": int(state_array[:args.context, 0].sum()),
            "future_induced_flows": int(state_array[args.context:, 0].sum()),
        })
        if args.max_samples is not None and len(contexts) >= args.max_samples: break
    if not contexts: raise ValueError("no valid Friday sequences")
    context_array = np.stack(contexts).astype(np.float32)
    future_array = np.stack(futures).astype(np.float32)
    edge_array = np.stack(future_edges).astype(np.float32)
    if not all(np.isfinite(value).all() for value in [context_array, future_array, edge_array]):
        raise ValueError("non-finite output sequence")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="friday-graph-sequences-", dir=out.parent))
    metadata.pop("technique_targets", None)
    profile = {
        "source_rows": len(frame), "candidate_start": first_complete, "candidate_end": last_sample_start,
        "samples": len(context_array), "skipped_candidates": skipped,
        "context_shape": list(context_array.shape), "future_shape": list(future_array.shape),
        "edge_shape": list(edge_array.shape), "active_future_fraction": float((future_array[:, :, 0] > 0).mean()),
        "unique_directed_pairs_in_capture": len(first_seen),
    }
    public_metadata = {
        **metadata, "source": "CICIDS2017 Friday PCAP observable telemetry only",
        "context_states": args.context, "future_horizon_states": args.horizon,
        "node_slots": [{"slot": index, "identity": "anonymized_context_host"} for index in range(3)],
        "directed_edge_slots": [{"slot": index, "source_slot": source, "destination_slot": destination}
                                for index, (source, destination) in enumerate(EDGE_PAIRS)],
        "feature_availability": {"syn_count": True, "ack_count": True, "rst_count": True, "fin_count": True},
        "roster_policy": "most frequent context edge plus highest-degree third host; deterministic slot anonymization; future never consulted",
        "novelty_policy": "directed edge is new only in its first five-second capture window; computed from present/past observable time",
        "split_policy": "single connected Friday capture retained as train-only pretraining data; no internal validation claim",
        "target_policy": "future observable state and edge presence only; no CIC labels loaded",
        "canonical_observations_sha256": canonical["output_sha256"],
        "internal_cidr": args.internal_cidr, "stride_seconds": args.stride_seconds, "profile": profile,
    }
    try:
        np.savez_compressed(work / "train.npz", context_states=context_array, future_states=future_array,
                            future_edge_presence=edge_array)
        pd.DataFrame(audits).to_csv(work / "sample_manifest.csv", index=False)
        (work / "feature_metadata.json").write_text(json.dumps(public_metadata, indent=2) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        for name in expected_outputs: os.replace(work / name, out / name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"Friday train: context={context_array.shape} future={future_array.shape} edge={edge_array.shape}")
    print(f"skipped={skipped}; pairs={len(first_seen):,}; metadata -> {out/'feature_metadata.json'}")
    return 0


if __name__ == "__main__": sys.exit(main())
