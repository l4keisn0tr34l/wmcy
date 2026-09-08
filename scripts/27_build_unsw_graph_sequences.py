#!/usr/bin/env python3
"""Build leakage-safe three-host graph-dynamics sequences from canonical UNSW.

This script reads observable files only. Host rosters are chosen from each
sample's context and never from its future or from UNSW attack labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGE_PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
FORBIDDEN = ("label", "attack", "technique", "tactic", "lateral", "actor", "target")
OBS_COLUMNS = [
    "timestamp", "source_ip", "source_port", "destination_ip", "destination_port",
    "total_fwd_packets", "total_backward_packets", "total_length_of_fwd_packets",
    "total_length_of_bwd_packets", "flow_duration",
]


def entropy(values: pd.Series) -> float:
    counts = values.value_counts(dropna=True).to_numpy(dtype=float)
    if not len(counts): return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def internal(host: str, network: ipaddress._BaseNetwork) -> float:
    try: return float(ipaddress.ip_address(host) in network)
    except ValueError: return 0.0


def load_capture_group(manifest: dict, group: dict) -> tuple[pd.DataFrame, int]:
    rows = []
    lookup = {row["dataset_id"]: row for row in manifest["segments"]}
    for dataset_id in group["dataset_ids"]:
        rows.append(pd.read_csv(lookup[dataset_id]["observations"], usecols=OBS_COLUMNS, low_memory=False))
    frame = pd.concat(rows, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=["timestamp", "source_ip", "destination_ip"])
    removed = before - len(frame)
    # Preserve repeated records: identical flow summaries can be legitimate
    # repeated observations. Source files overlap in clock time but are merged
    # into one capture group rather than treated as independent splits.
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    # Do not assume pandas' internal datetime unit (newer releases may retain
    # seconds rather than nanoseconds). Convert explicitly to Unix seconds.
    frame["epoch"] = frame.timestamp.map(lambda value: int(value.timestamp())).astype("int64")
    frame["window_epoch"] = (frame.epoch // 5) * 5
    for column in OBS_COLUMNS[2:]:
        if column not in {"destination_ip"}:
            if column not in {"source_ip", "timestamp"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).clip(lower=0)
    frame["bytes"] = frame.total_length_of_fwd_packets + frame.total_length_of_bwd_packets
    frame["packets"] = frame.total_fwd_packets + frame.total_backward_packets
    return frame, removed


def choose_roster(context: pd.DataFrame, group_name: str, start: int) -> tuple[str, ...] | None:
    degree = Counter(context.source_ip.astype(str)); degree.update(context.destination_ip.astype(str))
    if len(degree) < 3: return None
    # Seed the roster with an actually observed directed pair. Picking only the
    # three highest-degree hosts can select unrelated hubs and yield an empty
    # induced graph even though the context is busy.
    pair_counts = context.groupby(["source_ip", "destination_ip"], sort=True).size().sort_values(ascending=False)
    first_pair = next(((str(src), str(dst)) for (src, dst) in pair_counts.index if str(src) != str(dst)), None)
    if first_pair is None: return None
    hosts = list(first_pair)
    third = next(host for host, _ in sorted(degree.items(), key=lambda item: (-item[1], item[0])) if host not in hosts)
    hosts.append(third)
    seed = int.from_bytes(hashlib.sha256(f"{group_name}:{start}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed); rng.shuffle(hosts)
    return tuple(hosts)


def state_vector(frame: pd.DataFrame, roster: tuple[str, ...], seen: set[tuple[str, str]],
                 network: ipaddress._BaseNetwork) -> tuple[np.ndarray, np.ndarray]:
    host_set = set(roster)
    g = frame[frame.source_ip.isin(host_set) & frame.destination_ip.isin(host_set)]
    pairs = set(zip(g.source_ip.astype(str), g.destination_ip.astype(str)))
    new_pairs = pairs - seen
    seen.update(pairs)
    internal_map = {host: internal(host, network) for host in roster}
    out_fanout = g.groupby("source_ip").destination_ip.nunique() if len(g) else pd.Series(dtype=float)
    global_features = [
        len(g), len(set(g.source_ip).union(g.destination_ip)) if len(g) else 0,
        g.source_ip.nunique(), g.destination_ip.nunique(), len(pairs), len(new_pairs),
        sum(internal_map[s] * internal_map[d] for s, d in pairs),
        g.bytes.sum(), g.packets.sum(), 0, 0, 0,
        out_fanout.max() if len(out_fanout) else 0,
        out_fanout.mean() if len(out_fanout) else 0,
        entropy(g.destination_port) if len(g) else 0,
    ]
    node_features = []
    for host in roster:
        outgoing = g[g.source_ip.astype(str).eq(host)]
        incoming = g[g.destination_ip.astype(str).eq(host)]
        node_features.extend([
            len(outgoing), outgoing.bytes.sum(), outgoing.packets.sum(), outgoing.destination_ip.nunique(),
            outgoing.destination_port.nunique(), 0, 0,
            len(incoming), incoming.bytes.sum(), incoming.packets.sum(), incoming.source_ip.nunique(),
            incoming.source_port.nunique(), 0, 0, internal_map[host],
            sum(1 for pair in new_pairs if pair[0] == host),
            sum(1 for pair in new_pairs if pair[1] == host),
            float(len(outgoing) + len(incoming) > 0),
        ])
    edge_features = []; presence = []
    for source_slot, destination_slot in EDGE_PAIRS:
        source, destination = roster[source_slot], roster[destination_slot]
        edge = g[g.source_ip.astype(str).eq(source) & g.destination_ip.astype(str).eq(destination)]
        active = float(len(edge) > 0); presence.append(active)
        edge_features.extend([
            len(edge), edge.bytes.sum(), edge.packets.sum(), 0, 0, 0, 0,
            edge.flow_duration.mean() if len(edge) else 0,
            edge.destination_port.nunique(), internal_map[source] * internal_map[destination],
            float((source, destination) in new_pairs), active,
        ])
    vector = np.asarray(global_features + node_features + edge_features, dtype=np.float32)
    if vector.shape != (141,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid state vector {vector.shape}")
    return vector, np.asarray(presence, dtype=np.float32)


def build_group(frame: pd.DataFrame, group_name: str, context_states: int, horizon: int,
                stride: int, network: ipaddress._BaseNetwork, max_samples: int | None):
    grouped = {int(key): frame.iloc[index] for key, index in frame.groupby("window_epoch", sort=False).indices.items()}
    first = int(math.ceil(frame.epoch.min() / 5) * 5)
    last_start = int(math.floor((frame.epoch.max() - (context_states + horizon) * 5 + 1) / 5) * 5)
    contexts = []; futures = []; future_edges = []; audits = []; skipped = 0
    for start in range(first, last_start + 1, stride):
        windows = [grouped.get(start + offset * 5, frame.iloc[0:0]) for offset in range(context_states + horizon)]
        context_rows = pd.concat(windows[:context_states], ignore_index=True)
        roster = choose_roster(context_rows, group_name, start)
        if roster is None:
            skipped += 1; continue
        seen: set[tuple[str, str]] = set(); states = []; edges = []
        for window in windows:
            state, presence = state_vector(window, roster, seen, network)
            states.append(state); edges.append(presence)
        states_array = np.stack(states); edges_array = np.stack(edges)
        if states_array[:context_states, 0].sum() <= 0:
            skipped += 1; continue
        contexts.append(states_array[:context_states]); futures.append(states_array[context_states:])
        future_edges.append(edges_array[context_states:])
        roster_hashes = [hashlib.sha256(host.encode()).hexdigest()[:12] for host in roster]
        audits.append({
            "capture_group": group_name, "sample_start": pd.to_datetime(start, unit="s", utc=True).isoformat(),
            "context_end": pd.to_datetime(start + context_states * 5, unit="s", utc=True).isoformat(),
            "future_end": pd.to_datetime(start + (context_states + horizon) * 5, unit="s", utc=True).isoformat(),
            "roster_hashes": ";".join(roster_hashes), "context_induced_flows": int(states_array[:context_states, 0].sum()),
            "future_induced_flows": int(states_array[context_states:, 0].sum()),
        })
        if max_samples is not None and len(contexts) >= max_samples: break
    if not contexts: raise ValueError(f"no valid samples for {group_name}")
    return (np.stack(contexts).astype(np.float32), np.stack(futures).astype(np.float32),
            np.stack(future_edges).astype(np.float32), audits, skipped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-manifest", default=str(ROOT / "outputs/unsw/canonical/manifest.json"))
    ap.add_argument("--lab-feature-metadata", default=str(ROOT / "outputs/mvp_v2/sequences/feature_metadata.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/unsw/graph_sequences"))
    ap.add_argument("--context", type=int, default=3); ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--stride-seconds", type=int, default=30); ap.add_argument("--internal-cidr", default="149.171.126.0/24")
    ap.add_argument("--max-samples-per-group", type=int, default=None)
    args = ap.parse_args()
    if args.stride_seconds <= 0 or args.stride_seconds % 5: raise ValueError("stride must be a positive multiple of 5")
    manifest = json.loads(Path(args.canonical_manifest).read_text())
    metadata = json.loads(Path(args.lab_feature_metadata).read_text())
    contaminated = [name for name in metadata["state_feature_names"] if any(word in name.lower() for word in FORBIDDEN)]
    if contaminated: raise ValueError(f"contaminated state feature names: {contaminated}")
    groups = manifest["capture_groups"]
    if len(groups) != 2: raise ValueError("expected exactly two disconnected UNSW capture groups")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "validation"); profiles = []; audit_frames = []
    network = ipaddress.ip_network(args.internal_cidr)
    for split, group in zip(split_names, groups):
        frame, duplicates = load_capture_group(manifest, group)
        context, future, edge, audits, skipped = build_group(
            frame, group["capture_group"], args.context, args.horizon, args.stride_seconds,
            network, args.max_samples_per_group,
        )
        np.savez_compressed(out / f"{split}.npz", context_states=context, future_states=future,
                            future_edge_presence=edge)
        audit = pd.DataFrame(audits); audit.insert(0, "split", split); audit_frames.append(audit)
        profiles.append({
            "split": split, "capture_group": group["capture_group"], "source_rows": len(frame),
            "invalid_rows_removed": duplicates, "samples": len(context), "skipped_candidates": skipped,
            "context_shape": list(context.shape), "future_shape": list(future.shape),
            "active_future_fraction": float((future[:, :, 0] > 0).mean()),
        })
        print(f"{split}: context={context.shape} future={future.shape} edge={edge.shape} invalid_removed={duplicates:,}")
    pd.concat(audit_frames, ignore_index=True).to_csv(out / "sample_manifest.csv", index=False)
    metadata.pop("technique_targets", None)
    public_metadata = {
        **metadata, "source": "UNSW-NB15 observable telemetry only", "context_states": args.context,
        "future_horizon_states": args.horizon, "node_slots": [{"slot": i, "identity": "anonymized_context_host"} for i in range(3)],
        "directed_edge_slots": [{"slot": i, "source_slot": s, "destination_slot": d} for i, (s, d) in enumerate(EDGE_PAIRS)],
        "feature_availability": {"syn_count": False, "ack_count": False, "rst_count": False, "fin_count": False},
        "roster_policy": "most frequent context edge plus highest-degree third host; deterministic per-sample slot anonymization; future never consulted",
        "split_policy": "January and February connected capture groups remain disjoint",
        "target_policy": "future observable state and edge presence only; no UNSW labels loaded",
        "profiles": profiles,
    }
    (out / "feature_metadata.json").write_text(json.dumps(public_metadata, indent=2) + "\n")
    print(f"metadata -> {out / 'feature_metadata.json'}")
    return 0


if __name__ == "__main__": sys.exit(main())
