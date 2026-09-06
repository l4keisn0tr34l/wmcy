#!/usr/bin/env python3
"""Create fixed-shape chronological graph sequences for the MVP world model.

Only observable global/node/edge state values enter context_states. Ground-truth
ATT&CK and lateral-movement values are emitted as separate future targets and
sample-audit metadata. Whole-episode splits must already be assigned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KNOWN_HOSTS = ("10.77.0.20", "10.77.0.30", "10.77.0.40")
HOST_NAMES = {"ws1": "10.77.0.20", "srv1": "10.77.0.30", "srv2": "10.77.0.40"}
DIRECTED_PAIRS = tuple((src, dst) for src in KNOWN_HOSTS for dst in KNOWN_HOSTS if src != dst)
TECHNIQUES = ("T1046", "T1110.001", "T1021.004")


def split_values(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split(";") if part.strip()}


def state_lateral_edges(global_states: pd.DataFrame, truth: pd.DataFrame, window_seconds: int) -> np.ndarray:
    output = np.zeros((len(global_states), len(DIRECTED_PAIRS)), dtype=np.float32)
    if truth.empty:
        return output
    starts = pd.to_datetime(global_states.window_start, utc=True)
    ends = starts + pd.Timedelta(seconds=window_seconds)
    pair_index = {pair: index for index, pair in enumerate(DIRECTED_PAIRS)}
    truth = truth.copy()
    truth["start_time"] = pd.to_datetime(truth.start_time, utc=True)
    truth["end_time"] = pd.to_datetime(truth.end_time, utc=True)
    for event in truth[truth.tactic.astype(str).eq("Lateral Movement")].itertuples(index=False):
        src = HOST_NAMES.get(str(event.actor))
        dst = HOST_NAMES.get(str(event.target))
        if (src, dst) not in pair_index:
            raise ValueError(f"unknown lateral-movement host pair: {event.actor}->{event.target}")
        if event.end_time < event.start_time:
            raise ValueError("ground-truth event ends before it starts")
        if event.end_time == event.start_time:
            overlap = (event.start_time >= starts) & (event.start_time < ends)
        else:
            overlap = (event.start_time < ends) & (event.end_time > starts)
        if not overlap.any():
            raise ValueError(f"lateral event has no state: {event.actor}->{event.target}")
        output[np.flatnonzero(overlap.to_numpy()), pair_index[(src, dst)]] = 1.0
    return output


def build_episode_states(
    episode_dir: Path,
    global_features: list[str],
    node_features: list[str],
    edge_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    global_states = pd.read_csv(episode_dir / "states/global_states.csv")
    nodes = pd.read_csv(episode_dir / "states/node_states.csv.gz", low_memory=False)
    edges = pd.read_csv(episode_dir / "states/edge_states.csv.gz", low_memory=False)
    aligned = pd.read_csv(episode_dir / "state_ground_truth.csv")
    truth = pd.read_csv(episode_dir / "ground_truth.csv")

    num_states = len(global_states)
    node_width = len(node_features) + 1  # activity mask
    edge_width = len(edge_features) + 1  # presence mask
    node_tensor = np.zeros((num_states, len(KNOWN_HOSTS), node_width), dtype=np.float32)
    edge_tensor = np.zeros((num_states, len(DIRECTED_PAIRS), edge_width), dtype=np.float32)

    node_host_index = {host: index for index, host in enumerate(KNOWN_HOSTS)}
    edge_pair_index = {pair: index for index, pair in enumerate(DIRECTED_PAIRS)}
    is_internal_node_index = node_features.index("is_internal")
    internal_edge_index = edge_features.index("internal_edge")

    # Known enterprise inventory is available to the defender even during host
    # silence. Activity/presence masks distinguish a known-but-silent entity.
    node_tensor[:, :, is_internal_node_index] = 1.0
    edge_tensor[:, :, internal_edge_index] = 1.0

    for row in nodes.itertuples(index=False):
        host = str(row.host)
        if host not in node_host_index:
            continue
        state_id = int(row.state_id)
        values = [float(getattr(row, feature)) for feature in node_features]
        node_tensor[state_id, node_host_index[host], : len(node_features)] = values
        node_tensor[state_id, node_host_index[host], -1] = 1.0

    for row in edges.itertuples(index=False):
        pair = (str(row.source_ip), str(row.destination_ip))
        if pair not in edge_pair_index:
            continue
        state_id = int(row.state_id)
        values = [float(getattr(row, feature)) for feature in edge_features]
        edge_tensor[state_id, edge_pair_index[pair], : len(edge_features)] = values
        edge_tensor[state_id, edge_pair_index[pair], -1] = 1.0

    global_tensor = global_states[global_features].to_numpy(dtype=np.float32)
    states = np.concatenate(
        [global_tensor, node_tensor.reshape(num_states, -1), edge_tensor.reshape(num_states, -1)],
        axis=1,
    )
    if not np.isfinite(states).all():
        raise ValueError(f"{episode_dir}: observable state tensor contains non-finite values")

    edge_presence = edge_tensor[:, :, -1]
    future_lm = aligned.has_lateral_movement.to_numpy(dtype=np.float32)
    technique_targets = np.zeros((num_states, len(TECHNIQUES)), dtype=np.float32)
    for state_index, value in enumerate(aligned.technique_ids):
        present = split_values(value)
        for technique_index, technique in enumerate(TECHNIQUES):
            technique_targets[state_index, technique_index] = float(technique in present)
    lateral_edges = state_lateral_edges(global_states, truth, 5)
    return states, edge_presence, future_lm, technique_targets, lateral_edges, global_states


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-manifest", default=str(ROOT / "outputs/mvp/episode_manifest.csv"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp/sequences"))
    ap.add_argument("--context", type=int, default=3, help="number of past/current states")
    ap.add_argument("--horizon", type=int, default=6, help="number of future states")
    args = ap.parse_args()
    if args.context <= 0 or args.horizon <= 0:
        raise ValueError("context and horizon must be positive")

    manifest = pd.read_csv(args.episode_manifest, dtype={"episode_id": str, "split": str})
    episodes_dir = Path(args.episodes_dir)
    out_dir = Path(args.out_dir)

    example = episodes_dir / manifest.iloc[0].episode_id / "states"
    global_columns = pd.read_csv(example / "global_states.csv", nrows=0).columns.tolist()
    node_columns = pd.read_csv(example / "node_states.csv.gz", nrows=0).columns.tolist()
    edge_columns = pd.read_csv(example / "edge_states.csv.gz", nrows=0).columns.tolist()
    global_features = [c for c in global_columns if c not in {"state_id", "window_start"}]
    node_features = [c for c in node_columns if c not in {"state_id", "window_start", "host"}]
    edge_features = [c for c in edge_columns if c not in {"state_id", "window_start", "source_ip", "destination_ip"}]

    state_feature_names = list(global_features)
    for slot, host in enumerate(KNOWN_HOSTS):
        state_feature_names.extend([f"node_{slot}__{name}" for name in node_features + ["activity_mask"]])
    for slot, (src, dst) in enumerate(DIRECTED_PAIRS):
        state_feature_names.extend([f"edge_{slot}__{name}" for name in edge_features + ["presence_mask"]])

    forbidden = ("technique", "tactic", "actor", "target", "lateral", "label")
    contaminated = [name for name in state_feature_names if any(term in name.lower() for term in forbidden)]
    if contaminated:
        raise ValueError(f"ground-truth-like feature names found: {contaminated}")

    per_split: dict[str, dict[str, list[np.ndarray]]] = {}
    sample_rows: list[dict[str, object]] = []
    sample_id = 0
    for episode_row in manifest.itertuples(index=False):
        split = str(episode_row.split)
        store = per_split.setdefault(split, {
            "context_states": [], "future_states": [], "future_edge_presence": [],
            "future_lateral_movement": [], "future_techniques": [], "future_lateral_edges": [],
        })
        episode_dir = episodes_dir / episode_row.episode_id
        states, edge_presence, lm, techniques, lm_edges, global_states = build_episode_states(
            episode_dir, global_features, node_features, edge_features
        )
        times = pd.to_datetime(global_states.window_start, utc=True)
        L, H = args.context, args.horizon
        for context_end in range(L - 1, len(states) - H):
            future_slice = slice(context_end + 1, context_end + 1 + H)
            store["context_states"].append(states[context_end - L + 1 : context_end + 1])
            store["future_states"].append(states[future_slice])
            store["future_edge_presence"].append(edge_presence[future_slice])
            store["future_lateral_movement"].append(lm[future_slice])
            store["future_techniques"].append(techniques[future_slice])
            store["future_lateral_edges"].append(lm_edges[future_slice])

            future_lm_offsets = np.flatnonzero(lm[future_slice] > 0)
            sample_rows.append({
                "sample_id": sample_id,
                "split": split,
                "episode_id": episode_row.episode_id,
                "scenario": episode_row.scenario,
                "context_first_state": context_end - L + 1,
                "context_last_state": context_end,
                "future_first_state": context_end + 1,
                "future_last_state": context_end + H,
                "context_end_time": times.iloc[context_end],
                "lateral_movement_within_horizon": int(len(future_lm_offsets) > 0),
                "first_lateral_movement_lead_seconds": (
                    int((future_lm_offsets[0] + 1) * episode_row.window_seconds)
                    if len(future_lm_offsets) else ""
                ),
                "lateral_movement_already_observed": int(lm[: context_end + 1].any()),
            })
            sample_id += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for split, store in per_split.items():
        arrays = {name: np.stack(values).astype(np.float32) for name, values in store.items()}
        arrays["lateral_movement_within_horizon"] = (
            arrays["future_lateral_movement"].max(axis=1).astype(np.float32)
        )
        np.savez_compressed(out_dir / f"{split}.npz", **arrays)
        split_counts[split] = len(arrays["context_states"])

    sample_manifest = pd.DataFrame(sample_rows)
    sample_manifest.to_csv(out_dir / "sample_manifest.csv", index=False)
    metadata = {
        "context_states": args.context,
        "future_horizon_states": args.horizon,
        "window_seconds": 5,
        "context_seconds": args.context * 5,
        "future_horizon_seconds": args.horizon * 5,
        "state_feature_count": len(state_feature_names),
        "state_feature_names": state_feature_names,
        "global_feature_names": global_features,
        "node_feature_names": node_features + ["activity_mask"],
        "edge_feature_names": edge_features + ["presence_mask"],
        "node_slots": [{"slot": i, "ip": host} for i, host in enumerate(KNOWN_HOSTS)],
        "directed_edge_slots": [
            {"slot": i, "source_ip": src, "destination_ip": dst}
            for i, (src, dst) in enumerate(DIRECTED_PAIRS)
        ],
        "technique_targets": list(TECHNIQUES),
        "input_policy": "context_states contains observable features only; all security truth is separate target/audit data",
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"state feature width: {len(state_feature_names)}")
    print(f"sequence counts: {split_counts}")
    print(f"sample audit rows: {len(sample_manifest)} -> {out_dir/'sample_manifest.csv'}")
    for split in sorted(per_split):
        with np.load(out_dir / f"{split}.npz") as data:
            print(
                f"{split}: context={data['context_states'].shape} future={data['future_states'].shape} "
                f"future_lateral_positive={int(data['lateral_movement_within_horizon'].sum())}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
