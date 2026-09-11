"""Strict V5 telemetry/target separation and five-host sequence contract."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.cyberwm.v5_contract import HOSTS, IPS, PAIRS, TECHNIQUES, ACTIONS, GLOBAL, NODE, EDGE, WIDTH, roles


def require(condition, message):
    if not condition: raise ValueError(message)


def times(series):
    require(series.astype(str).str.contains(r"(?:Z|[+-]\d\d:?\d\d)$").all(), "timezone missing")
    return pd.to_datetime(series, utc=True, errors="raise", format="mixed")


def schema(frame, columns, name):
    require(frame.columns.tolist() == list(columns), f"{name}: unexpected columns/order (including forbidden extras)")


def load_observable_states(episode: Path):
    """No truth, scenario, role, action, or future-derived roster is read here."""
    global_df = pd.read_csv(episode / "states/global_states.csv")
    nodes = pd.read_csv(episode / "states/node_states.csv.gz")
    edges = pd.read_csv(episode / "states/edge_states.csv.gz")
    schema(global_df, ["state_id", "window_start"]+GLOBAL, "globals")
    schema(nodes, ["state_id", "window_start", "host"]+NODE[:-1], "nodes")
    schema(edges, ["state_id", "window_start", "source_ip", "destination_ip"]+EDGE[:-1], "edges")
    count = len(global_df); starts = times(global_df.window_start)
    require(count >= 9, "insufficient dense states")
    require(np.array_equal(global_df.state_id, np.arange(count)), "noncontiguous global IDs")
    require((starts.diff().iloc[1:] == pd.Timedelta(seconds=5)).all(), "nondense time")
    nt = np.zeros((count, 5, 18), np.float32); et = np.zeros((count, 20, 12), np.float32)
    nt[..., NODE.index("is_internal")] = 1; et[..., EDGE.index("internal_edge")] = 1
    for frame, ids in [(nodes, ["host"]), (edges, ["source_ip", "destination_ip"])]:
        require(not frame.duplicated(["state_id"]+ids).any(), "duplicate state entity")
        refs = frame.state_id.to_numpy()
        require(np.isin(refs, np.arange(count)).all(), "invalid state reference")
        require(np.array_equal(times(frame.window_start).to_numpy(), starts.iloc[refs.astype(int)].to_numpy()), "entity time mismatch")
    for row in nodes.itertuples(index=False):
        require(row.host in IPS, f"unknown observable host {row.host}; do not silently drop it")
        nt[row.state_id, IPS.index(row.host), :-1] = [getattr(row, f) for f in NODE[:-1]]
        nt[row.state_id, IPS.index(row.host), -1] = 1
    for row in edges.itertuples(index=False):
        pair = row.source_ip, row.destination_ip
        require(pair in PAIRS, f"unknown/self observable pair {pair}")
        et[row.state_id, PAIRS.index(pair), :-1] = [getattr(row, f) for f in EDGE[:-1]]
        et[row.state_id, PAIRS.index(pair), -1] = 1
    g = global_df[GLOBAL].to_numpy(np.float32)
    states = np.concatenate([g, nt.reshape(count, -1), et.reshape(count, -1)], axis=1)
    require(states.shape == (count, WIDTH) and np.isfinite(states).all(), "nonfinite state features")
    require((states >= -1e-6).all(), "negative telemetry feature")
    require(np.array_equal(g[:, GLOBAL.index("flow_count")], et[..., EDGE.index("flow_count")].sum(1)), "global/edge flow mismatch")
    require(np.array_equal(g[:, GLOBAL.index("unique_edges")], et[..., -1].sum(1)), "global/pair coverage mismatch")
    return states, et[..., -1], starts


def load_targets(episode: Path, starts):
    """Separate supervision; recompute exact half-open truth, including attempts."""
    truth = pd.read_csv(episode/"ground_truth.csv")
    schema(truth, "episode_id start_time end_time actor target technique_id technique tactic".split(), "truth")
    tech = np.zeros((len(starts), 3), np.float32); pairs = np.zeros((len(starts), 20), np.float32)
    begins, ends = times(truth.start_time), times(truth.end_time)
    for i, row in enumerate(truth.itertuples(index=False)):
        require(row.technique_id in TECHNIQUES and row.actor in HOSTS, "unknown technique/actor truth")
        require(row.target in HOSTS or row.target == "internal_subnet", "unknown target truth")
        a, b = begins.iloc[i], ends.iloc[i]
        require(b >= a, "reversed truth interval")
        mask = (starts <= a) & (a < starts+pd.Timedelta(seconds=5)) if a == b else (starts < b) & (starts+pd.Timedelta(seconds=5) > a)
        require(mask.any(), "unaligned event")
        tech[mask, TECHNIQUES.index(row.technique_id)] = 1
        if row.tactic == "Lateral Movement":
            require(row.technique_id == "T1021.004", "unexpected completed LM technique")
            pair = HOSTS[row.actor], HOSTS.get(row.target)
            require(pair in PAIRS, "invalid LM pair")
            pairs[mask, PAIRS.index(pair)] = 1
    lm = pairs.max(1)
    aligned = pd.read_csv(episode/"state_ground_truth.csv").fillna("")
    require(np.array_equal(aligned.state_id, np.arange(len(starts))), "aligned truth IDs differ")
    require(np.array_equal(times(aligned.window_start).to_numpy(), starts.to_numpy()), "aligned truth time differs")
    require(np.array_equal(aligned.has_lateral_movement, lm), "completed-LM alignment disagrees")
    for i, text in enumerate(aligned.technique_ids):
        require(set(filter(None, text.split(";"))) == {TECHNIQUES[k] for k in np.flatnonzero(tech[i])}, "technique alignment disagrees")
    return lm, tech, pairs, truth


def load_metadata(episode, starts, expected=None):
    frame = pd.read_csv(episode/"episode_metadata.csv", dtype=str)
    require(len(frame) == 1, "invalid metadata row count")
    m = frame.iloc[0].to_dict()
    require(m.get("runtime_version") == "v5_reviewed_1", "unreviewed capture runtime")
    require(m["node_count"] == "5" and m["window_seconds"] == "5", "graph/time metadata differs")
    require(m["planned_capture_duration_seconds"] == "150", "capture duration contract differs")
    a, b = times(pd.Series([m["capture_start"], m["capture_end"]]))
    require(150 <= (b-a).total_seconds() < 153, "capture duration drift")
    grid = pd.date_range(a.ceil("5s"), b.floor("5s"), freq="5s", inclusive="left")
    require(np.array_equal(starts.to_numpy(), grid.to_numpy()), "incomplete capture-bounded grid")
    ordered = roles(int(m["seed"]))
    require([m[k] for k in ["actor", "pivot", "target", "background_host_1", "background_host_2"]] == ordered, "role metadata mismatch")
    if expected is not None:
        for key in ["episode_id", "scenario", "seed", "background_profile", "defender_action", "topology_profile", "node_count"]:
            require(str(expected[key]) == m[key], f"plan mismatch: {key}")
    return m, a, b


def validate_action(episode, starts, meta, truth, lm):
    actions = pd.read_csv(episode/"defender_actions.csv", dtype=str)
    schema(actions, "episode_id start_time end_time action source target known_at_forecast_time details forecast_time effective_start_time".split(), "actions")
    if meta["defender_action"] == "none":
        require(actions.empty, "unexpected action row")
        return None
    require(len(actions) == 1, "expected one action")
    row = actions.iloc[0]
    require(row.action in ACTIONS and row.action == meta["defender_action"], "invalid action")
    require(row.episode_id == meta["episode_id"] and row.known_at_forecast_time == "true", "action provenance")
    require(row.source == meta["actor"] and row.target == meta["pivot"], "action pair metadata mismatch")
    decision, end, cutoff, effective = times(pd.Series([row.start_time, row.end_time, row.forecast_time, row.effective_start_time]))
    require(decision <= cutoff < effective <= end, "action not known before cutoff or applied inside context")
    require(cutoff == cutoff.floor("5s") and cutoff == pd.Timestamp(meta["forecast_time"]), "invalid action grid cutoff")
    require(0 < (effective-cutoff).total_seconds() < 1, "action apply drift")
    indexes = np.flatnonzero(starts == cutoff)
    require(len(indexes) == 1, "forecast grid absent")
    first = int(indexes[0]); require(first >= 3 and first+6 <= len(starts), "insufficient complete action context/future")
    remote = truth[truth.technique_id.eq("T1021.004")]
    require(len(remote) == 1, "expected one remote attempt/outcome")
    remote_start, remote_end = times(pd.Series([remote.iloc[0].start_time, remote.iloc[0].end_time]))
    require(end <= remote_start <= remote_end < cutoff+pd.Timedelta(seconds=30), "SSH not inside future or before application")
    expected = row.action == "permit_ssh"
    require(remote.iloc[0].tactic == ("Lateral Movement" if expected else "Lateral Movement Attempt"), "attempt/completion mismatch")
    require(not lm[:first].any() and bool(lm[first:first+6].any()) == expected, "action outcome/context mismatch")
    action_type = np.eye(2, dtype=np.float32)[ACTIONS.index(row.action)]
    action_pair = np.eye(20, dtype=np.float32)[PAIRS.index((HOSTS[row.source], HOSTS[row.target]))]
    return first, action_type, action_pair


def episode_samples(episode: Path, mode: str, expected=None):
    require(mode in ("passive", "action", "passive_action"), "invalid mode")
    states, edges, starts = load_observable_states(episode)
    lm, techniques, pairs, truth = load_targets(episode, starts)
    meta, capture_start, capture_end = load_metadata(episode, starts, expected)
    require(truth.episode_id.eq(meta["episode_id"]).all(), "truth episode IDs differ")
    if len(truth):
        require((times(truth.start_time) >= capture_start).all() and (times(truth.end_time) <= capture_end).all(), "truth outside capture")
    action = validate_action(episode, starts, meta, truth, lm)
    if mode == "passive":
        require(action is None, "ordinary passive training excludes intervention episodes")
        firsts = range(3, len(states)-5)
    else:
        require(action is not None, "aligned action sample requires chosen intervention")
        firsts = [action[0]]
    arrays = {k: [] for k in ["context_states", "future_states", "future_edge_presence", "future_lateral_movement",
                              "future_techniques", "future_lateral_edges", "lateral_movement_within_horizon"]}
    if mode == "action": arrays.update(action_type=[], action_pair=[])
    audit = []
    for first in firsts:
        future = slice(first, first+6)
        for key, val in [("context_states", states[first-3:first]), ("future_states", states[future]),
            ("future_edge_presence", edges[future]), ("future_lateral_movement", lm[future]),
            ("future_techniques", techniques[future]), ("future_lateral_edges", pairs[future]),
            ("lateral_movement_within_horizon", lm[future].max())]: arrays[key].append(val)
        if mode == "action":
            arrays["action_type"].append(action[1]); arrays["action_pair"].append(action[2])
        audit.append({"episode_id": meta["episode_id"], "context_first_state": first-3,
            "context_last_state": first-1, "future_first_state": first, "future_last_state": first+5,
            "prediction_available_time": str(starts.iloc[first]), "lateral_movement_already_observed": int(lm[:first].any()),
            "lateral_movement_within_horizon": int(lm[future].max()),
            "scenario": meta["scenario"], "background_profile": meta["background_profile"]})
    return {k: np.stack(v).astype(np.float32) for k, v in arrays.items()}, audit


def permutation_indices(order):
    require(sorted(order) == list(range(5)), "invalid host permutation")
    pairs = [(s, d) for s in range(5) for d in range(5) if s != d]
    edge_order = np.asarray([pairs.index((order[s], order[d])) for s, d in pairs])
    indices = np.r_[np.arange(15), np.concatenate([np.arange(15+18*i, 15+18*(i+1)) for i in order]),
                    np.concatenate([np.arange(105+12*i, 105+12*(i+1)) for i in edge_order])]
    return indices, edge_order
