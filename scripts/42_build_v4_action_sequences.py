#!/usr/bin/env python3
"""Build intervention-aligned V4 graph sequences, train split by default.

Passive telemetry and chosen actions are separate inputs. Attack/LM outcomes and
future graph values remain separate targets. V4 test requires an explicit unlock.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACTION_TYPES = ("permit_ssh", "block_ssh")
FORBIDDEN = ("label", "attack", "technique", "tactic", "lateral", "actor", "target", "scenario", "seed")


def load_sequence_module() -> Any:
    path = ROOT / "scripts/10_build_mvp_sequences.py"
    spec = importlib.util.spec_from_file_location("v4_base_sequences", path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def feature_contract(module: Any, episode: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    states = episode / "states"
    global_columns = pd.read_csv(states / "global_states.csv", nrows=0).columns.tolist()
    node_columns = pd.read_csv(states / "node_states.csv.gz", nrows=0).columns.tolist()
    edge_columns = pd.read_csv(states / "edge_states.csv.gz", nrows=0).columns.tolist()
    global_features = [name for name in global_columns if name not in {"state_id", "window_start"}]
    node_features = [name for name in node_columns if name not in {"state_id", "window_start", "host"}]
    edge_features = [name for name in edge_columns
                     if name not in {"state_id", "window_start", "source_ip", "destination_ip"}]
    names = list(global_features)
    for slot in range(len(module.KNOWN_HOSTS)):
        names.extend(f"node_{slot}__{name}" for name in node_features + ["activity_mask"])
    for slot in range(len(module.DIRECTED_PAIRS)):
        names.extend(f"edge_{slot}__{name}" for name in edge_features + ["presence_mask"])
    contaminated = [name for name in names if any(term in name.lower() for term in FORBIDDEN)]
    if contaminated: raise ValueError(f"forbidden observable feature names: {contaminated}")
    if len(names) != 141: raise ValueError(f"unexpected state width: {len(names)}")
    return global_features, node_features, edge_features, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "configs/mvp_v4_episode_plan.csv"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp_v4/action_sequences"))
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--unlock-test", action="store_true",
                    help="required to read V4 action-test episodes after model/protocol freeze")
    ap.add_argument("--context", type=int, default=3); ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.split == "test" and not args.unlock_test:
        raise PermissionError("V4 action test is sealed; pass --unlock-test only after protocol freeze")
    if (args.context, args.horizon) != (3, 6): raise ValueError("V4 protocol requires context=3, horizon=6")
    out = Path(args.out_dir); output_npz = out / f"{args.split}.npz"
    output_audit = out / f"{args.split}_sample_manifest.csv"
    output_metadata = out / f"{args.split}_feature_metadata.json"
    if not args.force and (output_npz.exists() or output_audit.exists() or output_metadata.exists()):
        raise FileExistsError(f"refusing to replace {args.split} action sequences without --force")
    plan = pd.read_csv(args.plan, dtype={"episode_id": str, "seed": str})
    selected = plan[plan.cohort.eq("action_conditioning") & plan.split.eq(args.split)].copy()
    if len(selected) != 12: raise ValueError(f"expected 12 {args.split} action episodes, found {len(selected)}")
    module = load_sequence_module(); episodes = Path(args.episodes_dir)
    # Determine temporal eligibility without reading outcomes or running a model.
    # If one member lacks the fixed six complete future windows, remove its
    # whole predefined permit/block family so paired analysis remains balanced.
    ineligible_families: dict[str, list[str]] = {}
    for planned in selected.itertuples(index=False):
        action_rows = pd.read_csv(episodes / planned.episode_id / "defender_actions.csv")
        grid = pd.DatetimeIndex(pd.to_datetime(
            pd.read_csv(episodes / planned.episode_id / "states/global_states.csv").window_start, utc=True
        ))
        action_grid = pd.to_datetime(action_rows.start_time.iloc[0], utc=True).floor("5s")
        matches = np.flatnonzero(grid == action_grid)
        eligible = len(matches) == 1 and int(matches[0]) >= args.context and int(matches[0]) + args.horizon <= len(grid)
        if not eligible:
            ineligible_families.setdefault(planned.paired_family, []).append(planned.episode_id)
    if ineligible_families:
        if args.split != "test" or ineligible_families != {"action_7002": ["lab_090"]}:
            raise ValueError(f"unexpected action-alignment exclusions: {ineligible_families}")
        selected = selected[~selected.paired_family.isin(ineligible_families)].copy()
    expected_samples = len(selected)
    if expected_samples not in ({12} if args.split == "train" else {10, 12}):
        raise ValueError(f"unexpected eligible {args.split} sample count: {expected_samples}")
    first_episode = episodes / selected.episode_id.iloc[0]
    global_features, node_features, edge_features, state_names = feature_contract(module, first_episode)
    stores: dict[str, list[np.ndarray]] = {
        "context_states": [], "action_type": [], "action_pair": [], "future_states": [],
        "future_edge_presence": [], "future_lateral_movement": [], "future_techniques": [],
        "future_lateral_edges": [], "lateral_movement_within_horizon": [],
    }
    audit_rows = []
    action_index = {name: index for index, name in enumerate(ACTION_TYPES)}
    pair_index = {pair: index for index, pair in enumerate(module.DIRECTED_PAIRS)}
    for sample_id, planned in enumerate(selected.itertuples(index=False)):
        episode = episodes / planned.episode_id
        # Only the selected split reaches this file-reading point.
        metadata = pd.read_csv(episode / "episode_metadata.csv", dtype={"seed": str})
        actions = pd.read_csv(episode / "defender_actions.csv")
        truth = pd.read_csv(episode / "ground_truth.csv")
        if len(metadata) != 1 or len(actions) != 1: raise ValueError(f"{planned.episode_id}: metadata/action count")
        meta = metadata.iloc[0]; action = actions.iloc[0]
        for key in ["episode_id", "scenario", "seed", "defender_action"]:
            if str(meta[key]) != str(getattr(planned, key)):
                raise ValueError(f"{planned.episode_id}: metadata {key} mismatch")
        if action["action"] != planned.defender_action or str(action["known_at_forecast_time"]).lower() != "true":
            raise ValueError(f"{planned.episode_id}: action identity/availability mismatch")
        action_start = pd.to_datetime(action["start_time"], utc=True)
        action_end = pd.to_datetime(action["end_time"], utc=True)
        grid_time = action_start.floor("5s")
        offset_seconds = (action_start - grid_time).total_seconds()
        if not 0.15 <= offset_seconds <= 0.30:
            raise ValueError(f"{planned.episode_id}: unexpected action/grid offset {offset_seconds}")
        states, edge_presence, lm, techniques, lm_edges, global_states = module.build_episode_states(
            episode, global_features, node_features, edge_features
        )
        times = pd.DatetimeIndex(pd.to_datetime(global_states.window_start, utc=True))
        matches = np.flatnonzero(times == grid_time)
        if len(matches) != 1: raise ValueError(f"{planned.episode_id}: action grid state not unique/present")
        future_first = int(matches[0]); context_first = future_first - args.context
        future_last = future_first + args.horizon - 1
        if context_first < 0 or future_last >= len(states):
            raise ValueError(f"{planned.episode_id}: insufficient action-aligned context/future")
        context_slice = slice(context_first, future_first); future_slice = slice(future_first, future_last + 1)
        context_end = times[future_first]
        if context_end > action_start or not (times[future_first] <= action_start < times[future_first] + pd.Timedelta(seconds=5)):
            raise ValueError(f"{planned.episode_id}: action not in first future grid state")
        truth = truth.copy(); truth["start_time"] = pd.to_datetime(truth.start_time, utc=True)
        truth["end_time"] = pd.to_datetime(truth.end_time, utc=True)
        remote = truth[truth.technique_id.astype(str).eq("T1021.004")]
        if len(remote) != 1 or action_end > remote.start_time.iloc[0]:
            raise ValueError(f"{planned.episode_id}: action does not precede one SSH result")
        expected_lm = int(planned.defender_action == "permit_ssh")
        realized_lm = int(lm[future_slice].max() > 0)
        if realized_lm != expected_lm or lm[:future_first].max(initial=0) > 0:
            raise ValueError(f"{planned.episode_id}: completed-LM context/outcome mismatch")
        tactic = remote.tactic.iloc[0]
        expected_tactic = "Lateral Movement" if expected_lm else "Lateral Movement Attempt"
        if tactic != expected_tactic: raise ValueError(f"{planned.episode_id}: remote tactic mismatch")
        source_ip = module.HOST_NAMES.get(str(action["source"])); target_ip = module.HOST_NAMES.get(str(action["target"]))
        if (source_ip, target_ip) not in pair_index: raise ValueError(f"{planned.episode_id}: unknown action pair")
        type_vector = np.zeros(len(ACTION_TYPES), dtype=np.float32); type_vector[action_index[planned.defender_action]] = 1
        pair_vector = np.zeros(len(module.DIRECTED_PAIRS), dtype=np.float32); pair_vector[pair_index[(source_ip, target_ip)]] = 1
        stores["context_states"].append(states[context_slice]); stores["action_type"].append(type_vector)
        stores["action_pair"].append(pair_vector); stores["future_states"].append(states[future_slice])
        stores["future_edge_presence"].append(edge_presence[future_slice])
        stores["future_lateral_movement"].append(lm[future_slice])
        stores["future_techniques"].append(techniques[future_slice])
        stores["future_lateral_edges"].append(lm_edges[future_slice])
        stores["lateral_movement_within_horizon"].append(np.asarray(realized_lm, dtype=np.float32))
        audit_rows.append({
            "sample_id": sample_id, "split": args.split, "episode_id": planned.episode_id,
            "paired_family": planned.paired_family, "scenario": planned.scenario, "seed": planned.seed,
            "action": planned.defender_action, "action_source": action["source"], "action_target": action["target"],
            "context_first_state": context_first, "context_last_state": future_first - 1,
            "future_first_state": future_first, "future_last_state": future_last,
            "context_end_grid_time": context_end, "action_start_time": action_start,
            "action_grid_offset_seconds": offset_seconds, "remote_start_time": remote.start_time.iloc[0],
            "lateral_movement_within_horizon": realized_lm,
        })
    arrays = {name: np.stack(values).astype(np.float32) for name, values in stores.items()}
    expected_shapes = {
        "context_states": (expected_samples, 3, 141), "action_type": (expected_samples, 2), "action_pair": (expected_samples, 6),
        "future_states": (expected_samples, 6, 141), "future_edge_presence": (expected_samples, 6, 6),
        "future_lateral_movement": (expected_samples, 6), "future_techniques": (expected_samples, 6, 3),
        "future_lateral_edges": (expected_samples, 6, 6), "lateral_movement_within_horizon": (expected_samples,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"invalid {name}: {arrays[name].shape}")
    per_action = expected_samples // 2
    if not np.allclose(arrays["action_type"].sum(axis=0), [per_action, per_action]): raise ValueError("action types not balanced")
    if int(arrays["lateral_movement_within_horizon"].sum()) != per_action: raise ValueError("LM outcomes not balanced")
    metadata_out = {
        "source": "controlled Docker V4 action-conditioning episodes",
        "split": args.split, "samples": len(selected), "context_states": args.context,
        "future_horizon_states": args.horizon, "window_seconds": 5,
        "timing_excluded_families": ineligible_families,
        "timing_exclusion_policy": "exclude both predefined paired-family members if either lacks six complete post-action windows; uses timing only before prediction",
        "state_feature_count": len(state_names), "state_feature_names": state_names,
        "global_feature_names": global_features, "node_feature_names": node_features + ["activity_mask"],
        "edge_feature_names": edge_features + ["presence_mask"],
        "node_slots": [{"slot": index, "ip": host} for index, host in enumerate(module.KNOWN_HOSTS)],
        "directed_edge_slots": [{"slot": index, "source_ip": source, "destination_ip": destination}
                                for index, (source, destination) in enumerate(module.DIRECTED_PAIRS)],
        "technique_targets": list(module.TECHNIQUES), "action_types": list(ACTION_TYPES),
        "action_pair_policy": "one-hot directed source/target pair known from chosen intervention; must be host-permuted with graph",
        "input_policy": "context_states is passive telemetry ending before intervention; action_type/action_pair are chosen and known; outcomes/truth are separate targets",
        "test_access": "explicit --unlock-test required" if args.split == "test" else "test episode files were not read",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"v4-action-{args.split}-", dir=out.parent))
    try:
        np.savez_compressed(work / output_npz.name, **arrays)
        pd.DataFrame(audit_rows).to_csv(work / output_audit.name, index=False)
        (work / output_metadata.name).write_text(json.dumps(metadata_out, indent=2) + "\n")
        out.mkdir(parents=True, exist_ok=True)
        for path in work.iterdir(): os.replace(path, out / path.name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"V4 action {args.split}: " + " ".join(f"{name}={value.shape}" for name, value in arrays.items()))
    print(f"test episode access: {metadata_out['test_access']}")
    return 0


if __name__ == "__main__": sys.exit(main())
