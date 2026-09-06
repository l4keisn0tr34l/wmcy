#!/usr/bin/env python3
"""Build audited episode and split manifests for the controlled MVP corpus.

These manifests contain scenario/ground-truth metadata and are never model input.
Sequences must be constructed separately within the assigned whole-episode split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def joined(values: pd.Series) -> str:
    return ";".join(sorted(set(values.dropna().astype(str))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "configs/mvp_episode_plan.csv"))
    ap.add_argument("--splits", default=str(ROOT / "configs/mvp_split_assignments.csv"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp"))
    ap.add_argument("--window-seconds", type=int, default=5)
    args = ap.parse_args()

    plan = pd.read_csv(args.plan, dtype={"episode_id": str, "scenario": str, "seed": str})
    splits = pd.read_csv(args.splits, dtype=str)
    episodes_dir = Path(args.episodes_dir)
    out_dir = Path(args.out_dir)

    if plan["episode_id"].duplicated().any():
        raise ValueError("episode plan contains duplicate IDs")
    if splits["episode_id"].duplicated().any():
        raise ValueError("split assignments contain duplicate IDs")
    if set(plan.episode_id) != set(splits.episode_id):
        missing = set(plan.episode_id) - set(splits.episode_id)
        extra = set(splits.episode_id) - set(plan.episode_id)
        raise ValueError(f"split/plan episode mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    allowed_splits = {"train", "validation", "test"}
    if not set(splits.split).issubset(allowed_splits):
        raise ValueError(f"unknown split values: {sorted(set(splits.split) - allowed_splits)}")

    split_by_episode = splits.set_index("episode_id")["split"].to_dict()
    rows: list[dict[str, object]] = []
    for planned in plan.itertuples(index=False):
        episode = episodes_dir / planned.episode_id
        for marker in ["INVALID_EPISODE.txt", "EXCLUDE_FROM_MVP.txt"]:
            if (episode / marker).exists():
                raise ValueError(f"planned episode {planned.episode_id} is excluded by {marker}")

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/07_validate_episode.py"),
                str(episode),
                "--window-seconds", str(args.window_seconds),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        metadata = pd.read_csv(episode / "episode_metadata.csv", dtype={"seed": str})
        if len(metadata) != 1:
            raise ValueError(f"{planned.episode_id}: metadata must have exactly one row")
        m = metadata.iloc[0]
        expected = (str(planned.episode_id), str(planned.scenario), str(planned.seed))
        actual = (str(m.episode_id), str(m.scenario), str(m.seed))
        if actual != expected:
            raise ValueError(f"{planned.episode_id}: metadata {actual} != plan {expected}")

        observations = pd.read_csv(episode / "observations.csv.gz", low_memory=False)
        global_states = pd.read_csv(episode / "states/global_states.csv")
        nodes = pd.read_csv(episode / "states/node_states.csv.gz", low_memory=False)
        edges = pd.read_csv(episode / "states/edge_states.csv.gz", low_memory=False)
        truth = pd.read_csv(episode / "ground_truth.csv")

        state_times = pd.to_datetime(global_states.window_start, utc=True)
        observation_times = pd.to_datetime(observations.timestamp, utc=True)
        state_start = state_times.iloc[0]
        state_end = state_times.iloc[-1] + pd.Timedelta(seconds=args.window_seconds)
        boundary_observations = int(
            ((observation_times < state_start) | (observation_times >= state_end)).sum()
        )
        lateral = truth[truth.tactic.astype(str).eq("Lateral Movement")]
        capture_start = pd.to_datetime(m.capture_start, utc=True)
        capture_end = pd.to_datetime(m.capture_end, utc=True)

        rows.append({
            "episode_id": planned.episode_id,
            "split": split_by_episode[planned.episode_id],
            "data_source": "controlled_docker_lab",
            "scenario": planned.scenario,
            "seed": planned.seed,
            "actor": m.actor,
            "pivot": m.pivot,
            "target": m.target,
            "capture_start": capture_start,
            "capture_end": capture_end,
            "capture_duration_seconds": (capture_end - capture_start).total_seconds(),
            "window_seconds": args.window_seconds,
            "state_start": state_start,
            "state_end": state_end,
            "num_states": len(global_states),
            "num_zero_traffic_states": int(global_states.flow_count.eq(0).sum()),
            "num_observations": len(observations),
            "num_boundary_observations_excluded_from_states": boundary_observations,
            "num_node_state_rows": len(nodes),
            "num_edge_state_rows": len(edges),
            "num_ground_truth_events": len(truth),
            "technique_ids": joined(truth.technique_id),
            "tactics": joined(truth.tactic),
            "has_lateral_movement": int(len(lateral) > 0),
            "num_lateral_movement_events": len(lateral),
            "validation_status": "pass",
        })

    manifest = pd.DataFrame(rows)
    train_scenarios = set(manifest.loc[manifest.split.eq("train"), "scenario"])
    all_scenarios = set(manifest.scenario)
    if train_scenarios != all_scenarios:
        raise ValueError(f"training split lacks scenarios: {sorted(all_scenarios - train_scenarios)}")

    # The known topology has six role permutations. Requiring all in train helps
    # prevent a fixed host identity from uniquely revealing the scenario role.
    train_roles = manifest.loc[manifest.split.eq("train"), ["actor", "pivot", "target"]].drop_duplicates()
    if len(train_roles) != 6:
        raise ValueError(f"training split contains {len(train_roles)}/6 host-role permutations")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "episode_manifest.csv", index=False)
    manifest[["episode_id", "split", "scenario"]].to_csv(out_dir / "split_manifest.csv", index=False)

    print(f"wrote {len(manifest)} validated episodes -> {out_dir/'episode_manifest.csv'}")
    print("split counts:")
    print(manifest.groupby(["split", "scenario"]).size().unstack(fill_value=0).to_string())
    print("totals:")
    print(
        manifest.groupby("split")[[
            "num_states", "num_observations", "num_ground_truth_events",
            "num_lateral_movement_events",
        ]].sum().to_string()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
