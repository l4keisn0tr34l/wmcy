#!/usr/bin/env python3
"""Validate fresh role-balanced V4 action/passive paired corpus design."""
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import random
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ["ws1", "srv1", "srv2"]
REQUIRED = {"episode_id", "scenario", "seed", "duration_seconds", "cohort",
            "split", "paired_family", "defender_action"}
ACTION_MAP = {"scan_guess_action_permit": "permit_ssh", "scan_guess_action_block": "block_ssh"}
COHORT_SCENARIOS = {
    "action_conditioning": set(ACTION_MAP),
    "passive_branching": {"scan_guess_then_stop", "one_hop"},
    "direct_credential": {"matched_legitimate_ssh", "credential_one_hop"},
}


def roles(seed: int) -> tuple[str, ...]:
    hosts = HOSTS.copy(); random.Random(seed).shuffle(hosts); return tuple(hosts)


def old_seeds(config_dir: Path) -> set[int]:
    seeds: set[int] = set()
    for path in config_dir.glob("mvp_v[23]*episode_plan.csv"):
        frame = pd.read_csv(path)
        if "seed" in frame: seeds.update(pd.to_numeric(frame.seed, errors="raise").astype(int))
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "configs/mvp_v4_episode_plan.csv"))
    ap.add_argument("--splits", default=str(ROOT / "configs/mvp_v4_split_assignments.csv"))
    args = ap.parse_args(); plan = pd.read_csv(args.plan, dtype={"episode_id": str})
    splits = pd.read_csv(args.splits, dtype=str)
    missing = REQUIRED - set(plan.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    if len(plan) != 36: raise ValueError(f"V4 must contain 36 episodes, found {len(plan)}")
    if plan.episode_id.duplicated().any(): raise ValueError("duplicate episode IDs")
    if splits.episode_id.duplicated().any() or set(splits.columns) != {"episode_id", "split"}:
        raise ValueError("invalid split-assignment schema/IDs")
    expected_split = plan.set_index("episode_id").split.sort_index()
    actual_split = splits.set_index("episode_id").split.sort_index()
    if not expected_split.equals(actual_split): raise ValueError("plan and split assignments disagree")
    expected_ids = [f"lab_{index:03d}" for index in range(73, 109)]
    if plan.episode_id.tolist() != expected_ids: raise ValueError("V4 IDs must be ordered lab_073..lab_108")
    if not plan.duration_seconds.eq(120).all(): raise ValueError("all captures must have equal 120s duration")
    if not plan.split.isin(["train", "test"]).all(): raise ValueError("only train/test are allowed")
    if set(plan.cohort) != set(COHORT_SCENARIOS): raise ValueError("unexpected/missing cohort")
    for cohort, expected in COHORT_SCENARIOS.items():
        actual = set(plan.loc[plan.cohort.eq(cohort), "scenario"])
        if actual != expected: raise ValueError(f"{cohort} scenarios {actual} != {expected}")
    for row in plan.itertuples(index=False):
        expected_action = ACTION_MAP.get(row.scenario, "none")
        if row.defender_action != expected_action:
            raise ValueError(f"{row.episode_id}: action {row.defender_action} != {expected_action}")
        if not re.fullmatch(r"(action|passive_prefix|direct)_\d+", row.paired_family):
            raise ValueError(f"invalid paired family: {row.paired_family}")
    old = old_seeds(ROOT / "configs")
    if set(plan.seed) & old: raise ValueError("V4 seed overlaps V2/V3 plan")
    for family, rows in plan.groupby("paired_family", sort=False):
        if len(rows) != 2 or rows.seed.nunique() != 1 or rows.split.nunique() != 1 or rows.cohort.nunique() != 1:
            raise ValueError(f"{family}: pair does not share seed/split/cohort")
        expected = COHORT_SCENARIOS[rows.cohort.iloc[0]]
        if set(rows.scenario) != expected: raise ValueError(f"{family}: wrong scenario alternatives")
    action = plan[plan.cohort.eq("action_conditioning")]
    for split in ["train", "test"]:
        rows = action[action.split.eq(split)]
        if len(rows) != 12 or rows.seed.nunique() != 6:
            raise ValueError(f"action {split} must have six pairs")
        if {roles(int(seed)) for seed in rows.seed.unique()} != set(permutations(HOSTS)):
            raise ValueError(f"action {split} does not cover all six role permutations")
    if set(action[action.split.eq("train")].seed) & set(action[action.split.eq("test")].seed):
        raise ValueError("action train/test seeds overlap")
    passive = plan[~plan.cohort.eq("action_conditioning")]
    if len(passive) != 12 or not passive.split.eq("test").all():
        raise ValueError("all 12 passive/direct episodes must remain sealed test")
    if {roles(int(seed)) for seed in passive.seed.unique()} != set(permutations(HOSTS)):
        raise ValueError("combined passive/direct test seeds must cover all role permutations")
    counts = plan.groupby(["cohort", "split", "scenario"]).size()
    print("V4 plan PASS")
    print(counts.to_string())
    print("action train role permutations: 6/6")
    print("action sealed-test role permutations: 6/6")
    print("combined passive/direct sealed-test role permutations: 6/6")
    print("fresh episodes=36; train=12; sealed test=24")
    return 0


if __name__ == "__main__": sys.exit(main())
