#!/usr/bin/env python3
"""Validate matched-prefix V3 plan and fresh-holdout split invariants."""
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import random
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HOSTS = ["ws1", "srv1", "srv2"]


def roles(seed: int) -> tuple[str, ...]:
    hosts = HOSTS.copy(); random.Random(seed).shuffle(hosts); return tuple(hosts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-plan", default=str(ROOT / "configs/mvp_v2_episode_plan.csv"))
    ap.add_argument("--new-plan", default=str(ROOT / "configs/mvp_v3_new_episode_plan.csv"))
    ap.add_argument("--full-plan", default=str(ROOT / "configs/mvp_v3_episode_plan.csv"))
    ap.add_argument("--splits", default=str(ROOT / "configs/mvp_v3_split_assignments.csv"))
    args = ap.parse_args()
    old = pd.read_csv(args.old_plan, dtype={"episode_id": str})
    new = pd.read_csv(args.new_plan, dtype={"episode_id": str})
    full = pd.read_csv(args.full_plan, dtype={"episode_id": str})
    splits = pd.read_csv(args.splits, dtype=str)
    for name, frame in [("old", old), ("new", new), ("full", full), ("splits", splits)]:
        if frame.episode_id.duplicated().any(): raise ValueError(f"duplicate IDs in {name}")
    if set(full.episode_id) != set(old.episode_id) | set(new.episode_id):
        raise ValueError("full V3 plan is not exactly old V2 plus new paired episodes")
    if set(full.episode_id) != set(splits.episode_id): raise ValueError("plan/split IDs differ")
    split = splits.set_index("episode_id").split
    if not set(old.episode_id).issubset(set(split[split.eq("train")].index)):
        raise ValueError("previously inspected V2 episodes must be V3 development train only")
    if set(old.episode_id) & set(split[~split.eq("train")].index):
        raise ValueError("old episodes leaked into fresh validation/test")
    expected_scenarios = {"scan_guess_then_stop", "one_hop"}
    if set(new.scenario) != expected_scenarios: raise ValueError("unexpected V3 new scenario")
    if not new.duration_seconds.eq(120).all(): raise ValueError("new episodes must have equal 120s duration")
    paired = new.groupby("seed")
    for seed, rows in paired:
        if len(rows) != 2 or set(rows.scenario) != expected_scenarios:
            raise ValueError(f"seed {seed} is not one stopped/progressing pair")
        assigned = set(split.loc[rows.episode_id])
        if len(assigned) != 1: raise ValueError(f"seed pair {seed} crosses splits")
    counts = new.assign(split=new.episode_id.map(split)).groupby(["split", "scenario"]).size().unstack(fill_value=0)
    expected_counts = {"train": 6, "validation": 3, "test": 3}
    for split_name, count in expected_counts.items():
        if any(counts.loc[split_name, scenario] != count for scenario in expected_scenarios):
            raise ValueError(f"wrong matched count in {split_name}")
    train_seeds = new.loc[new.episode_id.map(split).eq("train"), "seed"].drop_duplicates()
    train_roles = {roles(int(seed)) for seed in train_seeds}
    if train_roles != set(permutations(HOSTS)):
        raise ValueError(f"new training pairs cover {len(train_roles)}/6 role permutations")
    validation_seeds = set(new.loc[new.episode_id.map(split).eq("validation"), "seed"])
    test_seeds = set(new.loc[new.episode_id.map(split).eq("test"), "seed"])
    if validation_seeds & test_seeds: raise ValueError("validation/test seeds overlap")
    print("V3 plan PASS")
    print(counts.to_string())
    print(f"old development train={len(old)}, new train={2*len(train_seeds)}, "
          f"fresh validation={2*len(validation_seeds)}, fresh test={2*len(test_seeds)}")
    print("new train role permutations: 6/6")
    return 0


if __name__ == "__main__": sys.exit(main())
