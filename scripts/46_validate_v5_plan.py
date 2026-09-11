#!/usr/bin/env python3
"""Validate the time-boxed five-host V5 paired episode/split design."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_contract import HOSTS as INVENTORY, roles, capture_order
HOSTS = list(INVENTORY)
BACKGROUND = {"quiet", "web", "admin", "mixed"}
REQUIRED = {"episode_id", "scenario", "seed", "duration_seconds", "cohort", "split",
            "paired_family", "defender_action", "background_profile", "topology_profile", "node_count"}
COHORT_SCENARIOS = {
    "scan_action": {"scan_guess_action_permit", "scan_guess_action_block"},
    "credential_action": {"credential_action_permit", "credential_action_block"},
    "passive_prefix": {"scan_guess_then_stop", "one_hop"},
    "intent_probe": {"matched_legitimate_ssh", "credential_one_hop"},
    "benign_control": {"background_only", "legitimate_ssh"},
}
ACTION_MAP = {
    "scan_guess_action_permit": "permit_ssh", "credential_action_permit": "permit_ssh",
    "scan_guess_action_block": "block_ssh", "credential_action_block": "block_ssh",
}
EXPECTED_EPISODES = {
    ("train", "scan_action"): 12, ("train", "credential_action"): 12,
    ("train", "passive_prefix"): 8, ("train", "benign_control"): 8,
    ("validation", "scan_action"): 4, ("validation", "credential_action"): 4,
    ("validation", "passive_prefix"): 4, ("validation", "benign_control"): 4,
    ("test", "scan_action"): 4, ("test", "credential_action"): 4,
    ("test", "passive_prefix"): 4, ("test", "intent_probe"): 8,
    ("test", "benign_control"): 4,
}


def role_order(seed: int) -> tuple[str, ...]:
    return tuple(roles(seed))


def prior_seeds(config_dir: Path) -> set[int]:
    values: set[int] = set()
    for path in config_dir.glob("mvp*_episode_plan.csv"):
        if path.name == "mvp_v5_episode_plan.csv": continue
        frame = pd.read_csv(path)
        if "seed" in frame: values.update(pd.to_numeric(frame.seed, errors="raise").astype(int))
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "configs/mvp_v5_episode_plan.csv"))
    ap.add_argument("--splits", default=str(ROOT / "configs/mvp_v5_split_assignments.csv"))
    args = ap.parse_args(); plan = pd.read_csv(args.plan, dtype={"episode_id": str})
    splits = pd.read_csv(args.splits, dtype=str)
    missing = REQUIRED - set(plan.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    if len(plan) != 80: raise ValueError(f"V5 requires 80 episodes, found {len(plan)}")
    expected_ids = [f"lab_{number:03d}" for number in range(109, 189)]
    if plan.episode_id.tolist() != expected_ids or plan.episode_id.duplicated().any():
        raise ValueError("V5 IDs must be unique and ordered lab_109..lab_188")
    if set(splits.columns) != {"episode_id", "split"} or splits.episode_id.duplicated().any():
        raise ValueError("invalid split assignment schema")
    expected_split = plan.set_index("episode_id").split.sort_index()
    actual_split = splits.set_index("episode_id").split.sort_index()
    if not expected_split.equals(actual_split): raise ValueError("plan and split assignments disagree")
    if not plan.duration_seconds.eq(150).all(): raise ValueError("all V5 captures must be 150 seconds")
    if not plan.node_count.eq(5).all() or not plan.topology_profile.eq("flat_five_host").all():
        raise ValueError("V5 must use the declared fixed five-host graph")
    if not set(plan.background_profile).issubset(BACKGROUND): raise ValueError("unknown background profile")
    if set(plan.cohort) != set(COHORT_SCENARIOS): raise ValueError("missing/unexpected cohort")
    if set(plan.split) != {"train", "validation", "test"}: raise ValueError("missing/unexpected split")
    actual_counts = plan.groupby(["split", "cohort"]).size().to_dict()
    if actual_counts != EXPECTED_EPISODES: raise ValueError(f"cohort counts differ: {actual_counts}")
    if not plan.loc[plan.cohort.eq("intent_probe"), "split"].eq("test").all():
        raise ValueError("the unidentifiable intent probe must remain test-only")
    if set(plan.seed) & prior_seeds(ROOT / "configs"):
        raise ValueError("V5 seed overlaps an earlier corpus")
    triples: dict[str, set[tuple[str, ...]]] = {split: set() for split in ["train", "validation", "test"]}
    seen_triples: set[tuple[str, ...]] = set()
    for family, rows in plan.groupby("paired_family", sort=False):
        if len(rows) != 2 or rows.seed.nunique() != 1 or rows.split.nunique() != 1 \
                or rows.cohort.nunique() != 1 or rows.background_profile.nunique() != 1:
            raise ValueError(f"{family}: paired attributes disagree")
        cohort = rows.cohort.iloc[0]; split = rows.split.iloc[0]
        if set(rows.scenario) != COHORT_SCENARIOS[cohort]: raise ValueError(f"{family}: wrong alternatives")
        if not re.fullmatch(r"v5_(train|validation|test)_[a-z_]+_\d+", family):
            raise ValueError(f"invalid family ID: {family}")
        triple = role_order(int(rows.seed.iloc[0]))[:3]
        if triple in seen_triples: raise ValueError(f"reused actor/pivot/target ordering: {triple}")
        seen_triples.add(triple); triples[split].add(triple)
        for row in rows.itertuples(index=False):
            expected_action = ACTION_MAP.get(row.scenario, "none")
            if row.defender_action != expected_action:
                raise ValueError(f"{row.episode_id}: action mismatch")
    if len(seen_triples) != 40: raise ValueError("expected 40 unique ordered role triples")
    for split, split_triples in triples.items():
        for position, role in enumerate(["actor", "pivot", "target"]):
            present = {triple[position] for triple in split_triples}
            if present != set(HOSTS): raise ValueError(f"{split} {role} lacks host coverage: {present}")
        family_profiles = plan[plan.split.eq(split)].drop_duplicates("paired_family").background_profile
        profile_counts = Counter(family_profiles)
        if set(profile_counts) != BACKGROUND or max(profile_counts.values()) - min(profile_counts.values()) > 0:
            raise ValueError(f"{split}: background families not exactly balanced: {profile_counts}")
    for scenario, expected_action in ACTION_MAP.items():
        if not plan.loc[plan.scenario.eq(scenario), "defender_action"].eq(expected_action).all():
            raise ValueError(f"{scenario}: defender action mismatch")
    for cohort in ["scan_action", "credential_action", "passive_prefix", "benign_control"]:
        val_profiles = set(plan.loc[plan.cohort.eq(cohort) & plan.split.eq("validation"), "background_profile"])
        test_profiles = set(plan.loc[plan.cohort.eq(cohort) & plan.split.eq("test"), "background_profile"])
        if val_profiles & test_profiles or val_profiles | test_profiles != BACKGROUND:
            raise ValueError(f"{cohort}: validation/test profile rotation lost")
    order = sorted(plan.episode_id, key=capture_order)
    if order == plan.episode_id.tolist(): raise ValueError("capture order must not follow split/permit-before-block CSV order")
    total_seconds = int(plan.duration_seconds.sum())
    print("V5 plan PASS")
    print(plan.groupby(["split", "cohort", "scenario"]).size().to_string())
    print(f"episodes={len(plan)} families={plan.paired_family.nunique()} capture={total_seconds/3600:.2f}h")
    print("five-host actor/pivot/target coverage: complete in train, validation, and test")
    print("background profiles: exactly balanced by paired family within every split")
    print("intent probe: test-only; action train/validation/test remain disjoint")
    return 0


if __name__ == "__main__": sys.exit(main())
