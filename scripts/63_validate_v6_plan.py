#!/usr/bin/env python3
"""Fail-closed validation of the prospective V6 plan; no capture or inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v6_contract import (BACKGROUND_PROFILES, HOST_NAMES, SERVICE_PROFILES,
    TOPOLOGIES, capture_order, schedule, topology)

FIELDS = ["episode_id", "scenario", "seed", "duration_seconds", "cohort", "split",
          "evaluation_scope", "paired_family", "defender_action", "background_profile",
          "topology_profile", "active_node_count", "service_profile", "telemetry_profile",
          "prefix_contract", "forecast_focus"]
COUNTS = {
    "train": {"action_ablation": 10, "informative_passive": 10, "ambiguous_passive": 8,
              "two_hop_progression": 8, "benign_control": 8},
    "validation": {"action_ablation": 4, "informative_passive": 4, "ambiguous_passive": 4,
                   "two_hop_progression": 4, "benign_control": 4},
    "test": {"action_ablation": 8, "informative_passive": 8, "ambiguous_passive": 6,
             "two_hop_progression": 6, "benign_control": 4},
}
SCENARIOS = {
    "action_ablation": {"second_hop_action_permit", "second_hop_action_block"},
    "informative_passive": {"credential_pressure_progress", "matched_legitimate_admin"},
    "ambiguous_passive": {"shared_prefix_stop", "shared_prefix_progress"},
    "two_hop_progression": {"pivot_probe_then_stop", "pivot_probe_then_second_hop"},
    "benign_control": {"background_only", "legitimate_admin_ssh"},
}
ACTION = {"second_hop_action_permit": "permit_ssh", "second_hop_action_block": "block_ssh"}
PREFIX = {"action_ablation": "same_prefix_through_cutoff", "informative_passive": "observable_contrast",
          "ambiguous_passive": "same_prefix_through_cutoff", "two_hop_progression": "same_prefix_through_cutoff",
          "benign_control": "observable_contrast"}
FOCUS = {"action_ablation": "second_hop", "informative_passive": "first_hop",
         "ambiguous_passive": "first_hop", "two_hop_progression": "second_hop",
         "benign_control": "none_or_legitimate"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"plan schema/order differs: {reader.fieldnames}")
        return list(reader)


def previous_seeds() -> set[int]:
    seeds: set[int] = set()
    for path in (ROOT / "configs").glob("mvp*_episode_plan.csv"):
        if path.name == "mvp_v6_episode_plan.csv":
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("seed"):
                    seeds.add(int(row["seed"]))
    return seeds


def validate(plan_path: Path, split_path: Path) -> dict[str, int]:
    rows = read_rows(plan_path)
    if len(rows) != 192:
        raise ValueError(f"V6 requires192 episodes, found {len(rows)}")
    expected_ids = [f"lab_{number:03d}" for number in range(189, 381)]
    ids = [row["episode_id"] for row in rows]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError("V6 IDs must be unique and ordered lab_189..lab_380")
    with split_path.open(newline="") as handle:
        split_reader = csv.DictReader(handle)
        if split_reader.fieldnames != ["episode_id", "split"]:
            raise ValueError("split schema differs")
        split_rows = list(split_reader)
    if split_rows != [{"episode_id": row["episode_id"], "split": row["split"]} for row in rows]:
        raise ValueError("split file and plan disagree")
    if {row["split"] for row in rows} != set(COUNTS):
        raise ValueError("split coverage differs")
    if any(row["duration_seconds"] != "150" for row in rows):
        raise ValueError("all captures must be150 seconds")
    if any(row["telemetry_profile"] != "network_auth" for row in rows):
        raise ValueError("V6 requires packet plus authentication telemetry")
    if {row["topology_profile"] for row in rows} != set(TOPOLOGIES):
        raise ValueError("topology profile coverage differs")
    if any(row["background_profile"] not in BACKGROUND_PROFILES for row in rows):
        raise ValueError("unknown background")
    if any(row["service_profile"] not in SERVICE_PROFILES for row in rows):
        raise ValueError("unknown service profile")

    family_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        family_rows.setdefault(row["paired_family"], []).append(row)
    if len(family_rows) != 96:
        raise ValueError("V6 requires96 families")
    if len({int(group[0]["seed"]) for group in family_rows.values()}) != 96:
        raise ValueError("family seeds must be unique")
    if {int(group[0]["seed"]) for group in family_rows.values()} & previous_seeds():
        raise ValueError("V6 seed overlaps prior corpus")

    family_counts: Counter[tuple[str, str]] = Counter()
    role_triples: set[tuple[str, str, str, str]] = set()
    role_coverage: dict[str, list[set[str]]] = {split: [set(), set(), set()] for split in COUNTS}
    topology_counts: Counter[tuple[str, str, str]] = Counter()
    for family, group in family_rows.items():
        if len(group) != 2:
            raise ValueError(f"{family}: expected two alternatives")
        constant = ["seed", "duration_seconds", "cohort", "split", "evaluation_scope",
                    "background_profile", "topology_profile", "active_node_count", "service_profile",
                    "telemetry_profile", "prefix_contract", "forecast_focus"]
        for field in constant:
            if len({row[field] for row in group}) != 1:
                raise ValueError(f"{family}: paired {field} differs")
        row = group[0]; split = row["split"]; cohort = row["cohort"]; profile = row["topology_profile"]
        if not re.fullmatch(rf"v6_{split}_{cohort}_\d+", family):
            raise ValueError(f"{family}: malformed family ID")
        if cohort not in SCENARIOS or {item["scenario"] for item in group} != SCENARIOS[cohort]:
            raise ValueError(f"{family}: scenario alternatives differ")
        if row["prefix_contract"] != PREFIX[cohort] or row["forecast_focus"] != FOCUS[cohort]:
            raise ValueError(f"{family}: forecast/prefix contract differs")
        for item in group:
            if item["defender_action"] != ACTION.get(item["scenario"], "none"):
                raise ValueError(f"{item['episode_id']}: action differs")
        expected_scope = ("development" if split == "train" else "selection" if split == "validation"
                          else "sealed_topology_holdout" if profile == "dual_zone7_holdout"
                          else "sealed_in_domain")
        if row["evaluation_scope"] != expected_scope:
            raise ValueError(f"{family}: evaluation scope differs")
        if profile == "dual_zone7_holdout" and split != "test":
            raise ValueError("holdout topology leaked outside test")
        if split != "test" and profile not in {"flat5", "segmented7"}:
            raise ValueError("development topology differs")
        topo = topology(int(row["seed"]), profile)
        if int(row["active_node_count"]) != len(topo["active_hosts"]):
            raise ValueError(f"{family}: active node count differs")
        if topo["node_inventory"].shape != (7,) or topo["edge_topology"].shape != (42, 5):
            raise ValueError("topology tensor shape differs")
        if not np.isfinite(topo["edge_topology"]).all():
            raise ValueError("nonfinite topology")
        actor, pivot, target = topo["role_template"]
        identity = (profile, actor, pivot, target)
        if identity in role_triples:
            raise ValueError(f"reused topology/role triple: {identity}")
        role_triples.add(identity)
        for index, host in enumerate((actor, pivot, target)):
            role_coverage[split][index].add(host)
        sched = schedule(int(row["seed"]))
        if sched["decision_seconds"] + 5 + 30 > 145:
            raise ValueError(f"{family}: worst-case cutoff lacks future coverage")
        family_counts[(split, cohort)] += 1
        topology_counts[(split, cohort, profile)] += 1

    expected = {(split, cohort): count for split, cohorts in COUNTS.items() for cohort, count in cohorts.items()}
    if dict(family_counts) != expected:
        raise ValueError(f"cohort family counts differ: {family_counts}")
    for split, cohorts in COUNTS.items():
        for cohort, count in cohorts.items():
            profiles = [profile for profile in TOPOLOGIES if topology_counts[(split, cohort, profile)]]
            if split == "test":
                if topology_counts[(split, cohort, "dual_zone7_holdout")] != count // 2:
                    raise ValueError(f"{split}/{cohort}: holdout count differs")
                if not {"flat5", "segmented7"}.issubset(profiles):
                    raise ValueError(f"{split}/{cohort}: in-domain topology missing")
            elif set(profiles) != {"flat5", "segmented7"}:
                raise ValueError(f"{split}/{cohort}: development topology balance missing")
        groups = [group for family, group in family_rows.items() if group[0]["split"] == split]
        for field, allowed in (("background_profile", BACKGROUND_PROFILES), ("service_profile", SERVICE_PROFILES)):
            counts = Counter(group[0][field] for group in groups)
            if set(counts) != set(allowed) or max(counts.values()) - min(counts.values()) > 1:
                raise ValueError(f"{split}: {field} imbalance: {counts}")
        for profile in TOPOLOGIES:
            topology_groups = [group for group in groups if group[0]["topology_profile"] == profile]
            if not topology_groups:
                continue
            background_counts = Counter(group[0]["background_profile"] for group in topology_groups)
            service_counts = Counter(group[0]["service_profile"] for group in topology_groups)
            if len(topology_groups) >= 4 and (set(background_counts) != set(BACKGROUND_PROFILES)
                    or max(background_counts.values()) - min(background_counts.values()) > 1):
                raise ValueError(f"{split}/{profile}: topology-background confounding: {background_counts}")
            if set(service_counts) != set(SERVICE_PROFILES) or max(service_counts.values()) - min(service_counts.values()) > 1:
                raise ValueError(f"{split}/{profile}: topology-service confounding: {service_counts}")
            combinations = {(group[0]["background_profile"], group[0]["service_profile"])
                            for group in topology_groups}
            minimum_cross = min(8, max(1, len(topology_groups) - 2))
            if len(combinations) < minimum_cross:
                raise ValueError(f"{split}/{profile}: sparse/confounded background-service cross: {combinations}")
        for position, present in zip(("actor", "pivot", "target"), role_coverage[split]):
            if present != set(HOST_NAMES):
                raise ValueError(f"{split}: {position} identity coverage incomplete: {present}")

    order = sorted(ids, key=capture_order)
    if order == ids:
        raise ValueError("capture order follows plan order")
    return {"episodes": len(rows), "families": len(family_rows),
            "train_families": sum(group[0]["split"] == "train" for group in family_rows.values()),
            "validation_families": sum(group[0]["split"] == "validation" for group in family_rows.values()),
            "test_families": sum(group[0]["split"] == "test" for group in family_rows.values()),
            "holdout_test_families": sum(group[0]["evaluation_scope"] == "sealed_topology_holdout" for group in family_rows.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mvp_v6_episode_plan.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "configs/mvp_v6_split_assignments.csv")
    args = parser.parse_args()
    summary = validate(args.plan, args.splits)
    print("V6 prospective plan PASS")
    print(" ".join(f"{key}={value}" for key, value in summary.items()))
    print("holdout topology is test-only; V5 remains untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
