#!/usr/bin/env python3
"""Build the deterministic prospective V6 episode and split plans."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v6_contract import BACKGROUND_PROFILES, SERVICE_PROFILES, topology

COUNTS = {
    "train": {"action_ablation": 10, "informative_passive": 10, "ambiguous_passive": 8,
              "two_hop_progression": 8, "benign_control": 8},
    "validation": {"action_ablation": 4, "informative_passive": 4, "ambiguous_passive": 4,
                   "two_hop_progression": 4, "benign_control": 4},
    "test": {"action_ablation": 8, "informative_passive": 8, "ambiguous_passive": 6,
             "two_hop_progression": 6, "benign_control": 4},
}
SCENARIOS = {
    "action_ablation": ("second_hop_action_permit", "second_hop_action_block"),
    "informative_passive": ("credential_pressure_progress", "matched_legitimate_admin"),
    "ambiguous_passive": ("shared_prefix_stop", "shared_prefix_progress"),
    "two_hop_progression": ("pivot_probe_then_stop", "pivot_probe_then_second_hop"),
    "benign_control": ("background_only", "legitimate_admin_ssh"),
}
ACTIONS = {"second_hop_action_permit": "permit_ssh", "second_hop_action_block": "block_ssh"}
PREFIX = {"action_ablation": "same_prefix_through_cutoff", "informative_passive": "observable_contrast",
          "ambiguous_passive": "same_prefix_through_cutoff", "two_hop_progression": "same_prefix_through_cutoff",
          "benign_control": "observable_contrast"}
FOCUS = {"action_ablation": "second_hop", "informative_passive": "first_hop",
         "ambiguous_passive": "first_hop", "two_hop_progression": "second_hop",
         "benign_control": "none_or_legitimate"}
FIELDS = ["episode_id", "scenario", "seed", "duration_seconds", "cohort", "split",
          "evaluation_scope", "paired_family", "defender_action", "background_profile",
          "topology_profile", "active_node_count", "service_profile", "telemetry_profile",
          "prefix_contract", "forecast_focus"]


def prior_seeds() -> set[int]:
    found: set[int] = set()
    for path in (ROOT / "configs").glob("mvp*_episode_plan.csv"):
        if path.name == "mvp_v6_episode_plan.csv":
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("seed"):
                    found.add(int(row["seed"]))
    return found


def topology_sequence(split: str, count: int) -> list[str]:
    if split != "test":
        return ["flat5" if index % 2 == 0 else "segmented7" for index in range(count)]
    holdout = count // 2
    return ["dual_zone7_holdout"] * holdout + ["flat5" if index % 2 == 0 else "segmented7"
                                                    for index in range(count - holdout)]


def build_rows() -> list[dict[str, str | int]]:
    rng = random.Random(62001)
    blocked = prior_seeds()
    candidates = [seed for seed in range(20_000, 50_000) if seed not in blocked]
    rng.shuffle(candidates)
    used_seeds: set[int] = set(); used_role_triples: set[tuple[str, str, str, str]] = set()
    role_coverage = {split: [set(), set(), set()] for split in COUNTS}
    global_background: Counter[tuple[str, str]] = Counter()
    topology_background: Counter[tuple[str, str, str]] = Counter()
    global_service: Counter[tuple[str, str]] = Counter()
    topology_service: Counter[tuple[str, str, str]] = Counter()
    profile_cross: Counter[tuple[str, str, str, str]] = Counter()
    rows: list[dict[str, str | int]] = []
    episode_number = 189
    family_index = 0
    for split, cohorts in COUNTS.items():
        for cohort, count in cohorts.items():
            for local_index, topology_profile in enumerate(topology_sequence(split, count)):
                best = None; best_score = -1
                maximum_useful_score = sum(len(role_coverage[split][position]) < 7 for position in range(3))
                for candidate in candidates:
                    if candidate in used_seeds:
                        continue
                    candidate_topology = topology(candidate, topology_profile)
                    identity = (topology_profile, *candidate_topology["role_template"])
                    if identity in used_role_triples:
                        continue
                    score = sum(host not in role_coverage[split][position]
                                for position, host in enumerate(candidate_topology["role_template"]))
                    if score > best_score:
                        best = (candidate, candidate_topology, identity); best_score = score
                        if score == maximum_useful_score:
                            break
                if best is None:
                    raise RuntimeError("unable to allocate unique V6 role triple")
                seed, topo, identity = best
                used_seeds.add(seed); used_role_triples.add(identity)
                for position, host in enumerate(topo["role_template"]):
                    role_coverage[split][position].add(host)
                background = min(BACKGROUND_PROFILES, key=lambda value: (
                    global_background[(split, value)],
                    topology_background[(split, topology_profile, value)],
                    (BACKGROUND_PROFILES.index(value) - family_index) % len(BACKGROUND_PROFILES)))
                service = min(SERVICE_PROFILES, key=lambda value: (
                    global_service[(split, value)],
                    topology_service[(split, topology_profile, value)],
                    profile_cross[(split, topology_profile, background, value)],
                    (SERVICE_PROFILES.index(value) - family_index) % len(SERVICE_PROFILES)))
                global_background[(split, background)] += 1
                topology_background[(split, topology_profile, background)] += 1
                global_service[(split, service)] += 1
                topology_service[(split, topology_profile, service)] += 1
                profile_cross[(split, topology_profile, background, service)] += 1
                family = f"v6_{split}_{cohort}_{seed}"
                evaluation_scope = ("development" if split == "train" else "selection" if split == "validation"
                                    else "sealed_topology_holdout" if topology_profile == "dual_zone7_holdout"
                                    else "sealed_in_domain")
                active_count = len(topo["active_hosts"])
                for scenario in SCENARIOS[cohort]:
                    rows.append({"episode_id": f"lab_{episode_number:03d}", "scenario": scenario,
                        "seed": seed, "duration_seconds": 150, "cohort": cohort, "split": split,
                        "evaluation_scope": evaluation_scope, "paired_family": family,
                        "defender_action": ACTIONS.get(scenario, "none"),
                        "background_profile": background, "topology_profile": topology_profile,
                        "active_node_count": active_count, "service_profile": service,
                        "telemetry_profile": "network_auth", "prefix_contract": PREFIX[cohort],
                        "forecast_focus": FOCUS[cohort]})
                    episode_number += 1
                family_index += 1
    if episode_number != 381 or len(rows) != 192:
        raise AssertionError("V6 plan size drift")
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mvp_v6_episode_plan.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "configs/mvp_v6_split_assignments.csv")
    args = parser.parse_args()
    rows = build_rows()
    write_csv(args.plan, FIELDS, rows)
    write_csv(args.splits, ["episode_id", "split"],
              [{"episode_id": row["episode_id"], "split": row["split"]} for row in rows])
    print(f"V6 prospective plan: {len(rows)} episodes / {len(rows)//2} families / {sum(int(r['duration_seconds']) for r in rows)/3600:.1f} capture hours")
    print("No capture, model input, or V5 artifact was accessed or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
