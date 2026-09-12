#!/usr/bin/env python3
"""Synthetic prospective tests for V6 controller programs; no captures accessed."""
from __future__ import annotations

import csv
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v6_contract import PAIR_INDEX, HOSTS, TOPOLOGIES, topology
from src.cyberwm.v6_scenarios import (SAME_OBSERVABLE_PREFIX_PAIRS, SCENARIOS,
    controller_roles, observable_prefix_signature, scenario_program)


def plan_rows():
    with (ROOT / "configs/mvp_v6_episode_plan.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


class V6ScenarioTests(unittest.TestCase):
    def test_plan_scenarios_match_program_vocabulary(self):
        rows = plan_rows()
        self.assertEqual({row["scenario"] for row in rows}, set(SCENARIOS))
        for row in rows:
            program = scenario_program(int(row["seed"]), row["topology_profile"],
                                       row["service_profile"], row["scenario"])
            self.assertFalse(program["model_input"])
            self.assertEqual(program["schedule"]["minimum_capture_seconds"], 150)

    def test_controller_roles_are_active_and_two_hop_paths_allow_ssh(self):
        for seed in (20001, 27119, 999001):
            for profile in TOPOLOGIES:
                value = topology(seed, profile); roles = controller_roles(seed, profile)
                self.assertTrue(set(roles.values()).issubset(set(value["active_hosts"])))
                for source, destination in ((roles["actor"], roles["pivot"]),
                                            (roles["pivot"], roles["target"])):
                    pair = PAIR_INDEX[(HOSTS[source], HOSTS[destination])]
                    self.assertEqual(value["edge_topology"][pair, 2], 1)

    def test_declared_same_prefix_pairs_have_identical_observable_consequences(self):
        for seed in (20001, 27119, 999001):
            for profile in TOPOLOGIES:
                for service in ("ssh_password", "ssh_key"):
                    for left, right in SAME_OBSERVABLE_PREFIX_PAIRS:
                        lhs = observable_prefix_signature(scenario_program(seed, profile, service, left))
                        rhs = observable_prefix_signature(scenario_program(seed, profile, service, right))
                        self.assertEqual(lhs, rhs, (seed, profile, service, left, right))

    def test_action_is_chosen_before_cutoff_and_consequences_follow_it(self):
        for scenario in ("second_hop_action_permit", "second_hop_action_block"):
            program = scenario_program(999001, "segmented7", "ssh_password", scenario)
            decision = next(op for op in program["operations"] if op["kind"] == "defender_action_decision")
            effective = next(op for op in program["operations"] if op["kind"] == "defender_action_effective")
            focal = next(op for op in program["operations"] if op["kind"] == "focal_ssh")
            self.assertTrue(decision["known_before_cutoff"])
            self.assertFalse(decision["observable_consequence"])
            self.assertEqual(effective["phase"], "post_cutoff")
            self.assertLess(effective["after_cutoff_seconds"], focal["after_cutoff_seconds"])

    def test_attempted_and_completed_lm_semantics_are_distinct(self):
        blocked = scenario_program(999001, "segmented7", "ssh_key", "second_hop_action_block")
        permit = scenario_program(999001, "segmented7", "ssh_key", "second_hop_action_permit")
        blocked_focal = next(op for op in blocked["operations"] if op["kind"] == "focal_ssh")
        permit_focal = next(op for op in permit["operations"] if op["kind"] == "focal_ssh")
        self.assertFalse(blocked_focal["completed_lm"])
        self.assertEqual(blocked_focal["technique_ids"], ["T1021.004"])
        self.assertEqual(blocked_focal["expected_auth_result"], "blocked_before_auth")
        self.assertTrue(permit_focal["completed_lm"])
        self.assertIn("T1078", permit_focal["technique_ids"])
        self.assertEqual(permit_focal["expected_auth_result"], "accepted")

    def test_all_planned_cutoffs_leave_complete_six_state_future(self):
        rows = plan_rows()
        for row in rows:
            program = scenario_program(int(row["seed"]), row["topology_profile"],
                                       row["service_profile"], row["scenario"])
            # The next absolute boundary is strictly less than five seconds after decision.
            latest_cutoff = program["schedule"]["decision_seconds"] + 5
            self.assertLessEqual(latest_cutoff + program["future_horizon_seconds"], 150)
            for operation in program["operations"]:
                if operation["phase"] == "post_cutoff":
                    self.assertLess(operation["after_cutoff_seconds"], 30)

    def test_observable_contrast_cohorts_are_not_claimed_as_same_prefix(self):
        pressure = scenario_program(999001, "flat5", "ssh_password", "credential_pressure_progress")
        legitimate = scenario_program(999001, "flat5", "ssh_password", "matched_legitimate_admin")
        self.assertNotEqual(observable_prefix_signature(pressure), observable_prefix_signature(legitimate))
        background = scenario_program(999001, "flat5", "ssh_password", "background_only")
        admin = scenario_program(999001, "flat5", "ssh_password", "legitimate_admin_ssh")
        self.assertEqual(observable_prefix_signature(background), observable_prefix_signature(admin))
        self.assertNotEqual(background["operations"], admin["operations"])


if __name__ == "__main__":
    unittest.main()
