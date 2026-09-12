#!/usr/bin/env python3
"""Synthetic prospective V6 contract tests; never access capture/test model artifacts."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


contract = load_script("v6_contract_test", "src/cyberwm/v6_contract.py")
builder = load_script("v6_builder_test", "scripts/62_build_v6_plan.py")
validator = load_script("v6_validator_test", "scripts/63_validate_v6_plan.py")
policy = load_script("v6_policy_test", "src/cyberwm/v6_policy.py")


class V6ContractTests(unittest.TestCase):
    def test_feature_contract_separates_dynamic_and_static_topology(self):
        metadata = contract.feature_metadata()
        self.assertEqual(metadata["maximum_node_slots"], 7)
        self.assertEqual(metadata["directed_pair_slots"], 42)
        self.assertEqual(metadata["dynamic_state_feature_count"], contract.DYNAMIC_WIDTH)
        self.assertEqual(metadata["node_inventory_shape"], [7])
        self.assertEqual(metadata["edge_topology_shape"], [42, 5])
        self.assertIn("excluded from state MAE", metadata["target_policy"])
        self.assertEqual(metadata["technique_targets"], ["T1046", "T1110.001", "T1078", "T1021.004"])
        names = (metadata["global_dynamic_feature_names"] + metadata["node_dynamic_feature_names"]
                 + metadata["edge_dynamic_feature_names"])
        forbidden = {"scenario", "seed", "split", "role", "ground_truth", "future",
                     "action_result", "controller_intent", "label"}
        self.assertFalse(any(any(token in name.lower() for token in forbidden) for name in names))
        self.assertFalse(any("session" in name for name in names))

    def test_topology_tensors_are_finite_and_mask_inactive_pairs(self):
        for seed in (20001, 27119, 49999):
            for profile in contract.TOPOLOGIES:
                value = contract.topology(seed, profile)
                node = value["node_inventory"]; edge = value["edge_topology"]
                self.assertEqual(node.shape, (7,)); self.assertEqual(edge.shape, (42, 5))
                self.assertTrue(np.isfinite(edge).all())
                self.assertTrue(np.isin(node, [0, 1]).all())
                for index, (source, destination) in enumerate(contract.PAIRS):
                    expected = node[contract.IPS.index(source)] * node[contract.IPS.index(destination)]
                    self.assertEqual(edge[index, 0], expected)
                    if not expected:
                        self.assertTrue(np.array_equal(edge[index], np.zeros(5)))
                    else:
                        self.assertGreater(edge[index, 4], 0)
                        self.assertLessEqual(edge[index, 4], 1)

    def test_each_role_template_has_two_hop_ssh_path(self):
        for seed in range(20000, 20040):
            for profile in contract.TOPOLOGIES:
                value = contract.topology(seed, profile); actor, pivot, target = value["role_template"]
                first = contract.PAIR_INDEX[(contract.HOSTS[actor], contract.HOSTS[pivot])]
                second = contract.PAIR_INDEX[(contract.HOSTS[pivot], contract.HOSTS[target])]
                self.assertEqual(value["edge_topology"][first, 2], 1)
                self.assertEqual(value["edge_topology"][second, 2], 1)
                self.assertEqual(len({actor, pivot, target}), 3)

    def test_service_policy_matches_topology_and_never_leaves_inventory(self):
        for seed in (20001, 27119):
            for profile in contract.TOPOLOGIES:
                value = contract.topology(seed, profile)
                active = set(value["active_hosts"])
                rules = policy.rejection_rules(seed, profile)
                self.assertEqual(set(rules), set(contract.HOST_NAMES))
                for destination, commands in rules.items():
                    for command in commands:
                        self.assertIn(command[3], contract.IPS)
                        self.assertIn(command[5], {"22", "8080"})
                        self.assertEqual(command[-2:], ["--reject-with", "tcp-reset"])
                for source in contract.HOST_NAMES:
                    for service in policy.SERVICE_PORTS:
                        peers = policy.allowed_peers(seed, profile, source, service)
                        self.assertTrue(set(peers).issubset(set(contract.IPS)))
                        self.assertNotIn(contract.HOSTS[source], peers)
                        if source not in active:
                            self.assertEqual(peers, [])

    def test_dual_zone_blocks_direct_cross_zone_but_role_path_is_allowed(self):
        value = contract.topology(27119, "dual_zone7_holdout")
        zones = value["zones"]; jump = next(host for host, zone in zones.items() if zone == "jump")
        zone_a = next(host for host, zone in zones.items() if zone == "zone_a")
        zone_b = next(host for host, zone in zones.items() if zone == "zone_b")
        self.assertFalse(policy.service_allowed(27119, "dual_zone7_holdout", zone_a, zone_b, "ssh"))
        self.assertTrue(policy.service_allowed(27119, "dual_zone7_holdout", zone_a, jump, "ssh"))
        self.assertTrue(policy.service_allowed(27119, "dual_zone7_holdout", jump, zone_b, "ssh"))

    def test_schedule_is_deterministic_and_has_future_margin(self):
        for seed in range(20000, 20100):
            first = contract.schedule(seed); second = contract.schedule(seed)
            self.assertEqual(first, second)
            self.assertLess(first["first_hop_seconds"], first["pivot_evidence_seconds"])
            self.assertLess(first["pivot_evidence_seconds"], first["decision_seconds"])
            self.assertGreaterEqual(first["first_session_hold_seconds"], 35)
            self.assertLessEqual(first["decision_seconds"] + 5 + 30, 145)

    def test_plan_is_deterministic_and_matches_tracked_files(self):
        rows1 = builder.build_rows(); rows2 = builder.build_rows()
        self.assertEqual(rows1, rows2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = root / "plan.csv"; splits = root / "splits.csv"
            builder.write_csv(plan, builder.FIELDS, rows1)
            builder.write_csv(splits, ["episode_id", "split"],
                              [{"episode_id": row["episode_id"], "split": row["split"]} for row in rows1])
            self.assertEqual(plan.read_bytes(), (ROOT / "configs/mvp_v6_episode_plan.csv").read_bytes())
            self.assertEqual(splits.read_bytes(), (ROOT / "configs/mvp_v6_split_assignments.csv").read_bytes())
            summary = validator.validate(plan, splits)
            self.assertEqual(summary, {"episodes": 192, "families": 96, "train_families": 44,
                "validation_families": 20, "test_families": 32, "holdout_test_families": 16})

    def test_validator_rejects_holdout_leakage(self):
        rows = builder.build_rows()
        target = next(row for row in rows if row["topology_profile"] == "dual_zone7_holdout")
        family = target["paired_family"]
        for row in rows:
            if row["paired_family"] == family:
                row["split"] = "train"; row["evaluation_scope"] = "development"
                row["paired_family"] = row["paired_family"].replace("v6_test_", "v6_train_")
        self._expect_rejected(rows)

    def test_validator_rejects_service_topology_proxy(self):
        rows = builder.build_rows()
        for row in rows:
            row["service_profile"] = ("ssh_password" if row["topology_profile"] == "flat5"
                                      else "ssh_key")
        self._expect_rejected(rows)

    def test_validator_rejects_pair_attribute_or_action_drift(self):
        rows = builder.build_rows(); rows[1]["background_profile"] = "mixed"
        self._expect_rejected(rows)
        rows = builder.build_rows(); action = next(row for row in rows if row["defender_action"] != "none")
        action["defender_action"] = "none"
        self._expect_rejected(rows)

    def _expect_rejected(self, rows: list[dict]) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); plan = root / "plan.csv"; splits = root / "splits.csv"
            builder.write_csv(plan, builder.FIELDS, rows)
            builder.write_csv(splits, ["episode_id", "split"],
                              [{"episode_id": row["episode_id"], "split": row["split"]} for row in rows])
            with self.assertRaises(ValueError):
                validator.validate(plan, splits)


if __name__ == "__main__":
    unittest.main()
