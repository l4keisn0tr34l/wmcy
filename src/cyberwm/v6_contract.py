"""Prospective V6 graph/topology contract; constants are not inference labels.

V6 keeps dynamic telemetry separate from defender-known inventory/policy topology.
Scenario, seed, role, split, and outcome fields remain audit metadata only.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

HOSTS = {
    "ws1": "10.77.0.20", "ws2": "10.77.0.25", "ws3": "10.77.0.27",
    "srv1": "10.77.0.30", "srv2": "10.77.0.40",
    "admin1": "10.77.0.50", "jump1": "10.77.0.60",
}
HOST_NAMES = tuple(HOSTS)
IPS = tuple(HOSTS.values())
PAIRS = tuple((source, destination) for source in IPS for destination in IPS if source != destination)
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIRS)}
TOPOLOGIES = ("flat5", "segmented7", "dual_zone7_holdout")
BACKGROUND_PROFILES = ("quiet", "web", "admin", "mixed")
SERVICE_PROFILES = ("ssh_password", "ssh_key")
ACTIONS = ("permit_ssh", "block_ssh")
TECHNIQUES = ("T1046", "T1110.001", "T1078", "T1021.004")

# Dynamic packet features remain compatible in meaning with V5. Proposed host
# observables are limited to authentication/session records available to a
# defender; no controller action name, role, or success truth is a feature.
GLOBAL_DYNAMIC = "flow_count unique_hosts unique_src_hosts unique_dst_hosts unique_edges new_edges internal_edges total_bytes total_packets syn_count rst_count fin_count max_out_fanout mean_out_fanout dst_port_entropy auth_success_count auth_failure_count hosts_reporting_auth auth_telemetry_mask".split()
NODE_DYNAMIC = "outgoing_flows outgoing_bytes outgoing_packets out_neighbors unique_dst_ports out_syn out_rst incoming_flows incoming_bytes incoming_packets in_neighbors unique_src_ports in_syn in_rst is_internal new_out_neighbors new_in_neighbors activity_mask auth_success auth_failure password_auth publickey_auth new_remote_auth_source auth_telemetry_mask".split()
EDGE_DYNAMIC = "flow_count bytes_total packets_total syn_count ack_count rst_count fin_count mean_flow_duration unique_dst_ports internal_edge is_new_edge presence_mask auth_success auth_failure new_auth_edge auth_telemetry_mask".split()
TOPOLOGY_EDGE_FEATURES = ("inventory_pair", "same_zone", "ssh_allowed", "http_allowed", "route_hops_normalized")
DYNAMIC_WIDTH = len(GLOBAL_DYNAMIC) + len(IPS) * len(NODE_DYNAMIC) + len(PAIRS) * len(EDGE_DYNAMIC)


def _shuffled_hosts(seed: int, namespace: str) -> list[str]:
    values = list(HOST_NAMES)
    random.Random(f"v6-{namespace}-{seed}").shuffle(values)
    return values


def topology(seed: int, profile: str) -> dict[str, Any]:
    """Return defender-known inventory/zones/policy, never scenario roles."""
    if profile not in TOPOLOGIES:
        raise ValueError(f"unknown topology: {profile}")
    order = _shuffled_hosts(seed, f"topology-{profile}")
    if profile == "flat5":
        active = order[:5]
        zones = {host: "flat" for host in active}
        role_template = (active[0], active[1], active[2])
        admin = active[3]
    elif profile == "segmented7":
        active = order
        zones = {host: "client" for host in order[:3]}
        zones.update({host: "server" for host in order[3:5]})
        zones[order[5]] = "jump"; zones[order[6]] = "admin"
        role_template = (order[0], order[5], order[3])
        admin = order[6]
    else:
        active = order
        zones = {host: "zone_a" for host in order[:3]}
        zones.update({host: "zone_b" for host in order[3:6]})
        zones[order[6]] = "jump"
        role_template = (order[0], order[6], order[3])
        admin = order[1]

    node_inventory = np.asarray([host in active for host in HOST_NAMES], dtype=np.float32)
    edge = np.zeros((len(PAIRS), len(TOPOLOGY_EDGE_FEATURES)), dtype=np.float32)
    jump = next((host for host, zone in zones.items() if zone == "jump"), None)
    server_hosts = {host for host, zone in zones.items() if zone in {"server", "zone_b"}}
    for index, (source_ip, destination_ip) in enumerate(PAIRS):
        source = HOST_NAMES[IPS.index(source_ip)]; destination = HOST_NAMES[IPS.index(destination_ip)]
        if source not in active or destination not in active:
            continue
        same = zones[source] == zones[destination]
        if profile == "flat5":
            ssh = http = True; route_hops = 1
        elif profile == "segmented7":
            ssh = same or source == admin or source == jump or destination == jump
            http = same or destination in server_hosts or source == admin
            route_hops = 1 if same or source == jump or destination == jump else 2
        else:
            ssh = http = same or source == jump or destination == jump
            route_hops = 1 if same or source == jump or destination == jump else 2
        edge[index] = (1, float(same), float(ssh), float(http), route_hops / 2)
    actor, pivot, target = role_template
    if not edge[PAIR_INDEX[(HOSTS[actor], HOSTS[pivot])], 2] or not edge[PAIR_INDEX[(HOSTS[pivot], HOSTS[target])], 2]:
        raise AssertionError("role template must have a two-hop SSH path")
    return {"active_hosts": tuple(active), "zones": zones, "role_template": role_template,
            "node_inventory": node_inventory, "edge_topology": edge}


def schedule(seed: int) -> dict[str, int]:
    """Outcome-independent elapsed seconds; designed for a cutoff near100s."""
    rng = random.Random(f"v6-schedule-{seed}")
    decision = rng.randint(96, 106)
    return {"discovery_seconds": rng.randint(18, 24), "guessing_seconds": rng.randint(40, 47),
            "first_hop_seconds": rng.randint(80, 84), "pivot_evidence_seconds": rng.randint(89, 94),
            "decision_seconds": decision, "minimum_capture_seconds": 150,
            "first_session_hold_seconds": 35, "attempts": rng.randint(3, 5)}


def capture_order(episode_id: str) -> str:
    return hashlib.sha256(f"v6-order-62001-{episode_id}".encode()).hexdigest()


def feature_metadata() -> dict[str, Any]:
    return {"context_states": 3, "future_horizon_states": 6, "window_seconds": 5,
            "maximum_node_slots": len(IPS), "directed_pair_slots": len(PAIRS),
            "dynamic_state_feature_count": DYNAMIC_WIDTH,
            "global_dynamic_feature_names": list(GLOBAL_DYNAMIC),
            "node_dynamic_feature_names": list(NODE_DYNAMIC),
            "edge_dynamic_feature_names": list(EDGE_DYNAMIC),
            "node_inventory_shape": [len(IPS)],
            "edge_topology_shape": [len(PAIRS), len(TOPOLOGY_EDGE_FEATURES)],
            "edge_topology_feature_names": list(TOPOLOGY_EDGE_FEATURES),
            "technique_targets": list(TECHNIQUES), "action_types": list(ACTIONS),
            "input_policy": "past packet/auth telemetry plus defender-known inventory/policy; action separate only when chosen before cutoff",
            "excluded": ["scenario", "cohort", "seed", "split", "roles", "ground_truth",
                         "future_state", "future_edges", "future_action", "action_result", "controller_intent"],
            "target_policy": "decode dynamic future only; inventory/topology are conditioning tensors and excluded from state MAE",
            "normalization": "fit dynamic shared-slot statistics on deduplicated training contexts only; do not z-score binary topology masks"}
