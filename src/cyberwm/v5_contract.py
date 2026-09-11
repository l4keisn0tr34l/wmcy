"""V5 constants and deterministic nuisance schedules; never inference features."""
from __future__ import annotations
import hashlib
import random

HOSTS = {"ws1": "10.77.0.20", "ws2": "10.77.0.25", "srv1": "10.77.0.30",
         "srv2": "10.77.0.40", "admin1": "10.77.0.50"}
IPS = tuple(HOSTS.values())
PAIRS = tuple((s, d) for s in IPS for d in IPS if s != d)
TECHNIQUES = ("T1046", "T1110.001", "T1021.004")
ACTIONS = ("permit_ssh", "block_ssh")
PROFILES = ("quiet", "web", "admin", "mixed")
GLOBAL = "flow_count unique_hosts unique_src_hosts unique_dst_hosts unique_edges new_edges internal_edges total_bytes total_packets syn_count rst_count fin_count max_out_fanout mean_out_fanout dst_port_entropy".split()
NODE = "outgoing_flows outgoing_bytes outgoing_packets out_neighbors unique_dst_ports out_syn out_rst incoming_flows incoming_bytes incoming_packets in_neighbors unique_src_ports in_syn in_rst is_internal new_out_neighbors new_in_neighbors activity_mask".split()
EDGE = "flow_count bytes_total packets_total syn_count ack_count rst_count fin_count mean_flow_duration unique_dst_ports internal_edge is_new_edge presence_mask".split()
WIDTH = len(GLOBAL) + len(IPS)*len(NODE) + len(PAIRS)*len(EDGE)


def roles(seed: int) -> list[str]:
    hosts = list(HOSTS); random.Random(seed).shuffle(hosts); return hosts


def schedule(seed: int) -> dict:
    # Independent RNG: alternative scenario or action must not shift its draws.
    rng = random.Random(f"v5-schedule-{seed}")
    return {"discovery_seconds": rng.randint(20, 28), "guessing_seconds": rng.randint(43, 50),
            "decision_seconds": rng.randint(80, 103), "attempts": rng.randint(3, 4)}


def capture_order(episode_id: str) -> str:
    return hashlib.sha256(f"v5-order-51001-{episode_id}".encode()).hexdigest()


def feature_metadata() -> dict:
    return {"context_states": 3, "future_horizon_states": 6, "window_seconds": 5,
            "state_feature_count": WIDTH, "global_feature_names": GLOBAL,
            "node_feature_names": NODE, "edge_feature_names": EDGE,
            "state_feature_names": GLOBAL + [f"node_{i}__{f}" for i in range(5) for f in NODE]
                + [f"edge_{i}__{f}" for i in range(20) for f in EDGE],
            "node_slots": [{"slot": i, "ip": ip} for i, ip in enumerate(IPS)],
            "directed_edge_slots": [{"slot": i, "source_ip": s, "destination_ip": d}
                                    for i, (s, d) in enumerate(PAIRS)],
            "technique_targets": list(TECHNIQUES), "action_types": list(ACTIONS),
            "input_policy": "raw observable context only; action type/pair separate and known by forecast_time; audit metadata and future truth excluded",
            "normalization": "none here; fit shared-slot scaler on train contexts only"}
