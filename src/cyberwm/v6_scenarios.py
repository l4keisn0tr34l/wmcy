"""Prospective V6 controller programs; never inference inputs.

Programs specify intentional operations and truth semantics before capture. Actual
packet and OpenSSH timestamps remain authoritative observables at processing.
"""
from __future__ import annotations

from typing import Any

from src.cyberwm.v6_contract import HOST_NAMES, SERVICE_PROFILES, TOPOLOGIES, schedule, topology

SCENARIOS = (
    "second_hop_action_permit", "second_hop_action_block",
    "credential_pressure_progress", "matched_legitimate_admin",
    "shared_prefix_stop", "shared_prefix_progress",
    "pivot_probe_then_stop", "pivot_probe_then_second_hop",
    "background_only", "legitimate_admin_ssh",
)
SAME_OBSERVABLE_PREFIX_PAIRS = (
    ("second_hop_action_permit", "second_hop_action_block"),
    ("shared_prefix_stop", "shared_prefix_progress"),
    ("pivot_probe_then_stop", "pivot_probe_then_second_hop"),
)


def controller_roles(seed: int, topology_profile: str) -> dict[str, str]:
    """Return hidden controller roles; callers must not export these as features."""
    value = topology(seed, topology_profile)
    actor, pivot, target = value["role_template"]
    if topology_profile == "flat5":
        administrator = value["active_hosts"][3]
    elif topology_profile == "segmented7":
        administrator = next(host for host, zone in value["zones"].items() if zone == "admin")
    else:
        administrator = next(host for host in value["active_hosts"]
                             if host not in {actor, pivot, target})
    roles = {"actor": actor, "pivot": pivot, "target": target, "administrator": administrator}
    if not set(roles.values()).issubset(set(value["active_hosts"])):
        raise AssertionError("controller role outside active topology")
    return roles


def _operation(kind: str, phase: str, source_role: str | None = None,
               destination_role: str | None = None, **values: Any) -> dict[str, Any]:
    return {"kind": kind, "phase": phase, "source_role": source_role,
            "destination_role": destination_role, **values}


def scenario_program(seed: int, topology_profile: str, service_profile: str,
                     scenario: str) -> dict[str, Any]:
    if topology_profile not in TOPOLOGIES:
        raise ValueError(f"unknown topology profile: {topology_profile}")
    if service_profile not in SERVICE_PROFILES:
        raise ValueError(f"unknown service profile: {service_profile}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown V6 scenario: {scenario}")
    timing = schedule(seed); method = service_profile.removeprefix("ssh_")
    operations: list[dict[str, Any]] = []

    pressure = scenario in {"credential_pressure_progress", "shared_prefix_stop", "shared_prefix_progress"}
    two_hop = scenario in {"second_hop_action_permit", "second_hop_action_block",
                           "pivot_probe_then_stop", "pivot_probe_then_second_hop"}
    if pressure:
        operations.append(_operation("network_discovery", "pre_cutoff", "actor", None,
            elapsed_seconds=timing["discovery_seconds"], observable_consequence=True,
            technique_id="T1046", truth_semantics="intentional_discovery"))
        operations.append(_operation("authentication_pressure", "pre_cutoff", "actor", "pivot",
            elapsed_seconds=timing["guessing_seconds"], observable_consequence=True,
            attempts=timing["attempts"], auth_method=method, technique_id="T1110.001",
            truth_semantics="failed_authentication_only", completed_lm=False))
    if scenario == "matched_legitimate_admin":
        operations.append(_operation("legitimate_authentication_burst", "pre_cutoff",
            "administrator", "pivot", elapsed_seconds=timing["guessing_seconds"],
            observable_consequence=True, attempts=timing["attempts"], auth_method=method,
            technique_id=None, truth_semantics="benign_administration", completed_lm=False))
    if two_hop:
        operations.append(_operation("first_hop_session", "pre_cutoff", "actor", "pivot",
            elapsed_seconds=timing["first_hop_seconds"], hold_seconds=timing["first_session_hold_seconds"],
            observable_consequence=True, auth_method=method,
            technique_ids=["T1078", "T1021.004"], truth_semantics="completed_lateral_movement",
            completed_lm=True))
        operations.append(_operation("pivot_discovery", "pre_cutoff", "pivot", "target",
            elapsed_seconds=timing["pivot_evidence_seconds"], observable_consequence=True,
            technique_id="T1046", truth_semantics="intentional_target_service_discovery"))
    if scenario.startswith("second_hop_action_"):
        action = "permit_ssh" if scenario.endswith("permit") else "block_ssh"
        operations.append(_operation("defender_action_decision", "decision", "pivot", "target",
            elapsed_seconds=timing["decision_seconds"], observable_consequence=False,
            action=action, known_before_cutoff=True))
        operations.append(_operation("defender_action_effective", "post_cutoff", "pivot", "target",
            after_cutoff_seconds=0.2, observable_consequence=True, action=action))

    malicious_future = {
        "credential_pressure_progress": ("actor", "pivot"),
        "shared_prefix_progress": ("actor", "pivot"),
        "pivot_probe_then_second_hop": ("pivot", "target"),
        "second_hop_action_permit": ("pivot", "target"),
        "second_hop_action_block": ("pivot", "target"),
    }
    if scenario in malicious_future:
        source, destination = malicious_future[scenario]
        blocked = scenario == "second_hop_action_block"
        operations.append(_operation("focal_ssh", "post_cutoff", source, destination,
            after_cutoff_seconds=2.0, observable_consequence=True, auth_method=method,
            expected_auth_result="blocked_before_auth" if blocked else "accepted",
            technique_ids=["T1021.004"] if blocked else ["T1078", "T1021.004"],
            truth_semantics="lateral_movement_attempt" if blocked else "completed_lateral_movement",
            completed_lm=not blocked))
    elif scenario in {"matched_legitimate_admin", "legitimate_admin_ssh"}:
        operations.append(_operation("legitimate_future_ssh", "post_cutoff", "administrator", "target",
            after_cutoff_seconds=2.0, observable_consequence=True, auth_method=method,
            expected_auth_result="accepted", technique_ids=[],
            truth_semantics="benign_administration", completed_lm=False))

    return {"scenario": scenario, "seed": seed, "topology_profile": topology_profile,
            "service_profile": service_profile, "roles": controller_roles(seed, topology_profile),
            "schedule": timing, "operations": operations,
            "forecast_cutoff_rule": "first absolute five-second boundary after decision",
            "future_horizon_seconds": 30, "model_input": False}


def observable_prefix_signature(program: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Signature of consequences observable before cutoff, excluding hidden truth."""
    signature = []
    for operation in program["operations"]:
        if operation["phase"] != "pre_cutoff" or not operation["observable_consequence"]:
            continue
        signature.append((operation["kind"], operation["source_role"], operation["destination_role"],
                          operation.get("elapsed_seconds"), operation.get("hold_seconds"),
                          operation.get("attempts"), operation.get("auth_method")))
    return tuple(signature)
