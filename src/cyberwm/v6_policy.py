"""Generate allowlisted service-policy firewall rules from the V6 topology contract."""
from __future__ import annotations

from typing import Any

from src.cyberwm.v6_contract import HOSTS, HOST_NAMES, IPS, PAIR_INDEX, topology

SERVICE_PORTS = {"ssh": 22, "http": 8080}


def service_allowed(seed: int, profile: str, source: str, destination: str, service: str) -> bool:
    if source not in HOSTS or destination not in HOSTS or source == destination:
        raise ValueError("source/destination must be distinct V6 inventory hosts")
    if service not in SERVICE_PORTS:
        raise ValueError(f"unsupported service: {service}")
    value = topology(seed, profile)
    pair = PAIR_INDEX[(HOSTS[source], HOSTS[destination])]
    feature = 2 if service == "ssh" else 3
    return bool(value["edge_topology"][pair, feature])


def rejection_rules(seed: int, profile: str) -> dict[str, list[list[str]]]:
    """Return destination-host INPUT rejects for disallowed SSH/HTTP initiations.

    Rules constrain only the two declared service ports. They do not claim to
    implement a routed L3 network; ICMP/background reachability is separate.
    """
    rules: dict[str, list[list[str]]] = {host: [] for host in HOST_NAMES}
    for destination in HOST_NAMES:
        for source in HOST_NAMES:
            if source == destination:
                continue
            for service, port in SERVICE_PORTS.items():
                if not service_allowed(seed, profile, source, destination, service):
                    rules[destination].append(["-p", "tcp", "-s", HOSTS[source],
                        "--dport", str(port), "-j", "REJECT", "--reject-with", "tcp-reset"])
    return rules


def allowed_peers(seed: int, profile: str, source: str, service: str) -> list[str]:
    """Return destination IPs permitted by inventory and declared service policy."""
    return [HOSTS[destination] for destination in HOST_NAMES
            if destination != source and service_allowed(seed, profile, source, destination, service)]


def policy_manifest(seed: int, profile: str) -> dict[str, Any]:
    value = topology(seed, profile)
    return {"seed": seed, "topology_profile": profile,
            "active_hosts": list(value["active_hosts"]), "zones": value["zones"],
            "service_ports": SERVICE_PORTS,
            "rule_count_by_destination": {host: len(items) for host, items in rejection_rules(seed, profile).items()},
            "scope": "inventory SSH/HTTP service initiation policy; not arbitrary L3 routing"}
