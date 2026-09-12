#!/usr/bin/env python3
"""Realize V6 service policies in Docker and test only inventory endpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess as sp
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v6_contract import HOSTS, HOST_NAMES, topology
from src.cyberwm.v6_policy import policy_manifest, rejection_rules

COMPOSE = ["docker", "compose", "-p", "cyberwm_v6", "-f", str(ROOT / "lab/docker-compose-v6.yml")]
V5_COMPOSE = ["docker", "compose", "-p", "cyberwm_v5", "-f", str(ROOT / "lab/docker-compose-v5.yml")]
OUT = ROOT / "outputs/mvp_v6/review/policy_feasibility.json"
SEED = 999001  # smoke-only seed; absent from the prospective episode plan


def run(command: list[str], check: bool = True, **kwargs) -> sp.CompletedProcess:
    return sp.run(command, check=check, timeout=kwargs.pop("timeout", 45), **kwargs)


def compose_exec(host: str, command: list[str], check: bool = True) -> sp.CompletedProcess:
    if host not in HOST_NAMES:
        raise ValueError("host outside V6 inventory")
    return run(COMPOSE + ["exec", "-T", host] + command, check=check,
               stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=20)


def apply(profile: str) -> dict[str, int]:
    rules = rejection_rules(SEED, profile)
    installed = {}
    for destination, items in rules.items():
        compose_exec(destination, ["iptables", "-F", "INPUT"])
        for args in items:
            compose_exec(destination, ["iptables", "-A", "INPUT"] + args)
        listing = run(COMPOSE + ["exec", "-T", destination, "iptables", "-S", "INPUT"],
                      capture_output=True, text=True).stdout.splitlines()
        installed[destination] = sum(line.startswith("-A INPUT ") for line in listing)
        if installed[destination] != len(items):
            raise RuntimeError(f"installed policy count differs for {destination}")
    return installed


def ssh(source: str, destination: str) -> int:
    return compose_exec(source, ["ssh", "-i", "/home/lab/.ssh/id_ed25519", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=2", f"lab@{HOSTS[destination]}", "true"], check=False).returncode


def http(source: str, destination: str) -> int:
    return compose_exec(source, ["curl", "--noproxy", "*", "-fsS", "--max-time", "2",
                         f"http://{HOSTS[destination]}:8080/"], check=False).returncode


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"preserve existing feasibility result: {OUT}")
    for compose in (V5_COMPOSE, COMPOSE):
        if run(compose + ["ps", "--services", "--status", "running"], capture_output=True,
               text=True).stdout.strip():
            raise RuntimeError("V5/V6 containers already running")
    networks = run(["docker", "network", "ls", "--format", "{{.Name}}"], capture_output=True,
                   text=True).stdout.splitlines()
    if "cyberwm_v5_labnet_v5" in networks:
        raise RuntimeError("empty V5 network occupies authorized subnet; remove explicitly")
    started = stamp(); success = False
    try:
        run(COMPOSE + ["up", "-d", "--force-recreate", "--no-build"], timeout=90)
        image_ids = set()
        for service in HOST_NAMES:
            container = run(COMPOSE + ["ps", "-q", service], capture_output=True, text=True).stdout.strip()
            image_ids.add(run(["docker", "inspect", "--format", "{{.Image}}", container],
                              capture_output=True, text=True).stdout.strip())
        if len(image_ids) != 1:
            raise RuntimeError("V6 policy smoke does not use one shared image")

        dual = topology(SEED, "dual_zone7_holdout")
        zone_a = next(host for host, zone in dual["zones"].items() if zone == "zone_a")
        zone_b = next(host for host, zone in dual["zones"].items() if zone == "zone_b")
        jump = next(host for host, zone in dual["zones"].items() if zone == "jump")
        dual_counts = apply("dual_zone7_holdout")
        dual_rc = {"direct_cross_ssh": ssh(zone_a, zone_b),
                   "zone_a_to_jump_ssh": ssh(zone_a, jump),
                   "jump_to_zone_b_ssh": ssh(jump, zone_b),
                   "direct_cross_http": http(zone_a, zone_b),
                   "jump_to_zone_b_http": http(jump, zone_b)}
        if dual_rc["direct_cross_ssh"] == 0 or dual_rc["direct_cross_http"] == 0:
            raise RuntimeError("dual-zone direct cross-zone service was not blocked")
        if any(dual_rc[key] != 0 for key in
               ("zone_a_to_jump_ssh", "jump_to_zone_b_ssh", "jump_to_zone_b_http")):
            raise RuntimeError(f"a permitted dual-zone leg failed: {dual_rc}")

        flat = topology(SEED, "flat5")
        active = list(flat["active_hosts"]); inactive = next(host for host in HOST_NAMES if host not in active)
        source = next(host for host in active if host != inactive)
        destination = next(host for host in active if host != source)
        flat_counts = apply("flat5")
        flat_rc = {"active_to_active_ssh": ssh(source, destination),
                   "active_to_inactive_ssh": ssh(source, inactive),
                   "active_to_active_http": http(source, destination),
                   "active_to_inactive_http": http(source, inactive)}
        if flat_rc["active_to_active_ssh"] != 0 or flat_rc["active_to_active_http"] != 0:
            raise RuntimeError(f"flat active service failed: {flat_rc}")
        if flat_rc["active_to_inactive_ssh"] == 0 or flat_rc["active_to_inactive_http"] == 0:
            raise RuntimeError("flat inactive slot accepted service traffic")
        completed = stamp(); success = True
    finally:
        sp.run(COMPOSE + ["down", "-t", "2"], timeout=60, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    if not success:
        raise RuntimeError("V6 real service-policy feasibility failed; containers were stopped")
    result = {"status": "PASS_NON_CAPTURE_POLICY_FEASIBILITY",
        "scope": "inventory SSH/HTTP service policy; two permitted legs are not an L3 relay claim",
        "smoke_seed": SEED, "seed_is_absent_from_episode_plan": True,
        "started_utc": started, "completed_utc": completed, "shared_image_id": next(iter(image_ids)),
        "dual_zone": {"profile": "dual_zone7_holdout", "zone_a_host": zone_a,
            "zone_b_host": zone_b, "jump_host": jump, "returncodes": dual_rc,
            "installed_rule_count": dual_counts,
            "manifest": policy_manifest(SEED, "dual_zone7_holdout")},
        "flat": {"profile": "flat5", "active_source": source, "active_destination": destination,
            "inactive_destination": inactive, "returncodes": flat_rc,
            "installed_rule_count": flat_counts, "manifest": policy_manifest(SEED, "flat5")},
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            ROOT / "lab/docker-compose-v6.yml", ROOT / "src/cyberwm/v6_contract.py",
            ROOT / "src/cyberwm/v6_policy.py", Path(__file__).resolve())}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".tmp"); temporary.write_text(json.dumps(result, indent=2) + "\n"); temporary.replace(OUT)
    print(json.dumps({"status": result["status"], "out": str(OUT),
                      "dual_returncodes": dual_rc, "flat_returncodes": flat_rc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
