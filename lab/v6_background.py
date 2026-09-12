#!/usr/bin/env python3
"""Bounded benign V6 traffic worker; only controller-provided inventory peers."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time

INVENTORY = {"10.77.0.20", "10.77.0.25", "10.77.0.27", "10.77.0.30",
             "10.77.0.40", "10.77.0.50", "10.77.0.60"}


def peers(value: str) -> list[str]:
    parsed = [item for item in value.split(",") if item]
    if len(parsed) != len(set(parsed)) or not set(parsed).issubset(INVENTORY):
        raise argparse.ArgumentTypeError("peer list must be unique and inside V6 inventory")
    return parsed


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", choices=sorted(INVENTORY), required=True)
parser.add_argument("--profile", choices=["quiet", "web", "admin", "mixed"], required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--until", type=float, required=True)
parser.add_argument("--ping-peers", type=peers, required=True)
parser.add_argument("--web-peers", type=peers, required=True)
parser.add_argument("--ssh-peers", type=peers, required=True)
parser.add_argument("--auth-method", choices=["password", "publickey"], required=True)
args = parser.parse_args()
if args.source in args.ping_peers + args.web_peers + args.ssh_peers:
    parser.error("source cannot be its own peer")

rng = random.Random(f"v6-background-{args.seed}-{args.source}")
peer_sets = {"quiet": args.ping_peers, "web": args.web_peers, "admin": args.ssh_peers}
next_time = time.time()
while time.time() < args.until:
    time.sleep(max(0, min(next_time - time.time(), args.until - time.time())))
    remaining = args.until - time.time()
    if remaining <= 0:
        break
    kind = rng.choice(["quiet", "web", "admin"]) if args.profile == "mixed" else args.profile
    available = peer_sets[kind]
    if not available:
        print(json.dumps({"start_epoch": time.time(), "source": args.source,
                          "destination": None, "kind": kind, "returncode": None,
                          "status": "no_policy_allowed_peer"}), flush=True)
        next_time = time.time() + rng.uniform(3, 7)
        continue
    destination = rng.choice(available)
    if kind == "quiet":
        command = ["ping", "-c", "1", "-W", "1", destination]
    elif kind == "web":
        command = ["curl", "--noproxy", "*", "-fsS", "--max-time", "2",
                   f"http://{destination}:8080/"]
    elif args.auth_method == "password":
        command = ["sshpass", "-p", "labpass", "ssh", "-o", "StrictHostKeyChecking=no",
                   "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=2",
                   "-o", "PubkeyAuthentication=no", f"lab@{destination}", "true"]
    else:
        command = ["ssh", "-i", "/home/lab/.ssh/id_ed25519", "-o", "BatchMode=yes",
                   "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                   "-o", "ConnectTimeout=2", f"lab@{destination}", "true"]
    started = time.time()
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=min(5, remaining)).returncode
    except subprocess.TimeoutExpired:
        result = 124
    print(json.dumps({"start_epoch": started, "source": args.source, "destination": destination,
                      "kind": kind, "returncode": result, "status": "attempted"}), flush=True)
    next_time = time.time() + (rng.uniform(3, 7) if kind == "web" else rng.uniform(8, 14))
