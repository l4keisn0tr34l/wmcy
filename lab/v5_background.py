#!/usr/bin/env python3
"""Bounded benign traffic worker inside a V5 container; allowlisted peers only."""
import argparse
import json
import random
import subprocess
import time

IPS = ("10.77.0.20", "10.77.0.25", "10.77.0.30", "10.77.0.40", "10.77.0.50")
p = argparse.ArgumentParser()
p.add_argument("--source", choices=IPS, required=True)
p.add_argument("--profile", choices=["quiet", "web", "admin", "mixed"], required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--until", type=float, required=True)
a = p.parse_args()
rng = random.Random(f"v5-background-{a.seed}-{a.source}")
peers = [ip for ip in IPS if ip != a.source]
next_time = time.time()
while time.time() < a.until:
    time.sleep(max(0, min(next_time - time.time(), a.until - time.time())))
    remaining = a.until - time.time()
    if remaining <= 0: break
    kind = rng.choice(["quiet", "web", "admin"]) if a.profile == "mixed" else a.profile
    peer = rng.choice(peers)
    commands = {
        "quiet": ["ping", "-c", "1", "-W", "1", peer],
        "web": ["curl", "--noproxy", "*", "-fsS", "--max-time", "2", f"http://{peer}:8080/"],
        "admin": ["sshpass", "-p", "labpass", "ssh", "-o", "StrictHostKeyChecking=no",
                  "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=2",
                  f"lab@{peer}", "true"],
    }
    started = time.time()
    try:
        result = subprocess.run(commands[kind], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=min(5, remaining)).returncode
    except subprocess.TimeoutExpired: result = 124
    print(json.dumps({"start_epoch": started, "source": a.source, "destination": peer,
                      "kind": kind, "returncode": result}), flush=True)
    next_time = time.time() + rng.uniform(3, 7) if kind == "web" else time.time() + rng.uniform(8, 14)
