#!/usr/bin/env python3
"""Fail-closed V5 capture controller. No external adversarial targets accepted."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import signal
import subprocess as sp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_contract import HOSTS, PROFILES, roles, schedule
from src.cyberwm.v5_integrity import code_hashes, image_hashes

COMPOSE = ["docker", "compose", "-p", "cyberwm_v5", "-f", str(ROOT / "lab/docker-compose-v5.yml")]
SCENARIOS = ("background_only", "legitimate_ssh", "matched_legitimate_ssh", "credential_one_hop",
             "scan_guess_then_stop", "one_hop", "scan_guess_action_permit", "scan_guess_action_block",
             "credential_action_permit", "credential_action_block")
TRUTH_HEADER = "episode_id,start_time,end_time,actor,target,technique_id,technique,tactic".split(",")
ACTION_HEADER = "episode_id,start_time,end_time,action,source,target,known_at_forecast_time,details,forecast_time,effective_start_time".split(",")


def stamp(ns=None):
    ns = time.time_ns() if ns is None else ns
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ns // 10**9)) + f".{ns % 10**9:09d}+00:00"


def run(args, **kw):
    return sp.run(args, check=True, timeout=kw.pop("timeout", 30), **kw)


def exec_host(host, command, **kw):
    if host not in HOSTS: raise ValueError("host not in isolated lab inventory")
    return run(COMPOSE + ["exec", "-T", host] + command, **kw)


def sleep_until(ns):
    while (left := (ns-time.time_ns())/1e9) > 0: time.sleep(min(left, 0.5))


def append(path, row):
    with path.open("a", newline="") as f: csv.writer(f).writerow(row)


def ssh_command(destination, password="labpass", command="hostname && id"):
    return ["timeout", "8", "sshpass", "-p", password, "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "PubkeyAuthentication=no",
            "-o", "PreferredAuthentications=password", "-o", "ConnectTimeout=2",
            "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2", f"lab@{HOSTS[destination]}", command]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode_id")
    p.add_argument("--scenario", choices=SCENARIOS, default="background_only")
    p.add_argument("--background", choices=PROFILES, default="quiet")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--duration-seconds", type=int, choices=[150], default=150)
    p.add_argument("--out-dir", type=Path)
    a = p.parse_args()
    import re
    if not re.fullmatch(r"(?:lab_\d{3}|v5_smoke_[a-z0-9_]+)", a.episode_id) or a.seed < 0:
        p.error("invalid opaque episode ID or seed")
    out = (a.out_dir or ROOT / "lab/episodes" / a.episode_id).resolve()
    if not out.is_relative_to((ROOT / "lab/episodes").resolve()): p.error("output must be under lab/episodes")
    if out.exists(): raise FileExistsError(f"raw episode already exists: {out}")
    # One controller owns all five lab containers. Never reset another capture.
    with (ROOT / "lab/.v5_capture.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not a.episode_id.startswith("v5_smoke_"):
            run([sys.executable, str(ROOT / "scripts/49_freeze_v5_capture.py"), "--check"])
            with (ROOT / "configs/mvp_v5_episode_plan.csv").open() as f:
                rows = [r for r in csv.DictReader(f) if r["episode_id"] == a.episode_id]
            if len(rows) != 1 or any(rows[0][k] != str(v) for k, v in {
                "scenario": a.scenario, "seed": a.seed, "background_profile": a.background,
                "duration_seconds": a.duration_seconds}.items()):
                raise ValueError("episode arguments differ from frozen plan")
        import shutil
        if shutil.disk_usage(ROOT).free < 5*1024**3: raise RuntimeError("less than 5 GiB free; capture refused")
        supplies = list(Path("/sys/class/power_supply").glob("*/type"))
        if any(p.read_text().strip() == "Battery" for p in supplies):
            if not any(p.read_text().strip() == "Mains" and (p.parent/"online").read_text().strip() == "1" for p in supplies):
                raise RuntimeError("laptop must be on AC power")
        run(["sudo", "-v"])
        capture = None; log_handles = []; background = []; block = None; cleanup_ok = False
        def interrupted(signum, frame): raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
        try:
            run(COMPOSE + ["down"], timeout=60)
            run(COMPOSE + ["up", "-d", "--force-recreate", "--no-build"], timeout=90)
            network = json.loads(run(["docker", "network", "inspect", "cyberwm_v5_labnet_v5"], capture_output=True, text=True).stdout)[0]
            if not network["Internal"] or network["IPAM"]["Config"][0]["Subnet"] != "10.77.0.0/24":
                raise RuntimeError("network must be internal and on authorized CIDR")
            if {c["IPv4Address"].split('/')[0] for c in network["Containers"].values()} != set(HOSTS.values()):
                raise RuntimeError("network has missing or unexpected members")
            for host in HOSTS:
                for retry in range(30):
                    try:
                        exec_host(host, ["curl", "--noproxy", "*", "-fsS", "http://127.0.0.1:8080/"], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                        exec_host(host, ["bash", "-c", "echo >/dev/tcp/127.0.0.1/22"], stdout=sp.DEVNULL)
                        break
                    except sp.CalledProcessError:
                        if retry == 29: raise
                        time.sleep(.2)
                rules = exec_host(host, ["iptables", "-S", "INPUT"], capture_output=True, text=True).stdout.strip()
                if rules != "-P INPUT ACCEPT": raise RuntimeError(f"unexpected firewall on {host}: {rules}")
            order = roles(a.seed); actor, pivot, target, bg1, bg2 = order
            sched = schedule(a.seed)
            out.mkdir()
            truth = out / "ground_truth.csv"; actions = out / "defender_actions.csv"
            append(truth, TRUTH_HEADER); append(actions, ACTION_HEADER)
            (out / "schedule.json").write_text(json.dumps(sched, indent=2)+"\n")
            (out / "capture_provenance.json").write_text(json.dumps({"code_hashes": code_hashes(), "image_hashes": image_hashes()}, indent=2)+"\n")
            tcp_log = (out / "tcpdump.log").open("w"); log_handles.append(tcp_log)
            # Scope telemetry to inventory-to-inventory IPv4, not host multicast,
            # bridge gateway, or unrelated local-machine traffic.
            sources = " or ".join("src host " + ip for ip in HOSTS.values())
            destinations = " or ".join("dst host " + ip for ip in HOSTS.values())
            capture_filter = f"ip and ({sources}) and ({destinations})"
            capture = sp.Popen(["sudo", "-n", "tcpdump", "-U", "-i", "cyberwmv5", "-n", "-s", "0",
                                "-w", str(out / "network.pcap"), capture_filter], stderr=tcp_log, stdout=sp.DEVNULL)
            time.sleep(1)
            if capture.poll() is not None or "listening on cyberwmv5" not in (out/"tcpdump.log").read_text():
                raise RuntimeError("tcpdump failed readiness check")
            # Use an interior capture interval: readiness through stop request,
            # not process-launch/exit latency as if it were observed time.
            start_ns = time.time_ns(); deadline = start_ns + 150*10**9
            sleep_until(((time.time_ns() // (5*10**9))+1)*5*10**9 + 200_000_000)
            # All hosts can produce background; background identity does not
            # reveal actor/pivot roles. Seed and schedule do not depend on outcome.
            for host, ip in HOSTS.items():
                log = (out / f"background_{host}.jsonl").open("w"); log_handles.append(log)
                background.append(sp.Popen(COMPOSE + ["exec", "-T", host, "python3", "/opt/cyberwm/v5_background.py",
                    "--source", ip, "--profile", a.background, "--seed", str(a.seed), "--until", str(deadline/1e9-2)],
                    stdout=log, stderr=sp.STDOUT))
            def event(command, technique, name, tactic, destination):
                began = stamp(); exec_host(actor, command, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                append(truth, [a.episode_id, began, stamp(), actor, destination, technique, name, tactic])
            if a.scenario in ("scan_guess_then_stop", "one_hop", "scan_guess_action_permit", "scan_guess_action_block"):
                sleep_until(start_ns + sched["discovery_seconds"]*10**9)
                event(["timeout", "18", "nmap", "-sT", "-Pn", "--max-retries", "1", "-p", "22,80,443,445,3389,8080"]
                      + [ip for h, ip in HOSTS.items() if h != actor], "T1046", "Network Service Discovery", "Discovery", "internal_subnet")
                sleep_until(start_ns + sched["guessing_seconds"]*10**9)
                began = stamp()
                for i in range(sched["attempts"]):
                    result = sp.run(COMPOSE + ["exec", "-T", actor] + ssh_command(pivot, f"wrong{i}", "true"),
                                    stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=12)
                    if result.returncode != 5: raise RuntimeError(f"guess must fail authentication (sshpass 5), got {result.returncode}")
                append(truth, [a.episode_id, began, stamp(), actor, pivot, "T1110.001", "Password Guessing", "Credential Access"])
            sleep_until(start_ns + sched["decision_seconds"]*10**9)
            if time.time_ns() > start_ns + (sched["decision_seconds"]+1)*10**9:
                raise RuntimeError("prefix overran scheduled decision; reject episode")
            action_type = "permit_ssh" if a.scenario.endswith("_action_permit") else "block_ssh" if a.scenario.endswith("_action_block") else "none"
            decision = stamp()
            forecast_ns = ((time.time_ns() // (5*10**9))+1)*5*10**9
            # Action is chosen BEFORE cutoff; physical application occurs after.
            sleep_until(forecast_ns + 200_000_000)
            effective = stamp()
            if action_type == "block_ssh":
                rule = ["-p", "tcp", "-s", HOSTS[actor], "--dport", "22", "-j", "REJECT", "--reject-with", "tcp-reset"]
                block = (pivot, rule)  # record cleanup intent before invoking iptables
                exec_host(pivot, ["iptables", "-I", "INPUT", "1"]+rule)
            if action_type != "none":
                append(actions, [a.episode_id, decision, stamp(), action_type, actor, pivot, "true",
                                 "chosen_before_cutoff", stamp(forecast_ns), effective])
            if a.scenario not in ("background_only", "scan_guess_then_stop"):
                began = stamp()
                result = sp.run(COMPOSE + ["exec", "-T", actor]+ssh_command(pivot), stdout=sp.DEVNULL,
                                stderr=sp.DEVNULL, timeout=12)
                if result.returncode != (255 if action_type == "block_ssh" else 0):
                    raise RuntimeError(f"unexpected SSH result: {result.returncode}")
                if a.scenario not in ("legitimate_ssh", "matched_legitimate_ssh"):
                    append(truth, [a.episode_id, began, stamp(), actor, pivot, "T1021.004", "SSH",
                                   "Lateral Movement Attempt" if action_type == "block_ssh" else "Lateral Movement"])
            if time.time_ns() + 35*10**9 > deadline: raise RuntimeError("insufficient complete future margin")
            sleep_until(deadline)
            if capture.poll() is not None: raise RuntimeError("tcpdump exited during capture")
            end_ns = time.time_ns()
            run(["sudo", "-n", "kill", "-INT", str(capture.pid)])
            if capture.wait(timeout=10) != 0: raise RuntimeError("tcpdump unsuccessful exit")
            capture = None
            for worker in background:
                if worker.wait(timeout=8) != 0: raise RuntimeError("background worker failed")
            # Interior capture bounds end before stop request; metadata installed only
            # after clean workers, zero drops, and successful firewall cleanup.
            tcp_log.flush()
            if "0 packets dropped by kernel" not in (out/"tcpdump.log").read_text():
                raise RuntimeError("capture drop count missing/nonzero")
            if block:
                exec_host(block[0], ["iptables", "-D", "INPUT"]+block[1]); block = None
            for host in HOSTS:
                if exec_host(host, ["iptables", "-S", "INPUT"], capture_output=True, text=True).stdout.strip() != "-P INPUT ACCEPT":
                    raise RuntimeError("firewall cleanup failed")
            cleanup_ok = True
            meta = dict(episode_id=a.episode_id, scenario=a.scenario, seed=a.seed, capture_start=stamp(start_ns),
                capture_end=stamp(end_ns), actor=actor, pivot=pivot, target=target, background_host_1=bg1,
                background_host_2=bg2, background_profile=a.background, topology_profile="flat_five_host",
                node_count=5, window_seconds=5, planned_capture_duration_seconds=150, defender_action=action_type,
                runtime_version="v5_reviewed_1", forecast_time=stamp(forecast_ns))
        finally:
            try:
                if capture is not None and capture.poll() is None:
                    sp.run(["sudo", "-n", "kill", "-INT", str(capture.pid)], timeout=10, check=True)
                    capture.wait(timeout=10)
            finally:
                # Always kill container-side descendants, even if capture cleanup
                # raises. A disconnected docker exec client is not sufficient.
                try:
                    try:
                        if block: exec_host(block[0], ["iptables", "-D", "INPUT"]+block[1])
                    finally:
                        run(COMPOSE + ["stop", "-t", "2"], timeout=40)
                finally:
                    for worker in background:
                        if worker.poll() is None: worker.terminate(); worker.wait(timeout=5)
                    for handle in log_handles: handle.close()
                if not cleanup_ok: print("capture failed; preserve partial directory for quarantine", file=sys.stderr)
        if run(COMPOSE + ["ps", "--services", "--status", "running"], capture_output=True, text=True).stdout.strip():
            raise RuntimeError("containers still running after cleanup")
        (out/"cleanup.json").write_text(json.dumps({"firewall_clean": True, "workers_exited": True,
            "containers_stopped": True, "capture_drops": 0})+"\n")
        with (out/"episode_metadata.csv").open("x", newline="") as f:
            writer=csv.DictWriter(f, fieldnames=list(meta)); writer.writeheader(); writer.writerow(meta)
        print(f"captured {out} ({(end_ns-start_ns)/1e9:.3f}s); firewall clean and containers stopped", flush=True)
    return 0


if __name__ == "__main__": sys.exit(main())
