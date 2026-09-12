#!/usr/bin/env python3
"""Non-capture V6 auth-log feasibility smoke on the isolated Docker inventory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess as sp
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v6_auth import parse_auth_logs, validate_auth_event_schema

COMPOSE = ["docker", "compose", "-p", "cyberwm_v6", "-f", str(ROOT / "lab/docker-compose-v6.yml")]
V5_COMPOSE = ["docker", "compose", "-p", "cyberwm_v5", "-f", str(ROOT / "lab/docker-compose-v5.yml")]
SERVICES = {"ws1", "ws2", "ws3", "srv1", "srv2", "admin1", "jump1"}
OUT = ROOT / "outputs/mvp_v6/review/auth_feasibility.json"


def run(command: list[str], **kwargs) -> sp.CompletedProcess:
    return sp.run(command, check=True, timeout=kwargs.pop("timeout", 45), **kwargs)


def digest(path: Path) -> str:
    value = hashlib.sha256(); value.update(path.read_bytes()); return value.hexdigest()


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def exec_host(host: str, command: list[str], check: bool = True) -> sp.CompletedProcess:
    if host not in SERVICES:
        raise ValueError("host outside V6 inventory")
    return sp.run(COMPOSE + ["exec", "-T", host] + command, check=check, timeout=20,
                  stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "outputs/mvp_v6/review").resolve()):
        parser.error("output must remain under outputs/mvp_v6/review")
    if out.exists():
        raise FileExistsError(f"preserve existing feasibility result: {out}")
    for compose in (V5_COMPOSE, COMPOSE):
        running = run(compose + ["ps", "--services", "--status", "running"],
                      capture_output=True, text=True).stdout.strip()
        if running:
            raise RuntimeError(f"containers already running: {running}")
    if run(["docker", "network", "ls", "--format", "{{.Name}}"], capture_output=True,
           text=True).stdout.splitlines().__contains__("cyberwm_v5_labnet_v5"):
        raise RuntimeError("empty V5 network occupies authorized subnet; remove it explicitly before smoke")
    temp = Path(tempfile.mkdtemp(prefix="v6-auth-smoke-"))
    started = stamp(); success = False
    try:
        run(COMPOSE + ["up", "-d", "--force-recreate", "--no-build"], timeout=90)
        network = json.loads(run(["docker", "network", "inspect", "cyberwm_v6_labnet_v6"],
                                 capture_output=True, text=True).stdout)[0]
        if not network["Internal"] or network["IPAM"]["Config"][0]["Subnet"] != "10.77.0.0/24":
            raise RuntimeError("V6 network is not the authorized internal subnet")
        if len(network["Containers"]) != 7:
            raise RuntimeError("V6 network inventory differs")
        for attempt in range(30):
            if exec_host("ws1", ["bash", "-c", "echo >/dev/tcp/127.0.0.1/22"], check=False).returncode == 0:
                break
            if attempt == 29:
                raise RuntimeError("V6 sshd readiness failed")
            time.sleep(.2)
        ids = {}
        for service in SERVICES:
            container = run(COMPOSE + ["ps", "-q", service], capture_output=True, text=True).stdout.strip()
            ids[service] = run(["docker", "inspect", "--format", "{{.Image}}", container],
                               capture_output=True, text=True).stdout.strip()
        if len(set(ids.values())) != 1:
            raise RuntimeError(f"services do not share one image digest: {ids}")
        wrong = exec_host("ws1", ["sshpass", "-p", "wrong", "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "PubkeyAuthentication=no",
            "-o", "PreferredAuthentications=password", "-o", "ConnectTimeout=2",
            "lab@10.77.0.30", "true"], check=False)
        if wrong.returncode != 5:
            raise RuntimeError(f"expected password authentication rejection5, got {wrong.returncode}")
        wrong_key = exec_host("ws1", ["ssh", "-i", "/etc/ssh/ssh_host_ed25519_key",
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=2",
            "lab@10.77.0.30", "true"], check=False)
        if wrong_key.returncode != 255:
            raise RuntimeError(f"expected public-key authentication rejection255, got {wrong_key.returncode}")
        exec_host("jump1", ["ssh", "-i", "/home/lab/.ssh/id_ed25519", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=2", "lab@10.77.0.40", "true"])
        ended = stamp()
        parsed = {}
        for service in ("srv1", "srv2"):
            raw = run(COMPOSE + ["logs", "--no-color", "--no-log-prefix", "--timestamps", service],
                      capture_output=True, text=True).stdout
            (temp / f"{service}.raw.log").write_text(raw)
            parsed[service] = parse_auth_logs(service, raw.splitlines(), started, ended)
            validate_auth_event_schema(parsed[service])
        srv1_outcomes = {(event["event_type"], event["auth_method"]) for event in parsed["srv1"]}
        if len(parsed["srv1"]) != 2 or srv1_outcomes != {
                ("auth_failure", "password"), ("auth_failure", "publickey")}:
            raise RuntimeError(f"actual authentication failures differ: {parsed['srv1']}")
        if len(parsed["srv2"]) != 1 or parsed["srv2"][0]["event_type"] != "auth_success" \
                or parsed["srv2"][0]["auth_method"] != "publickey":
            raise RuntimeError("actual public-key success parse differs")
        success = True
    finally:
        sp.run(COMPOSE + ["down", "-t", "2"], timeout=60, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    if not success:
        raise RuntimeError(f"V6 auth feasibility failed; raw logs retained at {temp}")
    if run(COMPOSE + ["ps", "--services", "--status", "running"], capture_output=True,
           text=True).stdout.strip():
        raise RuntimeError("V6 containers remain running")
    result = {"status": "PASS_NON_CAPTURE_AUTH_FEASIBILITY", "scope":
        "isolated internal password/public-key rejection, public-key acceptance, and observable parsing; no PCAP/episode",
        "started_utc": started, "completed_utc": ended, "network_internal": True,
        "subnet": "10.77.0.0/24", "containers": 7, "shared_image_id": next(iter(ids.values())),
        "password_rejection_returncode": wrong.returncode,
        "publickey_rejection_returncode": wrong_key.returncode,
        "events": parsed, "raw_logs_retained": False,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
            ROOT / "lab/Dockerfile.v6", ROOT / "lab/docker-compose-v6.yml",
            ROOT / "src/cyberwm/v6_auth.py", Path(__file__).resolve())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n"); temporary.replace(out)
    for path in temp.iterdir(): path.unlink()
    temp.rmdir()
    print(json.dumps({"status": result["status"], "out": str(out),
                      "shared_image_id": result["shared_image_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
