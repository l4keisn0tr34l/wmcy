#!/usr/bin/env python3
"""Validate capture quality, then explicitly freeze V5 only after two real smokes."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_contract import HOSTS, capture_order
from src.cyberwm.v5_integrity import code_hashes, image_hashes, sha
from src.cyberwm.v5_sequences import episode_samples, require, times
FREEZE = ROOT/"configs/mvp_v5_capture_freeze.json"


def verify_episode(episode, expected, current_images=None):
    mode = "passive" if expected["defender_action"] == "none" else "action"
    arrays, audit = episode_samples(episode, mode, expected)
    require((episode/"network.pcap").stat().st_size > 24, "empty PCAP")
    provenance = json.loads((episode/"capture_provenance.json").read_text())
    require(provenance["code_hashes"] == code_hashes(), "capture code differs from reviewed source; do not retrofit provenance")
    if current_images is not None:
        require(provenance["image_hashes"] == current_images, "capture image drift")
    receipt = json.loads((episode/"cleanup.json").read_text())
    require(receipt == {"firewall_clean": True, "workers_exited": True, "containers_stopped": True, "capture_drops": 0}, "incomplete cleanup")
    require("0 packets dropped by kernel" in (episode/"tcpdump.log").read_text(), "PCAP drops")
    meta = pd.read_csv(episode/"episode_metadata.csv", dtype=str).iloc[0]
    start, end = times(pd.Series([meta.capture_start, meta.capture_end]))
    kinds = set(); successful_kinds = set(); totals = {}
    for host, ip in HOSTS.items():
        rows = [json.loads(line) for line in (episode/f"background_{host}.jsonl").read_text().splitlines()]
        require(len(rows) >= 5, f"missing sustained background on {host}")
        require(all(r["source"] == ip and r["destination"] in HOSTS.values() and r["destination"] != ip for r in rows), "background escapes inventory")
        require(all(start.timestamp() <= r["start_epoch"] < end.timestamp() for r in rows), "background outside capture")
        require(max(r["start_epoch"] for r in rows) > end.timestamp()-22, "premature background quiet tail")
        require(any(r["returncode"] == 0 for r in rows), "all background commands failed")
        kinds.update(r["kind"] for r in rows)
        successful_kinds.update(r["kind"] for r in rows if r["returncode"] == 0)
        totals[host] = len(rows)
    desired = {"quiet", "web", "admin"} if meta.background_profile == "mixed" else {meta.background_profile}
    require(kinds == desired and successful_kinds == desired, "background profile kinds missing/never successful")
    truth = pd.read_csv(episode/"ground_truth.csv")
    scenario = expected["scenario"]
    prefix = scenario in {"scan_guess_then_stop", "one_hop", "scan_guess_action_permit", "scan_guess_action_block"}
    remote = scenario in {"one_hop", "credential_one_hop", "scan_guess_action_permit", "scan_guess_action_block",
                          "credential_action_permit", "credential_action_block"}
    expected_techniques = (["T1046", "T1110.001"] if prefix else []) + (["T1021.004"] if remote else [])
    require(truth.technique_id.tolist() == expected_techniques, "scenario truth events differ from executed-action contract")
    expected_completed = remote and expected["defender_action"] != "block_ssh"
    require(int(truth.tactic.eq("Lateral Movement").sum()) == int(expected_completed), "scenario completion truth mismatch")
    return {"episode_id": meta.episode_id, "mode": mode, "samples": len(audit),
            "context_shape": list(arrays["context_states"].shape), "background_events_by_host": totals}


def check_freeze():
    if not FREEZE.is_file(): raise PermissionError("V5 NOT FROZEN: two validated smokes and explicit --freeze required")
    record = json.loads(FREEZE.read_text())
    require(record["code_hashes"] == code_hashes(), "reviewed V5 code/plan changed after freeze")
    require(record["image_hashes"] == image_hashes(), "Docker images changed after freeze")
    for name, digest in record["smoke_hashes"].items(): require(sha(ROOT/name) == digest, f"smoke modified: {name}")
    print("V5 capture freeze PASS (not a model/evaluation freeze)")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--episode", type=Path, help="capture QA only, not model evaluation")
    a = p.parse_args()
    if a.check: check_freeze(); return 0
    smoke_plan = pd.read_csv(ROOT/"configs/mvp_v5_smoke_plan.csv", dtype=str)
    if a.episode:
        plan = smoke_plan if a.episode.name.startswith("v5_smoke_") else pd.read_csv(ROOT/"configs/mvp_v5_episode_plan.csv", dtype=str)
        selected = plan[plan.episode_id.eq(a.episode.name)]
        require(len(selected) == 1, "episode absent from plan")
        print(json.dumps(verify_episode(a.episode, selected.iloc[0].to_dict()), indent=2)); return 0
    if FREEZE.exists(): raise FileExistsError("freeze already exists; never overwrite")
    # Missing smoke outputs fail before any expensive checks or state changes.
    for name in smoke_plan.episode_id:
        require((ROOT/"lab/episodes"/name/"episode_metadata.csv").is_file(), f"missing real smoke: {name}")
    subprocess.run([sys.executable, str(ROOT/"scripts/46_validate_v5_plan.py")], check=True)
    subprocess.run([sys.executable, str(ROOT/"scripts/48_test_v5_contract.py")], check=True)
    images = image_hashes(); summary=[]; smoke_hashes={}
    for row in smoke_plan.to_dict("records"):
        episode = ROOT/"lab/episodes"/row["episode_id"]
        subprocess.run([sys.executable, str(ROOT/"scripts/07_validate_episode.py"), str(episode), "--window-seconds", "5"], check=True)
        summary.append(verify_episode(episode, row, images))
        for path in sorted(episode.rglob("*")):
            if path.is_file(): smoke_hashes[str(path.relative_to(ROOT))] = sha(path)
    plan = pd.read_csv(ROOT/"configs/mvp_v5_episode_plan.csv", dtype=str)
    record = {"status": "capture_only_frozen", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "code_hashes": code_hashes(), "image_hashes": images, "smoke_hashes": smoke_hashes,
              "smoke_validation": summary, "capture_order": sorted(plan.episode_id, key=capture_order),
              "model_protocol": "NOT FROZEN; train/validation only until separately frozen"}
    with FREEZE.open("x") as f: json.dump(record, f, indent=2); f.write("\n")
    print(f"capture freeze written: {FREEZE}; full corpus NOT started")
    return 0


if __name__ == "__main__": sys.exit(main())
