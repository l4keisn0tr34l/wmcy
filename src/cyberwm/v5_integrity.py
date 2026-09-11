"""Capture implementation/image provenance, separate from model inputs."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "configs/mvp_v5_episode_plan.csv", "configs/mvp_v5_split_assignments.csv", "configs/mvp_v5_smoke_plan.csv",
    "lab/Dockerfile", "lab/docker-compose-v5.yml", "lab/run_v5_episode.sh", "lab/v5_runtime.py",
    "lab/v5_background.py", "lab/generate_v5_corpus.sh", "lab/smoke_v5.sh",
    "src/cyberwm/v5_contract.py", "src/cyberwm/v5_sequences.py", "src/cyberwm/v5_integrity.py",
    "src/cyberwm/common.py", "scripts/03_build_graph_states.py", "scripts/05_align_ground_truth.py",
    "scripts/06_pcap_to_canonical.py", "scripts/07_validate_episode.py", "scripts/08_process_lab_episode.py",
    "scripts/46_validate_v5_plan.py", "scripts/47_build_v5_sequences.py", "scripts/48_test_v5_contract.py",
    "scripts/49_freeze_v5_capture.py",
]


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def code_hashes(): return {p: sha(ROOT/p) for p in FILES}


def image_hashes():
    names = [f"cyberwm_v5-{h}" for h in ["ws1", "ws2", "srv1", "srv2", "admin1"]]
    data = json.loads(subprocess.run(["docker", "image", "inspect", *names], check=True,
                                    capture_output=True, text=True, timeout=20).stdout)
    return {name: item["Id"] for name, item in zip(names, data)}
