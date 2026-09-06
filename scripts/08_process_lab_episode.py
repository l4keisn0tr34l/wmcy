#!/usr/bin/env python3
"""Build and validate all derived files for one controlled lab episode.

Raw network.pcap, ground_truth.csv, and episode_metadata.csv are read-only. Derived
outputs are built in a temporary directory, validated there, and installed only
after the full pipeline passes.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_metadata(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one metadata row")
    required = {"episode_id", "capture_start", "capture_end", "window_seconds"}
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    return rows[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--internal-cidr", default="10.77.0.0/24")
    ap.add_argument("--window-seconds", type=int, default=5, help="used only without metadata")
    ap.add_argument("--force", action="store_true", help="replace existing derived outputs after validation")
    args = ap.parse_args()

    episode = Path(args.episode_dir).resolve()
    raw_pcap = episode / "network.pcap"
    ground_truth = episode / "ground_truth.csv"
    metadata_path = episode / "episode_metadata.csv"
    invalid_marker = episode / "INVALID_EPISODE.txt"
    if invalid_marker.exists():
        raise SystemExit(f"refusing to process quarantined episode: {invalid_marker}")
    for path in [raw_pcap, ground_truth]:
        if not path.is_file():
            raise SystemExit(f"missing raw episode file: {path}")

    metadata = read_metadata(metadata_path) if metadata_path.is_file() else None
    episode_id = metadata["episode_id"] if metadata else episode.name
    window_seconds = int(metadata["window_seconds"]) if metadata else args.window_seconds
    if window_seconds <= 0:
        raise SystemExit("window_seconds must be positive")

    destinations = [
        episode / "observations.csv.gz",
        episode / "states",
        episode / "state_ground_truth.csv",
    ]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise SystemExit(f"derived outputs already exist ({joined}); use --force to rebuild safely")

    with tempfile.TemporaryDirectory(prefix=".processing-", dir=episode) as temporary:
        work = Path(temporary)
        temp_observations = work / "observations.csv.gz"
        temp_states = work / "states"
        temp_aligned = work / "state_ground_truth.csv"
        shutil.copy2(ground_truth, work / "ground_truth.csv")
        if metadata_path.is_file():
            shutil.copy2(metadata_path, work / "episode_metadata.csv")

        run([
            sys.executable,
            str(SCRIPTS / "06_pcap_to_canonical.py"),
            str(raw_pcap),
            "--bucket-seconds", "1",
            "--dataset-id", episode_id,
            "--out", str(temp_observations),
        ])
        state_command = [
            sys.executable,
            str(SCRIPTS / "03_build_graph_states.py"),
            str(temp_observations),
            "--window", f"{window_seconds}s",
            "--internal-cidr", args.internal_cidr,
            "--out-dir", str(temp_states),
        ]
        if metadata:
            state_command.extend([
                "--capture-start", metadata["capture_start"],
                "--capture-end", metadata["capture_end"],
            ])
        run(state_command)
        run([
            sys.executable,
            str(SCRIPTS / "05_align_ground_truth.py"),
            str(temp_states / "global_states.csv"),
            str(work / "ground_truth.csv"),
            "--window-seconds", str(window_seconds),
            "--out", str(temp_aligned),
        ])
        run([
            sys.executable,
            str(SCRIPTS / "07_validate_episode.py"),
            str(work),
            "--window-seconds", str(window_seconds),
        ])

        # Install only validated derived outputs. Raw episode files are never moved.
        old_observations_csv = episode / "observations.csv"
        if destinations[0].exists():
            destinations[0].unlink()
        os.replace(temp_observations, destinations[0])
        if destinations[1].exists():
            shutil.rmtree(destinations[1])
        os.replace(temp_states, destinations[1])
        if destinations[2].exists():
            destinations[2].unlink()
        os.replace(temp_aligned, destinations[2])
        if old_observations_csv.exists():
            old_observations_csv.unlink()

    run([
        sys.executable,
        str(SCRIPTS / "07_validate_episode.py"),
        str(episode),
        "--window-seconds", str(window_seconds),
    ])
    print(f"processed episode successfully: {episode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
