#!/usr/bin/env python3
"""Validate V4 action records and outcome semantics after episode processing."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACTION_COLUMNS = ["episode_id", "start_time", "end_time", "action", "source", "target",
                  "known_at_forecast_time", "details"]


def parse_time(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    for column in columns:
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if parsed.isna().any(): raise ValueError(f"{path}: invalid {column}")
        frame[column] = parsed


def validate_episode(row: object, episode: Path, python: str) -> None:
    subprocess.run([python, str(ROOT / "scripts/07_validate_episode.py"), str(episode)], check=True,
                   stdout=subprocess.DEVNULL)
    metadata_path = episode / "episode_metadata.csv"; truth_path = episode / "ground_truth.csv"
    action_path = episode / "defender_actions.csv"
    if not action_path.is_file(): raise ValueError(f"{episode.name}: missing defender_actions.csv")
    metadata = pd.read_csv(metadata_path); truth = pd.read_csv(truth_path); actions = pd.read_csv(action_path)
    if len(metadata) != 1: raise ValueError(f"{episode.name}: metadata row count")
    for key in ["episode_id", "scenario", "seed", "defender_action"]:
        if str(metadata.iloc[0][key]) != str(getattr(row, key)):
            raise ValueError(f"{episode.name}: metadata {key} mismatch")
    if list(actions.columns) != ACTION_COLUMNS: raise ValueError(f"{action_path}: wrong schema")
    parse_time(metadata, ["capture_start", "capture_end"], metadata_path)
    parse_time(truth, ["start_time", "end_time"], truth_path)
    if len(actions): parse_time(actions, ["start_time", "end_time"], action_path)
    scenario = row.scenario; completed_lm = truth.tactic.astype(str).eq("Lateral Movement")
    attempted_lm = truth.tactic.astype(str).eq("Lateral Movement Attempt")
    if row.defender_action == "none":
        if len(actions): raise ValueError(f"{episode.name}: unexpected action rows")
    else:
        if len(actions) != 1 or actions.action.iloc[0] != row.defender_action:
            raise ValueError(f"{episode.name}: expected one {row.defender_action}")
        action = actions.iloc[0]
        if str(action.known_at_forecast_time).lower() != "true":
            raise ValueError(f"{episode.name}: action not known at forecast")
        if action.source != metadata.actor.iloc[0] or action.target != metadata.pivot.iloc[0]:
            raise ValueError(f"{episode.name}: action source/target differs from roles")
        if not (metadata.capture_start.iloc[0] <= action.start_time <= action.end_time <= metadata.capture_end.iloc[0]):
            raise ValueError(f"{episode.name}: action outside capture")
        guessing = truth[truth.technique_id.astype(str).eq("T1110.001")]
        remote = truth[truth.technique_id.astype(str).eq("T1021.004")]
        if len(guessing) != 1 or len(remote) != 1:
            raise ValueError(f"{episode.name}: action scenario needs one guess and remote event")
        if guessing.end_time.iloc[0] > action.start_time or action.end_time > remote.start_time.iloc[0]:
            raise ValueError(f"{episode.name}: action is not causally between guess and SSH")
    expected_completed = scenario in {"one_hop", "credential_one_hop", "scan_guess_action_permit"}
    expected_attempt = scenario == "scan_guess_action_block"
    if int(completed_lm.sum()) != int(expected_completed):
        raise ValueError(f"{episode.name}: completed LM truth mismatch")
    if int(attempted_lm.sum()) != int(expected_attempt):
        raise ValueError(f"{episode.name}: attempted LM truth mismatch")
    if scenario == "matched_legitimate_ssh" and len(truth):
        raise ValueError(f"{episode.name}: matched legitimate SSH must have no attack truth")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(ROOT / "configs/mvp_v4_episode_plan.csv"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args(); plan = pd.read_csv(args.plan, dtype={"episode_id": str, "seed": str})
    episodes_dir = Path(args.episodes_dir); checked = 0; missing = []
    for row in plan.itertuples(index=False):
        episode = episodes_dir / row.episode_id
        if not episode.is_dir(): missing.append(row.episode_id); continue
        validate_episode(row, episode, sys.executable); checked += 1
    if missing and not args.allow_missing: raise ValueError(f"missing {len(missing)} V4 episodes: {missing[:5]}")
    print(f"V4 action semantics PASS: checked={checked}, missing={len(missing)}")
    return 0


if __name__ == "__main__": sys.exit(main())
