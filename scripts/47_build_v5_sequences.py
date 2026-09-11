#!/usr/bin/env python3
"""Export one explicitly selected V5 split/mode; test is opt-in, smoke is separate."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v5_contract import feature_metadata
from src.cyberwm.v5_sequences import episode_samples, require


def selected_rows(plan, split, mode, unlock_test=False):
    if split == "test" and not unlock_test:
        raise PermissionError("V5 test sealed; --unlock-test only after model/protocol freeze")
    rows = plan[plan.split.eq(split)]
    rows = rows[rows.defender_action.eq("none") if mode == "passive" else rows.defender_action.ne("none")]
    require(len(rows) > 0, "empty split/mode")
    require(not rows.episode_id.duplicated().any(), "duplicate selected episode")
    return rows


def export(plan, split, mode, episodes, destination, unlock_test=False):
    # Gate before inspecting episode contents; write a directory atomically.
    rows = selected_rows(plan, split, mode, unlock_test)
    if destination.exists(): raise FileExistsError(f"refusing overwrite: {destination}")
    if split != "smoke":
        require(not rows.episode_id.str.startswith("v5_smoke_").any(), "smoke entered model split")
    stores = {}; audit = []; hashes = {}
    for row in rows.to_dict("records"):
        ep = episodes / row["episode_id"]
        arrays, manifest = episode_samples(ep, mode, row)
        for key, value in arrays.items(): stores.setdefault(key, []).append(value)
        for item in manifest:
            item.update(sample_id=len(audit), split=split, cohort=row["cohort"], paired_family=row["paired_family"])
            audit.append(item)
        paths = [ep/"episode_metadata.csv", ep/"ground_truth.csv", ep/"defender_actions.csv", ep/"state_ground_truth.csv",
                 ep/"states/global_states.csv", ep/"states/node_states.csv.gz", ep/"states/edge_states.csv.gz"]
        hashes[row["episode_id"]] = {str(p.relative_to(ep)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = {k: np.concatenate(v) for k, v in stores.items()}
    require(all(np.isfinite(a).all() for a in result.values()), "nonfinite arrays")
    if mode == "action" and split != "smoke":
        require(result["action_type"].sum(0).tolist() == [len(rows)/2]*2, "unbalanced action alternatives")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".v5-export-", dir=destination.parent))
    try:
        np.savez_compressed(temp/f"{split}.npz", **result)
        pd.DataFrame(audit).to_csv(temp/"sample_manifest.csv", index=False)
        metadata = {**feature_metadata(), "split": split, "mode": mode, "episodes": len(rows),
                    "samples": len(audit), "source_hashes": hashes, "test_access": split == "test",
                    "test_policy": "no test episode content is read for train/validation/smoke exports"}
        (temp/"feature_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        os.rename(temp, destination)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    return {k: list(a.shape) for k, a in result.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "validation", "test", "smoke"], default="train")
    p.add_argument("--mode", choices=["passive", "action", "passive_action"], default="passive")
    p.add_argument("--unlock-test", action="store_true")
    p.add_argument("--episodes-dir", type=Path, default=ROOT/"lab/episodes")
    p.add_argument("--out-dir", type=Path, default=ROOT/"outputs/mvp_v5/sequences")
    a = p.parse_args()
    if a.split == "test" and not a.unlock_test: raise PermissionError("V5 test sealed")
    if a.split != "smoke":
        subprocess.run([sys.executable, str(ROOT/"scripts/46_validate_v5_plan.py")], check=True)
    plan_path = ROOT/"configs"/("mvp_v5_smoke_plan.csv" if a.split == "smoke" else "mvp_v5_episode_plan.csv")
    plan = pd.read_csv(plan_path, dtype=str)
    dest = a.out_dir / a.mode / a.split
    shapes = export(plan, a.split, a.mode, a.episodes_dir, dest, a.unlock_test)
    print(json.dumps(shapes, indent=2)); print(f"V5 export -> {dest}")
    return 0


if __name__ == "__main__": sys.exit(main())
