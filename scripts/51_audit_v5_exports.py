#!/usr/bin/env python3
"""Audit immutable V5 train/validation exports without reading test episodes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODES = ("passive", "action", "passive_action")
SPLITS = ("train", "validation")
COMMON = ("context_states", "future_states", "future_edge_presence", "future_lateral_movement",
          "future_techniques", "future_lateral_edges", "lateral_movement_within_horizon")
FORBIDDEN_OBSERVABLE_TERMS = ("label", "attack", "technique", "tactic", "lateral", "actor", "pivot",
                              "target", "scenario", "seed", "cohort", "background_profile")
EXPECTED_SAMPLES = {
    ("passive", "train"): 336, ("passive", "validation"): 168,
    ("action", "train"): 24, ("action", "validation"): 8,
    ("passive_action", "train"): 24, ("passive_action", "validation"): 8,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences-dir", type=Path, default=ROOT / "outputs/mvp_v5/sequences")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mvp_v5/export_audit")
    args = ap.parse_args()
    if args.out_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    require(not any((args.sequences_dir / mode / "test").exists() for mode in MODES),
            "V5 test export exists; train/validation protocol seal violated")
    plan = pd.read_csv(ROOT / "configs/mvp_v5_episode_plan.csv", dtype=str)
    expected_ids = {split: set(plan.loc[plan.split.eq(split), "episode_id"]) for split in SPLITS}
    require(expected_ids["train"].isdisjoint(expected_ids["validation"]), "plan split overlap")
    data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    schemas: list[list[str]] = []; artifact_hashes: dict[str, str] = {}; summaries = []
    seen_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for mode in MODES:
        for split in SPLITS:
            directory = args.sequences_dir / mode / split
            npz_path = directory / f"{split}.npz"
            metadata_path = directory / "feature_metadata.json"
            manifest_path = directory / "sample_manifest.csv"
            require(all(path.is_file() for path in (npz_path, metadata_path, manifest_path)),
                    f"missing export files: {mode}/{split}")
            arrays = load_npz(npz_path); metadata = json.loads(metadata_path.read_text())
            manifest = pd.read_csv(manifest_path, dtype=str)
            count = EXPECTED_SAMPLES[(mode, split)]
            require(len(manifest) == count, f"{mode}/{split}: manifest count")
            require(set(manifest.split) == {split}, f"{mode}/{split}: split marker")
            require(set(manifest.episode_id).issubset(expected_ids[split]), f"{mode}/{split}: foreign episode")
            require(all(value.shape[0] == count and np.isfinite(value).all() for value in arrays.values()),
                    f"{mode}/{split}: nonfinite/count mismatch")
            require(arrays["context_states"].shape == (count, 3, 345), f"{mode}/{split}: context shape")
            require(arrays["future_states"].shape == (count, 6, 345), f"{mode}/{split}: future shape")
            require(arrays["future_edge_presence"].shape == (count, 6, 20), f"{mode}/{split}: edge shape")
            require(metadata["normalization"] == "none here; fit shared-slot scaler on train contexts only",
                    f"{mode}/{split}: normalization policy")
            require(metadata["test_access"] is False, f"{mode}/{split}: test access marker")
            require(set(metadata["source_hashes"]).issubset(expected_ids[split]), f"{mode}/{split}: source hash split")
            names = metadata["state_feature_names"]; schemas.append(names)
            contaminated = [name for name in names if any(term in name.lower() for term in FORBIDDEN_OBSERVABLE_TERMS)]
            require(not contaminated, f"{mode}/{split}: forbidden observable names {contaminated}")
            expected_keys = set(COMMON) | ({"action_type", "action_pair"} if mode == "action" else set())
            require(set(arrays) == expected_keys, f"{mode}/{split}: array keys {sorted(arrays)}")
            if mode == "action":
                require(arrays["action_type"].shape == (count, 2) and arrays["action_pair"].shape == (count, 20),
                        f"{mode}/{split}: action shape")
                require(np.all(arrays["action_type"].sum(axis=1) == 1) and
                        np.all(arrays["action_pair"].sum(axis=1) == 1), f"{mode}/{split}: action one-hot")
                require(np.all(arrays["action_type"].sum(axis=0) == count // 2), f"{mode}/{split}: action imbalance")
                require(arrays["lateral_movement_within_horizon"].sum() == count // 2,
                        f"{mode}/{split}: LM imbalance")
            if mode == "passive":
                per_episode = manifest.groupby("episode_id").size()
                require(per_episode.eq(21).all(), f"{mode}/{split}: expected 21 windows per episode")
            seen_by_split[split].update(manifest.episode_id)
            data[(mode, split)] = arrays
            rel = f"{mode}/{split}"
            for path in (npz_path, metadata_path, manifest_path):
                artifact_hashes[f"{rel}/{path.name}"] = sha256(path)
            summaries.append({"mode": mode, "split": split, "samples": count,
                              "episodes": manifest.episode_id.nunique(),
                              "lm_positive_samples": int(arrays["lateral_movement_within_horizon"].sum()),
                              "artifact_npz_sha256": artifact_hashes[f"{rel}/{npz_path.name}"]})
    require(all(schema == schemas[0] for schema in schemas[1:]), "feature schema differs across exports")
    for split in SPLITS:
        action = data[("action", split)]; aligned = data[("passive_action", split)]
        for name in COMMON:
            require(np.array_equal(action[name], aligned[name]), f"{split}: action/passive_action {name} differs")
        action_manifest = pd.read_csv(args.sequences_dir / "action" / split / "sample_manifest.csv", dtype=str)
        aligned_manifest = pd.read_csv(args.sequences_dir / "passive_action" / split / "sample_manifest.csv", dtype=str)
        require(action_manifest.equals(aligned_manifest), f"{split}: aligned manifests differ")
    train_families = set(plan.loc[plan.split.eq("train"), "paired_family"])
    validation_families = set(plan.loc[plan.split.eq("validation"), "paired_family"])
    test_families = set(plan.loc[plan.split.eq("test"), "paired_family"])
    require(train_families.isdisjoint(validation_families | test_families) and
            validation_families.isdisjoint(test_families), "paired-family split leakage")
    report = {
        "status": "PASS", "scope": "V5 train/validation export integrity; no test episode read",
        "state_feature_count": 345, "node_count": 5, "directed_pair_count": 20,
        "summaries": summaries, "common_action_aligned_arrays_exactly_equal": True,
        "feature_schemas_identical": True, "observable_feature_name_leakage": False,
        "whole_family_splits_disjoint": True, "v5_test_exports_absent": True,
        "artifact_sha256": artifact_hashes,
        "note": "scenario/cohort/family columns occur only in audit manifests, never NPZ model inputs",
    }
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v5-export-audit-", dir=args.out_dir.parent))
    try:
        (temporary / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        pd.DataFrame(summaries).to_csv(temporary / "summary.csv", index=False)
        os.rename(temporary, args.out_dir)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    print(json.dumps({key: report[key] for key in ["status", "summaries",
          "common_action_aligned_arrays_exactly_equal", "whole_family_splits_disjoint",
          "v5_test_exports_absent"]}, indent=2))
    print(f"V5 export audit -> {args.out_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
