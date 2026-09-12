#!/usr/bin/env python3
"""Fit the train-context-only V5 scaler and freeze model-training sources."""
from __future__ import annotations
import argparse
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
from src.cyberwm.v5_model_protocol import (assert_test_sealed, file_sha256, fit_v5_scaler,
                                           load_protocol, save_scaler)

SOURCES = (
    "configs/mvp_v5_training_protocol.json",
    "src/cyberwm/graph_rssm.py", "src/cyberwm/action_graph_rssm.py",
    "src/cyberwm/branching_graph_rssm.py", "src/cyberwm/branch_metrics.py",
    "src/cyberwm/device.py", "src/cyberwm/v5_model_protocol.py",
    "scripts/15_train_rssm.py", "scripts/34_train_branching_graph_rssm.py",
    "scripts/43_train_action_graph_rssm_v4.py", "scripts/51_audit_v5_exports.py",
    "scripts/52_freeze_v5_training_protocol.py", "scripts/53_train_v5_action.py",
    "scripts/54_train_v5_passive_branch.py", "scripts/55_test_v5_training_protocol.py",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--freeze-out", type=Path, default=ROOT / "configs/mvp_v5_training_freeze.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mvp_v5/model_protocol")
    args = ap.parse_args(); assert_test_sealed(); protocol = load_protocol()
    audit_path = ROOT / "outputs/mvp_v5/export_audit/audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "PASS" or not audit.get("v5_test_exports_absent"):
        raise ValueError("V5 export audit not passing/sealed")
    for relative, expected in audit["artifact_sha256"].items():
        actual = file_sha256(ROOT / "outputs/mvp_v5/sequences" / relative)
        if actual != expected: raise ValueError(f"export changed after audit: {relative}")
    scaler, keys = fit_v5_scaler()
    expected_rows = protocol["scaler"]["expected_unique_state_rows"]
    if len(keys) != expected_rows or len(set(keys)) != expected_rows: raise ValueError("scaler row deduplication failed")
    mean, scale = scaler.mean_, scaler.scale_
    if mean.shape != (345,) or scale.shape != (345,) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("invalid scaler moments")
    node_mean = mean[15:33]; edge_mean = mean[105:117]
    for slot in range(5):
        np.testing.assert_array_equal(mean[15 + slot*18:15 + (slot+1)*18], node_mean)
        np.testing.assert_array_equal(scale[15 + slot*18:15 + (slot+1)*18], scale[15:33])
    for slot in range(20):
        np.testing.assert_array_equal(mean[105 + slot*12:105 + (slot+1)*12], edge_mean)
        np.testing.assert_array_equal(scale[105 + slot*12:105 + (slot+1)*12], scale[105:117])
    source_hashes = {path: file_sha256(ROOT / path) for path in SOURCES}
    result = {"status": "CHECK_PASS", "protocol_id": protocol["protocol_id"],
              "unique_train_context_rows": len(keys), "state_width": len(mean),
              "test_exports_absent": True, "shared_node_edge_statistics": True,
              "source_sha256": source_hashes}
    if args.check_only:
        print(json.dumps(result, indent=2)); return 0
    if args.freeze_out.exists() or args.out_dir.exists(): raise FileExistsError("V5 training freeze already exists")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, check=True).stdout.strip()
    if status: raise RuntimeError("commit protocol/training sources before freeze; git tree is not clean")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, check=True).stdout.strip()
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v5-model-protocol-", dir=args.out_dir.parent))
    try:
        scaler_path = temporary / "shared_train_context_scaler.npz"; save_scaler(scaler_path, scaler, keys)
        pd.DataFrame(keys, columns=["episode_id", "state_index"]).to_csv(temporary / "scaler_source_rows.csv", index=False)
        scaler_hash = file_sha256(scaler_path)
        freeze = {**result, "status": "FROZEN_TRAIN_VALIDATION_ONLY", "git_commit": commit,
                  "protocol_sha256": source_hashes["configs/mvp_v5_training_protocol.json"],
                  "export_audit_sha256": file_sha256(audit_path), "export_artifact_sha256": audit["artifact_sha256"],
                  "scaler_sha256": scaler_hash,
                  "initialization_checkpoint_sha256": {name: entry["sha256"] for name, entry in protocol["initializations"].items()
                                                       if entry["source"] is not None},
                  "test_unlock": False,
                  "next_gate": "train/validate frozen candidates, then freeze selected checkpoint hashes and evaluator before one test run"}
        (temporary / "training_freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
        os.rename(temporary, args.out_dir)
        args.freeze_out.write_text(json.dumps(freeze, indent=2) + "\n")
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    print(json.dumps({key: freeze[key] for key in ["status", "git_commit", "protocol_sha256",
                                                    "scaler_sha256", "test_unlock", "next_gate"]}, indent=2))
    print(f"V5 training protocol -> {args.freeze_out}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
