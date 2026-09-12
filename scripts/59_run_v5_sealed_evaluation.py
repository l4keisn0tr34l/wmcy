#!/usr/bin/env python3
"""Atomically export and evaluate V5 test once under the evaluation freeze."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cyberwm.v5_model_protocol import file_sha256


def main()->int:
    freeze_path=ROOT/"configs/mvp_v5_evaluation_freeze.json"
    if not freeze_path.is_file():raise PermissionError("V5 evaluation freeze has not been created")
    freeze=json.loads(freeze_path.read_text())
    if freeze.get("status")!="FROZEN_FOR_ONE_SEALED_TEST" or not freeze.get("test_unlock"):
        raise PermissionError("V5 sealed-test evaluation is not frozen/unlocked")
    relative=str(Path(__file__).resolve().relative_to(ROOT))
    if freeze["source_sha256"].get(relative)!=file_sha256(Path(__file__).resolve()):
        raise ValueError("sealed wrapper changed after evaluation freeze")
    final=ROOT/"outputs/mvp_v5/sealed_test"
    if final.exists():raise FileExistsError("sealed V5 result already exists; rerun forbidden")
    if any((ROOT/"outputs/mvp_v5/sequences"/mode/"test").exists() for mode in ["passive","action","passive_action"]):
        raise PermissionError("unexpected legacy V5 test export exists")
    final.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix=".v5-sealed-test-",dir=final.parent));started=datetime.now(timezone.utc).isoformat()
    try:
        sequence_root=work/"sequences"
        for mode in ["passive","action","passive_action"]:
            subprocess.run([sys.executable,str(ROOT/"scripts/47_build_v5_sequences.py"),"--split","test",
                            "--mode",mode,"--unlock-test","--out-dir",str(sequence_root)],cwd=ROOT,check=True)
        with np.load(sequence_root/"action/test/test.npz") as a, np.load(sequence_root/"passive_action/test/test.npz") as p:
            common=set(p.files)
            if common!={"context_states","future_states","future_edge_presence","future_lateral_movement",
                        "future_techniques","future_lateral_edges","lateral_movement_within_horizon"}:
                raise ValueError("unexpected aligned test arrays")
            for name in common:
                if not np.array_equal(a[name],p[name]):raise ValueError(f"test action/aligned mismatch {name}")
        subprocess.run([sys.executable,str(ROOT/"scripts/57_evaluate_v5_models.py"),"--split","test",
                        "--sequences-dir",str(sequence_root),"--evaluation-freeze",str(freeze_path),
                        "--out",str(work/"report.json")],cwd=ROOT,check=True)
        files={str(path.relative_to(work)):file_sha256(path) for path in sorted(work.rglob("*")) if path.is_file()}
        provenance={"status":"SEALED_TEST_COMPLETE","started_utc":started,
                    "completed_utc":datetime.now(timezone.utc).isoformat(),"evaluation_freeze_sha256":file_sha256(freeze_path),
                    "git_commit":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True).stdout.strip(),
                    "artifact_sha256_before_provenance":files,"rerun_policy":"forbidden"}
        (work/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
        os.rename(work,final)
    except BaseException:
        if work.exists():
            failed=final.parent/f"_failed_sealed_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            os.rename(work,failed);print(f"sealed evaluation failed; evidence preserved at {failed}",file=sys.stderr)
        raise
    print(f"V5 sealed evaluation complete -> {final}")
    return 0


if __name__=="__main__":raise SystemExit(main())
