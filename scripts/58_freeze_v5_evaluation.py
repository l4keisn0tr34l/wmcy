#!/usr/bin/env python3
"""Freeze checkpoints, thresholds, evaluator, and metrics before one V5 test run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cyberwm.v5_model_protocol import file_sha256

REGIMES=("scratch","friday","v4")
SOURCES=(
 "configs/mvp_v5_training_protocol.json","configs/mvp_v5_training_freeze.json",
 "src/cyberwm/graph_rssm.py","src/cyberwm/action_graph_rssm.py","src/cyberwm/branching_graph_rssm.py",
 "src/cyberwm/branch_metrics.py","src/cyberwm/v5_contract.py","src/cyberwm/v5_sequences.py",
 "src/cyberwm/v5_model_protocol.py","scripts/15_train_rssm.py","scripts/47_build_v5_sequences.py",
 "scripts/56_audit_v5_validation_models.py","scripts/57_evaluate_v5_models.py",
 "scripts/58_freeze_v5_evaluation.py","scripts/59_run_v5_sealed_evaluation.py",
)


def main()->int:
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--check-only",action="store_true")
 ap.add_argument("--out",type=Path,default=ROOT/"configs/mvp_v5_evaluation_freeze.json")
 args=ap.parse_args()
 if (ROOT/"outputs/mvp_v5/sealed_test").exists():raise PermissionError("V5 sealed test already exists")
 if any((ROOT/"outputs/mvp_v5/sequences"/m/"test").exists() for m in ["passive","action","passive_action"]):
  raise PermissionError("test export exists before evaluation freeze")
 training_freeze_path=ROOT/"configs/mvp_v5_training_freeze.json";training_freeze=json.loads(training_freeze_path.read_text())
 if training_freeze.get("status")!="FROZEN_TRAIN_VALIDATION_ONLY" or training_freeze.get("test_unlock") is not False:
  raise ValueError("training freeze status")
 validation_path=ROOT/"outputs/mvp_v5/validation_audit/audit.json";validation=json.loads(validation_path.read_text())
 if validation.get("status")!="PASS_VALIDATION_ONLY_TEST_SEALED" or not validation.get("test_exports_absent"):
  raise ValueError("validation audit status")
 checkpoint_hashes=validation["checkpoint_sha256"]
 for track,stem in [("action","action_graph_rssm"),("passive","passive_branch")]:
  for regime in REGIMES:
   if file_sha256(ROOT/"models"/f"mvp_v5_{stem}_{regime}.pt")!=checkpoint_hashes[track][regime]:
    raise ValueError(f"checkpoint changed {track}/{regime}")
 thresholds={"action":{r:validation["action"][r]["lm"]["validation_threshold"] for r in REGIMES},
             "passive":{r:validation["passive"][r]["ordinary_validation"]["exact_lm"]["validation_best_threshold"] for r in REGIMES}}
 if validation["action_primary"]!="scratch" or validation["passive_primary"]!="scratch":raise ValueError("primary candidate changed")
 smoke=ROOT/"outputs/mvp_v5/evaluation_protocol/validation_smoke.json"
 smoke_report=json.loads(smoke.read_text())
 if smoke_report.get("status")!="VALIDATION_EVALUATOR_SMOKE_COMPLETE" or smoke_report.get("test_access") is not False:
  raise ValueError("validation evaluator smoke status")
 source_hashes={path:file_sha256(ROOT/path) for path in SOURCES}
 freeze={"status":"FROZEN_FOR_ONE_SEALED_TEST","protocol_id":"mvp_v5_sealed_evaluation_v1",
  "source_sha256":source_hashes,"git_commit":None,
  "runtime_versions":{"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,
                      "sklearn":sklearn.__version__,"torch":torch.__version__,"torch_cuda":torch.version.cuda},
  "training_freeze_sha256":file_sha256(training_freeze_path),
  "training_protocol_sha256":file_sha256(ROOT/"configs/mvp_v5_training_protocol.json"),
  "shared_scaler_sha256":validation["shared_scaler_sha256"],
  "validation_audit_sha256":file_sha256(validation_path),"validation_evaluator_smoke_sha256":file_sha256(smoke),
  "checkpoint_sha256":checkpoint_hashes,"primary_action":"scratch","primary_passive":"scratch",
  "thresholds":thresholds,"test_expected_samples":{"action":8,"passive":336,"passive_action":8},
  "test_metrics":["same-scaler state MAE overall/global/node/edge/active/quiet","future-edge AP",
   "LM AP/Brier/F1 at fixed0.5 and frozen validation threshold","ATT&CK micro/per-technique AP","LM-pair AP/top1",
   "factual-vs-opposite action state preference","passive expected/oracle/diversity with oracle coverage-only",
   "persistence references","scenario counts/diagnostics","test-only intent probe at metadata-frozen forecast cutoff"],
  "test_unlock":True,"allowed_runs":1,"post_test_retuning":False,
  "claim_limits":["one synthetic flat five-host topology","deterministic SSH permit/block",
   "small test cohorts","no enterprise/unseen-topology/causal-policy/calibrated-uncertainty claim"]}
 if args.check_only:
  print(json.dumps({k:freeze[k] for k in ["status","primary_action","primary_passive","checkpoint_sha256","thresholds","test_expected_samples"]},indent=2));return 0
 if args.out.exists():raise FileExistsError(args.out)
 status=subprocess.run(["git","status","--porcelain"],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
 if status:raise RuntimeError("commit evaluator/freeze sources before final freeze; git tree not clean")
 freeze["git_commit"]=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,stdout=subprocess.PIPE,check=True).stdout.strip()
 args.out.write_text(json.dumps(freeze,indent=2)+"\n")
 print(json.dumps({k:freeze[k] for k in ["status","git_commit","primary_action","primary_passive","test_unlock","allowed_runs"]},indent=2))
 print(f"V5 evaluation freeze -> {args.out}")
 return 0


if __name__=="__main__":raise SystemExit(main())
