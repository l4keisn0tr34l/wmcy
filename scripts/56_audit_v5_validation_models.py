#!/usr/bin/env python3
"""Independently reload and audit frozen V5 validation checkpoints; no test path."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.action_graph_rssm import ActionGraphRSSM
from src.cyberwm.branch_metrics import calibration_metrics
from src.cyberwm.branching_graph_rssm import BranchingGraphRSSM
from src.cyberwm.v5_model_protocol import assert_test_sealed, file_sha256, load_export, load_protocol, load_scaler


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def state_metrics(predicted: np.ndarray, target: np.ndarray, raw_target: np.ndarray) -> dict[str, float]:
    absolute = np.abs(predicted - target); active = raw_target != 0
    return {"overall_mae": float(absolute.mean()), "global_mae": float(absolute[..., :15].mean()),
            "node_mae": float(absolute[..., 15:105].mean()), "edge_mae": float(absolute[..., 105:].mean()),
            "active_mae": float(absolute[active].mean()), "quiet_mae": float(absolute[~active].mean())}


def action_audit(base_module: Any, data: dict[str, np.ndarray], scaler: Any,
                 report: dict[str, Any]) -> dict[str, Any]:
    target = base_module.normalize(scaler, data["future_states"]); result = {}
    expected_hashes = json.loads((ROOT / "outputs/mvp_v5/action_model/checkpoint_hashes.json").read_text())
    for regime in ["scratch", "friday", "v4"]:
        path = ROOT / "models" / f"mvp_v5_action_graph_rssm_{regime}.pt"
        if file_sha256(path) != expected_hashes[regime]: raise ValueError(f"action checkpoint hash {regime}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("v5_test_loaded") is not False: raise ValueError(f"action test marker {regime}")
        model = ActionGraphRSSM(**checkpoint["model_config"]); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        context = torch.from_numpy(base_module.normalize(scaler, data["context_states"]))
        action_type = torch.from_numpy(data["action_type"]); action_pair = torch.from_numpy(data["action_pair"])
        with torch.no_grad():
            output = model.forecast_action(context, action_type, action_pair, sample=False)
            opposite = model.forecast_action(context, action_type.flip(1), action_pair, sample=False)
            permit = model.forecast_action(context, torch.tensor([[1., 0.]]).expand(len(context), -1), action_pair, sample=False)
            block = model.forecast_action(context, torch.tensor([[0., 1.]]).expand(len(context), -1), action_pair, sample=False)
        decoded = output["decoded"].numpy(); edge_probability = torch.sigmoid(output["edge_logits"]).numpy()
        lm_probability = torch.sigmoid(output["lm_logits"]).numpy(); labels = data["lateral_movement_within_horizon"].astype(int)
        pair_probability = torch.sigmoid(output["pair_logits"]).numpy(); pair_target = data["future_lateral_edges"].max(axis=1)
        positive = pair_target.sum(axis=1) > 0
        factual_error = np.abs(decoded-target).mean(axis=(1,2)); opposite_error = np.abs(opposite["decoded"].numpy()-target).mean(axis=(1,2))
        threshold = report["results"][regime]["selected_validation_metrics"]["lm_threshold_from_validation"]
        metrics = {"state": state_metrics(decoded, target, data["future_states"]),
                   "future_edge_average_precision": float(average_precision_score(data["future_edge_presence"].ravel(), edge_probability.ravel())),
                   "lm": {"fixed_0_5": base_module.binary_metrics(labels, lm_probability, 0.5),
                          "validation_threshold": threshold,
                          "at_validation_threshold": base_module.binary_metrics(labels, lm_probability, threshold),
                          "calibration_diagnostic": calibration_metrics(labels, lm_probability)},
                   "pair": {"micro_average_precision": float(average_precision_score(pair_target.ravel(), pair_probability.ravel())),
                            "top1_positive_samples": float(np.mean(pair_target[positive, pair_probability[positive].argmax(1)] > 0)),
                            "positive_samples": int(positive.sum())},
                   "counterfactual": {"factual_lower_state_error_count": int((factual_error < opposite_error).sum()),
                                      "contexts": len(factual_error),
                                      "mean_factual_minus_opposite_state_error": float((factual_error-opposite_error).mean()),
                                      "mean_permit_minus_block_lm_probability": float((torch.sigmoid(permit["lm_logits"])-torch.sigmoid(block["lm_logits"])).mean())}}
        saved = report["results"][regime]["selected_validation_metrics"]
        if abs(metrics["state"]["overall_mae"] - saved["normalized_state_mae"]) > 1e-6:
            raise ValueError(f"action metric mismatch {regime}")
        chosen_seed = report["results"][regime]["selected_seed"]
        chosen = next(row for row in report["results"][regime]["candidates"] if row["seed"] == chosen_seed)
        trained = chosen["trained_cpu_audit"]
        if trained["future_perturbation_max_delta"] != 0 or trained["all_120_cpu_equivariance_max_delta"] >= 1e-5:
            raise ValueError(f"action structural gate {regime}")
        if metrics["counterfactual"]["mean_permit_minus_block_lm_probability"] <= 0:
            raise ValueError(f"action direction gate {regime}")
        result[regime] = metrics
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mvp_v5/validation_audit")
    args = ap.parse_args(); assert_test_sealed()
    if args.out_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    protocol = load_protocol(); freeze = json.loads((ROOT / "configs/mvp_v5_training_freeze.json").read_text())
    scaler_path = ROOT / "outputs/mvp_v5/model_protocol/shared_train_context_scaler.npz"
    if file_sha256(scaler_path) != freeze["scaler_sha256"]: raise ValueError("shared scaler hash")
    scaler = load_scaler(scaler_path); base_module = load_script("v5_validation_base", ROOT / "scripts/15_train_rssm.py")
    action_data, _, _ = load_export("action", "validation")
    action_report_path = ROOT / "outputs/mvp_v5/action_model/validation_report.json"
    action_report = json.loads(action_report_path.read_text())
    if action_report["test_access"] is not False: raise ValueError("action report test marker")
    action = action_audit(base_module, action_data, scaler, action_report)
    passive_data, _, passive_metadata = load_export("passive", "validation")
    aligned_data, _, _ = load_export("passive_action", "validation")
    passive_report_path = ROOT / "outputs/mvp_v5/passive_branch/validation_report.json"
    passive_report = json.loads(passive_report_path.read_text())
    if passive_report["test_access"] is not False: raise ValueError("passive report test marker")
    passive_module = load_script("v5_validation_passive", ROOT / "scripts/54_train_v5_passive_branch.py")
    expected_hashes = json.loads((ROOT / "outputs/mvp_v5/passive_branch/checkpoint_hashes.json").read_text())
    passive = {}
    for regime in ["scratch", "friday", "v4"]:
        path = ROOT / "models" / f"mvp_v5_passive_branch_{regime}.pt"
        if file_sha256(path) != expected_hashes[regime]: raise ValueError(f"passive checkpoint hash {regime}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("v5_test_loaded") is not False: raise ValueError(f"passive test marker {regime}")
        model = BranchingGraphRSSM(**checkpoint["model_config"]); model.load_state_dict(checkpoint["model_state_dict"])
        regular = passive_module.evaluate(base_module, model, passive_data, scaler, passive_metadata, torch.device("cpu"))
        aligned = passive_module.evaluate(base_module, model, aligned_data, scaler, passive_metadata, torch.device("cpu"))
        saved = passive_report["results"][regime]["selected_validation_metrics"]
        if abs(regular["future_graph"]["expected_forecast_mae"] - saved["future_graph"]["expected_forecast_mae"]) > 1e-6:
            raise ValueError(f"passive metric mismatch {regime}")
        chosen_seed = passive_report["results"][regime]["selected_seed"]
        chosen = next(row for row in passive_report["results"][regime]["candidates"] if row["seed"] == chosen_seed)
        audit = chosen["trained_cpu_audit"]
        if audit["future_perturbation_max_delta"] != 0 or max(audit["all_120_cpu_equivariance"].values()) >= 1e-5:
            raise ValueError(f"passive structural gate {regime}")
        if regular["future_graph"]["mean_pairwise_branch_mae"] <= 0: raise ValueError(f"passive collapse {regime}")
        passive[regime] = {"ordinary_validation": regular, "action_aligned_validation": aligned}
    report = {"status": "PASS_VALIDATION_ONLY_TEST_SEALED", "protocol_id": protocol["protocol_id"],
              "training_freeze_sha256": file_sha256(ROOT / "configs/mvp_v5_training_freeze.json"),
              "shared_scaler_sha256": file_sha256(scaler_path), "test_exports_absent": True,
              "action_primary": action_report["primary_initialization_selected_before_test"],
              "passive_primary": passive_report["primary_initialization_selected_before_test"],
              "action": action, "passive": passive,
              "checkpoint_sha256": {"action": json.loads((ROOT / "outputs/mvp_v5/action_model/checkpoint_hashes.json").read_text()),
                                    "passive": expected_hashes},
              "training_console_sha256": {"action": file_sha256(ROOT / "outputs/mvp_v5/action_model/training_console.log"),
                                          "passive": file_sha256(ROOT / "outputs/mvp_v5/passive_branch/training_console.log")},
              "warnings": ["validation is model-selection evidence, not sealed generalization",
                           "action LM labels are deterministic under permit/block; graph dynamics remain essential",
                           "passive oracle gains are candidate coverage only",
                           "small correlated-window calibration diagnostics are not deployment calibration"]}
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v5-validation-audit-", dir=args.out_dir.parent))
    try:
        (temporary / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        os.rename(temporary, args.out_dir)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    print(json.dumps({"status": report["status"], "action_primary": report["action_primary"],
        "passive_primary": report["passive_primary"], "action_fixed_0_5_f1": {k:v["lm"]["fixed_0_5"]["f1"] for k,v in action.items()},
        "passive_fixed_0_5_f1": {k:v["ordinary_validation"]["exact_lm"]["fixed_0_5"]["f1"] for k,v in passive.items()}}, indent=2))
    print(f"V5 validation audit -> {args.out_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
