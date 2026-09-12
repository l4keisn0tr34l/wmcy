#!/usr/bin/env python3
"""Train V5 passive alternative-future candidates on train/validation only."""
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.branch_metrics import branch_forecast_metrics, calibration_metrics
from src.cyberwm.branching_graph_rssm import BranchingGraphRSSM
from src.cyberwm.device import cpu_state_dict, device_summary, resolve_device
from src.cyberwm.v5_model_protocol import (assert_test_sealed, compatible_parameters, file_sha256,
                                           load_export, load_protocol, load_scaler)


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def verify_freeze(path: Path, this_script: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text())
    if freeze.get("status") != "FROZEN_TRAIN_VALIDATION_ONLY": raise ValueError("V5 training freeze absent")
    expected = freeze["source_sha256"].get(str(this_script.relative_to(ROOT)))
    if expected != file_sha256(this_script): raise ValueError("passive trainer changed after V5 protocol freeze")
    if freeze["protocol_sha256"] != file_sha256(ROOT / "configs/mvp_v5_training_protocol.json"):
        raise ValueError("V5 protocol changed after freeze")
    return freeze


def initialize(config: dict[str, Any], regime: str, seed: int, protocol: dict[str, Any],
               device: torch.device) -> tuple[BranchingGraphRSSM, dict[str, Any]]:
    torch.manual_seed(seed); model = BranchingGraphRSSM(**config).to(device)
    info: dict[str, Any] = {"source": None, "loaded_parameter_tensors": 0, "missing_keys": []}
    if regime == "scratch": return model, info
    entry = protocol["initializations"][regime]; path = ROOT / entry["source"]
    if file_sha256(path) != entry["sha256"]: raise ValueError(f"{regime} source checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    selected, missing = compatible_parameters(model, checkpoint["model_state_dict"])
    if not missing or not all(name.startswith("branch_") for name in missing):
        raise ValueError(f"unexpected {regime} passive missing keys: {missing}")
    info = {"source": entry["source"], "source_sha256": entry["sha256"],
            "loaded_parameter_tensors": len(selected), "missing_keys": missing}
    return model, info


def predictions(module: Any, model: BranchingGraphRSSM, data: dict[str, np.ndarray], scaler: Any,
                device: torch.device) -> dict[str, np.ndarray]:
    context = torch.from_numpy(module.normalize(scaler, data["context_states"])).to(device)
    chunks: dict[str, list[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), 128):
            output = model.forecast_branches(context[start:start + 128], sample=False)
            for name, value in output.items(): chunks.setdefault(name, []).append(value.cpu())
    return {name: torch.cat(values).numpy() for name, values in chunks.items()}


def evaluate(module: Any, model: BranchingGraphRSSM, data: dict[str, np.ndarray], scaler: Any,
             metadata: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = predictions(module, model, data, scaler, device)
    target = module.normalize(scaler, data["future_states"])
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    groups = {"global": slice(0, global_width), "node": slice(global_width, global_width + node_width),
              "edge": slice(global_width + node_width, metadata["state_feature_count"])}
    edge_probability = torch.sigmoid(torch.from_numpy(output["edge_logits"])).numpy()
    future = branch_forecast_metrics(output["decoded"].transpose(1, 0, 2, 3), target,
        output["branch_weights"], groups, edge_probability.transpose(1, 0, 2, 3), data["future_edge_presence"])
    labels = data["lateral_movement_within_horizon"].astype(int)
    lm_probability = output["branch_weights"][:, 1]
    threshold = module.best_f1_threshold(labels, lm_probability)
    pair_branches = torch.sigmoid(torch.from_numpy(output["pair_logits"])).numpy()
    pair_probability = (output["branch_weights"][..., None] * pair_branches).sum(axis=1)
    pair_target = data["future_lateral_edges"].max(axis=1)
    positive = pair_target.sum(axis=1) > 0
    pair_ap = float(average_precision_score(pair_target.ravel(), pair_probability.ravel()))
    pair_top1 = float(np.mean(pair_target[positive, pair_probability[positive].argmax(axis=1)] > 0)) if positive.any() else None
    branch_lm = torch.sigmoid(torch.from_numpy(output["lm_logits"])).numpy()
    trajectory = np.abs(output["decoded"] - target[:, None]).mean(axis=(2, 3))
    return {"future_graph": future,
            "exact_lm": {"fixed_0_5": module.binary_metrics(labels, lm_probability, 0.5),
                         "validation_best_threshold": threshold,
                         "at_validation_best_threshold": module.binary_metrics(labels, lm_probability, threshold),
                         "calibration_diagnostic": calibration_metrics(labels, lm_probability)},
            "lm_pair": {"average_precision": pair_ap, "top1_positive_samples": pair_top1,
                        "positive_samples": int(positive.sum())},
            "branch_interpretation": {"mean_branch_weight": output["branch_weights"].mean(axis=0).tolist(),
                "mean_branch_semantic_lm_probability": branch_lm.mean(axis=0).tolist(),
                "mean_realized_outcome_branch_mae": float(trajectory[np.arange(len(labels)), labels].mean()),
                "mean_opposite_outcome_branch_mae": float(trajectory[np.arange(len(labels)), 1-labels].mean())}}


def trained_cpu_audit(module: Any, model: BranchingGraphRSSM, data: dict[str, np.ndarray], scaler: Any,
                      state_permutations: np.ndarray, pair_permutations: np.ndarray,
                      branch_module: Any) -> dict[str, Any]:
    cpu = BranchingGraphRSSM(**model.branching_config()); cpu.load_state_dict(cpu_state_dict(model.state_dict())); cpu.eval()
    equivariance = branch_module.deterministic_equivariance(
        module, cpu, data, scaler, state_permutations, pair_permutations, torch.device("cpu"))
    context = module.normalize(scaler, data["context_states"][:4]); future = module.normalize(scaler, data["future_states"][:4])
    full = torch.from_numpy(np.concatenate([context, future], axis=1))
    with torch.no_grad():
        _, first = cpu.forward_branches(full, 3, sample=False)
        changed = full.clone(); changed[:, 3:] += 999.0
        _, second = cpu.forward_branches(changed, 3, sample=False)
    causal = max(float((first[name] - second[name]).abs().max()) for name in
                 ["decoded", "edge_logits", "lm_logits", "technique_logits", "pair_logits", "branch_weights"])
    if causal != 0.0: raise AssertionError(f"trained passive causality failed: {causal}")
    return {"future_perturbation_max_delta": causal, "all_120_cpu_equivariance": equivariance}


def initial_cpu_smoke(module: Any, branch_module: Any, model: BranchingGraphRSSM,
                      data: dict[str, np.ndarray], scaler: Any,
                      state_permutations: np.ndarray, pair_permutations: np.ndarray,
                      context_steps: int, groups: list[slice], weights: dict[str, float],
                      pos: dict[str, torch.Tensor], temperature: float) -> dict[str, Any]:
    audit = trained_cpu_audit(module, model, data, scaler, state_permutations, pair_permutations, branch_module)
    loader = module.make_loader(data, scaler, 16, False); batch = next(iter(loader))
    model.train(); model.zero_grad(set_to_none=True)
    terms = branch_module.loss_terms(module, model, batch, context_steps, groups, weights, pos,
                                     temperature, "lm-outcome", sample=True)
    if not torch.isfinite(terms["total"]): raise AssertionError("nonfinite initial branch smoke loss")
    terms["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise AssertionError("nonfinite initial branch smoke gradients")
    return {**audit, "finite_gradient_parameter_tensors": len(gradients)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--freeze", type=Path, default=ROOT / "configs/mvp_v5_training_freeze.json")
    ap.add_argument("--scaler", type=Path, default=ROOT / "outputs/mvp_v5/model_protocol/shared_train_context_scaler.npz")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mvp_v5/passive_branch")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args(); assert_test_sealed(); freeze = verify_freeze(args.freeze, Path(__file__).resolve())
    if file_sha256(args.scaler) != freeze["scaler_sha256"]: raise ValueError("V5 shared scaler changed after freeze")
    if args.out_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    protocol = load_protocol(); spec = protocol["passive_branch_track"]
    device = resolve_device(args.device); torch.set_num_threads(min(8, os.cpu_count() or 1))
    train, _, metadata = load_export("passive", "train")
    validation, _, validation_metadata = load_export("passive", "validation")
    aligned, _, aligned_metadata = load_export("passive_action", "validation")
    if not (metadata["state_feature_names"] == validation_metadata["state_feature_names"] == aligned_metadata["state_feature_names"]):
        raise ValueError("passive schemas differ")
    if (len(train["context_states"]), len(validation["context_states"]), len(aligned["context_states"])) != (
            spec["train_samples"], spec["validation_samples"], spec["aligned_validation_samples"]):
        raise ValueError("passive sample counts changed")
    scaler = load_scaler(args.scaler); base = protocol["graph"]
    config = {**base, "horizon": protocol["horizon_states"], "branch_count": 2, "branch_embedding_size": 32}
    module = load_script("v5_passive_base", ROOT / "scripts/15_train_rssm.py")
    branch_module = load_script("v5_branch_functions", ROOT / "scripts/34_train_branching_graph_rssm.py")
    context_steps = protocol["context_states"]; global_width = base["global_size"]
    node_width = base["node_size"] * base["node_count"]
    groups = [slice(0, global_width), slice(global_width, global_width + node_width), slice(global_width + node_width, 345)]
    weights = spec["loss_weights"]
    pos = {"edge": module.positive_weight(train["future_edge_presence"]),
           "lm": module.positive_weight(train["lateral_movement_within_horizon"]),
           "technique": module.positive_weight(train["future_techniques"].max(axis=1)),
           "pair": module.positive_weight(train["future_lateral_edges"].max(axis=1))}
    pos = {name: value.to(device) for name, value in pos.items()}
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    if len(state_permutations) != 120: raise ValueError("expected all 120 host permutations")
    state_indices = torch.from_numpy(state_permutations).long().to(device)
    pair_indices = torch.from_numpy(pair_permutations).long().to(device)
    validation_loader = module.make_loader(validation, scaler, spec["batch_size"], False)
    temporary = Path(tempfile.mkdtemp(prefix=".v5-passive-", dir=args.out_dir.parent))
    results: dict[str, Any] = {}; selected_states: dict[str, dict[str, torch.Tensor]] = {}
    try:
        for regime in ["scratch", "friday", "v4"]:
            candidates = []; regime_states: dict[int, dict[str, torch.Tensor]] = {}
            for seed in spec["seeds"]:
                module.seed_everything(seed); model, initialization = initialize(config, regime, seed, protocol, device)
                train_loader = module.make_loader(train, scaler, spec["batch_size"], True)
                # Require exact structural causality on CPU; CUDA float32 noise is
                # separately bounded as a diagnostic rather than called leakage.
                smoke_model = BranchingGraphRSSM(**config)
                smoke_model.load_state_dict(cpu_state_dict(model.state_dict()))
                smoke_pos = {name: value.cpu() for name, value in pos.items()}
                smoke = initial_cpu_smoke(module, branch_module, smoke_model, train, scaler,
                    state_permutations, pair_permutations, context_steps, groups, weights, smoke_pos,
                    spec["mixture_temperature"])
                module.seed_everything(seed); optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
                best_value = float("inf"); best_epoch = 0; best_state = cpu_state_dict(model.state_dict()); history = []
                for epoch in range(1, spec["epochs"] + 1):
                    model.train(); total = 0.0; count = 0
                    for raw in train_loader:
                        batch = module.augment_batch(module.move(raw, device), state_indices, pair_indices)
                        optimizer.zero_grad(set_to_none=True)
                        terms = branch_module.loss_terms(module, model, batch, context_steps, groups, weights, pos,
                                                         spec["mixture_temperature"], "lm-outcome", sample=True)
                        if not torch.isfinite(terms["total"]): raise AssertionError("nonfinite V5 passive loss")
                        terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"])
                        optimizer.step(); total += float(terms["total"].detach()) * len(batch[0]); count += len(batch[0])
                    values = branch_module.validate(module, model, validation_loader, context_steps, groups, weights,
                                                     pos, spec["mixture_temperature"], "lm-outcome", device)
                    if values["selection"] < best_value - 1e-8:
                        best_value = values["selection"]; best_epoch = epoch; best_state = cpu_state_dict(model.state_dict())
                    if epoch == 1 or epoch % 25 == 0 or epoch == spec["epochs"]:
                        history.append({"epoch": epoch, "train_total": total/count, **values})
                        print(f"{regime} seed={seed} epoch={epoch} train={total/count:.5f} val={values['selection']:.5f}")
                model.load_state_dict(best_state); validation_metrics = evaluate(module, model, validation, scaler, metadata, device)
                aligned_metrics = evaluate(module, model, aligned, scaler, metadata, device)
                trained_audit = trained_cpu_audit(module, model, validation, scaler, state_permutations,
                                                  pair_permutations, branch_module)
                summary = {"seed": seed, "initialization": initialization, "smoke_tests": smoke,
                           "trained_cpu_audit": trained_audit,
                           "best_epoch": best_epoch, "best_validation_selection": best_value,
                           "validation_metrics": validation_metrics, "action_aligned_validation_metrics": aligned_metrics,
                           "history": history}
                candidates.append(summary); regime_states[seed] = best_state
            chosen = min(candidates, key=lambda row: (round(row["best_validation_selection"], 8), row["seed"]))
            if chosen["validation_metrics"]["future_graph"]["mean_pairwise_branch_mae"] <= 0:
                raise AssertionError(f"{regime}: selected passive branches have zero diversity")
            selected_states[regime] = regime_states[int(chosen["seed"])]
            results[regime] = {"candidates": candidates, "selected_seed": chosen["seed"],
                               "selected_epoch": chosen["best_epoch"],
                               "selected_validation_selection": chosen["best_validation_selection"],
                               "selected_validation_metrics": chosen["validation_metrics"],
                               "selected_action_aligned_validation_metrics": chosen["action_aligned_validation_metrics"]}
        primary = min(results, key=lambda name: (round(results[name]["selected_validation_selection"], 8), name))
        report = {"status": "VALIDATION_COMPLETE_TEST_SEALED", "scope": "V5 passive branch train/validation only",
                  "test_access": False, "runtime": device_summary(device), "protocol": spec,
                  "model_config": config, "parameter_count": sum(p.numel() for p in BranchingGraphRSSM(**config).parameters()),
                  "shared_scaler_sha256": file_sha256(args.scaler), "results": results,
                  "primary_initialization_selected_before_test": primary,
                  "prespecified_test_candidates": ["scratch", "friday", "v4"],
                  "oracle_warning": spec["oracle_metric_policy"], "limitations": protocol["claim_limits"]}
        (temporary / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
        hashes = {}
        for regime, state in selected_states.items():
            path = temporary / f"mvp_v5_passive_branch_{regime}.pt"
            torch.save({"model_state_dict": state, "model_config": config, "scaler_path": str(args.scaler.relative_to(ROOT)),
                        "scaler_sha256": file_sha256(args.scaler), "feature_metadata": metadata,
                        "protocol_id": protocol["protocol_id"], "validation": results[regime],
                        "v5_test_loaded": False}, path)
            hashes[regime] = file_sha256(path)
        (temporary / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        args.out_dir.parent.mkdir(parents=True, exist_ok=True); os.rename(temporary, args.out_dir)
        args.models_dir.mkdir(parents=True, exist_ok=True)
        for regime in selected_states:
            source = args.out_dir / f"mvp_v5_passive_branch_{regime}.pt"; destination = args.models_dir / source.name
            if destination.exists(): raise FileExistsError(destination)
            shutil.copy2(source, destination)
        print(json.dumps({"primary": primary, "validation": {name: value["selected_validation_selection"] for name, value in results.items()}}, indent=2))
        print(f"V5 passive validation -> {args.out_dir}")
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return 0


if __name__ == "__main__": raise SystemExit(main())
