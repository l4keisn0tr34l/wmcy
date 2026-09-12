#!/usr/bin/env python3
"""Train frozen V5 action candidates using train/validation only; no test path."""
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
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.action_graph_rssm import ActionGraphRSSM
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
    if expected != file_sha256(this_script): raise ValueError("action trainer changed after V5 protocol freeze")
    if freeze["protocol_sha256"] != file_sha256(ROOT / "configs/mvp_v5_training_protocol.json"):
        raise ValueError("V5 protocol changed after freeze")
    return freeze


def initialize(config: dict[str, Any], regime: str, seed: int,
               protocol: dict[str, Any], device: torch.device
               ) -> tuple[ActionGraphRSSM, dict[str, Any]]:
    torch.manual_seed(seed); model = ActionGraphRSSM(**config).to(device)
    info: dict[str, Any] = {"source": None, "loaded_parameter_tensors": 0, "missing_keys": []}
    if regime == "scratch": return model, info
    entry = protocol["initializations"][regime]; path = ROOT / entry["source"]
    if file_sha256(path) != entry["sha256"]: raise ValueError(f"{regime} source checkpoint changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    selected, missing = compatible_parameters(model, checkpoint["model_state_dict"])
    if regime == "friday" and (not missing or not all(name.startswith("action_") for name in missing)):
        raise ValueError(f"unexpected Friday missing keys: {missing}")
    if regime == "v4" and missing: raise ValueError(f"unexpected V4 missing keys: {missing}")
    info = {"source": entry["source"], "source_sha256": entry["sha256"],
            "loaded_parameter_tensors": len(selected), "missing_keys": missing}
    return model, info


def trained_cpu_audit(module: Any, model: ActionGraphRSSM, data: dict[str, np.ndarray], scaler: Any,
                      state_permutations: np.ndarray, pair_permutations: np.ndarray) -> dict[str, Any]:
    cpu = ActionGraphRSSM(**model.action_config()); cpu.load_state_dict(cpu_state_dict(model.state_dict())); cpu.eval()
    context = torch.from_numpy(module.normalize(scaler, data["context_states"][:4]))
    action_type = torch.from_numpy(data["action_type"][:4]); action_pair = torch.from_numpy(data["action_pair"][:4])
    with torch.no_grad(): base = cpu.forecast_action(context, action_type, action_pair, sample=False)
    delta = 0.0
    for state_order, pair_order in zip(state_permutations, pair_permutations):
        si = torch.from_numpy(state_order); pi = torch.from_numpy(pair_order)
        with torch.no_grad(): predicted = cpu.forecast_action(context[..., si], action_type, action_pair[..., pi], sample=False)
        delta = max(delta, float((predicted["decoded"] - base["decoded"][..., si]).abs().max()),
                    float((predicted["edge_logits"] - base["edge_logits"][..., pi]).abs().max()),
                    float((predicted["pair_logits"] - base["pair_logits"][..., pi]).abs().max()),
                    float((predicted["lm_logits"] - base["lm_logits"]).abs().max()))
    full = torch.from_numpy(module.normalize(scaler, np.concatenate([data["context_states"][:4], data["future_states"][:4]], axis=1)))
    with torch.no_grad():
        _, first = cpu.forward_action(full, 3, action_type, action_pair, sample=False)
        changed = full.clone(); changed[:, 3:] += 999.0
        _, second = cpu.forward_action(changed, 3, action_type, action_pair, sample=False)
    causal = max(float((first[name] - second[name]).abs().max()) for name in
                 ["decoded", "edge_logits", "lm_logits", "technique_logits", "pair_logits"])
    if causal != 0.0 or delta >= 1e-5: raise AssertionError(f"trained action audit failed: causal={causal} equiv={delta}")
    return {"future_perturbation_max_delta": causal, "all_120_cpu_equivariance_max_delta": delta}


def initial_cpu_smoke(module: Any, action_module: Any, model: ActionGraphRSSM,
                      data: dict[str, np.ndarray], scaler: Any,
                      state_permutations: np.ndarray, pair_permutations: np.ndarray,
                      context_steps: int, groups: list[slice], weights: dict[str, float],
                      pos: dict[str, torch.Tensor]) -> dict[str, Any]:
    audit = trained_cpu_audit(module, model, data, scaler, state_permutations, pair_permutations)
    loader = action_module.make_loader(data, scaler, module, 6, False); batch = next(iter(loader))
    model.train(); model.zero_grad(set_to_none=True)
    terms = action_module.loss_terms(module, model, batch, context_steps, groups, weights, pos, sample=True)
    if not torch.isfinite(terms["total"]): raise AssertionError("nonfinite initial action smoke loss")
    terms["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise AssertionError("nonfinite initial action smoke gradients")
    return {**audit, "finite_gradient_parameter_tensors": len(gradients)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--freeze", type=Path, default=ROOT / "configs/mvp_v5_training_freeze.json")
    ap.add_argument("--scaler", type=Path, default=ROOT / "outputs/mvp_v5/model_protocol/shared_train_context_scaler.npz")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/mvp_v5/action_model")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args(); assert_test_sealed(); freeze = verify_freeze(args.freeze, Path(__file__).resolve())
    if file_sha256(args.scaler) != freeze["scaler_sha256"]: raise ValueError("V5 shared scaler changed after freeze")
    if args.out_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    protocol = load_protocol(); spec = protocol["action_track"]
    device = resolve_device(args.device); torch.set_num_threads(min(8, os.cpu_count() or 1))
    train, _, metadata = load_export("action", "train")
    validation, _, validation_metadata = load_export("action", "validation")
    if metadata["state_feature_names"] != validation_metadata["state_feature_names"]:
        raise ValueError("action train/validation schema mismatch")
    if (len(train["context_states"]), len(validation["context_states"])) != (spec["train_samples"], spec["validation_samples"]):
        raise ValueError("action sample count changed")
    scaler = load_scaler(args.scaler)
    base = protocol["graph"]
    config = {**base, "horizon": protocol["horizon_states"], "action_type_count": 2, "action_embedding": 32}
    module = load_script("v5_action_base", ROOT / "scripts/15_train_rssm.py")
    action_module = load_script("v5_action_functions", ROOT / "scripts/43_train_action_graph_rssm_v4.py")
    context_steps = protocol["context_states"]
    global_width = base["global_size"]; node_width = base["node_size"] * base["node_count"]
    groups = [slice(0, global_width), slice(global_width, global_width + node_width), slice(global_width + node_width, 345)]
    weights = spec["loss_weights"]
    pos = {
        "edge": module.positive_weight(train["future_edge_presence"]).to(device),
        "lm": module.positive_weight(train["lateral_movement_within_horizon"]).to(device),
        "technique": module.positive_weight(train["future_techniques"].max(axis=1)).to(device),
        "pair": module.positive_weight(train["future_lateral_edges"].max(axis=1)).to(device),
    }
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    if len(state_permutations) != 120: raise ValueError("expected all 120 host permutations")
    state_indices = torch.from_numpy(state_permutations).long().to(device)
    pair_indices = torch.from_numpy(pair_permutations).long().to(device)
    validation_loader = action_module.make_loader(validation, scaler, module, len(validation["context_states"]), False)
    temporary = Path(tempfile.mkdtemp(prefix=".v5-action-", dir=args.out_dir.parent))
    results: dict[str, Any] = {}; selected_states: dict[str, dict[str, torch.Tensor]] = {}
    try:
        for regime in ["scratch", "friday", "v4"]:
            candidates = []; states: dict[int, dict[str, torch.Tensor]] = {}
            for seed in spec["seeds"]:
                module.seed_everything(seed)
                model, initialization = initialize(config, regime, seed, protocol, device)
                # Structural causality is exact on CPU. CUDA float32 graph reductions
                # can differ by ~1e-8 when unrelated future values change memory/kernel
                # execution, so do not impose a false bitwise-zero CUDA contract.
                smoke_model = ActionGraphRSSM(**config)
                smoke_model.load_state_dict(cpu_state_dict(model.state_dict()))
                smoke_pos = {name: value.cpu() for name, value in pos.items()}
                smoke = initial_cpu_smoke(module, action_module, smoke_model, train, scaler, state_permutations,
                    pair_permutations, context_steps, groups, weights, smoke_pos)
                train_loader = action_module.make_loader(train, scaler, module, spec["batch_size"], True)
                audit_loader = action_module.make_loader(train, scaler, module, len(train["context_states"]), False)
                initial = action_module.deterministic_objective(module, model, audit_loader, context_steps,
                                                                 groups, weights, pos, device)
                module.seed_everything(seed); optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
                history = []
                for epoch in range(1, spec["epochs"] + 1):
                    model.train(); total = 0.0
                    for raw in train_loader:
                        batch = action_module.augment(module.move(raw, device), state_indices, pair_indices)
                        optimizer.zero_grad(set_to_none=True)
                        terms = action_module.loss_terms(module, model, batch, context_steps, groups, weights, pos, sample=True)
                        if not torch.isfinite(terms["total"]): raise AssertionError("nonfinite V5 action loss")
                        terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"])
                        optimizer.step(); total += float(terms["total"].detach()) * len(batch[0])
                    if epoch == 1 or epoch % 50 == 0 or epoch == spec["epochs"]:
                        objective = action_module.deterministic_objective(module, model, audit_loader, context_steps,
                                                                          groups, weights, pos, device)
                        history.append({"epoch": epoch, "stochastic_train_total": total / len(train["context_states"]),
                                        "deterministic_train_total": objective["total"]})
                        print(f"{regime} seed={seed} epoch={epoch} train={total/len(train['context_states']):.5f}")
                train_final = action_module.deterministic_objective(module, model, audit_loader, context_steps,
                                                                     groups, weights, pos, device)
                val_terms = action_module.deterministic_objective(module, model, validation_loader, context_steps,
                                                                   groups, weights, pos, device)
                selection = val_terms["future_state"] + weights["edge"] * val_terms["edge"]
                val_metrics = action_module.training_metrics(module, model, validation, scaler, device)
                val_metrics["lm_threshold_from_validation"] = val_metrics.pop("lm_threshold_from_training")
                trained_audit = trained_cpu_audit(module, model, validation, scaler, state_permutations, pair_permutations)
                if train_final["total"] >= initial["total"]: raise AssertionError(f"{regime}/{seed} failed to fit train")
                summary = {"seed": seed, "initialization": initialization, "smoke_tests": smoke,
                           "trained_cpu_audit": trained_audit, "initial_train": initial, "final_train": train_final, "validation_objective": val_terms,
                           "validation_selection": selection, "validation_metrics": val_metrics, "history": history}
                candidates.append(summary); states[seed] = cpu_state_dict(model.state_dict())
            chosen = min(candidates, key=lambda row: (round(row["validation_selection"], 8), row["seed"]))
            if chosen["validation_metrics"]["same_context_counterfactual"]["mean_permit_minus_block_lm_probability"] <= 0:
                raise AssertionError(f"{regime}: selected action candidate does not rank permit risk above block")
            selected_states[regime] = states[int(chosen["seed"])]
            results[regime] = {"candidates": candidates, "selected_seed": chosen["seed"],
                               "selected_validation_selection": chosen["validation_selection"],
                               "selected_validation_metrics": chosen["validation_metrics"]}
        primary = min(results, key=lambda name: (round(results[name]["selected_validation_selection"], 8), name))
        report = {"status": "VALIDATION_COMPLETE_TEST_SEALED", "scope": "V5 action train/validation only",
                  "test_access": False, "runtime": device_summary(device), "protocol": spec,
                  "model_config": config, "parameter_count": sum(p.numel() for p in ActionGraphRSSM(**config).parameters()),
                  "shared_scaler_sha256": file_sha256(args.scaler), "results": results,
                  "primary_initialization_selected_before_test": primary,
                  "prespecified_test_candidates": ["scratch", "friday", "v4"],
                  "fixed_primary_probability_threshold": 0.5,
                  "limitations": protocol["claim_limits"]}
        (temporary / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
        hashes = {}
        for regime, state in selected_states.items():
            path = temporary / f"mvp_v5_action_graph_rssm_{regime}.pt"
            torch.save({"model_state_dict": state, "model_config": config, "scaler_path": str(args.scaler.relative_to(ROOT)),
                        "scaler_sha256": file_sha256(args.scaler), "feature_metadata": metadata,
                        "protocol_id": protocol["protocol_id"], "validation": results[regime],
                        "fixed_probability_threshold": 0.5, "v5_test_loaded": False}, path)
            hashes[regime] = file_sha256(path)
        (temporary / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        args.out_dir.parent.mkdir(parents=True, exist_ok=True); os.rename(temporary, args.out_dir)
        args.models_dir.mkdir(parents=True, exist_ok=True)
        for regime in selected_states:
            source = args.out_dir / f"mvp_v5_action_graph_rssm_{regime}.pt"
            destination = args.models_dir / source.name
            if destination.exists(): raise FileExistsError(destination)
            shutil.copy2(source, destination)
        print(json.dumps({"primary": primary, "validation": {name: value["selected_validation_selection"] for name, value in results.items()}}, indent=2))
        print(f"V5 action validation -> {args.out_dir}")
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    return 0


if __name__ == "__main__": raise SystemExit(main())
