#!/usr/bin/env python3
"""Freeze scratch/Friday action-conditioned models using V4 action train only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.action_graph_rssm import ActionGraphRSSM  # noqa: E402
from src.cyberwm.device import cpu_state_dict, device_summary, resolve_device  # noqa: E402
from src.cyberwm.graph_rssm import fit_graph_feature_scaler  # noqa: E402

FRIDAY_SHA256 = "f8633c796f0523fc4537212ced3584d2f448ba55ac22be0cf94092a86e359ba0"


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def make_loader(data: dict[str, np.ndarray], scaler: Any, module: Any,
                batch_size: int, shuffle: bool) -> DataLoader:
    context = module.normalize(scaler, data["context_states"])
    future = module.normalize(scaler, data["future_states"])
    tensors = [
        torch.from_numpy(context), torch.from_numpy(future),
        torch.from_numpy(data["future_edge_presence"]),
        torch.from_numpy(data["lateral_movement_within_horizon"]),
        torch.from_numpy(data["future_techniques"]),
        torch.from_numpy(data["future_lateral_edges"]),
        torch.from_numpy(data["action_type"]), torch.from_numpy(data["action_pair"]),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def augment(batch: tuple[torch.Tensor, ...], state_indices: torch.Tensor,
            pair_indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    context, future, edges, lm, techniques, lm_pairs, action_type, action_pair = batch
    choices = torch.randint(len(state_indices), (len(context),), device=context.device)
    context_out = context.clone(); future_out = future.clone(); edges_out = edges.clone()
    pairs_out = lm_pairs.clone(); action_pair_out = action_pair.clone()
    for choice in choices.unique():
        mask = choices == choice; state_order = state_indices[choice]; pair_order = pair_indices[choice]
        context_out[mask] = context[mask][..., state_order]
        future_out[mask] = future[mask][..., state_order]
        edges_out[mask] = edges[mask][..., pair_order]
        pairs_out[mask] = lm_pairs[mask][..., pair_order]
        action_pair_out[mask] = action_pair[mask][..., pair_order]
    return context_out, future_out, edges_out, lm, techniques, pairs_out, action_type, action_pair_out


def loss_terms(module: Any, model: ActionGraphRSSM, batch: tuple[torch.Tensor, ...],
               context_steps: int, groups: list[slice], weights: dict[str, float],
               pos: dict[str, torch.Tensor], sample: bool) -> dict[str, torch.Tensor]:
    context, future, edges, lm, techniques, lm_pairs, action_type, action_pair = batch
    full = torch.cat([context, future], dim=1)
    posterior, imagined = model.forward_action(
        full, context_steps, action_type, action_pair, sample=sample
    )
    technique_target = techniques.max(dim=1).values; pair_target = lm_pairs.max(dim=1).values
    terms = {
        "future_state": module.grouped_mse(imagined["decoded"], future, groups),
        "reconstruction": module.grouped_mse(posterior["decoded"], full, groups),
        "edge": F.binary_cross_entropy_with_logits(imagined["edge_logits"], edges, pos_weight=pos["edge"]),
        "kl": module.gaussian_kl(
            posterior["post_mean"], posterior["post_std"],
            posterior["prior_mean"], posterior["prior_std"],
        ).mean(),
        "lm": F.binary_cross_entropy_with_logits(imagined["lm_logits"], lm, pos_weight=pos["lm"]),
        "technique": F.binary_cross_entropy_with_logits(
            imagined["technique_logits"], technique_target, pos_weight=pos["technique"]
        ),
        "pair": F.binary_cross_entropy_with_logits(imagined["pair_logits"], pair_target, pos_weight=pos["pair"]),
    }
    terms["total"] = sum(weights[name] * terms[name] for name in weights)
    return terms


def deterministic_objective(module: Any, model: ActionGraphRSSM, loader: DataLoader,
                            context_steps: int, groups: list[slice], weights: dict[str, float],
                            pos: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    model.eval(); totals = {name: 0.0 for name in [*weights, "total"]}; count = 0
    with torch.no_grad():
        for raw in loader:
            batch = module.move(raw, device)
            terms = loss_terms(module, model, batch, context_steps, groups, weights, pos, sample=False)
            size = len(batch[0]); count += size
            for name in totals: totals[name] += float(terms[name]) * size
    return {name: value / count for name, value in totals.items()}


def forecast(model: ActionGraphRSSM, context: torch.Tensor, action_type: torch.Tensor,
             action_pair: torch.Tensor) -> dict[str, np.ndarray]:
    model.eval()
    with torch.no_grad(): output = model.forecast_action(context, action_type, action_pair, sample=False)
    return {name: value.detach().cpu().numpy() for name, value in output.items()}


def training_metrics(module: Any, model: ActionGraphRSSM, data: dict[str, np.ndarray],
                     scaler: Any, device: torch.device) -> dict[str, Any]:
    context = torch.from_numpy(module.normalize(scaler, data["context_states"])).to(device)
    action_type = torch.from_numpy(data["action_type"]).to(device)
    action_pair = torch.from_numpy(data["action_pair"]).to(device)
    output = forecast(model, context, action_type, action_pair)
    target = module.normalize(scaler, data["future_states"])
    absolute = np.abs(output["decoded"] - target)
    raw_active = data["future_states"] != 0
    edge_probability = torch.sigmoid(torch.from_numpy(output["edge_logits"])).numpy()
    lm_probability = torch.sigmoid(torch.from_numpy(output["lm_logits"])).numpy()
    lm_label = data["lateral_movement_within_horizon"].astype(int)
    threshold = module.best_f1_threshold(lm_label, lm_probability)
    pair_probability = torch.sigmoid(torch.from_numpy(output["pair_logits"])).numpy()
    pair_target = data["future_lateral_edges"].max(axis=1)
    positive = pair_target.sum(axis=1) > 0
    pair_top1 = float(np.mean(pair_target[positive, pair_probability[positive].argmax(axis=1)] > 0))
    permit = torch.tensor([[1.0, 0.0]], device=device).expand(len(context), -1)
    block = torch.tensor([[0.0, 1.0]], device=device).expand(len(context), -1)
    permit_output = forecast(model, context, permit, action_pair)
    block_output = forecast(model, context, block, action_pair)
    permit_lm = torch.sigmoid(torch.from_numpy(permit_output["lm_logits"])).numpy()
    block_lm = torch.sigmoid(torch.from_numpy(block_output["lm_logits"])).numpy()
    return {
        "normalized_state_mae": float(absolute.mean()),
        "active_normalized_state_mae": float(absolute[raw_active].mean()),
        "quiet_normalized_state_mae": float(absolute[~raw_active].mean()),
        "future_edge_average_precision": float(average_precision_score(
            data["future_edge_presence"].ravel(), edge_probability.ravel()
        )),
        "lm": module.binary_metrics(lm_label, lm_probability, threshold),
        "lm_threshold_from_training": threshold,
        "pair_top1_positive_samples": pair_top1,
        "same_context_counterfactual": {
            "mean_permit_lm_probability": float(permit_lm.mean()),
            "mean_block_lm_probability": float(block_lm.mean()),
            "mean_permit_minus_block_lm_probability": float((permit_lm - block_lm).mean()),
            "mean_normalized_state_branch_mae": float(np.abs(
                permit_output["decoded"] - block_output["decoded"]
            ).mean()),
            "mean_edge_probability_mae": float(np.abs(
                torch.sigmoid(torch.from_numpy(permit_output["edge_logits"])).numpy()
                - torch.sigmoid(torch.from_numpy(block_output["edge_logits"])).numpy()
            ).mean()),
        },
    }


def smoke_tests(module: Any, model: ActionGraphRSSM, data: dict[str, np.ndarray], scaler: Any,
                state_permutations: np.ndarray, pair_permutations: np.ndarray,
                context_steps: int, groups: list[slice], weights: dict[str, float],
                pos: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    context = torch.from_numpy(module.normalize(scaler, data["context_states"][:6])).to(device)
    future = torch.from_numpy(module.normalize(scaler, data["future_states"][:6])).to(device)
    action_type = torch.from_numpy(data["action_type"][:6]).to(device)
    action_pair = torch.from_numpy(data["action_pair"][:6]).to(device)
    full = torch.cat([context, future], dim=1); model.eval()
    with torch.no_grad():
        posterior_a, output_a = model.forward_action(full, context_steps, action_type, action_pair, sample=False)
        changed = full.clone(); changed[:, context_steps:] += 999
        posterior_b, output_b = model.forward_action(changed, context_steps, action_type, action_pair, sample=False)
        causal = max(float((posterior_a[name][:, :context_steps] - posterior_b[name][:, :context_steps]).abs().max())
                     for name in ["h", "node_h", "edge_h", "z"])
        causal = max(causal, float((output_a["decoded"] - output_b["decoded"]).abs().max()))
        base = model.forecast_action(context, action_type, action_pair, sample=False)
        equivariance = 0.0
        for state_order, pair_order in zip(state_permutations, pair_permutations):
            state_order_t = torch.from_numpy(state_order).to(device)
            pair_order_t = torch.from_numpy(pair_order).to(device)
            predicted = model.forecast_action(
                context[..., state_order_t], action_type, action_pair[..., pair_order_t], sample=False
            )
            equivariance = max(
                equivariance, float((predicted["decoded"] - base["decoded"][..., state_order_t]).abs().max()),
                float((predicted["edge_logits"] - base["edge_logits"][..., pair_order_t]).abs().max()),
                float((predicted["pair_logits"] - base["pair_logits"][..., pair_order_t]).abs().max()),
                float((predicted["lm_logits"] - base["lm_logits"]).abs().max()),
            )
    if causal != 0: raise AssertionError(f"future leakage delta {causal}")
    if equivariance > 2e-6: raise AssertionError(f"action equivariance delta {equivariance}")
    loader = make_loader(data, scaler, module, len(data["context_states"]), False)
    batch = module.move(next(iter(loader)), device); model.train(); model.zero_grad(set_to_none=True)
    terms = loss_terms(module, model, batch, context_steps, groups, weights, pos, sample=True)
    terms["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise AssertionError("non-finite action-model gradients")
    return {"future_perturbation_max_delta": causal,
            "deterministic_equivariance_max_delta": equivariance,
            "finite_gradient_parameter_tensors": len(gradients)}


def initialize_model(config: dict[str, Any], regime: str, friday_path: Path,
                     seed: int, device: torch.device) -> tuple[ActionGraphRSSM, list[str]]:
    torch.manual_seed(seed); model = ActionGraphRSSM(**config).to(device); missing: list[str] = []
    if regime == "friday_initialized":
        if file_sha256(friday_path) != FRIDAY_SHA256: raise ValueError("Friday checkpoint hash changed")
        checkpoint = torch.load(friday_path, map_location="cpu", weights_only=False)
        result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        missing = list(result.missing_keys)
        if result.unexpected_keys or not missing or not all(name.startswith("action_") for name in missing):
            raise ValueError(f"unexpected Friday initialization mismatch: {result}")
    return model, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v4/action_sequences"))
    ap.add_argument("--friday-checkpoint", default=str(ROOT / "models/friday_graph_rssm_pretrained_fixed.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp_v4/action_model"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=400); ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--learning-rate", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=43001)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(); device = resolve_device(args.device); torch.set_num_threads(min(8, os.cpu_count() or 1))
    if (args.epochs, args.batch_size, args.learning_rate, args.seed) != (400, 12, 3e-4, 43001):
        raise ValueError("V4 protocol frozen at epochs=400,batch=12,lr=3e-4,seed=43001")
    directory = Path(args.sequences_dir)
    if (directory / "test.npz").exists() or (directory / "test_sample_manifest.csv").exists():
        raise ValueError("V4 action test arrays must not exist during protocol freeze")
    report_path = Path(args.out_dir) / "train_protocol.json"
    checkpoint_paths = {
        regime: Path(args.models_dir) / f"mvp_v4_action_graph_rssm_{regime}.pt"
        for regime in ["scratch", "friday_initialized"]
    }
    if not args.force and (report_path.exists() or any(path.exists() for path in checkpoint_paths.values())):
        raise FileExistsError("refusing to replace frozen V4 action protocol outputs without --force")
    with np.load(directory / "train.npz") as loaded:
        data = {name: loaded[name] for name in loaded.files}
    metadata = json.loads((directory / "train_feature_metadata.json").read_text())
    module = load_script("v4_action_training", ROOT / "scripts/15_train_rssm.py")
    graph_script = load_script("v4_action_graph_config", ROOT / "scripts/23_train_graph_rssm.py")
    config = {**graph_script.graph_config(metadata), "action_type_count": 2, "action_embedding": 32}
    context_steps = metadata["context_states"]
    scaler = fit_graph_feature_scaler(data["context_states"], 15, 18, 3, 12, 6)
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    state_indices = torch.from_numpy(state_permutations).long().to(device)
    pair_indices = torch.from_numpy(pair_permutations).long().to(device)
    groups = [slice(0, 15), slice(15, 15 + 3 * 18), slice(15 + 3 * 18, 141)]
    weights = {"future_state": 1.0, "reconstruction": 0.1, "edge": 0.25, "kl": 0.01,
               "lm": 0.2, "technique": 0.1, "pair": 0.1}
    pos = {
        "edge": module.positive_weight(data["future_edge_presence"]).to(device),
        "lm": module.positive_weight(data["lateral_movement_within_horizon"]).to(device),
        "technique": module.positive_weight(data["future_techniques"].max(axis=1)).to(device),
        "pair": module.positive_weight(data["future_lateral_edges"].max(axis=1)).to(device),
    }
    train_loader = make_loader(data, scaler, module, args.batch_size, True)
    audit_loader = make_loader(data, scaler, module, args.batch_size, False)
    results = {}; checkpoints = {}; action_initial_states = {}
    for regime in ["scratch", "friday_initialized"]:
        model, missing = initialize_model(config, regime, Path(args.friday_checkpoint), args.seed, device)
        action_initial_states[regime] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                                         if name.startswith("action_")}
        smoke = smoke_tests(module, model, data, scaler, state_permutations, pair_permutations,
                            context_steps, groups, weights, pos, device)
        initial = deterministic_objective(module, model, audit_loader, context_steps, groups, weights, pos, device)
        torch.manual_seed(args.seed); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train(); total = 0.0
            for raw in train_loader:
                batch = augment(module.move(raw, device), state_indices, pair_indices)
                optimizer.zero_grad(set_to_none=True)
                terms = loss_terms(module, model, batch, context_steps, groups, weights, pos, sample=True)
                if not torch.isfinite(terms["total"]): raise AssertionError("non-finite action training loss")
                terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
                total += float(terms["total"].detach()) * len(batch[0])
            if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
                objective = deterministic_objective(
                    module, model, audit_loader, context_steps, groups, weights, pos, device
                )
                history.append({"epoch": epoch, "stochastic_train_total": total / len(data["context_states"]),
                                "deterministic_train_total": objective["total"]})
                print(f"{regime} epoch={epoch} train={total/len(data['context_states']):.4f} "
                      f"deterministic={objective['total']:.4f}")
        final = deterministic_objective(module, model, audit_loader, context_steps, groups, weights, pos, device)
        metrics = training_metrics(module, model, data, scaler, device)
        if final["total"] >= initial["total"]: raise AssertionError(f"{regime} did not fit action train")
        checkpoints[regime] = cpu_state_dict(model.state_dict())
        results[regime] = {"initial_training_objective": initial, "final_training_objective": final,
                           "training_metrics": metrics, "smoke_tests": smoke, "history": history,
                           "friday_missing_action_keys": missing}
    for name in action_initial_states["scratch"]:
        if not torch.equal(action_initial_states["scratch"][name], action_initial_states["friday_initialized"][name]):
            raise AssertionError(f"action-layer initialization differs: {name}")
    eligibility = {
        regime: {
            "causality": values["smoke_tests"]["future_perturbation_max_delta"] == 0,
            "equivariance": values["smoke_tests"]["deterministic_equivariance_max_delta"] < 2e-6,
            "finite_gradients": values["smoke_tests"]["finite_gradient_parameter_tensors"] > 0,
            "nontrivial_state_action_effect": values["training_metrics"]["same_context_counterfactual"]["mean_normalized_state_branch_mae"] > 0.02,
            "permit_risk_above_block": values["training_metrics"]["same_context_counterfactual"]["mean_permit_minus_block_lm_probability"] > 0.5,
        } for regime, values in results.items()
    }
    report = {
        "scope": "V4 action-train-only fixed protocol; test remains sealed",
        "test_access": "no V4 test array/episode loaded", "selection": "none; carry both initializations to sealed test",
        "protocol": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "weights": weights,
                     "initializations": ["scratch", "fixed Friday dynamics"]},
        "runtime_device": device_summary(device), "model_config": config,
        "parameter_count": sum(parameter.numel() for parameter in ActionGraphRSSM(**config).parameters()),
        "train_samples": len(data["context_states"]), "results": results,
        "eligibility_gates": eligibility,
        "all_candidates_eligible_for_sealed_test": all(all(gates.values()) for gates in eligibility.values()),
        "fixed_probability_threshold": 0.5,
        "frozen_test_metrics": ["state/active/quiet MAE", "edge AP", "LM Brier/AP/F1 at train threshold",
                                "pair top-1", "same-context permit/block probability and graph deltas"],
        "limitations": ["only 12 action-train samples", "action deterministically controls SSH outcome in this lab",
                        "matched capture contexts are not identical", "training fit is not generalization evidence"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, indent=2) + "\n")
    for regime, path in checkpoint_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": checkpoints[regime], "model_config": config,
                    "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                    "feature_metadata": metadata, "protocol": report["protocol"],
                    "training_result": results[regime],
                    "lm_threshold_from_training": results[regime]["training_metrics"]["lm_threshold_from_training"],
                    "fixed_probability_threshold": 0.5,
                    "v4_test_loaded": False}, path)
    hashes = {regime: file_sha256(path) for regime, path in checkpoint_paths.items()}
    (Path(args.out_dir) / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({regime: {"train_total": values["final_training_objective"]["total"],
                               "state_mae": values["training_metrics"]["normalized_state_mae"],
                               "lm_f1": values["training_metrics"]["lm"]["f1"],
                               "counterfactual_lm_delta": values["training_metrics"]["same_context_counterfactual"]["mean_permit_minus_block_lm_probability"]}
                      for regime, values in results.items()}, indent=2))
    print(f"protocol -> {report_path}\nhashes -> {hashes}"); return 0


if __name__ == "__main__": sys.exit(main())
