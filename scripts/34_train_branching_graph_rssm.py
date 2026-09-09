#!/usr/bin/env python3
"""Train an explicit two-branch graph RSSM using V3 train/validation only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.branch_metrics import branch_forecast_metrics, calibration_metrics  # noqa: E402
from src.cyberwm.branching_graph_rssm import BranchingGraphRSSM  # noqa: E402
from src.cyberwm.device import cpu_state_dict, device_summary, resolve_device  # noqa: E402
from src.cyberwm.graph_rssm import fit_graph_feature_scaler  # noqa: E402


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def grouped_branch_mse(predicted: torch.Tensor, target: torch.Tensor,
                       groups: list[slice]) -> torch.Tensor:
    # [batch,branch,horizon,feature] -> [batch,branch]
    return torch.stack([
        (predicted[..., group] - target[:, None, ..., group]).square().mean(dim=(2, 3))
        for group in groups
    ], dim=-1).mean(dim=-1)


def responsible_bce(logits: torch.Tensor, target: torch.Tensor,
                    responsibilities: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    expanded = target[:, None].expand_as(logits)
    raw = F.binary_cross_entropy_with_logits(logits, expanded, pos_weight=pos_weight, reduction="none")
    if raw.ndim > 2: raw = raw.mean(dim=tuple(range(2, raw.ndim)))
    return (responsibilities * raw).sum(dim=1).mean()


def loss_terms(module: Any, model: BranchingGraphRSSM, batch: tuple[torch.Tensor, ...],
               context_steps: int, groups: list[slice], weights: dict[str, float],
               pos_weights: dict[str, torch.Tensor], temperature: float,
               responsibility_mode: str, sample: bool) -> dict[str, torch.Tensor]:
    context, future, future_edges, lm, techniques, lm_pairs = batch
    full = torch.cat([context, future], dim=1)
    posterior, branches = model.forward_branches(full, context_steps, sample=sample)
    trajectory_error = grouped_branch_mse(branches["decoded"], future, groups)
    log_weights = torch.log_softmax(branches["branch_logits"], dim=-1)
    if responsibility_mode == "lm-outcome":
        if model.branch_count != 2:
            raise ValueError("lm-outcome responsibilities require exactly two branches")
        # Branch 0 means no LM in the horizon; branch 1 means LM. The future
        # outcome is a training target only. Branch probabilities remain context-only.
        responsibilities = F.one_hot(lm.long(), num_classes=2).to(trajectory_error.dtype)
        assigned_error = (responsibilities * trajectory_error).sum(dim=1)
        gate_nll = -(responsibilities * log_weights).sum(dim=1)
        mixture_state = (assigned_error + temperature * gate_nll).mean()
    elif responsibility_mode == "trajectory":
        mixture_state = (-temperature * torch.logsumexp(
            log_weights - trajectory_error / temperature, dim=1
        )).mean()
        responsibilities = torch.softmax(
            log_weights.detach() - trajectory_error.detach() / temperature, dim=1
        )
    else:
        raise ValueError(f"unknown responsibility mode: {responsibility_mode}")
    reconstruction = module.grouped_mse(posterior["decoded"], full, groups)
    kl = module.gaussian_kl(
        posterior["post_mean"], posterior["post_std"],
        posterior["prior_mean"], posterior["prior_std"],
    ).clamp_min(weights["free_nats"]).mean()
    technique_target = techniques.max(dim=1).values
    pair_target = lm_pairs.max(dim=1).values
    edge = responsible_bce(branches["edge_logits"], future_edges, responsibilities, pos_weights["edge"])
    lm_loss = responsible_bce(branches["lm_logits"], lm, responsibilities, pos_weights["lm"])
    technique = responsible_bce(
        branches["technique_logits"], technique_target, responsibilities, pos_weights["technique"]
    )
    pair = responsible_bce(branches["pair_logits"], pair_target, responsibilities, pos_weights["pair"])
    mean_weight = branches["branch_weights"].mean(dim=0)
    balance = (mean_weight * torch.log(mean_weight * model.branch_count + 1e-8)).sum()
    total = (weights["future_state"] * mixture_state + weights["reconstruction"] * reconstruction
             + weights["edge"] * edge + weights["kl"] * kl + weights["lm"] * lm_loss
             + weights["technique"] * technique + weights["pair"] * pair
             + weights["branch_balance"] * balance)
    return {
        "total": total, "mixture_state": mixture_state,
        "oracle_state_mse": trajectory_error.min(dim=1).values.mean(),
        "reconstruction": reconstruction, "edge": edge, "kl": kl, "lm": lm_loss,
        "technique": technique, "pair": pair, "branch_balance": balance,
        "responsibility_entropy": -(responsibilities * torch.log(responsibilities + 1e-8)).sum(dim=1).mean(),
    }


def validate(module: Any, model: BranchingGraphRSSM, loader: Any, context_steps: int,
             groups: list[slice], weights: dict[str, float], pos_weights: dict[str, torch.Tensor],
             temperature: float, responsibility_mode: str,
             device: torch.device) -> dict[str, float]:
    totals: dict[str, float] = {}; count = 0; model.eval()
    with torch.no_grad():
        for raw in loader:
            batch = module.move(raw, device)
            terms = loss_terms(module, model, batch, context_steps, groups, weights,
                               pos_weights, temperature, responsibility_mode, sample=False)
            size = len(batch[0]); count += size
            for name, value in terms.items(): totals[name] = totals.get(name, 0.0) + float(value) * size
    result = {name: value / count for name, value in totals.items()}
    result["selection"] = (result["mixture_state"] + weights["edge"] * result["edge"]
                           + weights["lm"] * result["lm"]
                           + weights["technique"] * result["technique"]
                           + weights["pair"] * result["pair"])
    return result


def smoke_tests(module: Any, model: BranchingGraphRSSM, loader: Any, raw_context: np.ndarray,
                scaler: Any, state_permutations: np.ndarray, pair_permutations: np.ndarray,
                context_steps: int, groups: list[slice], weights: dict[str, float],
                pos_weights: dict[str, torch.Tensor], temperature: float,
                responsibility_mode: str, device: torch.device) -> dict[str, float]:
    batch = module.move(tuple(value[:16] for value in next(iter(loader))), device)
    full = torch.cat([batch[0], batch[1]], dim=1); model.eval()
    with torch.no_grad():
        posterior_a, branches_a = model.forward_branches(full, context_steps, sample=False)
        changed = full.clone(); changed[:, context_steps:] += 123.0
        posterior_b, branches_b = model.forward_branches(changed, context_steps, sample=False)
        causal = max(
            float((posterior_a[name][:, :context_steps] - posterior_b[name][:, :context_steps]).abs().max())
            for name in ["h", "node_h", "edge_h", "z"]
        )
        causal = max(causal, float((branches_a["branch_weights"] - branches_b["branch_weights"]).abs().max()),
                     float((branches_a["decoded"] - branches_b["decoded"]).abs().max()))
        if causal != 0.0: raise AssertionError(f"future changed context-only branches: {causal}")
        if branches_a["decoded"].shape != (16, model.branch_count, model.horizon, model.observation_size):
            raise AssertionError("unexpected branching future shape")
        context = torch.from_numpy(module.normalize(scaler, raw_context[:8])).to(device)
        base = model.forecast_branches(context, sample=False); equivariance = 0.0
        for state_order, pair_order in zip(state_permutations, pair_permutations):
            permuted = torch.from_numpy(module.normalize(scaler, raw_context[:8, :, state_order])).to(device)
            prediction = model.forecast_branches(permuted, sample=False)
            equivariance = max(equivariance,
                float((prediction["branch_weights"] - base["branch_weights"]).abs().max()),
                float((prediction["decoded"] - base["decoded"][..., state_order]).abs().max()),
                float((prediction["edge_logits"] - base["edge_logits"][..., pair_order]).abs().max()),
                float((prediction["pair_logits"] - base["pair_logits"][..., pair_order]).abs().max()),
                float((prediction["lm_logits"] - base["lm_logits"]).abs().max()))
        if equivariance > 2e-6: raise AssertionError(f"branch model not equivariant: {equivariance}")
    tiny = deepcopy(model); optimizer = torch.optim.Adam(tiny.parameters(), lr=1e-3); losses = []
    tiny.train()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        terms = loss_terms(module, tiny, batch, context_steps, groups, weights,
                           pos_weights, temperature, responsibility_mode, sample=True)
        if not torch.isfinite(terms["total"]): raise AssertionError("non-finite tiny loss")
        terms["total"].backward(); nn.utils.clip_grad_norm_(tiny.parameters(), 10.0)
        optimizer.step(); losses.append(float(terms["total"].detach()))
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    if last >= first: raise AssertionError(f"tiny-overfit failed: {first}->{last}")
    return {"causal_context_and_branch_max_delta": causal,
            "deterministic_equivariance_max_delta": equivariance,
            "tiny_overfit_first": first, "tiny_overfit_last": last}


def train_seed(module: Any, seed: int, train_loader: Any, validation_loader: Any,
               config: dict[str, Any], context_steps: int, groups: list[slice],
               weights: dict[str, float], pos_weights: dict[str, torch.Tensor],
               state_indices: torch.Tensor, edge_indices: torch.Tensor,
               temperature: float, responsibility_mode: str,
               epochs: int, patience: int, learning_rate: float,
               device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    module.seed_everything(seed); model = BranchingGraphRSSM(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = cpu_state_dict(model.state_dict()); best_value = math.inf; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for raw in train_loader:
            batch = module.augment_batch(module.move(raw, device), state_indices, edge_indices)
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(module, model, batch, context_steps, groups, weights,
                               pos_weights, temperature, responsibility_mode, sample=True)
            terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step(); total += float(terms["total"].detach()) * len(batch[0]); count += len(batch[0])
        values = validate(module, model, validation_loader, context_steps, groups,
                          weights, pos_weights, temperature, responsibility_mode, device)
        history.append({"epoch": epoch, "train": total / count, **values})
        if values["selection"] < best_value - 1e-5:
            best_value = values["selection"]; best_epoch = epoch
            best = cpu_state_dict(model.state_dict()); stale = 0
        else: stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"seed={seed} epoch={epoch} train={total/count:.4f} val={values['selection']:.4f} "
                  f"mix={values['mixture_state']:.4f} oracle={values['oracle_state_mse']:.4f}")
        if stale >= patience: break
    return best, {"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                  "best_validation_selection": best_value, "history": history}


def evaluate_validation(module: Any, model: BranchingGraphRSSM, validation: dict[str, np.ndarray],
                        scaler: Any, metadata: dict[str, Any], device: torch.device,
                        contract_script: Any, responsibility_mode: str) -> dict[str, Any]:
    context = torch.from_numpy(module.normalize(scaler, validation["context_states"])).to(device)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(context), 128):
            outputs.append(model.forecast_branches(context[start:start + 128], sample=False))
    names = ["decoded", "edge_logits", "lm_logits", "technique_logits", "pair_logits",
             "branch_weights", "branch_logits"]
    combined = {name: torch.cat([value[name].cpu() for value in outputs], dim=0).numpy() for name in names}
    target = module.normalize(scaler, validation["future_states"])
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    groups = {"global": slice(0, global_width), "node": slice(global_width, global_width + node_width),
              "edge": slice(global_width + node_width, metadata["state_feature_count"])}
    future = branch_forecast_metrics(
        combined["decoded"].transpose(1, 0, 2, 3), target, combined["branch_weights"], groups,
        edge_probability_branches=torch.sigmoid(torch.from_numpy(combined["edge_logits"])).numpy().transpose(1, 0, 2, 3),
        edge_targets=validation["future_edge_presence"],
    )
    weights = combined["branch_weights"]
    branch_lm = torch.sigmoid(torch.from_numpy(combined["lm_logits"])).numpy()
    semantic_mixture_probability = (weights * branch_lm).sum(axis=1)
    lm_labels = validation["lateral_movement_within_horizon"].astype(int)
    # In the outcome-conditioned model, branch 1 probability is itself the
    # exact-outcome probability. The semantic head describes risk within each branch.
    lm_probability = (weights[:, 1] if responsibility_mode == "lm-outcome"
                      else semantic_mixture_probability)
    threshold = module.best_f1_threshold(lm_labels, lm_probability)
    trajectory = np.abs(combined["decoded"] - target[:, None]).mean(axis=(2, 3))
    assignment = trajectory.argmin(axis=1)
    contingency = [[int(((assignment == branch) & (lm_labels == label)).sum()) for label in [0, 1]]
                   for branch in range(model.branch_count)]
    error_by_branch_and_outcome = [
        [float(trajectory[lm_labels == label, branch].mean()) for label in [0, 1]]
        for branch in range(model.branch_count)
    ]
    absolute = np.abs(combined["decoded"] - target[:, None])
    active_mask = validation["future_states"] != 0
    active_error = [[
        float(absolute[lm_labels == label, branch][active_mask[lm_labels == label]].mean())
        for label in [0, 1]
    ] for branch in range(model.branch_count)]
    quiet_error = [[
        float(absolute[lm_labels == label, branch][~active_mask[lm_labels == label]].mean())
        for label in [0, 1]
    ] for branch in range(model.branch_count)]
    edge_probability = torch.sigmoid(torch.from_numpy(combined["edge_logits"])).numpy()
    edge_ap = [[float(average_precision_score(
        validation["future_edge_presence"][lm_labels == label].ravel(),
        edge_probability[lm_labels == label, branch].ravel(),
    )) for label in [0, 1]] for branch in range(model.branch_count)]
    feature_diversity = np.abs(combined["decoded"][:, 0] - combined["decoded"][:, 1])
    mean_feature_diversity = feature_diversity.mean(axis=(0, 1))
    top_indices = np.argsort(mean_feature_diversity)[::-1][:12]
    top_features = [{"feature": metadata["state_feature_names"][int(index)],
                     "normalized_mean_absolute_branch_delta": float(mean_feature_diversity[index])}
                    for index in top_indices]
    conditioned_realized = float(trajectory[np.arange(len(trajectory)), lm_labels].mean())
    conditioned_opposite = float(trajectory[np.arange(len(trajectory)), 1 - lm_labels].mean())
    episode = contract_script.episode_summary(
        module, Path(args_global.sequences_dir), lm_probability, lm_labels, threshold,
        Path(args_global.paired_plan),
    )
    return {
        "future_graph": future,
        "exact_lm": {"threshold_selected_on_validation": threshold,
                     "classification": module.binary_metrics(lm_labels, lm_probability, threshold),
                     "calibration": calibration_metrics(lm_labels, lm_probability)},
        "branch_interpretation": {
            "branch_meanings": (["no_lm_within_horizon", "lm_within_horizon"]
                                if responsibility_mode == "lm-outcome" else ["exchangeable"] * model.branch_count),
            "mean_branch_lm_probability": branch_lm.mean(axis=0).tolist(),
            "mean_branch_weight": weights.mean(axis=0).tolist(),
            "semantic_head_mixture_brier": float(np.mean((semantic_mixture_probability - lm_labels) ** 2)),
            "nearest_future_branch_by_exact_lm_label_counts": contingency,
            "nearest_future_branch_assignment": assignment.tolist(),
            "state_mae_by_branch_and_outcome": error_by_branch_and_outcome,
            "active_state_mae_by_branch_and_outcome": active_error,
            "quiet_state_mae_by_branch_and_outcome": quiet_error,
            "future_edge_ap_by_branch_and_outcome": edge_ap,
            "top_branch_divergent_features": top_features,
            "outcome_conditioned_realized_branch_mae": (
                conditioned_realized if responsibility_mode == "lm-outcome" else None
            ),
            "opposite_outcome_branch_mae": (
                conditioned_opposite if responsibility_mode == "lm-outcome" else None
            ),
            "conditioned_branch_state_gain": (
                conditioned_opposite - conditioned_realized
                if responsibility_mode == "lm-outcome" else None
            ),
        },
        "dangerous_precursor_operational_view": episode,
    }


def deterministic_equivariance(module: Any, model: BranchingGraphRSSM, validation: dict[str, np.ndarray],
                               scaler: Any, state_permutations: np.ndarray,
                               pair_permutations: np.ndarray, device: torch.device) -> dict[str, float]:
    raw = validation["context_states"][:32]
    base_context = torch.from_numpy(module.normalize(scaler, raw)).to(device); model.eval()
    with torch.no_grad(): base = model.forecast_branches(base_context, sample=False)
    values = {name: 0.0 for name in ["branch_weight", "state", "edge", "pair", "lm", "technique"]}
    for state_order, pair_order in zip(state_permutations, pair_permutations):
        context = torch.from_numpy(module.normalize(scaler, raw[..., state_order])).to(device)
        with torch.no_grad(): predicted = model.forecast_branches(context, sample=False)
        values["branch_weight"] = max(values["branch_weight"], float((predicted["branch_weights"] - base["branch_weights"]).abs().max()))
        values["state"] = max(values["state"], float((predicted["decoded"] - base["decoded"][..., state_order]).abs().max()))
        values["edge"] = max(values["edge"], float((predicted["edge_logits"] - base["edge_logits"][..., pair_order]).abs().max()))
        values["pair"] = max(values["pair"], float((predicted["pair_logits"] - base["pair_logits"][..., pair_order]).abs().max()))
        values["lm"] = max(values["lm"], float((predicted["lm_logits"] - base["lm_logits"]).abs().max()))
        values["technique"] = max(values["technique"], float((predicted["technique_logits"] - base["technique_logits"]).abs().max()))
    # Trained logits can be large, so float32 reduction-order noise reaches a few e-6.
    if max(values.values()) > 1e-5: raise AssertionError(f"equivariance failed: {values}")
    return {f"{name}_max_delta": value for name, value in values.items()}


def main() -> int:
    global args_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v3/sequences"))
    ap.add_argument("--paired-plan", default=str(ROOT / "configs/mvp_v3_new_episode_plan.csv"))
    ap.add_argument("--baseline-contract", default=str(ROOT / "outputs/mvp_v3/branching/contract_baseline_validation.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v3/branching/outcome_two_branch_validation.json"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/mvp_v3_outcome_branching_graph_rssm_validation.pt"))
    ap.add_argument("--branches", type=int, default=2); ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--responsibility-mode", choices=["trajectory", "lm-outcome"], default="lm-outcome")
    ap.add_argument("--epochs", type=int, default=250); ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--resume-candidates", action="store_true",
                    help="reuse compatible per-seed checkpoints after an interrupted audit")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args_global = ap.parse_args(); torch.set_num_threads(min(8, os.cpu_count() or 1)); device = resolve_device(args_global.device)
    module = load_script("branch_training", ROOT / "scripts/15_train_rssm.py")
    graph_script = load_script("branch_graph_config", ROOT / "scripts/23_train_graph_rssm.py")
    contract_script = load_script("branch_contract", ROOT / "scripts/33_evaluate_branching_contract.py")
    sequence_dir = Path(args_global.sequences_dir); metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    # Deliberately load train and validation only. This script has no test path.
    data = {split: module.load_split(sequence_dir, split) for split in ["train", "validation"]}
    base_config = graph_script.graph_config(metadata)
    config = {**base_config, "branch_count": args_global.branches, "branch_embedding_size": 32}
    scaler = fit_graph_feature_scaler(
        data["train"]["context_states"], base_config["global_size"], base_config["node_size"],
        base_config["node_count"], base_config["edge_size"], base_config["pair_count"],
    )
    context_steps = metadata["context_states"]; global_width = base_config["global_size"]
    node_width = base_config["node_size"] * base_config["node_count"]
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, metadata["state_feature_count"])]
    weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25, "kl": 0.01,
               "lm": 0.2, "technique": 0.1, "pair": 0.1,
               "branch_balance": 0.0 if args_global.responsibility_mode == "lm-outcome" else 0.02,
               "free_nats": 0.0}
    pos_weights = {
        "edge": module.positive_weight(data["train"]["future_edge_presence"]),
        "lm": module.positive_weight(data["train"]["lateral_movement_within_horizon"]),
        "technique": module.positive_weight(data["train"]["future_techniques"].max(axis=1)),
        "pair": module.positive_weight(data["train"]["future_lateral_edges"].max(axis=1)),
    }
    pos_weights = {name: value.to(device) for name, value in pos_weights.items()}
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    state_indices = torch.from_numpy(state_permutations).long().to(device)
    pair_indices = torch.from_numpy(pair_permutations).long().to(device)
    train_loader = module.make_loader(data["train"], scaler, args_global.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args_global.batch_size, False)
    module.seed_everything(123); initial = BranchingGraphRSSM(**config).to(device)
    smoke = smoke_tests(module, initial, train_loader, data["train"]["context_states"], scaler,
                        state_permutations, pair_permutations, context_steps, groups, weights,
                        pos_weights, args_global.temperature, args_global.responsibility_mode, device)
    states = {}; candidates = []
    for seed in [int(value) for value in args_global.seeds.split(",") if value.strip()]:
        candidate_path = (Path(args_global.out).parent
                          / f"candidate_{args_global.responsibility_mode.replace('-', '_')}_seed_{seed}.pt")
        if args_global.resume_candidates and candidate_path.exists():
            candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
            if candidate["model_config"] != config:
                raise ValueError(f"incompatible resumed candidate: {candidate_path}")
            state, summary = candidate["model_state_dict"], candidate["summary"]
            print(f"reused seed={seed} candidate from {candidate_path}")
        else:
            state, summary = train_seed(
                module, seed, train_loader, validation_loader, config, context_steps, groups,
                weights, pos_weights, state_indices, pair_indices, args_global.temperature,
                args_global.responsibility_mode, args_global.epochs, args_global.patience,
                args_global.learning_rate, device,
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": state, "summary": summary, "model_config": config}, candidate_path)
        states[seed] = state; candidates.append(summary)
    selected = min(candidates, key=lambda row: row["best_validation_selection"]); seed = int(selected["seed"])
    model = BranchingGraphRSSM(**config).to(device); model.load_state_dict(states[seed]); model.eval()
    evaluation = evaluate_validation(
        module, model, data["validation"], scaler, metadata, device, contract_script,
        args_global.responsibility_mode,
    )
    # Audit on CPU to avoid CUDA reduction-order noise in this numerical symmetry check.
    audit_model = deepcopy(model).to(torch.device("cpu"))
    equivariance = deterministic_equivariance(
        module, audit_model, data["validation"], scaler, state_permutations, pair_permutations,
        torch.device("cpu")
    )
    baseline = json.loads(Path(args_global.baseline_contract).read_text())
    future = evaluation["future_graph"]
    branch_lm = evaluation["branch_interpretation"]["mean_branch_lm_probability"]
    gates = {
        "noncollapsed_best_branch_effective_count": future["best_branch_effective_count"] >= 1.2,
        "nontrivial_pairwise_diversity": future["mean_pairwise_branch_mae"] >= 0.005,
        "positive_oracle_coverage_gain": future["oracle_gain_over_expected_mae"] > 0.005,
        "outcome_semantic_separation": (
            abs(branch_lm[1] - branch_lm[0]) >= 0.1
            if args_global.responsibility_mode == "lm-outcome" else None
        ),
        "positive_outcome_active_state_specialization": (
            evaluation["branch_interpretation"]["active_state_mae_by_branch_and_outcome"][1][1]
            < evaluation["branch_interpretation"]["active_state_mae_by_branch_and_outcome"][0][1]
            if args_global.responsibility_mode == "lm-outcome" else None
        ),
        "positive_outcome_edge_specialization": (
            evaluation["branch_interpretation"]["future_edge_ap_by_branch_and_outcome"][1][1]
            > evaluation["branch_interpretation"]["future_edge_ap_by_branch_and_outcome"][0][1]
            if args_global.responsibility_mode == "lm-outcome" else None
        ),
        "equivariance": max(equivariance.values()) < 1e-5,
        "causality": smoke["causal_context_and_branch_max_delta"] == 0.0,
    }
    required_gates = [value for value in gates.values() if value is not None]
    report = {
        "scope": f"explicit two-branch graph RSSM ({args_global.responsibility_mode}); V3 train/validation only",
        "test_access": "V3 test is not loaded by this script",
        "runtime": device_summary(device), "model_config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": {"weights": weights, "mixture_temperature": args_global.temperature,
                     "responsibility_mode": args_global.responsibility_mode,
                     "selection_rule": "minimum validation joint trajectory/outcome + weighted edge/semantic objective",
                     "candidates": candidates, "selected_seed": seed,
                     "selected_epoch": selected["best_epoch"]},
        "smoke_tests": smoke, "equivariance": equivariance,
        "validation": evaluation,
        "eligibility_gates": gates, "eligible_for_fresh_evaluation": all(required_gates),
        "ordinary_rssm_validation_reference": {
            "expected_mae": baseline["future_graph"]["expected_forecast_mae"],
            "oracle_mae": baseline["future_graph"]["oracle_best_branch_mae"],
            "pairwise_diversity": baseline["future_graph"]["mean_pairwise_branch_mae"],
            "lm_brier": baseline["exact_lm_within_30_seconds"]["calibration"]["brier"],
            "lm_ece": baseline["exact_lm_within_30_seconds"]["calibration"]["expected_calibration_error"],
        },
        "limitations": [
            "Only V3 validation is evaluated; no generalization result exists yet.",
            "Two branches are exchangeable and should be interpreted by decoded behavior, not fixed index name.",
            "Oracle best-branch MAE is coverage, not deployable prediction error.",
            "Validation ECE uses 90 correlated windows and is not deployment calibration evidence.",
            "No attacker/defender action variable is observed.",
        ],
    }
    output = Path(args_global.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    checkpoint = Path(args_global.checkpoint_out); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": cpu_state_dict(model.state_dict()), "model_config": config,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_metadata": metadata, "training": report["training"],
                "validation_only": True, "v3_test_loaded": False}, checkpoint)
    print("\n===== two-branch validation summary =====")
    print(f"seed={seed} expected/oracle={future['expected_forecast_mae']:.3f}/{future['oracle_best_branch_mae']:.3f} "
          f"diversity={future['mean_pairwise_branch_mae']:.3f} effective={future['best_branch_effective_count']:.2f} "
          f"LM Brier/ECE={evaluation['exact_lm']['calibration']['brier']:.3f}/"
          f"{evaluation['exact_lm']['calibration']['expected_calibration_error']:.3f} gates={gates}")
    print(f"metrics -> {output}\ncheckpoint -> {checkpoint}"); return 0


args_global: argparse.Namespace
if __name__ == "__main__": sys.exit(main())
