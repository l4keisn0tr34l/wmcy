#!/usr/bin/env python3
"""Train and evaluate a permutation-equivariant graph RSSM on V2."""
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.graph_rssm import GraphRSSM, fit_graph_feature_scaler  # noqa: E402


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def graph_config(metadata: dict[str, Any]) -> dict[str, int]:
    return {
        "global_size": len(metadata["global_feature_names"]),
        "node_size": len(metadata["node_feature_names"]),
        "node_count": len(metadata["node_slots"]),
        "edge_size": len(metadata["edge_feature_names"]),
        "pair_count": len(metadata["directed_edge_slots"]),
        "horizon": metadata["future_horizon_states"],
        "global_hidden": 64, "node_hidden": 32, "edge_hidden": 32,
        "stochastic_size": 16, "local_embedding": 32,
        "technique_count": len(metadata["technique_targets"]),
    }


def smoke_tests(module: Any, model: GraphRSSM, train_loader: Any,
                state_permutations: np.ndarray, pair_permutations: np.ndarray,
                scaler: Any, raw_context: np.ndarray, context_steps: int,
                groups: list[slice], weights: dict[str, float],
                pos_weights: dict[str, torch.Tensor]) -> dict[str, float]:
    raw_batch = next(iter(train_loader))
    batch = tuple(tensor[:16] for tensor in raw_batch)
    full = torch.cat([batch[0], batch[1]], dim=1)
    model.eval()
    with torch.no_grad():
        posterior_a, future_a = model(full, context_steps, sample=False)
        changed = full.clone(); changed[:, context_steps:] += 123.0
        posterior_b, _ = model(changed, context_steps, sample=False)
        causal_delta = max(float((posterior_a[name][:, :context_steps]
                                  - posterior_b[name][:, :context_steps]).abs().max())
                           for name in ["h", "node_h", "edge_h", "z"])
        if causal_delta != 0.0:
            raise AssertionError(f"future changed graph context posterior: {causal_delta}")
        if future_a["decoded"].shape != (len(full), full.shape[1] - context_steps, 141):
            raise AssertionError("unexpected graph future shape")

        base_context = torch.from_numpy(module.normalize(scaler, raw_context[:8]))
        base = model.forecast(base_context, sample=False)
        equivariance_delta = 0.0
        for state_order, pair_order in zip(state_permutations, pair_permutations):
            permuted_context = torch.from_numpy(
                module.normalize(scaler, raw_context[:8, :, state_order])
            )
            predicted = model.forecast(permuted_context, sample=False)
            deltas = [
                float((predicted["decoded"] - base["decoded"][..., state_order]).abs().max()),
                float((predicted["edge_logits"] - base["edge_logits"][..., pair_order]).abs().max()),
                float((predicted["pair_logits"] - base["pair_logits"][..., pair_order]).abs().max()),
                float((predicted["lm_logits"] - base["lm_logits"]).abs().max()),
                float((predicted["technique_logits"] - base["technique_logits"]).abs().max()),
            ]
            equivariance_delta = max(equivariance_delta, *deltas)
        if equivariance_delta > 2e-6:
            raise AssertionError(f"graph model is not equivariant: {equivariance_delta}")

    tiny = deepcopy(model); optimizer = torch.optim.Adam(tiny.parameters(), lr=1e-3)
    losses = []; tiny.train()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        terms = module.loss_terms(tiny, batch, context_steps, groups, weights, pos_weights, sample=True)
        if not torch.isfinite(terms["total"]):
            raise AssertionError("non-finite graph smoke loss")
        terms["total"].backward(); nn.utils.clip_grad_norm_(tiny.parameters(), 10.0)
        optimizer.step(); losses.append(float(terms["total"].detach()))
    first = float(np.mean(losses[:5])); last = float(np.mean(losses[-5:]))
    if last >= first:
        raise AssertionError(f"graph tiny-overfit did not decrease: {first} -> {last}")
    return {"causal_context_max_delta": causal_delta,
            "deterministic_equivariance_max_delta": equivariance_delta,
            "overfit_first_loss": first, "overfit_last_loss": last}


def train_seed(module: Any, seed: int, train_loader: Any, validation_loader: Any,
               config: dict[str, int], context_steps: int, groups: list[slice],
               weights: dict[str, float], pos_weights: dict[str, torch.Tensor],
               state_indices: torch.Tensor, edge_indices: torch.Tensor,
               max_epochs: int, patience: int, learning_rate: float
               ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    module.seed_everything(seed)
    model = GraphRSSM(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = deepcopy(model.state_dict()); best_validation = math.inf
    best_epoch = 0; stale = 0; epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0; count = 0
        for raw_batch in train_loader:
            batch = module.augment_batch(raw_batch, state_indices, edge_indices)
            optimizer.zero_grad(set_to_none=True)
            terms = module.loss_terms(
                model, batch, context_steps, groups, weights, pos_weights, sample=True
            )
            terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step(); total += float(terms["total"].detach()) * len(batch[0])
            count += len(batch[0])
        validation = module.validate_objective(
            model, validation_loader, context_steps, groups, weights, pos_weights,
            torch.device("cpu"), selection_mode="joint",
        )
        epochs_run = epoch
        if validation["selection"] < best_validation - 1e-5:
            best_validation = validation["selection"]; best_epoch = epoch
            best_state = deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"seed={seed} epoch={epoch} train={total/count:.4f} "
                  f"val_joint={validation['selection']:.4f}")
        if stale >= patience:
            break
    return best_state, {"seed": seed, "best_epoch": best_epoch,
                        "best_validation_selection": best_validation,
                        "epochs_run": epochs_run}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp_v2/graph_rssm"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/mvp_v2_graph_rssm.pt"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--mc-samples", type=int, default=20)
    args = ap.parse_args()

    module = load_script("cyberwm_rssm_training", ROOT / "scripts/15_train_rssm.py")
    ablation = load_script("cyberwm_rssm_ablation", ROOT / "scripts/18_run_rssm_ablations.py")
    pair_audit = load_script("cyberwm_pair_audit", ROOT / "scripts/21_audit_pair_equivariance.py")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    # Test is not loaded until validation has selected a seed.
    data = {split: module.load_split(sequence_dir, split) for split in ["train", "validation"]}
    config = graph_config(metadata)
    scaler = fit_graph_feature_scaler(
        data["train"]["context_states"], config["global_size"], config["node_size"],
        config["node_count"], config["edge_size"], config["pair_count"],
    )
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    state_indices = torch.from_numpy(state_permutations).long()
    edge_indices = torch.from_numpy(pair_permutations).long()
    context_steps = metadata["context_states"]
    global_width = config["global_size"]
    node_width = config["node_size"] * config["node_count"]
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, metadata["state_feature_count"])]
    weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25,
               "kl": 0.01, "lm": 0.2, "technique": 0.1, "pair": 0.1,
               "free_nats": 0.0}
    pos_weights = {
        "edge": module.positive_weight(data["train"]["future_edge_presence"]),
        "lm": module.positive_weight(data["train"]["lateral_movement_within_horizon"]),
        "technique": module.positive_weight(data["train"]["future_techniques"].max(axis=1)),
        "pair": module.positive_weight(data["train"]["future_lateral_edges"].max(axis=1)),
    }
    train_loader = module.make_loader(data["train"], scaler, args.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args.batch_size, False)
    initial_model = GraphRSSM(**config)
    smoke = smoke_tests(
        module, initial_model, train_loader, state_permutations, pair_permutations,
        scaler, data["train"]["context_states"], context_steps, groups, weights, pos_weights,
    )
    candidates = []; states = {}
    print("===== graph RSSM training; validation-only seed selection =====")
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        state, summary = train_seed(
            module, seed, train_loader, validation_loader, config, context_steps,
            groups, weights, pos_weights, state_indices, edge_indices,
            args.epochs, args.patience, args.learning_rate,
        )
        states[seed] = state; candidates.append(summary)
    selected = min(candidates, key=lambda row: row["best_validation_selection"])
    selected_seed = int(selected["seed"])
    model = GraphRSSM(**config); model.load_state_dict(states[selected_seed]); model.eval()

    # Freeze selection before loading/evaluating test.
    data["test"] = module.load_split(sequence_dir, "test")
    dynamics, mc = ablation.dynamics_metrics(
        module, model, data, scaler, metadata, args.mc_samples, args.batch_size,
        selected_seed + 800_000,
    )
    semantics = ablation.internal_semantic_metrics(module, mc, data, metadata, sequence_dir)
    threshold = semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    validation_audit, _ = pair_audit.evaluate_split(
        module, model, scaler, data["validation"], state_permutations, pair_permutations,
        args.mc_samples, args.batch_size, selected_seed + 810_000, threshold,
    )
    test_audit, _ = pair_audit.evaluate_split(
        module, model, scaler, data["test"], state_permutations, pair_permutations,
        args.mc_samples, args.batch_size, selected_seed + 820_000, threshold,
    )
    probabilities = {split: mc[split]["lm"].mean(axis=0) for split in ["validation", "test"]}
    labels = {split: data[split]["lateral_movement_within_horizon"].astype(int)
              for split in ["validation", "test"]}
    sample_manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    episode_frame, episode_summary, sample_frame = module.episode_alert_report(
        sample_manifest, probabilities, labels, threshold, Path(args.episodes_dir)
    )
    reference = json.loads((ROOT / "outputs/mvp_v2/rssm/metrics.json").read_text())
    kl_reference = json.loads((ROOT / "outputs/mvp_v2/rssm/kl_tuning.json").read_text())
    metrics = {
        "scope": "permutation-equivariant graph RSSM on equal-duration V2",
        "test_isolation": "test loaded only after validation selected seed",
        "model_config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training": {"weights": weights, "candidates": candidates,
                     "selected_seed": selected_seed,
                     "selection_rule": "validation joint dynamics + semantic weighted objective"},
        "smoke_tests": smoke,
        **dynamics, **semantics,
        "equivariance": {"validation": validation_audit, "test": test_audit},
        "episode_alert_summary": episode_summary,
        "references": {
            "published_flattened": {
                "state_prediction": reference["state_prediction"]["test"],
                "future_edge_presence": reference["future_edge_presence"]["test"],
                "future_lateral_movement": reference["future_lateral_movement"]["test"],
            },
            "kl_tuned_flattened": kl_reference["final_test"],
        },
        "limitations": [
            "Only 24 episodes and 12 LM events; overlapping windows are correlated.",
            "This first graph model is trained only on V2, without public pretraining.",
            "The fixed three-host roster is known, although model operations are shared/equivariant.",
            "Validation/test episodes have been inspected in prior flattened-model experiments.",
            "Monte Carlo spread is not calibrated uncertainty.",
        ],
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    episode_frame.to_csv(out / "episode_alerts.csv", index=False)
    sample_frame.to_csv(out / "sample_predictions.csv", index=False)
    np.savez_compressed(
        out / "predictions.npz",
        validation_state_draws=mc["validation"]["state"], test_state_draws=mc["test"]["state"],
        validation_lm_draws=mc["validation"]["lm"], test_lm_draws=mc["test"]["lm"],
        validation_pair_draws=mc["validation"]["pair"], test_pair_draws=mc["test"]["pair"],
    )
    checkpoint = Path(args.checkpoint_out); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": config,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_metadata": metadata, "training": metrics["training"]}, checkpoint)

    state = metrics["state_prediction"]["test"]
    lm = metrics["future_lateral_movement"]["test"]
    pre = metrics["future_lateral_movement"]["test_before_any_observed_lateral"]
    pair = metrics["future_lateral_pair_ranking"]["test"]
    eq = test_audit["summary"]
    print("\n===== graph RSSM test summary =====")
    print(f"seed={selected_seed} state={state['normalized_mae']:.3f} "
          f"active={state['active_normalized_mae']:.3f} "
          f"edge_AP={metrics['future_edge_presence']['test']['average_precision']:.3f} "
          f"LM_F1/AP={lm['f1']:.3f}/{lm['average_precision']:.3f} "
          f"pre_F1={pre['f1']:.3f} pair_identity={pair['top1_accuracy_any_true_lm_pair']:.3f} "
          f"pair_perm_mean={eq['pair_top1_accuracy_any_true_pair']['mean']:.3f} "
          f"pair_eq_mae={eq['pair_equivariance_mae']['mean']:.6f}")
    print(f"metrics -> {out / 'metrics.json'}\ncheckpoint -> {checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
