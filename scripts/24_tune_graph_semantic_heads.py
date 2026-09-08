#!/usr/bin/env python3
"""Recover graph-RSSM semantic performance while freezing learned dynamics."""
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
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.graph_rssm import GraphRSSM  # noqa: E402


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def checkpoint_scaler(checkpoint: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
    scaler.scale_ = np.asarray(checkpoint["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_); scaler.n_samples_seen_ = 1
    return scaler


def freeze_to_semantic_heads(model: GraphRSSM) -> list[str]:
    prefixes = ("lm_head.", "technique_head.", "pair_embedding.", "pair_head.")
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(prefixes)
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise AssertionError("no semantic parameters selected")
    return trainable


def train_seed(module: Any, seed: int, initial_state: dict[str, torch.Tensor],
               model_config: dict[str, int], train_loader: Any, validation_loader: Any,
               context_steps: int, groups: list[slice], weights: dict[str, float],
               pos_weights: dict[str, torch.Tensor], state_indices: torch.Tensor,
               edge_indices: torch.Tensor, epochs: int, patience: int, learning_rate: float
               ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    module.seed_everything(seed)
    model = GraphRSSM(**model_config); model.load_state_dict(initial_state)
    trainable_names = freeze_to_semantic_heads(model)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    best_state = deepcopy(model.state_dict()); best_validation = math.inf
    best_epoch = 0; stale = 0; epochs_run = 0
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for raw_batch in train_loader:
            batch = module.augment_batch(raw_batch, state_indices, edge_indices)
            optimizer.zero_grad(set_to_none=True)
            terms = module.loss_terms(
                model, batch, context_steps, groups, weights, pos_weights, sample=True
            )
            terms["total"].backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 10.0
            )
            optimizer.step(); total += float(terms["total"].detach()) * len(batch[0])
            count += len(batch[0])
        validation = module.validate_objective(
            model, validation_loader, context_steps, groups, weights, pos_weights,
            torch.device("cpu"), selection_mode="semantics",
        )
        epochs_run = epoch
        if validation["selection"] < best_validation - 1e-5:
            best_validation = validation["selection"]; best_epoch = epoch
            best_state = deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"seed={seed} epoch={epoch} train_sem={total/count:.4f} "
                  f"val_sem={validation['selection']:.4f}")
        if stale >= patience:
            break
    return best_state, {"seed": seed, "best_epoch": best_epoch,
                        "best_validation_semantic_loss": best_validation,
                        "epochs_run": epochs_run, "trainable_parameter_names": trainable_names}


def validation_score(module: Any, pair_audit: Any, model: GraphRSSM,
                     validation: dict[str, np.ndarray], scaler: StandardScaler,
                     metadata: dict[str, Any], state_permutations: np.ndarray,
                     pair_permutations: np.ndarray, mc_samples: int, batch_size: int,
                     seed: int) -> tuple[float, dict[str, Any]]:
    mc = module.mc_predictions(
        model, validation["context_states"], scaler, torch.device("cpu"),
        mc_samples, batch_size, seed,
    )
    lm = mc["lm"].mean(axis=0)
    lm_target = validation["lateral_movement_within_horizon"].astype(int)
    lm_ap = float(average_precision_score(lm_target, lm))
    technique = mc["technique"].mean(axis=0)
    technique_target = validation["future_techniques"].max(axis=1).astype(int)
    technique_aps = [float(average_precision_score(technique_target[:, index], technique[:, index]))
                     for index in range(technique.shape[1])]
    audit, _ = pair_audit.evaluate_split(
        module, model, scaler, validation, state_permutations, pair_permutations,
        mc_samples, batch_size, seed + 10_000, threshold=None,
    )
    pair_ap = audit["summary"]["pair_micro_average_precision"]["mean"]
    pair_top1 = audit["summary"]["pair_top1_accuracy_any_true_pair"]["mean"]
    score = lm_ap + float(np.mean(technique_aps)) + pair_ap + pair_top1
    return score, {"lm_average_precision": lm_ap, "technique_average_precisions": technique_aps,
                   "mean_technique_average_precision": float(np.mean(technique_aps)),
                   "permutation_mean_pair_average_precision": pair_ap,
                   "permutation_mean_pair_top1": pair_top1,
                   "combined_selection_score": score,
                   "equivariance": audit["summary"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--initial-checkpoint", default=str(ROOT / "models/mvp_v2_graph_rssm.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v2/graph_rssm/semantic_tuning.json"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/mvp_v2_graph_rssm_semantic.pt"))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--mc-validation", type=int, default=20)
    ap.add_argument("--mc-test", type=int, default=100)
    args = ap.parse_args()

    module = load_script("cyberwm_rssm_training", ROOT / "scripts/15_train_rssm.py")
    ablation = load_script("cyberwm_rssm_ablation", ROOT / "scripts/18_run_rssm_ablations.py")
    pair_audit = load_script("cyberwm_pair_audit", ROOT / "scripts/21_audit_pair_equivariance.py")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    scaler = checkpoint_scaler(checkpoint)
    model_config = checkpoint["model_config"]
    initial_state = checkpoint["model_state_dict"]
    data = {split: module.load_split(sequence_dir, split) for split in ["train", "validation"]}
    context_steps = metadata["context_states"]
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, metadata["state_feature_count"])]
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    state_indices = torch.from_numpy(state_permutations).long()
    edge_indices = torch.from_numpy(pair_permutations).long()
    weights = {"future_state": 0.0, "reconstruction": 0.0, "edge": 0.0,
               "kl": 0.0, "lm": 0.2, "technique": 0.1, "pair": 0.1,
               "free_nats": 0.0}
    pos_weights = {
        "edge": module.positive_weight(data["train"]["future_edge_presence"]),
        "lm": module.positive_weight(data["train"]["lateral_movement_within_horizon"]),
        "technique": module.positive_weight(data["train"]["future_techniques"].max(axis=1)),
        "pair": module.positive_weight(data["train"]["future_lateral_edges"].max(axis=1)),
    }
    train_loader = module.make_loader(data["train"], scaler, args.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args.batch_size, False)
    semantic_prefixes = ("lm_head.", "technique_head.", "pair_embedding.", "pair_head.")
    frozen_reference = {name: value.detach().clone() for name, value in initial_state.items()
                        if not name.startswith(semantic_prefixes)}

    candidates = []; states = {}
    print("===== frozen graph semantic-head training =====")
    for index, seed in enumerate(int(value) for value in args.seeds.split(",") if value.strip()):
        state, training = train_seed(
            module, seed, initial_state, model_config, train_loader, validation_loader,
            context_steps, groups, weights, pos_weights, state_indices, edge_indices,
            args.epochs, args.patience, args.learning_rate,
        )
        for name, value in state.items():
            if name in frozen_reference and not torch.equal(value, frozen_reference[name]):
                raise AssertionError(f"frozen graph tensor changed: {name}")
        model = GraphRSSM(**model_config); model.load_state_dict(state); model.eval()
        score, validation = validation_score(
            module, pair_audit, model, data["validation"], scaler, metadata,
            state_permutations, pair_permutations, args.mc_validation,
            args.batch_size, 900_000 + index,
        )
        candidates.append({"seed": seed, "training": training, "validation": validation,
                           "selection_score": score})
        states[seed] = state
        print(f"seed={seed} val_LM_AP={validation['lm_average_precision']:.3f} "
              f"tech_AP={validation['mean_technique_average_precision']:.3f} "
              f"pair_AP/top1={validation['permutation_mean_pair_average_precision']:.3f}/"
              f"{validation['permutation_mean_pair_top1']:.3f} score={score:.3f}")
    selected = max(candidates, key=lambda row: row["selection_score"])
    selected_seed = int(selected["seed"])
    model = GraphRSSM(**model_config); model.load_state_dict(states[selected_seed]); model.eval()

    # Test becomes available only after semantic seed selection is fixed.
    data["test"] = module.load_split(sequence_dir, "test")
    dynamics, mc = ablation.dynamics_metrics(
        module, model, data, scaler, metadata, args.mc_test, args.batch_size,
        selected_seed + 1_000_000,
    )
    semantics = ablation.internal_semantic_metrics(module, mc, data, metadata, sequence_dir)
    threshold = semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    validation_audit, _ = pair_audit.evaluate_split(
        module, model, scaler, data["validation"], state_permutations, pair_permutations,
        args.mc_test, args.batch_size, selected_seed + 1_010_000, threshold,
    )
    test_audit, _ = pair_audit.evaluate_split(
        module, model, scaler, data["test"], state_permutations, pair_permutations,
        args.mc_test, args.batch_size, selected_seed + 1_020_000, threshold,
    )
    probability = {split: mc[split]["lm"].mean(axis=0) for split in ["validation", "test"]}
    labels = {split: data[split]["lateral_movement_within_horizon"].astype(int)
              for split in ["validation", "test"]}
    sample_manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    episode_frame, episode_summary, sample_frame = module.episode_alert_report(
        sample_manifest, probability, labels, threshold, Path(args.episodes_dir)
    )
    initial_metrics = json.loads((ROOT / "outputs/mvp_v2/graph_rssm/metrics.json").read_text())
    report = {
        "scope": "frozen dynamics with validation-selected graph semantic heads",
        "test_isolation": "test loaded only after semantic seed selection",
        "frozen_tensor_check": "all non-semantic tensors bit-identical to initial graph checkpoint",
        "candidates": candidates, "selected_seed": selected_seed,
        "selection_rule": "validation LM AP + mean ATT&CK AP + permutation-mean pair AP + top-1",
        **dynamics, **semantics,
        "equivariance": {"validation": validation_audit, "test": test_audit},
        "episode_alert_summary": episode_summary,
        "initial_graph_reference": {
            "state_prediction": initial_metrics["state_prediction"]["test"],
            "future_edge_presence": initial_metrics["future_edge_presence"]["test"],
            "future_lateral_movement": initial_metrics["future_lateral_movement"]["test"],
            "future_techniques": {name: value["test"]
                                  for name, value in initial_metrics["future_techniques"].items()},
            "future_lateral_pair_ranking": initial_metrics["future_lateral_pair_ranking"]["test"],
        },
        "limitations": [
            "Only 24 controlled episodes and 12 LM events.",
            "The same validation split selects head checkpoint, seed, and LM threshold.",
            "Test episodes have been inspected in earlier experiments.",
            "Frozen head recovery cannot alter graph dynamics or edge presence decoding.",
            "One hundred Monte Carlo rollouts reduce but do not eliminate sampling noise.",
        ],
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    episode_frame.to_csv(output.parent / "semantic_episode_alerts.csv", index=False)
    sample_frame.to_csv(output.parent / "semantic_sample_predictions.csv", index=False)
    checkpoint_output = Path(args.checkpoint_out); checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_metadata": metadata,
                "selection": {"seed": selected_seed, "rule": report["selection_rule"],
                              "score": selected["selection_score"]}}, checkpoint_output)

    state = report["state_prediction"]["test"]
    lm = report["future_lateral_movement"]["test"]
    pre = report["future_lateral_movement"]["test_before_any_observed_lateral"]
    pair = test_audit["summary"]["pair_top1_accuracy_any_true_pair"]
    print("\n===== graph semantic-head test summary =====")
    print(f"seed={selected_seed} state={state['normalized_mae']:.3f} "
          f"edge_AP={report['future_edge_presence']['test']['average_precision']:.3f} "
          f"LM_F1/AP={lm['f1']:.3f}/{lm['average_precision']:.3f} "
          f"pre_F1={pre['f1']:.3f} pair_identity="
          f"{report['future_lateral_pair_ranking']['test']['top1_accuracy_any_true_lm_pair']:.3f} "
          f"pair_perm={pair['minimum']:.3f}-{pair['maximum']:.3f}")
    print(f"metrics -> {output}\ncheckpoint -> {checkpoint_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
