#!/usr/bin/env python3
"""Compare frozen, unfrozen, joint, and zero-KL RSSM training regimes on V2."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_rssm_module() -> Any:
    path = ROOT / "scripts/15_train_rssm.py"
    spec = importlib.util.spec_from_file_location("cyberwm_rssm_training", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dynamics_metrics(module: Any, model: torch.nn.Module, data: dict[str, dict[str, np.ndarray]],
                     scaler: StandardScaler, metadata: dict[str, Any], mc_samples: int,
                     batch_size: int, seed: int, device: torch.device | None = None
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    device = device or torch.device("cpu")
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    groups = {"global": slice(0, global_width), "node": slice(global_width, global_width + node_width),
              "edge": slice(global_width + node_width, metadata["state_feature_count"])}
    output: dict[str, Any] = {"state_prediction": {}, "future_edge_presence": {}}
    mc_outputs = {}
    for offset, split in enumerate(["validation", "test"]):
        mc = module.mc_predictions(model, data[split]["context_states"], scaler, device,
                                   mc_samples, batch_size, seed + 1000 * (offset + 1))
        mc_outputs[split] = mc
        state_mean = mc["state"].mean(axis=0)
        context = module.normalize(scaler, data[split]["context_states"])
        future = module.normalize(scaler, data[split]["future_states"])
        output["state_prediction"][split] = module.state_metrics(
            state_mean, future, context, data[split]["future_states"], groups
        )
        edge_probability = mc["edge"].mean(axis=0)
        edge_target = data[split]["future_edge_presence"]
        output["future_edge_presence"][split] = {
            "average_precision": float(average_precision_score(edge_target.ravel(), edge_probability.ravel())),
            "positive_rate": float(edge_target.mean()),
        }
    test_mc = mc_outputs["test"]
    state_std = test_mc["state"].std(axis=0)
    state_mean = test_mc["state"].mean(axis=0)
    test_future = module.normalize(scaler, data["test"]["future_states"])
    error = np.abs(state_mean - test_future)
    active = data["test"]["future_states"][:, :, 0] > 0
    output["uncertainty"] = {
        "mc_samples": mc_samples,
        "mean_normalized_state_std": float(state_std.mean()),
        "active_mean_normalized_state_std": float(state_std[active].mean()),
        "quiet_mean_normalized_state_std": float(state_std[~active].mean()),
        "state_std_absolute_error_correlation": float(np.corrcoef(
            state_std.mean(axis=2).ravel(), error.mean(axis=2).ravel())[0, 1]),
    }
    return output, mc_outputs


def internal_semantic_metrics(module: Any, mc_outputs: dict[str, dict[str, np.ndarray]],
                              data: dict[str, dict[str, np.ndarray]], metadata: dict[str, Any],
                              sequences_dir: Path) -> dict[str, Any]:
    probability = {split: {name: values.mean(axis=0) for name, values in mc_outputs[split].items()}
                   for split in ["validation", "test"]}
    labels = {split: data[split]["lateral_movement_within_horizon"].astype(int) for split in data}
    threshold = module.best_f1_threshold(labels["validation"], probability["validation"]["lm"])
    lm = {
        "threshold_selected_on_validation": threshold,
        "validation": module.binary_metrics(labels["validation"], probability["validation"]["lm"], threshold),
        "test": module.binary_metrics(labels["test"], probability["test"]["lm"], threshold),
    }
    sample_manifest = pd.read_csv(sequences_dir / "sample_manifest.csv")
    for split in ["validation", "test"]:
        audit = sample_manifest[sample_manifest.split.eq(split)].reset_index(drop=True)
        mask = audit.lateral_movement_already_observed.eq(0).to_numpy()
        lm[f"{split}_before_any_observed_lateral"] = module.binary_metrics(
            labels[split][mask], probability[split]["lm"][mask], threshold
        )
    techniques = {}
    for index, name in enumerate(metadata["technique_targets"]):
        techniques[name] = {}
        for split in ["validation", "test"]:
            target = data[split]["future_techniques"].max(axis=1)[:, index].astype(int)
            techniques[name][split] = module.binary_metrics(
                target, probability[split]["technique"][:, index], 0.5
            )
    pairs = {}
    for split in ["validation", "test"]:
        target = data[split]["future_lateral_edges"].max(axis=1)
        positive = target.max(axis=1) > 0
        top = probability[split]["pair"][positive].argmax(axis=1)
        correct = target[positive][np.arange(positive.sum()), top] > 0
        pairs[split] = {"positive_samples": int(positive.sum()),
                        "top1_accuracy_any_true_lm_pair": float(correct.mean())}
    return {"future_lateral_movement": lm, "future_techniques": techniques,
            "future_lateral_pair_ranking": pairs}


def host_sensitivity_internal(module: Any, model: torch.nn.Module, test: dict[str, np.ndarray],
                              scaler: StandardScaler, state_permutations: np.ndarray,
                              threshold: float, batch_size: int, seed: int,
                              device: torch.device | None = None) -> dict[str, Any]:
    device = device or torch.device("cpu")
    rows, scores = [], []
    labels = test["lateral_movement_within_horizon"].astype(int)
    for index, order in enumerate(state_permutations):
        mc = module.mc_predictions(model, test["context_states"][..., order], scaler,
                                   device, 5, batch_size, seed + index)
        probability = mc["lm"].mean(axis=0)
        rows.append({"permutation_index": index,
                     "average_precision": float(average_precision_score(labels, probability)),
                     "f1": float(f1_score(labels, probability >= threshold, zero_division=0))})
        scores.append(probability)
    stack = np.stack(scores)
    return {"rows": rows,
            "mean_per_sample_probability_range": float(np.mean(stack.max(axis=0) - stack.min(axis=0))),
            "max_per_sample_probability_range": float(np.max(stack.max(axis=0) - stack.min(axis=0)))}


def train_candidates(module: Any, name: str, seeds: list[int], train_loader: Any,
                     validation_loader: Any, model_config: dict[str, int], context_steps: int,
                     groups: list[slice], weights: dict[str, float], pos_weights: dict[str, torch.Tensor],
                     state_indices: torch.Tensor, edge_indices: torch.Tensor, epochs: int, patience: int,
                     learning_rate: float, initial_state: dict[str, torch.Tensor] | None = None,
                     selection_mode: str = "dynamics", freeze_backbone: bool = False,
                     ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    states = {}; summaries = []
    print(f"\n===== {name} =====")
    for seed in seeds:
        state, summary = module.train_seed(
            seed, train_loader, validation_loader, model_config, context_steps, groups,
            weights, pos_weights, state_indices, edge_indices, torch.device("cpu"),
            epochs, patience, learning_rate, initial_state=initial_state,
            selection_mode=selection_mode, freeze_backbone=freeze_backbone,
        )
        states[seed] = state; summaries.append(summary)
    selected = min(summaries, key=lambda row: row["best_validation_selection"])
    return states[int(selected["seed"])], {
        "selected_seed": selected["seed"], "selected_epoch": selected["best_epoch"],
        "selected_validation": selected["best_validation_selection"],
        "candidates": [{key: value for key, value in row.items() if key != "history"} for row in summaries],
    }


def save_checkpoint(path: Path, model: torch.nn.Module, model_config: dict[str, int],
                    scaler: StandardScaler, metadata: dict[str, Any], training: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_metadata": metadata, "training": training}, path)


def compact_variant(dynamics: dict[str, Any], semantics: dict[str, Any],
                    training: dict[str, Any], sensitivity: dict[str, Any]) -> dict[str, Any]:
    return {"training": training, **dynamics, **semantics, "host_permutation_sensitivity": sensitivity}


def main() -> int:
    global args_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v2/rssm/ablations.json"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--head-epochs", type=int, default=1000)
    ap.add_argument("--head-patience", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--mc-samples", type=int, default=20)
    args_global = ap.parse_args()

    module = load_rssm_module()
    torch.set_num_threads(min(8, __import__("os").cpu_count() or 1))
    sequence_dir = Path(args_global.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    data = {split: module.load_split(sequence_dir, split) for split in ["train", "validation", "test"]}
    state_permutations, edge_permutations = module.state_and_edge_permutation_indices(metadata)
    scaler = module.fit_context_scaler(data["train"]["context_states"])
    context_steps = metadata["context_states"]; horizon = metadata["future_horizon_states"]
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, metadata["state_feature_count"])]
    model_config = {"observation_size": metadata["state_feature_count"], "embedding_size": 64,
                    "deterministic_size": 64, "stochastic_size": 16, "horizon": horizon,
                    "technique_count": len(metadata["technique_targets"]),
                    "pair_count": len(metadata["directed_edge_slots"])}
    base_weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25,
                    "kl": 0.1, "lm": 0.2, "technique": 0.1, "pair": 0.1,
                    "free_nats": 1.0}
    train_techniques = data["train"]["future_techniques"].max(axis=1)
    train_pairs = data["train"]["future_lateral_edges"].max(axis=1)
    pos_weights = {"edge": module.positive_weight(data["train"]["future_edge_presence"]),
                   "lm": module.positive_weight(data["train"]["lateral_movement_within_horizon"]),
                   "technique": module.positive_weight(train_techniques),
                   "pair": module.positive_weight(train_pairs)}
    state_indices = torch.from_numpy(state_permutations).long()
    edge_indices = torch.from_numpy(edge_permutations).long()
    train_loader = module.make_loader(data["train"], scaler, args_global.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args_global.batch_size, False)
    seeds = [int(value) for value in args_global.seeds.split(",") if value.strip()]

    self_weights = {**base_weights, "lm": 0.0, "technique": 0.0, "pair": 0.0}
    self_state, self_training = train_candidates(
        module, "self-supervised pretraining", seeds, train_loader, validation_loader,
        model_config, context_steps, groups, self_weights, pos_weights, state_indices,
        edge_indices, args_global.epochs, args_global.patience, args_global.learning_rate,
    )
    self_model = module.CompactRSSM(**model_config); self_model.load_state_dict(self_state); self_model.eval()

    semantic_weights = {**base_weights, "future_state": 0.0, "reconstruction": 0.0,
                        "edge": 0.0, "kl": 0.0}
    frozen_state, frozen_training = train_candidates(
        module, "frozen semantic heads", seeds, train_loader, validation_loader,
        model_config, context_steps, groups, semantic_weights, pos_weights, state_indices,
        edge_indices, args_global.head_epochs, args_global.head_patience,
        args_global.learning_rate, initial_state=self_state, selection_mode="semantics",
        freeze_backbone=True,
    )
    frozen_model = module.CompactRSSM(**model_config)
    frozen_model.load_state_dict(frozen_state); frozen_model.eval()
    frozen_dynamics, frozen_mc = dynamics_metrics(
        module, frozen_model, data, scaler, metadata, args_global.mc_samples,
        args_global.batch_size, int(frozen_training["selected_seed"]),
    )
    frozen_semantics = internal_semantic_metrics(module, frozen_mc, data, metadata, sequence_dir)
    frozen_threshold = frozen_semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    frozen_sensitivity = host_sensitivity_internal(
        module, frozen_model, data["test"], scaler, state_permutations, frozen_threshold,
        args_global.batch_size, int(frozen_training["selected_seed"]) + 3000,
    )

    fine_state, fine_training = train_candidates(
        module, "self-supervised then unfrozen", seeds, train_loader, validation_loader,
        model_config, context_steps, groups, base_weights, pos_weights, state_indices,
        edge_indices, args_global.epochs, args_global.patience, args_global.learning_rate,
        initial_state=self_state, selection_mode="joint",
    )
    fine_model = module.CompactRSSM(**model_config); fine_model.load_state_dict(fine_state); fine_model.eval()
    fine_dynamics, fine_mc = dynamics_metrics(module, fine_model, data, scaler, metadata,
                                              args_global.mc_samples, args_global.batch_size,
                                              int(fine_training["selected_seed"]))
    fine_semantics = internal_semantic_metrics(module, fine_mc, data, metadata, sequence_dir)
    fine_threshold = fine_semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    fine_sensitivity = host_sensitivity_internal(
        module, fine_model, data["test"], scaler, state_permutations, fine_threshold,
        args_global.batch_size, int(fine_training["selected_seed"]) + 5000,
    )

    matched_joint_state, matched_joint_training = train_candidates(
        module, "joint from scratch with matched selection", seeds, train_loader,
        validation_loader, model_config, context_steps, groups, base_weights, pos_weights,
        state_indices, edge_indices, args_global.epochs, args_global.patience,
        args_global.learning_rate, selection_mode="joint",
    )
    matched_joint_model = module.CompactRSSM(**model_config)
    matched_joint_model.load_state_dict(matched_joint_state); matched_joint_model.eval()
    matched_joint_dynamics, matched_joint_mc = dynamics_metrics(
        module, matched_joint_model, data, scaler, metadata, args_global.mc_samples,
        args_global.batch_size, int(matched_joint_training["selected_seed"]),
    )
    matched_joint_semantics = internal_semantic_metrics(
        module, matched_joint_mc, data, metadata, sequence_dir
    )
    matched_joint_threshold = matched_joint_semantics["future_lateral_movement"][
        "threshold_selected_on_validation"
    ]
    matched_joint_sensitivity = host_sensitivity_internal(
        module, matched_joint_model, data["test"], scaler, state_permutations,
        matched_joint_threshold, args_global.batch_size,
        int(matched_joint_training["selected_seed"]) + 6000,
    )

    no_kl_weights = {**base_weights, "kl": 0.0}
    no_kl_state, no_kl_training = train_candidates(
        module, "joint zero-KL", seeds, train_loader, validation_loader,
        model_config, context_steps, groups, no_kl_weights, pos_weights, state_indices,
        edge_indices, args_global.epochs, args_global.patience, args_global.learning_rate,
        selection_mode="joint",
    )
    no_kl_model = module.CompactRSSM(**model_config); no_kl_model.load_state_dict(no_kl_state); no_kl_model.eval()
    no_kl_dynamics, no_kl_mc = dynamics_metrics(module, no_kl_model, data, scaler, metadata,
                                                args_global.mc_samples, args_global.batch_size,
                                                int(no_kl_training["selected_seed"]))
    no_kl_semantics = internal_semantic_metrics(module, no_kl_mc, data, metadata, sequence_dir)
    no_kl_threshold = no_kl_semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    no_kl_sensitivity = host_sensitivity_internal(
        module, no_kl_model, data["test"], scaler, state_permutations, no_kl_threshold,
        args_global.batch_size, int(no_kl_training["selected_seed"]) + 7000,
    )

    joint_metrics = json.loads((ROOT / "outputs/mvp_v2/rssm/metrics.json").read_text())
    joint = {
        "training": {"selected_seed": joint_metrics["training_config"]["selected_seed"],
                     "selection_rule": joint_metrics["training_config"]["selection_rule"]},
        "state_prediction": joint_metrics["state_prediction"],
        "future_edge_presence": joint_metrics["future_edge_presence"],
        "uncertainty": joint_metrics["uncertainty"],
        "future_lateral_movement": joint_metrics["future_lateral_movement"],
        "future_techniques": joint_metrics["future_techniques"],
        "future_lateral_pair_ranking": joint_metrics["future_lateral_pair_ranking"],
        "host_permutation_sensitivity": joint_metrics["host_permutation_sensitivity"],
    }

    variants = {
        "published_joint_dynamics_selected": joint,
        "frozen_two_stage": compact_variant(frozen_dynamics, frozen_semantics, frozen_training,
                                             frozen_sensitivity),
        "pretrained_then_unfrozen": compact_variant(fine_dynamics, fine_semantics, fine_training,
                                                     fine_sensitivity),
        "joint_from_scratch_matched_selection": compact_variant(
            matched_joint_dynamics, matched_joint_semantics, matched_joint_training,
            matched_joint_sensitivity,
        ),
        "joint_zero_kl": compact_variant(no_kl_dynamics, no_kl_semantics, no_kl_training,
                                         no_kl_sensitivity),
    }
    report = {
        "scope": "V2 RSSM representation-training ablation; identical episode splits and architecture",
        "selection_policy": "all checkpoints and LM thresholds selected on validation only; frozen heads use semantic selection and jointly trained variants use matched joint selection",
        "model_config": model_config,
        "regimes": {
            "published_joint_dynamics_selected": "previously reported joint model selected only on validation state+edge dynamics",
            "frozen_two_stage": "telemetry + edge + KL pretraining; identical internal linear semantic heads trained with RSSM backbone frozen",
            "pretrained_then_unfrozen": "same self-supervised checkpoint followed by full joint fine-tuning",
            "joint_from_scratch_matched_selection": "joint random initialization with the same joint validation selection as fine-tuning",
            "joint_zero_kl": "joint random initialization and matched selection with KL weight zero",
        },
        "variants": variants,
        "limitations": [
            "Only 24 episodes and 12 LM events.",
            "Overlapping sequence windows are correlated.",
            "Fine-tuning and frozen heads start each seed from one validation-selected self-supervised checkpoint.",
            "The published joint reference used dynamics-only selection; the matched joint variant controls this difference.",
            "No DANN, transformer, ensemble, or action variable is tested.",
        ],
    }

    models_dir = Path(args_global.models_dir)
    save_checkpoint(models_dir / "mvp_v2_rssm_selfsupervised.pt", self_model, model_config,
                    scaler, metadata, self_training)
    save_checkpoint(models_dir / "mvp_v2_rssm_frozen_heads.pt", frozen_model, model_config,
                    scaler, metadata, frozen_training)
    save_checkpoint(models_dir / "mvp_v2_rssm_pretrained_unfrozen.pt", fine_model, model_config,
                    scaler, metadata, fine_training)
    save_checkpoint(models_dir / "mvp_v2_rssm_joint_matched.pt", matched_joint_model,
                    model_config, scaler, metadata, matched_joint_training)
    save_checkpoint(models_dir / "mvp_v2_rssm_zero_kl.pt", no_kl_model, model_config,
                    scaler, metadata, no_kl_training)
    output = Path(args_global.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nTEST SUMMARY")
    for name, variant in variants.items():
        state = variant["state_prediction"]["test"]
        lm = variant["future_lateral_movement"]["test"]
        pre = variant["future_lateral_movement"]["test_before_any_observed_lateral"]
        print(f"{name:28s} state={state['normalized_mae']:.3f} active={state['active_normalized_mae']:.3f} "
              f"LM_F1={lm['f1']:.3f} LM_AP={lm.get('average_precision', float('nan')):.3f} "
              f"pre_F1={pre['f1']:.3f} edge_AP={variant['future_edge_presence']['test']['average_precision']:.3f} "
              f"pair={variant['future_lateral_pair_ranking']['test']['top1_accuracy_any_true_lm_pair']:.3f}")
    print(f"ablations -> {output}")
    return 0


args_global: argparse.Namespace
if __name__ == "__main__":
    sys.exit(main())
