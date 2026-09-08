#!/usr/bin/env python3
"""Train invariant LM/ATT&CK readouts from graph-RSSM imagined future states."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.graph_rssm import GraphRSSM  # noqa: E402


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validation_readout_score(module: Any, model: GraphRSSM, validation: dict[str, np.ndarray],
                             scaler: Any, mc_samples: int, batch_size: int, seed: int
                             ) -> tuple[float, dict[str, Any]]:
    mc = module.mc_predictions(model, validation["context_states"], scaler, torch.device("cpu"),
                               mc_samples, batch_size, seed)
    lm_target = validation["lateral_movement_within_horizon"].astype(int)
    lm_ap = float(average_precision_score(lm_target, mc["lm"].mean(axis=0)))
    technique_target = validation["future_techniques"].max(axis=1).astype(int)
    technique_probability = mc["technique"].mean(axis=0)
    technique_ap = [float(average_precision_score(technique_target[:, index],
                                                  technique_probability[:, index]))
                    for index in range(technique_probability.shape[1])]
    score = lm_ap + float(np.mean(technique_ap))
    return score, {"lm_average_precision": lm_ap,
                   "technique_average_precisions": technique_ap,
                   "mean_technique_average_precision": float(np.mean(technique_ap)),
                   "selection_score": score}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--initial-checkpoint", default=str(ROOT / "models/mvp_v2_graph_rssm.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v2/graph_rssm/rich_semantic.json"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/mvp_v2_graph_rssm_rich_semantic.pt"))
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
    semantic_training = load_script("cyberwm_graph_semantic", ROOT / "scripts/24_tune_graph_semantic_heads.py")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    scaler = semantic_training.checkpoint_scaler(checkpoint)
    model_config = {**checkpoint["model_config"], "semantic_from_decoded": True}
    module.seed_everything(25_700)
    initial_model = GraphRSSM(**model_config)
    transferable = {name: value for name, value in checkpoint["model_state_dict"].items()
                    if not name.startswith(("lm_head.", "technique_head."))}
    incompatible = initial_model.load_state_dict(transferable, strict=False)
    expected_missing = {"lm_head.0.weight", "lm_head.0.bias", "lm_head.2.weight", "lm_head.2.bias",
                        "technique_head.0.weight", "technique_head.0.bias",
                        "technique_head.2.weight", "technique_head.2.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise AssertionError(f"unexpected rich-readout initialization mismatch: {incompatible}")
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    readout_prefixes = ("lm_head.", "technique_head.")
    frozen_reference = {name: value.detach().clone() for name, value in initial_state.items()
                        if not name.startswith(readout_prefixes)}

    # Test stays unloaded until validation selects a seed.
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
               "kl": 0.0, "lm": 0.2, "technique": 0.1, "pair": 0.0,
               "free_nats": 0.0}
    pos_weights = {
        "edge": module.positive_weight(data["train"]["future_edge_presence"]),
        "lm": module.positive_weight(data["train"]["lateral_movement_within_horizon"]),
        "technique": module.positive_weight(data["train"]["future_techniques"].max(axis=1)),
        "pair": module.positive_weight(data["train"]["future_lateral_edges"].max(axis=1)),
    }
    train_loader = module.make_loader(data["train"], scaler, args.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args.batch_size, False)

    candidates = []; states = {}
    print("===== decoded-future invariant semantic readout =====")
    for index, seed in enumerate(int(value) for value in args.seeds.split(",") if value.strip()):
        state, training = semantic_training.train_seed(
            module, seed, initial_state, model_config, train_loader, validation_loader,
            context_steps, groups, weights, pos_weights, state_indices, edge_indices,
            args.epochs, args.patience, args.learning_rate,
        )
        for name, value in state.items():
            if name in frozen_reference and not torch.equal(value, frozen_reference[name]):
                raise AssertionError(f"frozen graph tensor changed: {name}")
        model = GraphRSSM(**model_config); model.load_state_dict(state); model.eval()
        score, validation = validation_readout_score(
            module, model, data["validation"], scaler, args.mc_validation,
            args.batch_size, 1_100_000 + index,
        )
        candidates.append({"seed": seed, "training": training,
                           "validation": validation, "selection_score": score})
        states[seed] = state
        print(f"seed={seed} validation LM_AP={validation['lm_average_precision']:.3f} "
              f"technique_AP={validation['mean_technique_average_precision']:.3f} "
              f"score={score:.3f}")
    selected = max(candidates, key=lambda row: row["selection_score"])
    selected_seed = int(selected["seed"])
    model = GraphRSSM(**model_config); model.load_state_dict(states[selected_seed]); model.eval()

    data["test"] = module.load_split(sequence_dir, "test")
    dynamics, mc = ablation.dynamics_metrics(
        module, model, data, scaler, metadata, args.mc_test, args.batch_size,
        selected_seed + 1_200_000,
    )
    semantics = ablation.internal_semantic_metrics(module, mc, data, metadata, sequence_dir)
    threshold = semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    validation_audit, _ = pair_audit.evaluate_split(
        module, model, scaler, data["validation"], state_permutations, pair_permutations,
        args.mc_test, args.batch_size, selected_seed + 1_210_000, threshold,
    )
    test_audit, _ = pair_audit.evaluate_split(
        module, model, scaler, data["test"], state_permutations, pair_permutations,
        args.mc_test, args.batch_size, selected_seed + 1_220_000, threshold,
    )
    probability = {split: mc[split]["lm"].mean(axis=0) for split in ["validation", "test"]}
    labels = {split: data[split]["lateral_movement_within_horizon"].astype(int)
              for split in ["validation", "test"]}
    sample_manifest = pd.read_csv(sequence_dir / "sample_manifest.csv")
    episode_frame, episode_summary, sample_frame = module.episode_alert_report(
        sample_manifest, probability, labels, threshold, Path(args.episodes_dir)
    )
    initial = json.loads((ROOT / "outputs/mvp_v2/graph_rssm/metrics.json").read_text())
    report = {
        "scope": "invariant semantic readout from decoded graph-RSSM future states",
        "input_to_readout": "per-step global + node mean/max + directed-edge mean/max",
        "test_isolation": "test loaded only after validation selected seed",
        "frozen_tensor_check": "all non-LM/non-technique tensors bit-identical",
        "candidates": candidates, "selected_seed": selected_seed,
        "selection_rule": "validation LM AP + mean ATT&CK AP",
        **dynamics, **semantics,
        "equivariance": {"validation": validation_audit, "test": test_audit},
        "episode_alert_summary": episode_summary,
        "initial_graph_reference": {
            "state_prediction": initial["state_prediction"]["test"],
            "future_edge_presence": initial["future_edge_presence"]["test"],
            "future_lateral_movement": initial["future_lateral_movement"]["test"],
            "future_techniques": {name: values["test"]
                                  for name, values in initial["future_techniques"].items()},
        },
        "limitations": [
            "Only 24 episodes and 12 LM events.",
            "Decoded future features may carry reconstruction error into semantic readout.",
            "Validation/test have been inspected in prior experiments.",
            "The test split is loaded after selection but is no longer a fresh project holdout.",
            "Uncertainty remains uncalibrated.",
        ],
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    episode_frame.to_csv(output.parent / "rich_semantic_episode_alerts.csv", index=False)
    sample_frame.to_csv(output.parent / "rich_semantic_sample_predictions.csv", index=False)
    checkpoint_output = Path(args.checkpoint_out); checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_config": model_config,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_metadata": metadata,
                "selection": {"seed": selected_seed, "rule": report["selection_rule"],
                              "score": selected["selection_score"]}}, checkpoint_output)

    state = report["state_prediction"]["test"]
    lm = report["future_lateral_movement"]["test"]
    pre = report["future_lateral_movement"]["test_before_any_observed_lateral"]
    print("\n===== rich graph semantic test summary =====")
    print(f"seed={selected_seed} state={state['normalized_mae']:.3f} "
          f"edge_AP={report['future_edge_presence']['test']['average_precision']:.3f} "
          f"LM_F1/AP={lm['f1']:.3f}/{lm['average_precision']:.3f} "
          f"pre_F1/AP={pre['f1']:.3f}/{pre['average_precision']:.3f} "
          f"pair={report['future_lateral_pair_ranking']['test']['top1_accuracy_any_true_lm_pair']:.3f}")
    print(f"metrics -> {output}\ncheckpoint -> {checkpoint_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
