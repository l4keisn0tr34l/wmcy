#!/usr/bin/env python3
"""Pretrain graph RSSM dynamics on UNSW, then fine-tune on controlled lab truth."""
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
from src.cyberwm.device import cpu_state_dict, device_summary, resolve_device  # noqa: E402


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def public_split(module: Any, directory: Path, split: str, horizon: int,
                 technique_count: int, pair_count: int) -> dict[str, np.ndarray]:
    data = module.load_split(directory, split); count = len(data["context_states"])
    expected = {"context_states", "future_states", "future_edge_presence"}
    if set(data) != expected: raise ValueError(f"unexpected public arrays: {sorted(data)}")
    data["lateral_movement_within_horizon"] = np.zeros(count, dtype=np.float32)
    data["future_techniques"] = np.zeros((count, horizon, technique_count), dtype=np.float32)
    data["future_lateral_edges"] = np.zeros((count, horizon, pair_count), dtype=np.float32)
    return data


def train_graph(module: Any, initial_state: dict[str, torch.Tensor], seed: int,
                train_loader: Any, validation_loader: Any, config: dict[str, int],
                context_steps: int, groups: list[slice], weights: dict[str, float],
                pos_weights: dict[str, torch.Tensor], state_indices: torch.Tensor,
                edge_indices: torch.Tensor, epochs: int, patience: int, learning_rate: float,
                selection_mode: str, device: torch.device = torch.device("cpu")
                ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    module.seed_everything(seed)
    model = GraphRSSM(**config).to(device); model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best = cpu_state_dict(model.state_dict()); best_value = math.inf; best_epoch = 0; stale = 0
    first_validation = None; history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0; count = 0
        for raw_batch in train_loader:
            batch = module.augment_batch(module.move(raw_batch, device), state_indices, edge_indices)
            optimizer.zero_grad(set_to_none=True)
            terms = module.loss_terms(model, batch, context_steps, groups, weights, pos_weights, sample=True)
            terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step(); total += float(terms["total"].detach()) * len(batch[0]); count += len(batch[0])
        validation = module.validate_objective(
            model, validation_loader, context_steps, groups, weights, pos_weights,
            device, selection_mode=selection_mode,
        )
        if first_validation is None: first_validation = validation["selection"]
        if validation["selection"] < best_value - 1e-5:
            best_value = validation["selection"]; best_epoch = epoch
            best = cpu_state_dict(model.state_dict()); stale = 0
        else: stale += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"seed={seed} epoch={epoch} train={total/count:.4f} val_{selection_mode}={validation['selection']:.4f}")
            history.append({"epoch": epoch, "train": total/count, "validation": validation["selection"]})
        if stale >= patience: break
    return best, {"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch,
                  "initial_validation_selection": first_validation,
                  "best_validation_selection": best_value, "history": history}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-sequences", default=str(ROOT / "outputs/unsw/graph_sequences"))
    ap.add_argument("--lab-sequences", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp_v2/graph_rssm"))
    ap.add_argument("--pretrain-checkpoint", default=str(ROOT / "models/unsw_graph_rssm_pretrained.pt"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/mvp_v2_graph_rssm_unsw_pretrained.pt"))
    ap.add_argument("--pretrain-epochs", type=int, default=160); ap.add_argument("--pretrain-patience", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=240); ap.add_argument("--finetune-patience", type=int, default=45)
    ap.add_argument("--public-batch-size", type=int, default=64); ap.add_argument("--lab-batch-size", type=int, default=32)
    ap.add_argument("--pretrain-learning-rate", type=float, default=3e-4)
    ap.add_argument("--finetune-learning-rate", type=float, default=2e-4)
    ap.add_argument("--seeds", default="7,17,27"); ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = resolve_device(args.device)
    print("device:", device_summary(device))
    module = load_script("rssm_training", ROOT / "scripts/15_train_rssm.py")
    graph_script = load_script("graph_training", ROOT / "scripts/23_train_graph_rssm.py")
    ablation = load_script("rssm_ablation", ROOT / "scripts/18_run_rssm_ablations.py")
    pair_audit = load_script("pair_audit", ROOT / "scripts/21_audit_pair_equivariance.py")
    public_dir, lab_dir = Path(args.public_sequences), Path(args.lab_sequences)
    public_metadata = json.loads((public_dir / "feature_metadata.json").read_text())
    lab_metadata = json.loads((lab_dir / "feature_metadata.json").read_text())
    if public_metadata["target_policy"] != "future observable state and edge presence only; no UNSW labels loaded":
        raise ValueError("public target policy is not leakage-safe")
    if public_metadata["profiles"][0]["capture_group"] == public_metadata["profiles"][1]["capture_group"]:
        raise ValueError("public train and validation share a capture group")
    config = graph_script.graph_config(lab_metadata); context_steps = lab_metadata["context_states"]
    public_data = {split: public_split(module, public_dir, split, config["horizon"],
                                       config["technique_count"], config["pair_count"])
                   for split in ["train", "validation"]}
    public_scaler = fit_graph_feature_scaler(
        public_data["train"]["context_states"], config["global_size"], config["node_size"],
        config["node_count"], config["edge_size"], config["pair_count"],
    )
    permutations, edge_permutations = module.state_and_edge_permutation_indices(lab_metadata)
    state_indices = torch.from_numpy(permutations).long().to(device)
    edge_indices = torch.from_numpy(edge_permutations).long().to(device)
    global_width = config["global_size"]; node_width = config["node_size"] * config["node_count"]
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, lab_metadata["state_feature_count"])]
    public_weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25,
                      "kl": 0.01, "lm": 0.0, "technique": 0.0, "pair": 0.0, "free_nats": 0.0}
    public_pos = {"edge": module.positive_weight(public_data["train"]["future_edge_presence"]),
                  "lm": torch.ones(1), "technique": torch.ones(config["technique_count"]),
                  "pair": torch.ones(config["pair_count"])}
    public_pos = {name: value.to(device) for name, value in public_pos.items()}
    public_train_loader = module.make_loader(public_data["train"], public_scaler, args.public_batch_size, True)
    public_validation_loader = module.make_loader(public_data["validation"], public_scaler, args.public_batch_size, False)
    module.seed_everything(7); initial_model = GraphRSSM(**config).to(device)
    smoke = graph_script.smoke_tests(
        module, initial_model, public_train_loader, permutations, edge_permutations,
        public_scaler, public_data["train"]["context_states"], context_steps, groups,
        public_weights, public_pos, device,
    )
    print("===== UNSW observable-only dynamics pretraining =====")
    pretrained_state, pretraining = train_graph(
        module, initial_model.state_dict(), 7, public_train_loader, public_validation_loader,
        config, context_steps, groups, public_weights, public_pos, state_indices, edge_indices,
        args.pretrain_epochs, args.pretrain_patience, args.pretrain_learning_rate, "dynamics", device,
    )
    if pretraining["best_validation_selection"] >= pretraining["initial_validation_selection"]:
        raise AssertionError("public validation dynamics did not improve")
    pretrain_path = Path(args.pretrain_checkpoint); pretrain_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": cpu_state_dict(pretrained_state), "model_config": config,
                "scaler_mean": public_scaler.mean_, "scaler_scale": public_scaler.scale_,
                "feature_metadata": public_metadata, "training": pretraining,
                "semantic_supervision": False}, pretrain_path)

    # Refit all preprocessing on lab training contexts. Lab test remains unopened.
    lab_data = {split: module.load_split(lab_dir, split) for split in ["train", "validation"]}
    lab_scaler = fit_graph_feature_scaler(
        lab_data["train"]["context_states"], config["global_size"], config["node_size"],
        config["node_count"], config["edge_size"], config["pair_count"],
    )
    lab_weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25,
                   "kl": 0.01, "lm": 0.2, "technique": 0.1, "pair": 0.1, "free_nats": 0.0}
    lab_pos = {"edge": module.positive_weight(lab_data["train"]["future_edge_presence"]),
               "lm": module.positive_weight(lab_data["train"]["lateral_movement_within_horizon"]),
               "technique": module.positive_weight(lab_data["train"]["future_techniques"].max(axis=1)),
               "pair": module.positive_weight(lab_data["train"]["future_lateral_edges"].max(axis=1))}
    lab_pos = {name: value.to(device) for name, value in lab_pos.items()}
    train_loader = module.make_loader(lab_data["train"], lab_scaler, args.lab_batch_size, True)
    validation_loader = module.make_loader(lab_data["validation"], lab_scaler, args.lab_batch_size, False)
    candidates = []; states = {}
    print("===== controlled-lab fine-tuning; validation-only selection =====")
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        state, summary = train_graph(
            module, pretrained_state, seed, train_loader, validation_loader, config,
            context_steps, groups, lab_weights, lab_pos, state_indices, edge_indices,
            args.finetune_epochs, args.finetune_patience, args.finetune_learning_rate, "joint", device,
        )
        states[seed] = state; candidates.append(summary)
    selected = min(candidates, key=lambda row: row["best_validation_selection"]); selected_seed = int(selected["seed"])
    model = GraphRSSM(**config).to(device); model.load_state_dict(states[selected_seed]); model.eval()

    # Selection is now frozen. The repeatedly inspected V2 test is diagnostic only.
    lab_data["test"] = module.load_split(lab_dir, "test")
    dynamics, mc = ablation.dynamics_metrics(
        module, model, lab_data, lab_scaler, lab_metadata, args.mc_samples,
        args.lab_batch_size, selected_seed + 900_000, device=device,
    )
    semantics = ablation.internal_semantic_metrics(module, mc, lab_data, lab_metadata, lab_dir)
    threshold = semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    validation_audit, _ = pair_audit.evaluate_split(
        module, model, lab_scaler, lab_data["validation"], permutations, edge_permutations,
        args.mc_samples, args.lab_batch_size, selected_seed + 910_000, threshold, device=device,
    )
    test_audit, _ = pair_audit.evaluate_split(
        module, model, lab_scaler, lab_data["test"], permutations, edge_permutations,
        args.mc_samples, args.lab_batch_size, selected_seed + 920_000, threshold, device=device,
    )
    probabilities = {split: mc[split]["lm"].mean(axis=0) for split in ["validation", "test"]}
    labels = {split: lab_data[split]["lateral_movement_within_horizon"].astype(int)
              for split in ["validation", "test"]}
    sample_manifest = pd.read_csv(lab_dir / "sample_manifest.csv")
    episode_frame, episode_summary, sample_frame = module.episode_alert_report(
        sample_manifest, probabilities, labels, threshold, Path(args.episodes_dir)
    )
    baseline = json.loads((Path(args.out_dir) / "metrics.json").read_text())
    metrics = {
        "scope": "UNSW observable-dynamics pretraining followed by controlled V2 graph RSSM fine-tuning",
        "public_truth_loaded": False,
        "test_isolation": "V2 test loaded only after public checkpoint and lab validation selected fine-tune seed/epoch; test is diagnostic because previously inspected",
        "public_data": {"train_capture_group": public_metadata["profiles"][0]["capture_group"],
                        "validation_capture_group": public_metadata["profiles"][1]["capture_group"],
                        "train_samples": len(public_data["train"]["context_states"]),
                        "validation_samples": len(public_data["validation"]["context_states"]),
                        "semantic_loss_weights": {k: public_weights[k] for k in ["lm", "technique", "pair"]}},
        "model_config": config, "parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime": device_summary(device),
        "smoke_tests": smoke, "pretraining": pretraining,
        "fine_tuning": {"candidates": candidates, "selected_seed": selected_seed,
                         "weights": lab_weights, "selection_rule": "lab validation joint objective"},
        **dynamics, **semantics,
        "equivariance": {"validation": validation_audit, "test": test_audit},
        "episode_alert_summary": episode_summary,
        "reference_graph_rssm": {
            "validation_state_mae": baseline["state_prediction"]["validation"]["normalized_mae"],
            "test_state_mae": baseline["state_prediction"]["test"]["normalized_mae"],
            "test_active_mae": baseline["state_prediction"]["test"]["active_normalized_mae"],
            "validation_edge_ap": baseline["future_edge_presence"]["validation"]["average_precision"],
            "test_edge_ap": baseline["future_edge_presence"]["test"]["average_precision"],
            "validation_lm_ap": baseline["future_lateral_movement"]["validation"]["average_precision"],
            "test_lm_ap": baseline["future_lateral_movement"]["test"]["average_precision"],
        },
        "limitations": [
            "UNSW has only two connected capture groups and large January/February intensity shift.",
            "UNSW TCP flag-count fields are unavailable and zero-marked, unlike lab telemetry.",
            "Public rosters are context-selected induced triads rather than complete enterprise inventories.",
            "V2 test has been inspected repeatedly; transfer conclusions should prioritize validation and require a fresh holdout.",
        ],
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    result_path = out / "public_pretraining.json"
    result_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    episode_frame.to_csv(out / "public_pretraining_episode_alerts.csv", index=False)
    sample_frame.to_csv(out / "public_pretraining_sample_predictions.csv", index=False)
    checkpoint = Path(args.checkpoint_out); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": cpu_state_dict(model.state_dict()), "model_config": config,
                "scaler_mean": lab_scaler.mean_, "scaler_scale": lab_scaler.scale_,
                "feature_metadata": lab_metadata, "pretraining": pretraining,
                "fine_tuning": metrics["fine_tuning"]}, checkpoint)
    state = metrics["state_prediction"]; edge = metrics["future_edge_presence"]
    lm = metrics["future_lateral_movement"]
    print("\n===== public-pretrained graph RSSM diagnostic =====")
    print(f"validation state={state['validation']['normalized_mae']:.3f} edge_AP={edge['validation']['average_precision']:.3f} LM_AP={lm['validation']['average_precision']:.3f}")
    print(f"test state={state['test']['normalized_mae']:.3f} active={state['test']['active_normalized_mae']:.3f} edge_AP={edge['test']['average_precision']:.3f} LM_F1/AP={lm['test']['f1']:.3f}/{lm['test']['average_precision']:.3f}")
    print(f"metrics -> {result_path}\ncheckpoints -> {pretrain_path}, {checkpoint}")
    return 0


if __name__ == "__main__": sys.exit(main())
