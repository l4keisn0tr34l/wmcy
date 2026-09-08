#!/usr/bin/env python3
"""Compare scratch and UNSW-pretrained graph RSSMs on fresh paired V3."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.graph_rssm import GraphRSSM, fit_graph_feature_scaler  # noqa: E402

SEMANTIC_PREFIXES = ("lm_head.", "technique_head.", "pair_embedding.", "pair_head.")


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def paired_initial_states(module: Any, config: dict[str, int], pretrained: dict[str, torch.Tensor],
                          seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    module.seed_everything(seed); scratch = GraphRSSM(**config).state_dict()
    transfer = deepcopy(scratch); transferred = []
    for name, value in pretrained.items():
        if name.startswith(SEMANTIC_PREFIXES): continue
        if name not in transfer or transfer[name].shape != value.shape:
            raise ValueError(f"incompatible public parameter: {name}")
        transfer[name] = value.detach().clone(); transferred.append(name)
    semantic_delta = max(
        float((scratch[name] - transfer[name]).abs().max())
        for name in scratch if name.startswith(SEMANTIC_PREFIXES)
    )
    if semantic_delta != 0.0: raise AssertionError("semantic initializations differ")
    dynamics_changed = max(
        float((scratch[name] - transfer[name]).abs().max()) for name in transferred
    )
    if dynamics_changed == 0.0: raise AssertionError("public dynamics were not transferred")
    return scratch, transfer, {"semantic_initialization_max_delta": semantic_delta,
                               "transferred_parameter_tensors": len(transferred),
                               "dynamics_initialization_max_delta": dynamics_changed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v3/sequences"))
    ap.add_argument("--episode-manifest", default=str(ROOT / "outputs/mvp_v3/episode_manifest.csv"))
    ap.add_argument("--episodes-dir", default=str(ROOT / "lab/episodes"))
    ap.add_argument("--public-checkpoint", default=str(ROOT / "models/unsw_graph_rssm_pretrained.pt"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs/mvp_v3/graph_rssm"))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--seeds", default="7,17,27"); ap.add_argument("--mc-samples", type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    module = load_script("rssm_training_v3", ROOT / "scripts/15_train_rssm.py")
    graph_script = load_script("graph_training_v3", ROOT / "scripts/23_train_graph_rssm.py")
    pretrain_script = load_script("public_pretraining_v3", ROOT / "scripts/28_pretrain_graph_rssm_unsw.py")
    ablation = load_script("rssm_ablation_v3", ROOT / "scripts/18_run_rssm_ablations.py")
    pair_audit = load_script("pair_audit_v3", ROOT / "scripts/21_audit_pair_equivariance.py")
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    episode_manifest = pd.read_csv(args.episode_manifest, dtype={"episode_id": str})
    split_ids = {split: set(episode_manifest.loc[episode_manifest.split.eq(split), "episode_id"])
                 for split in ["train", "validation", "test"]}
    if any(split_ids[a] & split_ids[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("V3 episode splits overlap")
    if split_ids["test"] != {f"lab_{number:03d}" for number in range(67, 73)}:
        raise ValueError("unexpected sealed V3 test IDs")

    public_checkpoint = torch.load(args.public_checkpoint, map_location="cpu", weights_only=False)
    if public_checkpoint.get("semantic_supervision") is not False:
        raise ValueError("public checkpoint does not declare zero semantic supervision")
    config = graph_script.graph_config(metadata)
    if config != public_checkpoint["model_config"]:
        raise ValueError("public/lab graph configurations differ")

    # Only train and validation are loaded before all model/threshold selection.
    data = {split: module.load_split(sequence_dir, split) for split in ["train", "validation"]}
    scaler = fit_graph_feature_scaler(
        data["train"]["context_states"], config["global_size"], config["node_size"],
        config["node_count"], config["edge_size"], config["pair_count"],
    )
    permutations, edge_permutations = module.state_and_edge_permutation_indices(metadata)
    state_indices = torch.from_numpy(permutations).long(); edge_indices = torch.from_numpy(edge_permutations).long()
    context_steps = metadata["context_states"]; global_width = config["global_size"]
    node_width = config["node_size"] * config["node_count"]
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, metadata["state_feature_count"])]
    weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25,
               "kl": 0.01, "lm": 0.2, "technique": 0.1, "pair": 0.1, "free_nats": 0.0}
    pos_weights = {
        "edge": module.positive_weight(data["train"]["future_edge_presence"]),
        "lm": module.positive_weight(data["train"]["lateral_movement_within_horizon"]),
        "technique": module.positive_weight(data["train"]["future_techniques"].max(axis=1)),
        "pair": module.positive_weight(data["train"]["future_lateral_edges"].max(axis=1)),
    }
    train_loader = module.make_loader(data["train"], scaler, args.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args.batch_size, False)
    smoke_model = GraphRSSM(**config)
    smoke = graph_script.smoke_tests(
        module, smoke_model, train_loader, permutations, edge_permutations, scaler,
        data["train"]["context_states"], context_steps, groups, weights, pos_weights,
    )

    states: dict[str, dict[int, dict[str, torch.Tensor]]] = {"scratch": {}, "unsw_pretrained": {}}
    summaries: dict[str, list[dict[str, Any]]] = {"scratch": [], "unsw_pretrained": []}
    initialization_audit = {}
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        scratch_initial, transfer_initial, audit = paired_initial_states(
            module, config, public_checkpoint["model_state_dict"], seed
        )
        initialization_audit[str(seed)] = audit
        for regime, initial in [("scratch", scratch_initial), ("unsw_pretrained", transfer_initial)]:
            print(f"===== V3 {regime} seed {seed}; validation-only selection =====")
            state, summary = pretrain_script.train_graph(
                module, initial, seed, train_loader, validation_loader, config, context_steps,
                groups, weights, pos_weights, state_indices, edge_indices, args.epochs,
                args.patience, args.learning_rate, "joint",
            )
            states[regime][seed] = state; summaries[regime].append(summary)

    selected = {}; models = {}; validation_mc = {}; frozen_thresholds = {}
    regime_seed_offsets = {"scratch": 1_000_000, "unsw_pretrained": 1_100_000}
    for regime in states:
        choice = min(summaries[regime], key=lambda row: row["best_validation_selection"])
        seed = int(choice["seed"]); selected[regime] = choice
        model = GraphRSSM(**config); model.load_state_dict(states[regime][seed]); model.eval(); models[regime] = model
        base_seed = seed + regime_seed_offsets[regime]
        validation_mc[regime] = module.mc_predictions(
            model, data["validation"]["context_states"], scaler, torch.device("cpu"),
            args.mc_samples, args.batch_size, base_seed + 1000,
        )
        labels = data["validation"]["lateral_movement_within_horizon"].astype(int)
        frozen_thresholds[regime] = module.best_f1_threshold(
            labels, validation_mc[regime]["lm"].mean(axis=0)
        )
    selection_record = {
        regime: {"selected_seed": int(selected[regime]["seed"]),
                 "selected_epoch": int(selected[regime]["best_epoch"]),
                 "selected_validation_joint": selected[regime]["best_validation_selection"],
                 "lm_threshold_selected_on_validation": frozen_thresholds[regime]}
        for regime in selected
    }
    print("===== both regimes and LM thresholds frozen; loading sealed V3 test once =====")
    data["test"] = module.load_split(sequence_dir, "test")

    results = {}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.models_dir); model_dir.mkdir(parents=True, exist_ok=True)
    for regime, model in models.items():
        seed = int(selected[regime]["seed"]); base_seed = seed + regime_seed_offsets[regime]
        dynamics, mc = ablation.dynamics_metrics(
            module, model, data, scaler, metadata, args.mc_samples, args.batch_size, base_seed,
        )
        if not np.array_equal(mc["validation"]["lm"], validation_mc[regime]["lm"]):
            raise AssertionError("validation MC changed after threshold freeze")
        semantics = ablation.internal_semantic_metrics(module, mc, data, metadata, sequence_dir)
        calculated_threshold = semantics["future_lateral_movement"]["threshold_selected_on_validation"]
        if abs(calculated_threshold - frozen_thresholds[regime]) > 1e-12:
            raise AssertionError("semantic evaluation changed frozen threshold")
        validation_audit, _ = pair_audit.evaluate_split(
            module, model, scaler, data["validation"], permutations, edge_permutations,
            args.mc_samples, args.batch_size, base_seed + 20_000, frozen_thresholds[regime],
        )
        test_audit, _ = pair_audit.evaluate_split(
            module, model, scaler, data["test"], permutations, edge_permutations,
            args.mc_samples, args.batch_size, base_seed + 30_000, frozen_thresholds[regime],
        )
        probabilities = {split: mc[split]["lm"].mean(axis=0) for split in ["validation", "test"]}
        labels = {split: data[split]["lateral_movement_within_horizon"].astype(int)
                  for split in ["validation", "test"]}
        episodes, episode_summary, samples = module.episode_alert_report(
            pd.read_csv(sequence_dir / "sample_manifest.csv"), probabilities, labels,
            frozen_thresholds[regime], Path(args.episodes_dir),
        )
        episodes.to_csv(out / f"{regime}_episode_alerts.csv", index=False)
        samples.to_csv(out / f"{regime}_sample_predictions.csv", index=False)
        results[regime] = {
            "training": {"candidates": summaries[regime], **selection_record[regime]},
            **dynamics, **semantics,
            "equivariance": {"validation": validation_audit, "test": test_audit},
            "episode_alert_summary": episode_summary,
        }
        torch.save({"model_state_dict": model.state_dict(), "model_config": config,
                    "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                    "feature_metadata": metadata, "training": results[regime]["training"],
                    "sealed_test_policy": "model predictions only after both regimes and thresholds frozen; prior external schema/count integrity check disclosed"},
                   model_dir / f"mvp_v3_graph_rssm_{regime}.pt")

    report = {
        "scope": "paired-prefix V3 comparison on a fresh sealed controlled-lab holdout",
        "test_isolation": "model test predictions were produced only after scratch/pretrained seed+epoch and both LM thresholds were frozen on validation; no post-test model selection",
        "pretraining_integrity_disclosure": "before training, test arrays were opened externally only for schema/finiteness/shape/count validation, not model prediction or selection",
        "episode_splits": {key: sorted(value) for key, value in split_ids.items()},
        "sample_counts": {split: len(data[split]["context_states"]) for split in data},
        "initialization_policy": "per seed, semantic initialization is identical; UNSW condition replaces only non-semantic dynamics/edge tensors",
        "initialization_audit": initialization_audit,
        "public_checkpoint_semantic_supervision": False,
        "weights": weights, "smoke_tests": smoke,
        "results": results,
        "limitations": [
            "Fresh validation/test contain three stopped/progressing pairs each.",
            "Overlapping windows within each episode are correlated.",
            "Seed matching controls roles/random-delay calls, but real command runtimes make absolute event times differ.",
            "All episodes use one three-container lab topology.",
            "Monte Carlo spread is not calibrated uncertainty.",
        ],
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("\n===== sealed V3 test summary (reporting, not selection) =====")
    for regime, result in results.items():
        state = result["state_prediction"]["test"]; edge = result["future_edge_presence"]["test"]
        lm = result["future_lateral_movement"]["test"]; pre = result["future_lateral_movement"]["test_before_any_observed_lateral"]
        pair = result["future_lateral_pair_ranking"]["test"]
        episode = result["episode_alert_summary"]["splits"]["test"]
        print(f"{regime:16s} state={state['normalized_mae']:.3f} active={state['active_normalized_mae']:.3f} "
              f"edge_AP={edge['average_precision']:.3f} LM_F1/AP={lm['f1']:.3f}/{lm['average_precision']:.3f} "
              f"pre_F1={pre['f1']:.3f} pair={pair['top1_accuracy_any_true_lm_pair']:.3f} "
              f"progress={episode['progressing_detected_before_first_lm']}/{episode['progressing_episodes']} "
              f"negative_alerts={episode['nonprogressing_episodes_with_any_30s_false_alert']}/{episode['nonprogressing_episodes']}")
    print(f"comparison -> {out / 'comparison.json'}")
    return 0


if __name__ == "__main__": sys.exit(main())
