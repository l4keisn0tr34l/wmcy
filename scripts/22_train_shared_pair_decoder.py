#!/usr/bin/env python3
"""Train and audit a shared-weight source-target pair decoder on a frozen RSSM."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--initial-checkpoint", default=str(ROOT / "models/mvp_v2_rssm_kl_tuned.pt"))
    ap.add_argument("--reference-audit", default=str(ROOT / "outputs/mvp_v2/rssm/pair_equivariance_audit.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v2/rssm/shared_pair_decoder.json"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/mvp_v2_rssm_shared_pair.pt"))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--mc-validation", type=int, default=50)
    ap.add_argument("--mc-test", type=int, default=100)
    args = ap.parse_args()

    module = load_script("cyberwm_rssm_training", ROOT / "scripts/15_train_rssm.py")
    audit = load_script("cyberwm_pair_audit", ROOT / "scripts/21_audit_pair_equivariance.py")
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())
    # Test remains unloaded until validation has selected the seed.
    data = {split: module.load_split(sequence_dir, split) for split in ["train", "validation"]}
    scaler = module.fit_context_scaler(data["train"]["context_states"])
    state_permutations, pair_permutations = module.state_and_edge_permutation_indices(metadata)
    context_steps = metadata["context_states"]
    horizon = metadata["future_horizon_states"]
    global_size = len(metadata["global_feature_names"])
    node_size = len(metadata["node_feature_names"])
    edge_size = len(metadata["edge_feature_names"])
    node_count = len(metadata["node_slots"])
    model_config = {
        "observation_size": metadata["state_feature_count"],
        "embedding_size": 64, "deterministic_size": 64, "stochastic_size": 16,
        "horizon": horizon, "technique_count": len(metadata["technique_targets"]),
        "pair_count": len(metadata["directed_edge_slots"]),
        "shared_pair_decoder": True, "global_size": global_size,
        "node_size": node_size, "node_count": node_count, "edge_size": edge_size,
    }
    global_width = global_size
    node_width = node_size * node_count
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, metadata["state_feature_count"])]

    initial_checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    # Backward compatibility: the legacy/default model must still load exactly.
    legacy_model = module.CompactRSSM(**initial_checkpoint["model_config"])
    incompatible = legacy_model.load_state_dict(initial_checkpoint["model_state_dict"])
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"legacy checkpoint incompatibility: {incompatible}")

    module.seed_everything(22_700)
    initial_model = module.CompactRSSM(**model_config)
    backbone_state = {
        name: value for name, value in initial_checkpoint["model_state_dict"].items()
        if not name.startswith("pair_head.")
    }
    incompatible = initial_model.load_state_dict(backbone_state, strict=False)
    expected_missing = {"pair_head.0.weight", "pair_head.0.bias", "pair_head.2.weight", "pair_head.2.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise AssertionError(f"unexpected shared-head initialization mismatch: {incompatible}")
    initial_state = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    frozen_reference = {name: value.detach().clone() for name, value in initial_state.items()
                        if not name.startswith("pair_head.")}

    with torch.no_grad():
        shape = initial_model.forecast(
            torch.from_numpy(module.normalize(scaler, data["validation"]["context_states"][:2])),
            sample=False,
        )["pair_logits"].shape
    if shape != (2, len(metadata["directed_edge_slots"])):
        raise AssertionError(f"unexpected shared pair shape: {shape}")

    pair_weights = {"future_state": 0.0, "reconstruction": 0.0, "edge": 0.0,
                    "kl": 0.0, "lm": 0.0, "technique": 0.0, "pair": 1.0,
                    "free_nats": 0.0}
    train_pairs = data["train"]["future_lateral_edges"].max(axis=1)
    pos_weights = {
        "edge": module.positive_weight(data["train"]["future_edge_presence"]),
        "lm": module.positive_weight(data["train"]["lateral_movement_within_horizon"]),
        "technique": module.positive_weight(data["train"]["future_techniques"].max(axis=1)),
        "pair": module.positive_weight(train_pairs),
    }
    train_loader = module.make_loader(data["train"], scaler, args.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args.batch_size, False)
    state_indices = torch.from_numpy(state_permutations).long()
    edge_indices = torch.from_numpy(pair_permutations).long()

    candidates = []; candidate_states = {}
    print("===== shared pair decoder training; validation-only selection =====")
    for index, seed in enumerate(int(value) for value in args.seeds.split(",") if value.strip()):
        state, training = module.train_seed(
            seed, train_loader, validation_loader, model_config, context_steps, groups,
            pair_weights, pos_weights, state_indices, edge_indices, torch.device("cpu"),
            args.epochs, args.patience, args.learning_rate, initial_state=initial_state,
            selection_mode="pair", freeze_backbone=True,
        )
        for name, value in state.items():
            if not name.startswith("pair_head.") and not torch.equal(value, frozen_reference[name]):
                raise AssertionError(f"frozen tensor changed: {name}")
        model = module.CompactRSSM(**model_config); model.load_state_dict(state); model.eval()
        validation_audit, _ = audit.evaluate_split(
            module, model, scaler, data["validation"], state_permutations,
            pair_permutations, args.mc_validation, args.batch_size, 600_000 + index,
            threshold=None,
        )
        summary = validation_audit["summary"]
        selection_score = (
            summary["pair_micro_average_precision"]["mean"]
            + summary["pair_top1_accuracy_any_true_pair"]["mean"]
            - summary["pair_equivariance_mae"]["mean"]
        )
        candidate_states[seed] = state
        candidates.append({
            "seed": seed,
            "training": {key: value for key, value in training.items() if key != "history"},
            "validation_selection_score": selection_score,
            "validation": validation_audit,
        })
        print(f"seed={seed} val_pair_AP_mean={summary['pair_micro_average_precision']['mean']:.3f} "
              f"top1_mean={summary['pair_top1_accuracy_any_true_pair']['mean']:.3f} "
              f"eq_mae={summary['pair_equivariance_mae']['mean']:.3f} score={selection_score:.3f}")

    selected = max(candidates, key=lambda row: row["validation_selection_score"])
    selected_seed = int(selected["seed"])
    selected_state = candidate_states[selected_seed]
    selected_model = module.CompactRSSM(**model_config)
    selected_model.load_state_dict(selected_state); selected_model.eval()

    # The test split is loaded only after the validation selection above is fixed.
    data["test"] = module.load_split(sequence_dir, "test")
    test_audit, _ = audit.evaluate_split(
        module, selected_model, scaler, data["test"], state_permutations,
        pair_permutations, args.mc_test, args.batch_size, 700_000 + selected_seed,
        threshold=float(selected["validation"]["threshold"]),
    )
    reference = json.loads(Path(args.reference_audit).read_text())[
        "models"
    ]["kl_tuned_candidate"]
    pair_parameter_count = sum(parameter.numel() for parameter in selected_model.pair_head.parameters())
    total_parameter_count = sum(parameter.numel() for parameter in selected_model.parameters())
    report = {
        "scope": "shared-weight pair-head experiment on frozen KL-tuned RSSM",
        "input": "imagined decoded source-node, destination-node, and directed-edge features",
        "test_isolation": "test loaded only after validation selected the seed",
        "architecture": {
            "same_scorer_for_all_directed_pairs": True,
            "pair_input_width": horizon * (2 * node_size + edge_size),
            "hidden_width": 64,
            "pair_head_parameters": pair_parameter_count,
            "total_parameters": total_parameter_count,
        },
        "candidate_seeds": candidates,
        "selected_seed": selected_seed,
        "selection_rule": "maximize validation permutation-mean pair AP + top-1 - equivariance MAE",
        "selected_validation_score": selected["validation_selection_score"],
        "screening_used_test": False,
        "test": test_audit,
        "reference_kl_tuned_independent_head": reference,
        "limitations": [
            "The flattened RSSM encoder/decoder remains slot-specific and is not equivariant.",
            "Only 13 validation and 14 test pair-positive windows exist.",
            "Validation and test episodes have been inspected in earlier experiments.",
            "The shared head has more parameters than the old linear pair head.",
            "Success here would not replace new-episode validation.",
        ],
    }
    for split_audit in [selected["validation"], test_audit]:
        if len(split_audit["permutations"]) != 6:
            raise AssertionError("missing permutation")
        for row in split_audit["permutations"]:
            if row["pair"]["positive_samples"] not in {13, 14}:
                raise AssertionError("pair target support changed")
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("non-finite metric")

    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checkpoint_output = Path(args.checkpoint_out); checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": selected_state, "model_config": model_config,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_metadata": metadata, "selection": {
                    "seed": selected_seed, "rule": report["selection_rule"],
                    "validation_score": selected["validation_selection_score"],
                }}, checkpoint_output)
    summary = test_audit["summary"]
    print("\n===== selected shared-head test audit =====")
    print(f"seed={selected_seed} pair_AP={summary['pair_micro_average_precision']} "
          f"top1={summary['pair_top1_accuracy_any_true_pair']} "
          f"eq_mae={summary['pair_equivariance_mae']} "
          f"choice_consistency={summary['pair_top_choice_consistency_with_identity']}")
    print(f"metrics -> {output}\ncheckpoint -> {checkpoint_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
