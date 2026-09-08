#!/usr/bin/env python3
"""Tune RSSM KL weight/free-nats with validation-only screening and gates."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_kl(module: Any, model: torch.nn.Module, split: dict[str, np.ndarray], scaler: Any,
           context_steps: int, batch_size: int) -> float:
    context = module.normalize(scaler, split["context_states"])
    future = module.normalize(scaler, split["future_states"])
    loader = DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(context), torch.from_numpy(future)),
        batch_size=batch_size, shuffle=False,
    )
    total = 0.0; count = 0
    model.eval()
    with torch.no_grad():
        for context_batch, future_batch in loader:
            full = torch.cat([context_batch, future_batch], dim=1)
            posterior, _ = model(full, context_steps=context_steps, sample=False)
            value = module.gaussian_kl(
                posterior["post_mean"], posterior["post_std"],
                posterior["prior_mean"], posterior["prior_std"],
            ).mean()
            total += float(value) * len(full); count += len(full)
    return total / count


def validation_metrics(module: Any, ablation: Any, model: torch.nn.Module,
                       validation: dict[str, np.ndarray], scaler: Any,
                       metadata: dict[str, Any], state_permutations: np.ndarray,
                       context_steps: int, mc_samples: int, batch_size: int,
                       seed: int) -> dict[str, Any]:
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    groups = {
        "global": slice(0, global_width),
        "node": slice(global_width, global_width + node_width),
        "edge": slice(global_width + node_width, metadata["state_feature_count"]),
    }
    mc = module.mc_predictions(
        model, validation["context_states"], scaler, torch.device("cpu"),
        mc_samples, batch_size, seed,
    )
    state_mean = mc["state"].mean(axis=0)
    state_std = mc["state"].std(axis=0)
    context = module.normalize(scaler, validation["context_states"])
    future = module.normalize(scaler, validation["future_states"])
    state = module.state_metrics(
        state_mean, future, context, validation["future_states"], groups
    )
    state_error = np.abs(state_mean - future)
    edge_probability = mc["edge"].mean(axis=0)
    edge_target = validation["future_edge_presence"]
    lm_probability = mc["lm"].mean(axis=0)
    lm_target = validation["lateral_movement_within_horizon"].astype(int)
    threshold = module.best_f1_threshold(lm_target, lm_probability)
    sensitivity = ablation.host_sensitivity_internal(
        module, model, validation, scaler, state_permutations, threshold,
        batch_size, seed + 50_000,
    )
    return {
        "state_normalized_mae": state["normalized_mae"],
        "active_state_normalized_mae": state["active_normalized_mae"],
        "edge_average_precision": float(average_precision_score(edge_target.ravel(), edge_probability.ravel())),
        "lm_average_precision": float(average_precision_score(lm_target, lm_probability)),
        "lm_f1_at_validation_selected_threshold": module.binary_metrics(
            lm_target, lm_probability, threshold
        )["f1"],
        "lm_threshold": threshold,
        "mean_normalized_state_spread": float(state_std.mean()),
        "spread_error_correlation": float(np.corrcoef(
            state_std.mean(axis=2).ravel(), state_error.mean(axis=2).ravel()
        )[0, 1]),
        "raw_posterior_prior_kl": raw_kl(
            module, model, validation, scaler, context_steps, batch_size
        ),
        "host_permutation_mean_lm_range": sensitivity["mean_per_sample_probability_range"],
        "host_permutation_max_lm_range": sensitivity["max_per_sample_probability_range"],
    }


def candidate_score(metrics: dict[str, Any], reference: dict[str, Any]) -> float:
    return (
        metrics["state_normalized_mae"] / reference["state_normalized_mae"]
        - 0.25 * metrics["edge_average_precision"] / reference["edge_average_precision"]
        - 0.25 * metrics["lm_average_precision"] / reference["lm_average_precision"]
    )


def eligibility(metrics: dict[str, Any], reference: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    gates = {
        "state_within_3_percent": metrics["state_normalized_mae"] <= 1.03 * reference["state_normalized_mae"],
        "edge_ap_at_least_90_percent": metrics["edge_average_precision"] >= 0.90 * reference["edge_average_precision"],
        "lm_ap_at_least_90_percent": metrics["lm_average_precision"] >= 0.90 * reference["lm_average_precision"],
        "spread_at_least_half": metrics["mean_normalized_state_spread"] >= 0.50 * reference["mean_normalized_state_spread"],
        "host_mean_range_at_most_1_5x": metrics["host_permutation_mean_lm_range"] <= 1.50 * reference["host_permutation_mean_lm_range"],
    }
    return all(gates.values()), gates


def finite_metrics(metrics: dict[str, Any]) -> None:
    for name, value in metrics.items():
        if isinstance(value, (float, int)) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite metric {name}={value}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v2/sequences"))
    ap.add_argument("--reference-checkpoint", default=str(ROOT / "models/mvp_v2_rssm.pt"))
    ap.add_argument("--out", default=str(ROOT / "outputs/mvp_v2/rssm/kl_tuning.json"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/mvp_v2_rssm_kl_tuned.pt"))
    ap.add_argument("--screen-epochs", type=int, default=300)
    ap.add_argument("--confirm-epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mc-screen", type=int, default=10)
    ap.add_argument("--mc-confirm", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--screen-seed", type=int, default=7)
    ap.add_argument("--confirm-seeds", default="7,17,27")
    ap.add_argument("--max-settings", type=int, default=0,
                    help="For smoke tests only: truncate the fixed settings list when positive")
    args = ap.parse_args()

    module = load_script("cyberwm_rssm_training", ROOT / "scripts/15_train_rssm.py")
    ablation = load_script("cyberwm_rssm_ablation", ROOT / "scripts/18_run_rssm_ablations.py")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    sequence_dir = Path(args.sequences_dir)
    metadata = json.loads((sequence_dir / "feature_metadata.json").read_text())

    # Test is deliberately not loaded until the final validation selection is frozen.
    data = {split: module.load_split(sequence_dir, split) for split in ["train", "validation"]}
    scaler = module.fit_context_scaler(data["train"]["context_states"])
    state_permutations, edge_permutations = module.state_and_edge_permutation_indices(metadata)
    context_steps = metadata["context_states"]
    horizon = metadata["future_horizon_states"]
    global_width = len(metadata["global_feature_names"])
    node_width = len(metadata["node_feature_names"]) * len(metadata["node_slots"])
    group_slices = [slice(0, global_width), slice(global_width, global_width + node_width),
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
    train_loader = module.make_loader(data["train"], scaler, args.batch_size, True)
    validation_loader = module.make_loader(data["validation"], scaler, args.batch_size, False)

    reference_checkpoint = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=False)
    reference_model = module.CompactRSSM(**model_config)
    reference_model.load_state_dict(reference_checkpoint["model_state_dict"]); reference_model.eval()
    reference = validation_metrics(
        module, ablation, reference_model, data["validation"], scaler, metadata,
        state_permutations, context_steps, args.mc_confirm, args.batch_size, 90_007,
    )
    finite_metrics(reference)

    settings = [(0.0, 0.0)] + [
        (weight, free_nats)
        for weight in [0.01, 0.03, 0.1, 0.3]
        for free_nats in [0.0, 1.0]
    ]
    if args.max_settings > 0:
        settings = settings[:args.max_settings]
    screening = []
    print("===== validation-only KL screening =====")
    for index, (kl_weight, free_nats) in enumerate(settings):
        setting_id = f"kl_{kl_weight:g}_free_{free_nats:g}"
        weights = {**base_weights, "kl": kl_weight, "free_nats": free_nats}
        state, training = module.train_seed(
            args.screen_seed, train_loader, validation_loader, model_config, context_steps,
            group_slices, weights, pos_weights, state_indices, edge_indices,
            torch.device("cpu"), args.screen_epochs, args.patience, args.learning_rate,
            selection_mode="joint",
        )
        model = module.CompactRSSM(**model_config); model.load_state_dict(state); model.eval()
        metrics = validation_metrics(
            module, ablation, model, data["validation"], scaler, metadata,
            state_permutations, context_steps, args.mc_screen, args.batch_size,
            100_000 + index,
        )
        finite_metrics(metrics)
        eligible, gates = eligibility(metrics, reference)
        score = candidate_score(metrics, reference)
        screening.append({"setting_id": setting_id, "kl_weight": kl_weight,
                          "free_nats": free_nats, "seed": args.screen_seed,
                          "training": {key: value for key, value in training.items() if key != "history"},
                          "validation": metrics, "validation_score": score,
                          "eligible": eligible, "gates": gates})
        print(f"{setting_id:20s} state={metrics['state_normalized_mae']:.3f} "
              f"edge={metrics['edge_average_precision']:.3f} lm={metrics['lm_average_precision']:.3f} "
              f"spread={metrics['mean_normalized_state_spread']:.3f} "
              f"host={metrics['host_permutation_mean_lm_range']:.3f} "
              f"eligible={eligible} score={score:.3f}")

    eligible_screening = sorted(
        [row for row in screening if row["eligible"]], key=lambda row: row["validation_score"]
    )
    fallback = False
    if len(eligible_screening) < min(2, len(settings)):
        fallback = True
        eligible_screening = sorted(screening, key=lambda row: row["validation_score"])
    shortlist = eligible_screening[:min(2, len(eligible_screening))]
    confirm_seeds = [int(value) for value in args.confirm_seeds.split(",") if value.strip()]

    confirmation = []
    selected_states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    print("\n===== shortlisted multi-seed confirmation (validation only) =====")
    for setting_index, row in enumerate(shortlist):
        weights = {**base_weights, "kl": row["kl_weight"], "free_nats": row["free_nats"]}
        for seed_index, seed in enumerate(confirm_seeds):
            state, training = module.train_seed(
                seed, train_loader, validation_loader, model_config, context_steps,
                group_slices, weights, pos_weights, state_indices, edge_indices,
                torch.device("cpu"), args.confirm_epochs, args.patience, args.learning_rate,
                selection_mode="joint",
            )
            model = module.CompactRSSM(**model_config); model.load_state_dict(state); model.eval()
            metrics = validation_metrics(
                module, ablation, model, data["validation"], scaler, metadata,
                state_permutations, context_steps, args.mc_confirm, args.batch_size,
                200_000 + 100 * setting_index + seed_index,
            )
            finite_metrics(metrics)
            eligible, gates = eligibility(metrics, reference)
            score = candidate_score(metrics, reference)
            selected_states[(row["setting_id"], seed)] = state
            confirmation.append({"setting_id": row["setting_id"],
                                 "kl_weight": row["kl_weight"],
                                 "free_nats": row["free_nats"], "seed": seed,
                                 "training": {key: value for key, value in training.items() if key != "history"},
                                 "validation": metrics, "validation_score": score,
                                 "eligible": eligible, "gates": gates})
            print(f"{row['setting_id']:20s} seed={seed:2d} state={metrics['state_normalized_mae']:.3f} "
                  f"edge={metrics['edge_average_precision']:.3f} lm={metrics['lm_average_precision']:.3f} "
                  f"spread={metrics['mean_normalized_state_spread']:.3f} "
                  f"eligible={eligible} score={score:.3f}")

    eligible_confirmation = [row for row in confirmation if row["eligible"]]
    selection_fallback = not eligible_confirmation
    selection_pool = eligible_confirmation or confirmation
    selected = min(selection_pool, key=lambda row: row["validation_score"])
    selected_state = selected_states[(selected["setting_id"], int(selected["seed"]))]
    selected_model = module.CompactRSSM(**model_config)
    selected_model.load_state_dict(selected_state); selected_model.eval()

    # Only now, after setting and seed selection are fixed, load/evaluate test.
    data["test"] = module.load_split(sequence_dir, "test")
    final_dynamics, final_mc = ablation.dynamics_metrics(
        module, selected_model, data, scaler, metadata, args.mc_confirm, args.batch_size,
        int(selected["seed"]) + 300_000,
    )
    final_semantics = ablation.internal_semantic_metrics(
        module, final_mc, data, metadata, sequence_dir
    )
    threshold = final_semantics["future_lateral_movement"]["threshold_selected_on_validation"]
    test_sensitivity = ablation.host_sensitivity_internal(
        module, selected_model, data["test"], scaler, state_permutations, threshold,
        args.batch_size, int(selected["seed"]) + 400_000,
    )
    final_test = {
        "state_prediction": final_dynamics["state_prediction"]["test"],
        "future_edge_presence": final_dynamics["future_edge_presence"]["test"],
        "uncertainty": final_dynamics["uncertainty"],
        "future_lateral_movement": final_semantics["future_lateral_movement"]["test"],
        "future_lateral_movement_before_any_observed_lateral": final_semantics[
            "future_lateral_movement"
        ]["test_before_any_observed_lateral"],
        "future_techniques": {name: values["test"]
                              for name, values in final_semantics["future_techniques"].items()},
        "future_lateral_pair_ranking": final_semantics["future_lateral_pair_ranking"]["test"],
        "host_permutation_sensitivity": test_sensitivity,
        "raw_posterior_prior_kl": raw_kl(
            module, selected_model, data["test"], scaler, context_steps, args.batch_size
        ),
    }
    published = json.loads((ROOT / "outputs/mvp_v2/rssm/metrics.json").read_text())
    report = {
        "scope": "validation-only KL weight/free-nats tuning on equal-duration V2",
        "test_isolation": "test split was loaded only after setting and seed selection",
        "reference_validation": reference,
        "gates": {
            "state": "<= 1.03x reference validation normalized MAE",
            "edge_ap": ">= 0.90x reference validation AP",
            "lm_ap": ">= 0.90x reference validation AP",
            "spread": ">= 0.50x reference validation mean spread",
            "host_mean_range": "<= 1.50x reference validation mean permutation range",
        },
        "score": "state_MAE/reference - 0.25*(edge_AP/reference) - 0.25*(LM_AP/reference); lower is better",
        "screening_used_test": False,
        "screening": screening,
        "shortlist": [row["setting_id"] for row in shortlist],
        "shortlist_fallback": fallback,
        "confirmation_used_test": False,
        "confirmation": confirmation,
        "selected": selected,
        "selection_fallback": selection_fallback,
        "final_test": final_test,
        "published_reference_test": {
            "state_normalized_mae": published["state_prediction"]["test"]["normalized_mae"],
            "active_state_normalized_mae": published["state_prediction"]["test"]["active_normalized_mae"],
            "edge_average_precision": published["future_edge_presence"]["test"]["average_precision"],
            "lm_f1": published["future_lateral_movement"]["test"]["f1"],
            "lm_average_precision": published["future_lateral_movement"]["test"]["average_precision"],
            "pre_first_lm_f1": published["future_lateral_movement"]["test_before_any_observed_lateral"]["f1"],
            "pair_top1": published["future_lateral_pair_ranking"]["test"]["top1_accuracy_any_true_lm_pair"],
            "mean_state_spread": published["uncertainty"]["mean_normalized_state_std"],
        },
        "limitations": [
            "Only 24 episodes and 12 LM events; overlapping windows are correlated.",
            "The same validation episodes support checkpoint, setting, seed, and threshold selection.",
            "Repeated project experiments have already inspected this fixed test split.",
            "Spread and spread-error correlation are diagnostics, not calibrated uncertainty.",
            "The scalar ranking weights and eligibility gates are supported design choices, not learned truths.",
        ],
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checkpoint_output = Path(args.checkpoint_out); checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": selected_state, "model_config": model_config,
                "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "feature_metadata": metadata, "selection": selected}, checkpoint_output)

    state = final_test["state_prediction"]
    lm = final_test["future_lateral_movement"]
    pre = final_test["future_lateral_movement_before_any_observed_lateral"]
    print("\n===== final validation-selected test result =====")
    print(f"selected={selected['setting_id']} seed={selected['seed']} fallback={selection_fallback}")
    print(f"state={state['normalized_mae']:.3f} active={state['active_normalized_mae']:.3f} "
          f"edge_AP={final_test['future_edge_presence']['average_precision']:.3f} "
          f"LM_F1/AP={lm['f1']:.3f}/{lm['average_precision']:.3f} "
          f"pre_F1={pre['f1']:.3f} "
          f"pair={final_test['future_lateral_pair_ranking']['top1_accuracy_any_true_lm_pair']:.3f} "
          f"spread={final_test['uncertainty']['mean_normalized_state_std']:.3f}")
    print(f"metrics -> {output}\ncheckpoint -> {checkpoint_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
