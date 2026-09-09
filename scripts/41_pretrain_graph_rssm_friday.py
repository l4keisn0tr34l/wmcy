#!/usr/bin/env python3
"""Fixed-epoch GraphRSSM dynamics pretraining on connected Friday train data only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.device import cpu_state_dict, device_summary, resolve_device  # noqa: E402
from src.cyberwm.graph_rssm import GraphRSSM, fit_graph_feature_scaler  # noqa: E402


def load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def public_train(module: Any, directory: Path, horizon: int,
                 technique_count: int, pair_count: int) -> dict[str, np.ndarray]:
    data = module.load_split(directory, "train"); count = len(data["context_states"])
    expected = {"context_states", "future_states", "future_edge_presence"}
    if set(data) != expected: raise ValueError(f"unexpected Friday arrays: {sorted(data)}")
    data["lateral_movement_within_horizon"] = np.zeros(count, dtype=np.float32)
    data["future_techniques"] = np.zeros((count, horizon, technique_count), dtype=np.float32)
    data["future_lateral_edges"] = np.zeros((count, horizon, pair_count), dtype=np.float32)
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/cic2017_friday_pcap/graph_sequences"))
    ap.add_argument("--lab-feature-metadata", default=str(ROOT / "outputs/mvp_v2/sequences/feature_metadata.json"))
    ap.add_argument("--checkpoint-out", default=str(ROOT / "models/friday_graph_rssm_pretrained_fixed.pt"))
    ap.add_argument("--report-out", default=str(ROOT / "outputs/cic2017_friday_pcap/pretraining/fixed_pretraining.json"))
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=41001)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(); torch.set_num_threads(min(8, os.cpu_count() or 1)); device = resolve_device(args.device)
    if (args.epochs, args.seed, args.batch_size, args.learning_rate) != (100, 41001, 128, 3e-4):
        raise ValueError("Friday protocol is frozen at epochs=100, seed=41001, batch=128, lr=3e-4")
    if not args.force and (Path(args.checkpoint_out).exists() or Path(args.report_out).exists()):
        raise FileExistsError("refusing to replace Friday pretraining outputs without --force")
    directory = Path(args.sequences_dir)
    if (directory / "validation.npz").exists() or (directory / "test.npz").exists():
        raise ValueError("Friday connected capture must not contain validation/test arrays")
    metadata = json.loads((directory / "feature_metadata.json").read_text())
    lab_metadata = json.loads(Path(args.lab_feature_metadata).read_text())
    if metadata["state_feature_names"] != lab_metadata["state_feature_names"]:
        raise ValueError("Friday/lab feature contracts differ")
    required_policy = "single connected Friday capture retained as train-only pretraining data; no internal validation claim"
    if metadata.get("split_policy") != required_policy: raise ValueError("Friday split policy changed")
    module = load_script("friday_rssm_training", ROOT / "scripts/15_train_rssm.py")
    graph_script = load_script("friday_graph_training", ROOT / "scripts/23_train_graph_rssm.py")
    config = graph_script.graph_config(lab_metadata); context_steps = metadata["context_states"]
    data = public_train(module, directory, config["horizon"], config["technique_count"], config["pair_count"])
    scaler = fit_graph_feature_scaler(
        data["context_states"], config["global_size"], config["node_size"], config["node_count"],
        config["edge_size"], config["pair_count"],
    )
    permutations, edge_permutations = module.state_and_edge_permutation_indices(lab_metadata)
    state_indices = torch.from_numpy(permutations).long().to(device)
    edge_indices = torch.from_numpy(edge_permutations).long().to(device)
    global_width = config["global_size"]; node_width = config["node_size"] * config["node_count"]
    groups = [slice(0, global_width), slice(global_width, global_width + node_width),
              slice(global_width + node_width, metadata["state_feature_count"])]
    weights = {"future_state": 1.0, "reconstruction": 0.25, "edge": 0.25,
               "kl": 0.01, "lm": 0.0, "technique": 0.0, "pair": 0.0, "free_nats": 0.0}
    pos_weights = {
        "edge": module.positive_weight(data["future_edge_presence"]).to(device),
        "lm": torch.ones(1, device=device), "technique": torch.ones(config["technique_count"], device=device),
        "pair": torch.ones(config["pair_count"], device=device),
    }
    train_loader = module.make_loader(data, scaler, args.batch_size, True)
    audit_loader = module.make_loader(data, scaler, args.batch_size, False)
    module.seed_everything(args.seed); model = GraphRSSM(**config).to(device)
    smoke = graph_script.smoke_tests(
        module, model, train_loader, permutations, edge_permutations, scaler,
        data["context_states"], context_steps, groups, weights, pos_weights, device,
    )
    initial = module.validate_objective(
        model, audit_loader, context_steps, groups, weights, pos_weights, device, selection_mode="dynamics"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate); history = []
    print("device:", device_summary(device)); print("===== fixed Friday observable dynamics pretraining =====")
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; count = 0
        for raw in train_loader:
            batch = module.augment_batch(module.move(raw, device), state_indices, edge_indices)
            optimizer.zero_grad(set_to_none=True)
            terms = module.loss_terms(model, batch, context_steps, groups, weights, pos_weights, sample=True)
            if not torch.isfinite(terms["total"]): raise AssertionError("non-finite Friday loss")
            terms["total"].backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
            total += float(terms["total"].detach()) * len(batch[0]); count += len(batch[0])
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            audit = module.validate_objective(
                model, audit_loader, context_steps, groups, weights, pos_weights,
                device, selection_mode="dynamics",
            )
            row = {"epoch": epoch, "stochastic_train_total": total / count,
                   "deterministic_training_dynamics_objective": audit["selection"]}
            history.append(row)
            print(f"epoch={epoch} train={total/count:.4f} deterministic_train={audit['selection']:.4f}")
    final = module.validate_objective(
        model, audit_loader, context_steps, groups, weights, pos_weights, device, selection_mode="dynamics"
    )
    if final["selection"] >= initial["selection"]:
        raise AssertionError("fixed Friday pretraining did not improve its training objective")
    report = {
        "scope": "Friday connected-capture training-only observable graph dynamics pretraining",
        "selection": "none; fixed seed and exactly 100 epochs",
        "validation_or_test_loaded": False, "runtime_device": device_summary(device),
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weights": weights,
        "semantic_weights_exactly_zero": all(weights[name] == 0 for name in ["lm", "technique", "pair"]),
        "samples": len(data["context_states"]), "model_config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "smoke_tests": smoke, "initial_training_objective": initial,
        "final_training_objective": final, "history": history,
        "claim_limit": "training-fit initialization only; transfer/generalization not evaluated",
    }
    report_path = Path(args.report_out); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    checkpoint = Path(args.checkpoint_out); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": cpu_state_dict(model.state_dict()), "model_config": config,
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "feature_metadata": metadata,
        "training": report, "training_only": True, "validation_or_test_loaded": False,
    }, checkpoint)
    print(f"report -> {report_path}\ncheckpoint -> {checkpoint}"); return 0


if __name__ == "__main__": sys.exit(main())
