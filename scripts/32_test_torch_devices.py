#!/usr/bin/env python3
"""Test CPU/CUDA GraphRSSM parity, gradients, equivariance, and speed on train only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.device import device_summary, resolve_device  # noqa: E402
from src.cyberwm.graph_rssm import GraphRSSM, fit_graph_feature_scaler  # noqa: E402


def load_training():
    spec = importlib.util.spec_from_file_location("device_test_training", ROOT / "scripts/15_train_rssm.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def benchmark(model: GraphRSSM, context: torch.Tensor, future: torch.Tensor,
              device: torch.device, steps: int) -> float:
    model.train(); full = torch.cat([context, future], dim=1)
    for _ in range(3):
        model.zero_grad(set_to_none=True); _, predicted = model(full, context.shape[1], sample=False)
        F.mse_loss(predicted["decoded"], future).backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True); _, predicted = model(full, context.shape[1], sample=False)
        F.mse_loss(predicted["decoded"], future).backward()
    if device.type == "cuda": torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/mvp_v3/sequences"))
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--benchmark-steps", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "outputs/device_test.json"))
    args = ap.parse_args(); torch.set_num_threads(min(8, os.cpu_count() or 1))
    target_device = resolve_device(args.device); module = load_training()
    directory = Path(args.sequences_dir); metadata = json.loads((directory / "feature_metadata.json").read_text())
    # Deliberately load train only: this hardware test cannot consume validation/test evidence.
    train = module.load_split(directory, "train")
    config = {
        "global_size": len(metadata["global_feature_names"]),
        "node_size": len(metadata["node_feature_names"]), "node_count": len(metadata["node_slots"]),
        "edge_size": len(metadata["edge_feature_names"]), "pair_count": len(metadata["directed_edge_slots"]),
        "horizon": metadata["future_horizon_states"], "global_hidden": 64, "node_hidden": 32,
        "edge_hidden": 32, "stochastic_size": 16, "local_embedding": 32,
        "technique_count": len(metadata["technique_targets"]),
    }
    scaler = fit_graph_feature_scaler(
        train["context_states"], config["global_size"], config["node_size"], config["node_count"],
        config["edge_size"], config["pair_count"],
    )
    count = min(args.batch_size, len(train["context_states"]))
    raw_context = train["context_states"][:count]; raw_future = train["future_states"][:count]
    context_cpu = torch.from_numpy(module.normalize(scaler, raw_context))
    future_cpu = torch.from_numpy(module.normalize(scaler, raw_future))
    module.seed_everything(7321); cpu_model = GraphRSSM(**config).eval()
    baseline_state = {name: value.detach().clone() for name, value in cpu_model.state_dict().items()}
    with torch.no_grad(): cpu_output = cpu_model.forecast(context_cpu, sample=False)["decoded"]
    target_model = GraphRSSM(**config).to(target_device); target_model.load_state_dict(baseline_state); target_model.eval()
    context_target = context_cpu.to(target_device); future_target = future_cpu.to(target_device)
    with torch.no_grad(): target_output = target_model.forecast(context_target, sample=False)["decoded"].cpu()
    parity_delta = float((cpu_output - target_output).abs().max())
    if parity_delta > 2e-5: raise AssertionError(f"CPU/device deterministic delta too large: {parity_delta}")

    state_permutations, _ = module.state_and_edge_permutation_indices(metadata)
    with torch.no_grad(): base = target_model.forecast(context_target[:16], sample=False)["decoded"]
    equivariance_delta = 0.0
    for order in state_permutations:
        permuted = torch.from_numpy(module.normalize(scaler, raw_context[:16, :, order])).to(target_device)
        with torch.no_grad(): prediction = target_model.forecast(permuted, sample=False)["decoded"]
        equivariance_delta = max(equivariance_delta, float((prediction - base[..., order]).abs().max()))
    if equivariance_delta > 2e-5: raise AssertionError(f"device equivariance delta too large: {equivariance_delta}")

    # One explicit finite backward pass before timing.
    target_model.zero_grad(set_to_none=True)
    _, imagined = target_model(torch.cat([context_target, future_target], dim=1), context_cpu.shape[1], sample=False)
    loss = F.mse_loss(imagined["decoded"], future_target); loss.backward()
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                           for parameter in target_model.parameters())
    if not finite_gradients: raise AssertionError("non-finite gradients")

    # Use independent models because benchmarking accumulates no optimizer state but overwrites gradients.
    cpu_benchmark_model = GraphRSSM(**config); cpu_benchmark_model.load_state_dict(baseline_state)
    target_benchmark_model = GraphRSSM(**config).to(target_device); target_benchmark_model.load_state_dict(baseline_state)
    cpu_seconds = benchmark(cpu_benchmark_model, context_cpu, future_cpu, torch.device("cpu"), args.benchmark_steps)
    target_seconds = benchmark(target_benchmark_model, context_target, future_target,
                               target_device, args.benchmark_steps)
    result = {
        "scope": "train-only hardware test; validation/test not loaded",
        "runtime": device_summary(target_device), "batch_size": count,
        "deterministic_cpu_device_max_delta": parity_delta,
        "deterministic_equivariance_max_delta": equivariance_delta,
        "finite_backward_loss": float(loss.detach().cpu()), "finite_gradients": finite_gradients,
        "cpu_forward_backward_seconds_per_step": cpu_seconds,
        "selected_device_forward_backward_seconds_per_step": target_seconds,
        "speedup_vs_cpu": cpu_seconds / target_seconds,
        "cuda_peak_memory_bytes": (torch.cuda.max_memory_allocated(target_device)
                                   if target_device.type == "cuda" else 0),
    }
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__": sys.exit(main())
