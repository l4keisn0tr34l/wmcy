#!/usr/bin/env python3
"""Pre-freeze regression tests for V5 train/validation model plumbing."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.action_graph_rssm import ActionGraphRSSM
from src.cyberwm.branching_graph_rssm import BranchingGraphRSSM
from src.cyberwm.v5_model_protocol import (assert_test_sealed, compatible_parameters, fit_v5_scaler,
                                           load_export, load_protocol)


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> int:
    assert_test_sealed(); protocol = load_protocol(); scaler, keys = fit_v5_scaler()
    assert len(keys) == len(set(keys)) == 440 and scaler.mean_.shape == (345,)
    action, _, metadata = load_export("action", "train")
    aligned, _, _ = load_export("passive_action", "train")
    for name in aligned: np.testing.assert_array_equal(action[name], aligned[name])
    passive, _, _ = load_export("passive", "train")
    base_module = load_script("v5_test_base", ROOT / "scripts/15_train_rssm.py")
    action_module = load_script("v5_test_action", ROOT / "scripts/43_train_action_graph_rssm_v4.py")
    branch_module = load_script("v5_test_branch", ROOT / "scripts/34_train_branching_graph_rssm.py")
    base = protocol["graph"]; groups = [slice(0, 15), slice(15, 105), slice(105, 345)]
    state_permutations, pair_permutations = base_module.state_and_edge_permutation_indices(metadata)
    assert len(state_permutations) == len(pair_permutations) == 120
    action_model = ActionGraphRSSM(**base, horizon=6, action_type_count=2, action_embedding=32)
    action_pos = {"edge": base_module.positive_weight(action["future_edge_presence"]),
                  "lm": base_module.positive_weight(action["lateral_movement_within_horizon"]),
                  "technique": base_module.positive_weight(action["future_techniques"].max(axis=1)),
                  "pair": base_module.positive_weight(action["future_lateral_edges"].max(axis=1))}
    loader = action_module.make_loader(action, scaler, base_module, 6, False); batch = next(iter(loader))
    terms = action_module.loss_terms(base_module, action_model, batch, 3, groups,
                                     protocol["action_track"]["loss_weights"], action_pos, sample=True)
    assert torch.isfinite(terms["total"]); terms["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in action_model.parameters() if p.grad is not None)
    branch_model = BranchingGraphRSSM(**base, horizon=6, branch_count=2, branch_embedding_size=32)
    branch_pos = {"edge": base_module.positive_weight(passive["future_edge_presence"]),
                  "lm": base_module.positive_weight(passive["lateral_movement_within_horizon"]),
                  "technique": base_module.positive_weight(passive["future_techniques"].max(axis=1)),
                  "pair": base_module.positive_weight(passive["future_lateral_edges"].max(axis=1))}
    branch_loader = base_module.make_loader(passive, scaler, 16, False); branch_batch = next(iter(branch_loader))
    branch_terms = branch_module.loss_terms(base_module, branch_model, branch_batch, 3, groups,
        protocol["passive_branch_track"]["loss_weights"], branch_pos, 0.25, "lm-outcome", sample=True)
    assert torch.isfinite(branch_terms["total"]); branch_terms["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in branch_model.parameters() if p.grad is not None)
    # Actual source dictionaries: old scalers excluded, shared tensor contracts explicit.
    for regime, expected_action_missing, expected_branch_missing in [("friday", 11, 9), ("v4", 0, 9)]:
        checkpoint = torch.load(ROOT / protocol["initializations"][regime]["source"], map_location="cpu", weights_only=False)
        a = ActionGraphRSSM(**base, horizon=6, action_type_count=2, action_embedding=32)
        loaded_a, missing_a = compatible_parameters(a, checkpoint["model_state_dict"])
        b = BranchingGraphRSSM(**base, horizon=6, branch_count=2, branch_embedding_size=32)
        loaded_b, missing_b = compatible_parameters(b, checkpoint["model_state_dict"])
        assert len(loaded_a) in {82, 93} and len(missing_a) == expected_action_missing
        assert len(loaded_b) == 82 and len(missing_b) == expected_branch_missing
        assert len(checkpoint["scaler_mean"]) == 141
    print({"status": "PASS", "test_exports_absent": True, "scaler_unique_context_rows": 440,
           "action_gradient_tensors": sum(p.grad is not None for p in action_model.parameters()),
           "branch_gradient_tensors": sum(p.grad is not None for p in branch_model.parameters()),
           "all_host_permutations": len(state_permutations), "old_scalers_reused": False})
    return 0


if __name__ == "__main__": raise SystemExit(main())
