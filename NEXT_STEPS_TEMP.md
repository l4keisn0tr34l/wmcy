# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current graph status

- Graph RSSM dynamics are strong: state MAE 0.252, active 0.895, edge AP 0.394, exact host equivariance, robust pair top-1 0.357.
- Initial pooled-latent LM F1/AP is 0.476/0.738.
- Frozen continuation of the same semantic heads did not recover generalization: selected test LM F1/AP 0.455/0.707. Keep as a negative result.

## Current code-writing batch: future-state invariant semantic readout

1. Add an optional backward-compatible GraphRSSM semantic readout from decoded imagined future states.
2. For each future step, concatenate:
   - decoded global features;
   - mean and max over decoded node features;
   - mean and max over decoded directed-edge features.
3. This readout is permutation invariant but preserves outlier/intensity information lost by mean-pooled latent state alone.
4. Initialize all graph dynamics, decoders, edge head, and pair head from `mvp_v2_graph_rssm.pt`.
5. Initialize only new LM/ATT&CK heads randomly; freeze every other tensor.
6. Train seeds 7/17/27 with validation semantic selection; test remains unloaded until seed selection.
7. Verify frozen tensors bit-identical and exact host equivariance.
8. Evaluate LM/pre-LM, ATT&CK, episode lead/false alerts, state/edge, pair, spread.
9. Save `models/mvp_v2_graph_rssm_rich_semantic.pt` and `outputs/mvp_v2/graph_rssm/rich_semantic.json`.

## Decision gate

- Adopt graph RSSM as preferred candidate only if rich readout materially closes LM AP/F1 gap without sacrificing exact equivariance or graph dynamics.
- Otherwise retain split conclusion: graph RSSM is best dynamics model; flattened RSSM remains best semantic model, and more varied/hard-negative data is the bottleneck.
