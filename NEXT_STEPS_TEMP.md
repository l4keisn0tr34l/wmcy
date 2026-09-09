# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Verified branch contract

Committed at `5eb4c65`. V3 validation only:

```text
20-draw expected/oracle state MAE: 0.3250 / 0.3131
oracle gain:                       0.0119
pairwise diversity:                0.0341
LM draw spread:                    0.0114
LM Brier / ECE:                    0.2048 / 0.2286
```

Best draw uses 18.2 effective draws, suggesting diffuse stochastic noise rather than a small interpretable mode set. V3 test remains unused in this development phase.

## Current batch: explicit two-branch equivariant GraphRSSM

1. Add `src/cyberwm/branching_graph_rssm.py`.
2. Reuse the verified equivariant graph observation model and recurrent dynamics.
3. Infer two context-only branch probabilities from the invariant context graph latent.
4. Apply learned invariant branch shifts to global stochastic/deterministic state and shared shifts to every node/edge latent; this preserves host equivariance.
5. Roll both branches through the same six-step prior/decoder.
6. Train with proper probability-weighted mixture trajectory loss, not oracle-only loss.
7. Derive detached future-state responsibilities for branch-specific edge/LM/ATT&CK/pair supervision; future is target-only and never enters branch weights.
8. Add a mild batch-level branch-usage penalty; do not add arbitrary output-diversity reward.
9. Add `scripts/34_train_branching_graph_rssm.py` using V3 train/validation only, CUDA batch128, seeds7/17/27, validation-only selection.
10. Evaluate expected and oracle MAE, branch utilization/diversity, LM calibration, matched-prefix burden, and exact host equivariance.
11. Save validation-only checkpoint/artifact. Do not load V3 test.

## Gates

- future perturbation changes context posterior/branch weights by exactly zero;
- deterministic host equivariance below2e-6 for every branch and branch weights invariant;
- finite gradients and tiny-overfit decrease;
- branch weights use context latent only;
- no scenario/seed/truth input;
- report collapse honestly if effective branch use/diversity/coverage fail;
- no model is promoted without a later fresh V4 evaluation.
