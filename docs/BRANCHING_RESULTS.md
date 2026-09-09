# Explicit Branching Graph RSSM Results

## Status

**Validation-only architecture checkpoint. Eligible for a fresh sealed evaluation; not a V3 test result and not deployment evidence.**

`scripts/34_train_branching_graph_rssm.py` deliberately loads only V3 train and validation. The V3 test split remains frozen and was not reopened.

## What goes in

Three normalized five-second graph states containing observable telemetry only:

- 15 global features;
- three anonymous host slots × 18 node features;
- six directed anonymous host-pair slots × 12 edge features.

No scenario, seed, host role, attack label, ATT&CK truth, LM truth, or future value enters inference.

## Transformation

`BranchingGraphRSSM` retains shared permutation-equivariant graph encoders, recurrent dynamics, and decoders. An invariant context graph latent produces two probabilities. A learned branch embedding shifts global/stochastic state and shifts every node/edge state identically under host relabeling. Both candidates then use the same six-step transition model.

The selected model assigns explicit meanings during training:

```text
branch 0: no LM within the six-state / 30-second future
branch 1: LM within the six-state / 30-second future
```

Future LM truth chooses the realized branch only in the loss. It never enters the branch gate or model input. The objective is a fixed-scale joint outcome/trajectory likelihood: selected-branch trajectory error plus context-gate negative log likelihood. Edge, ATT&CK, LM, and pair heads use the same outcome responsibility. No arbitrary output-diversity reward is used.

## Outputs

For each context:

- two context-only branch probabilities;
- two six-state future graph trajectories;
- future edge probabilities per branch;
- ATT&CK, LM, and source-target pair probabilities per branch.

The branch-weighted mean is the inference-time point forecast. Oracle best-branch error is reported only as candidate-set coverage.

## Validation protocol

- V3 train: 540 overlapping samples from 36 whole episodes;
- V3 validation: 90 samples from six new matched-prefix episodes;
- three fixed seeds: 7, 17, 27;
- CUDA batch 128;
- up to 250 epochs, patience 40;
- selected seed 27 at epoch 127 by the joint validation objective;
- 372,197 parameters.

## Results

| Validation metric | Ordinary RSSM, 20 draws | Outcome two-branch RSSM |
|---|---:|---:|
| expected normalized state MAE | 0.325 | 0.327 |
| oracle best-candidate state MAE | 0.313 | 0.306 |
| oracle coverage gain | 0.012 | 0.021 |
| candidate pairwise diversity | 0.034 | 0.406 |
| future-edge AP | 0.783 | 0.763 |
| exact-LM AP | 0.607 | 0.655 |
| exact-LM Brier | 0.205 | 0.114 |
| exact-LM diagnostic ECE | 0.229 | 0.046 |
| exact-LM validation F1 | 0.653 | 0.652 |

The branch gate's mean probabilities are 0.788/0.212, close to the validation exact-outcome prevalence of 19/90. The conditional LM heads are nearly 0/1 by supervised construction; that separation is not an emergent discovery.

### Does the LM branch represent different telemetry?

For LM-positive validation targets:

| Metric | no-LM branch | LM branch |
|---|---:|---:|
| active-feature normalized MAE | 0.967 | **0.720** |
| quiet-feature normalized MAE | **0.058** | 0.213 |
| future-edge AP | 0.800 | **0.879** |

The LM branch better covers active changes and edges, while deliberately predicting extra activity that hurts error on quiet entries. Major branch differences include per-edge mean duration, total bytes/packets, FIN counts, and anonymous-node traffic volumes.

Across all entries, the no-LM branch is still closer for many LM-positive windows (0.295 versus 0.344 total MAE) because zeros/quiet features dominate. This is why active and quiet metrics are both reported; the result is promising but not conclusive.

### Matched prefixes

The outcome model still alerts on all 3/3 stopped and 3/3 progressing validation episodes. Mean absolute stopped/progressing maximum-risk gap is 0.064, and the progressing member ranks higher in only 2/3 pairs. Explicit branching improves calibrated representation of alternatives but does not recover unobserved future intent.

## Invariants and gates

- future perturbation changes context posterior and branch forecast by exactly `0`;
- maximum deterministic host-relabeling discrepancy is `5.72e-6` in large pair logits (float32 numerical scale); branch-weight delta is `1.19e-7`;
- finite gradients and tiny-overfit pass;
- both branches have nontrivial diversity/coverage;
- the LM branch improves active-state MAE and edge AP on LM-positive futures.

## What is deliberately excluded

- labels/scenario/seed/actor/target from inference;
- future state or future action from branch weights;
- identity-dependent branch shifts;
- V3 test data;
- post-hoc V3 test thresholding;
- claims of calibrated deployment uncertainty.

## Remaining limitation

This is a supervised outcome-conditioned mixture over a tiny lab topology, not a learned causal intent variable. Its strong validation Brier/ECE can reflect only 90 correlated windows and must not be called calibrated uncertainty. The next valid evidence must come from fresh role-balanced V4 episodes frozen after the architecture and evaluation contract.
