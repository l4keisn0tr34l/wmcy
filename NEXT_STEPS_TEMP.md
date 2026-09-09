# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Resumed checkpoint

Repository clean at `1fdf7fb`. CUDA is available on RTX 3050 with torch 2.9.1+cu128. V3 test is frozen/inspected and must not be used in development.

## Current batch: branch-aware contract before architecture

Implement a reusable validation-only branching forecast audit:

1. Add `src/cyberwm/branch_metrics.py` with metrics for `[draw/branch, sample, horizon, feature]` forecasts:
   - mixture/mean trajectory MAE;
   - oracle best-branch trajectory MAE;
   - global/node/edge group MAE;
   - branch/draw diversity;
   - hard best-branch utilization and effective branch count;
   - edge AP from probability-weighted branches;
   - exact LM Brier/log loss/ECE and reliability bins;
   - explicit distinction between exact outcome and dangerous precursor alert burden.
2. Add `scripts/33_evaluate_branching_contract.py`.
3. Load only V3 validation and the already selected scratch checkpoint; never load V3 test.
4. Treat existing 20 stochastic rollouts as uniformly weighted candidate futures to establish whether current RSSM stochasticity covers materially different futures.
5. Include episode-level stopped/progressing risk summaries using validation only; this is evaluation metadata, never input.
6. Write `outputs/mvp_v3/branching/contract_baseline_validation.json`.
7. Unit-test exact synthetic cases: identical branches, complementary perfect branches, collapsed weights, ECE boundaries, finite outputs.

## Following batch

Implement `src/cyberwm/branching_graph_rssm.py` with two learned latent branches, context-only branch weights, shared equivariant graph rollout, and proper mixture trajectory likelihood. Train on V3 train and select only on V3 validation using CUDA. Do not open V3 test.
