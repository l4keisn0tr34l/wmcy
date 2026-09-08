# RSSM Representation-Training Ablations

## Status

Completed on equal-duration V2. These are controlled-lab ablations, not enterprise-generalization evidence.

Verified artifact:

```text
outputs/mvp_v2/rssm/ablations.json
```

Reproduce:

```bash
.venv/bin/python scripts/18_run_rssm_ablations.py
```

The final run used seeds 7/17/27, validation-only checkpoint/threshold selection, 20 Monte Carlo test rollouts, a maximum of 400 epochs for RSSM training, and up to 1,000 epochs for the frozen linear semantic heads.

## Question

Does the RSSM learn security-useful structure from observable telemetry alone, and what changes when semantic supervision can adapt the latent dynamics?

A second question tests the stochastic bottleneck: what happens when KL regularization is removed?

## Inputs and leakage controls

Each sample provides only three observable five-second graph states as context. The model imagines six future states (30 seconds). Training targets are future observable states/edges and separately stored future LM/ATT&CK/pair truth.

The following never enter model input:

- CIC/scenario labels;
- ATT&CK truth;
- actor or target metadata;
- state IDs or timestamps;
- future states or edges;
- prior LM truth.

Scaling is fitted on original training contexts only. Host permutation augmentation is consistent across context, future, edge, and pair targets. Splits remain whole episodes. Every checkpoint and decision threshold is selected on validation; test is evaluation only.

## Matched regimes

1. **Published joint, dynamics-selected** — previous reported model trained jointly from scratch and selected by validation future-state plus edge loss.
2. **Frozen two-stage** — RSSM pretrained with future telemetry, reconstruction, edge, and KL losses. The RSSM is frozen and its existing linear LM/ATT&CK/pair heads alone are trained.
3. **Pretrained then unfrozen** — starts from the same validation-selected self-supervised RSSM and then allows all parameters to adapt under joint losses.
4. **Joint from scratch, matched selection** — random initialization, joint losses, and the same joint validation objective used for the unfrozen condition.
5. **Joint zero-KL** — same matched joint training but KL weight is zero.

Frozen and unfrozen comparisons use the same internal PyTorch head architecture, BCE weighting, augmentation, stochastic imagined trajectories, and optimizer family. This corrected an exploratory run that had used separately tuned scikit-learn heads for the frozen condition; those exploratory numbers are superseded.

## Test results

Lower state MAE is better. Higher AP/F1/top-1 is better.

| Regime | State MAE | Active MAE | Quiet MAE | Edge AP | LM F1 | LM AP | Pre-first-LM F1 | Pair top-1 | Mean spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Published joint, dynamics-selected | 0.280 | 1.031 | 0.130 | 0.222 | 0.800 | **0.910** | 0.769 | 0.143 | 0.076 |
| Frozen two-stage | 0.279 | 1.047 | **0.125** | 0.240 | 0.757 | 0.816 | 0.727 | 0.071 | 0.079 |
| Pretrained then unfrozen | **0.275** | 1.040 | **0.122** | 0.210 | 0.800 | 0.869 | 0.815 | 0.214 | 0.067 |
| Joint scratch, matched selection | 0.293 | 1.041 | 0.143 | 0.152 | **0.839** | 0.839 | 0.815 | 0.071 | 0.085 |
| Joint zero-KL | 0.280 | **0.984** | 0.139 | **0.316** | **0.839** | 0.904 | **0.846** | **0.286** | **0.025** |

The pair top-1 values correspond to 1/14, 3/14, or 4/14 positive test windows and therefore have high sampling uncertainty.

## ATT&CK test F1

| Regime | T1046 scan | T1110.001 password guessing | T1021.004 SSH |
|---|---:|---:|---:|
| Published joint, dynamics-selected | 0.683 | **0.772** | **0.839** |
| Frozen two-stage | 0.703 | 0.746 | 0.765 |
| Pretrained then unfrozen | 0.718 | 0.746 | 0.824 |
| Joint scratch, matched selection | 0.737 | 0.710 | 0.824 |
| Joint zero-KL | **0.757** | 0.714 | **0.839** |

## What was learned

### 1. Self-supervised latent dynamics contain useful security signal

With the RSSM frozen, linear heads reach LM F1 0.757 and LM AP 0.816. This is evidence that telemetry prediction/reconstruction produces a latent trajectory associated with future security behavior. It does not show that the representation is universally sufficient.

### 2. Unfreezing helps some downstream objectives

Relative to the matched frozen condition, unfreezing changes:

- LM F1: 0.757 → 0.800;
- pre-first-LM F1: 0.727 → 0.815;
- pair top-1: 0.071 → 0.214;
- state MAE: 0.279 → 0.275.

Edge AP decreases from 0.240 to 0.210, and not every ATT&CK score improves. Therefore, adaptation helps some security semantics but is not universally beneficial.

### 3. Pretraining changes the trade-off

Compared with the matched-selection joint-from-scratch model, pretraining then unfreezing improves state MAE (0.275 versus 0.293), edge AP (0.210 versus 0.152), LM AP (0.869 versus 0.839), and pair top-1 (0.214 versus 0.071). Thresholded LM F1 is lower (0.800 versus 0.839). Pretraining appears more useful for dynamics and ranking than for every threshold-dependent decision.

### 4. Removing KL improves several point metrics but damages the stochastic model

The zero-KL variant has the best active-state MAE, edge AP, pre-first-LM F1, and pair top-1 in this ablation. However, mean normalized rollout spread falls from 0.076 in the published model to 0.025—a roughly 67% reduction. The worst per-sample LM probability range under host relabeling rises to 0.814.

This does **not** justify declaring zero-KL the new model. It suggests the current KL weight/free-nats setting may over-regularize this tiny corpus and should be tuned on validation. Without KL alignment, the posterior and rollout prior no longer form the intended regularized latent-state model, and low spread is not evidence of calibrated confidence.

## What is deliberately excluded

This experiment does not add:

- a transformer;
- DANN/domain adaptation;
- Ridge/RSSM fusion;
- defensive actions;
- test-driven checkpoint selection.

DANN has no justified domain label in the current single controlled environment. Fusion requires validated complementary errors. Actions require intervention/no-intervention data.

## Limitations

- Only 24 controlled episodes and 12 LM events exist.
- Test has 90 overlapping windows and 14 LM-positive windows; windows are not independent episodes.
- The self-supervised checkpoint itself is selected on the same validation split used later for heads.
- Frozen-head validation loss continued improving slowly; the selected seed-7 checkpoint was epoch 947 of a fixed 1,000-epoch cap.
- The model remains a fixed-slot flattened encoder, not a permutation-equivariant graph network.
- Monte Carlo spread is diagnostic, not calibrated uncertainty.
- Multiple ablation variants increase the risk of over-interpreting one small test set, even though model selection was validation-only.

## Next experiment

Run a small validation-only grid over KL weight/free nats and report both performance and stochastic diagnostics. A candidate is acceptable only if it preserves useful spread and host stability while improving dynamics/semantic validation metrics. Then improve pair/edge prediction with a shared-weight pair decoder and collect matched hard-negative episodes.
