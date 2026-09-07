# Compact RSSM V2 Experiment

Date: 2026-09-07

## Purpose

Test a genuine recurrent stochastic latent world model over chronological network graph states. This is not a current-flow attack classifier and is independent of the unavailable senior implementation.

## Input and temporal contract

```text
input:  3 observable graph states x 5 seconds = 15 seconds
rollout: 6 future states x 5 seconds = 30 seconds
state width: 141 observable global/node/edge features
```

No scenario, actor, target, state ID, timestamp, ATT&CK truth, LM truth, or future observation enters inference. Normalization is fitted on original training contexts only. Whole episodes remain isolated in 12/6/6 train/validation/test splits.

## Architecture

```text
141-feature observation
    -> 64-dimensional MLP embedding
    -> posterior q(z_t | h_t, o_t)

previous stochastic z
    -> GRUCell -> deterministic h_t (64)
    -> prior p(z_t | h_t), diagonal Gaussian z_t (16)

[h_t, z_t] (80)
    -> observable-state decoder (141)
    -> directed-edge presence head (6)

six imagined [h,z] states
    -> LM-within-horizon head
    -> three ATT&CK heads
    -> six directed LM-pair heads
```

The posterior observes chronological states during training. Forecasting observes only the three context states and then rolls the stochastic prior open-loop for six steps.

## Training regime

- CPU PyTorch 2.9.1; external GPU not required.
- Adam, learning rate `3e-4`, batch 32, gradient clip 10.
- Candidate seeds 7, 17, 27.
- Maximum 400 epochs, validation patience 50.
- Selection metric: validation open-loop grouped state MSE plus 0.25 edge BCE.
- Selected seed 7, epoch 240; test was evaluated only after selection.
- Runtime: approximately 76 seconds on local CPU.
- Consistent random host permutation augmentation is applied to context, future states, edge targets, and LM-pair targets.

Losses:

```text
open-loop future global/node/edge grouped MSE  weight 1.00
posterior reconstruction grouped MSE          weight 0.25
future edge-presence BCE                      weight 0.25
posterior-prior KL, one free nat               weight 0.10
future LM BCE                                 weight 0.20
ATT&CK multi-label BCE                        weight 0.10
LM-pair BCE                                   weight 0.10
```

The dynamics/reconstruction/KL part is self-supervised from telemetry. Because semantic losses backpropagate into the shared latent model, the complete regime is hybrid/self-supervised-plus-auxiliary-supervision, not purely self-supervised.

## Correctness checks

```text
future-observation perturbation -> context latent max change: 0.0
tiny-overfit mean initial loss: 1.872
tiny-overfit mean final loss:   1.752
all losses/checkpoint outputs finite
```

The causality test verifies that altering future observations cannot change the inferred context latent or forecast starting point.

## Results

### Primary future-state rollout

| Test metric | RSSM | PCA/Ridge | Persistence |
|---|---:|---:|---:|
| All normalized state MAE | **0.280** | 0.354 | 0.384 |
| Active-state MAE | **1.031** | 1.089 | 1.137 |
| Quiet-state MAE | **0.130** | 0.207 | 0.233 |

RSSM aggregate MAE by horizon versus persistence:

```text
horizon       +5s    +10s   +15s   +20s   +25s   +30s
RSSM         0.371   0.362  0.320  0.236  0.205  0.189
persistence  0.391   0.421  0.400  0.353  0.365  0.373
```

On active future states, RSSM is worse than persistence at +5 and +10 seconds, then better at +15 through +30 seconds. Quiet states are 450/540 test future state-windows, so aggregate metrics must always be accompanied by active metrics.

### Security interpretation from imagined futures

| Test metric | RSSM | PCA/Ridge |
|---|---:|---:|
| Future-LM F1 | **0.800** | 0.645 |
| Pre-first-LM F1 | **0.769** | 0.640 |
| Future-LM AP | **0.910** | 0.581 |
| Future-edge AP | 0.222 | **0.230** |
| LM-pair top-1 | **0.143** | 0.000 |

Validation future-LM F1 is 0.741 and pre-first-LM F1 is 0.720. The test threshold, 0.668, was selected on validation.

ATT&CK test results:

```text
T1046 Network Service Discovery    F1 0.683   AP 0.802
T1110.001 Password Guessing        F1 0.772   AP 0.784
T1021.004 SSH lateral movement     F1 0.839   AP 0.905
```

### Stochastic uncertainty diagnostics

Twenty Monte Carlo prior rollouts give:

```text
mean normalized state std                  0.076
active-state mean std                      0.125
quiet-state mean std                       0.066
state spread / absolute-error correlation  0.613
mean LM-probability std                    0.061
```

Higher spread on active states and positive error correlation are useful signals, but Monte Carlo spread is not yet calibrated probabilistic uncertainty.

### Host permutation sensitivity

Across all six equivalent host relabelings:

```text
LM AP range                         0.725-0.950
LM F1 range                         0.800-0.875
state MAE range                     0.280-0.292
mean per-sample LM probability range 0.111
maximum per-sample range             0.680
```

Mean sensitivity is slightly lower than Ridge's 0.120, but individual outliers remain. The flattened fixed-slot encoder is not permutation equivariant.

## Decision

Retain RSSM as the preferred next model candidate because it materially improves multi-step state prediction, active-state prediction, LM forecasting, and useful stochastic diagnostics. Do not claim universal superiority: edge AP is slightly worse than Ridge and exact pair ranking remains poor.

## Limitations and next work

1. Only 24 controlled episodes and 12 LM events exist.
2. Overlapping sequence windows are correlated.
3. The encoder is flattened fixed-slot MLP, not graph message passing.
4. Security supervision jointly shapes the latent model; a pure self-supervised/two-stage ablation is still needed.
5. Edge and LM-pair decoders need improvement.
6. Episode-level RSSM alert/lead-time reporting and a V2 replay are still needed.
7. No defensive action variable or intervention data exists.
8. No enterprise, cross-domain, or unseen-playbook claim is supported.

Artifacts:

```text
models/mvp_v2_rssm.pt
outputs/mvp_v2/rssm/metrics.json
```
