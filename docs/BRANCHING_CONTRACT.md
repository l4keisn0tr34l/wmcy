# Branch-Aware Forecasting Contract

## Why

V3 demonstrates that the same dangerous discovery/guessing prefix can stop or continue to SSH. A single mean future and one binary LM decision conflate two questions:

1. Is this prefix dangerous?
2. Will LM actually complete inside the next 30 seconds?

The first is operational risk; the second can depend on unobserved future action.

## Forecast representation

A branching model produces `K` candidate graph futures:

```text
branch probability π_k(context)
future states S_hat[k, 1:H]
future edges E_hat[k, 1:H]
LM/ATT&CK/pair risk per branch
```

Branch weights must use context only. Future telemetry/truth is used only to score training targets.

## Metrics

### Inference-time expected forecast

Use branch probabilities to compute the expected future. Report normalized MAE and edge AP. This is the deployable point forecast.

### Oracle best-branch coverage

For each realized target, choose the closest predicted branch and report its MAE. This measures whether the candidate set covered reality. It is not an inference-time score and must never be presented as ordinary prediction accuracy.

### Diversity and collapse

Report:

- mean pairwise branch MAE;
- weighted branch standard deviation;
- best-branch utilization;
- effective number of utilized branches;
- mean inference weights/effective weight count.

Useful branches need both nontrivial diversity and target coverage. Diversity alone can be meaningless noise.

### Exact LM outcome

Report proper scoring and discrimination:

- Brier score;
- log loss;
- diagnostic ECE/reliability bins;
- AP and validation-thresholded precision/recall/F1.

### Dangerous-prefix operational burden

Separately report progressing-episode detection and stopped-prefix alert burden. An alert on a stopped dangerous prefix is false for the narrow eventual-outcome target but can still be valid risk detection.

## Existing RSSM validation baseline

Twenty ordinary stochastic V3 scratch-RSSM draws, weighted uniformly:

```text
expected state MAE:          0.3250
oracle best-draw MAE:        0.3131
oracle gain:                 0.0119
pairwise draw diversity:     0.0341
weighted draw std:           0.0291
exact-LM Brier:              0.2048
exact-LM ECE:                0.2286
mean LM draw std:            0.0114
stopped/progress alert rate: 3/3 and 3/3
```

The best draw is spread across 18.2 effective draws, suggesting small stochastic perturbations rather than a small number of interpretable modes. Oracle improvement is only 0.012 normalized MAE, and LM draw variation is very small. This motivates explicit learned branches.

## Methodological scope

These values use V3 validation only. `scripts/33_evaluate_branching_contract.py` does not load V3 test. With only 90 correlated validation windows, ECE is diagnostic rather than calibration evidence.
