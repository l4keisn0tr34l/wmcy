# V3 Matched-Prefix Graph-RSSM Results

## Data and protocol

V3 contains 48 validated equal-duration controlled episodes:

```text
train:       36 episodes / 540 sequences / 91 LM-positive horizons
validation:   6 new episodes /  90 sequences / 19 LM-positive horizons
test:         6 new episodes /  90 sequences / 18 LM-positive horizons
```

All previously inspected V2 episodes are declared training data. Validation and test each contain three same-seed pairs:

```text
scan_guess_then_stop: T1046 → T1110.001 → no successful movement
one_hop:              T1046 → T1110.001 → T1021.004 movement
```

The pair shares roles and seeded control flow, but real nmap/SSH/runtime variation means event starts are not numerically identical.

Two regimes were declared before model test prediction:

1. graph RSSM from scratch;
2. the same per-seed semantic initialization with only non-semantic graph dynamics replaced by the observable-only UNSW checkpoint.

Seeds 7/17/27 and epochs were selected independently on V3 validation joint objective. Each LM threshold was frozen from validation Monte Carlo predictions. Model predictions on test were then produced once for both regimes. Before training, test arrays were opened only for schema, finiteness, shape, and expected-count integrity—not model evaluation; this is disclosed rather than calling the file literally untouched.

## Fresh-test results

| Metric | Scratch graph RSSM | UNSW-initialized |
|---|---:|---:|
| normalized state MAE ↓ | **0.294** | 0.321 |
| active-state MAE ↓ | **0.556** | 0.579 |
| quiet-state MAE ↓ | **0.130** | 0.159 |
| future edge AP ↑ | **0.808** | 0.796 |
| LM F1 ↑ | 0.560 | **0.596** |
| LM AP ↑ | 0.477 | **0.504** |
| pre-first-LM F1 ↑ | 0.571 | **0.596** |
| pair top-1 | 0.333 (6/18) | 0.333 (6/18) |
| mean state spread | **0.026** | 0.022 |
| spread/error correlation | **0.768** | 0.533 |
| progressing episodes detected | 3/3 | 3/3 |
| stopped episodes alerting | 3/3 | 3/3 |
| mean exact lead | 23.5 s | 23.5 s |

The scratch model was also better on the prespecified validation joint objective (`0.952` versus `1.060`) and validation LM AP (`0.611` versus `0.492`). Public initialization is therefore **not selected** for V3. Its small test LM increase does not override worse validation, dynamics, edges, and stochastic diagnostics.

Edge AP is not directly comparable with V2 because V3 test edge prevalence is 0.131 versus approximately 0.060 in V2.

## Matched-pair audit

On test episode maxima:

```text
Scratch:
  stopped mean max risk      0.9343
  progressing mean max risk  0.9348
  mean absolute pair gap     0.0027

UNSW initialized:
  stopped mean max risk      0.9272
  progressing mean max risk  0.9202
  mean absolute pair gap     0.0165
```

Both members alert in every test pair. The model does not reliably infer which dangerous precursor will be followed by the scenario controller's successful SSH.

## Interpretation: an observability boundary

This result is not simply “the classifier failed.” At a shared discovery/guessing prefix, the later external decision to execute SSH is absent from passive network telemetry. Two futures remain plausible:

```text
same observed prefix → attacker stops
                     → attacker moves laterally
```

A passive world model should represent this branching compromise risk; it cannot deterministically recover unobserved future intent. Alerting on the stopped member can therefore be a false positive for the narrow “LM completes in 30 seconds” target while still being a defensible risk alert.

To distinguish intervention-dependent futures, the dataset needs additional causal variables such as attacker action, defender action, credential state, or paired intervention/no-intervention trajectories. The immediate operational objective should separately score:

- dangerous-progression risk;
- exact eventual LM outcome;
- uncertainty over branching futures.

## Equivariance qualification

Mapped pair probabilities remain equivariant to approximately `1e-8`. However, some pair scores are nearly tied, so tiny Monte Carlo/floating-point changes can change `argmax`; permutation top-1 ranges from 6/18 to 7/18. Claim score-level equivariance, not perfectly stable discrete ranking.

## Artifacts

```text
models/mvp_v3_graph_rssm_scratch.pt
models/mvp_v3_graph_rssm_unsw_pretrained.pt
outputs/mvp_v3/episode_manifest.csv
outputs/mvp_v3/sequences/
outputs/mvp_v3/graph_rssm/comparison.json
outputs/mvp_v3/graph_rssm/paired_prefix_audit.json
outputs/mvp_v3/graph_rssm/*_episode_alerts.csv
```

No further model or threshold tuning should use this test split.
