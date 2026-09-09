# Frozen Passive Branch Model on V4

## Scope

The V3 validation-frozen outcome-conditioned checkpoint was applied without training or tuning:

```text
models/mvp_v3_outcome_branching_graph_rssm_validation.pt
SHA-256 4f0524b5c25a9bd0e4f591362be948f0df12fa684b5140d55a7abcb147be874b
```

Evaluation uses:

1. 180 overlapping windows from the 12 fresh passive-branching/direct-credential episodes;
2. the 10 eligible V4 pre-action contexts, with the chosen permit/block input deliberately withheld.

The LM decision threshold `0.3707732260` was frozen on V3 validation. The evaluator needed two wiring-error retries: the first failed before prediction, and the second formed passive prediction tensors but crashed before computing a metric because the metric helper was imported from the wrong module. Neither retry altered the checkpoint, threshold, cohort, or metric contract; this document reports the first completed metric artifact.

## Forecast results

| Metric | V4 passive/direct windows |
|---|---:|
| samples / LM-positive | 180 / 27 |
| expected state MAE | 0.248 |
| oracle branch state MAE | 0.237 |
| oracle coverage gain | 0.011 |
| branch state diversity | 0.501 |
| expected future-edge AP | 0.556 |
| LM F1 | 0.483 |
| LM AP | 0.472 |
| LM Brier | 0.107 |
| LM ROC AUC | 0.712 |

The outcome-conditioned branch matching the known LM label has state MAE `0.256`, versus `0.584` for the opposite semantic branch. This supports transferred alternative-future specialization. Oracle MAE remains a coverage diagnostic, not an inference-time score.

## Episode-level boundary

At the frozen threshold:

- stopped scan/guess episodes alerted: **3/3**;
- progressing scan/guess episodes alerted before the first positive horizon: **0/3**;
- matched legitimate SSH episodes alerted: **1/3**;
- direct-credential progressing episodes alerted before the first positive horizon: **0/3**.

The V3 precursor rule did not transfer as an early-warning discriminator. On the fresh passive matched-prefix pairs it alerts on every stopped episode and misses every progressing episode before LM enters the horizon. Direct credential movement also lacks the scan/guess precursor learned by this model.

Window-level cohort AP was `0.619` for passive branching and only `0.089` for direct credential movement. Overlapping windows are correlated, so the six pair-level outcomes are the more honest operational summary.

## Passive versus chosen-action input

On the same 10 pre-action contexts used for action evaluation, but without supplying the chosen action:

```text
passive branch state MAE: 0.141
passive LM AP/F1/Brier:    0.519 / 0.400 / 0.329
action scratch state MAE: 0.060
action scratch LM AP/F1:  1.000 / 1.000
```

Passive permit/block pair probabilities differ because matched captures are not exact clones, but the differences do not recover outcomes reliably. With the already-chosen intervention supplied, the action model forecasts the controlled outcome and gives lower factual future-state error on 10/10 contexts.

This is direct evidence for the project's observability claim:

> Passive telemetry can represent dangerous alternative futures, but it cannot reliably infer an unobserved future intervention or hidden intent. A decision known at forecast time is a legitimate conditioning variable and materially changes forecast quality.

It is not evidence that arbitrary post hoc metadata should enter the model; only an action actually chosen before its packet consequences is admissible.

## Limitations

- 180 windows arise from only 12 episodes and are heavily correlated;
- five action pairs remain after timing-only exclusion;
- all episodes use one three-host Docker topology;
- direct credential behavior is one narrow scenario family;
- branch semantics are outcome-supervised;
- Brier/ECE values are diagnostics, not deployment calibration;
- no enterprise or general causal-policy conclusion.

## Artifact

```text
outputs/mvp_v4/passive_branch/sealed_test.json
scripts/45_evaluate_passive_branch_v4.py
```
