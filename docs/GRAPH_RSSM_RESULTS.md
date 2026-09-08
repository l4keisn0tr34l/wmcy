# Permutation-Equivariant Graph RSSM — First V2 Result

## Status

Implemented and trained. The graph RSSM materially improves future-state/edge forecasting and enforces host-relabeling equivariance, but its first jointly trained semantic LM head is weaker than the flattened RSSM. This checkpoint is a successful dynamics foundation, not yet the final combined model.

Artifacts:

```text
models/mvp_v2_graph_rssm.pt
outputs/mvp_v2/graph_rssm/metrics.json
outputs/mvp_v2/graph_rssm/predictions.npz
outputs/mvp_v2/graph_rssm/episode_alerts.csv
outputs/mvp_v2/graph_rssm/sample_predictions.csv
```

Reproduce:

```bash
.venv/bin/python scripts/23_train_graph_rssm.py
```

## Input

Each observable state is parsed structurally rather than treated as an undifferentiated vector:

```text
15 global features
3 hosts × 18 node features
6 directed pairs × 12 edge features
```

Three five-second states form context; the model rolls its prior forward six steps (30 seconds). Labels, scenarios, actor/target metadata, timestamps, state IDs, and future values do not enter `forecast(context)`.

## Transformation

- Global features use a global encoder.
- One shared node encoder processes every host.
- One shared edge encoder processes every directed pair together with its source/destination node encodings.
- Incoming/outgoing edge messages are aggregated at each node.
- Recurrent state remains structured as one invariant global hidden state, three equivariant node states, and six equivariant edge states.
- A 16-dimensional stochastic global latent uses prior/posterior KL with weight 0.01 and no free-nats floor.
- Shared decoders reconstruct every future node and edge.
- Invariant pooled graph trajectories feed LM/ATT&CK heads; one shared pair head scores each directed pair.

Train-context normalization shares statistics across node slots and separately across edge slots. This removes a subtle source of slot dependence in preprocessing.

## Tests

```text
future-to-context causal delta:       0.0
deterministic equivariance max delta: 4.47e-8
tiny-overfit loss:                    1.780 -> 1.421
parameter count:                      358,115
```

All state, edge, and pair permutation maps are tested. Under 20 stochastic rollouts, mapped prediction equivariance MAE remains approximately 1e-8 because stochastic noise is global/invariant.

## Test comparison

| Metric | Graph RSSM | Published flattened RSSM | Ridge | Persistence |
|---|---:|---:|---:|---:|
| State MAE | **0.252** | 0.280 | 0.354 | 0.384 |
| Active-state MAE | **0.895** | 1.031 | 1.089 | 1.137 |
| Quiet-state MAE | **0.124** | 0.130 | 0.207 | 0.233 |
| Edge AP | **0.394** | 0.222 | 0.230 | — |
| LM F1 | 0.476 | **0.800** | 0.645 | — |
| LM AP | 0.738 | **0.910** | 0.581 | — |
| Pre-first-LM F1 | 0.526 | **0.769** | 0.640 | — |
| Pair top-1 | **0.357 (5/14)** | 0.143 | 0.000 | — |

Unlike the flattened models, graph pair top-1 is exactly 0.357 under every host relabeling. The result is still only 5/14 correlated positive windows and requires new episodes.

## ATT&CK test F1

```text
T1046       0.718
T1110.001   0.746
T1021.004   0.737
```

## Episode behavior

```text
progressing test episodes detected: 2/2
mean exact warning lead:            23.9 seconds
non-progressing test episodes alert: 2/4
```

The graph model still does not solve scan-only/failed-guessing false alerts.

## Uncertainty

```text
mean state spread:              0.036
active spread:                  0.064
quiet spread:                   0.031
spread/error correlation:       0.722
```

Spread is lower than the flattened KL-tuned candidate but remains related to error. It is not calibrated uncertainty.

## Interpretation

This establishes that preserving graph structure improves the primary world-model objective and future edge forecasting while eliminating fixed-slot host relabeling failure. The weaker LM head does not invalidate the dynamics result; it shows that the jointly selected checkpoint optimized a different trade-off and that pooled semantic heads need separate training/fine-tuning.

## Information deliberately excluded

- no public attack labels;
- no current/future scenario identity;
- no actor or target truth;
- no state index or absolute time;
- no future graph as input;
- no fixed host-ID embedding.

These exclusions prevent the graph model from memorizing attack roles or schedules.

## Limitations

- The current graph size is still fixed at three known hosts.
- Only 24 controlled episodes and 12 LM events exist.
- The model has not yet used public graph-dynamics pretraining.
- Validation/test episodes have been inspected in prior experiments.
- Semantic heads use pooled graph trajectories and may need richer node-to-global attention.
- Exact pair ranking still has only 14 positive test windows.

## Semantic follow-up

Frozen continuation of the original pooled heads did not improve test LM AP. A decoded-future invariant mean/max readout raises LM F1 from 0.476 to 0.667 and pre-first-LM F1 from 0.526 to 0.667 while preserving dynamics/equivariance, but LM AP remains approximately 0.744. See `docs/GRAPH_SEMANTIC_RESULTS.md`.

The immediate bottleneck is now varied semantic data—especially matched non-progression negatives—rather than additional V2-only head tuning.
