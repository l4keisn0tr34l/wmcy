# Graph RSSM Semantic-Head Follow-ups

## Status

Two frozen-dynamics follow-ups are complete. Additional training of the original pooled-latent heads did not improve generalization. An invariant readout from decoded future graph states improved thresholded LM F1 but did not close the flattened model's LM ranking gap.

Artifacts:

```text
outputs/mvp_v2/graph_rssm/semantic_tuning.json
outputs/mvp_v2/graph_rssm/rich_semantic.json
models/mvp_v2_graph_rssm_semantic.pt
models/mvp_v2_graph_rssm_rich_semantic.pt
```

## Experiment A — Continue pooled-latent heads

All graph dynamics/decoder tensors were frozen bit-identically. Only LM, ATT&CK, and pair modules continued training with validation-only seed selection.

```text
initial graph LM F1/AP: 0.476 / 0.738
continued-head LM F1/AP: 0.455 / 0.707
```

This is a negative result. Additional optimization of the same pooled representation did not improve held-out semantics.

## Experiment B — Read the imagined future graph directly

For every predicted future state, the invariant semantic readout uses:

```text
decoded global features
node-feature mean and max over hosts
edge-feature mean and max over directed pairs
```

Mean/max pooling is invariant to host order while retaining outlier intensity that simple latent means can lose. All graph dynamics, state/edge decoders, and pair modules remain frozen.

Test result:

| Metric | Initial graph | Rich invariant readout | Flattened RSSM |
|---|---:|---:|---:|
| State MAE | 0.252 | 0.253* | 0.280 |
| Edge AP | 0.394 | 0.392* | 0.222 |
| LM F1 | 0.476 | **0.667** | 0.800 |
| LM AP | 0.738 | **0.744** | 0.910 |
| Pre-first-LM F1 | 0.526 | **0.667** | 0.769 |
| Pair top-1 | 0.357 | 0.357 | 0.143 |

`*` State/edge tensors are frozen; tiny differences are Monte Carlo estimation noise.

ATT&CK test F1:

```text
T1046       0.737
T1110.001   0.786
T1021.004   0.722
```

Episode behavior remains:

```text
2/2 progressing episodes detected
23.9-second mean exact lead
2/4 non-progressing episodes alert
```

## Interpretation

The graph model is clearly stronger and methodologically safer for future-state, edge, and pair dynamics. Its semantic validation AP is close to its test AP (~0.74), whereas the flattened model's test LM AP 0.91 is much higher than its validation AP 0.74. The flattened semantic advantage may therefore include split luck or residual slot-specific behavior rather than universally better semantics.

Do not discard the graph model because it does not reproduce the flattened test score. The more defensible conclusion is:

- graph structure solves the identified equivariance failure and improves world prediction;
- invariant semantic readout reaches useful but not sufficient LM performance;
- additional hard-negative/progression episodes are now more valuable than repeatedly tuning heads on V2.

## Limitations

- Test is no longer a fresh holdout after many model iterations.
- Only 14 positive test windows exist.
- Rich readout depends on imperfect decoded future features.
- False alerts remain unchanged.
- Public dynamics pretraining and new controlled semantic episodes are not yet included.
