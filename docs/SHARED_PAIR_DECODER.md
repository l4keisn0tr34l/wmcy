# Shared-Weight Pair Decoder Experiment

## Status

Completed and rejected as the replacement pair head. It reduces score-level equivariance error but worsens validation/test pair ranking. This negative result narrows the problem to the slot-specific RSSM representation, not only the final output layer.

Artifacts:

```text
outputs/mvp_v2/rssm/shared_pair_decoder.json
models/mvp_v2_rssm_shared_pair.pt
```

Reproduce:

```bash
.venv/bin/python scripts/22_train_shared_pair_decoder.py
```

## Architecture

The old pair head gives each output slot independent weights over the flattened imagined latent trajectory.

The candidate instead gathers, for every directed source-target pair and all six imagined steps:

```text
predicted source-node features (18)
predicted destination-node features (18)
predicted directed-edge features (12)
```

This produces `6 × (18 + 18 + 12) = 288` values per candidate pair. One shared `288 -> 64 -> 1` MLP scores all six directed pairs.

```text
same scorer for pair 0
same scorer for pair 1
...
same scorer for pair 5
```

The pair head has 18,561 parameters; the complete model has 95,096. All non-pair parameters were initialized from the validation-selected lower-KL checkpoint and frozen. Automated checks verified every non-pair tensor remained bit-identical after training.

## Selection

Seeds 7/17/27 were trained with consistent host-permutation augmentation. Test data was not loaded until seed selection was complete.

Seed selection maximized on validation:

```text
permutation-mean pair AP
+ permutation-mean pair top-1
- permutation-mean pair equivariance MAE
```

Seed 7 was selected.

## Results across all six host relabelings

| Split/metric | Independent tuned head | Shared pair head |
|---|---:|---:|
| Validation mean pair AP | **0.279** | 0.271 |
| Validation mean top-1 | **0.269** | 0.218 |
| Validation mean equivariance MAE | 0.038 | **0.030** |
| Test mean pair AP | **0.267** | 0.232 |
| Test mean top-1 | **0.274** | 0.202 |
| Test mean equivariance MAE | 0.031 | **0.021** |

The shared head test top-1 ranges from 0/14 to 6/14 across equivalent relabelings. Identity-order top-1 is 3/14. Top-choice consistency after mapping back to original host coordinates remains near chance.

## Interpretation

The shared scorer removes direct output-slot-specific pair weights and measurably narrows probability differences under relabeling. However, it receives decoded node/edge features from a flattened fixed-slot encoder and decoder. Those features already change incorrectly when host slots are renamed. Sharing only the final scorer cannot repair the upstream representation.

Therefore:

- the experiment is useful as an architectural diagnosis;
- the shared head is not adopted;
- the previous 9/14 identity result remains rejected as robust evidence;
- next pair work should start with a permutation-equivariant node/edge encoder rather than adding more pair-head complexity.

## Remaining limitations

- Only 13 validation and 14 test pair-positive windows exist.
- The shared head is larger than the original linear head.
- Validation/test episodes are repeatedly inspected research splits.
- The decoder consumes predicted normalized feature slots, which remain topology-specific.
- New role-balanced episodes are required even after a graph encoder is implemented.
