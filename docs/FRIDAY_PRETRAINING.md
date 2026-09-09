# Fixed-Setting Friday Graph Dynamics Pretraining

## Status

**Training-fit initialization checkpoint only. No transfer/generalization claim.**

Implementation: `scripts/41_pretrain_graph_rssm_friday.py`.

## Protocol frozen before execution

```text
architecture: GraphRSSM, 358,115 parameters
source:       Friday train.npz only, 5,775 samples
seed:         41001
CUDA batch:   128
epochs:       exactly 100
optimizer:    Adam, 3e-4
augmentation: all consistent host relabelings
selection:    none
```

The script fails if Friday validation/test arrays exist. It never loads a lab split, V3 validation, or V3 test.

## Inputs

Three observable five-second graph states with 141 features, using the same global/node/edge feature contract as the lab model. Scaling is fitted on Friday training contexts only.

## Objective

```text
future six-step normalized state MSE: weight 1.00
full observed reconstruction MSE:     weight 0.25
future edge BCE:                       weight 0.25
Gaussian KL:                           weight 0.01
LM / ATT&CK / LM-pair losses:          weight exactly 0
```

Zero-filled semantic tensors exist only to satisfy the shared batch API; their losses have exactly zero contribution and cannot train semantic behavior.

## Training-only observed result

| Deterministic training metric | Initial | Epoch 100 |
|---|---:|---:|
| dynamics selection expression | 1.174 | 0.350 |
| future-state loss | 0.905 | 0.220 |
| reconstruction loss | 0.939 | 0.016 |
| edge loss | 1.076 | 0.516 |
| KL | 0.130 | 0.037 |

These are fits on the same connected capture used for optimization. They are not validation metrics.

Runtime on the RTX 3050 was 7m07.68s; maximum process RSS was 1,826,172 KB (mostly dataset loading/arrays, not VRAM evidence).

## Gates

```text
causal context max delta:           0
host-equivariance max delta:        4.47e-8
tiny-overfit mean loss:             1.131 → 1.016
semantic weights exactly zero:      true
CPU-portable checkpoint tensors:    required
```

## Outputs

```text
models/friday_graph_rssm_pretrained_fixed.pt
outputs/cic2017_friday_pcap/pretraining/fixed_pretraining.json
```

Checkpoint SHA-256:

```text
f8633c796f0523fc4537212ced3584d2f448ba55ac22be0cf94092a86e359ba0
```

## Correct interpretation

Observed fact: the model can fit substantial Friday graph/TCP dynamics under a fixed training protocol.

Hypothesis: this initialization may improve data efficiency or graph forecasting on fresh controlled episodes.

Not established: improved V4 dynamics, LM semantics, uncertainty calibration, attacker intent inference, or action effects. The checkpoint is only a candidate initializer for the predetermined V4 action-training protocol. It must not trigger another V3 test evaluation.
