# Frozen V4 Action-Model Protocol

## Status

**Protocol and two checkpoints frozen before V4 action-test access.** Training used only 12 action-train samples. No V4 test array or episode was loaded by the training script.

## Architecture

`ActionGraphRSSM` extends the shared equivariant GraphRSSM. After observing three passive states, it encodes:

- action type: `[permit_ssh, block_ssh]`;
- one selected directed action pair among six anonymous host pairs.

The action-pair encoder is shared across edges. Outgoing/incoming action messages shift node latents; pooled action information shifts global/stochastic latents; the selected edge latent is shifted directly. The same six-step graph transition/decoders then forecast future state, edges, ATT&CK, LM, and LM pair.

When the action pair is relabeled consistently with hosts, action forecasts remain permutation equivariant.

## Fixed training protocol

```text
samples:       12 (six permit, six block)
seed:          43001
batch:         12
optimizer:     Adam 3e-4
epochs:        exactly 400
selection:     none
```

Both prespecified initializations proceed to sealed test:

1. scratch;
2. fixed Friday dynamics checkpoint.

The train-context scaler and action-layer random initialization are identical. Host permutation augmentation also permutes the action pair and graph/pair targets consistently.

Loss weights:

```text
future state 1.00
reconstruction 0.10
future edge 0.25
KL 0.01
LM 0.20
ATT&CK 0.10
LM pair 0.10
```

## Training-fit diagnostics—not generalization

| Train-only metric | Scratch | Friday initialized |
|---|---:|---:|
| final total objective | 0.081 | 0.027 |
| normalized state MAE | 0.047 | 0.043 |
| active state MAE | 0.157 | 0.108 |
| edge AP | 1.000 | 1.000 |
| LM F1 | 1.000 | 1.000 |
| pair top-1 | 1.000 | 1.000 |
| same-context permit−block LM probability | 1.000 | 0.994 |
| same-context state difference | 0.111 | 0.131 |

These near-perfect values show strong fit to a deterministic 12-sample lab intervention set. They must not be presented as test performance. The LM head can learn the direct action/outcome relationship; future graph-state accuracy is therefore essential in sealed evaluation.

## Invariants

| Gate | Scratch | Friday initialized |
|---|---:|---:|
| future perturbation context/forecast delta | 0 | 0 |
| host/action-pair equivariance max delta | 5.96e-8 | 1.19e-6 |
| finite gradient tensors | 93 | 93 |
| nontrivial state action effect | pass | pass |
| permit risk above block | pass | pass |

## Frozen checkpoints

```text
models/mvp_v4_action_graph_rssm_scratch.pt
SHA-256 f69ece2d24ac07593d35666a83ea52ef8a768d1f51d4dcab13e7c1dc9f1c876f

models/mvp_v4_action_graph_rssm_friday_initialized.pt
SHA-256 6b4cf070da873f04cd1fecb17cf351b9f9ba07bcbf45275b80aa69a8b100e351
```

Both checkpoints contain CPU tensors, the shared train scaler, fixed `0.5` threshold, and their train-derived LM threshold. Neither is selected over the other.

## Frozen sealed-test contract

`scripts/44_evaluate_action_graph_rssm_v4.py` verifies checkpoint hashes before reading test and reports both models:

- all/active/quiet normalized state MAE;
- future-edge AP;
- LM Brier/AP and F1 at fixed 0.5 and frozen train threshold;
- LM pair AP/top-1;
- metrics by factual permit/block action;
- same-context permit-versus-block LM, graph, and edge changes;
- whether the factual chosen action gives lower future-state error than the opposite action;
- all-six-permutation equivariance.

The evaluator refuses to overwrite an existing sealed report. V4 action test must be built once with an explicit unlock only after this protocol commit.

## Limitations

- only six train intervention pairs;
- permit/block deterministically controls SSH outcome in this lab;
- matched packet captures are not exact cloned prefixes;
- action semantics are supervised rather than discovered;
- no enterprise or calibrated-policy claim.
