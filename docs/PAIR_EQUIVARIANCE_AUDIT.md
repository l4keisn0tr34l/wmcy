# Pair and Host-Relabeling Equivariance Audit

## Status

Completed after KL tuning. The tuned pair top-1 gain fails the host-relabeling robustness gate and must not be presented as robust graph reasoning.

Artifact:

```text
outputs/mvp_v2/rssm/pair_equivariance_audit.json
```

Reproduce:

```bash
.venv/bin/python scripts/21_audit_pair_equivariance.py
```

## Question

If host slots are consistently renamed, a future source-target prediction should rename in the same way. The physical scenario has not changed.

For all six permutations of the three hosts, the audit:

1. consistently permutes observable context node/edge slots;
2. permutes future edge/pair targets using the matching directed-pair map;
3. runs 100 Monte Carlo forecasts without retraining;
4. maps pair choices back to original coordinates;
5. compares states, edge probabilities, LM probabilities, and pair scores with reordered identity predictions.

Ground truth is used only after forecasting for metrics.

## Main result

### KL-tuned candidate

Across six equivalent test relabelings:

```text
pair top-1:       0.143 to 0.571  (2/14 to 8/14)
pair top-1 mean:  0.274
pair score equivariance MAE mean: 0.031
pair-choice consistency mean including identity: 0.178
```

Identity contributes 1.0 consistency to the six-way mean. Across the five non-identity permutations, only about 1.3% of sample top choices map back to the identity top choice.

The earlier 20-rollout 9/14 and this audit's 100-rollout identity 8/14 differ because finite Monte Carlo estimates can change close rankings. More importantly, equivalent host relabeling changes the answer substantially.

### Published joint RSSM

```text
pair top-1:       0.000 to 0.786  (0/14 to 11/14)
pair top-1 mean:  0.250
pair score equivariance MAE mean: 0.045
pair-choice consistency mean including identity: 0.183
```

This model is also strongly slot-sensitive.

## Identity-split support

The 14 positive test windows contain true pairs only in two of six identity slots:

```text
slot 1: 12 positive memberships
slot 2:  6 positive memberships
other slots: 0
```

Some windows contain both true hops, so memberships sum to 18 while positive windows total 14. Episode-level splitting is correct, but only two progressing test episodes cannot cover all directed pairs simultaneously. Host augmentation is therefore essential—and current architecture does not convert it into equivariance.

## Other outputs also vary

For the tuned candidate across test relabelings:

```text
edge AP: 0.133 to 0.284
LM AP:   0.733 to 0.917
state equivariance MAE: 0.000 to 0.075 normalized units
```

LM ranking remains comparatively useful, but predictions are not invariant enough for strong host-independent claims.

## Why this happens

The current model flattens fixed node and edge slots into a 141-feature vector. Its encoder, decoder, edge head, and pair head all have slot-specific weights. Consistent permutation augmentation encourages robustness but does not mathematically enforce it.

The current pair head is:

```text
all six imagined latent features, flattened
    -> one linear layer
    -> six independently weighted pair logits
```

Nothing forces one source-target pair to be processed with the same function as another pair.

## Decision

Reject the tuned 0.643 pair top-1 as a headline. Retain it only as the identity-order point estimate with an explicit failed robustness audit.

Next:

1. replace independent pair-output weights with a shared pair scorer;
2. build each candidate pair representation from predicted source-node, destination-node, and directed-edge future features;
3. apply the same scorer to all six directed pairs;
4. evaluate identity accuracy and all-permutation equivariance;
5. ultimately replace the flattened state encoder/decoder with permutation-equivariant graph message passing.

A shared pair head alone cannot make the fixed-slot encoder equivariant, but it removes one direct source of output-slot memorization.

## Limitations

- Only 13 validation and 14 test positive windows exist.
- Validation/test episodes have been inspected repeatedly.
- One hundred rollouts reduce but do not eliminate Monte Carlo noise.
- Pair equivariance failure is an architectural diagnostic, not proof that every aggregate RSSM output is unusable.
