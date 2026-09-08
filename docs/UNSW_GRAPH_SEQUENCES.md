# UNSW Three-Host Graph Sequences

## Result

`scripts/27_build_unsw_graph_sequences.py` created lab-compatible chronological graph tensors from canonical UNSW observables:

```text
public train (January capture):    1,481 samples × 3 context × 141 features
public validation (February):      1,383 samples × 3 context × 141 features
future target per sample:          6 states × 141 features
future directed-edge target:       6 states × 6 ordered pairs
```

The two connected capture groups remain disjoint. Overlapping 45-second samples are correlated within each group; they are not independent episodes.

## Causal roster construction

For every sample, the three-host roster is selected from the 15-second context only:

1. take the most frequent observed context edge;
2. add the highest-degree third context host;
3. deterministically shuffle the three slot positions;
4. retain only flows whose two endpoints are in that roster;
5. roll forward six five-second states without changing the roster.

Future activity and attack labels are never consulted. Raw IP strings appear only as one-way hashes in the separate audit manifest and never enter tensors.

## Feature contract

The output follows the exact 141-field lab layout:

```text
15 global + 3 × 18 node + 6 × 12 directed-edge features
```

UNSW lacks explicit per-flow SYN/ACK/RST/FIN count fields. Those positions are zero and marked unavailable in metadata; they are not fabricated from broad labels or connection state.

New-edge/new-neighbor features are causal within each sample: a pair is new only on its first state appearance.

## Outputs

```text
outputs/unsw/graph_sequences/train.npz
outputs/unsw/graph_sequences/validation.npz
outputs/unsw/graph_sequences/sample_manifest.csv
outputs/unsw/graph_sequences/feature_metadata.json
```

The NPZ files contain only context observations, future observations, and future edge presence. They contain no attack, ATT&CK, or LM targets.

## Limitation

Traffic intensity differs sharply between the January and February captures. This is useful domain-shift pressure but means public-validation loss may be pessimistic. Public pretraining must still be judged by downstream controlled-lab validation before reporting the already-inspected V2 test.
