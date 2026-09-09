# Friday Causal Graph Sequences

## Status

Complete train-only public dynamics sequence build. No model has been trained on it yet.

## Input

```text
outputs/cic2017_friday_pcap/canonical/observations.csv.gz
2,102,560 chronological observable directed events
```

The builder reads timestamp, endpoint/port, packet/byte, duration, and TCP flag fields only. It never reads CIC CSV labels.

## Transformation

`scripts/39_build_friday_graph_sequences.py`:

1. assigns events to dense five-second windows inside complete capture bounds;
2. constructs every candidate as three context states plus six future states;
3. chooses three hosts from context only:
   - endpoints of the most frequent observed directed context edge;
   - the highest-degree remaining context host;
4. deterministically shuffles those hosts into anonymous slots using dataset ID and sample start;
5. constructs the induced three-host global/node/edge graph for all nine states;
6. preserves observed SYN/ACK/RST/FIN counts;
7. marks a directed edge new only in its first five-second capture window;
8. emits future graph states and edge presence as self-supervised targets.

The first-seen pair map is precomputed for efficiency, but the value used for a current edge is its minimum observed timestamp and is exactly reproducible from telemetry up to that edge's first appearance. Future roster activity is never consulted.

## Output

```text
outputs/cic2017_friday_pcap/graph_sequences/train.npz
outputs/cic2017_friday_pcap/graph_sequences/sample_manifest.csv
outputs/cic2017_friday_pcap/graph_sequences/feature_metadata.json
```

Observed shapes:

```text
context_states:       [5775, 3, 141]
future_states:        [5775, 6, 141]
future_edge_presence: [5775, 6, 6]
```

Additional profile:

```text
candidate windows skipped for <3 context hosts: 13
unique directed pairs in capture:              35,297
future windows with induced flows:              57.78%
stride:                                          5 seconds
configured internal CIDR:                       192.168.10.0/24
full build runtime:                              7m22.29s
maximum RSS:                                     1,466,268 KB
```

The arrays are finite; edge targets are binary; event-derived SYN/RST/FIN features are nonzero; no forbidden semantic field appears in feature metadata. `scripts/40_test_friday_graph_sequences.py` also checks canonical-file lineage, chronology, capture bounds, causal novelty, and context-only roster behavior.

## Why train only

Friday is one connected working-hours capture. Randomly splitting its overlapping windows would leak adjacent traffic and host behavior. There is no independent Friday capture group in the current local source, so the output intentionally contains no `validation.npz` or `test.npz`.

Any future training must use fixed pretraining settings or select against an independent development source. It must not use frozen V3 test for selection. Transfer evidence should come from the predetermined V4 protocol.

## What is excluded

- CIC attack labels;
- ATT&CK and LM truth;
- future host activity during roster selection;
- raw host identities from model arrays;
- absolute timestamp/state index from model arrays;
- preprocessing statistics (not fitted at this stage).

Roster hashes and UTC times exist only in the audit manifest.

## Limitations

- Three hosts are an induced subgraph of a much larger capture; relevant hosts/edges can be omitted.
- Frequent-edge roster selection favors active hubs and is not proven optimal for attacker progression.
- Five-second-stride samples overlap heavily and are correlated.
- The configured internal CIDR depends on the documented CIC topology.
- Packet-bucket events differ from CICFlowMeter bidirectional connection records.
- `flow_duration` remains within-one-second-bucket duration.
- No public progression/LM semantic result can be claimed.

This dataset is suitable for observable graph-dynamics pretraining, not semantic supervision.
