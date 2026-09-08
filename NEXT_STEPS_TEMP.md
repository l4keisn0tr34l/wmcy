# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Resumed checkpoint

UNSW canonicalization is complete and committed at `609a7a5`; repository clean. All 2,540,047 rows are in five chronological segments grouped into two non-overlapping capture units. No labels enter observations.

## Current code-writing batch: UNSW graph-dynamics sequences

Implement `scripts/27_build_unsw_graph_sequences.py`:

1. Read only canonical observable files; never read row ground truth.
2. Merge segments by the two connected capture groups and stable-sort/deduplicate overlapping source boundaries.
3. Create 45-second samples: three 5-second context states plus six future states.
4. Select each three-host roster using context traffic only, never future activity.
5. Randomize/anonymize roster slot order deterministically per sample; raw IP values never enter tensors.
6. Retain only directed flows between selected hosts for a consistent three-node induced graph.
7. Build the exact lab-compatible 141-feature layout.
8. Leave unavailable UNSW SYN/ACK/RST/FIN counts as explicit unavailable-zero fields and record this domain limitation; do not infer flags from attack labels/state.
9. Compute edge novelty only from the sample's past/current chronology.
10. Split by capture group: January capture for public train, February capture for public validation. Never randomly split overlapping windows.
11. Write train/validation NPZ, sample manifest, feature metadata, and profile under `outputs/unsw/graph_sequences/`.

## Validation

- exact `[N,3,141]` context and `[N,6,141]` future shapes;
- finite values and monotonic five-second states;
- no label/attack/ATT&CK/LM fields;
- roster selected from context only;
- host slot order deterministic but anonymized;
- train/validation capture groups disjoint;
- causal edge novelty;
- compile, bounded smoke, full build, inspect output.

## Following experiment

Pretrain the graph RSSM only on UNSW future-state/reconstruction/edge/KL losses, then fine-tune on controlled V2 telemetry + semantic truth. Compare against graph RSSM trained only on V2 using validation first; current V2 test is reporting-only and no longer a fresh selection set.
