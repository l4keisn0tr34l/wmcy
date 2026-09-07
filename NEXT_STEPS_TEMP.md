# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Completed in the latest batch

- `lab/run_episode.sh` now enforces a common capture deadline (default/minimum 120 seconds) and records `planned_capture_duration_seconds`.
- `lab/generate_mvp_corpus.sh` reads optional `duration_seconds`, passes it to each capture, and validates it on resume while remaining compatible with legacy metadata.
- `scripts/09_build_episode_manifests.py` verifies planned versus metadata/actual duration for new plans.
- Added `configs/mvp_v2_episode_plan.csv` and `configs/mvp_v2_split_assignments.csv` for `lab_025`-`lab_048`.
- V2 has 12 train / 6 validation / 6 test episodes; each scenario appears 2/1/1 per split; all six train role permutations occur twice; all six directed train LM pairs occur once.
- Shell syntax, Python compilation, invalid-duration rejection, old-manifest backward compatibility, plan invariants, and diff checks pass.

## Current invariant

Existing `lab_001`-`lab_024` raw episodes are immutable. New captures must use opaque IDs `lab_025`-`lab_048`, remain inside `10.77.0.0/24`, capture for 120 seconds, and be written atomically through the existing processing pipeline.

## Immediate interactive step

From repository root:

```bash
./lab/generate_mvp_corpus.sh configs/mvp_v2_episode_plan.csv
```

The user must enter sudo credentials for host tcpdump. Expected wall time is roughly 48 minutes plus processing. The runner is resumable and refuses mismatched/partial raw episodes.

## After capture completes

1. Validate all 24 episodes and audit actual capture/state-duration spread.
2. Build V2 manifests into `outputs/mvp_v2/` using `configs/mvp_v2_split_assignments.csv`.
3. Build V2 fixed-shape sequences.
4. Retrain persistence and PCA/Ridge baselines into separate V2 model/output paths.
5. Rerun shortcut diagnostics, especially forbidden state-index and host permutation tests.
6. Reject/repair V2 if late negative windows are absent or state index remains unrealistically predictive.
7. Only after V2 passes, overwrite this file with the compact RSSM implementation plan and begin RSSM code.

## RSSM decision

Use a compact RSSM as the next candidate architecture, not as a replacement for data repair. Keep Ridge as the benchmark. RSSM must use a GRU deterministic state, diagonal-Gaussian prior/posterior, future-state/edge decoder, and auxiliary future ATT&CK/LM heads. Accept it only if it beats persistence/Ridge on multi-step forecasting and does not worsen shortcut/permutation sensitivity.
