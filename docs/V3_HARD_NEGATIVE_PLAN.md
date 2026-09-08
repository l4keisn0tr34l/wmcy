# V3 Matched Hard Negatives and Fresh Holdout

**Status:** all 24 planned episodes captured, processed, and validated; fresh evaluation complete. See `docs/V3_RESULTS.md`.

## Motivation

Current graph models alert on scan-only/failed-guessing episodes because those precursors often precede successful movement in the small lab corpus. Public pretraining improves ranking but worsens negative-episode alerting to 3/4. More semantic-head tuning on the inspected V2 test would not resolve this data ambiguity.

## New scenario

`scan_guess_then_stop` executes:

```text
benign baseline → T1046 discovery → T1110.001 failed guessing → matched wait → no SSH movement
```

For the same seed, `one_hop` executes the same randomized roles, attempt count, and discovery/guessing delay sequence, then performs T1021.004 SSH movement. The stopped episode is malicious precursor activity with LM target zero; it is not mislabeled benign traffic.

## Paired corpus

`configs/mvp_v3_new_episode_plan.csv` defines 12 seed-matched pairs (24 new 120-second episodes):

```text
train:       6 stopped + 6 progressing; all six role permutations
validation:  3 stopped + 3 progressing
sealed test: 3 stopped + 3 progressing
```

A seed pair never crosses splits.

`configs/mvp_v3_episode_plan.csv` combines the 24 existing equal-duration V2 episodes with the 24 new episodes. Because old V2 validation/test have already been inspected, all old episodes become declared V3 development training data. Only new episodes populate V3 validation/test.

Validate before capture:

```bash
.venv/bin/python scripts/29_validate_v3_plan.py
```

Generate interactively (host tcpdump requires sudo):

```bash
docker compose -f lab/docker-compose.yml up -d
./lab/generate_mvp_corpus.sh configs/mvp_v3_new_episode_plan.csv
```

Then build V3:

```bash
.venv/bin/python scripts/09_build_episode_manifests.py \
  --plan configs/mvp_v3_episode_plan.csv \
  --splits configs/mvp_v3_split_assignments.csv \
  --out-dir outputs/mvp_v3
.venv/bin/python scripts/10_build_mvp_sequences.py \
  --episode-manifest outputs/mvp_v3/episode_manifest.csv \
  --out-dir outputs/mvp_v3/sequences
```

## Leakage prevention

- pairing is episode-level, never window-level;
- paired seed/scenario membership is audit metadata, not model input;
- stopped-prefix ATT&CK truth is retained separately;
- current V2 test is not reused as V3 validation/test;
- fresh test must remain unopened until model/threshold selection is frozen.

## Remaining limitation

Traffic is still from a three-container controlled lab. Matching seeded control flow reduces a major semantic confound but does not establish enterprise deployment generalization.
