# Current State — 2026-09-07

This file records the current verified implementation. See `docs/MVP_STATUS.md` for the complete plain-language checkpoint, file map, dataset inventory, hardware assessment, and MVP timeline.

## Repository

```text
/home/paprika/Documents/153/wm
```

The project is a predictive cyber-defense world model, not a current-flow IDS classifier.

## Implemented data path

```text
controlled Docker action
    -> network.pcap + separate ground_truth.csv + episode_metadata.csv
    -> canonical directed observations
    -> dense five-second global/node/edge states
    -> state-aligned ATT&CK truth
    -> validation gate
```

Implemented scripts:

```text
scripts/01_profile_cic.py
scripts/02_canonicalize_cic.py
scripts/03_build_graph_states.py
scripts/04_build_sequence_index.py
scripts/05_align_ground_truth.py
scripts/06_pcap_to_canonical.py
scripts/07_validate_episode.py
scripts/08_process_lab_episode.py
```

## Latest verified corrections

1. Global states are dense: every adjacent lab state is exactly five seconds apart.
2. Empty traffic windows are explicit zero-valued global states.
3. Canonical PCAP observations are timestamp-sorted and have explicit UTC offsets.
4. Interval events use half-open overlap; true zero-duration events are aligned as points.
5. Reversed ground-truth intervals are rejected.
6. New episodes record explicit capture start/end metadata.
7. State construction uses only complete windows inside capture bounds.
8. The validator checks timing, references, finite values, leakage, capture/event coverage, and aligned truth.
9. The atomic processor validates temporary derived outputs before installing them.
10. Raw PCAP/truth/metadata files are not modified by processing.

## Episode status

### `lab_001`

Legacy episode rebuilt successfully for regression testing:

```text
68 observations
20 states
4/4 events aligned
2/2 lateral movement events aligned
```

It lacks current capture metadata and has old whole-second truth, so it is not the preferred MVP episode.

### `lab_002`

Invalid and quarantined. Locale-dependent commas in nanosecond timestamps corrupted CSV width. Original files remain unchanged and `INVALID_EPISODE.txt` excludes it from use.

### `lab_003`

Current valid smoke-test episode:

```text
1,085 packets
82 observations
14 complete five-second states
30 node-state rows
29 directed-edge rows
4/4 events aligned
2/2 lateral movement events aligned
4 explicit quiet states
```

The randomized path was `srv1 -> srv2 -> ws1`. Atomic processing and validation pass.

## Planned MVP corpus

`configs/mvp_episode_plan.csv` defines 20 opaque episode IDs across:

- benign ping;
- legitimate SSH;
- scan only;
- failed password guessing;
- one-hop movement;
- two-hop movement.

All six actor/pivot/target role permutations are represented. The replacement 20-episode plan completed and all episodes pass validation. `lab_004` remains valid but explicitly excluded because only two complete states remained after partial-boundary removal; replacement `lab_024` is part of the completed corpus.

Verified corpus totals:

```text
297 complete five-second states
1,122 canonical observations
34 ATT&CK events
12 lateral-movement events
```

All 12 lateral events have their intended actor-to-target directed traffic edge in the same state. This corpus is retained as an engineering smoke test but its semantic metrics are superseded because scenario lengths were unequal.

## Equal-duration V2 corpus

All 24 planned `lab_025`-`lab_048` episodes pass validation:

```text
552 complete five-second states (23 per episode)
1,068 canonical observations
36 ATT&CK events
12 lateral-movement events
360 sequences (15 per episode)
```

Every capture is 120.009-120.013 seconds. Splits are 12/6/6 episodes and 180/90/90 sequences. Every scenario appears in every split, and all six directed LM pairs occur exactly once in training. Late negative samples continue through context state 16.

## Public data currently present

Local raw data under `/home/paprika/Documents/153/ds` includes:

- CICIDS2017: approximately 2.0 GB;
- CICIDS2018: approximately 6.5 GB;
- UNSW-NB15: approximately 688 MB;
- Friday CICIDS2017 PCAP: approximately 8.3 GB.

Existing processed Friday CIC output has 286,467 observations and 150 exact one-minute graph states. It is useful for broad dynamics but not clean five-second lateral-movement progression truth.

## World-model status

The V2 interpretable baseline is:

```text
141-feature observable graph state
  -> train-context-only scaling
  -> 8-component PCA latent state
  -> Ridge six-step latent trajectory
  -> reconstructed future states
  -> future ATT&CK / LM / pair interpretation
```

Fifteen seconds of context predicts thirty seconds of future. Honest V2 held-out test results are:

```text
future-state normalized MAE: 0.354 (persistence: 0.384)
active-future-state MAE:     1.089 (persistence: 1.137)
quiet-future-state MAE:      0.207 (persistence: 0.233)
future-LM F1:                0.645
pre-first-LM F1:             0.640
future-edge AP:              0.230
LM pair top-1:               0.000
```

The forbidden state-index diagnostic fell from AP 1.000 to 0.153, near target prevalence 0.156. Actor-only AP is 0.132; slot-specific masks (0.278) are nearly identical to permutation-invariant counts (0.281). The main prior shortcut is removed, although host permutation sensitivity remains.

The model detects both progressing test episodes before first LM, with mean exact lead 21.4 seconds, but 3/4 non-progressing test episodes produce at least one false alert. A global-only direct diagnostic reaches AP 0.770 versus the latent model's AP 0.581, so the current semantic head does not yet establish graph-dynamics value.

## RSSM status

A compact passive RSSM is implemented and trained independently on V2. It uses a 64-dimensional GRU state, 16-dimensional diagonal-Gaussian stochastic state, posterior telemetry inference, and six-step open-loop prior rollout.

```text
RSSM test state MAE:       0.280 (Ridge 0.354; persistence 0.384)
RSSM active-state MAE:     1.031 (Ridge 1.089; persistence 1.137)
RSSM future-LM F1 / AP:    0.800 / 0.910
RSSM pre-first-LM F1:      0.769
RSSM edge AP:              0.222 (Ridge 0.230)
RSSM LM-pair top-1:        0.143
```

Twenty stochastic rollouts produce mean state spread 0.076; spread/error correlation is 0.613. Host-permutation mean score range is 0.111, but individual outliers remain. Episode evaluation detects 2/2 progressing test episodes before LM with mean exact lead 26.4 seconds; scan-only and failed-guessing produce alerts while benign ping and legitimate SSH do not. See `docs/RSSM_RESULTS.md`.

A matched representation-training ablation is also complete. Frozen telemetry-pretrained RSSM features reach LM F1 0.757; unfreezing raises LM F1 to 0.800, pre-first-LM F1 from 0.727 to 0.815, and pair top-1 from 0.071 to 0.214. A zero-KL model improves several point metrics but reduces mean rollout spread from 0.076 to 0.025, so it is not promoted as the selected model. See `docs/RSSM_ABLATIONS.md`.

Not yet implemented:

- learned graph message-passing encoder;
- calibrated uncertainty;
- action-conditioned dynamics;
- cross-domain evaluation.

## Immediate next milestone

The replay, standalone report, and frozen/unfrozen/KL ablations are complete. Next tune KL weight/free nats on validation while preserving useful rollout spread, improve edge/pair decoding, and add matched scan/guessing hard negatives. Action-conditioned defensive intervention remains later work.

See `docs/MVP_STATUS.md` for full details.
