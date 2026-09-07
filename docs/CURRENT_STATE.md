# Current State — 2026-09-06

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

All 12 lateral events have their intended actor-to-target directed traffic edge in the same state.

## Public data currently present

Local raw data under `/home/paprika/Documents/153/ds` includes:

- CICIDS2017: approximately 2.0 GB;
- CICIDS2018: approximately 6.5 GB;
- UNSW-NB15: approximately 688 MB;
- Friday CICIDS2017 PCAP: approximately 8.3 GB.

Existing processed Friday CIC output has 286,467 observations and 150 exact one-minute graph states. It is useful for broad dynamics but not clean five-second lateral-movement progression truth.

## World-model status

The first interpretable latent multi-horizon baseline is implemented and trained:

```text
observable graph-state vector
  -> train-only scaling
  -> 32-component PCA latent state
  -> Ridge six-step latent trajectory
  -> reconstructed future states
  -> future ATT&CK / LM / pair interpretation
```

Whole-episode split is 12 train / 4 validation / 4 test. Fifteen seconds of context predicts thirty seconds of future. Sequence counts are 73 / 33 / 31.

Observed controlled test results:

```text
future-state normalized MAE: 0.676 (persistence: 0.767)
future-LM F1: 0.963
pre-first-LM F1: 0.957
future-edge AP: 0.412 (prevalence: 0.176)
LM pair top-1: 0.429
```

These are provisional correlated-window results from four held-out controlled episodes. `docs/SHORTCUT_AUDIT.md` documents a critical unequal-capture-length confound: on test, all complete samples at context state 7 or later are positive. The metric change also reflects correcting scaler/PCA fitting from train-context-plus-future to train-context-only preprocessing.

Episode-level evaluation detects 2/2 progressing validation and 2/2 progressing test episodes before first LM, with mean exact leads of 27.9 and 27.0 seconds. Failed guessing produces a validation false alert; current test non-progressing episodes do not alert. Unequal episode length still makes this provisional.

A deterministic self-contained HTML/JSON replay exists for held-out `lab_023` at context state 8. The first actual LM event occurs 22.8 seconds after prediction availability; LM probability is 67.8%, T1021.004 is 96.2%, and the correct `srv1 -> srv2` pair ranks first. This selected pair occurred twice in training.

Not yet implemented:

- autoregressive probabilistic rollout;
- learned graph message-passing encoder;
- cross-domain evaluation.

## Immediate next milestone

1. Generate equal-duration replacement captures so negative scenarios provide late windows.
2. Balance directed LM pairs/roles and broaden action timing.
3. Retrain and require state-index, no-history, global-only, and host-permutation diagnostics.
4. Only then polish/freeze the replay and consider a neural recurrent baseline.

See `docs/MVP_STATUS.md` for full details.
