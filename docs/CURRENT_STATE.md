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

All six actor/pivot/target role permutations are represented. `lab/generate_mvp_corpus.sh` has been syntax/plan checked but has not yet been run.

## Public data currently present

Local raw data under `/home/paprika/Documents/153/ds` includes:

- CICIDS2017: approximately 2.0 GB;
- CICIDS2018: approximately 6.5 GB;
- UNSW-NB15: approximately 688 MB;
- Friday CICIDS2017 PCAP: approximately 8.3 GB.

Existing processed Friday CIC output has 286,467 observations and 150 exact one-minute graph states. It is useful for broad dynamics but not clean five-second lateral-movement progression truth.

## World-model status

No trained world model exists yet.

Not yet implemented:

- episode-level split manifest;
- fixed-shape graph/sequence loader;
- latent state encoder;
- learned future-state transition;
- multi-step rollout;
- future-state and future-edge decoders;
- future ATT&CK/lateral-movement heads;
- evaluation report and judge replay.

## Immediate next milestone

1. Generate and validate the planned 20-episode corpus.
2. Inspect class, role, timing, state-count, and event-count distributions.
3. Split whole episodes before creating sequences.
4. Build model-facing arrays with persistent known-host rows and activity masks.
5. Train an interpretable CPU baseline before adding a small neural dynamics model.

Do not claim model performance until held-out episode results exist.
