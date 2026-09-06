# MVP Status — Predictive Cyber World Model

**Verified:** 2026-09-06  
**Repository:** `/home/paprika/Documents/153/wm`

## 1. Objective in plain language

This project is not intended to answer only:

```text
Is the current network connection benign or malicious?
```

It is intended to examine recent network history, form an internal summary of the evolving network/security situation, and predict what the network is likely to do next. The predicted future is then interpreted as possible ATT&CK behavior, compromise risk, or lateral movement.

The intended path is:

```text
recent observable network history
    -> internal network/security state
    -> predicted future states
    -> predicted future host-to-host communication
    -> ATT&CK / lateral-movement interpretation
```

## 2. Methodology

### Record packet evidence

Three Docker hosts communicate only on the isolated subnet `10.77.0.0/24`:

```text
ws1  = 10.77.0.20
srv1 = 10.77.0.30
srv2 = 10.77.0.40
```

Host `tcpdump` records the packets in `lab/episodes/<id>/network.pcap`. This is the network evidence.

### Keep the answer key separate

Because the controller knows which action it executes, it separately records action start/end time, actor, target, ATT&CK technique, and tactic in `ground_truth.csv`. Scenario and capture metadata reside in `episode_metadata.csv`.

Neither file is model input. They are targets and audit information only.

### Convert packets to directed conversations

`scripts/06_pcap_to_canonical.py` groups packets by one-second bucket and directed 5-tuple. It records observable packet, byte, TCP-flag, port, protocol, and duration information in `observations.csv.gz`.

Canonical rows are sorted by their actual timestamp and written with an explicit UTC offset. ATT&CK labels are excluded.

### Create five-second network snapshots

`scripts/03_build_graph_states.py` creates:

- a whole-network row for each state;
- one row for each active host;
- one row for each active directed host pair.

Quiet five-second periods are explicit zero-traffic global states. With capture metadata, only fully recorded windows inside the capture interval are retained. Therefore adjacent state IDs have a consistent physical meaning.

### Align known actions to the state timeline

`scripts/05_align_ground_truth.py` converts all times to UTC and records which known actions overlap each state. Normal events are time intervals; zero-duration events are points assigned to the single state containing that timestamp. Reversed event intervals are rejected.

### Validate before use

`scripts/07_validate_episode.py` checks fixed spacing, IDs, references, timestamps, finite global values, quiet-state representation, capture coverage, event coverage, lateral-movement flags, and absence of ground-truth columns in model-facing tables.

`scripts/08_process_lab_episode.py` builds derived files in a temporary directory, validates them, and installs them only after the full pipeline passes. Raw PCAP, truth, and metadata are never modified.

## 3. What is implemented

### Public-data preprocessing

- CICIDS2017 schema/timestamp profiling;
- separation of CIC observations from raw labels;
- canonical observation generation;
- one-minute graph-state construction for minute-resolution CIC CSVs;
- chronological context/future sequence indexing;
- evidence-backed mappings for selected CIC behaviors.

### Controlled-lab preprocessing

- isolated three-host Docker network;
- packet capture;
- exact action/ATT&CK truth recording;
- PCAP-to-canonical conversion;
- dense five-second graph states;
- explicit capture-bound handling;
- short/instant event alignment;
- leakage and integrity validation;
- atomic processing of derived episode files.

### Scenario generation

`lab/run_episode.sh` supports:

```text
benign_ping
legitimate_ssh
scan_only
failed_guessing
one_hop
two_hop
```

It randomizes actor/pivot/target roles, action pauses, baseline length, and password-attempt count from a recorded seed. It refuses to overwrite raw episode files and validates manifest width/timestamp format before declaring success.

### Planned MVP corpus

`configs/mvp_episode_plan.csv` contains 20 opaque episode IDs:

- 3 benign ping;
- 3 legitimate SSH;
- 3 scan-only;
- 3 failed-guessing;
- 4 one-hop progression;
- 4 two-hop progression.

All six actor/pivot/target permutations occur at least three times. `lab/generate_mvp_corpus.sh` generates, processes, and validates each episode, stopping on the first failure.

The first bulk attempt stopped safely on a partial-boundary policy error in `lab_004`. After boundary handling, traffic alignment, minimum duration, and safe resume were corrected, the replacement 20-episode plan completed. All 20 planned episodes pass validation.

## 4. Current episode status

### `lab_001` — legacy, rebuilt for debugging

Location: `lab/episodes/lab_001/`

Verified derived counts:

```text
68 canonical observations
20 global states
31 node-state rows
32 edge-state rows
4/4 events aligned
2/2 lateral-movement events aligned
7 explicit quiet states
```

Limitations: old whole-second truth timestamps and no explicit capture metadata. It is useful for regression/debugging but is not the preferred MVP episode.

### `lab_002` — quarantined

Location: `lab/episodes/lab_002/`

The host locale caused `date --iso-8601=ns` to use a comma as the fractional separator, corrupting CSV width. Original files were preserved and `INVALID_EPISODE.txt` explicitly excludes it from training/evaluation. The timestamp writer and schema checks were fixed afterward.

### `lab_004` — valid but excluded from MVP sequences

The first bulk attempt produced a valid benign capture, but controlled traffic began inside the partially captured opening window. Complete-window processing deliberately excluded five boundary observations, leaving only two complete states. The raw/derived episode passes integrity validation but is marked `EXCLUDE_FROM_MVP.txt` because it is too short for the planned context/future sequence. Its plan slot was replaced by `lab_024`.

Future captures align traffic to a five-second boundary and retain a longer future tail.

### `lab_003` — current valid smoke test

Location: `lab/episodes/lab_003/`

Verified facts:

```text
1,085 captured packets
82 canonical observations
14 complete five-second states
30 node-state rows
29 directed-edge rows
4/4 events aligned
2/2 lateral-movement events aligned
4 explicit quiet states
```

The randomized path was `srv1 -> srv2 -> ws1`. Capture metadata bounds are used. The entire atomic processing and validation pipeline passes.

PCAP timestamps have microsecond precision; controller truth has nanosecond-formatted timestamps. No extra packet precision is fabricated.

## 5. Current public/external data

Raw datasets are outside the repository under `/home/paprika/Documents/153/ds` and remain immutable.

### CICIDS2017 — approximately 2.0 GB

```text
/home/paprika/Documents/153/ds/cicids2017/TrafficLabelling 
/home/paprika/Documents/153/ds/cicids2017/MachineLearningCVE
```

The inspected TrafficLabelling copy has host identities but minute-resolution timestamps. It is useful for one-minute dynamics/background work, not precise five-second progression truth.

### CICIDS2018 — approximately 6.5 GB

```text
/home/paprika/Documents/153/ds/cicids2018/Processed Traffic Data for ML Algorithms
```

Ten large CICFlowMeter CSVs are present. A representative file has 80 columns and second-resolution-looking timestamps such as `14/02/2018 08:31:01`. Its first fields begin with destination port and protocol; source and destination IP fields were not observed in that representative header. Exact graph suitability must be checked before integration.

### UNSW-NB15 — approximately 688 MB

```text
/home/paprika/Documents/153/ds/un/CSV Files
```

The four original `UNSW-NB15_*.csv` files have 49 fields per row and preserve source/destination IPs, ports, protocol, service, duration, and traffic statistics. A separate feature-definition CSV exists. Reduced training/testing tables have 45 columns and should not automatically be preferred over the host-rich originals.

### Additional Friday PCAP — approximately 8.3 GB

```text
/home/paprika/Documents/153/ds/Friday-WorkingHours.pcap
```

This may provide precise public packet timing, but processing it now would consume time/storage and does not automatically provide clean multi-stage lateral-movement truth.

### Existing processed Friday output

Local generated files under `outputs/friday/` contain:

```text
286,467 canonical observations
286,467 separate raw-label rows
150 one-minute global states
20,553 node-state rows
30,613 directed-edge rows
143 chronological context/future samples
150 aligned state-truth rows
```

The 150 global states run from `2017-07-07 13:00 UTC` through `15:29 UTC` at exact one-minute increments.

## 6. Important repository locations

### Root project control

```text
AGENTS.md                    agent safety/methodology rules
MEMORY.md                    durable project memory
TODO.md                      ordered work queue
README.md                    repository overview
requirements.txt             Python dependencies
.gitignore                   excludes local/generated data and environments
PI_FIRST_SESSION.md          Pi startup guidance
CONTEXT_PACK_README.md       context-pack explanation
```

### Documentation

```text
docs/PROJECT_CONTEXT.md      objective and conceptual design
docs/CURRENT_STATE.md        verified implementation state
docs/MVP_STATUS.md           this complete MVP checkpoint
docs/DATA_PIPELINE.md        preprocessing semantics
docs/DATA_SOURCES.md         source roles and limitations
docs/MITRE_MAPPING.md        ATT&CK truth policy
docs/LEAKAGE_AND_SPLITS.md   causal input/split rules
docs/ROADMAP.md              long-term roadmap
docs/RESEARCH_CONTEXT.md     paper analysis
docs/DECISIONS.md            durable design decisions
docs/COMMANDS.md             reproducible commands
docs/OUTPUT_LOCATIONS.md     output paths
```

### Scripts

```text
scripts/01_profile_cic.py              inspect CIC schema/time resolution
scripts/02_canonicalize_cic.py         separate CIC observations and labels
scripts/03_build_graph_states.py       dense graph-state construction
scripts/04_build_sequence_index.py     chronological context/future indexes
scripts/05_align_ground_truth.py       state-level truth alignment
scripts/06_pcap_to_canonical.py        PCAP conversion
scripts/07_validate_episode.py         correctness/leakage validator
scripts/08_process_lab_episode.py      atomic lab pipeline
src/cyberwm/common.py                  shared utility functions
```

### Lab

```text
lab/Dockerfile                         host image
lab/docker-compose.yml                 isolated topology
lab/run_episode.sh                     one parameterized episode
lab/generate_mvp_corpus.sh             planned bulk generation
lab/episodes/<id>/                     local raw/derived episodes (Git-ignored)
```

### Configuration

```text
configs/cic2017_known_mitre.csv         supported public ATT&CK mappings
configs/mvp_episode_plan.csv            reproducible MVP capture plan
```

### Generated data

```text
outputs/                                local public-data outputs (Git-ignored)
lab/episodes/                           local lab data (Git-ignored)
```

### Research paper

```text
/home/paprika/Documents/153/rp/1809.01999v1.pdf
```

This is Ha and Schmidhuber's 15-page *Recurrent World Models Facilitate Policy Evolution*. It supports learned compressed state, predictive recurrent dynamics, probabilistic futures, and rollout, but its image/RL architecture will not be copied literally.

## 7. Current MVP dataset and baseline model

The completed 20-episode corpus contains:

```text
297 complete five-second states
1,122 canonical observations
34 ATT&CK events
12 lateral-movement events
```

All 12 lateral events have the intended actor-to-target directed edge in the same state.

Whole-episode split:

```text
train       12 episodes / 169 states / 6 LM events
validation   4 episodes /  65 states / 3 LM events
test         4 episodes /  63 states / 3 LM events
```

`scripts/10_build_mvp_sequences.py` creates 141-feature observable graph-state vectors with three persistent known-host slots, six directed internal host-pair slots, and activity/presence masks. It creates sequences independently inside each split:

```text
15-second context -> 30-second future
train:      73 sequences / 28 future-LM positives
validation: 33 sequences / 14 future-LM positives
test:       31 sequences / 14 future-LM positives
```

`scripts/11_train_mvp_baseline.py` trains a train-only StandardScaler + PCA state encoder, Ridge six-step latent trajectory predictor, future-state reconstruction, and future semantic/pair heads. Validation selected 32 PCA components and Ridge alpha 100.

Observed untouched-test results:

```text
normalized future-state MAE                 0.612
persistence baseline MAE                    0.728
future-LM precision / recall / F1            0.778 / 1.000 / 0.875
pre-first-LM precision / recall / F1         0.750 / 1.000 / 0.857
future-edge presence average precision       0.414 (positive prevalence 0.176)
future LM source-target top-1 accuracy       0.429
```

These results are correlated-window metrics from only four held-out controlled episodes. They support an MVP demonstration but do not establish enterprise, unseen-playbook, or cross-domain generalization. Exact LM pair prediction is currently weak.

Generated local artifacts:

```text
outputs/mvp/episode_manifest.csv
outputs/mvp/split_manifest.csv
outputs/mvp/sequences/
outputs/mvp/model/baseline_metrics.json
models/mvp_baseline.joblib
```

Still missing:

- an inference/replay interface;
- episode-level alert/lead-time summary beyond overlapping sample metrics;
- an autoregressive probabilistic rollout model;
- a learned graph message-passing encoder;
- robust exact source-target ranking;
- cross-dataset and unseen-playbook evaluation.

## 8. Current limitations and risks

1. Twenty planned current-format episodes are complete and validated; `lab_003` remains the separate smoke test.
2. All six generator branches have now run successfully in the planned corpus.
3. Twenty episodes are enough for an MVP smoke test, not strong generalization evidence.
4. Overlapping windows within held-out episodes are correlated, so sample-level metrics must not be described as 31 independent incidents.
5. Topology, SSH port, and credentials remain simple even though host roles vary.
6. Edge rows aggregate by host pair. Password guessing and successful SSH may use the same pair, so the first movement is not necessarily a new pair; the second hop often is.
7. Source state tables omit silent-host rows; the MVP sequence exporter now inserts the causally known three-host inventory with zero activity and an activity mask.
8. Public datasets are not yet integrated into one combined training contract.
9. The current baseline improves aggregate future-state MAE and LM forecasting on the tiny held-out controlled split, but exact LM pair ranking is weak.
10. Documentation and metrics must distinguish this controlled-lab proof of concept from enterprise generalization.

## 9. Hardware and dependency assessment

Verified local hardware:

```text
NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM
AMD Ryzen 7 7435HS, 16 logical CPUs
15 GB RAM
26 GB free project-disk space
```

Current virtual environment contains NumPy, pandas, and scikit-learn. PyTorch and TensorFlow are not installed.

The MVP does not require a remote GPU. A compact PCA/Ridge/MLP baseline can run on CPU. A small PyTorch recurrent/graph model should fit on the local RTX 3050 after installation. If infrastructure is easy to request, an NVIDIA GPU with 8–16 GB VRAM is useful but should not delay the MVP. Data quality and episode variety are currently the bottlenecks.

## 10. Distance to the world model

### Judge-facing MVP

Corpus capture, whole-episode splits, fixed-shape arrays, and the first latent multi-horizon baseline are complete. The remaining judge-facing path is:

1. add an inference/replay command for one held-out episode;
2. show observed graph history, predicted future state, future edges, ATT&CK/LM probability, and actual outcome;
3. add episode-level warning lead-time and false-alert summaries;
4. improve or clearly qualify exact source-target ranking;
5. freeze a reproducible demo command and screenshots.

A narrow controlled-lab MVP is now roughly one focused development day from a presentable replay, assuming no inference/UI defect appears.

### Full research system

Strong unseen-playbook generalization, calibrated multimodal uncertainty, multiple telemetry domains, and defense planning remain a longer-term effort measured in weeks or months.

## 11. Immediate next steps

1. Build a deterministic inference/replay script around `models/mvp_baseline.joblib`.
2. Compute episode-level warning lead time and false-alert behavior on validation/test episodes.
3. Produce a judge-facing held-out replay, initially for `lab_023` or `lab_022`.
4. Add a small neural recurrent model only if it can be completed without destabilizing the working baseline.
5. State the controlled-lab scope, correlated-window limitation, and weak exact-pair ranking honestly.
