# MEMORY.md — Durable Project Memory

> Pi core automatically loads `AGENTS.md`, not this file. `AGENTS.md` explicitly instructs Pi to read this file at session start. An optional Pi memory extension can provide additional user-wide memory, but it is not required for this project.

## Identity of the project

This is a **predictive cyber-defense world-model** project.

The final system must model how an enterprise network/security state evolves and forecast attacker progression **before compromise/progression completes**.

Core formulation:

\[
S_{t-L+1:t} \rightarrow z_t \rightarrow \hat z_{t+1:t+H} \rightarrow \hat S_{t+1:t+H}
\]

Security heads interpret the predicted future:

- future ATT&CK techniques/tactics;
- lateral-movement probability;
- likely source-target movement edge;
- future compromise/risk.

The project must never collapse into a current-flow IDS classifier.

---

## Durable architectural decisions so far

### 1. Observable telemetry and ground truth are separate

Observable network data is model input.

Attack labels, MITRE truth, lateral-movement truth, scenario actor/target truth, and future values are targets/evaluation only.

### 2. The base world state is graph-structured

A state \(S_t\) has:

- global network features;
- per-host/node features;
- per-directed-edge communication features.

This preserves "who talks to whom", which is essential for lateral movement.

### 3. Time is explicit

Data is divided into chronological windows.

For precise lab PCAP data the intended initial resolution is 5 seconds.
Uploaded CICIDS2017 TrafficLabelling CSVs only preserve minute-level timestamps, so those CSVs must not be treated as 5-second data.

### 4. Generated/emulated data is primary progression truth

A controlled Docker lab produces actual packets plus an exact ATT&CK action manifest.

This is preferred for multi-stage/lateral-movement supervision because common IDS datasets do not cleanly contain exact attack trajectories.

### 5. Public datasets still matter

CIC/UNSW/CSE-CIC data can support:

- self-supervised network dynamics learning;
- background/benign diversity;
- attack-regime diversity where labels are reliable;
- domain-transfer/generalization evaluation.

### 6. MITRE is not the world model

MITRE provides semantic labels for observed/generated behaviors.

The core model predicts the future network/latent state.
MITRE is an auxiliary future-interpretation head.

### 7. Lateral movement should eventually be graph-aware

Desired targets include:

\[
P(\text{LM within horizon})
\]

and more specifically:

\[
P(i \rightarrow j \text{ is a future lateral-movement edge})
\]

### 8. ATT&CK output should be multi-label

Multiple techniques can occur in one future horizon.
Do not force one mutually exclusive stage unless an experiment explicitly requires that abstraction.

---

## Current repository data flow

```text
raw CIC CSV
    ↓ 02_canonicalize_cic.py
outputs/<dataset>/observations.csv.gz
    ↓ 03_build_graph_states.py
outputs/<dataset>/states/
    ├── global_states.csv
    ├── node_states.csv.gz
    └── edge_states.csv.gz
    ↓ 04_build_sequence_index.py
outputs/<dataset>/sequence_index.csv
```

Lab:

```text
lab/run_episode.sh
    ↓
lab/episodes/<episode>/
    ├── network.pcap
    ├── ground_truth.csv
    └── episode_metadata.csv
    ↓ 08_process_lab_episode.py
06_pcap_to_canonical.py
    ↓ observations.csv.gz
03_build_graph_states.py
    ↓ dense capture-bounded states/
05_align_ground_truth.py
    ↓ state_ground_truth.csv
07_validate_episode.py
    ↓ pass/fail integrity and leakage gate
```

---

## Current lab topology

Private Docker subnet:

```text
10.77.0.0/24
```

Hosts:

```text
ws1  = 10.77.0.20
srv1 = 10.77.0.30
srv2 = 10.77.0.40
```

The parameterized generator supports:

- benign ping;
- legitimate SSH;
- T1046 scan only;
- T1110.001 failed guessing only;
- one-hop discovery/guessing/SSH movement;
- two-hop discovery/guessing/SSH movement.

Actor, pivot, target, timing, baseline length, and attempt count vary from a recorded seed.

---

## Current known limitations / bugs

1. `lab_003` is the first current, capture-bounded episode to pass the complete atomic pipeline and validator. It has 14 complete five-second states, 4/4 aligned events, and both lateral hops.
2. `lab_001` passes after rebuilding but is legacy: whole-second truth and no capture metadata.
3. `lab_002` is quarantined because locale-dependent commas corrupted its timestamp CSV fields; raw files remain unchanged.
4. A balanced 20-episode MVP plan exists. Its first bulk attempt stopped safely on `lab_004`; that valid benign capture is excluded from MVP sequences because partial-boundary removal leaves only two complete states. Replacement `lab_024` is planned.
5. Future captures align controlled traffic to a five-second boundary and retain at least 31 seconds after the final action; the bulk runner can resume without overwriting raw files.
6. Only the two-hop and benign-ping branches have been live-capture tested so far.
7. Fixed topology, SSH service/credentials, and limited background traffic still permit shortcuts despite role randomization.
8. Silent hosts have no node rows; the future model loader must provide a causal known-host roster with zero activity and masks.
9. Host-pair edge aggregation does not distinguish a new service relationship from renewed activity on an existing pair.
10. The world-model architecture and training pipeline have not been implemented.

---

## Current CICIDS2017 facts from the uploaded copy

Three TrafficLabelling CSVs were inspected:

- Tuesday: 445,909 rows; FTP-Patator and SSH-Patator attack rows.
- Thursday afternoon Infiltration: 288,602 rows; only 36 rows labelled Infiltration.
- Friday afternoon PortScan: 286,467 rows; 158,930 PortScan rows.
- They contain 85 columns.
- Their `Timestamp` field is only minute-resolution in the uploaded export.

CIC raw labels are not sufficient to reconstruct a clean multi-stage progression timeline. In particular, the Thursday infiltration scenario must be decomposed from documented behavior; a broad `Infiltration` label is not one ATT&CK technique.

---

## Current high-confidence CIC -> ATT&CK mapping file

`configs/cic2017_known_mitre.csv`

Contains:

- FTP-Patator -> T1110.001 Password Guessing -> Credential Access.
- SSH-Patator -> T1110.001 Password Guessing -> Credential Access.
- Friday PortScan -> T1046 Network Service Discovery -> Discovery.

Mappings include confidence/evidence.

Do not force broad labels such as `Infiltration` to one technique.

---

## Evaluation principles

Never random-split overlapping windows from the same attack timeline.

Split by whole episode/scenario/day before sequence creation.

Important final metrics:

- next-state prediction error, by feature type;
- future-edge prediction PR-AUC / ranking;
- MITRE multi-label precision/recall/F1 or mAP;
- lateral-movement event recall;
- target-host top-k accuracy/ranking;
- false-positive rate on benign episodes;
- warning lead time;
- calibration/uncertainty;
- unseen playbook/scenario generalization;
- cross-dataset/domain-shift behavior.

---

## Important research memories

### Ha & Schmidhuber, World Models (2018)
Encoder + probabilistic latent dynamics + controller.
Key idea for this project: explicitly model a distribution over future latent states, not just current labels.

### StageFinder (2026)
GNN graph embeddings + LSTM current-stage estimation.
Most relevant detail: self-supervised pretraining predicts the **next graph embedding** and uses temporal contrastive loss before supervised stage fine-tuning.

### DeepStage (2026)
POMDP + provenance graph + stage belief + hierarchical PPO.
Useful for belief-state framing and CALDERA-style labeled episodes.
Not itself an explicit learned world-transition model.

### Guo & Xie (2025)
TCN/ResNet/BiGRU IDS paper.
Useful: causal/dilated TCN idea and imbalance treatment.
Critical warning: their tensor appears to treat feature columns as a sequence rather than actual chronological states.

### Cao et al. (2022)
CNN/BiGRU IDS.
Useful: multi-stat aggregation, redundancy-aware features.
Warning: grayscale reshaping gives arbitrary spatial semantics.

### KillChainGraph (2025)
Useful for ATT&CK semantic mapping/visualization.
Not sufficient evidence for actual temporal attack-transition dynamics; it links techniques largely via semantic similarity.

### Mane & Rao XAI
Useful explanation toolbox: SHAP, LIME, exemplars, contrastive explanations, rules.
Different explanation types serve different users.

---

## Working style

The user wants:

- practical implementation, not only research talk;
- explanation of every step and why it exists;
- no hallucinated dataset fields or outputs;
- continuity with the final world-model goal;
- weak assumptions challenged rather than accepted;
- direct, technical, intelligent communication.

After important changes, update this file only with durable facts/decisions, not temporary debugging chatter.
