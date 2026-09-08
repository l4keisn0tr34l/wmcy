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
4. The replacement 20-episode MVP corpus completed: 297 complete five-second states, 1,122 canonical observations, 34 ATT&CK events, and 12 lateral-movement events. All episodes pass validation and all 12 LM events have the intended directed edge in the same state.
5. Whole-episode splits are 12 train / 4 validation / 4 test. The 15-second-context/30-second-horizon exporter produces 73 / 33 / 31 sequences with only observable context features.
6. The first CPU latent baseline is trained: train-context-only scaling/PCA, direct six-step Ridge latent trajectory, state reconstruction, and future semantic/pair heads.
7. A shortcut audit found critical scenario-dependent capture-length censoring: on test, all complete samples at context state 7 or later are positive because negative captures end sooner. A forbidden state-index-only diagnostic gets AP 1.000. Current metrics are smoke tests, not final evidence; see `docs/SHORTCUT_AUDIT.md`.
8. Scaler/PCA originally fitted on train context plus train futures. This was corrected to train-context-only preprocessing. Corrected provisional test results are state MAE 0.676 versus 0.767 persistence, future-LM F1 0.963, pre-first-LM F1 0.957, edge AP 0.412, and pair top-1 0.429.
9. Raw actor metadata alone has AP 0.498, so no evidence says one common attack IP dominates. Fixed IP slots still matter: equivalent host relabeling changes LM probability by 0.234 on average. Training LM truth covers only 4/6 directed pairs and none sourced by srv2.
10. Simple last-state-only and global-only diagnostics achieve high semantic AP, so current LM results do not prove temporal history or graph identity is necessary. The primary future-state objective still beats persistence on this provisional split.
11. Episode-level evaluation detects both progressing episodes in validation and test before first LM (mean exact lead 27.9/27.0 s); failed guessing causes a validation false alert and current test negatives do not alert. Unequal duration confounds these results.
12. A held-out `lab_023` replay at context state 8 has no prior LM, exact event 22.8 s later, LM score 67.8%, T1021.004 96.2%, and correct `srv1->srv2` pair ranked first. It is selected after test inspection, and that pair occurred twice in training.
13. Fixed topology, SSH service/credentials, deterministic action order, capture order, and limited background traffic permit shortcuts despite role randomization.
14. Source state tables omit silent hosts; the MVP exporter inserts the known roster with zero activity and masks.
15. Host-pair edge aggregation does not distinguish a new service relationship from renewed activity on an existing pair.
16. Historical checkpoint: the first model was only PCA/Ridge. This is superseded by the compact recurrent stochastic RSSM described below; a learned message-passing graph encoder remains unimplemented.
17. Equal-duration V2 is complete: 24/24 episodes pass, each capture is 120.009-120.013 s, each has 23 states/15 samples, and totals are 552 states, 1,068 observations, 36 ATT&CK events, and 12 LM events. Splits are 12/6/6 episodes and 180/90/90 samples; all six directed LM pairs occur once in train.
18. V2 removes the dominant shortcut: forbidden state-index AP is 0.153 at target prevalence 0.156 (old AP 1.000), actor-only AP 0.132, slot masks AP 0.278 versus invariant masks 0.281. Host-relabeling sensitivity remains.
19. Honest V2 Ridge test results: state MAE 0.354 vs 0.384 persistence; active-state 1.089 vs 1.137; quiet-state 0.207 vs 0.233; LM F1 0.645; pre-first F1 0.640; edge AP 0.230; pair top-1 0.000. Global-only direct AP 0.770 exceeds latent LM AP 0.581.
20. A compact RSSM is now the next candidate: GRU deterministic state, diagonal-Gaussian stochastic prior/posterior, future state/edge decoder, and auxiliary semantic heads. Ridge remains the benchmark.
21. The senior's separate CICIDS2018 implementation is unavailable because they judged its results too poor to continue. Treat its verbal description as motivation only; do not depend on, integrate with, or claim empirical comparison against it.
22. Compact passive RSSM is implemented with 64-D GRU state, 16-D Gaussian stochastic state, posterior observation inference, and six-step prior rollout. Three seeds were selected by validation only; seed 7 epoch 240 won. CPU runtime was 76 s.
23. RSSM V2 test: state MAE 0.280 vs Ridge 0.354/persistence 0.384; active MAE 1.031 vs 1.089/1.137; future-LM F1/AP 0.800/0.910; pre-first F1 0.769; edge AP 0.222 (Ridge 0.230); pair top-1 0.143. MC state spread/error correlation is 0.613.
24. RSSM training is hybrid: telemetry prediction/reconstruction/KL is self-supervised, but auxiliary LM/ATT&CK/pair losses backpropagate into the shared latent model. Do not describe the whole regime as purely self-supervised.
25. RSSM episode test: 2/2 progressing episodes detected before LM with mean exact lead 26.4 s; 2/4 non-progressing episodes alert. Benign ping/legitimate SSH do not; scan-only/failed guessing do.
26. Selected V2 replay `lab_048` context state 6: no prior LM, score 87.8% ± 5.1%, threshold 66.8%, exact first LM 28.7 s later, correct top `srv1->ws1` pair at 76.5%. It was selected after aggregate test inspection; aggregate pair top-1 is only 0.143.
27. Standalone senior-facing report is generated by `scripts/17_build_rssm_report.py` at `outputs/mvp_v2/report/rssm_eod_report.html`; it embeds CSS, two SVG diagrams, metrics, replay, and limitations with no external assets.
28. Senior independently reported transformer failure on scarce data, stronger RSSM forecasting, weak downstream classes, and improvement when RSSM was unfrozen. Their artifacts/protocol are unavailable, so this is an external hypothesis only. Defer DANN and ensembles until domain/complementarity evidence exists.
29. Matched V2 ablation completed: frozen two-stage LM F1/pre-F1/pair = 0.757/0.727/0.071; pretrained-unfrozen = 0.800/0.815/0.214. Zero-KL gives active MAE 0.984, edge AP 0.316, pre-F1 0.846, pair 0.286 but collapses mean spread to 0.025 from 0.076 and has worst permutation range 0.814. Keep zero-KL as an ablation; tune KL/free-nats. See `docs/RSSM_ABLATIONS.md`.

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
