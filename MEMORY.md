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
27. Standalone senior-facing report is generated by `scripts/17_build_rssm_report.py` at `outputs/mvp_v2/report/rssm_eod_report.html` and copied to `/home/paprika/Downloads/rssm_eod_report.html`; it embeds CSS/SVG/metrics/replay/limitations with no external assets. Current SHA-256 `7b886a2d0da7bbc856deda7a621f33d9ff34eda22197bcc79698ba1597fe25e5` includes branching, complete V4 action/passive sealed results, explicit cross-scaler comparability caution, and full Friday canonical/graph-sequence/fixed-pretraining results.
28. Senior independently reported transformer failure on scarce data, stronger RSSM forecasting, weak downstream classes, and improvement when RSSM was unfrozen. Their artifacts/protocol are unavailable, so this is an external hypothesis only. Defer DANN and ensembles until domain/complementarity evidence exists.
29. Matched V2 ablation completed: frozen two-stage LM F1/pre-F1/pair = 0.757/0.727/0.071; pretrained-unfrozen = 0.800/0.815/0.214. Zero-KL gives active MAE 0.984, edge AP 0.316, pre-F1 0.846, pair 0.286 but collapses mean spread to 0.025 from 0.076 and has worst permutation range 0.814. Keep zero-KL as an ablation. See `docs/RSSM_ABLATIONS.md`.
30. Test-isolated validation KL grid selected KL 0.01/free 0/seed 7. Twenty-rollout test: state 0.276, edge AP 0.285, LM F1/AP 0.800/0.926, identity-order pair 0.643 (9/14), spread 0.068. A 100-rollout checkpoint evaluation gives LM F1/AP 0.828/0.914 and unchanged episode behavior: 2/2 progressing about 26.4 s early, 2/4 negatives alert. Use as research initialization only; exact KL choice was close and fixed test is repeatedly inspected. See `docs/KL_TUNING.md`.
31. Six-permutation pair audit rejects the tuned identity pair score as robust: tuned top-1 ranges 2/14–8/14; non-identity choices map back to the identity choice only ~1.3%. Published model is also unstable. See `docs/PAIR_EQUIVARIANCE_AUDIT.md`.
32. Frozen shared pair scorer (same 288→64→1 MLP per pair) lowers test permutation-mean equivariance MAE 0.031→0.021 but worsens mean AP 0.267→0.232 and top-1 0.274→0.202. Reject the head; fixed-slot encoder/decoder is upstream bottleneck. See `docs/SHARED_PAIR_DECODER.md`.
33. First 358,115-param graph RSSM has shared normalization/encoding/recurrent node+edge states/decoders and exact host equivariance. V2 test: state MAE 0.252, active 0.895, edge AP 0.394, robust pair top-1 0.357, but initial LM F1/AP 0.476/0.738. See `docs/GRAPH_RSSM_RESULTS.md`.
34. Frozen pooled-head continuation worsens graph LM AP to 0.707. Invariant readout from decoded future global + node/edge mean/max raises LM F1/pre-F1 to 0.667/0.667 with AP 0.744, preserving dynamics/equivariance. Stop V2-only head tuning; add data/public dynamics. See `docs/GRAPH_SEMANTIC_RESULTS.md`.
35. UNSW temporal adapter processed all 2,540,047 host-rich rows into five sorted/gap-bounded segments and separate truth. Raw files have 52k–108k time inversions. Overlapping file intervals form only two connected capture groups and must not be split independently. No TCP flags, ATT&CK, or LM truth is fabricated. See `docs/UNSW_ADAPTER.md`.
36. Context-only roster extraction yields 1,481 January train and 1,383 February validation samples with the exact 141-feature lab graph layout. Rosters use the most frequent context edge plus a third context host and deterministic slot anonymization; no future or truth is consulted. See `docs/UNSW_GRAPH_SEQUENCES.md`.
37. UNSW observable-only pretraining (public semantic weights exactly zero) transfers: validation edge AP 0.430, LM AP 0.819, robust pair 9/13; diagnostic test state/active MAE 0.251/0.877, edge AP 0.405, LM F1/AP 0.778/0.785, robust pair 10/14. But false-alert episodes worsen to 3/4 and spread/error correlation to 0.199. Candidate, not unconditional replacement. See `docs/PUBLIC_PRETRAINING_RESULTS.md`.
38. V3 adds 12 seed-matched `scan_guess_then_stop`/`one_hop` pairs. Old V2 is development train; only new episodes form 6-episode validation and 6-episode test. Total: 48 episodes/720 samples. See `docs/V3_HARD_NEGATIVE_PLAN.md`.
39. Fresh V3 test: scratch/public-init graph models state MAE 0.294/0.321, edge AP 0.808/0.796, LM F1/AP 0.560/0.477 and 0.596/0.504, pair 6/18 both; 3/3 progress detected 23.5s early and 3/3 stopped episodes alert. Scratch wins validation joint 0.952 vs1.060 and is selected; public transfer rejected for V3. See `docs/V3_RESULTS.md`.
40. Matched prefixes expose observability: future controller SSH is absent from shared passive telemetry. Scratch test pair max risks differ only 0.0027 on average. Treat as branching risk, not deterministic intent inference. Score equivariance ~1e-8 but near-tied pair argmax varies 6/18–7/18.
41. CUDA is verified on RTX 3050 Mobile 4GB with torch2.9.1+cu128. CPU/CUDA deterministic delta 7.45e-8, equivariance 2.24e-8, finite gradients, ~202MB peak at batch128. Batch128 is1.78x per-step faster; batch32 is0.86x. Primary scripts support `--device auto|cpu|cuda`; checkpoint states save on CPU. See `docs/GPU_EXECUTION.md`.
42. Validation-only branch contract: 20 ordinary V3 scratch draws expected/oracle state MAE0.325/0.313, pairwise diversity0.034, weighted std0.029, LM draw std0.011, exact LM Brier/ECE0.205/0.229. Best draws span18.2 effective draws, suggesting diffuse noise not interpretable modes. Both stopped/progress validation episodes alert3/3. See `docs/BRANCHING_CONTRACT.md`.
43. Explicit outcome-conditioned 2-branch equivariant GraphRSSM (372,197 params), V3 train/val only: expected/oracle state MAE0.327/0.306, diversity0.406, edge AP0.763, LM AP/Brier/ECE0.655/0.114/0.046. LM-positive active MAE noLM/LM branches0.967/0.720 and edge AP0.800/0.879, but total MAE still favors generic noLM branch because quiet entries dominate. Alerts stopped/progress3/3. Eligible only for fresh V4 evaluation; does not infer intent. See `docs/BRANCHING_RESULTS.md`.
44. V4 complete:36 fresh ~120s episodes; action train12/test12 across all six role permutations, sealed passive stop/progress6, sealed matched legitimate/direct credential6. Totals828 states,2,559 observations,90 ATT&CK events,24 actions,399 zero states. General/action validators pass36/36; no residual firewall rules. Train action sequences are12 one-per-intervention samples; test remains locked. Paired train context normalized MAE mean0.164(range0.005–0.329), so captures are matched but not identical. See `docs/V4_ACTION_PLAN.md` and `docs/V4_CAPTURE_RESULTS.md`.
45. Full Friday PCAPNG canonicalization complete: 8,839,309,056 bytes;9,997,874 packets;9,915,680 IPv4;82,194 skipped;7,094 adjacent inversions max14us;2,102,560 one-second directed events. Native partition/merge adapter ran1m20.72s at47,480KB RSS. Output32MB gzip SHA e099281ef8903ed0697a3e6612935f503e4f68cec926274a0dac6813f9fc64a5. Observables only/no labels. See `docs/FRIDAY_PCAP_ADAPTER.md`.
46. Friday causal graph sequences: 5,775 train-only samples, context[3,141], future[6,141], edge[6,6], stride5s,13 candidates skipped,35,297 capture pairs,57.78% future induced windows active. Roster uses context frequent edge+third degree host and deterministic slot anonymization; novelty is capture-causal; real flags retained. No val/test split because one connected capture. See `docs/FRIDAY_GRAPH_SEQUENCES.md`.
47. Fixed Friday GraphRSSM pretraining:358,115 params, seed41001, CUDA batch128, exactly100 epochs, no selection, semantic weights0. Training-only dynamics objective1.174→0.350; future state0.905→0.220; edge1.076→0.516; causality0, equivariance4.47e-8. Runtime7m07.68s. Checkpoint SHA f8633c796f0523fc4537212ced3584d2f448ba55ac22be0cf94092a86e359ba0. Candidate initializer only; no transfer result. See `docs/FRIDAY_PRETRAINING.md`.
48. V4 action protocol frozen before test: ActionGraphRSSM372,947 params; train12 only; seed43001,batch12,Adam3e-4,400 epochs, no selection. Scratch/Friday train state MAE0.047/0.043, active0.157/0.108, LM/edge/pair perfect training fit, action LM deltas1.000/0.994; causality0 and equivariance5.96e-8/1.19e-6. Frozen SHAs scratch f69ece2d24ac07593d35666a83ea52ef8a768d1f51d4dcab13e7c1dc9f1c876f, Friday6b4cf070da873f04cd1fecb17cf351b9f9ba07bcbf45275b80aa69a8b100e351. See `docs/V4_ACTION_MODEL_PROTOCOL.md`.
49. One-shot V4 action test: initial build found lab090 had only5 complete future windows; before prediction excluded whole timing-invalid action_7002 family (lab089/090), leaving10 episodes/5 pairs. Scratch/Friday state MAE.0603/.0719, active.2327/.2475, quiet.0323/.0435; both edge AP1, LM AP/F1@.5=1, pair AP/top1=1, and factual action lower state error10/10. Scratch Brier8.91e-7 vs Friday1.60e-4. Train-derived thresholds overfit (F1.750/.889). ECE diagnostic only. See `docs/V4_ACTION_RESULTS.md`.
50. Frozen passive branch on V4 without tuning: 180 windows/12 episodes, expected/oracle state MAE.248/.237, diversity.501, edge AP.556, LM F1/AP/Brier.483/.472/.107. At frozen V3 threshold: alerts stopped3/3 but progressing scan-guess0/3 pre-positive; direct credential0/3, legitimate false alerts1/3. On same10 action contexts without action, state MAE.141 and LM F1/AP/Brier.400/.519/.329 vs scratch action model.060 and1/1. State MAEs use different frozen scalers and are not a direct ratio. Supports observability boundary, not calibrated/general causal claims. See `docs/V4_PASSIVE_BRANCH_RESULTS.md`.
51. Remote CSE-CIC-IDS2018 worker: authorized SSH `yashdeep@10.141.90.34`; dedicated HDD root `/media/yashdeep/New Volume 21/yashdeep_cyberwm` on `/dev/sda` has ~4TB free, 62GiB RAM, RTX PRO4000 24GB. No existing CIC/PCAP duplicate found. Official S3 enumerates42 objects/452.75GiB. All remote data/env/cache/temp/output must remain under HDD root. User guide `/home/paprika/Downloads/CSE_CIC_IDS2018_REMOTE_DOWNLOAD.txt`; see `docs/CSE_CIC_IDS2018_REMOTE.md`.
52. V5 initial draft (now reviewed; see next item): 80 paired episodes lab109-188, 150s, one flat five-host graph, 40/16/24 train/val/test, 40 unique role triples with complete host-role coverage, exact split-level balance across quiet/web/admin/mixed backgrounds. Cohorts scan action20, credential action20, passive prefix16, benign16, test-only intent probe8. Raw wall time3h20. Five-container HTTP/SSH/firewall smoke passes against original draft only. No capture yet. See `docs/V5_PLAN.md`.
53. V5 pre-capture review implemented: independent family-seeded timing, hash acquisition order, all-host sustained background, rotated validation/test cohort profiles, internal inventory-only capture, fresh containers plus fail-closed cleanup/provenance, action chosen before cutoff/applied after. Plan SHA b6676819c7c9cdf13e7a8b40b1861f209fb0a85310708028864d3de3b26a0946; splits unchanged. Exporter47: passive/action/passive_action, 345 features/20 pairs, no fitted scaler, test locked. Test48:12 synthetic tests PASS; all120 CUDA permutations max3.35e-8 in final repeat; exact CPU future perturbation0, CUDA2.98e-8; finite gradients and actual Friday/V4 parameter shape loading pass, old scalers never reused. No old-test inference. Gate49 blocks both corpus entry points and cannot freeze absent real smokes. First v5_smoke_001 attempt was user-confirmed manually shut down at ~112s and quarantined. Entry points self-enforce blocking shutdown:sleep:idle inhibitor. Two replacement smokes pass; capture-only freeze SHA53069f98ed2f8e686f973da6aa6f62f22f4d9b735def1da9868e7247783a20df records36 smoke hashes and80 randomized IDs. See `docs/V5_PRECAPTURE_REVIEW.md`.
54. V5 corpus complete and consistency-audited before model preprocessing:80/80 raw+derived and general/V5 validators PASS; captures150.000094–150.000615s, all29 dense complete states (145s after causal partial-boundary exclusion), zero drops, attack-vs-benign mean duration gap15us, paired max duration gap0.456ms, action future margin min42.294828s.40 paired families preserve seed/profile/split/state count. Residual absolute-grid phase: paired cutoff index differs by one in2/20 action families; max action paired cutoff-offset gap3.3595s scan/2.7230s credential—pairs similar, not packet-identical.13 background nonzero statuses:12 expected post-block consequences,1 bounded tail timeout. First lab158 attempt quarantined after SSH transport255; replacement valid. Audit script50/artifacts and `docs/V5_CORPUS_AUDIT.md`.
55. V5 train/validation exports built, test still absent: passive336/168 from16/8 episodes with21 windows each; action24/8; passive_action24/8. Script51 audit PASS:345-feature schemas identical, finite expected arrays only, one-hot action balance12/12 train and4/4 val, action/passive_action common arrays and manifests exact, whole families disjoint, forbidden truth absent from observable names. Passive train/val LM-positive windows24/12; action12/4. Immutable NPZ SHAs recorded in `outputs/mvp_v5/export_audit/audit.json`.
56. V5 model protocol v1 was frozen pre-fit but first action invocation stopped before optimizer creation: inherited V4 CUDA smoke demanded exact0 future delta and observed4.470348e-8, consistent with prior V5 CUDA2.980232e-8. No model/output/validation/test access. Log SHA5e3198d... and original freeze/scaler archived as invalid_prefit. V1.1 changes numerical audit only: CPU exact causality0 + all120 equivariance<1e-5; CUDA <=1e-6 diagnostic; training remains CUDA. Data/scaler/init/seeds/budgets/loss/selection/threshold unchanged. See `docs/V5_PROTOCOL_INCIDENT.md`. Corrected v1.1 then froze from commit0f8c1e2: protocol SHA96734fbd..., deterministic scaler unchanged SHA3928af5..., `test_unlock=false`; training not restarted yet.

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
