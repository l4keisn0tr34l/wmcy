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

Validation-only KL tuning selected weight 0.01/free-nats 0/seed 7 without loading test during screening. Its 20-rollout test state MAE is 0.276, edge AP 0.285, LM AP 0.926, and identity-order pair top-1 0.643, while spread remains 0.068. A 100-rollout episode check still alerts on 2/4 negatives. See `docs/KL_TUNING.md`.

The pair result fails a six-permutation robustness audit: tuned pair top-1 ranges from 2/14 to 8/14 and non-identity top choices almost never map back to the identity choice. It is fixed-slot sensitive and is rejected as robust evidence. See `docs/PAIR_EQUIVARIANCE_AUDIT.md`.

A frozen shared-weight pair scorer reduces permutation-mean score equivariance MAE from 0.031 to 0.021, but worsens test permutation-mean pair AP from 0.267 to 0.232 and top-1 from 0.274 to 0.202. It is not adopted; the upstream flattened encoder/decoder must become graph-equivariant. See `docs/SHARED_PAIR_DECODER.md`.

The first permutation-equivariant graph RSSM is now trained. It improves test state MAE to 0.252, active-state MAE to 0.895, edge AP to 0.394, and pair top-1 to a relabeling-stable 0.357. Its initial LM F1/AP is 0.476/0.738. An invariant decoded-future readout improves LM F1 to 0.667 and pre-first-LM F1 to 0.667, but LM AP remains 0.744. See `docs/GRAPH_RSSM_RESULTS.md` and `docs/GRAPH_SEMANTIC_RESULTS.md`.

Not yet implemented:

- calibrated uncertainty;
- action-conditioned dynamics;
- fresh cross-domain or sealed-holdout evaluation.

## Public temporal integration

All 2,540,047 original host-rich UNSW-NB15 rows are now canonicalized into five sorted, gap-bounded segments. Observable events and raw attack labels are separate. The segments form only two connected capture groups because source-file times overlap; they cannot be randomly split.

Context-only roster extraction produced 1,481 January training and 1,383 February validation sequences in the same 141-feature, 3-context/6-future graph contract as the lab. The public tensors contain no attack/ATT&CK/LM targets.

Observable-only UNSW pretraining followed by controlled semantic fine-tuning improves validation edge AP 0.361→0.430, validation LM AP 0.728→0.819, and robust pair top-1 0.357→0.692. Diagnostic test state MAE is 0.251, active MAE 0.877, edge AP 0.405, LM F1/AP 0.778/0.785, and pair top-1 0.714 under every host relabeling. However, false-alert episodes worsen to 3/4 and stochastic spread/error correlation falls to 0.199. Treat this as a strong transfer candidate, not an unconditional replacement. See `docs/PUBLIC_PRETRAINING_RESULTS.md`.

## Immediate next milestone

V3 is complete: 48 validated episodes and 720 sequences. `scan_guess_then_stop` shares the seeded discovery/guessing control flow of paired `one_hop` but no successful SSH. Old V2 is development train; validation/test each contain three new stopped/progressing pairs.

On fresh test, scratch/public-initialized graph RSSMs reach state MAE 0.294/0.321, edge AP 0.808/0.796, LM F1 0.560/0.596, and both detect 3/3 progressing episodes 23.5 s early—but both alert on 3/3 stopped episodes. Episode maximum risks are nearly identical within scratch pairs (mean absolute gap 0.0027). Public initialization is not selected because scratch wins the validation joint objective and dynamics/edge/stochastic metrics. See `docs/V3_RESULTS.md`.

The paired result exposes an observability boundary: a future scenario-controller decision to perform SSH is not present in the shared passive-telemetry prefix. Next work must represent branching risk/uncertainty and eventually action/intervention variables rather than promise deterministic recovery of unobserved intent.

CUDA execution is now verified on the local RTX 3050. PyTorch 2.9.1+cu128, device-portable training, finite backward gradients, and graph equivariance all pass. Batch 128 is 1.78× faster per forward/backward step with only ~202 MB peak allocated VRAM; batch 32 is slightly slower than CPU, so future GPU runs should use larger prespecified batches. See `docs/GPU_EXECUTION.md`.

A validation-only branch-aware contract is implemented. Twenty ordinary RSSM draws improve expected-to-oracle state MAE only 0.325→0.313, with pairwise diversity 0.034 and mean LM draw spread 0.011. Best-draw assignments use 18.2 effective draws, indicating diffuse perturbations rather than two interpretable futures. Exact-LM Brier/ECE are 0.205/0.229. See `docs/BRANCHING_CONTRACT.md`.

An explicit 372,197-parameter two-branch equivariant GraphRSSM is now trained on V3 train/validation only. The outcome-conditioned model gives expected/oracle state MAE 0.327/0.306, diversity 0.406, edge AP 0.763, and exact-LM AP/Brier/diagnostic-ECE 0.655/0.114/0.046. On LM-positive futures, its LM branch improves active-feature MAE 0.967→0.720 and edge AP 0.800→0.879, at the cost of more quiet-entry error. It still alerts on all stopped and progressing validation episodes, so it represents alternatives but does not infer hidden intent. It is eligible for fresh V4 evaluation, not for V3 test reuse. See `docs/BRANCHING_RESULTS.md`.

V4 capture is complete: all 36 fresh equal-duration episodes pass general and action-semantic validation, with 828 complete states, 2,559 observations, 90 ATT&CK events, and 24 chosen actions. No firewall rules remain. Twelve action-train episodes are aligned into one intervention sample each; V4 test remains locked and no model prediction has occurred. Matched permit/block training contexts are similar but not identical (mean normalized MAE 0.164). Two 372,947-parameter action-conditioned checkpoints—scratch and fixed-Friday initialized—were frozen after an identical 400-epoch train-only protocol. On one sealed evaluation of 10 temporally eligible action-test episodes, scratch/Friday state MAE was 0.060/0.072, active MAE 0.233/0.248, edge and LM AP were 1.0, and the factual action reduced state error on 10/10 contexts. Scratch won state dynamics; Friday initialization did not transfer beneficially here. A timing-only complete-family exclusion reduced test from six to five pairs before any prediction. See `docs/V4_ACTION_RESULTS.md`.

The already-frozen passive outcome-branch model was then evaluated without tuning on 180 fresh V4 passive/direct windows: expected/oracle state MAE 0.248/0.237, edge AP 0.556, LM F1/AP 0.483/0.472. Operationally it alerted on 3/3 stopped scan/guess episodes but 0/3 progressing ones before a positive horizon; it also missed 3/3 direct-credential progressions and alerted on 1/3 matched legitimate episodes. On the same 10 action-aligned contexts without action input, LM F1/AP was 0.400/0.519 versus 1.0/1.0 with chosen action. See `docs/V4_PASSIVE_BRANCH_RESULTS.md`.

V5 now contains 80 paired 150-second episodes on a five-host isolated graph: 40 train, 16 validation, and 24 sealed test. It adds balanced quiet/web/admin/mixed background traffic, scan-based and direct-credential permit/block actions, passive stopped/progressing pairs, benign controls, and a test-only intent observability probe. Astra review corrected action/cutoff timing, background duration/source-role shortcuts, acquisition ordering, cohort-profile correlation, cleanup/isolation, and missing freeze enforcement. The 345-feature/20-pair exporter passes 12 synthetic tests and CUDA permutation/finite-gradient plus exact CPU temporal-independence checks. The revised container configuration has not yet undergone full smoke capture; earlier HTTP/SSH/firewall checks were against the draft. The first smoke was manually interrupted and quarantined; two replacements passed before capture freeze. The full V5 corpus is now complete: all 80 planned episodes are raw/derived-complete and independently pass general plus strict V5 validation. Raw durations span 150.000094–150.000615s; every episode has 29 complete five-second states and zero packet drops. Attack-intent versus benign/legitimate mean duration differs by only 15 microseconds, and maximum paired duration difference is 0.456ms, so the historical outcome-dependent censoring shortcut is absent. All actions retain at least 42.294828s after cutoff. Residual limitation: independent absolute grid phase moves paired cutoffs by up to one state (2/20 action families), so pairs are not packet-identical. The first rejected `lab_158` transport attempt is quarantined; its replacement validates. Train/validation-only exports are now built and audited: passive336/168 samples, action24/8, and aligned-passive24/8. Their345-feature schemas match, action/aligned-passive common arrays are exactly equal, whole-family splits remain disjoint, and no test sequence artifact exists. The train/validation model protocol is now implemented and pre-freeze tested but not yet executed: one shared-slot scaler will fit exactly440 deduplicated observable train-context states; scratch/Friday/V4 action and passive-branch candidates use fixed budgets/seeds and validation-only selection. Protocol regression tests pass with finite action/branch gradients, all120 permutation contracts, compatible historical parameter loads, and no old scaler reuse. The train/validation freeze is now installed: protocol SHA `cc5801a59bce74366455e2b269bc61924e8b68582156187f42415c24fcc7de3d`, scaler SHA `3928af5e911006b713374c826e59162e68c7762d9d95ace15fa2b437ddf44be6`, source commit `659f20a`. Training and test model evaluation have not started; `test_unlock=false`. See `docs/V5_MODEL_PROTOCOL.md`.

The full 8.839 GB Friday PCAPNG is now canonicalized by a native disk-bounded adapter: 9,997,874 packets, 9,915,680 IPv4 packets, 7,094 small timestamp inversions, and 2,102,560 chronological directed one-second events in 1m20.72s with 47 MB maximum RSS. Labels remain absent/separate. A causal context-only three-host builder produces 5,775 train-only `[3,141]→[6,141]` graph sequences with real TCP flags. No validation/test split is fabricated inside the one connected capture. A fixed seed/100-epoch observable-only GraphRSSM pretraining run reduces its training dynamics expression 1.174→0.350; all semantic weights are zero. This is only a candidate initializer for V4, not transfer evidence. See `docs/FRIDAY_PCAP_ADAPTER.md` and `docs/FRIDAY_GRAPH_SEQUENCES.md`.

The standalone senior report now includes explicit branching, complete V4 chosen-action/passive results, and Friday scale evidence. Current report SHA-256: `7b886a2d0da7bbc856deda7a621f33d9ff34eda22197bcc79698ba1597fe25e5`.

See `docs/MVP_STATUS.md` for full details.
