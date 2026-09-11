# Roadmap — From Current Data Layer to Final World Model

This roadmap is ordered to avoid building a sophisticated model on invalid data.

---

# Phase 0 — Make the current state representation correct

**Status (2026-09-06):** Core implementation is complete and verified on fresh `lab_003`: dense states, complete capture-bounded windows, point-event alignment, both lateral hops, and automated validation pass. Bulk scenario validation remains part of the next corpus milestone.

## Goal

Guarantee:

\[
S_{t+1}
\]

means exactly one configured time step after `S_t`.

## Work

1. Make `03_build_graph_states.py` generate dense windows.
2. Represent empty global windows explicitly.
3. Decide node/edge representation for empty windows.
4. Make `05_align_ground_truth.py` robust to short/instant events.
5. Normalize all internal timestamps to UTC-aware.
6. Generate a fresh current-format episode while retaining legacy raw episodes.
7. Verify both lateral hops.
8. Add a validator script.

## Exit criterion

For a 5-second episode:

```text
diff(window_start) == 5 seconds
```

for every adjacent state.

Every ground-truth event is either aligned to at least one state or explicitly reported as unaligned with reason.

---

# Phase 1 — Turn the one demo script into a dataset generator

**Status (2026-09-06):** MVP smoke corpus complete: 20 validated planned episodes across six scenario types with randomized roles/timing. This is sufficient for pipeline/model smoke testing, not final-scale training.

One fixed script is not training data.

Create multiple scenario classes.

## Benign

- idle/low traffic;
- web-like traffic;
- DNS-like traffic;
- file/service access;
- legitimate admin SSH;
- legitimate multi-host automation.

## Suspicious but non-progressing

- scan only;
- scan + no successful access;
- password guesses that all fail;
- one unusual new edge with no compromise.

## Progressing attacks

- discovery -> guessing -> SSH movement;
- guessing -> SSH without discovery;
- valid credential SSH without guessing;
- one-hop movement;
- two-hop movement;
- movement to different targets;
- repeated discovery after movement.

## Randomization

Randomize:

- actor host;
- target host;
- timing;
- pauses;
- number of attempts;
- order where semantically valid;
- background load;
- host-role mapping;
- IP/node mapping if practical.

## Exit criterion

Initial smoke-test corpus:

```text
20-50 varied episodes
```

Then scale substantially for real training.

---

# Phase 2 — Finalize the data contract

## Canonical event schema

Keep portable primitives shared across sources.

Consider adding:

- directional packet/byte rates;
- TCP state/flag ratios;
- inter-arrival statistics;
- per-service summaries;
- connection success/failure proxies;
- longer-horizon baseline features.

Only add a feature if:

1. it is observable;
2. it is causally computable;
3. its semantics are stable;
4. it can be reproduced across target data sources or clearly marked source-specific.

## Graph state contract

Formalize tensors/tables for:

```text
global features
node features
edge features
edge index
state timestamp
episode ID
```

Prefer Parquet / efficient binary tensors for scale, while retaining human-readable CSV samples for debugging.

---

# Phase 3 — Build dataset manifests and splits

**Status (2026-09-06):** MVP episode and split manifests implemented: 12 train / 4 validation / 4 test whole episodes. Training contains all six scenarios and all six host-role permutations.

Create:

```text
episode_manifest.csv
split_manifest.csv
```

Each episode should include metadata:

```text
episode_id
duration
topology variant
scenario family
techniques present
lateral movement present
number of hops
data source
```

Split at episode/scenario level before sequence generation.

At least one test should hold out a complete attack-path/playbook variant.

---

# Phase 4 — Self-supervised dynamics pretraining

Before ATT&CK supervision, teach the model general network evolution.

Inspired by StageFinder:

\[
z_t = E(S_t)
\]

Predict:

\[
\hat z_{t+1}
\]

with a next-step objective.

Potential addition:

temporal contrastive loss so the history representation is closer to the true next state than unrelated future negatives.

Public CIC/UNSW/CSE-CIC telemetry can be useful here even when ATT&CK ground truth is weak. The full precise Friday PCAPNG is now available as 2,102,560 observable one-second events through a 47 MB-RSS disk-bounded adapter. Context-only induced-subgraph construction produces 5,775 train-only graph sequences with real TCP flags. A fixed 100-epoch, zero-semantic-weight GraphRSSM run fits the training dynamics expression 1.174→0.350; fresh V4 transfer evaluation remains and this training fit is not generalization evidence.

---

# Phase 5 — First real latent world model

**Status (2026-09-07):** Equal-duration V2 removes the original state-index/capture-length shortcut. A compact stochastic RSSM is implemented and improves test state MAE to 0.280 (Ridge 0.354; persistence 0.384) and LM F1 to 0.800 (Ridge 0.645). Replay/reporting and matched frozen/unfrozen/zero-KL ablations are complete. Frozen telemetry-pretrained features carry useful semantic signal, while unfreezing improves pre-first-LM and pair results. Zero-KL improves several point metrics but collapses rollout spread. Validation-only tuning selected KL 0.01/free-nats 0, preserving spread while improving several test point estimates; the large identity-order pair result fails equivariance audit, and a shared final pair scorer localizes the issue upstream. A graph RSSM achieves score-level host equivariance and strong V2 dynamics. Observable-only UNSW pretraining improves some V2 metrics but is not selected on fresh V3. Matched-prefix V3 shows both models alert on every stopped and progressing pair: future controller SSH is not observable in the shared prefix. A validation-only outcome-conditioned two-branch graph RSSM now produces distinct future trajectories and improves candidate coverage, LM Brier/ECE, LM-positive active-feature MAE, and positive edge AP, while still alerting on every stopped/progressing validation pair. It is frozen for fresh V4 evaluation. The 36-episode V4 corpus is now captured and validated, covering role-balanced permit/block pairs plus sealed stopped/progressing and matched legitimate/direct-credential controls. Twelve action-train intervention-aligned samples are built. Scratch and fixed-Friday initialized action-conditioned models were frozen without selecting a winner and evaluated once on five temporally eligible test pairs: scratch won state dynamics while both perfectly ranked edge/LM/pair outcomes. The frozen passive branch model has now been evaluated on V4: its alternative-future representation transfers better than its gate/early-warning rule, which alerts stopped prefixes but misses fresh progressing/direct-credential episodes. V4 reporting/integrity consolidation is complete. The time-boxed V5 corpus is now captured: 80 validated episodes on a larger five-host graph with balanced quiet/web/admin/mixed backgrounds, direct-credential and scan-based actions, a real validation split, and a test-only intent probe. Equal-duration audit passes: all captures are ~150s, all have29 complete dense states and zero drops, and paired duration delta is at most0.456ms. It deliberately remains one flat topology and must not be described as unseen-topology evidence. Model/scaler/training protocol is not yet frozen; no V3/V4 retuning or V5 test model access is allowed.

## Provisional architecture

Not final; chosen for debugability.

### State encoder

Graph message passing over host nodes and directed communication edges.

Output:

```text
z_t
```

plus optional per-node latent states.

### Temporal dynamics

Start with a causal recurrent model or TCN over:

```text
z_(t-L+1:t)
```

Predict:

```text
z_hat_(t+1)
```

Then add autoregressive or direct multi-horizon rollout.

### Why not jump immediately to a Transformer?

We need a simple, inspectable baseline that validates whether the state representation contains learnable dynamics.
Architecture sophistication comes after data correctness.

---

# Phase 6 — Decode the future world

The model should not stop at latent prediction.

Add:

## Global state decoder

Predict future:

- traffic volume;
- new edges;
- fan-out;
- entropy;
- other global features.

## Future-edge decoder

For candidate host pairs:

\[
P((i,j)\in E_{t+h})
\]

This is central for lateral-movement forecasting.

## Per-node future decoder

Predict host behavior/state changes.

Use appropriate losses by variable type rather than one blind MSE over heterogeneous features.

---

# Phase 7 — Security semantic heads

Attach to predicted/forecast latent state.

## MITRE technique head

Multi-label future technique probabilities.

## Tactic head

Multi-label where useful.

## Lateral movement head

At least:

\[
P(LM \text{ within } H)
\]

Better:

\[
P(i\rightarrow j \text{ is a future LM edge})
\]

## Future compromise / hazard head

Possible later extension:

```text
probability host j is compromised within horizon
```

or time-to-event/hazard modeling.

---

# Phase 8 — Multi-horizon forecasts

Report:

```text
10 s
30 s
60 s
5 min
```

where supported by data.

The probability should reflect growing uncertainty with horizon.

Measure rollout error accumulation.

---

# Phase 9 — Evaluation

## World dynamics

- next-state error;
- per-feature-type metrics;
- rollout error vs horizon;
- edge prediction ranking/PR-AUC.

## Security forecasting

- future ATT&CK precision/recall/F1/mAP;
- LM recall;
- LM false-positive rate;
- target-host top-k;
- lead time;
- calibration.

## Generalization

- held-out episode variants;
- held-out playbooks;
- unseen target host/role mappings;
- cross-dataset/public-to-lab transfer.

## Ablations

- global-only vs graph;
- no history vs temporal history;
- no edge novelty;
- no self-supervised pretraining;
- different temporal models;
- different window sizes.

---

# Phase 10 — Explainability

Possible outputs:

- influential global features;
- influential host nodes;
- influential communication edges;
- historical state intervals that mattered;
- predicted future edge that caused the ATT&CK interpretation;
- uncertainty.

Use attention only as supplementary evidence.
Use attribution/ablation/counterfactual checks where possible.

---

# Phase 11 — Judge-facing demo

Desired UI/story:

```text
Observed:
- workstation A expands internal fan-out
- repeated service/credential pressure on server B
- edge novelty rising

Forecast:
- new A -> B SSH relationship within 30 s: 72%
- T1021.004 / Lateral Movement: 68%
- broader compromise risk within 60 s: 74%

Evidence:
- new internal peers
- destination-service pattern
- recent credential pressure
- predicted future edge topology
```

The demo should show both:

1. the predicted future world;
2. the security interpretation.

---

# Phase 12 — Optional defense actions

Only after the predictive model is robust.

Extend dynamics to action-conditioned form:

\[
P(S_{t+1}\mid S_t,a_t)
\]

Potentially compare futures under:

```text
do nothing
isolate host
block edge/service
increase monitoring
```

This would enable counterfactual defense planning.

Do not build an RL policy first and call it a world model.
