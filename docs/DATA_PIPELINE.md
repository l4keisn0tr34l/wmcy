# Data Pipeline — What Every File and Script Does

This document explains the current preprocessing layer in both layman and technical terms.

## Big picture

A world model needs a sequence of network states:

\[
S_0,S_1,S_2,\dots,S_T
\]

Raw network datasets do not naturally arrive in that form.

The current pipeline converts:

```text
packets / flow rows
        ↓
common observable events
        ↓
time-windowed graph states
        ↓
past/future sequence definitions
```

ATT&CK truth is maintained separately.

---

# 1. Raw data

## Lab

```text
lab/episodes/<episode>/network.pcap
```

Layman meaning:

> A packet-level recording of what happened on the private network.

The PCAP contains evidence, not attack labels.

## CIC

Original `TrafficLabelling` CSVs live outside or alongside the project depending on local setup.

Earlier local directory:

```text
~/ds/cicids2017/TrafficLabelling 
```

Note: the directory name previously appeared with a trailing space. Inspect the actual filesystem before scripting around it.

---

# 2. `01_profile_cic.py`

## Input

One or more raw CIC CSVs.

## What it does

- reads schema;
- counts rows;
- counts labels;
- records timestamp examples;
- determines whether timestamps appear to contain seconds;
- records inferred timestamp resolution.

## Output

Example:

```text
outputs/cic_profile.json
```

## Why the world model needs it

Time resolution is part of the physics of the training data.

If CIC only contains minute timestamps, we cannot honestly create 5-second CIC states.

This step prevents fabricated temporal precision.

---

# 3. `02_canonicalize_cic.py`

## Input

Raw CICIDS2017 TrafficLabelling CSV.

## What it does

1. Normalizes column names.
2. Parses CIC's ambiguous working-hours timestamps.
3. Preserves all observable CIC columns.
4. Removes `Label` from the observation table.
5. Writes raw CIC label into a separate truth table.
6. Converts `inf/-inf` values to missing values.
7. Does not impute values yet.

## Output

```text
outputs/<name>/observations.csv.gz
outputs/<name>/row_ground_truth.csv.gz
```

## Why

The model must not see the answer.

`observations.csv.gz` is model-facing telemetry.
`row_ground_truth.csv.gz` is evaluation/supervision metadata.

## Important current behavior

The canonicalizer preserves many CICFlowMeter features, but the current graph-state builder does **not** consume all 80+ features.

That is intentional for now: the state builder uses a portable primitive subset shared with lab PCAP data.

---

# 4. `06_pcap_to_canonical.py`

## Input

```text
network.pcap
```

## What it does

Groups packets by:

```text
time bucket
source IP
source port
destination IP
destination port
protocol
```

For each directed group it records primitives such as:

```text
timestamp
source_ip
destination_ip
source_port
destination_port
protocol
packet count
byte count
SYN
ACK
RST
FIN
flow duration
```

## Output

```text
lab/episodes/<episode>/observations.csv.gz
```

## Why

PCAP is too low-level for the first state representation.

This script creates a common event language that downstream graph code can understand.

It does **not** attempt to reproduce every CICFlowMeter feature.

---

# 5. Canonical observation schema

The exact column set may be richer for CIC, but downstream graph construction currently relies on:

```text
event_id
dataset_id
timestamp

source_ip
destination_ip
source_port
destination_port
protocol

total_fwd_packets
total_backward_packets

total_length_of_fwd_packets
total_length_of_bwd_packets

syn_flag_count
ack_flag_count
rst_flag_count
fin_flag_count

flow_duration
```

CIC may contain many extra columns in the observation file.

Ground-truth fields must not be present as observation features.

---

# 6. `03_build_graph_states.py`

This is the most important current transformation.

## Input

Canonical observation events.

Example:

```text
lab/episodes/lab_001/observations.csv.gz
```

## Config

Lab:

```text
--window 5s
--internal-cidr 10.77.0.0/24
```

CIC CSV:

```text
--window 60s
--internal-cidr 192.168.10.0/24
```

## What it does

For each time window it constructs:

\[
S_t=(X_t^{global}, X_t^{node}, X_t^{edge})
\]

Hosts are graph nodes.
Directed source->destination communications are graph edges.

## Outputs

```text
states/global_states.csv
states/node_states.csv.gz
states/edge_states.csv.gz
```

---

## 6A. Global state

One row per state.

Current features:

```text
state_id
window_start

flow_count
unique_hosts
unique_src_hosts
unique_dst_hosts

unique_edges
new_edges
internal_edges

total_bytes
total_packets

syn_count
rst_count
fin_count

max_out_fanout
mean_out_fanout

dst_port_entropy
```

### Why

These describe the overall network regime.

Example:

```text
new_edges rises sharply
max_out_fanout rises
port entropy rises
```

may represent a transition into broad internal discovery.

They are not hard-coded attack rules; they are observations.

---

## 6B. Node state

One row per active host per state.

Outgoing features:

```text
outgoing_flows
outgoing_bytes
outgoing_packets
out_neighbors
unique_dst_ports
out_syn
out_rst
```

Incoming features:

```text
incoming_flows
incoming_bytes
incoming_packets
in_neighbors
unique_src_ports
in_syn
in_rst
```

Additional:

```text
is_internal
new_out_neighbors
new_in_neighbors
```

### Why

Lateral movement is host-specific.

We need to know which host's behavior changed, not only that "the network changed".

---

## 6C. Edge state

One row per directed source->destination pair per state.

Current features:

```text
flow_count
bytes_total
packets_total
syn_count
ack_count
rst_count
fin_count
mean_flow_duration
unique_dst_ports
internal_edge
is_new_edge
```

### Why

A future malicious movement may correspond to a new or changing host-host relationship.

This allows future work to learn:

\[
P(i\rightarrow j \text{ appears/changes in the future})
\]

---

## Edge novelty definition

The current script marks an ordered host pair as `is_new_edge=1` in the first state in which that pair appears.

Conceptually this can be computed causally from past observations only.

Any future rewrite must preserve that causal interpretation.

---

## Dense-window behavior

The script builds a dense fixed-time grid. For current generated episodes, explicit capture start/end metadata is supplied and only complete windows inside those bounds are retained. Canonical observations from partial opening/closing windows remain in the audit observation file but are deliberately excluded from graph aggregation and reported. Observations outside the raw capture bounds are errors. For legacy inputs without capture metadata, the builder falls back to the first through last observed window.

An empty complete five-second interval is still emitted in `global_states.csv` with zero-valued observable features.

The intended invariant is:

```text
state t+1 = exactly one configured window after state t
```

For empty windows, `node_states.csv.gz` and `edge_states.csv.gz` contain no rows for that `state_id`; the global row is the explicit representation of "no observed traffic in this interval."

---

# 7. `04_build_sequence_index.py`

## Input

```text
global_states.csv
```

## What it does

Defines which states are input history and which states are future targets.

Example:

```text
context = 12
horizon = 6
```

means:

\[
[S_{t-11},...,S_t]
\]

is the context and:

\[
[S_{t+1},...,S_{t+6}]
\]

is the future.

## Output

```text
sequence_index.csv
```

with fields like:

```text
sample_id
context_state_ids
future_state_ids
context_end_state
target_first_state
target_last_state
```

## Important

This script assumes state IDs are chronological adjacent states. The current graph-state builder now guarantees dense fixed-time global states. Sequence creation must still happen separately inside whole-episode train/validation/test splits; never randomly split overlapping sequences afterward.

---

# 8. `ground_truth.csv`

## Location

```text
lab/episodes/<episode>/ground_truth.csv
```

## Source

The scenario controller writes it because it knows the exact action it executed.

Current fields:

```text
episode_id
start_time
end_time
actor
target
technique_id
technique
tactic
```

## Example

```text
ws1 -> srv1
T1021.004
SSH
Lateral Movement
```

## Why it exists

This is the "answer key" for security semantics.

It is not model input.

---

# 9. `05_align_ground_truth.py`

## Inputs

```text
states/global_states.csv
ground_truth.csv
```

## What it does

Converts both timelines to UTC-aware timestamps and checks which ATT&CK events overlap each state window.

State windows are treated as half-open intervals:

```text
[window_start, window_end)
```

Instantaneous / zero-duration events are treated directly as points and align to the one half-open state containing their timestamp. No artificial duration or timestamp precision is added. Events ending before they start are rejected.

## Output

```text
state_ground_truth.csv
```

Current fields:

```text
state_id
window_start
technique_ids
tactics
actors
targets
has_lateral_movement
```

Multiple techniques/tactics are semicolon-separated.

## Why

This gives state-level semantic supervision.

For example:

```text
S_11 overlaps a real T1021.004 event
```

Later, if `S_11` is in a future horizon, the model can be trained/evaluated on whether it forecast that security behavior.

## Important

This script does not detect attacks from telemetry.
It only aligns known experiment truth to state time.

---

# 10. `07_validate_episode.py`

Checks one processed lab episode for fixed state spacing, valid IDs/references, chronological timezone-aware observations, finite global features, explicit quiet states, capture/event coverage, aligned ATT&CK/lateral truth, and forbidden ground-truth columns in model-facing data.

It reads data and reports pass/fail; it does not repair files.

---

# 11. `08_process_lab_episode.py`

Runs PCAP conversion, capture-bounded state construction, truth alignment, and validation in a temporary directory. Derived outputs are installed only after validation succeeds. Raw PCAP, truth, and episode metadata remain unchanged.

---

# 12. `09_build_episode_manifests.py`

Revalidates every planned episode, verifies metadata against the capture plan, and writes audited episode/split summaries. Scenario, role, and ATT&CK fields remain manifest metadata and are never model input.

# 13. `10_build_mvp_sequences.py`

Creates fixed-shape observable state vectors for the three known lab hosts and six directed internal host pairs. Silent known hosts and absent pairs receive zeros plus activity/presence masks. It then creates 15-second context and 30-second future arrays separately inside train, validation, and test episode splits. Future state, edge, ATT&CK, LM, and LM-pair targets are separate arrays.

# 14. `11_train_mvp_baseline.py`

Fits preprocessing only on observable training-context states, encodes states through PCA, predicts six future latent states with Ridge regression, reconstructs future observable states, and interprets predicted futures with security heads. Validation chooses PCA/Ridge settings and the LM threshold; test is used only after selection.

This is a direct multi-horizon baseline, not yet an autoregressive probabilistic neural rollout.

# 15. `12_replay_mvp.py`

Loads one fixed held-out sequence, performs prediction without reading future truth, then joins actual future truth only for evaluation/display. It writes self-contained JSON and HTML with context, predicted/actual future features, LM/ATT&CK scores, pair ranking, exact event lead time, and explicit limitations.

# 16. `13_evaluate_episode_alerts.py`

Aggregates horizon-window predictions into episode-level detection, exact warning lead time, and non-progressing false-alert summaries for validation/test episodes.

# 17. `14_audit_mvp_shortcuts.py`

Audits forbidden feature names, split/scenario role balance, LM pair coverage, direct global/last-state/identity/timing diagnostics, and sensitivity to six equivalent host relabelings. This stage identified unequal capture-duration censoring and fixed-slot sensitivity; see `docs/SHORTCUT_AUDIT.md`.

# 18. `15_train_rssm.py`

Trains a compact recurrent state-space model with deterministic GRU memory and a stochastic diagonal-Gaussian latent state. During inference it observes only three context states and performs six open-loop prior transitions. Observable-state and edge decoders model future telemetry; LM, ATT&CK, and pair heads interpret only imagined future latents. The script performs a causality test, tiny-overfit test, validation-only seed/epoch selection, Monte Carlo uncertainty diagnostics, active/quiet state evaluation, host-permutation sensitivity analysis, and episode alert reporting.

# 19. `16_replay_rssm.py`

Loads saved RSSM Monte Carlo predictions rather than rerunning or selecting the model, joins held-out truth only for display, and renders a self-contained V2 JSON/HTML replay. The output includes prediction uncertainty, ATT&CK scores, pair ranking, actual future states/events, exact lead time, aggregate metrics, and explicit selected-example limitations.

# 20. `17_build_rssm_report.py`

Builds one standalone senior-facing HTML report from verified V2 manifests, Ridge/RSSM metrics, shortcut audits, ablations, and replay JSON. CSS, pipeline/RSSM SVG diagrams, charts, tables, disclosures, and limitations are inline; no external assets or runtime network access are required.

# 21. `18_run_rssm_ablations.py`

Runs validation-selected frozen two-stage, pretrained/unfrozen, matched joint-from-scratch, and zero-KL comparisons using the same RSSM architecture and whole-episode V2 split. Frozen and unfrozen conditions use identical internal linear semantic heads. It keeps future telemetry and semantic targets outside context input and writes a structured metrics JSON plus local checkpoints.

# 22. `19_tune_rssm_kl.py`

Screens KL weights/free-nats on train/validation only, enforces predeclared validation gates for dynamics, semantics, spread, and host sensitivity, confirms a shortlist across three seeds, and loads test only after freezing the setting and seed. It records raw posterior/prior KL and avoids equating low spread with calibrated confidence.

# 23. `20_evaluate_rssm_checkpoint.py`

Loads a saved RSSM checkpoint and reproducibly recomputes validation-thresholded episode alerts and Monte Carlo LM predictions. It is used to increase the tuned candidate's rollout count without retraining or changing model selection.

# 24. `21_audit_pair_equivariance.py`

Applies all six consistent host relabelings to observable context, remaps evaluation targets, and measures whether state, edge, LM, ATT&CK, and pair predictions transform consistently. It performs no training or model selection and exposes fixed-slot pair shortcuts that identity-order top-1 hides.

# 25. `22_train_shared_pair_decoder.py`

Freezes the lower-KL RSSM and trains one shared pair scorer over each imagined source-node, destination-node, and directed-edge trajectory. It selects a seed by validation permutation-mean ranking before loading test, verifies all non-pair tensors remain bit-identical, and records the negative result that final-head sharing cannot repair an upstream fixed-slot representation.

# 26. `23_train_graph_rssm.py` + `src/cyberwm/graph_rssm.py`

Parse each state into global/node/directed-edge tensors, share normalization and neural operations across host/pair slots, aggregate incoming/outgoing messages, retain structured recurrent node/edge states, and roll a stochastic global prior forward. Shared decoders produce an exactly host-relabeling-equivariant future graph. Validation selects the seed before test loading; causality, tiny-overfit, shared-scaler, and exact permutation tests run before training.

# 27. `24_tune_graph_semantic_heads.py`

Freezes graph dynamics bit-identically and continues pooled LM/ATT&CK/shared-pair heads with validation semantic selection. The negative result separates representation/readout limitations from dynamics optimization.

# 28. `25_train_graph_rich_semantic.py`

Freezes graph dynamics and trains invariant LM/ATT&CK readouts from decoded future global features plus node/edge mean/max summaries. This preserves exact host invariance while exposing outlier behavior hidden by mean-only latent pooling.

---

# 29. `26_canonicalize_unsw.py`

Reads original host-rich UNSW files using their 49-field definition, separates observable flow primitives from raw attack truth, stable-sorts heavily out-of-order Unix timestamps, and creates gap-bounded segments. Overlapping source-file intervals are assigned to connected capture groups that must remain together across any split.

---

# 30. `27_build_unsw_graph_sequences.py`

Merges source segments by connected capture group and creates 45-second three-host induced-graph samples. Each roster is selected from the first 15 seconds only, then held fixed across the 30-second future. Slot order is anonymized. Output tensors use the exact 141-feature lab layout and contain observable state/edge targets only.

---

# 31. `28_pretrain_graph_rssm_unsw.py`

Fits public normalization on January context observations only, pretrains graph future/reconstruction/edge/KL dynamics with all semantic weights zero, and selects the public epoch on February dynamics. It then refits preprocessing on controlled lab train, fine-tunes telemetry plus semantic heads, selects on lab validation, and only then loads diagnostic V2 test.

---

# 32. `29_validate_v3_plan.py`

Checks that every new stopped/progressing seed pair remains in one split, old inspected V2 episodes are development train only, validation/test contain only new IDs, and new training pairs cover all six role permutations.

# 33. `30_train_graph_rssm_v3.py`

Fits preprocessing on V3 train only and compares matched scratch versus public-dynamics initialization. Semantic-head initial tensors are identical per seed; only non-semantic dynamics differ. Both seeds/epochs and LM thresholds are frozen on V3 validation before any model test prediction.

# 34. `31_audit_v3_matched_pairs.py`

Post-selection only: compares episode-level maximum risk and alerts within same-seed stopped/progressing pairs. It changes no model or threshold and exposes whether the passive prefix identifies eventual controller action.

# 35. `32_test_torch_devices.py`

Loads V3 train only and validates CPU/CUDA deterministic parity, graph equivariance, finite gradients, peak VRAM, and forward/backward speed. It cannot consume validation/test evidence.

# 36. `33_evaluate_branching_contract.py` + `src/cyberwm/branch_metrics.py`

Loads V3 validation only and treats existing stochastic rollouts as uniformly weighted candidate futures. It separates inference-weighted forecast error from oracle coverage, audits diversity/utilization, reports exact-outcome proper scores/calibration, and separately summarizes stopped/progressing precursor alert burden.

# 37. `34_train_branching_graph_rssm.py` + `src/cyberwm/branching_graph_rssm.py`

Loads V3 train/validation only. An invariant context latent predicts two outcome probabilities and initializes two equivariant future rollouts. Future LM truth selects the realized training branch but never enters inference. The branch-weighted future is the deployable mean; oracle best-branch error is coverage only. See `docs/BRANCHING_RESULTS.md`.

# 38. V4 intervention records and validators

`run_episode.sh` writes chosen defender actions to separate `defender_actions.csv`. Permit/block action scenarios align the decision after a five-second boundary, enabling a context that ends before the intervention. `35_validate_v4_plan.py` validates pair/split/role invariants before capture; `36_validate_v4_actions.py` validates action timing and completed-versus-attempted LM semantics afterward. No V4 data exists yet.

# 39. Where each datapoint resides

## Raw synthetic packet evidence

```text
lab/episodes/lab_001/network.pcap
```

## Synthetic attack/action truth

```text
lab/episodes/lab_001/ground_truth.csv
```

## Synthetic canonical events

```text
lab/episodes/lab_001/observations.csv.gz
```

## Synthetic graph-state datapoints

```text
lab/episodes/lab_001/states/global_states.csv
lab/episodes/lab_001/states/node_states.csv.gz
lab/episodes/lab_001/states/edge_states.csv.gz
```

## Synthetic state-aligned MITRE truth

```text
lab/episodes/lab_001/state_ground_truth.csv
```

## CIC canonical observation rows

Example:

```text
outputs/friday/observations.csv.gz
```

## CIC raw label truth

```text
outputs/friday/row_ground_truth.csv.gz
```

## CIC graph states

```text
outputs/friday/states/global_states.csv
outputs/friday/states/node_states.csv.gz
outputs/friday/states/edge_states.csv.gz
```

## CIC past/future sequence definitions

```text
outputs/friday/sequence_index.csv
```

## Known CIC -> MITRE event map

```text
configs/cic2017_known_mitre.csv
```

---

# 40. What will actually be fed to the model

Only past observable state information:

```text
global states
node states
edge states
graph topology
past history
```

Potentially later:

- known enterprise topology;
- additional host telemetry;
- normalized portable flow features.

Not fed:

```text
CIC Label
ground_truth.csv
state_ground_truth.csv
MITRE technique/tactic truth
has_lateral_movement
future state values
future edges
future actor/target
attack start/end times
```

See `LEAKAGE_AND_SPLITS.md`.

---

# 41. How this becomes a world model

After the state layer is correct:

```text
S_(t-L+1) ... S_t
         ↓
encoder + temporal dynamics
         ↓
z_hat_(t+1) ... z_hat_(t+H)
         ↓
future-state / edge decoders
```

Then security heads use the same future latent representation to forecast:

```text
future MITRE technique probabilities
future lateral-movement probability
future source-target movement edge
```

The world-state prediction remains primary.
