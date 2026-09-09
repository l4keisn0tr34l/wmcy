# Cyber World Model — Starter Build

This repository is deliberately **not an IDS classifier**. Its job is to turn raw network telemetry into chronological, graph-structured network states that can train a model of network dynamics:

\[
P(S_{t+1:t+H} \mid S_{t-L:t})
\]

The first supervised security tasks (MITRE technique/tactic prediction and lateral-movement forecasting) are auxiliary heads attached to the learned dynamics. They do not replace next-state prediction.

For the current verified MVP checkpoint, implementation status, file map, dataset inventory, hardware assessment, limitations, and next steps, read:

```text
docs/MVP_STATUS.md
```

**Current status:** equal-duration V2 is complete and removes the original late-negative censoring shortcut. An independent compact RSSM now performs six-step stochastic prior rollout: test state MAE is 0.280 versus 0.354 Ridge and 0.384 persistence; future-LM F1 is 0.800 versus 0.645 Ridge. CUDA training is verified on the local RTX 3050 (`--device auto|cpu|cuda`); use prespecified larger batches because batch 32 is launch-overhead bound. A permutation-equivariant graph RSSM reaches V2 state MAE 0.252 and edge AP 0.394. Observable-only UNSW pretraining helps some V2 metrics, but fresh matched-prefix V3 selects scratch initialization: state MAE 0.294, edge AP 0.808, 3/3 progressing episodes detected 23.5 seconds early—and 3/3 same-prefix stopped episodes alert. This reveals a passive-observability boundary for unobserved future attacker action. A validation-only outcome-conditioned two-branch graph RSSM now represents no-LM/LM alternatives explicitly (expected/oracle state MAE 0.327/0.306; exact-LM Brier 0.114), but still alerts on all stopped/progressing pairs; it is frozen for fresh V4 evaluation, not another V3 test. The full 8.839 GB Friday PCAPNG is also canonicalized into 2,102,560 label-free directed events and 5,775 train-only context/future graph sequences with disk-bounded, context-causal adapters. See `docs/BRANCHING_RESULTS.md`, `docs/FRIDAY_PCAP_ADAPTER.md`, and the linked result documents; model findings remain controlled-lab evidence, not enterprise generalization.

## Permanent data flow

1. **Raw telemetry** — CIC/UNSW/Zeek/PCAP-derived flows.
2. **Canonical events** — same column names and semantics regardless of source dataset.
3. **Dynamic graph states** — one graph per time window: hosts are nodes, communications are directed edges, network-wide values are global features.
4. **Sequence index** — context states `[t-L+1 ... t]` paired with future states `[t+1 ... t+H]`.
5. **Ground truth** — separate ATT&CK event timeline from scenario documentation or our own emulation. Ground-truth labels are NEVER included in observable model input.
6. **World model** — graph encoder + temporal dynamics model + state/edge decoders.
7. **Security interpretation** — auxiliary heads predict future ATT&CK techniques/tactics and lateral-movement edges from predicted latent futures.

## Why CICIDS2017 is handled specially

The uploaded `TrafficLabelling` CSVs contain source/destination IPs and 80+ CICFlowMeter features, but timestamps are rounded to one minute. We therefore:

- preserve CIC rows as useful flow observations;
- build **1-minute graph states** from the CSVs for self-supervised dynamics work;
- record `timestamp_resolution_sec=60` so later code never mistakes these for precise 5-second data;
- do NOT use CICIDS2017 CSVs as the main lateral-movement ground truth.

For our own emulated episodes (or precise PCAP/Zeek data), the exact same graph-state builder can use 5 s or 10 s windows.

## First build commands

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional NVIDIA/CUDA RSSM environment:
.venv/bin/python -m pip install -r requirements-rssm-cu128.txt
```

Put the CIC files in a directory, then profile them:

```bash
python scripts/01_profile_cic.py /path/to/cicids2017/*.csv --out outputs/cic_profile.json
```

Canonicalize one file:

```bash
python scripts/02_canonicalize_cic.py \
  /path/to/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv \
  --dataset-id cicids2017_thursday_infiltration \
  --out-dir outputs/thursday
```

Build 1-minute dynamic graph states:

```bash
python scripts/03_build_graph_states.py \
  outputs/thursday/observations.csv.gz \
  --window 60s \
  --out-dir outputs/thursday/states
```

Build a sequence index using 12 past states and 5 future states:

```bash
python scripts/04_build_sequence_index.py \
  outputs/thursday/states/global_states.csv \
  --context 12 --horizon 5 \
  --out outputs/thursday/sequence_index.csv
```

This produces world-model training structure without training an IDS.

## MITRE ground truth

MITRE mapping is not inferred from CIC feature values during dataset creation. We keep a separate event manifest such as:

```csv
episode_id,start_time,end_time,actor,target,technique_id,technique,tactic
lab_001,2026-09-05T18:20:00,2026-09-05T18:20:20,ws1,internal,T1046,Network Service Discovery,Discovery
lab_001,2026-09-05T18:20:40,2026-09-05T18:21:00,ws1,srv1,T1110.001,Password Guessing,Credential Access
lab_001,2026-09-05T18:21:10,2026-09-05T18:21:40,ws1,srv1,T1021.004,SSH,Lateral Movement
```

The scenario runner writes this because it knows exactly which action it executed. During training, state windows are aligned to this manifest by timestamp. At inference, the model predicts technique probabilities from predicted future latent states.

## Next model architecture (after the data layer is verified)

The intended first real model is:

```text
G_t --Graph Encoder--> z_t
z_{t-L+1:t} --Dynamics Model--> z_hat_{t+1:t+H}

z_hat future --> state decoder       (future network state)
             --> edge decoder        (future host-host links)
             --> MITRE technique head
             --> tactic head
             --> lateral-movement target-host head
```

The core objective remains future-state modeling. ATT&CK/lateral-movement losses add security meaning to the learned dynamics.
