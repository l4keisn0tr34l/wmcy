# Controlled cyber world-model lab

This Docker lab generates real packets plus separate exact action/ATT&CK truth on the owned private subnet `10.77.0.0/24`.

## Topology

```text
ws1  10.77.0.20
srv1 10.77.0.30
srv2 10.77.0.40
```

All three containers run SSH and the tools needed by the controlled scenarios. Do not change targets to external networks.

## Start

From `lab/`:

```bash
docker compose up -d --build
docker compose ps
```

## Generate one episode

Use an opaque ID; keep the scenario name only in separate metadata:

```bash
./run_episode.sh lab_003 --scenario two_hop --seed 2003
```

Supported scenarios:

```text
benign_ping
legitimate_ssh
scan_only
failed_guessing
one_hop
two_hop
```

The runner asks for sudo because host `tcpdump` captures the private bridge. It refuses to overwrite existing raw episodes and writes:

```text
episodes/<id>/network.pcap
episodes/<id>/ground_truth.csv
episodes/<id>/episode_metadata.csv
```

`ground_truth.csv` and `episode_metadata.csv` are targets/audit information, never model input.

## Process and validate one episode

From repository root:

```bash
.venv/bin/python scripts/08_process_lab_episode.py lab/episodes/lab_003
```

Use `--force` only to rebuild derived files; raw files remain unchanged:

```bash
.venv/bin/python scripts/08_process_lab_episode.py lab/episodes/lab_003 --force
```

Derived output:

```text
observations.csv.gz
states/global_states.csv
states/node_states.csv.gz
states/edge_states.csv.gz
state_ground_truth.csv
```

Validate independently:

```bash
.venv/bin/python scripts/07_validate_episode.py lab/episodes/lab_003 --window-seconds 5
```

## Generate the MVP corpus

From repository root:

```bash
./lab/generate_mvp_corpus.sh
```

This reads `configs/mvp_episode_plan.csv`, keeps sudo authorization alive, runs each episode, immediately processes/validates it, and stops at the first failure.

## Data policy

`lab/episodes/` is Git-ignored. Raw captures and manifests stay local and immutable. Publish only explicitly selected, reviewed demo artifacts later.
