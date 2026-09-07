# Command Reference

Run from `/home/paprika/Documents/153/wm` unless stated otherwise.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lab containers

```bash
cd lab
docker compose up -d --build
docker compose ps
cd ..
```

## One controlled episode

The capture command requires interactive sudo authorization for host `tcpdump`:

```bash
cd lab
./run_episode.sh lab_003 --scenario two_hop --seed 2003 --duration-seconds 120
cd ..
```

Available scenarios:

```text
benign_ping
legitimate_ssh
scan_only
failed_guessing
one_hop
two_hop
```

Use opaque episode IDs so the model cannot learn the scenario from an ID.

## Process and validate one episode

```bash
.venv/bin/python scripts/08_process_lab_episode.py lab/episodes/lab_003
```

Rebuild derived files safely:

```bash
.venv/bin/python scripts/08_process_lab_episode.py lab/episodes/lab_003 --force
```

Independent validation:

```bash
.venv/bin/python scripts/07_validate_episode.py \
  lab/episodes/lab_003 --window-seconds 5
```

## Generate the planned MVP corpus

```bash
./lab/generate_mvp_corpus.sh
```

Original smoke-test plan:

```text
configs/mvp_episode_plan.csv
```

Equal-duration V2 replacement plan (next interactive command):

```bash
./lab/generate_mvp_corpus.sh configs/mvp_v2_episode_plan.csv
```

Split assignments are fixed in `configs/mvp_v2_split_assignments.csv`.

Request a clean pause after the currently active episode:

```bash
touch lab/.pause_corpus
```

Resume later:

```bash
rm -f lab/.pause_corpus
./lab/generate_mvp_corpus.sh configs/mvp_v2_episode_plan.csv
```

Do not use Ctrl+Z; it suspends scenario execution while wall-clock packet-capture timing can continue.

## Manual lab pipeline (debugging)

```bash
.venv/bin/python scripts/06_pcap_to_canonical.py \
  lab/episodes/lab_003/network.pcap \
  --bucket-seconds 1 \
  --dataset-id lab_003 \
  --out lab/episodes/lab_003/observations.csv.gz
```

For current episodes, prefer `08_process_lab_episode.py` because it reads capture bounds and validates temporary output before replacement.

## CIC profile

```bash
.venv/bin/python scripts/01_profile_cic.py \
  "/home/paprika/Documents/153/ds/cicids2017/TrafficLabelling /Tuesday-WorkingHours.pcap_ISCX.csv" \
  --out outputs/cic_profile.json
```

## CIC canonicalization and one-minute states

```bash
.venv/bin/python scripts/02_canonicalize_cic.py \
  "/path/to/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" \
  --dataset-id cicids2017_friday_portscan \
  --out-dir outputs/friday

.venv/bin/python scripts/03_build_graph_states.py \
  outputs/friday/observations.csv.gz \
  --window 60s \
  --internal-cidr 192.168.10.0/24 \
  --out-dir outputs/friday/states
```

The uploaded CICIDS2017 CSV timestamps have only minute precision; never build five-second states from them.

## Sequence index

```bash
.venv/bin/python scripts/04_build_sequence_index.py \
  outputs/friday/states/global_states.csv \
  --context 5 \
  --horizon 3 \
  --out outputs/friday/sequence_index.csv
```

Split whole episodes/days before sequence generation for training experiments.

## Build MVP manifests and sequences

```bash
.venv/bin/python scripts/09_build_episode_manifests.py
.venv/bin/python scripts/10_build_mvp_sequences.py \
  --context 3 --horizon 6
```

This means 15 seconds of observable context predicts 30 seconds of future. Splits come from `configs/mvp_split_assignments.csv` and are applied before sequence creation.

## Train the CPU latent-dynamics baseline

```bash
.venv/bin/python scripts/11_train_mvp_baseline.py
```

Local outputs:

```text
models/mvp_baseline.joblib
outputs/mvp/model/baseline_metrics.json
```

## Shortcut/leakage audit

```bash
.venv/bin/python scripts/14_audit_mvp_shortcuts.py
```

Read `docs/SHORTCUT_AUDIT.md` before citing controlled-corpus metrics.

## Episode alerts and held-out replay

```bash
.venv/bin/python scripts/13_evaluate_episode_alerts.py
.venv/bin/python scripts/12_replay_mvp.py \
  --episode lab_023 \
  --context-end-state 8
```

Open directly:

```text
outputs/mvp/replays/lab_023_context_8.html
```

Or serve locally:

```bash
python3 -m http.server 8000 --directory outputs/mvp/replays
```

Then open `http://localhost:8000/lab_023_context_8.html`.

## Git checkpoint

Generated data and `.venv` are ignored:

```bash
git status
git add -A
git commit -m "message"
git push origin main
```
