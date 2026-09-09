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

Build and evaluate V2 separately from original outputs:

```bash
.venv/bin/python scripts/09_build_episode_manifests.py \
  --plan configs/mvp_v2_episode_plan.csv \
  --splits configs/mvp_v2_split_assignments.csv \
  --out-dir outputs/mvp_v2
.venv/bin/python scripts/10_build_mvp_sequences.py \
  --episode-manifest outputs/mvp_v2/episode_manifest.csv \
  --out-dir outputs/mvp_v2/sequences --context 3 --horizon 6
.venv/bin/python scripts/11_train_mvp_baseline.py \
  --sequences-dir outputs/mvp_v2/sequences \
  --model-out models/mvp_v2_baseline.joblib \
  --metrics-out outputs/mvp_v2/model/baseline_metrics.json
.venv/bin/python scripts/13_evaluate_episode_alerts.py \
  --model models/mvp_v2_baseline.joblib \
  --sequences-dir outputs/mvp_v2/sequences \
  --out-dir outputs/mvp_v2/model
.venv/bin/python scripts/14_audit_mvp_shortcuts.py \
  --sequences-dir outputs/mvp_v2/sequences \
  --episode-manifest outputs/mvp_v2/episode_manifest.csv \
  --model models/mvp_v2_baseline.joblib \
  --out outputs/mvp_v2/model/shortcut_audit.json
```

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

## Train the compact V2 RSSM

Install the CPU dependency once:

```bash
.venv/bin/python -m pip install -r requirements-rssm.txt
```

Train three fixed seeds with validation-only selection:

```bash
.venv/bin/python scripts/15_train_rssm.py
```

Local outputs:

```text
models/mvp_v2_rssm.pt
outputs/mvp_v2/rssm/metrics.json
```

The current compact run takes approximately 76 seconds on CPU; an external GPU is unnecessary. It also writes reproducible Monte Carlo predictions and episode alerts.

Generate the selected held-out V2 replay:

```bash
.venv/bin/python scripts/16_replay_rssm.py
```

Open:

```text
outputs/mvp_v2/replays/lab_048_context_6_rssm.html
```

Build the standalone senior-facing HTML report with inline diagrams:

```bash
.venv/bin/python scripts/17_build_rssm_report.py
```

Run the matched frozen/unfrozen/zero-KL experiment:

```bash
.venv/bin/python scripts/18_run_rssm_ablations.py
```

Read:

```text
outputs/mvp_v2/rssm/ablations.json
docs/RSSM_ABLATIONS.md
```

Tune KL/free-nats with test-isolated screening and evaluate the selected checkpoint:

```bash
.venv/bin/python scripts/19_tune_rssm_kl.py
.venv/bin/python scripts/20_evaluate_rssm_checkpoint.py --mc-samples 100
```

Read:

```text
outputs/mvp_v2/rssm/kl_tuning.json
docs/KL_TUNING.md
```

Audit state/edge/pair behavior under all six host relabelings:

```bash
.venv/bin/python scripts/21_audit_pair_equivariance.py
```

Read `docs/PAIR_EQUIVARIANCE_AUDIT.md` before citing pair top-1.

Train/evaluate the frozen shared-weight pair-head diagnostic:

```bash
.venv/bin/python scripts/22_train_shared_pair_decoder.py
```

The head was rejected; see `docs/SHARED_PAIR_DECODER.md`.

Train the permutation-equivariant graph RSSM:

```bash
.venv/bin/python scripts/23_train_graph_rssm.py
```

Read `docs/GRAPH_RSSM_RESULTS.md` for the first dynamics/semantic trade-off.

Run frozen pooled-head and decoded-future invariant semantic follow-ups:

```bash
.venv/bin/python scripts/24_tune_graph_semantic_heads.py
.venv/bin/python scripts/25_train_graph_rich_semantic.py
```

See `docs/GRAPH_SEMANTIC_RESULTS.md`.

Build the updated report and send/open it:

```bash
.venv/bin/python scripts/17_build_rssm_report.py
```

```text
outputs/mvp_v2/report/rssm_eod_report.html
```

## UNSW-NB15 temporal canonicalization

```bash
.venv/bin/python scripts/26_canonicalize_unsw.py \
  /home/paprika/Documents/153/ds/un/CSV\ Files/UNSW-NB15_{1,2,3,4}.csv \
  --features /home/paprika/Documents/153/ds/un/CSV\ Files/NUSW-NB15_features.csv \
  --out-dir outputs/unsw/canonical
```

This writes observable events and raw attack truth separately, sorts out-of-order rows, and keeps overlapping source segments in two capture groups.

Build public three-host graph-dynamics sequences without loading truth:

```bash
.venv/bin/python scripts/27_build_unsw_graph_sequences.py \
  --out-dir outputs/unsw/graph_sequences
```

Pretrain graph dynamics on UNSW and fine-tune semantics on controlled V2:

```bash
.venv/bin/python scripts/28_pretrain_graph_rssm_unsw.py
```

Public semantic loss weights are zero; V2 test loads only after public and lab-validation selection.

## GPU execution

```bash
.venv/bin/python -m pip install -r requirements-rssm-cu128.txt
.venv/bin/python scripts/32_test_torch_devices.py --device cuda --batch-size 128
```

Primary training scripts default to `--device auto`. Use larger prespecified batches on this small model; batch 32 is slower on the RTX 3050. See `docs/GPU_EXECUTION.md`.

## Branch-aware validation contract

```bash
.venv/bin/python scripts/33_evaluate_branching_contract.py --device cuda
```

This loads V3 validation only and writes coverage/diversity/calibration diagnostics. Oracle best-branch MAE is coverage, not deployable point accuracy.

## Explicit outcome-conditioned branches (V3 development only)

```bash
.venv/bin/python scripts/34_train_branching_graph_rssm.py \
  --device cuda --batch-size 128 --epochs 250 --patience 40 \
  --seeds 7,17,27
```

This loads only V3 train/validation. `--resume-candidates` reuses compatible per-seed checkpoints after an interrupted post-training audit. Do not use that flag after changing model/loss settings.

## Friday PCAPNG canonicalization

```bash
.venv/bin/python scripts/37_canonicalize_pcap_scalable.py \
  /home/paprika/Documents/153/ds/Friday-WorkingHours.pcap \
  --dataset-id cicids2017_friday_working_hours_pcap \
  --bucket-seconds 1 --partitions 64 \
  --out outputs/cic2017_friday_pcap/canonical/observations.csv.gz \
  --manifest outputs/cic2017_friday_pcap/canonical/manifest.json

.venv/bin/python scripts/38_test_scalable_pcap_adapter.py
```

Use `--max-packets 100000` for a bounded smoke test. Existing outputs are protected unless `--force` is explicit. Labels are never read.

Build train-only causal Friday graph sequences:

```bash
.venv/bin/python scripts/39_build_friday_graph_sequences.py
.venv/bin/python scripts/40_test_friday_graph_sequences.py
```

Use `--force` only for an intentional rebuild. This deliberately emits no validation/test split from the one connected Friday capture.

## V4 plan validation and interactive capture

```bash
.venv/bin/python scripts/35_validate_v4_plan.py
bash -n lab/run_episode.sh lab/generate_mvp_corpus.sh
docker compose -f lab/docker-compose.yml config >/dev/null
```

When the laptop can remain awake for roughly 72 minutes plus processing:

```bash
cd lab
docker compose down
docker compose up -d --build
./generate_mvp_corpus.sh ../configs/mvp_v4_episode_plan.csv
cd ..
.venv/bin/python scripts/36_validate_v4_actions.py
```

The capture command is interactive because host `tcpdump` requires sudo. Do not run it unattended.

## V3 matched hard-negative plan

```bash
.venv/bin/python scripts/29_validate_v3_plan.py
docker compose -f lab/docker-compose.yml up -d
./lab/generate_mvp_corpus.sh configs/mvp_v3_new_episode_plan.csv
```

Capture requires interactive sudo. See `docs/V3_HARD_NEGATIVE_PLAN.md` before building V3 manifests.

Build and evaluate V3 (already completed; do not rerun to tune against test):

```bash
.venv/bin/python scripts/09_build_episode_manifests.py \
  --plan configs/mvp_v3_episode_plan.csv \
  --splits configs/mvp_v3_split_assignments.csv \
  --out-dir outputs/mvp_v3
.venv/bin/python scripts/10_build_mvp_sequences.py \
  --episode-manifest outputs/mvp_v3/episode_manifest.csv \
  --out-dir outputs/mvp_v3/sequences
.venv/bin/python scripts/30_train_graph_rssm_v3.py
.venv/bin/python scripts/31_audit_v3_matched_pairs.py
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
