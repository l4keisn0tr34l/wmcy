# Output Locations — Quick Reference

Generated data is local and Git-ignored.

## Controlled lab episode

For `lab/episodes/<episode_id>/`:

### Raw immutable evidence/truth

```text
network.pcap                 captured packet evidence
ground_truth.csv             exact scenario action/ATT&CK truth
episode_metadata.csv         scenario seed, roles, and capture bounds
INVALID_EPISODE.txt          present only when an episode is quarantined
```

### Derived model/evaluation data

```text
observations.csv.gz          canonical observable directed conversations
states/global_states.csv     dense fixed-time whole-network states
states/node_states.csv.gz    per-active-host state rows
states/edge_states.csv.gz    per-directed-host-pair state rows
state_ground_truth.csv       state-aligned ATT&CK/lateral targets
```

Current status:

```text
lab_001  legacy regression episode; rebuilt and validates without metadata bounds
lab_002  invalid/quarantined; never use for training/evaluation
lab_003  current valid capture-bounded smoke test
```

## Public CIC example

```text
outputs/cic_profile.json
outputs/friday/observations.csv.gz
outputs/friday/row_ground_truth.csv.gz
outputs/friday/states/global_states.csv
outputs/friday/states/node_states.csv.gz
outputs/friday/states/edge_states.csv.gz
outputs/friday/sequence_index.csv
outputs/friday/state_ground_truth.csv
```

## Configuration

```text
configs/cic2017_known_mitre.csv
configs/mvp_episode_plan.csv
configs/mvp_split_assignments.csv
configs/mvp_v2_episode_plan.csv
configs/mvp_v2_split_assignments.csv
```

## MVP manifests, sequences, and model

```text
outputs/mvp/episode_manifest.csv
outputs/mvp/split_manifest.csv
outputs/mvp/sequences/train.npz
outputs/mvp/sequences/validation.npz
outputs/mvp/sequences/test.npz
outputs/mvp/sequences/sample_manifest.csv
outputs/mvp/sequences/feature_metadata.json
outputs/mvp/model/baseline_metrics.json
outputs/mvp/model/episode_alerts.csv
outputs/mvp/model/episode_alert_summary.json
outputs/mvp/model/shortcut_audit.json
outputs/mvp/replays/lab_023_context_8.json
outputs/mvp/replays/lab_023_context_8.html
models/mvp_baseline.joblib
```

These are generated locally and Git-ignored. Reproducible scripts/configuration are tracked.

Equal-duration V2 counterparts are stored separately:

```text
outputs/mvp_v2/episode_manifest.csv
outputs/mvp_v2/split_manifest.csv
outputs/mvp_v2/sequences/
outputs/mvp_v2/model/baseline_metrics.json
outputs/mvp_v2/model/episode_alerts.csv
outputs/mvp_v2/model/episode_alert_summary.json
outputs/mvp_v2/model/shortcut_audit.json
outputs/mvp_v2/rssm/metrics.json
outputs/mvp_v2/rssm/predictions.npz
outputs/mvp_v2/rssm/sample_predictions.csv
outputs/mvp_v2/rssm/episode_alerts.csv
outputs/mvp_v2/rssm/ablations.json
outputs/mvp_v2/rssm/kl_tuning.json
outputs/mvp_v2/rssm/kl_tuned/
outputs/mvp_v2/rssm/pair_equivariance_audit.json
outputs/mvp_v2/rssm/shared_pair_decoder.json
outputs/mvp_v2/graph_rssm/metrics.json
outputs/mvp_v2/graph_rssm/semantic_tuning.json
outputs/mvp_v2/graph_rssm/rich_semantic.json
outputs/mvp_v2/replays/lab_048_context_6_rssm.json
outputs/mvp_v2/replays/lab_048_context_6_rssm.html
outputs/mvp_v2/report/rssm_eod_report.html
models/mvp_v2_baseline.joblib
models/mvp_v2_rssm.pt
models/mvp_v2_rssm_selfsupervised.pt
models/mvp_v2_rssm_frozen_heads.pt
models/mvp_v2_rssm_pretrained_unfrozen.pt
models/mvp_v2_rssm_joint_matched.pt
models/mvp_v2_rssm_zero_kl.pt
models/mvp_v2_rssm_kl_tuned.pt
models/mvp_v2_rssm_shared_pair.pt
models/mvp_v2_graph_rssm.pt
models/mvp_v2_graph_rssm_semantic.pt
models/mvp_v2_graph_rssm_rich_semantic.pt
```

## UNSW public temporal data

```text
outputs/unsw/canonical/manifest.json
outputs/unsw/canonical/<segment>/observations.csv.gz
outputs/unsw/canonical/<segment>/row_ground_truth.csv.gz
```

Five segments belong to only two connected capture groups; see `docs/UNSW_ADAPTER.md`.

```text
outputs/unsw/graph_sequences/train.npz
outputs/unsw/graph_sequences/validation.npz
outputs/unsw/graph_sequences/sample_manifest.csv
outputs/unsw/graph_sequences/feature_metadata.json
```

These contain 1,481 January and 1,383 February context/future graph samples; see `docs/UNSW_GRAPH_SEQUENCES.md`.

```text
models/unsw_graph_rssm_pretrained.pt
models/mvp_v2_graph_rssm_unsw_pretrained.pt
outputs/mvp_v2/graph_rssm/public_pretraining.json
outputs/mvp_v2/graph_rssm/public_pretraining_episode_alerts.csv
outputs/mvp_v2/graph_rssm/public_pretraining_sample_predictions.csv
```

See `docs/PUBLIC_PRETRAINING_RESULTS.md`.

## Status and methodology

```text
docs/MVP_STATUS.md
docs/CURRENT_STATE.md
docs/DATA_PIPELINE.md
docs/SHORTCUT_AUDIT.md
docs/RSSM_RESULTS.md
docs/RSSM_ABLATIONS.md
docs/KL_TUNING.md
docs/PAIR_EQUIVARIANCE_AUDIT.md
docs/SHARED_PAIR_DECODER.md
docs/GRAPH_RSSM_RESULTS.md
docs/GRAPH_SEMANTIC_RESULTS.md
docs/UNSW_ADAPTER.md
docs/UNSW_GRAPH_SEQUENCES.md
docs/PUBLIC_PRETRAINING_RESULTS.md
docs/V3_HARD_NEGATIVE_PLAN.md
docs/V3_RESULTS.md
docs/GPU_EXECUTION.md
docs/BRANCHING_CONTRACT.md
```

## V3 fresh matched-prefix evaluation

```text
outputs/mvp_v3/episode_manifest.csv
outputs/mvp_v3/sequences/
outputs/mvp_v3/graph_rssm/comparison.json
outputs/mvp_v3/graph_rssm/paired_prefix_audit.json
outputs/mvp_v3/graph_rssm/scratch_episode_alerts.csv
outputs/mvp_v3/graph_rssm/unsw_pretrained_episode_alerts.csv
models/mvp_v3_graph_rssm_scratch.pt
models/mvp_v3_graph_rssm_unsw_pretrained.pt
```

## Branch-aware validation

```text
outputs/mvp_v3/branching/contract_baseline_validation.json
```

See `docs/BRANCHING_CONTRACT.md`. V3 test is not loaded.

## Hardware validation

```text
outputs/device_test.json
```

Train-only CPU/CUDA parity, equivariance, gradient, speed, and VRAM audit.

## External raw datasets

```text
/home/paprika/Documents/153/ds/cicids2017
/home/paprika/Documents/153/ds/cicids2018
/home/paprika/Documents/153/ds/un
/home/paprika/Documents/153/ds/Friday-WorkingHours.pcap
```
