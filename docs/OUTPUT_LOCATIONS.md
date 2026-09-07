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
outputs/mvp_v2/replays/lab_048_context_6_rssm.json
outputs/mvp_v2/replays/lab_048_context_6_rssm.html
models/mvp_v2_baseline.joblib
models/mvp_v2_rssm.pt
```

## Status and methodology

```text
docs/MVP_STATUS.md
docs/CURRENT_STATE.md
docs/DATA_PIPELINE.md
docs/SHORTCUT_AUDIT.md
docs/RSSM_RESULTS.md
```

## External raw datasets

```text
/home/paprika/Documents/153/ds/cicids2017
/home/paprika/Documents/153/ds/cicids2018
/home/paprika/Documents/153/ds/un
/home/paprika/Documents/153/ds/Friday-WorkingHours.pcap
```
