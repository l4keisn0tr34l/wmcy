# UNSW Graph-RSSM Dynamics Pretraining

## Scope

`scripts/28_pretrain_graph_rssm_unsw.py` pretrained the permutation-equivariant graph RSSM on public UNSW observable dynamics, then fine-tuned it on controlled V2 telemetry and semantic truth.

UNSW attack category/label files were never loaded. Public loss weights were:

```text
future state 1.00
reconstruction 0.25
edge presence 0.25
KL 0.01
LM / ATT&CK / LM pair 0.00
```

The public scaler was fit only on January contexts. Public epoch 60 was selected on the disconnected February capture; its dynamics objective improved from 5.109 at epoch 1 to 1.817. Before transfer, a new scaler was fit only on controlled-lab training contexts.

Fine-tuning seeds 7/17/27 were compared on lab validation only. Seed 17, epoch 54 won with joint objective 0.802. V2 test was loaded afterward, but is diagnostic because it has already been inspected in prior experiments.

## Results

| Metric | Graph RSSM, lab only | UNSW-pretrained graph RSSM |
|---|---:|---:|
| validation normalized state MAE | **0.259** | 0.260 |
| validation edge AP | 0.361 | **0.430** |
| validation LM AP | 0.728 | **0.819** |
| validation LM F1 | 0.667 | **0.813** |
| validation pair top-1 | 0.357 | **0.692 (9/13)** |
| test normalized state MAE | 0.252 | **0.251** |
| test active-state MAE | 0.895 | **0.877** |
| test edge AP | 0.394 | **0.405** |
| test LM AP | 0.738 | **0.785** |
| test LM F1 | 0.476 | **0.778** |
| test pre-first-LM F1 | 0.476 | **0.774** |
| test pair top-1 | 0.357 | **0.714 (10/14)** |

The pair result is exactly host-equivariant: top-1 is 9/13 on validation and 10/14 on diagnostic test under every relabeling, with pair-probability equivariance MAE around `5e-9`. This passes the robustness test that invalidated the flattened model's identity-order pair headline.

The model detects both progressing test episodes before first LM with mean exact lead 28.9 seconds.

## Important trade-offs

Public pretraining is useful but not an unconditional replacement:

- non-progressing test episodes with any alert worsen from 2/4 to 3/4;
- validation false-alert episodes worsen from 1/4 to 3/4;
- mean Monte Carlo state spread falls from 0.036 to 0.023;
- spread/error correlation falls from 0.722 to 0.199.

Monte Carlo spread is not calibrated uncertainty, but this degradation is still a warning that transfer made stochastic variation less informative. The candidate improves edge, LM, active-state, and robust pair point metrics while worsening false-alert behavior and the stochastic diagnostic.

## Interpretation

**Observed:** public telemetry-only pretraining transfers useful graph/edge structure and improves robust LM pair ranking after controlled semantic fine-tuning.

**Not established:** deployment generalization, clean LM semantics in UNSW, calibrated uncertainty, or lower operational false-alert rate.

The appropriate next step is matched hard-negative lab data plus a fresh sealed holdout—not additional tuning against the current V2 test.

## Artifacts

```text
models/unsw_graph_rssm_pretrained.pt
models/mvp_v2_graph_rssm_unsw_pretrained.pt
outputs/mvp_v2/graph_rssm/public_pretraining.json
outputs/mvp_v2/graph_rssm/public_pretraining_episode_alerts.csv
outputs/mvp_v2/graph_rssm/public_pretraining_sample_predictions.csv
```
