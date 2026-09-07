# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## EOD-ready RSSM checkpoint

Implementation:

- `requirements-rssm.txt`: CPU PyTorch 2.9.1;
- `scripts/15_train_rssm.py`: 79,421-parameter passive RSSM, three-seed validation selection, causality/overfit checks, MC predictions, shortcut and episode outputs;
- `scripts/16_replay_rssm.py`: saved-prediction V2 JSON/HTML replay;
- `docs/RSSM_RESULTS.md`: architecture, training regime, results, and limitations.

Primary V2 test comparison:

```text
                         RSSM    Ridge   persistence
state MAE               0.280    0.354      0.384
active-state MAE        1.031    1.089      1.137
quiet-state MAE         0.130    0.207      0.233
future-LM F1            0.800    0.645
pre-first-LM F1         0.769    0.640
future-edge AP          0.222    0.230
LM-pair top-1           0.143    0.000
```

Other facts:

- validation LM F1 0.741;
- T1046/T1110.001/T1021.004 test F1 0.683/0.772/0.839;
- state spread/error correlation 0.613 over 20 stochastic rollouts;
- host-permutation mean LM probability range 0.111, max 0.680;
- selected seed 7 epoch 240; CPU runtime 76 seconds;
- RSSM loses to persistence on active +5/+10 s horizons, wins +15 through +30 s;
- 2/2 progressing test episodes detected before LM, mean exact lead 26.4 s;
- 2/4 non-progressing test episodes alert: scan-only and failed guessing;
- benign ping and legitimate SSH do not alert.

Selected held-out replay:

```text
episode lab_048, context state 6
no LM previously observed
LM score 87.8% ± 5.1%; validation threshold 66.8%
exact first LM 28.7 seconds later
top pair srv1 -> ws1 at 76.5%; actual srv1 -> ws1
```

Replay is explicitly labeled as selected after aggregate test inspection. Aggregate pair top-1 remains 0.143.

Artifacts:

```text
models/mvp_v2_rssm.pt
outputs/mvp_v2/rssm/metrics.json
outputs/mvp_v2/rssm/predictions.npz
outputs/mvp_v2/rssm/sample_predictions.csv
outputs/mvp_v2/rssm/episode_alerts.csv
outputs/mvp_v2/replays/lab_048_context_6_rssm.html
```

## Next code batch

1. Visually inspect/freeze the HTML replay and prepare a concise senior-facing slide/script.
2. Run a pure self-supervised/two-stage RSSM versus current joint-loss ablation.
3. Improve edge decoder AP and exact LM-pair ranking.
4. Reduce scan-only/failed-guessing false alerts with harder matched negative episodes or a progression-specific head.
5. Consider shared-weight graph encoding after the ablation; do not add action conditioning without intervention data.

## Communication rules

- Call the complete RSSM regime hybrid, not wholly self-supervised.
- Cite V2 only; never cite old confounded F1 0.963.
- Always show persistence/Ridge, active-state, validation, false-alert, and pair limitations.
- Do not claim calibrated uncertainty, enterprise generalization, or action-conditioned counterfactuals.
- Senior repository is unavailable; this is an independent result.
