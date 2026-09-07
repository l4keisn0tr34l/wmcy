# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Completed RSSM checkpoint

- CPU PyTorch 2.9.1 installed via `requirements-rssm.txt`; external GPU was unnecessary.
- `scripts/15_train_rssm.py` implements a 79,421-parameter passive RSSM:
  - observation MLP 141 -> 64;
  - deterministic GRU state 64;
  - diagonal-Gaussian stochastic state 16;
  - posterior observation inference and six-step prior rollout;
  - state, edge, LM, three ATT&CK, and six LM-pair outputs.
- Causality test max delta 0.0; tiny-overfit loss decreased 1.872 -> 1.752.
- Seeds 7/17/27 trained with validation-only selection; seed 7 epoch 240 selected.
- Full CPU runtime approximately 76 seconds.
- Model/checkpoint loading and metric invariants pass.

## Honest V2 RSSM test result

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

- Validation LM F1 0.741; validation pre-first F1 0.720.
- ATT&CK test F1: T1046 0.683, T1110.001 0.772, T1021.004 0.839.
- Twenty-rollout state spread/error correlation 0.613; active spread 0.125 vs quiet 0.066.
- Host-permutation mean LM score range 0.111 (Ridge 0.120), but max outlier range 0.680.
- RSSM loses to persistence on active horizons +5/+10 s, then wins at +15 through +30 s.
- Edge AP is slightly worse than Ridge and exact pair ranking remains weak.
- Complete training is hybrid because semantic losses backpropagate into the latent model; do not call it wholly self-supervised.

Artifacts:

```text
models/mvp_v2_rssm.pt
outputs/mvp_v2/rssm/metrics.json
docs/RSSM_RESULTS.md
```

## Next code batch

1. Extend RSSM output with reproducible validation/test per-sample prediction NPZ/CSV.
2. Compute episode-level first-alert, exact lead-time, and non-progressing false-alert reports using the validation-selected threshold.
3. Build a self-contained V2 RSSM HTML/JSON chronological replay from a fixed held-out sample; disclose if selected after test inspection.
4. Add architecture/results summary suitable for EOD presentation.
5. Then improve edge/pair heads and run a pure self-supervised/two-stage versus joint-loss ablation.

## Acceptance and communication rules

- Cite RSSM state MAE 0.280 and LM F1 0.800 only with V2 controlled-lab/tiny-corpus caveats.
- Never cite old confounded F1 0.963.
- Report active-state and horizon metrics alongside aggregate MAE.
- Do not claim calibrated uncertainty, enterprise generalization, or action-conditioned counterfactuals.
- Senior repository is unavailable; this is an independent implementation.
