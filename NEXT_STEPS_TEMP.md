# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Verified current state

V2 corpus is complete and accepted for the RSSM experiment:

- 24/24 `lab_025`-`lab_048` episodes pass validation;
- capture duration 120.008644-120.012759 seconds;
- exactly 23 states and 15 sequence samples per episode;
- totals: 552 states, 1,068 observations, 36 ATT&CK events, 12 LM events;
- splits: 12/6/6 episodes and 180/90/90 samples;
- all six directed LM pairs occur once in training;
- late negatives continue through context state 16;
- state-index AP 0.153 at target prevalence 0.156 (old confounded AP 1.000);
- actor-only AP 0.132; slot-mask AP 0.278 vs invariant-mask AP 0.281.

Official local artifacts are under `outputs/mvp_v2/` and `models/mvp_v2_baseline.joblib`.

Honest V2 Ridge test results:

- state MAE 0.354 vs 0.384 persistence;
- active-state MAE 1.089 vs 1.137;
- quiet-state MAE 0.207 vs 0.233;
- future-LM F1 0.645; pre-first F1 0.640;
- edge AP 0.230; pair top-1 0.000;
- global-only direct AP 0.770 vs latent LM AP 0.581;
- 2/2 progressing test episodes detected, but 3/4 non-progressing test episodes alert.

`scripts/11_train_mvp_baseline.py` now records active/quiet and per-horizon state metrics.

## Next code batch: compact passive RSSM

1. Confirm disk/PyTorch installation choice; local RTX 3050 4 GB is sufficient and external GPU is not required.
2. Add PyTorch without removing the current scikit environment.
3. Implement an RSSM over 9-state V2 sequences:
   - observation width 141;
   - MLP embedding 64;
   - GRU deterministic state 64;
   - diagonal-Gaussian stochastic state 16;
   - posterior conditioning for 3 context states;
   - six-step stochastic prior rollout;
   - decoder back to normalized 141-feature observable states;
   - explicit edge-presence logits;
   - future LM, three ATT&CK, and six LM-pair auxiliary heads from predicted future latents only.
4. Training losses:
   - equal-weight global/node/edge normalized prediction MSE;
   - edge-presence BCE;
   - posterior-prior KL with free nats and conservative beta;
   - auxiliary semantic BCE with class weighting;
   - no scenario, actor, target, state index, timestamp, or labels as input.
5. Use train-only normalization, validation early stopping, fixed seeds, gradient clipping, and consistent host permutation augmentation.
6. Run tiny-overfit and tensor/causality tests before full training.
7. Compare against V2 persistence/Ridge using aggregate, active/quiet, and per-horizon state MAE; edge AP; pre-first-LM metrics; episode false alerts/lead time; stochastic calibration; and host-permutation sensitivity.

## Acceptance rule

RSSM is retained only if multi-step state/edge forecasting or uncertainty improves without increasing shortcut sensitivity. Do not select architecture on test. Ridge remains the baseline. The old F1 0.963 is superseded and must not be cited as valid V2 performance.

The senior's implementation is permanently unavailable because they judged its results too poor to continue. Do not wait for it, depend on it, claim comparison with it, or plan repository integration. Their description is architectural motivation only. Action-conditioned firewall imagination is not part of this immediate model.
