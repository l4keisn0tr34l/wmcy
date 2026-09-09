# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## V4 action test completed once

The one-shot action test ran after excluding the timing-invalid `action_7002` family. No retraining/tuning occurred.

```text
eligible test: 10 episodes / 5 paired families
scratch: state MAE .0603, active .2327, edge AP1.0, LM F1@.5 1.0, Brier8.91e-7
Friday:  state MAE .0719, active .2475, edge AP1.0, LM F1@.5 1.0, Brier1.60e-4
both: pair top1 1.0, factual action lower state error 10/10
```

The frozen train-derived thresholds were too extreme: scratch/Friday F1 .750/.889 despite perfect AP. Do not call 10-window ECE calibrated uncertainty. Scratch beat Friday state/active MAE and Brier; edge/pair/F1@.5 tied.

## Current batch: frozen passive branch V4 evaluation

1. Add one no-training evaluator for checkpoint SHA `4f0524...`.
2. Evaluate all sliding windows from predetermined passive-branching and direct-credential cohorts (`lab_097`–`lab_108`).
3. Separately evaluate the same 10 pre-action contexts used by the action-test model, but without action input.
4. Use V3 validation-frozen LM threshold `0.3707732260` and also report threshold-free AP/Brier.
5. Report expected/oracle/conditioned state coverage, expected edge AP, branch diversity, matched episode alert behavior, and cohort counts.
6. Do not tune or retrain the passive model and do not rerun the action evaluator.
7. Document both results, timing exclusion, limitations, report, commit, push.

Do not reopen V3 test.
