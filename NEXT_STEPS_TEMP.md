# V5 final Astra review gate — test still sealed

## Completed

-80/80 corpus and train/validation exports audited.
- V1.2 training protocol frozen after preserving two numerical-wrapper incidents; no test access in either.
- Action validation primary: scratch seed51003, selection1.0192, same-scaler state MAE.4727, edge AP.3499, LM/pair F1/AP1 under deterministic action, factual state wins7/8.
- Passive primary: scratch seed51003 epoch79, expected/oracle MAE.4609/.4597, LM AP.1615, F1@.5=0, pair AP.0092, oracle gain.0012. This is weak passive warning/coverage evidence.
- Six checkpoints independently reload/hash/recompute and pass CPU causality0/all120 equivariance<1e-5.
- Scripts57–59 implement validation-smoked evaluator, pretest freeze, runtime lock, atomic three-mode test export/evaluation, intent probe, and failed-run preservation.
- No V5 test export, prediction, freeze, or sealed result exists. Repository clean/pushed at0158a0a before this note.

## Mandatory next gate

Use Astra for final review of `docs/V5_EVALUATION_PROTOCOL.md`, scripts57–59, validation audit, thresholds, intent cutoff, and claim limits. Do not run script58 without review because it creates `test_unlock=true`.

After accepted review only:

1. `.venv/bin/python scripts/58_freeze_v5_evaluation.py`
2. inspect and commit `configs/mvp_v5_evaluation_freeze.json`
3. `.venv/bin/python scripts/59_run_v5_sealed_evaluation.py` exactly once
4. inspect immutable `outputs/mvp_v5/sealed_test/`; no post-test retuning/rerun
5. update report/presentation with successes and failures.
