# Supersede V5 pre-fit freeze after CUDA bitwise-zero smoke mismatch

## Incident facts

- First action trainer invocation stopped before optimizer construction/training at inherited V4 smoke assertion: future perturbation delta4.470348e-8 on CUDA versus exact-zero assertion.
- This is consistent with already documented CUDA float32 reduction noise (V5 contract observed2.980232e-8), not evidence of future leakage. CPU structural audit is exact zero.
- No V5 model/checkpoint/action output exists; no validation result selected; no test export/inference exists.
- Failed log preserved under outputs/mvp_v5/protocol_incidents with SHA5e3198d0.... Original freeze/scaler archived as model_protocol_v1_invalid_prefit and configs/mvp_v5_training_freeze_v1_invalid_prefit.json.

## Current correction

- Version protocol as v1.1 before any fit.
- Run structural future-causality and all120 equivariance gates on CPU, requiring exact zero causality and <1e-5 equivariance.
- Treat CUDA causality deltas <=1e-6 only as numerical diagnostics; never call them bitwise causal equality.
- Instantiate a CPU copy from the exact initialized candidate for inherited smoke tests; CUDA remains required for optimization.
- Apply same correction to action and passive trainers.
- Add incident record/docs/test, commit changed sources, create a new clean-tree freeze/scaler, then retry training.
- Do not alter capture files, exports, outcomes, model hyperparameters, seeds, budgets, validation rule, thresholds, or test seal.
