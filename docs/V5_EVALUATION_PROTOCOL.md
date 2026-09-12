# V5 sealed-evaluation protocol

## Status

**Implemented and validation-smoke tested; awaiting final critical review before creating the evaluation freeze. V5 test has not been exported or inferred.**

Relevant sources:

```text
scripts/57_evaluate_v5_models.py
scripts/58_freeze_v5_evaluation.py
scripts/59_run_v5_sealed_evaluation.py
outputs/mvp_v5/evaluation_protocol/validation_smoke.json
```

## Frozen-before-test inputs

The evaluation freeze will hash:

- active V5 training freeze and protocol;
- shared train-context scaler;
- independent validation audit;
- all six retained checkpoint files;
- scratch primary selections for action and passive tracks;
- fixed0.5 and secondary validation-derived thresholds;
- evaluator, exporter, atomic wrapper, model, and metric sources;
- validation-mode evaluator smoke output;
- Python/NumPy/Pandas/scikit-learn/Torch/CUDA runtime versions;
- expected test sample counts and exact metric list.

The freeze script refuses if any test export or sealed report already exists or if the Git tree is dirty.

## One-run mechanism

`scripts/59_run_v5_sealed_evaluation.py` requires a matching evaluation freeze. It:

1. refuses an existing sealed-test result or legacy test export;
2. creates one temporary output root;
3. exports passive, action, and action-aligned-passive test arrays into that root;
4. verifies action and aligned-passive common arrays are bitwise equal;
5. runs the hashed evaluator once;
6. hashes all output artifacts and records timestamps/commit;
7. atomically renames the complete root to `outputs/mvp_v5/sealed_test/`.

If export/evaluation fails after test access, evidence is preserved under `_failed_sealed_test_*`; the project must inspect and disclose it rather than casually rerun or tune.

## Test scopes

```text
action:         8 intervention-aligned episodes
passive:        336 windows from16 non-action episodes
passive_action: 8 contexts exactly aligned with action test
```

The test-only intent probe is evaluated only after unlock. For each of its eight episodes, the evaluator selects the unique passive window whose prediction-availability time equals the metadata-frozen forecast cutoff. It then compares four malicious direct-credential episodes with four matched legitimate SSH episodes, including paired-family probability differences. It does not select a new threshold on test.

## Metrics

For scratch, Friday, and V4 initialization:

- normalized state MAE under the same train-only scaler: overall/global/node/edge/active/quiet;
- persistence state and edge references;
- future-edge average precision;
- ATT&CK micro and per-technique AP where both classes exist;
- completed-LM AP/Brier/F1 at fixed0.5;
- secondary F1 at the already frozen validation threshold;
- completed-LM pair AP/top-1;
- action factual-versus-opposite state preference;
- passive expected/oracle/diversity metrics;
- scenario-level counts and diagnostics;
- test-only intent-probe ranking/probability differences.

Scratch is the primary action and passive model because validation selected it. Friday/V4 test values are prespecified transfer diagnostics, not post-test candidate selection.

## Leakage prevention

The evaluator fits nothing. It does not optimize thresholds, epochs, seeds, scalers, branches, or model weights. It verifies checkpoint/scaler/source hashes before prediction. ATT&CK, LM, action outcomes, scenarios, and intent labels remain evaluation targets/audit strata, never context features.

## Interpretation limits

- Perfect action LM semantics can follow deterministic permit/block and must not overshadow state/edge forecasting.
- Oracle branch MAE is future coverage, not deployable inference accuracy.
- Small correlated-window Brier/ECE values do not establish calibration.
- Intent differences can reveal observable telemetry differences in independently generated pairs; they do not prove hidden-intent recovery.
- One flat synthetic topology cannot establish enterprise or unseen-topology generalization.

## Final gate

Before running `scripts/58_freeze_v5_evaluation.py` without `--check-only`, perform the requested critical Astra-style review of this document, scripts57–59, validation audit, thresholds, and claim limits. After the resulting freeze is committed, execute script59 once and do not rerun or retune based on the result.
