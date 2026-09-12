# V5 sealed-evaluation protocol — reviewed contract

## Review gate

Astra accepted the corrected implementation on2026-09-12 after code, validation-artifact, and synthetic-fixture review. See `docs/V5_FINAL_REVIEW.md`. No V5 test content was read during review. This document describes the immutable pre-test contract; execution status/results belong in `docs/CURRENT_STATE.md` and the sealed report.

## Inputs and transformation

- Three complete five-second observable graph states (345 features) predict six future graph states/edge-presence arrays.
- Action models additionally receive only a chosen pre-cutoff action type and directed pair. Passive branching does not.
- LM/ATT&CK/LM-pair heads output horizon-level scores. Future truth is used only to score forecasts.
- One shared train-context-only scaler is applied; nothing is fitted during evaluation.
- Scratch is the validation-selected primary for both tracks. Friday/V4 are prespecified transfer diagnostics, not post-test candidate selection.

## Frozen lineage before test I/O

Script58 verifies active training sources and all train/validation export hashes, the actual shared scaler, checkpoint/report selection lineage, original checkpoint thresholds, and the reviewed evaluator smoke. It records:

- all six checkpoint hashes and selected seed/epoch;
- source hashes covering the models, metric helpers, exporter, plan validator, seal helper, wrapper, tests, reviewed documents, plans/splits, and training freeze;
- validation audit/reports, scaler, checkpoint and reviewed-smoke artifact hashes;
- Python/NumPy/Pandas/scikit-learn/Torch/CUDA versions;
- primary models, exact secondary thresholds, expected samples, metrics, and claim limits.

The freeze must be committed in a clean tree. Before export, the wrapper rechecks **every** source/artifact hash and runtime. CUDA must be available. Tests cannot be unlocked through a changed scaler/exporter/dependency unnoticed.

## Permanent single-attempt mechanism

Script59 exclusively creates and fsyncs:

```text
outputs/mvp_v5/sealed_test_attempt/claim.json
```

before reading any test episode. This directory is never removed, even on failure. An existing claim, completed result, failed legacy result, or partial legacy staging root blocks another attempt. Concurrent calls cannot both acquire the directory.

The claiming wrapper exports three modes to `sealed_test_attempt/work/sequences/`. Only its child evaluator can exclusively create `inference_started.json`, with the matching freeze and exact claimed output paths. Direct evaluator reruns/alternate outputs are refused. A failure preserves the complete work directory and consumes the attempt.

After successful evaluation, predictions/report/arrays/logs are hashed and the work directory is atomically installed as:

```text
outputs/mvp_v5/sealed_test/
```

The permanent claim remains. This is an accidental-retry/mutation guard, not protection against a malicious filesystem owner.

## Bundle checks

Expected test scope:

```text
action:         8 contexts / 8 episodes / 4 paired families
passive:        336 windows / 16 episodes / 8 paired families
passive_action: 8 contexts identical to action, without action input
```

Check exact schema/slot order/array keys/shapes, finite telemetry, binary target/action fields, one-hot actions, plan membership/strata, ordered sample IDs, 21 unique sliding windows per passive episode, complete3→6 context/future indexing, source hashes, and LM/pair target agreement. Action/aligned-passive arrays and manifests must match exactly.

## Thresholds and metrics

Primary LM threshold is0.5. Secondary thresholds are copied exactly from each original selected checkpoint's ordinary validation record; never reoptimized in the evaluator, aligned test, or intent probe.

For all three initializations:

- same-scaler state MAE overall/global/node/edge/nonzero-feature/zero-feature and by six forecast steps;
- persistence state/edge references and edge positive rates;
- future-edge AP;
- ATT&CK micro/per-technique AP with support counts;
- completed-LM AP/Brier/F1 at primary and frozen secondary thresholds;
- completed-LM pair AP/top1;
- factual-versus-opposite action state preference;
- passive weighted-point/oracle/diversity measures;
- scenario/background-profile/episode metrics.

Primary passive model additionally reports pre-first-event episode alerts and exact lead to successful SSH action start, plus the test-only intent probe. Saved prediction arrays allow later displays/audits without another inference run.

## Intent and warning alignment

For each of eight intent-probe episodes, select the unique passive sample whose prediction-availability timestamp equals the **metadata-frozen pre-SSH forecast cutoff**. Enforce UTC/five-second alignment, no previously observed LM, all four paired families, and expected malicious/legitimate horizon labels. Report probability differences and metrics with the already frozen thresholds. Neither outcome nor prediction score chooses the cutoff.

Warning detection requires a strictly pre-onset alert with controller-recorded successful SSH action start inside its30s horizon. Earlier alerts and nonprogressing false alerts are separate. This timestamp is not an independently measured compromise-completion instant.

## Limitations

Perfect action labels can follow the given action/pair; they cannot establish learned graph forecasting by themselves. Passive/action training cohorts differ, so the aligned comparison is not a pure conditioning ablation. Paired captures are independent, not identical counterfactuals. Overlapping windows and small families do not support deployment-calibration or strong statistical claims. Oracle error measures candidate coverage; deterministic nonlinear decoding is not an exact probabilistic expectation. One flat synthetic topology cannot establish enterprise/unseen-topology generalization or policy causality.

## Authorized sequence

After accepted review, from committed clean sources:

```bash
.venv/bin/python scripts/58_freeze_v5_evaluation.py
# inspect and commit configs/mvp_v5_evaluation_freeze.json
.venv/bin/python scripts/59_run_v5_sealed_evaluation.py
```

Run script59 once. On any failure, preserve evidence and stop. On success, no post-test retuning or inference rerun; consume saved artifacts for reporting.
