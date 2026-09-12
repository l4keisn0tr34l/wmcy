# Astra final V5 review — seal hardening before any test access

Observed blockers: source/scaler/artifact hash inventories recorded but not fully enforced before export; no durable consumed-run claim or concurrent-run exclusion; direct evaluator can bypass wrapper; failed attempts do not block retry. Passive CPU audit recomputes secondary threshold rather than using original checkpoint value. Evaluation misses promised profile diagnostics and saved predictions.

Invariant: before test I/O validate full frozen dependency/artifact lineage; atomically consume one permanent run claim; only its matching staging evaluator may execute once. Failures consume the claim and preserve evidence. No model/seed/budget/primary threshold changes. Secondary thresholds come exactly from selected checkpoint validation fields. Use synthetic fixtures and validation only during review. Preserve historical outputs.

Implement new evaluation-only seal helper and tests, harden scripts57–59, preserve forecasts and add profile/horizon/episode warning diagnostics. Do not modify capture/training frozen files. Inspect validation outputs, write review acceptance with limitations, commit reviewed sources, create/inspect/commit freeze, then run wrapper exactly once if accepted. No V5 test files are read before that boundary.
