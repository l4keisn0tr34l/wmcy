# V5 final evaluation review — Astra, 2026-09-12

## Decision

**Accept the corrected evaluator for one sealed V5 evaluation.** Review used code, frozen plans, training/validation artifacts, and synthetic fixtures only. No V5 test episode content, test array, or test prediction was accessed during this review. The test-only intent path was tested using eight fabricated fixtures, not the real intent episodes.

Acceptance is narrow: report the prespecified controlled-lab results and failures. It is not endorsement of enterprise readiness, useful passive warning, calibrated uncertainty, or causal policy value.

## Blocking implementation findings and fixes

1. **Hash inventories were not enforced in full.** The old wrapper checked only itself; the evaluator did not check the actual scaler against the seal. The corrected helper verifies every frozen source and artifact before test access, including models, scaler, plans, validation lineage, exporter dependencies, and runtime. The freeze must match its committed Git blob in a clean tree.
2. **One-run policy was advisory.** Failed/concurrent attempts and alternate evaluator output paths could bypass it. The wrapper now atomically creates a permanent, fsynced `sealed_test_attempt/` before any export. It is never deleted. Failures consume the attempt. An exclusive inference ticket permits only the claiming wrapper's child evaluator, under fixed staging paths, once. Existing failed/partial legacy runs also block access.
3. **Secondary passive thresholds drifted in the CPU audit.** That audit called a threshold search again. The evaluator now uses the original selected checkpoint's validation threshold exactly (Friday `0.06516405940055847`, not the CPU reoptimized `0.06516405194997787`). Primary thresholds remain exactly0.5. No threshold is searched on test or on the aligned/intent test subsets.
4. **Some promised evidence was absent.** Added background-profile/episode diagnostics, six per-step state errors, primary passive pre-first-action warning/false-alert summaries, and saved forecast arrays. These are reporting additions before test, not candidate-selection changes. Saved arrays enable subsequent reporting without test inference reruns.
5. **Test-bundle checks were incomplete.** Enforce the 345-feature whitelist/order, array keys/shapes/finiteness/binary values, plan membership/strata, unique windows, 21 passive or one action sample per episode, source hashes, target consistency, and exact common arrays/manifests for action and aligned passive.

Capture and training source files, data, scaler values, model weights, seeds, epochs, losses, primary selections, and primary thresholds are unchanged.

## Validation lineage

All active training-freeze source/export hashes match. All six retained checkpoint hashes and embedded validation records match the reports. Selection rules reproduce:

| Track | Scratch | Friday | V4 |
|---|---|---|---|
| Action seed/epoch |51003/400|51003/400|51003/400|
| Passive seed/epoch |51003/79|51003/39|51002/114|

Scratch remains primary in both tracks by their separate original validation objectives. This does not imply scratch is best on every individual metric.

Original selected secondary thresholds:

| Track | Scratch | Friday | V4 |
|---|---:|---:|---:|
| Action |0.999037504196167|0.9830637574195862|0.9994845390319824|
| Passive |0.10029404610395432|0.06516405940055847|0.058327820152044296|

The extreme action thresholds are brittle diagnostics, not calibrated decision rules.

## Model/input and temporal review

- `forecast_action` observes context only, then encodes the chosen type/pair and rolls forward six prior states. Passive branching sees neither action nor target outcome at inference.
- Heads output **horizon-level** LM, ATT&CK, and LM-pair scores; graph state/communication edges are six-step outputs. Older prose claiming six-step semantic logits is incorrect.
- The scaler uses440 unique context-visible training states with statistics shared across anonymous host/pair slots. Old141-feature scalers are not reused.
- Roles, seeds, scenario, action result, truth, and absolute time remain outside observable tensors. Metadata is used for audit/evaluation alignment only.
- Frozen runtime schedules set the forecast cutoff before the focal SSH command in both intent alternatives. Every nominal decision is80–103s after capture start; the next five-second boundary lies within the complete context/future coverage. Selecting that boundary is prespecified, not hindsight optimization.
- The intent result checks all eight episodes/four families, one exact UTC-grid cutoff per episode, no previously observed LM, and correct malicious/legitimate horizon labels. It is not a packet-identical counterfactual experiment.
- Warning lead is to the controller-recorded start of a successful SSH action, **not** an independently observed compromise-completion time. Only strictly pre-onset alerts with onset inside30s count as forecast detections; earlier alerts are reported separately.

## Tests and inspected output

`scripts/60_test_v5_evaluation_seal.py`: eight synthetic tests pass for source/scaler corruption, repeated/concurrent claims, inference-ticket replay/path/freeze mismatch, failed/partial-run refusal, direct evaluator gating before bundle loading, and exact intent cutoff selection/rejection.

The reviewed evaluator ran on CUDA with validation only. All six models reproduced independently audited point state metrics within`1e-6`; fixed0.5 F1 remained action1/passive0. Fifty-four forecast arrays were preserved and checked for finiteness and expected shape. Source/scaler/model/test-seal checks pass.

Reviewed smoke:

```text
outputs/mvp_v5/evaluation_protocol/reviewed_validation_smoke/report.json
outputs/mvp_v5/evaluation_protocol/reviewed_validation_smoke/report.predictions.npz
```

## Remaining methodological limits

- Eight action-test contexts are only four paired families; passive overlapping windows are not independent observations. Report episode/family counts, not inflated sample certainty.
- Shared-scaler aligned errors are numerically comparable, but action/passive models have different training cohorts and objectives. This is **not** a controlled ablation isolating action conditioning.
- The action input specifies the intervention pair and the lab outcome is deterministic. Perfect LM/pair scores can be obtained from that information without sophisticated telemetry inference. State/edge forecasting and persistence references are essential.
- Branch-weighted deterministic latent-mean decoding is an operational point forecast, not an exact distributional expectation through a nonlinear model. Oracle error is best candidate coverage only and need not beat the weighted mean in every subset.
- AP/top1 can move near floating-point ties. CUDA execution is frozen, with no post-test numerical/model retuning. No new structural permutation sweep is run on test.
- The integrity gate prevents accidental mutation/retry, not a malicious owner editing local files. It is not cryptographic access control.
- Raw test content is first read only after the permanent claim; its hashes are recorded by export/provenance rather than pre-read during review.

## Execution authorization

Commit corrected sources and this review; create the evaluation freeze from the clean tree; inspect and commit that freeze. Then execute script59 once. Preserve every output/failure. Any failure consumes the attempt and requires a disclosed decision, not automatic retry. After success, use saved predictions and report artifacts only; do not retrain, retune, or rerun test.
