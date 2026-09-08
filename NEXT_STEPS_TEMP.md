# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Completed experiment

A matched V2 representation-training ablation is complete in `outputs/mvp_v2/rssm/ablations.json`.

Regimes use identical architecture/data/splits and validation-only selection:

1. published joint model selected on validation dynamics;
2. self-supervised telemetry/edge/KL pretraining, frozen RSSM, identical internal PyTorch semantic heads;
3. same pretraining followed by full unfreezing;
4. joint from scratch with matched joint validation selection;
5. joint zero-KL with matched selection.

Key test values:

```text
regime                              state active edgeAP LM-F1 LM-AP pre-F1 pair  spread
published joint                      .280  1.031   .222  .800  .910   .769 .143   .076
frozen two-stage                     .279  1.047   .240  .757  .816   .727 .071   .079
pretrained then unfrozen             .275  1.040   .210  .800  .869   .815 .214   .067
joint scratch, matched selection     .293  1.041   .152  .839  .839   .815 .071   .085
joint zero-KL                        .280   .984   .316  .839  .904   .846 .286   .025
```

Interpretation:
- Frozen self-supervised latents support nontrivial semantics, but unfreezing improves LM F1, pre-first-LM F1, and pair ranking relative to the matched frozen condition.
- Pretraining improves dynamics/edge/pair relative to matched-selection joint-from-scratch, but not every thresholded semantic metric.
- Zero-KL gives strong point metrics but collapses mean stochastic spread by about 67% versus the published joint model and has a worst host-permutation LM range of 0.814. It is an ablation, not the new selected model. The result motivates KL-weight/free-nats tuning, not immediate KL removal.
- Only 14 positive test windows and 14 pair-positive windows exist; differences are fragile.

## Current code-writing batch

1. Add `docs/RSSM_ABLATIONS.md` with methods, inputs/outputs, leakage controls, exact metrics, interpretation, and limitations.
2. Extend `scripts/17_build_rssm_report.py` to add an inline ablation table and conclusions from verified JSON.
3. Regenerate/parse-check the standalone report and copy it to Downloads.
4. Update commands, output locations, status, roadmap, decisions, TODO, and memory.
5. Compile scripts, inspect artifacts, run `git diff --check`, commit, and push.

## Next experimental batch after documentation

1. Tune KL weight/free-nats on validation only (small grid), monitoring point forecasting, spread, host sensitivity, and LM/edge ranking.
2. Improve edge/pair prediction with a shared-weight pair decoder rather than independent fixed-slot output weights.
3. Add matched scan/guessing non-progression episodes before claiming reduced false alerts.
4. Defer transformer, DANN, ensemble, and action conditioning for the reasons already recorded.

## Epistemic constraints

- Do not call zero-KL categorically superior.
- Do not claim spread is calibrated uncertainty.
- Do not treat overlapping test windows as independent events.
- Do not merge senior-reported results with project results.
- Keep the published joint RSSM as the current report headline until broader data and calibration support replacement.
