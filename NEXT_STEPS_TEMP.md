# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## V4 milestone complete

Action and passive evaluations are frozen and inspected. Do not retrain, retune thresholds, or rerun either sealed evaluator.

### Chosen-action model: 10 episodes / five pairs

```text
scratch state/active MAE: .0603/.2327
Friday state/active MAE:  .0719/.2475
both edge AP / LM AP / pair top1: 1.0 / 1.0 / 1.0
factual action lower state error: 10/10 for both
```

`lab_090` had only five complete post-action states. The complete `action_7002` family (`lab_089/090`) was excluded using timing only before prediction.

### Frozen passive branch: 180 windows / 12 episodes

```text
expected/oracle state MAE: .248/.237
edge AP: .556
LM F1/AP/Brier: .483/.472/.107
stopped scan/guess alerted: 3/3
progressing scan/guess pre-positive alerted: 0/3
direct credential pre-positive alerted: 0/3
matched legitimate alerted: 1/3
```

On the same 10 pre-action contexts without action: state MAE .141 and LM F1/AP .400/.519. This supports the observability boundary and chosen-action conditioning; it is not calibrated or enterprise evidence.

## Report

```text
/home/paprika/Downloads/rssm_eod_report.html
outputs/mvp_v2/report/rssm_eod_report.html
SHA-256 b341b5a103aef28614731b2f5d14150c5dc9c33dfdbe7668b85b7b511c61e208
```

## Next work

1. Run final integrity/docs checks, commit, and push this V4 consolidation.
2. Visually inspect the standalone report in a browser; content/assets checks already pass.
3. Do not tune against V3 or V4.
4. Design a new corpus with additional topologies, host counts, background processes, direct-credential paths, and interventions.
5. Create topology/scenario-disjoint train/validation/test before model changes.
6. Evaluate public initialization across multiple disconnected precise captures.
7. Add calibration only after enough independent validation episodes exist.
