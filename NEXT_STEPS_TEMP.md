# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Verified V3 experiment

All 48 episodes and 720 sequences pass. The prespecified scratch/public-initialized GraphRSSMs were selected independently on V3 validation; thresholds frozen before model test predictions.

```text
Fresh test               Scratch       UNSW init
state MAE                 0.294         0.321
active MAE                0.556         0.579
edge AP                   0.808         0.796
LM F1 / AP                0.560/0.477   0.596/0.504
pre-first-LM F1           0.571         0.596
pair top-1                0.333         0.333
progress episodes found   3/3           3/3
stopped episodes alert    3/3           3/3
mean lead                 23.5s         23.5s
spread/error corr         0.768         0.533
```

UNSW initialization does not transfer positively under the fresh matched distribution: scratch is better for state/edge/uncertainty; UNSW is slightly better for LM point scores. Exact score equivariance remains around 1e-8, but near-tied argmax pair choices can vary under floating-point/MC perturbations.

Important interpretation: same-seed stopped/progressing episodes intentionally share the observable discovery/guessing prefix. Whether the external scenario controller performs later SSH is not in passive telemetry. Alerting on both can represent valid compromise risk; exact binary eventual-outcome discrimination is partly unidentifiable without action/intent/intervention variables.

Disclosure: before training, test NPZ was opened only for schema/finiteness/count integrity, not model predictions. Model test predictions occurred only after both regimes and thresholds were frozen.

## Current batch

1. Implement `scripts/31_audit_v3_matched_pairs.py`.
2. Pair stopped/progressing episode summaries by seed and split.
3. Report max-risk overlap, paired ordering, both-alert rate, action timing mismatch, and per-regime differences.
4. Do not alter model, threshold, or test selection.
5. Document the observed identifiability boundary and public-transfer result.
6. Update standalone HTML with fresh-holdout results and limitations.
7. Compile, integrity-check, commit, and push.
