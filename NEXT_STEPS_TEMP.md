# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current verified status

- Matched frozen/unfrozen/zero-KL ablation: `docs/RSSM_ABLATIONS.md`.
- Validation-only KL grid: `docs/KL_TUNING.md`.
- Lower-KL candidate: KL 0.01/free 0/seed 7, test state MAE 0.276, edge AP 0.285, LM AP 0.926, pair top-1 9/14, spread 0.068.
- A 100-rollout evaluation gives LM F1/AP 0.828/0.914 and unchanged episode behavior (2/2 progressing, 2/4 negative episodes alert).
- Git clean at `fa16ff4`.

## Current code-writing batch: pair equivariance audit

Implement `scripts/21_audit_pair_equivariance.py` and compare:

```text
models/mvp_v2_rssm.pt
models/mvp_v2_rssm_kl_tuned.pt
```

For all six consistent host relabelings on validation and test:

1. Permute observable context node/edge slots only through the existing state permutation map.
2. Permute future edge/pair targets through the matching directed-pair map.
3. Run 100 Monte Carlo rollouts with no retraining or model selection.
4. Measure LM AP/F1, edge AP, pair AP/top-1, and pair top-choice consistency.
5. Compare permuted pair scores against the identity prediction reordered into the same coordinates; report MAE and correlation.
6. Compare decoded state/edge predictions against consistently reordered identity predictions.
7. Report per-pair support and correct counts so 9/14 cannot hide slot concentration.
8. Save `outputs/mvp_v2/rssm/pair_equivariance_audit.json`.

Validation:

- all six permutations present;
- pair/state index maps are true permutations;
- target positive counts are invariant;
- metrics finite/in range;
- no ground-truth field enters model input;
- compile and `git diff --check`.

## Decision gate

- If tuned pair accuracy and reordered scores are stable, retain the existing head provisionally but still require new episodes.
- If top-1 or scores vary materially by host relabeling, implement a shared-weight pair decoder before citing the gain.
- Do not select or retrain a model using audit test outcomes.
