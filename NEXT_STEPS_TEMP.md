# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current verified status

Matched representation ablations are complete and documented in `docs/RSSM_ABLATIONS.md`. The updated standalone report is `/home/paprika/Downloads/rssm_eod_report.html` (SHA-256 `4e94e5bc98744bbb504abca2a81d70505bc31e761e3e7c991990eaab71547ff3`). Git is clean at `1e52e21`.

## Current code-writing batch: KL/free-nats validation grid

Implement `scripts/19_tune_rssm_kl.py` with these invariants:

1. Use the unchanged V2 whole-episode 12/6/6 split and 141-feature observable input.
2. Fit scaling on training contexts only.
3. Keep architecture, augmentation, loss weights other than KL, and semantic heads unchanged.
4. Screen KL weights `{0, 0.01, 0.03, 0.1, 0.3}` and free-nats `{0, 1}` where meaningful, using seed 7 and validation only.
5. Evaluate no test arrays during screening/shortlisting.
6. Require candidates to preserve at least half the reference validation rollout spread, remain within 3% of reference validation state MAE, retain at least 90% of reference validation edge/LM AP, and avoid >1.5x reference mean host-permutation range.
7. Rank eligible candidates with an explicit state-primary validation score: state-MAE ratio minus 0.25 edge-AP ratio minus 0.25 LM-AP ratio.
8. Confirm the top two eligible settings across seeds 7/17/27, still using validation only.
9. Select one setting/seed on validation, then evaluate test exactly once in this script.
10. Save complete validation screening, eligibility, selection, final test, model checkpoint, and limitations.
11. Report raw posterior/prior KL, Monte Carlo spread, and host-permutation sensitivity so low spread is never mistaken for calibrated confidence.

Tests:

- compile and short smoke grid;
- assert screening artifact has no test metrics;
- assert all values finite and ranges valid;
- assert final selection satisfies recorded gates or is marked fallback;
- inspect final JSON and checkpoint;
- `git diff --check`.

## After KL tuning

1. Document the result and decide whether the published model changes; do not switch solely for one favorable test metric.
2. Add a shared-weight source-target pair decoder.
3. Generate matched scan/guessing hard negatives.
4. Defer transformer, DANN, ensemble, and action conditioning.
