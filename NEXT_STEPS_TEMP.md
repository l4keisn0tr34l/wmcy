# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Completed in this autonomous experiment cycle

1. Matched frozen/unfrozen/zero-KL RSSM ablation (`scripts/18_run_rssm_ablations.py`).
2. Validation-only nine-setting KL/free-nats grid with test loaded only after selection (`scripts/19_tune_rssm_kl.py`).
3. Generic 100-rollout checkpoint/episode evaluator (`scripts/20_evaluate_rssm_checkpoint.py`).
4. Six-way host/pair equivariance audit (`scripts/21_audit_pair_equivariance.py`).
5. Frozen shared-weight pair decoder experiment (`scripts/22_train_shared_pair_decoder.py`).
6. Updated standalone report and full methodological documentation.

## Durable findings

- Frozen telemetry-pretrained latents contain useful security signal: LM F1 0.757.
- Unfreezing improves matched LM F1 to 0.800, pre-first-LM F1 0.727→0.815, and pair top-1 0.071→0.214, but not every metric.
- Zero-KL improves point metrics but collapses mean spread 0.076→0.025; do not adopt it.
- Validation-only tuning selected KL 0.01/free 0/seed 7. Test-isolated 20-rollout estimates: state 0.276, edge AP 0.285, LM AP 0.926, spread 0.068. A 100-rollout episode check gives LM F1/AP 0.828/0.914, 2/2 progressing episodes detected ~26.4 s early, and 2/4 negative episodes alerting.
- Tuned identity pair top-1 is not robust. Across equivalent relabelings it ranges 2/14–8/14 and non-identity choices almost never map back to the identity choice.
- A shared final pair scorer lowers score equivariance MAE 0.031→0.021 but worsens permutation-mean AP 0.267→0.232 and top-1 0.274→0.202. Do not adopt it. The fixed-slot encoder/decoder is the upstream bottleneck.

## Current preferred model/report position

- Keep the published joint RSSM as the stable headline: state MAE 0.280 and LM F1 0.800.
- Use the KL 0.01 checkpoint only as research initialization.
- Do not headline pair top-1; explicitly cite failed equivariance.
- Updated standalone report: `/home/paprika/Downloads/rssm_eod_report.html`.

## Next major work

1. Design a permutation-equivariant graph encoder/decoder with shared node/edge message passing.
2. Generate matched scan/failed-guessing non-progression episodes to attack false alerts.
3. Define a fresh holdout before selecting further architecture variants; the current test has been inspected repeatedly.
4. Calibrate uncertainty only after expanding data.
5. Continue deferring transformers, DANN, ensembles, and action conditioning.

## Permission status

No sudo, Docker restart, credential entry, or interactive permission was required for this cycle.
