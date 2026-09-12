# Supersede V5 v1.1 implementation after stale inherited equivariance gate

## Incident facts

- Corrected action run completed scratch optimization for seeds51001-3 in a temporary directory, then stopped before Friday optimization.
- Friday initial CPU smoke measured equivariance2.622604e-6. Frozen v1.1 scientific gate is <1e-5, but inherited V4 helper still hardcoded <2e-6.
- Scratch validation was computed internally for seed selection but no report/checkpoint/output was installed; temp cleanup succeeded. No result is available for tuning. No test export/inference.
- Partial log SHA0270c6f0... preserved. V1.1 freeze/scaler archived as invalid_partial.

## Correction

- Version protocol implementation as v1.2 without changing the frozen scientific threshold (<1e-5).
- Replace inherited prefit smoke wrappers with V5-owned CPU checks that read the protocol gate: exact future causality0, all120 equivariance<1e-5, finite one-batch gradients.
- Keep inherited loss/objective code, data, scaler, checkpoints, architecture, seeds,400/250 epoch budgets, selection rules, thresholds, and metrics unchanged.
- Test the actual scratch/Friday/V4 initialized models through the new smoke path before clean-tree freeze.
- Restart all action candidates from zero after new freeze; do not reuse temporary scratch states.
