# V5 pre-fit protocol incident: CUDA bitwise-zero smoke assertion

## Status

**Resolved by superseding the first training freeze before any optimizer step, checkpoint, validation selection, test export, or test inference.** This is an implementation-contract correction, not model tuning.

## Observed failure

The first invocation of:

```bash
.venv/bin/python scripts/53_train_v5_action.py --device cuda
```

stopped in the inherited V4 pre-training smoke helper:

```text
AssertionError: future leakage delta 4.470348358154297e-08
```

The failure occurred before optimizer construction for the first scratch candidate. `outputs/mvp_v5/action_model/` and all `models/mvp_v5_*` checkpoints were absent afterward. V5 test exports remained absent.

Raw log:

```text
outputs/mvp_v5/protocol_incidents/action_prefit_cuda_smoke_failure.log
SHA-256 5e3198d04ed426b20b2747a94e4f268312cdb25d89fd32308cddebd908edfae3
```

## Interpretation

Mathematically, changing future observations must not alter the context-anchored rollout. On CPU this delta is exactly zero. The already completed V5 CUDA contract test observed `2.9802322387695312e-08` under the same type of perturbation, while gradients remained finite and CPU causality was exact. The new `4.47e-8` value is the same scale as ordinary float32 CUDA reduction-order noise and far below the frozen `1e-5` equivariance tolerance.

Therefore:

- it is not evidence that future targets enter the model input;
- it is also not honest to call the CUDA result bitwise equal;
- the inherited exact-zero CUDA assertion was inconsistent with evidence known before training.

## Correction

Protocol v1 is archived as invalid pre-fit:

```text
configs/mvp_v5_training_freeze_v1_invalid_prefit.json
outputs/mvp_v5/model_protocol_v1_invalid_prefit/
```

Protocol v1.1 changes only numerical audit execution:

- exact future-perturbation causality is required on CPU (`0.0`);
- all120 deterministic host permutations are audited on CPU (`<1e-5`);
- CUDA deltas up to `1e-6` are numerical diagnostics, not bitwise-causality claims;
- optimization still runs on CUDA;
- smoke tests use a CPU copy of the exact initialized candidate.

No data, split, scaler policy, initialization, architecture, loss weight, seed, epoch, batch, validation rule, threshold, metric, or test gate changed.

## Second stale-helper stop under v1.1

After the v1.1 freeze, all three scratch seeds completed in a temporary directory. Before Friday optimization began, the inherited V4 helper rejected the frozen Friday initialization at CPU equivariance `2.6226043701171875e-06` because that helper hardcoded `<2e-6`. The v1.1 protocol had already frozen the scientific gate at `<1e-5`. Thus the candidate was inside the declared protocol and the wrapper was stale.

No scratch report/checkpoint was installed; temporary states were deleted and will not be resumed. Although scratch validation was calculated internally by the fixed script, no metric was emitted or retained for changing choices. No Friday/V4 optimization or test access occurred.

```text
outputs/mvp_v5/protocol_incidents/action_partial_scratch_stale_equivariance_helper.log
SHA-256 0270c6f0e01198fa962ad0ee1dca0eaa113e8a1d2b3e65723488adee5340cffb
configs/mvp_v5_training_freeze_v1_1_invalid_partial.json
outputs/mvp_v5/model_protocol_v1_1_invalid_partial/
```

Protocol v1.2 replaces inherited smoke gating with V5-owned checks that implement the already declared contract: CPU causality exactly0, all120 CPU equivariance below`1e-5`, and finite one-batch gradients. Inherited model losses remain unchanged. Prefreeze tests exercise scratch, Friday, and V4 source dictionaries through this exact path.

## Methodological consequence

Both corrections concern fail-closed numerical plumbing. Neither changes data, scaler, initialization sources, architecture, optimizer, loss, seed, budget, selection, threshold, metric, or test policy. All action candidates restart from zero under a new freeze; temporary scratch states are not reused. The later independent evaluation freeze remains mandatory.
