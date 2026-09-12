# V5 train/validation export audit and model-protocol freeze

## Completed facts

- V5 corpus audit passes80/80. Capture freeze still passes.
- Generated only train/validation exports for passive, action, and passive_action.
- Shapes: passive train/val336/168; action train/val24/8; passive_action train/val24/8. Every sample has context[3,345], future[6,345],20 directed pairs.
- No V5 test sequence artifact exists. No model inference has run.

## Current batch

1. Add/run a reproducible export audit: exact common-array equality between action and passive_action; finite values; feature schema; whole-family split isolation; episode/cohort counts; target/action one-hot validity; ground-truth absence from observable feature names; immutable hashes; explicit absence of test exports.
2. Inspect V4/Friday/V3 model and training code plus actual checkpoint shapes before drafting V5 protocol.
3. Define one shared-slot scaler fitted only from V5 train observable contexts. No future/validation/test values enter scale fitting.
4. Freeze candidate initialization rules, seeds/budget, validation objective, passive/action comparisons, thresholds, uncertainty/branch reporting, checkpoint eligibility, and test unlock gate before training.
5. Do not export/read V5 test arrays or run any test predictions. Do not modify frozen capture-generating files or Docker images.

## Methodology posture

- Action model receives only a pre-cutoff chosen action and pair; passive model never receives it.
- Compare normalized state errors only under the exact same V5 train scaler.
- Carry protocol-eligible candidates according to a prespecified validation rule; training loss is not selection evidence.
- Oracle branch metrics are coverage only. Small validation calibration diagnostics are not deployment calibration.
- This critical protocol is the next Astra-review checkpoint if Astra is available; otherwise proceed conservatively and preserve an explicit pre-test review gate.
