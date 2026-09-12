# V5 validation complete — independent checkpoint audit and sealed-evaluation freeze

## Observed validation facts (test still absent)

Action selected dynamics scores scratch1.019175, V41.092597, Friday1.213829; scratch is primary. Selected seed51003 for all. Shared-scaler state MAE scratch.472703, V4.498718, Friday.537277; edge AP only.349903/.339970/.357509. LM ranking/F1 at validation-derived threshold is1 but deterministic action makes this easy; thresholds are extreme and fixed.5 must be independently reported. Permit-minus-block mean risk .984588/.999539/.993121. All causality0 and CPU equivariance<1e-5.

Passive branch selected composite scratch.931808(seed51003 epoch79), V4.944622(seed51002 epoch114), Friday.957200(seed51003 epoch39); scratch primary. Expected/oracle state MAE scratch.460888/.459664, V4.455362/.453578, Friday.465553/.464357. Oracle gains only.00122/.00178/.00120 despite diversity. LM AP scratch.1615, Friday.0651, V4.0623; fixed.5 F1 all0. Aligned action-context fixed.5 F1 all0. Passive alternatives do not establish robust early warning.

## Current batch

- Preserve v1/v1.1 incidents; v1.2 training outputs are immutable.
- Add an independent validation-checkpoint audit that reloads every checkpoint, verifies hashes/scaler/protocol/test seal, recomputes primary fixed.5 and state/edge/LM/pair/counterfactual metrics, and checks saved selection/eligibility.
- Document validation evidence without presenting it as test/generalization.
- Implement the sealed evaluator and a separate evaluation-freeze gate. Freeze selected checkpoint hashes, primary candidates, thresholds, metrics, evaluator source, and no-test status before export.
- Appropriate Astra checkpoint is immediately before evaluation freeze/test unlock. If unavailable, preserve freeze and do not open test until explicit final review.
