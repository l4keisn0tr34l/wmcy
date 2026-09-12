# V5 one-shot evaluation completed — 2026-09-12

Astra review accepted after evaluation-only guard fixes. Reviewed sources committed b1c5901; freeze inspected/committed e1497ff. Script59 executed exactly once at12:00UTC and succeeded. The permanent consumed attempt is outputs/mvp_v5/sealed_test_attempt/. Do not rerun any V5 test inference, change frozen files, retune thresholds, retrain candidates against this test, or delete evidence.

Current documentation batch: record immutable results and identities in docs/V5_TEST_RESULTS.md, CURRENT_STATE, MEMORY, TODO, DECISIONS, and OUTPUT_LOCATIONS. No model/evaluator/capture code changes.

Verified all12 sealed artifact hashes, all six checkpoint/scaler/freeze/threshold identities,54 finite saved forecast arrays, and nine point MAEs recomputed from saved predictions only. No failed sealed attempt occurred. Read-only inspection snippets had key/signature/reshape mistakes that were corrected; these never called model inference or modified results.

Primary action state MAE .459835 vs persistence .539809; edge AP .411418; LM F1@.5=1 under deterministic permit/block; factual action state wins8/8. Primary passive expected/oracle MAE .472887/.470838 vs persistence .538841; edge AP .369933; LM AP .200029, F1@.5=0. Frozen low threshold detects3/6 progression episodes but falsely alerts7/10 nonprogressing episodes. Intent-only cutoff F1@.5=0, secondary .4; no hidden-intent recovery claim.

Aligned same-scaler scratch passive MAE .468434 versus action .459835: small overall action advantage, but active-feature error is worse (.787490 versus .661994) and training cohorts differ, so not a pure action-conditioning ablation. Friday transfer does not improve scratch dynamics. V4 has lower passive point MAE, but scratch remains the frozen primary.

Next independent work: update the standalone senior HTML/judge presentation from saved artifacts, preserving weak passive/uncertainty results and methodological limits. Remote CSE-CIC-IDS2018 download remains unstarted and independent; respect approved HDD-only scope.
