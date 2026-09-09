# V4 Action-Conditioned Sealed Results

## One-shot status

The two checkpoints frozen in `docs/V4_ACTION_MODEL_PROTOCOL.md` were evaluated once. Neither model was retrained or selected using test.

The initial test build stopped before producing arrays or model predictions because `lab_090` supplied only five complete post-action windows, below the frozen six-state horizon. The complete predefined `action_7002` permit/block family (`lab_089`, `lab_090`) was excluded using intervention/capture timing only. Evaluation therefore contains 10 episodes / five paired families.

## Results

| Frozen metric | Scratch | Friday initialized |
|---|---:|---:|
| normalized state MAE | **0.0603** | 0.0719 |
| active normalized state MAE | **0.2327** | 0.2475 |
| quiet normalized state MAE | **0.0323** | 0.0435 |
| future-edge AP | 1.000 | 1.000 |
| LM AP | 1.000 | 1.000 |
| LM F1 at fixed 0.5 | 1.000 | 1.000 |
| LM Brier | **8.91e-7** | 1.60e-4 |
| LM-pair AP / top-1 | 1.000 / 1.000 | 1.000 / 1.000 |
| same-context permit−block LM probability | 0.999 | 0.995 |
| same-context state branch difference | 0.102 | 0.123 |
| factual action lower state error | 10/10 | 10/10 |
| opposite−factual state MAE | 0.0737 | 0.0672 |

Scratch is the better dynamics forecast on this sealed cohort; Friday initialization did not improve V4 state prediction. Edge and primary semantic classification tie. This is a small controlled-lab result, not evidence against public initialization generally.

## Threshold lesson

The train-derived thresholds were extremely high because both models overfit the 12 deterministic training actions:

```text
scratch threshold 0.99973 -> test F1 0.750
Friday threshold 0.99807 -> test F1 0.889
```

At the prespecified fixed threshold 0.5 both reach F1 1.0, and both have AP 1.0. This is evidence that train-optimal threshold transfer is brittle. The 10-sample ECE/Brier values are diagnostics only—not deployment-calibrated uncertainty.

## What is supported

Observed on five fresh permit/block pairs:

1. separately supplied chosen action changes latent rollout and decoded future graph state;
2. the factual chosen action gives lower state error than the opposite action for every eligible episode;
3. the model forecasts the SSH/LM pair and permit-versus-block completion outcome;
4. host/action-pair relabeling remains equivariant (all maximum discrepancies below `3.82e-6`);
5. scratch outperforms the fixed Friday initializer on state dynamics here.

## What is not supported

- The LM score is not a discovery of causal intent: action type deterministically controls outcome in this lab and is an explicit input.
- Opposite-action futures are design-based counterfactuals; the unfactual outcome is not simultaneously observed for one exact packet context.
- Matched captures are not exact clones.
- Five correlated pairs on one three-container topology do not establish enterprise generalization.
- Perfect semantic/edge ranks should not obscure nonzero active-state error.

## Artifacts

```text
outputs/mvp_v4/action_model/sealed_test.json
outputs/mvp_v4/action_sequences/test.npz
outputs/mvp_v4/action_sequences/test_sample_manifest.csv
scripts/44_evaluate_action_graph_rssm_v4.py
```
