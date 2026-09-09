# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Frozen V4 action protocol

The action model and evaluator are complete using V4 action train only. V4 test arrays still do not exist.

```text
architecture: ActionGraphRSSM, 372,947 parameters
train samples: 12
protocol: seed43001, batch12, Adam3e-4, exactly400 epochs
selection: none; carry both candidates
```

Frozen checkpoints:

```text
scratch SHA-256:
f69ece2d24ac07593d35666a83ea52ef8a768d1f51d4dcab13e7c1dc9f1c876f

Friday initialized SHA-256:
6b4cf070da873f04cd1fecb17cf351b9f9ba07bcbf45275b80aa69a8b100e351
```

Train-only state MAE scratch/Friday is 0.047/0.043; active MAE 0.157/0.108. Both perfectly fit train LM/edge/pair targets and strongly separate permit/block risk. This is overfit training evidence only.

## Next batch: one-shot action-test unlock

Only after the protocol/evaluator commit:

1. Run `scripts/42_build_v4_action_sequences.py --split test --unlock-test` once.
2. Inspect only schema/finiteness/counts before prediction.
3. Run `scripts/44_evaluate_action_graph_rssm_v4.py --device cuda` once.
4. Do not rerun/tune either model against resulting test metrics.
5. Document factual state/edge/LM/pair results, action-effect direction, factual-versus-opposite state error, initialization trade-off, and six-pair limitation.
6. Then build/evaluate the already-frozen passive outcome-branch model on predetermined V4 passive/direct/action test rows under a separately frozen script.
7. Regenerate report, commit, and push.

Do not reopen V3 test.
