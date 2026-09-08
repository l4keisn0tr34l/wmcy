# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current verified project result

- Equal-duration V2: 24 episodes, duration shortcut removed.
- Compact hybrid RSSM: state MAE 0.280 vs Ridge 0.354/persistence 0.384; LM F1 0.800; pre-first LM F1 0.769.
- RSSM forecasting is strong relative to baselines, but edge AP 0.222 and pair top-1 0.143 remain weak.
- Episode test: 2/2 progressing detected, mean exact lead 26.4 s; scan-only and failed guessing create false alerts.
- Standalone report: `/home/paprika/Downloads/rssm_eod_report.html`.

## New independent senior context

The senior independently reported that transformers failed on limited data, RSSM forecasting was better, downstream classes remained weak, and allowing downstream losses to unfreeze/adapt the RSSM improved most rare classes. They also reported RSSM+DANN and KL ablations. Their repository, data contract, splits, checkpoints, and metrics are unavailable, so these are external hypotheses only and must never be merged with or numerically compared to our results.

Our current RSSM is already analogous to an unfrozen joint model because LM/ATT&CK/pair losses backpropagate through the latent dynamics. This may contribute to its stronger semantic results. It is not evidence that a frozen self-supervised RSSM would perform equally well.

## Next code batch: representation-training ablation

Implement the following on identical V2 splits, architecture, seeds, and validation-only selection:

1. **Joint/unfrozen RSSM — current reference**
   - telemetry prediction + reconstruction + KL + semantic losses;
   - full model trainable.

2. **Frozen two-stage RSSM**
   - pretrain only telemetry prediction + reconstruction + edge + KL;
   - freeze encoder, GRU, prior/posterior, and decoder;
   - train LM/ATT&CK/pair heads on imagined future latents only.

3. **Unfrozen fine-tuned two-stage RSSM**
   - initialize from the same self-supervised checkpoint;
   - attach heads and fine-tune the full model jointly;
   - compare against training jointly from scratch.

4. **KL ablation**
   - same joint model but KL weight zero;
   - check forecasting, latent stochastic spread, and downstream semantics.

Report forecasting separately from downstream security metrics. Select checkpoints on validation only and evaluate test once per frozen experiment definition. Use the same active/quiet/per-horizon, LM/pre-first-LM, ATT&CK, edge, pair, uncertainty, false-alert, and host-permutation metrics.

## Explicit deferrals

- **Transformer:** do not add with only 24 episodes; no evidence it is appropriate.
- **DANN:** defer until there are meaningful source domains (for example lab versus public telemetry) and a valid domain label. Scenario is not a safe substitute for domain.
- **Fusion/ensemble:** first measure validation error correlation and oracle gain. Do not combine Ridge/RSSM merely because one edge AP differs by 0.008; fusion requires complementary validated signal.
- **Action conditioning:** defer until intervention/no-intervention tuples exist.

## Interpretation gate

- If frozen two-stage semantics approach the joint model, self-supervised dynamics carry useful security information.
- If unfreezing is much better, semantic supervision is shaping the latent representation and the supervision bottleneck remains important.
- If KL removal improves point prediction but destroys useful spread/generalization, retain stochastic regularization.
- Never describe the senior’s independent scores as project results or compare their class F1 numerically to our horizon/multi-label metrics.
